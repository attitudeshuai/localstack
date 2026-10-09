"""
Base class and utilities for provider stores.

Stores provide storage for AWS service providers and are analogous to Moto's BackendDict.

By convention, Stores are to be defined in `models` submodule of the service
by subclassing BaseStore e.g. `localstack.services.sqs.models.SqsStore`
Also by convention, cross-region and cross-account attributes are declared in CAPITAL_CASE

    class SqsStore(BaseStore):
        queues: dict[str, SqsQueue] =  LocalAttribute(default=dict)
        DELETED: dict[str, float] = CrossRegionAttribute(default=dict)

Stores are then wrapped in AccountRegionBundle

    sqs_stores = AccountRegionBundle('sqs', SqsStore)

Access patterns are as follows

    account_id = '001122334455'
    sqs_stores[account_id]  # -> RegionBundle
    sqs_stores[account_id]['ap-south-1']  # -> SqsStore
    sqs_stores[account_id]['ap-south-1'].queues  # -> {}

There should be a single declaration of a Store for a given service. If a service
has both Community and Pro providers, it must be declared as in Community codebase.
All Pro attributes must be declared within.

While not recommended, store classes may define member helper functions and properties.

In addition to the in-place reset semantics of ``AccountRegionBundle`` and
``RegionBundle``, ``GenerationalAccountRegionBundle`` provides scoped reset with
generation isolation:

    gen_stores = GenerationalAccountRegionBundle('sqs', SqsStore)
    gen_stores.reset()                                    # service scope
    gen_stores.reset(account_id=account_id)               # account scope
    gen_stores.reset(account_id=account_id, region_name=region)  # region scope

Each reset publishes a new generation; data outside the reset scope is retained
verbatim. References to stores of a superseded generation remain readable as
plain Python objects but are rejected with ``StaleGenerationError`` on any
subsequent data access, instead of silently reading or writing cleared data.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from itertools import count
from threading import RLock
from typing import Any, TypeVar

from localstack import config
from localstack.utils.aws.aws_stack import get_valid_regions_for_service

LOCAL_ATTR_PREFIX = "attr_"


class StaleGenerationError(Exception):
    """
    Raised when a store reference from a superseded generation is accessed after
    a scoped reset. Callers must re-fetch the store from the bundle to continue.
    """


def _raise_if_store_stale(obj: Any) -> None:
    """
    Reject access to stores belonging to a superseded generation.

    This is a no-op for stores held by the legacy in-place reset containers,
    which never set a ``_bundle`` back-reference.
    """
    bundle = getattr(obj, "_bundle", None)
    if bundle is not None:
        bundle._validate_store(obj)


#
# Descriptor protocol classes
#


class LocalAttribute:
    """
    Descriptor protocol for marking store attributes as local to a region.
    """

    def __init__(self, default: Callable | int | float | str | bool | None):
        """
        :param default: Default value assigned to the local attribute. Must be a scalar
            or a callable.
        """
        self.default = default

    def __set_name__(self, owner, name):
        self.name = LOCAL_ATTR_PREFIX + name

    def __get__(self, obj: BaseStoreType, objtype=None) -> Any:
        if obj is not None:
            _raise_if_store_stale(obj)
        if not hasattr(obj, self.name):
            if isinstance(self.default, Callable):
                value = self.default()
            else:
                value = self.default
            setattr(obj, self.name, value)

        return getattr(obj, self.name)

    def __set__(self, obj: BaseStoreType, value: Any):
        _raise_if_store_stale(obj)
        setattr(obj, self.name, value)


class CrossRegionAttribute:
    """
    Descriptor protocol for marking store attributes as shared across all regions.
    """

    def __init__(self, default: Callable | int | float | str | bool | None):
        """
        :param default: The default value assigned to the cross-region attribute.
            This must be a scalar or a callable.
        """
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj: BaseStoreType, objtype=None) -> Any:
        if obj is not None:
            _raise_if_store_stale(obj)
        self._check_region_store_association(obj)

        if self.name not in obj._global:
            if isinstance(self.default, Callable):
                obj._global[self.name] = self.default()
            else:
                obj._global[self.name] = self.default

        return obj._global[self.name]

    def __set__(self, obj: BaseStoreType, value: Any):
        _raise_if_store_stale(obj)
        self._check_region_store_association(obj)

        obj._global[self.name] = value

    def _check_region_store_association(self, obj):
        if not hasattr(obj, "_global"):
            # Raise if a Store is instantiated outside of a RegionBundle
            raise AttributeError(
                "Could not resolve cross-region attribute because there is no associated RegionBundle"
            )


class CrossAccountAttribute:
    """
    Descriptor protocol for marking a store attributes as shared across all regions and accounts.

    This should be used for resources that are identified by ARNs.
    """

    def __init__(self, default: Callable | int | float | str | bool | None):
        """
        :param default: The default value assigned to the cross-account attribute.
            This must be a scalar or a callable.
        """
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj: BaseStoreType, objtype=None) -> Any:
        if obj is not None:
            _raise_if_store_stale(obj)
        self._check_account_store_association(obj)

        if self.name not in obj._universal:
            if isinstance(self.default, Callable):
                obj._universal[self.name] = self.default()
            else:
                obj._universal[self.name] = self.default

        return obj._universal[self.name]

    def __set__(self, obj: BaseStoreType, value: Any):
        _raise_if_store_stale(obj)
        self._check_account_store_association(obj)

        obj._universal[self.name] = value

    def _check_account_store_association(self, obj):
        if not hasattr(obj, "_universal"):
            # Raise if a Store is instantiated outside an AccountRegionBundle
            raise AttributeError(
                "Could not resolve cross-account attribute because there is no associated AccountRegionBundle"
            )


#
# Base models
#


class BaseStore:
    """
    Base class for defining stores for LocalStack providers.
    """

    _service_name: str
    _account_id: str
    _region_name: str
    _global: dict[str, Any] | _SharedDict
    _universal: dict[str, Any] | _SharedDict
    _bundle: GenerationalAccountRegionBundle[Any] | None
    _root_generation: Generation | None
    _account_generation: Generation | None
    _region_generation: Generation | None

    def __repr__(self):
        try:
            repr_templ = "<{name} object for {service_name} at {account_id}/{region_name}>"
            return repr_templ.format(
                name=self.__class__.__name__,
                service_name=self._service_name,
                account_id=self._account_id,
                region_name=self._region_name,
            )
        except AttributeError:
            return super().__repr__()

    def check_live(self) -> None:
        """
        Raise ``StaleGenerationError`` if this store belongs to a superseded generation.

        Stores held by the legacy in-place reset containers are always considered live.
        """
        _raise_if_store_stale(self)

    def is_live(self) -> bool:
        """
        Return whether this store reference is still backed by a live generation.
        """
        try:
            self.check_live()
        except StaleGenerationError:
            return False
        return True

    @property
    def generation(self) -> Generation:
        """The service-wide (root) generation this store belongs to."""
        return self._root_generation

    @property
    def account_generation(self) -> Generation:
        """The account-scoped generation this store belongs to."""
        return self._account_generation

    @property
    def region_generation(self) -> Generation:
        """The region-scoped generation this store belongs to."""
        return self._region_generation


BaseStoreType = TypeVar("BaseStoreType", bound=BaseStore)


#
# Encapsulations
#


class RegionBundle[BaseStoreType: BaseStore](dict):
    """
    Encapsulation for stores across all regions for a specific AWS account ID.
    """

    def __init__(
        self,
        service_name: str,
        store: type[BaseStoreType],
        account_id: str,
        validate: bool = True,
        lock: RLock = None,
        universal: dict = None,
    ):
        self.store = store
        self.account_id = account_id
        self.service_name = service_name
        self.validate = validate
        self.lock = lock or RLock()

        self.valid_regions = get_valid_regions_for_service(service_name)

        # Keeps track of all cross-region attributes. This dict is maintained at
        # a region level (hence in RegionBundle). A ref is passed to every store
        # intialised in this region so that backref is possible.
        self._global = {}

        # Keeps track of all cross-account attributes. This dict is maintained at
        # the account level (ie. AccountRegionBundle). A ref is passed down from
        # AccountRegionBundle to RegionBundle to individual stores to enable backref.
        self._universal = universal

    def __getitem__(self, region_name) -> BaseStoreType:
        if (
            not config.ALLOW_NONSTANDARD_REGIONS
            and self.validate
            and region_name not in self.valid_regions
        ):
            raise ValueError(
                f"'{region_name}' is not a valid AWS region name for {self.service_name}"
            )

        with self.lock:
            if region_name not in self.keys():
                store_obj = self.store()

                store_obj._global = self._global
                store_obj._universal = self._universal
                store_obj._service_name = self.service_name
                store_obj._account_id = self.account_id
                store_obj._region_name = region_name

                self[region_name] = store_obj

        return super().__getitem__(region_name)

    def reset(self, _reset_universal: bool = False):
        """
        Clear all store data.

        This only deletes the data held in the stores. All instantiated stores
        are retained. This includes data shared by all stores in this account
        and marked by the CrossRegionAttribute descriptor.

        Data marked by CrossAccountAttribute descriptor is only cleared when
        `_reset_universal` is set. Note that this escapes the logical boundary of
        the account associated with this RegionBundle and affects *all* accounts.
        Hence this argument is not intended for public use and is only used when
        invoking this method from AccountRegionBundle.
        """
        # For safety, clear data in all referenced store instances, if any
        for store_inst in self.values():
            attrs = list(store_inst.__dict__.keys())
            for attr in attrs:
                # reset the cross-region attributes
                if attr == "_global":
                    store_inst._global.clear()

                if attr == "_universal" and _reset_universal:
                    store_inst._universal.clear()

                # reset the local attributes
                elif attr.startswith(LOCAL_ATTR_PREFIX):
                    delattr(store_inst, attr)

        self._global.clear()

        with self.lock:
            self.clear()


class AccountRegionBundle[BaseStoreType: BaseStore](dict):
    """
    Encapsulation for all stores for all AWS account IDs.
    """

    def __init__(self, service_name: str, store: type[BaseStoreType], validate: bool = True):
        """
        :param service_name: Name of the service. Must be a valid service defined in botocore.
        :param store: Class definition of the Store
        :param validate: Whether to raise if invalid region names or account IDs are used during subscription
        """
        self.service_name = service_name
        self.store = store
        self.validate = validate
        self.lock = RLock()

        # Keeps track of all cross-account attributes. This dict is maintained at
        # the account level (hence in AccountRegionBundle). A ref is passed to
        # every region bundle, which in turn passes it to every store in it.
        self._universal = {}

    def __getitem__(self, account_id: str) -> RegionBundle[BaseStoreType]:
        if self.validate and not re.match(r"\d{12}", account_id):
            raise ValueError(f"'{account_id}' is not a valid AWS account ID")

        with self.lock:
            if account_id not in self.keys():
                self[account_id] = RegionBundle(
                    service_name=self.service_name,
                    store=self.store,
                    account_id=account_id,
                    validate=self.validate,
                    lock=self.lock,
                    universal=self._universal,
                )

        return super().__getitem__(account_id)

    def reset(self):
        """
        Clear all store data.

        This only deletes the data held in the stores. All instantiated stores are retained.
        """
        # For safety, clear all referenced region bundles, if any
        for region_bundle in self.values():
            region_bundle.reset(_reset_universal=True)

        self._universal.clear()

        with self.lock:
            self.clear()

    def iter_stores(self) -> Iterator[tuple[str, str, BaseStoreType]]:
        """
        Iterate over a flattened view of all stores in this AccountRegionBundle, where each record is a
        tuple of account id, region name, and the store within that account and region. Example::

        :return: an iterator
        """
        for account_id, region_stores in self.items():
            for region_name, store in region_stores.items():
                yield account_id, region_name, store


#
# Generational encapsulations
#


class Generation:
    """
    Opaque generation token. Generation identity is compared with ``is``.
    """

    __slots__ = ("scope", "seq")

    def __init__(self, scope: str, seq: int):
        self.scope = scope
        self.seq = seq

    def __repr__(self) -> str:
        return f"<Generation {self.scope}#{self.seq}>"


class _SharedDict:
    """
    Indirection around the dict backing a shared scope.

    Reset replaces the inner dict wholesale instead of clearing it in place, so
    data belonging to superseded generations remains physically complete, while
    all live holders of this envelope immediately observe the replacement.
    """

    __slots__ = ("data",)

    def __init__(self, data: dict[str, Any] | None = None):
        self.data = {} if data is None else data

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def replace(self, data: dict[str, Any] | None = None) -> None:
        self.data = {} if data is None else data


class _RegionScope[BaseStoreType: BaseStore]:
    __slots__ = ("store", "generation")

    def __init__(self, store: BaseStoreType, generation: Generation):
        self.store = store
        self.generation = generation


class _AccountScope[BaseStoreType: BaseStore]:
    __slots__ = ("regions", "generation", "global_envelope")

    def __init__(self, generation: Generation, global_envelope: _SharedDict):
        self.regions: dict[str, _RegionScope[BaseStoreType]] = {}
        self.generation = generation
        self.global_envelope = global_envelope


class GenerationalRegionBundle[BaseStoreType: BaseStore]:
    """
    Live view on all region stores of a specific AWS account ID.

    The view always resolves the account's current generation, while
    ``keys()``, ``values()`` and ``items()`` return point-in-time snapshots.
    Iteration is therefore unaffected by concurrent resets or first access to
    additional regions.
    """

    def __init__(
        self,
        bundle: GenerationalAccountRegionBundle[BaseStoreType],
        account_id: str,
    ):
        self._bundle: GenerationalAccountRegionBundle[BaseStoreType] = bundle
        self.account_id = account_id
        self.service_name = bundle.service_name

    def __getitem__(self, region_name: str) -> BaseStoreType:
        return self._bundle._get_store(self.account_id, region_name)

    def get(self, region_name: str, default: BaseStoreType | None = None) -> BaseStoreType | None:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            if account_scope is None:
                return default
            region_scope = account_scope.regions.get(region_name)
            return region_scope.store if region_scope is not None else default

    def __contains__(self, region_name: object) -> bool:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            return account_scope is not None and region_name in account_scope.regions

    def __len__(self) -> int:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            return len(account_scope.regions) if account_scope is not None else 0

    def __iter__(self) -> Iterator[str]:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            region_names = list(account_scope.regions.keys()) if account_scope is not None else []
        yield from region_names

    def keys(self) -> list[str]:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            return list(account_scope.regions.keys()) if account_scope is not None else []

    def values(self) -> list[BaseStoreType]:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            if account_scope is None:
                return []
            return [region_scope.store for region_scope in account_scope.regions.values()]

    def items(self) -> list[tuple[str, BaseStoreType]]:
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            if account_scope is None:
                return []
            return [
                (region_name, region_scope.store)
                for region_name, region_scope in account_scope.regions.items()
            ]

    @property
    def global_data(self) -> dict[str, Any]:
        """
        The live dict of cross-region data for this account. A reset with
        cross-region cleanup replaces this dict rather than clearing it.
        """
        with self._bundle.lock:
            account_scope = self._bundle._accounts.get(self.account_id)
            if account_scope is None:
                return {}
            return account_scope.global_envelope.data

    def reset(
        self,
        *,
        cross_region: bool | None = None,
        cross_account: bool | None = None,
    ) -> None:
        """
        Reset this account scope. By default, local and cross-region data are
        cleared, while cross-account data is preserved.
        """
        self._bundle.reset(
            self.account_id,
            cross_region=cross_region,
            cross_account=cross_account,
        )


class GenerationalAccountRegionBundle[BaseStoreType: BaseStore]:
    """
    Encapsulation of all stores for all AWS account IDs, with scoped reset and
    generation isolation.

    Access patterns are identical to ``AccountRegionBundle``::

        bundle[account_id][region_name]

    Reset scopes::

        bundle.reset()                                     # service scope
        bundle.reset(account_id=account_id)                # account scope
        bundle.reset(account_id=account_id, region_name=r) # region scope

    Cleanup of cross-region and cross-account shared data is declared
    separately via ``cross_region`` and ``cross_account``. When left
    unspecified, defaults reproduce the legacy reset semantics:

    - service scope: both cross-region and cross-account data are cleared
    - account scope: cross-region data cleared, cross-account data preserved
    - region scope: neither shared scope is cleared (local data only)

    A reset never mutates the object graph of the superseded generation: it
    publishes fresh state and swaps it in atomically under a lock. Concurrent
    access therefore sees either the complete old generation or the complete
    new generation. References into a superseded generation remain ordinary
    Python objects, but data access on them raises ``StaleGenerationError``.
    """

    def __init__(self, service_name: str, store: type[BaseStoreType], validate: bool = True):
        """
        :param service_name: Name of the service. Must be a valid service defined in botocore.
        :param store: Class definition of the Store
        :param validate: Whether to raise if invalid region names or account IDs are used during subscription
        """
        self.service_name = service_name
        self.store = store
        self.validate = validate
        self.lock = RLock()

        self.valid_regions = get_valid_regions_for_service(service_name)

        self._gen_counter = count(1)
        self._root_generation = Generation("service", next(self._gen_counter))

        self._accounts: dict[str, _AccountScope[BaseStoreType]] = {}
        self._views: dict[str, GenerationalRegionBundle[BaseStoreType]] = {}

        # Cross-account attributes, shared via an envelope across every
        # generation of every account.
        self._universal = _SharedDict()

        # Cross-region envelopes preserved across a service-scope reset
        # (cross_region=False), reattached when the account is subscribed again.
        self._preserved_global: dict[str, _SharedDict] = {}

    @property
    def generation(self) -> Generation:
        """The current service-wide (root) generation."""
        return self._root_generation

    def __getitem__(self, account_id: str) -> GenerationalRegionBundle[BaseStoreType]:
        if self.validate and not re.match(r"\d{12}", account_id):
            raise ValueError(f"'{account_id}' is not a valid AWS account ID")

        with self.lock:
            self._get_or_create_account_scope(account_id)
            return self._get_view(account_id)

    def get(
        self, account_id: str, default: GenerationalRegionBundle[BaseStoreType] | None = None
    ) -> GenerationalRegionBundle[BaseStoreType] | None:
        with self.lock:
            if account_id not in self._accounts:
                return default
            return self._get_view(account_id)

    def __contains__(self, account_id: object) -> bool:
        with self.lock:
            return account_id in self._accounts

    def __len__(self) -> int:
        with self.lock:
            return len(self._accounts)

    def __iter__(self) -> Iterator[str]:
        with self.lock:
            account_ids = list(self._accounts.keys())
        yield from account_ids

    def keys(self) -> list[str]:
        with self.lock:
            return list(self._accounts.keys())

    def values(self) -> list[GenerationalRegionBundle[BaseStoreType]]:
        with self.lock:
            return [self._get_view(account_id) for account_id in self._accounts]

    def items(
        self,
    ) -> list[tuple[str, GenerationalRegionBundle[BaseStoreType]]]:
        with self.lock:
            return [(account_id, self._get_view(account_id)) for account_id in self._accounts]

    def iter_stores(self) -> Iterator[tuple[str, str, BaseStoreType]]:
        """
        Iterate over a flattened snapshot of all stores, yielding tuples of
        account id, region name, and store. Concurrent resets and first access
        never interrupt the traversal.
        """
        with self.lock:
            account_ids = list(self._accounts.keys())

        for account_id in account_ids:
            with self.lock:
                account_scope = self._accounts.get(account_id)
                region_scopes = (
                    list(account_scope.regions.items()) if account_scope is not None else []
                )
            for region_name, region_scope in region_scopes:
                yield account_id, region_name, region_scope.store

    def reset(
        self,
        account_id: str | None = None,
        region_name: str | None = None,
        *,
        cross_region: bool | None = None,
        cross_account: bool | None = None,
    ) -> None:
        """
        Reset state within the given scope.

        :param account_id: Account scope; when omitted, the whole service is reset.
        :param region_name: Region scope; requires ``account_id``.
        :param cross_region: Whether cross-region shared data is cleared.
        :param cross_account: Whether cross-account shared data is cleared
            (escapes the account boundary and affects every account).
        """
        if region_name is not None and account_id is None:
            raise ValueError("region_name requires account_id to be specified")

        if account_id is None:
            default_cross_region = True
            default_cross_account = True
        elif region_name is None:
            default_cross_region = True
            default_cross_account = False
        else:
            default_cross_region = False
            default_cross_account = False

        if cross_region is None:
            cross_region = default_cross_region
        if cross_account is None:
            cross_account = default_cross_account

        if account_id is None:
            self._reset_service(cross_region, cross_account)
        elif region_name is None:
            self._reset_account(account_id, cross_region, cross_account)
        else:
            self._reset_region(account_id, region_name, cross_region, cross_account)

    def _reset_service(self, cross_region: bool, cross_account: bool) -> None:
        with self.lock:
            # Publish the new generation before any state is replaced.
            new_root_generation = Generation("service", next(self._gen_counter))

            if cross_account:
                self._universal.replace()

            if cross_region:
                self._preserved_global = {}
            else:
                # Preserve cross-region data per account. The old account
                # scopes are detached and never mutated.
                self._preserved_global = {
                    account_id: account_scope.global_envelope
                    for account_id, account_scope in self._accounts.items()
                }

            self._accounts = {}
            self._root_generation = new_root_generation

    def _reset_account(self, account_id: str, cross_region: bool, cross_account: bool) -> None:
        if self.validate and not re.match(r"\d{12}", account_id):
            raise ValueError(f"'{account_id}' is not a valid AWS account ID")

        with self.lock:
            if cross_account:
                self._universal.replace()

            old_account_scope = self._accounts.get(account_id)
            if not cross_region and old_account_scope is not None:
                global_envelope = old_account_scope.global_envelope
            else:
                global_envelope = _SharedDict()

            # Pending service-scope preservation for this account is superseded.
            self._preserved_global.pop(account_id, None)

            new_account_scope = _AccountScope[BaseStoreType](
                generation=Generation("account", next(self._gen_counter)),
                global_envelope=global_envelope,
            )
            self._accounts[account_id] = new_account_scope

    def _reset_region(
        self,
        account_id: str,
        region_name: str,
        cross_region: bool,
        cross_account: bool,
    ) -> None:
        if self.validate and not re.match(r"\d{12}", account_id):
            raise ValueError(f"'{account_id}' is not a valid AWS account ID")
        self._validate_region(region_name)

        with self.lock:
            if cross_account:
                self._universal.replace()

            account_scope = self._get_or_create_account_scope(account_id)

            if cross_region:
                account_scope.global_envelope.replace()

            self._build_store(account_scope, account_id, region_name)

    def _get_view(self, account_id: str) -> GenerationalRegionBundle[BaseStoreType]:
        view = self._views.get(account_id)
        if view is None:
            view = GenerationalRegionBundle(self, account_id)
            self._views[account_id] = view
        return view

    def _get_or_create_account_scope(self, account_id: str) -> _AccountScope[BaseStoreType]:
        account_scope = self._accounts.get(account_id)
        if account_scope is None:
            global_envelope = self._preserved_global.pop(account_id, None)
            if global_envelope is None:
                global_envelope = _SharedDict()
            account_scope = _AccountScope[BaseStoreType](
                generation=Generation("account", next(self._gen_counter)),
                global_envelope=global_envelope,
            )
            self._accounts[account_id] = account_scope
        return account_scope

    def _validate_region(self, region_name: str) -> None:
        if (
            not config.ALLOW_NONSTANDARD_REGIONS
            and self.validate
            and region_name not in self.valid_regions
        ):
            raise ValueError(
                f"'{region_name}' is not a valid AWS region name for {self.service_name}"
            )

    def _get_store(self, account_id: str, region_name: str) -> BaseStoreType:
        self._validate_region(region_name)
        with self.lock:
            account_scope = self._get_or_create_account_scope(account_id)
            region_scope = account_scope.regions.get(region_name)
            if region_scope is None:
                return self._build_store(account_scope, account_id, region_name)
            return region_scope.store

    def _build_store(
        self,
        account_scope: _AccountScope[BaseStoreType],
        account_id: str,
        region_name: str,
    ) -> BaseStoreType:
        # Caller must hold self.lock.
        store_obj = self.store()

        store_obj._bundle = self
        store_obj._global = account_scope.global_envelope
        store_obj._universal = self._universal
        store_obj._service_name = self.service_name
        store_obj._account_id = account_id
        store_obj._region_name = region_name

        store_obj._root_generation = self._root_generation
        store_obj._account_generation = account_scope.generation
        region_generation = Generation("region", next(self._gen_counter))
        store_obj._region_generation = region_generation

        account_scope.regions[region_name] = _RegionScope[BaseStoreType](
            store_obj, region_generation
        )
        return store_obj

    def _validate_store(self, store_obj: BaseStore) -> None:
        reason = None
        with self.lock:
            if store_obj._root_generation is not self._root_generation:
                reason = "the service state was reset"
            elif (account_scope := self._accounts.get(store_obj._account_id)) is None:
                reason = "the account state was reset"
            elif account_scope.generation is not store_obj._account_generation:
                reason = "the account state was reset"
            elif (region_scope := account_scope.regions.get(store_obj._region_name)) is None:
                reason = "the region state was reset"
            elif region_scope.generation is not store_obj._region_generation:
                reason = "the region state was reset"

        if reason is not None:
            raise StaleGenerationError(
                f"Store reference {store_obj!r} is no longer valid because {reason}; "
                f"fetch the store again from the bundle"
            )
