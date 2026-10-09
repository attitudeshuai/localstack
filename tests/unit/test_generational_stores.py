"""
Unit tests for the generational state container with scoped reset.

These tests are self-contained (no repository fixtures) so they can also be
executed in stripped-down environments, e.g.::

    pytest tests/unit/test_generational_stores.py --noconftest
"""

import threading
import unittest

import pytest

from localstack.services.stores import (
    BaseStore,
    CrossAccountAttribute,
    CrossRegionAttribute,
    Generation,
    GenerationalAccountRegionBundle,
    LocalAttribute,
    StaleGenerationError,
)

ACCOUNT1 = "696969696969"
ACCOUNT2 = "424242424242"
EU_REGION = "eu-central-1"
AP_REGION = "ap-south-1"


class SampleStore(BaseStore):
    CROSS_ACCOUNT_ATTR = CrossAccountAttribute(default=list)
    CROSS_REGION_ATTR = CrossRegionAttribute(default=list)
    region_specific_attr = LocalAttribute(default=list)


def make_bundle(validate: bool = False) -> GenerationalAccountRegionBundle:
    return GenerationalAccountRegionBundle("zzz", SampleStore, validate=validate)


#
# Access semantics
#


class TestAccessSemantics:
    def test_lazy_initialisation_and_identity(self):
        stores = make_bundle()

        assert stores[ACCOUNT1] is stores[ACCOUNT1]
        assert stores[ACCOUNT1][EU_REGION] is stores[ACCOUNT1][EU_REGION]
        assert stores[ACCOUNT1][EU_REGION] is not stores[ACCOUNT1][AP_REGION]

        store = stores[ACCOUNT1][EU_REGION]
        assert store._service_name == "zzz"
        assert store._account_id == ACCOUNT1
        assert store._region_name == EU_REGION
        assert store.generation is stores.generation
        assert store.is_live()
        store.check_live()

    def test_local_region_isolation(self):
        stores = make_bundle()
        store_eu = stores[ACCOUNT1][EU_REGION]
        store_ap = stores[ACCOUNT1][AP_REGION]

        store_eu.region_specific_attr.extend([1, 2, 3])
        assert store_ap.region_specific_attr == []

    def test_cross_region_sharing_within_account(self):
        stores = make_bundle()
        store_eu = stores[ACCOUNT1][EU_REGION]
        store_ap = stores[ACCOUNT1][AP_REGION]
        account2 = stores[ACCOUNT2][EU_REGION]

        store_eu.CROSS_REGION_ATTR.extend([4, 5, 6])
        assert store_ap.CROSS_REGION_ATTR == [4, 5, 6]
        # cross-region scope does not cross the account boundary
        assert account2.CROSS_REGION_ATTR == []

    def test_cross_account_sharing(self):
        stores = make_bundle()
        store1 = stores[ACCOUNT1][EU_REGION]
        store2 = stores[ACCOUNT2][AP_REGION]

        store1.CROSS_ACCOUNT_ATTR.extend([100j, 200j])
        assert store2.CROSS_ACCOUNT_ATTR == [100j, 200j]

    def test_get_without_creation(self):
        stores = make_bundle()

        assert stores.get(ACCOUNT1) is None
        assert stores.get(ACCOUNT1, "default") == "default"

        stores[ACCOUNT1]
        assert stores.get(ACCOUNT1) is not None
        assert stores[ACCOUNT1].get(EU_REGION) is None
        assert stores[ACCOUNT1].get(EU_REGION, "default") == "default"

        store = stores[ACCOUNT1][EU_REGION]
        assert stores[ACCOUNT1].get(EU_REGION) is store

    def test_contains_len_iteration(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION]
        stores[ACCOUNT2][AP_REGION]

        assert ACCOUNT1 in stores
        assert ACCOUNT2 in stores
        assert "111111111111" not in stores
        assert len(stores) == 2
        assert set(stores) == {ACCOUNT1, ACCOUNT2}
        assert EU_REGION in stores[ACCOUNT1]
        assert len(stores[ACCOUNT1]) == 1

    def test_validating_bundle(self):
        stores = GenerationalAccountRegionBundle("sns", SampleStore, validate=True)
        with pytest.raises(Exception) as exc:
            stores["not-an-account"]
        exc.match("not a valid AWS account ID")

        stores[ACCOUNT1]
        with pytest.raises(Exception) as exc:
            stores[ACCOUNT1]["invalid-region"]
        exc.match("not a valid AWS region")

        assert stores[ACCOUNT1]["us-east-1"]

    @unittest.mock.patch("localstack.config.ALLOW_NONSTANDARD_REGIONS", True)
    def test_nonstandard_regions(self):
        stores = GenerationalAccountRegionBundle("sns", SampleStore, validate=True)
        assert stores[ACCOUNT1]["pluto-south-3"]


#
# Scoped reset
#


class TestRegionScopeReset:
    def test_local_data_cleared_only_for_target_region(self):
        stores = make_bundle()
        target = stores[ACCOUNT1][EU_REGION]
        sibling = stores[ACCOUNT1][AP_REGION]
        other_account = stores[ACCOUNT2][EU_REGION]

        target.region_specific_attr.extend([1, 2, 3])
        sibling.region_specific_attr.extend([4, 5, 6])
        other_account.region_specific_attr.extend([7, 8, 9])

        stores.reset(account_id=ACCOUNT1, region_name=EU_REGION)

        assert not target.is_live()
        assert sibling.is_live()
        assert other_account.is_live()
        assert sibling.region_specific_attr == [4, 5, 6]
        assert other_account.region_specific_attr == [7, 8, 9]

        new_store = stores[ACCOUNT1][EU_REGION]
        assert new_store is not target
        assert new_store.is_live()
        assert new_store.region_specific_attr == []
        assert new_store._account_id == ACCOUNT1
        assert new_store._region_name == EU_REGION

    def test_shared_scopes_preserved_by_default(self):
        stores = make_bundle()
        target = stores[ACCOUNT1][EU_REGION]
        sibling = stores[ACCOUNT1][AP_REGION]
        other_account = stores[ACCOUNT2][AP_REGION]

        target.CROSS_REGION_ATTR.extend(["a", "b"])
        target.CROSS_ACCOUNT_ATTR.extend([100j])

        stores.reset(account_id=ACCOUNT1, region_name=EU_REGION)

        new_store = stores[ACCOUNT1][EU_REGION]
        assert new_store.CROSS_REGION_ATTR == ["a", "b"]
        assert sibling.CROSS_REGION_ATTR == ["a", "b"]
        assert new_store.CROSS_ACCOUNT_ATTR == [100j]
        assert other_account.is_live()

    def test_stale_reference_explicitly_rejected(self):
        stores = make_bundle()
        stale = stores[ACCOUNT1][EU_REGION]
        stale.region_specific_attr.append("x")
        stale.CROSS_REGION_ATTR.append("y")
        stale.CROSS_ACCOUNT_ATTR.append("z")

        stores.reset(account_id=ACCOUNT1, region_name=EU_REGION)

        with pytest.raises(StaleGenerationError) as exc:
            stale.check_live()
        with pytest.raises(StaleGenerationError):
            stale.region_specific_attr  # noqa: B018
        with pytest.raises(StaleGenerationError):
            stale.CROSS_REGION_ATTR  # noqa: B018
        with pytest.raises(StaleGenerationError):
            stale.CROSS_ACCOUNT_ATTR  # noqa: B018
        with pytest.raises(StaleGenerationError):
            stale.region_specific_attr = []
        with pytest.raises(StaleGenerationError):
            stale.CROSS_REGION_ATTR = []
        with pytest.raises(StaleGenerationError):
            stale.CROSS_ACCOUNT_ATTR = []

        # metadata remains inspectable, error message is explicit
        assert stale._region_name == EU_REGION
        message = str(exc.value)
        assert ACCOUNT1 in message
        assert EU_REGION in message
        assert "region state was reset" in message

    def test_cross_region_flag_escapes_region_boundary(self):
        stores = make_bundle()
        target = stores[ACCOUNT1][EU_REGION]
        sibling = stores[ACCOUNT1][AP_REGION]
        other_account = stores[ACCOUNT2][EU_REGION]

        target.CROSS_REGION_ATTR.append("shared")
        sibling.region_specific_attr.append("keep")
        other_account.CROSS_REGION_ATTR.append("other-account")

        stores.reset(
            account_id=ACCOUNT1,
            region_name=EU_REGION,
            cross_region=True,
        )

        # explicitly declared: siblings see the cleared cross-region scope ...
        assert sibling.is_live()
        assert sibling.CROSS_REGION_ATTR == []
        # ... but their local data is untouched ...
        assert sibling.region_specific_attr == ["keep"]
        # ... and other accounts are unaffected.
        assert other_account.CROSS_REGION_ATTR == ["other-account"]

    def test_cross_account_flag_escapes_account_boundary(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION].CROSS_ACCOUNT_ATTR.append("global")
        other_account = stores[ACCOUNT2][EU_REGION]

        stores.reset(
            account_id=ACCOUNT1,
            region_name=EU_REGION,
            cross_account=True,
        )

        assert other_account.is_live()
        assert other_account.CROSS_ACCOUNT_ATTR == []

    def test_invalid_region_keeps_generation_unchanged(self):
        stores = make_bundle(validate=True)
        with pytest.raises(ValueError):
            stores.reset(account_id=ACCOUNT1, region_name="invalid-region")
        # nothing was reset
        assert ACCOUNT1 not in stores


class TestAccountScopeReset:
    def test_default_clears_local_and_cross_region(self):
        stores = make_bundle()
        store_eu = stores[ACCOUNT1][EU_REGION]
        store_ap = stores[ACCOUNT1][AP_REGION]
        other_account = stores[ACCOUNT2][EU_REGION]

        store_eu.region_specific_attr.append(1)
        store_ap.CROSS_REGION_ATTR.append("x")
        store_eu.CROSS_ACCOUNT_ATTR.append("global")
        other_account.region_specific_attr.append(2)
        other_account.CROSS_REGION_ATTR.append("y")

        stores.reset(account_id=ACCOUNT1)

        assert not store_eu.is_live()
        assert not store_ap.is_live()
        assert other_account.is_live()

        assert stores[ACCOUNT1][EU_REGION].region_specific_attr == []
        assert stores[ACCOUNT1][AP_REGION].CROSS_REGION_ATTR == []
        # cross-account data is preserved by default
        assert stores[ACCOUNT1][EU_REGION].CROSS_ACCOUNT_ATTR == ["global"]
        # other accounts are retained verbatim
        assert other_account.region_specific_attr == [2]
        assert other_account.CROSS_REGION_ATTR == ["y"]

    def test_region_view_reset_matches_account_scope(self):
        stores = make_bundle()
        store = stores[ACCOUNT1][EU_REGION]
        store.region_specific_attr.append(1)
        stores[ACCOUNT1].reset()
        assert not store.is_live()
        assert stores[ACCOUNT1][EU_REGION].region_specific_attr == []

    def test_cross_region_false_preserves_shared_data(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION].CROSS_REGION_ATTR.append("keep")

        stores.reset(account_id=ACCOUNT1, cross_region=False)

        assert stores[ACCOUNT1][EU_REGION].CROSS_REGION_ATTR == ["keep"]

    def test_cross_account_true_clears_universal(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION].CROSS_ACCOUNT_ATTR.append("global")
        other_account = stores[ACCOUNT2][EU_REGION]

        stores.reset(account_id=ACCOUNT1, cross_account=True)

        assert other_account.is_live()
        assert other_account.CROSS_ACCOUNT_ATTR == []

    def test_reset_unknown_account(self):
        stores = make_bundle()
        # resetting a never-accessed account simply materialises an empty scope
        stores.reset(account_id=ACCOUNT1)
        assert ACCOUNT1 in stores
        assert len(stores[ACCOUNT1]) == 0


class TestServiceScopeReset:
    def test_default_clears_everything(self):
        stores = make_bundle()
        store1 = stores[ACCOUNT1][EU_REGION]
        store2 = stores[ACCOUNT2][AP_REGION]
        store1.CROSS_ACCOUNT_ATTR.append("global")

        old_generation = stores.generation
        stores.reset()

        assert stores.generation is not old_generation
        assert not store1.is_live()
        assert not store2.is_live()
        assert stores[ACCOUNT1][EU_REGION].CROSS_ACCOUNT_ATTR == []
        assert stores[ACCOUNT1][EU_REGION].region_specific_attr == []

    def test_cross_account_false_preserves_universal(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION].CROSS_ACCOUNT_ATTR.append("global")

        stores.reset(cross_account=False)

        assert stores[ACCOUNT1][EU_REGION].CROSS_ACCOUNT_ATTR == ["global"]

    def test_cross_region_false_reattaches_preserved_data(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION].CROSS_REGION_ATTR.append("keep")
        stores[ACCOUNT1][EU_REGION].region_specific_attr.append("local")

        stores.reset(cross_region=False)

        new_store = stores[ACCOUNT1][EU_REGION]
        assert new_store.CROSS_REGION_ATTR == ["keep"]
        # local data is always cleared by a service reset
        assert new_store.region_specific_attr == []

    def test_pending_preservation_dropped_by_following_reset(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION].CROSS_REGION_ATTR.append("keep")

        # first reset preserves the envelope, but the account is not re-subscribed
        stores.reset(cross_region=False)
        # second reset rebuilds preservation from the (empty) live account set
        stores.reset(cross_region=False)

        assert stores[ACCOUNT1][EU_REGION].CROSS_REGION_ATTR == []


class TestResetApiValidation:
    def test_region_name_requires_account(self):
        stores = make_bundle()
        with pytest.raises(ValueError):
            stores.reset(region_name=EU_REGION)

    def test_invalid_account_in_validating_bundle(self):
        stores = make_bundle(validate=True)
        with pytest.raises(ValueError):
            stores.reset(account_id="bad-account")


#
# Iteration safety
#


class TestIterationSafety:
    def test_iterate_accounts_while_subscribing_new_ones(self):
        stores = make_bundle()
        stores[ACCOUNT1]
        seen = []
        for account_id in stores:
            seen.append(account_id)
            # new accounts first-accessed during traversal must not interrupt it
            stores[f"00000000000{len(seen)}"]

        assert seen == [ACCOUNT1]
        # and the newly added accounts are visible afterwards
        assert len(stores) == 2

    def test_iterate_regions_while_accessing_new_ones(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION]
        seen = []
        for region_name in stores[ACCOUNT1]:
            seen.append(region_name)
            stores[ACCOUNT1][f"region-{len(seen)}"]

        assert seen == [EU_REGION]
        assert len(stores[ACCOUNT1]) == 2

    def test_snapshot_methods_survive_concurrent_reset(self):
        stores = make_bundle()
        stores[ACCOUNT1][EU_REGION]
        errors = []

        def reset_worker():
            try:
                for _ in range(200):
                    stores.reset(account_id=ACCOUNT1)
                    stores[ACCOUNT1][EU_REGION]
            except Exception as e:  # noqa: BLE001 - fail loudly below
                errors.append(e)

        thread = threading.Thread(target=reset_worker)
        thread.start()
        try:
            for _ in range(200):
                stores.keys()
                stores.values()
                stores.items()
                stores[ACCOUNT1].keys()
                stores[ACCOUNT1].values()
                stores[ACCOUNT1].items()
                list(stores.iter_stores())
        finally:
            thread.join()

        assert not errors

    def test_iter_stores_during_concurrent_resets(self):
        stores = make_bundle()
        for account in (ACCOUNT1, ACCOUNT2):
            for region in (EU_REGION, AP_REGION):
                stores[account][region].region_specific_attr.append("data")

        stop = threading.Event()
        errors = []

        def reset_worker():
            seq = 0
            try:
                while not stop.is_set():
                    seq += 1
                    stores.reset(account_id=ACCOUNT1, region_name=EU_REGION)
                    stores[ACCOUNT1][EU_REGION]
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        thread = threading.Thread(target=reset_worker)
        thread.start()
        try:
            # every traversal must terminate without RuntimeError and yield
            # intact store objects
            for account_id, region_name, store in stores.iter_stores():
                # yielded objects are never mutated, even if superseded
                assert store._account_id == account_id
                assert store._region_name == region_name
        finally:
            stop.set()
            thread.join()

        assert not errors


#
# Concurrency: read either the complete old generation or the complete new one
#


class TestGenerationIsolationConcurrency:
    def test_concurrent_resets_and_reads_see_complete_generations(self):
        stores = make_bundle()
        frozen = [1, 2, 3]
        stores[ACCOUNT1][EU_REGION].region_specific_attr.extend(frozen)

        stop = threading.Event()
        errors = []
        observations = []
        lock = threading.Lock()

        def reader():
            while not stop.is_set():
                store = stores[ACCOUNT1][EU_REGION]
                try:
                    value = list(store.region_specific_attr)
                except StaleGenerationError:
                    continue
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    continue
                with lock:
                    observations.append(tuple(value))

        def resetter():
            try:
                while not stop.is_set():
                    stores.reset(account_id=ACCOUNT1, region_name=EU_REGION)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def creator():
            seq = 0
            try:
                while not stop.is_set():
                    seq += 1
                    account_id = f"{seq % 1000:012d}"
                    stores[account_id][f"region-{seq % 50}"]
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads += [threading.Thread(target=resetter) for _ in range(2)]
        threads += [threading.Thread(target=creator) for _ in range(2)]

        for thread in threads:
            thread.start()

        threading.Event().wait(1.0)
        stop.set()
        for thread in threads:
            thread.join()

        assert not errors
        assert observations
        # every read is either the complete frozen old generation or the
        # complete empty new generation; nothing in between.
        assert set(observations) <= {(1, 2, 3), ()}
        assert (1, 2, 3) in observations

    def test_generation_tokens_are_hierarchical(self):
        stores = make_bundle()
        store = stores[ACCOUNT1][EU_REGION]

        account_reset_generation_before = store.account_generation
        stores.reset(account_id=ACCOUNT1)
        new_store = stores[ACCOUNT1][EU_REGION]

        assert new_store.generation is store.generation  # root untouched
        assert new_store.account_generation is not account_reset_generation_before
        assert isinstance(new_store.generation, Generation)
