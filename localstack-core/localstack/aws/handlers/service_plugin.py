"""Handlers extending the base logic of service handlers with lazy-loading and plugin mechanisms."""

import logging
import threading

from plux import PluginDisabled

from localstack.http import Response
from localstack.services.plugins import Service, ServiceManager
from localstack.utils.sync import SynchronizedDefaultDict

from ...utils.bootstrap import is_api_enabled
from ..api import RequestContext
from ..chain import Handler, HandlerChain
from ..protocol.service_router import determine_aws_service_model_for_data_plane
from .service import PluginNotIncludedInUserLicenseError, ServiceRequestRouter

LOG = logging.getLogger(__name__)

# RequestContext attribute holding per-request finalizers (e.g. in-flight request counters). The
# ``ServiceRequestFinalizer`` chain finalizer drains this list at the very end of every request.
SERVICE_REQUEST_FINALIZERS = "service_request_finalizers"
# RequestContext attribute holding the services already tracked as in flight for the current request,
# ensuring the data-plane loader and the regular loader never count the same request twice.
SERVICE_REQUEST_TRACKED = "service_request_tracked"


class ServiceLoader(Handler):
    def __init__(
        self, service_manager: ServiceManager, service_request_router: ServiceRequestRouter
    ):
        """
        This handler encapsulates service lazy-loading. It loads services from the given ServiceManager and uses them
        to populate the given ServiceRequestRouter.

        :param service_manager: the service manager used to load services
        :param service_request_router: the service request router to populate
        """
        self.service_manager = service_manager
        self.service_request_router = service_request_router
        self.service_locks = SynchronizedDefaultDict(threading.RLock)
        self.loaded_services = set()

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        return self.require_service(chain, context, response)

    def require_service(self, _: HandlerChain, context: RequestContext, response: Response):
        if not context.service:
            return

        service_name: str = context.service.service_name

        if service_name not in self.loaded_services:
            if not self.service_manager.exists(service_name):
                raise NotImplementedError
            elif not is_api_enabled(service_name):
                raise NotImplementedError(
                    f"Service '{service_name}' is not enabled. Please check your 'SERVICES' configuration variable."
                )

            request_router = self.service_request_router
            try:
                # Ensure the Service is loaded and set to ServiceState.RUNNING if not in an erroneous state.
                service_plugin: Service = self.service_manager.require(service_name)
            except PluginDisabled as e:
                if e.reason == "This feature is not part of the active license agreement":
                    raise PluginNotIncludedInUserLicenseError()
                raise

            with self.service_locks[service_name]:
                # try again to avoid race conditions
                if service_name not in self.loaded_services:
                    self.loaded_services.add(service_name)
                    if isinstance(service_plugin, Service):
                        request_router.add_skeleton(service_plugin.skeleton)
                    else:
                        LOG.warning(
                            "found plugin for '%s', but cannot attach service plugin of type '%s'",
                            service_name,
                            type(service_plugin),
                        )
        else:
            # already registered in the router, but still consult the shared lifecycle state machine on
            # every request, so the handler chain judges ERROR / STOPPING / DISABLED states exactly like
            # the service manager and the shutdown hooks do.
            self.service_manager.require(service_name)

        self._track_in_flight_request(context, service_name)

    def _track_in_flight_request(self, context: RequestContext, service_name: str) -> None:
        """
        Registers this request as in flight against the service for its entire chain lifetime: the matching
        ``end_request`` is executed by the ``ServiceRequestFinalizer`` once the chain finalizes (i.e. after
        the service request has been fully dispatched, including on errors or early termination).
        """
        ticket = self.service_manager.begin_request(service_name)
        if ticket is None:
            # the service is not currently serving requests (stopping/stopped/disabled) - nothing to track
            return

        tracked_services = context.get(SERVICE_REQUEST_TRACKED)
        if tracked_services is None:
            tracked_services = set()
            setattr(context, SERVICE_REQUEST_TRACKED, tracked_services)
        if service_name in tracked_services:
            # the data-plane loader may already have registered this request
            self.service_manager.end_request(service_name, ticket)
            return
        tracked_services.add(service_name)

        finalizers = context.get(SERVICE_REQUEST_FINALIZERS)
        if finalizers is None:
            finalizers = []
            setattr(context, SERVICE_REQUEST_FINALIZERS, finalizers)

        def _finalize(*_args):
            self.service_manager.end_request(service_name, ticket)

        finalizers.append(_finalize)


class ServiceRequestFinalizer(Handler):
    """
    Chain finalizer draining per-request finalizers registered on the context (currently the in-flight
    request bookkeeping of the ``ServiceLoader``). Runs exactly once at the end of every request, including
    after exceptions and terminated chains.
    """

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        finalizers = context.get(SERVICE_REQUEST_FINALIZERS)
        if not finalizers:
            return
        for finalizer in reversed(list(finalizers)):
            try:
                finalizer(chain, context, response)
            except Exception as e:
                LOG.debug("error while running service request finalizer: %s", e)


class ServiceLoaderForDataPlane(Handler):
    """
    Specific lightweight service loader that loads services based only on hostname indicators. This allows
    us to correctly load services when things like lambda function URLs or APIGW REST APIs are called
    before the services were actually loaded.
    """

    def __init__(self, service_loader: ServiceLoader):
        self.service_loader = service_loader

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        if context.service:
            return

        if service := determine_aws_service_model_for_data_plane(context.request):
            context.service = service
            self.service_loader.require_service(chain, context, response)
