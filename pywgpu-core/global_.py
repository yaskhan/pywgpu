from typing import Any, Dict, Optional
from .hub import Hub
from .registry import Registry
from .instance import Instance
from pywgpu_types.instance import InstanceDescriptor
from pywgpu_types.adapter import RequestAdapterOptions


class Global:
    """
    Entrance to the wgpu-core logic.
    """

    def __init__(self, name: str, descriptor: InstanceDescriptor) -> None:
        self.instance = Instance(name, descriptor)
        self.hub = Hub()
        self.surfaces = Registry()

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a report of all resources.

        Returns:
            A dictionary containing reports from the hub and instance.
        """
        hub_report = self.hub.generate_report()
        return {
            "hub": hub_report,
            "surfaces": self.surfaces.generate_report(),
        }

    # Method implementations mirroring crate::Global
    def instance_request_adapter(self, options: RequestAdapterOptions) -> Any:
        """
        Request an adapter from the instance.

        Args:
            options: RequestAdapterOptions describing the requirements.

        Returns:
            An adapter ID or adapter info.
        """
        adapter_info = self.instance.request_adapter(options)
        # Store the adapter in the hub
        future_id = self.hub.adapters.prepare(None)
        adapter_id = future_id.assign(adapter_info)
        return adapter_id

    def adapter_request_device(
        self, adapter_id: Any, desc: Any, trace_path: Optional[str] = None
    ) -> Any:
        """
        Request a device from an adapter.

        Args:
            adapter_id: The adapter ID to use.
            desc: Device descriptor.
            trace_path: Optional path for tracing.

        Returns:
            A device ID and queue ID.

        Raises:
            Exception: If the adapter doesn't exist or device creation fails.
        """
        # Get the adapter
        adapter = self.hub.adapters.get(adapter_id)

        # TODO: Implement actual device creation
        # For now, return placeholder IDs
        device_id = self.hub.devices.prepare(None).assign(None)
        queue_id = self.hub.queues.prepare(None).assign(None)

        return (device_id, queue_id)
