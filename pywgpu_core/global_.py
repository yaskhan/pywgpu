from typing import Any, Dict, Optional
from .hub import Hub
from .registry import Registry
from .instance import Instance
from .device import Device
from .device.life import WaitIdleError
from .device import DeviceError
from pywgpu_types.instance import InstanceDescriptor
from pywgpu_types.adapter import RequestAdapterOptions
from pywgpu_types.device import DeviceDescriptor


class Global:
    """
    Entrance to the wgpu-core logic.

    This mirrors the Global struct from wgpu-core/src/global.rs
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
        self, adapter_id: Any, desc: DeviceDescriptor, trace_path: Optional[str] = None
    ) -> Any:
        """
        Request a device from an adapter.

        Logic from wgpu-core/src/device/global.rs

        Args:
            adapter_id: The adapter ID to use.
            desc: Device descriptor.
            trace_path: Optional path for tracing.

        Returns:
            A device ID and queue ID.

        Raises:
            DeviceError: If the adapter doesn't exist or device creation fails.
            WaitIdleError: If there's an error during device initialization.
        """
        # Get the adapter
        adapter = self.hub.adapters.get(adapter_id)

        try:
            # Create the device using the adapter
            device = Device.new(adapter, desc, trace_path)

            # Prepare IDs for device and queue
            device_id = self.hub.devices.prepare(None).assign(device)
            queue_id = self.hub.queues.prepare(None).assign(device.queue)

            return (device_id, queue_id)

        except (DeviceError, WaitIdleError) as e:
            # Create invalid device ID if creation fails
            device_id = self.hub.devices.prepare(None).assign(
                Device.new_invalid(adapter, desc.label, str(e))
            )
            queue_id = self.hub.queues.prepare(None).assign(None)
            raise e
