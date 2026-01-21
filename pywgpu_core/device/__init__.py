from typing import Any, Optional
from .queue import Queue
from .life import LifetimeTracker
from .resource import Device, BufferDescriptor, TextureDescriptor
from .ops import DeviceOps
from .bgl import EntryMap, BindGroupLayoutEntry
from .ray_tracing import (
    RayTracing,
    TlasDescriptor,
    BlasDescriptor,
    TlasInstance,
    BlasGeometry,
)


class DeviceLogic(Device):
    """
    Main Device logic container.
    """

    def __init__(self, adapter: Any, desc: Any) -> None:
        super().__init__()
        self.queue = Queue(self)
        self.life = LifetimeTracker()
        self.ops = DeviceOps()
        self.ray_tracing = RayTracing()
        self.adapter = adapter
        self.desc = desc

    def get_device_id(self) -> Any:
        """Get the device ID.

        Returns:
            The device ID if registered in hub, None otherwise.
        """
        # Device ID is typically set when registered in hub
        return getattr(self, "_device_id", None)

    def get_queue_id(self) -> Any:
        """Get the queue ID.

        Returns:
            The queue ID if registered in hub, None otherwise.
        """
        # Queue ID is typically set when registered in hub
        return getattr(self.queue, "_queue_id", None)

    def get_adapter(self) -> Any:
        """Get the adapter."""
        return self.adapter

    def get_descriptor(self) -> Any:
        """Get the device descriptor."""
        return self.desc

    def is_valid(self) -> bool:
        """Check if the device is valid."""
        return self.valid

    def mark_invalid(self, reason: str = "Unknown") -> None:
        """Mark the device as invalid."""
        self.valid = False
        # Log the reason
        print(f"Device marked invalid: {reason}")

    def wait_idle(self) -> None:
        """Wait for the device to become idle.

        This blocks until all submitted work has completed.
        """
        # Get HAL device
        hal_device = getattr(self, "hal_device", None) or getattr(
            self, "_hal_device", None
        )
        if hal_device and hasattr(hal_device, "wait_idle"):
            try:
                hal_device.wait_idle()
            except Exception as e:
                raise RuntimeError(f"Failed to wait for device idle: {e}") from e

    def poll(self, maintain: Any = None) -> Any:
        """Poll device for events.

        Args:
            maintain: Maintenance level.

        Returns:
            Poll result.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_poll"):
            device_id = self.get_device_id()
            if device_id is not None:
                return self.ops.device_poll(device_id, maintain)
        return None

    def tick(self) -> None:
        """Tick the device.

        Processes pending operations and updates device state.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_tick"):
            device_id = self.get_device_id()
            if device_id is not None:
                self.ops.device_tick(device_id)

        # Tick lifetime tracker
        if hasattr(self.life, "tick"):
            self.life.tick()

    def destroy(self) -> None:
        """Destroy the device.

        Releases all resources and marks the device as invalid.
        """
        # Mark device as invalid
        self.mark_invalid("Device destroyed")

        # Delegate to ops if available
        if hasattr(self.ops, "device_destroy"):
            device_id = self.get_device_id()
            if device_id is not None:
                self.ops.device_destroy(device_id)

    def set_lost_closure(self, closure: Any) -> None:
        """Set lost closure for device.

        Args:
            closure: Callback to invoke when device is lost.
        """
        # Store closure
        self._lost_closure = closure

        # Delegate to ops if available
        if hasattr(self.ops, "device_set_lost_closure"):
            device_id = self.get_device_id()
            if device_id is not None:
                self.ops.device_set_lost_closure(device_id, closure)

    def get_timestamp_period(self) -> float:
        """Get timestamp period for device.

        Returns:
            Timestamp period in nanoseconds.
        """
        return self.queue.get_timestamp_period()

    def get_hal_counters(self) -> Any:
        """Get HAL counters for device.

        Returns:
            Dictionary of HAL counters.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_get_hal_counters"):
            device_id = self.get_device_id()
            if device_id is not None:
                return self.ops.device_get_hal_counters(device_id)
        return {}

    def generate_allocator_report(self) -> Any:
        """Generate allocator report for device.

        Returns:
            Allocator report or None.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_generate_allocator_report"):
            device_id = self.get_device_id()
            if device_id is not None:
                return self.ops.device_generate_allocator_report(device_id)
        return None

    def configure_surface(self, surface_id: Any, config: Any) -> Optional[Any]:
        """Configure surface for device.

        Args:
            surface_id: The surface ID.
            config: Surface configuration.

        Returns:
            Error if configuration fails, None otherwise.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_configure_surface"):
            device_id = self.get_device_id()
            if device_id is not None:
                return self.ops.device_configure_surface(device_id, surface_id, config)
        return None

    def create_render_bundle_encoder(self, desc: Any) -> Any:
        """Create a render bundle encoder.

        Args:
            desc: Render bundle encoder descriptor.

        Returns:
            The render bundle encoder.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_create_render_bundle_encoder"):
            device_id = self.get_device_id()
            if device_id is not None:
                encoder_id, error = self.ops.device_create_render_bundle_encoder(
                    device_id, desc
                )
                if error:
                    raise RuntimeError(
                        f"Failed to create render bundle encoder: {error}"
                    )
                return encoder_id
        return None

    def finish_render_bundle(self, bundle_encoder: Any, desc: Any) -> Any:
        """Finish a render bundle.

        Args:
            bundle_encoder: The render bundle encoder.
            desc: Render bundle descriptor.

        Returns:
            The render bundle.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "render_bundle_encoder_finish"):
            bundle_id, error = self.ops.render_bundle_encoder_finish(
                bundle_encoder, desc
            )
            if error:
                raise RuntimeError(f"Failed to finish render bundle: {error}")
            return bundle_id
        return None

    def create_pipeline_cache(self, desc: Any) -> Any:
        """Create a pipeline cache.

        Args:
            desc: Pipeline cache descriptor.

        Returns:
            The pipeline cache.
        """
        # Delegate to ops if available
        if hasattr(self.ops, "device_create_pipeline_cache"):
            device_id = self.get_device_id()
            if device_id is not None:
                cache_id, error = self.ops.device_create_pipeline_cache(device_id, desc)
                if error:
                    raise RuntimeError(f"Failed to create pipeline cache: {error}")
                return cache_id
        return None
