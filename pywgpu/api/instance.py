from typing import Optional, List, Union, TYPE_CHECKING
from pydantic import Field
from pywgpu_types.descriptors import InstanceDescriptor, RequestAdapterOptions

if TYPE_CHECKING:
    from .adapter import Adapter
    from .surface import Surface, SurfaceTarget


class Instance:
    """
    Handle to an instance of wgpu.

    The Instance is the entry point for the wgpu API. It is used to create
    :class:`Surface` and request :class:`Adapter`.
    """

    def __init__(self, descriptor: Optional[InstanceDescriptor] = None) -> None:
        """
        Initializes a new wgpu instance.

        Args:
            descriptor: Description of the instance to create. If None, default
                backends and options will be used.
        """
        from ..backend import get_backend

        self._descriptor = descriptor or InstanceDescriptor()
        self._inner = get_backend(self._descriptor)

    async def request_adapter(
        self, options: Optional[RequestAdapterOptions] = None
    ) -> Optional["Adapter"]:
        """
        Requests an adapter from the instance.

        An adapter represents a specific graphics/compute device (e.g., a
        discrete GPU or integrated graphics).

        Args:
            options: Search options for the adapter, such as power preference
                or surface compatibility.

        Returns:
            The requested adapter, or None if no suitable adapter was found.
        """
        from .adapter import Adapter

        # Check if we have an inner implementation with request_adapter method
        if hasattr(self, "_inner") and hasattr(self._inner, "request_adapter"):
            adapter_inner = await self._inner.request_adapter(options)
            return Adapter(adapter_inner) if adapter_inner else None
        else:
            # For now, return None since we don't have a real backend
            # In a real implementation, this would use the system's GPU detection
            return None

    def create_surface(self, target: "SurfaceTarget") -> "Surface":
        """
        Creates a new surface targeting a given window/canvas/surface/etc.

        A surface represents a platform-specific graphics surface (e.g., a
        window or a web canvas) that can be drawn to.

        Args:
            target: The platform-specific target to create the surface for.

        Returns:
            A new surface targeting the given target.

        Raises:
            ValueError: If the target is invalid or not supported.
        """
        from .surface import Surface

        # Check if we have an inner implementation with create_surface method
        if hasattr(self, "_inner") and hasattr(self._inner, "create_surface"):
            surface_inner = self._inner.create_surface(target)
            return Surface(surface_inner)
        else:
            # For now, return a basic surface
            # In a real implementation, this would handle platform-specific surface creation
            raise NotImplementedError(
                "Surface creation requires backend implementation"
            )

    def poll_all_devices(self, force_wait: bool = False) -> bool:
        """
        Polls all devices currently in use by this instance.

        This can be used to advance asynchronous operations on all devices
        simultaneously.

        Args:
            force_wait: If True, this method will block until all devices
                have completed their current work.

        Returns:
            True if all devices are idle after polling.
        """
        # Check if we have an inner implementation
        if hasattr(self, "_inner") and hasattr(self._inner, "poll_all_devices"):
            return self._inner.poll_all_devices(force_wait)
        else:
            # For now, return True (no devices to poll)
            # In a real implementation, this would poll all devices
            return True

    def generate_report(self) -> dict:
        """
        Generates a report about the current state of the instance and all its
        resources.

        Returns:
            A dictionary containing structural information about the instance's
            internal state.
        """
        # Check if we have an inner implementation
        if hasattr(self, "_inner") and hasattr(self._inner, "generate_report"):
            return self._inner.generate_report()
        else:
            # Generate a basic report for the Python wrapper
            return {
                "instance_descriptor": {
                    "backends": getattr(self._descriptor, "backends", []),
                    "flags": getattr(self._descriptor, "flags", []),
                },
                "backend_support": {
                    "vulkan": False,
                    "direct3d12": False,
                    "metal": False,
                    "opengl": False,
                    "webgpu": False,
                },
                "active_devices": [],
                "created_surfaces": [],
            }
