from typing import Optional, Tuple, TYPE_CHECKING, Any
from pywgpu_types.descriptors import DeviceDescriptor

if TYPE_CHECKING:
    from .device import Device
    from .queue import Queue
    from .surface import Surface

class Adapter:
    """
    Handle to a physical graphics and/or compute device.
    
    An Adapter represents a specific GPU (discrete or integrated) or a 
    software renderer. It is used to query capabilities and request a 
    logical :class:`Device`.
    """
    
    def __init__(self, inner: Any) -> None:
        """
        Internal constructor. Use Instance.request_adapter to create an Adapter.
        """
        self._inner = inner

    async def request_device(
        self,
        descriptor: Optional[DeviceDescriptor] = None
    ) -> Tuple['Device', 'Queue']:
        """
        Requests a connection to a physical device (a logical device).

        Args:
            descriptor: Description of the logical device to create,
                including required features and limits.

        Returns:
            A tuple containing the :class:`Device` and its associated :class:`Queue`.

        Raises:
            RequestDeviceError: If the device could not be created.
        """
        from .device import Device
        from .queue import Queue

        # Delegate to the inner implementation
        if hasattr(self._inner, 'request_device'):
            device_inner, queue_inner = await self._inner.request_device(descriptor)
            return Device(device_inner), Queue(queue_inner)
        else:
            raise NotImplementedError("Backend does not support request_device")

    def features(self) -> Any:
        """
        Returns the features that this adapter supports.
        
        Returns:
            A set-like object containing supported :class:`FeatureName`.
        """
        if hasattr(self._inner, 'features'):
            return self._inner.features()
        else:
            raise NotImplementedError("Backend does not support features")

    def limits(self) -> Any:
        """
        Returns the limits that this adapter supports.
        
        Returns:
            A :class:`Limits` object containing hardware constraints.
        """
        if hasattr(self._inner, 'limits'):
            return self._inner.limits()
        else:
            raise NotImplementedError("Backend does not support limits")

    def get_info(self) -> Any:
        """
        Returns information about the adapter.
        
        Returns:
            An :class:`AdapterInfo` object containing name, vendor, etc.
        """
        if hasattr(self._inner, 'get_info'):
            return self._inner.get_info()
        else:
            raise NotImplementedError("Backend does not support get_info")

    def is_surface_supported(self, surface: 'Surface') -> bool:
        """
        Returns whether the given surface is supported by this adapter.

        Args:
            surface: The surface to check.

        Returns:
            True if the adapter can present to the surface.
        """
        if hasattr(self._inner, 'is_surface_supported'):
            return self._inner.is_surface_supported(surface._inner)
        else:
            raise NotImplementedError("Backend does not support is_surface_supported")

    def get_downlevel_capabilities(self) -> Any:
        """
        Returns the downlevel capabilities of this adapter.
        
        This is useful for compatibility with older hardware.
        """
        if hasattr(self._inner, 'get_downlevel_capabilities'):
            return self._inner.get_downlevel_capabilities()
        else:
            raise NotImplementedError("Backend does not support get_downlevel_capabilities")

    def get_texture_format_features(self, format: str) -> Any:
        """
        Returns the features supported by a specific texture format on this adapter.
        
        Args:
            format: The texture format to query.
        """
        if hasattr(self._inner, 'get_texture_format_features'):
            return self._inner.get_texture_format_features(format)
        else:
            raise NotImplementedError("Backend does not support get_texture_format_features")
