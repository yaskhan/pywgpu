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
        pass

    def features(self) -> Any:
        """
        Returns the features that this adapter supports.
        
        Returns:
            A set-like object containing supported :class:`FeatureName`.
        """
        pass

    def limits(self) -> Any:
        """
        Returns the limits that this adapter supports.
        
        Returns:
            A :class:`Limits` object containing hardware constraints.
        """
        pass

    def get_info(self) -> Any:
        """
        Returns information about the adapter.
        
        Returns:
            An :class:`AdapterInfo` object containing name, vendor, etc.
        """
        pass

    def is_surface_supported(self, surface: 'Surface') -> bool:
        """
        Returns whether the given surface is supported by this adapter.
        
        Args:
            surface: The surface to check.
            
        Returns:
            True if the adapter can present to the surface.
        """
        pass

    def get_downlevel_capabilities(self) -> Any:
        """
        Returns the downlevel capabilities of this adapter.
        
        This is useful for compatibility with older hardware.
        """
        pass

    def get_texture_format_features(self, format: str) -> Any:
        """
        Returns the features supported by a specific texture format on this adapter.
        
        Args:
            format: The texture format to query.
        """
        pass
