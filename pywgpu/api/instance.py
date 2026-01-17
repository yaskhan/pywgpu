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
        self._descriptor = descriptor or InstanceDescriptor()

    async def request_adapter(
        self, 
        options: Optional[RequestAdapterOptions] = None
    ) -> Optional['Adapter']:
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
        pass

    def create_surface(self, target: 'SurfaceTarget') -> 'Surface':
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
        pass

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
        pass

    def generate_report(self) -> dict:
        """
        Generates a report about the current state of the instance and all its 
        resources.
        
        Returns:
            A dictionary containing structural information about the instance's 
            internal state.
        """
        pass
