from typing import Any, Optional, TYPE_CHECKING
from pywgpu_types.surface import SurfaceConfiguration

if TYPE_CHECKING:
    from .adapter import Adapter
    from .device import Device
    from .texture import TextureFormat
    from .surface_texture import SurfaceTexture

class Surface:
    """
    Handle to a presentable surface.
    
    A Surface represents a platform-specific window or canvas that can be 
    presented to.
    
    Created with :meth:`Instance.create_surface`.
    """
    
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def get_capabilities(self, adapter: 'Adapter') -> Any:
        """
        Returns the capabilities of the surface when used with the given adapter.
        
        Returns:
            SurfaceCapabilities object containing supported formats, present modes, etc.
        """
        pass

    def get_current_texture(self) -> 'SurfaceTexture':
        """
        Returns the next texture to be presented to the surface.
        
        Returns:
            A SurfaceTexture object.
            
        Raises:
            SurfaceError: If a texture could not be acquired (e.g. timeout, outdated).
        """
        pass

    def configure(self, device: 'Device', config: SurfaceConfiguration) -> None:
        """
        Configures the surface for presentation.
        
        Args:
            device: The device to present with.
            config: Configuration (format, usage, size, etc.).
        """
        pass

    def unconfigure(self) -> None:
        """Removes the configuration from the surface."""
        pass
