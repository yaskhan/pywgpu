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
        if hasattr(self._inner, 'get_capabilities'):
            return self._inner.get_capabilities(adapter._inner)
        else:
            # Return some basic default capabilities for mock purposes
            return {
                "formats": ["rgba8unorm"],
                "present_modes": ["fifo"],
                "alpha_modes": ["opaque"],
            }


    def get_current_texture(self) -> 'SurfaceTexture':
        """
        Returns the next texture to be presented to the surface.

        Returns:
            A SurfaceTexture object.

        Raises:
            SurfaceError: If a texture could not be acquired (e.g. timeout, outdated).
        """
        from .surface_texture import SurfaceTexture
        if hasattr(self._inner, 'get_current_texture'):
            texture_inner = self._inner.get_current_texture()
            return SurfaceTexture(texture_inner)
        else:
            raise NotImplementedError("Backend does not support get_current_texture")

    def configure(self, device: 'Device', config: SurfaceConfiguration) -> None:
        """
        Configures the surface for presentation.
        
        Args:
            device: The device to present with.
            config: Configuration (format, usage, size, etc.).
        """
        if hasattr(self._inner, 'configure'):
            self._inner.configure(device._inner, config)
        else:
            raise NotImplementedError("Backend does not support configure")

    def unconfigure(self) -> None:
        """Removes the configuration from the surface."""
        if hasattr(self._inner, 'unconfigure'):
            self._inner.unconfigure()
        else:
            raise NotImplementedError("Backend does not support unconfigure")
