from typing import Optional, TYPE_CHECKING, Any
from pywgpu_types.descriptors import TextureDescriptor

if TYPE_CHECKING:
    from .texture_view import TextureView

class Texture:
    """
    Handle to a texture on the GPU.
    
    Created with :meth:`Device.create_texture`.
    """
    
    def __init__(self, inner: Any, descriptor: TextureDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    @property
    def width(self) -> int:
        """The width of the texture."""
        return self._descriptor.get('size', {}).get('width', 0)

    @property
    def height(self) -> int:
        """The height of the texture."""
        return self._descriptor.get('size', {}).get('height', 1)

    @property
    def depth_or_array_layers(self) -> int:
        """The depth or layer count of the texture."""
        return self._descriptor.get('size', {}).get('depth_or_array_layers', 1)

    @property
    def mip_level_count(self) -> int:
        """The number of mip levels."""
        return self._descriptor.get('mip_level_count', 1)

    @property
    def sample_count(self) -> int:
        """The number of samples per pixel."""
        return self._descriptor.get('sample_count', 1)

    @property
    def dimension(self) -> str:
        """The dimension of the texture (1d, 2d, 3d)."""
        return self._descriptor.get('dimension', '2d')

    @property
    def format(self) -> str:
        """The texture format."""
        return self._descriptor.get('format', '')

    @property
    def usage(self) -> int:
        """The allowed usage flags."""
        return self._descriptor.get('usage', 0)

    def create_view(self, descriptor: Optional[Any] = None) -> 'TextureView':
        """
        Creates a view of this texture.
        
        Args:
            descriptor: View descriptor. If None, default view is created.
        """
        from .texture_view import TextureView
        if hasattr(self._inner, 'create_view'):
            view_inner = self._inner.create_view(descriptor)
            return TextureView(view_inner, self)
        else:
            # Fallback for mock
            return TextureView(None, self)

    def destroy(self) -> None:
        """Destroys the texture."""
        if hasattr(self._inner, 'destroy'):
            self._inner.destroy()
        # If no inner destroy method, do nothing (resources managed by backend)

    def as_image_copy(self) -> Any:
        """Returns an ImageCopyTexture referring to this entire texture."""
        from .queue import ImageCopyTexture
        return ImageCopyTexture(
            texture=self,
            mip_level=0,
            origin=[0, 0, 0],
            aspect='all'
        )

