from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import ExternalTextureDescriptor

class ExternalTexture:
    """
    Handle to a texture imported from an external source (e.g. video, canvas).
    
    Created with :meth:`Device.create_external_texture`.
    """
    
    def __init__(self, inner: Any, descriptor: ExternalTextureDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def destroy(self) -> None:
        """Destroys the external texture."""
        if hasattr(self._inner, 'destroy'):
            self._inner.destroy()
        # If no inner destroy method, do nothing (resources managed by backend)
