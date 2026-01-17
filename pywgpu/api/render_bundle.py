from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import RenderBundleDescriptor

class RenderBundle:
    """
    Handle to a pre-recorded bundle of commands.
    
    Created with :meth:`RenderBundleEncoder.finish`.
    """
    
    def __init__(self, inner: Any, descriptor: RenderBundleDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor
