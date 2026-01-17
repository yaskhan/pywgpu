from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import BindGroupDescriptor

class BindGroup:
    """
    Handle to a bind group.
    
    A BindGroup represents a set of resources (buffers, textures, samplers) 
    bound together for use by a shader.
    
    Created with :meth:`Device.create_bind_group`.
    """
    
    def __init__(self, inner: Any, descriptor: BindGroupDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor
