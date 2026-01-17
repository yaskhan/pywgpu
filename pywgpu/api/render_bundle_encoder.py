from typing import Any, List, Optional, TYPE_CHECKING
from pywgpu_types.descriptors import RenderBundleEncoderDescriptor

if TYPE_CHECKING:
    from .render_pipeline import RenderPipeline
    from .bind_group import BindGroup
    from .buffer import Buffer
    from .render_bundle import RenderBundle

class RenderBundleEncoder:
    """
    Encodes commands into a render bundle.
    
    Created with :meth:`Device.create_render_bundle_encoder`.
    """
    
    def __init__(self, inner: Any, descriptor: RenderBundleEncoderDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def set_pipeline(self, pipeline: 'RenderPipeline') -> None:
        pass

    def set_bind_group(
        self, 
        index: int, 
        bind_group: 'BindGroup', 
        offsets: List[int] = []
    ) -> None:
        pass

    def set_vertex_buffer(
        self, 
        slot: int, 
        buffer: 'Buffer', 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        pass

    def set_index_buffer(
        self, 
        buffer: 'Buffer', 
        index_format: str, 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        pass

    def draw(
        self, 
        vertices: int, 
        instances: int = 1, 
        first_vertex: int = 0, 
        first_instance: int = 0
    ) -> None:
        pass

    def draw_indexed(
        self, 
        indices: int, 
        instances: int = 1, 
        first_index: int = 0, 
        base_vertex: int = 0, 
        first_instance: int = 0
    ) -> None:
        pass

    def draw_indirect(self, indirect_buffer: 'Buffer', indirect_offset: int) -> None:
        pass

    def draw_indexed_indirect(self, indirect_buffer: 'Buffer', indirect_offset: int) -> None:
        pass

    def finish(self, descriptor: Optional[Any] = None) -> 'RenderBundle':
        """Finishes recording and returns a RenderBundle."""
        pass
