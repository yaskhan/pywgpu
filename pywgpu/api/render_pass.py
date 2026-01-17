from typing import Any, Optional, Union, List, TYPE_CHECKING
from pywgpu_types.pass_desc import RenderPassDescriptor

if TYPE_CHECKING:
    from .render_pipeline import RenderPipeline
    from .bind_group import BindGroup
    from .buffer import Buffer
    from .query_set import QuerySet

class RenderPass:
    """
    In-progress recording of a render pass.
    
    Created with :meth:`CommandEncoder.begin_render_pass`.
    """
    
    def __init__(self, inner: Any, descriptor: RenderPassDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def set_pipeline(self, pipeline: 'RenderPipeline') -> None:
        """Sets the active render pipeline via a handle."""
        pass

    def set_bind_group(
        self, 
        index: int, 
        bind_group: 'BindGroup', 
        offsets: List[int] = []
    ) -> None:
        """Sets the active bind group for a given bind group index."""
        pass

    def set_vertex_buffer(
        self, 
        slot: int, 
        buffer: 'Buffer', 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Sets the active vertex buffer for a given slot."""
        pass

    def set_index_buffer(
        self, 
        buffer: 'Buffer', 
        index_format: str, 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Sets the active index buffer."""
        pass

    def draw(
        self, 
        vertices: int, 
        instances: int = 1, 
        first_vertex: int = 0, 
        first_instance: int = 0
    ) -> None:
        """Draws primitives."""
        pass

    def draw_indexed(
        self, 
        indices: int, 
        instances: int = 1, 
        first_index: int = 0, 
        base_vertex: int = 0, 
        first_instance: int = 0
    ) -> None:
        """Draws indexed primitives."""
        pass

    def draw_indirect(self, indirect_buffer: 'Buffer', indirect_offset: int) -> None:
        """Draws primitives using parameters from a buffer."""
        pass

    def draw_indexed_indirect(self, indirect_buffer: 'Buffer', indirect_offset: int) -> None:
        """Draws indexed primitives using parameters from a buffer."""
        pass

    def set_viewport(
        self, 
        x: float, 
        y: float, 
        width: float, 
        height: float, 
        min_depth: float, 
        max_depth: float
    ) -> None:
        """Sets the viewport."""
        pass

    def set_scissor_rect(self, x: int, y: int, width: int, height: int) -> None:
        """Sets the scissor rectangle."""
        pass

    def set_blend_constant(self, color: Union[List[float], Any]) -> None:
        """Sets the blend constant color."""
        pass

    def set_stencil_reference(self, reference: int) -> None:
        """Sets the stencil reference value."""
        pass

    def begin_occlusion_query(self, query_index: int) -> None:
        pass

    def end_occlusion_query(self) -> None:
        pass

    def execute_bundles(self, bundles: List[Any]) -> None:
        """Executes the given render bundles."""
        pass

    def end(self) -> None:
        """Ends the render pass."""
        pass
