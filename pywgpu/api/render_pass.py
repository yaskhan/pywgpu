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
        if hasattr(self._inner, 'set_pipeline'):
            self._inner.set_pipeline(pipeline._inner)
        else:
            raise NotImplementedError("Backend does not support set_pipeline")

    def set_bind_group(
        self, 
        index: int, 
        bind_group: 'BindGroup', 
        offsets: List[int] = []
    ) -> None:
        """Sets the active bind group for a given bind group index."""
        if hasattr(self._inner, 'set_bind_group'):
            self._inner.set_bind_group(index, bind_group._inner, offsets)
        else:
            raise NotImplementedError("Backend does not support set_bind_group")

    def set_vertex_buffer(
        self, 
        slot: int, 
        buffer: 'Buffer', 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Sets the active vertex buffer for a given slot."""
        if hasattr(self._inner, 'set_vertex_buffer'):
            self._inner.set_vertex_buffer(slot, buffer._inner, offset, size)
        else:
            raise NotImplementedError("Backend does not support set_vertex_buffer")

    def set_index_buffer(
        self, 
        buffer: 'Buffer', 
        index_format: str, 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Sets the active index buffer."""
        if hasattr(self._inner, 'set_index_buffer'):
            self._inner.set_index_buffer(buffer._inner, index_format, offset, size)
        else:
            raise NotImplementedError("Backend does not support set_index_buffer")

    def draw(
        self, 
        vertices: int, 
        instances: int = 1, 
        first_vertex: int = 0, 
        first_instance: int = 0
    ) -> None:
        """Draws primitives."""
        if hasattr(self._inner, 'draw'):
            self._inner.draw(vertices, instances, first_vertex, first_instance)
        else:
            raise NotImplementedError("Backend does not support draw")

    def draw_indexed(
        self, 
        indices: int, 
        instances: int = 1, 
        first_index: int = 0, 
        base_vertex: int = 0, 
        first_instance: int = 0
    ) -> None:
        """Draws indexed primitives."""
        if hasattr(self._inner, 'draw_indexed'):
            self._inner.draw_indexed(indices, instances, first_index, base_vertex, first_instance)
        else:
            raise NotImplementedError("Backend does not support draw_indexed")

    def draw_indirect(self, indirect_buffer: 'Buffer', indirect_offset: int) -> None:
        """Draws primitives using parameters from a buffer."""
        if hasattr(self._inner, 'draw_indirect'):
            self._inner.draw_indirect(indirect_buffer._inner, indirect_offset)
        else:
            raise NotImplementedError("Backend does not support draw_indirect")

    def draw_indexed_indirect(self, indirect_buffer: 'Buffer', indirect_offset: int) -> None:
        """Draws indexed primitives using parameters from a buffer."""
        if hasattr(self._inner, 'draw_indexed_indirect'):
            self._inner.draw_indexed_indirect(indirect_buffer._inner, indirect_offset)
        else:
            raise NotImplementedError("Backend does not support draw_indexed_indirect")

    def draw_mesh_tasks(self, group_count_x: int, group_count_y: int, group_count_z: int) -> None:
        """Draws primitives using mesh shaders."""
        if hasattr(self._inner, 'draw_mesh_tasks'):
            self._inner.draw_mesh_tasks(group_count_x, group_count_y, group_count_z)
        else:
            raise NotImplementedError("Backend does not support draw_mesh_tasks")

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
        if hasattr(self._inner, 'set_viewport'):
            self._inner.set_viewport(x, y, width, height, min_depth, max_depth)
        else:
            raise NotImplementedError("Backend does not support set_viewport")

    def set_scissor_rect(self, x: int, y: int, width: int, height: int) -> None:
        """Sets the scissor rectangle."""
        if hasattr(self._inner, 'set_scissor_rect'):
            self._inner.set_scissor_rect(x, y, width, height)
        else:
            raise NotImplementedError("Backend does not support set_scissor_rect")

    def set_blend_constant(self, color: Union[List[float], Any]) -> None:
        """Sets the blend constant color."""
        if hasattr(self._inner, 'set_blend_constant'):
            self._inner.set_blend_constant(color)
        else:
            raise NotImplementedError("Backend does not support set_blend_constant")

    def set_stencil_reference(self, reference: int) -> None:
        """Sets the stencil reference value."""
        if hasattr(self._inner, 'set_stencil_reference'):
            self._inner.set_stencil_reference(reference)
        else:
            raise NotImplementedError("Backend does not support set_stencil_reference")

    def begin_occlusion_query(self, query_index: int) -> None:
        if hasattr(self._inner, 'begin_occlusion_query'):
            self._inner.begin_occlusion_query(query_index)
        else:
            raise NotImplementedError("Backend does not support begin_occlusion_query")

    def end_occlusion_query(self) -> None:
        if hasattr(self._inner, 'end_occlusion_query'):
            self._inner.end_occlusion_query()
        else:
            raise NotImplementedError("Backend does not support end_occlusion_query")

    def execute_bundles(self, bundles: List[Any]) -> None:
        """Executes the given render bundles."""
        if hasattr(self._inner, 'execute_bundles'):
            self._inner.execute_bundles(bundles)
        else:
            raise NotImplementedError("Backend does not support execute_bundles")

    def end(self) -> None:
        """Ends the render pass."""
        if hasattr(self._inner, 'end'):
            self._inner.end()
        else:
            raise NotImplementedError("Backend does not support end")
