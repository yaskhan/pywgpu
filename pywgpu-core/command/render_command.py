"""
Render commands for command encoding.

This module defines render commands that can be recorded into a render pass.
These commands are used to encode render operations for execution on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RenderCommand:
    """
    Render command for encoding.
    
    This enum represents different types of render commands that can be
    recorded into a render pass.
    
    Attributes:
        set_bind_group: Set a bind group.
        set_pipeline: Set a render pipeline.
        set_index_buffer: Set an index buffer.
        set_vertex_buffer: Set a vertex buffer.
        set_immediate: Set immediate data.
        draw: Draw geometry.
        draw_indexed: Draw indexed geometry.
        draw_mesh_tasks: Draw mesh tasks.
        draw_indirect: Draw geometry indirectly.
        multi_draw_indirect_count: Draw geometry indirectly with count.
        push_debug_group: Push a debug group.
        pop_debug_group: Pop a debug group.
        insert_debug_marker: Insert a debug marker.
        write_timestamp: Write a timestamp.
        begin_occlusion_query: Begin an occlusion query.
        end_occlusion_query: End an occlusion query.
        begin_pipeline_statistics_query: Begin a pipeline statistics query.
        end_pipeline_statistics_query: End a pipeline statistics query.
        execute_bundle: Execute a render bundle.
        set_blend_constant: Set blend constant.
        set_stencil_reference: Set stencil reference.
        set_viewport: Set viewport.
        set_scissor: Set scissor.
    """

    SET_BIND_GROUP = "set_bind_group"
    SET_PIPELINE = "set_pipeline"
    SET_INDEX_BUFFER = "set_index_buffer"
    SET_VERTEX_BUFFER = "set_vertex_buffer"
    SET_IMMEDIATE = "set_immediate"
    DRAW = "draw"
    DRAW_INDEXED = "draw_indexed"
    DRAW_MESH_TASKS = "draw_mesh_tasks"
    DRAW_INDIRECT = "draw_indirect"
    MULTI_DRAW_INDIRECT_COUNT = "multi_draw_indirect_count"
    PUSH_DEBUG_GROUP = "push_debug_group"
    POP_DEBUG_GROUP = "pop_debug_group"
    INSERT_DEBUG_MARKER = "insert_debug_marker"
    WRITE_TIMESTAMP = "write_timestamp"
    BEGIN_OCCLUSION_QUERY = "begin_occlusion_query"
    END_OCCLUSION_QUERY = "end_occlusion_query"
    BEGIN_PIPELINE_STATISTICS_QUERY = "begin_pipeline_statistics_query"
    END_PIPELINE_STATISTICS_QUERY = "end_pipeline_statistics_query"
    EXECUTE_BUNDLE = "execute_bundle"
    SET_BLEND_CONSTANT = "set_blend_constant"
    SET_STENCIL_REFERENCE = "set_stencil_reference"
    SET_VIEWPORT = "set_viewport"
    SET_SCISSOR = "set_scissor"


@dataclass
class SetBindGroup:
    """
    Command to set a bind group.
    
    Attributes:
        index: The bind group index.
        num_dynamic_offsets: Number of dynamic offsets.
        bind_group: The bind group to set.
    """

    index: int
    num_dynamic_offsets: int
    bind_group: Optional[Any]


@dataclass
class SetPipeline:
    """
    Command to set a render pipeline.
    
    Attributes:
        pipeline: The render pipeline.
    """

    pipeline: Any


@dataclass
class SetIndexBuffer:
    """
    Command to set an index buffer.
    
    Attributes:
        buffer: The buffer to set.
        index_format: The index format.
        offset: Offset into the buffer.
        size: Size of the data to use.
    """

    buffer: Any
    index_format: Any
    offset: int
    size: Optional[int]


@dataclass
class SetVertexBuffer:
    """
    Command to set a vertex buffer.
    
    Attributes:
        slot: The vertex buffer slot.
        buffer: The buffer to set.
        offset: Offset into the buffer.
        size: Size of the data to use.
    """

    slot: int
    buffer: Any
    offset: int
    size: Optional[int]


@dataclass
class SetImmediate:
    """
    Command to set immediate data.
    
    Attributes:
        offset: Byte offset within immediate data storage.
        size_bytes: Number of bytes to write.
        values_offset: Index in immediates_data of the start of data.
    """

    offset: int
    size_bytes: int
    values_offset: int


@dataclass
class Draw:
    """
    Command to draw geometry.
    
    Attributes:
        vertex_count: Number of vertices to draw.
        instance_count: Number of instances to draw.
        first_vertex: Index of the first vertex.
        first_instance: Index of the first instance.
    """

    vertex_count: int
    instance_count: int
    first_vertex: int
    first_instance: int


@dataclass
class DrawIndexed:
    """
    Command to draw indexed geometry.
    
    Attributes:
        index_count: Number of indices to draw.
        instance_count: Number of instances to draw.
        first_index: Index of the first index.
        base_vertex: Base vertex offset.
        first_instance: Index of the first instance.
    """

    index_count: int
    instance_count: int
    first_index: int
    base_vertex: int
    first_instance: int


@dataclass
class DrawMeshTasks:
    """
    Command to draw mesh tasks.
    
    Attributes:
        group_count_x: Workgroup count in X dimension.
        group_count_y: Workgroup count in Y dimension.
        group_count_z: Workgroup count in Z dimension.
    """

    group_count_x: int
    group_count_y: int
    group_count_z: int


@dataclass
class DrawIndirect:
    """
    Command to draw geometry indirectly.
    
    Attributes:
        buffer: The buffer containing draw parameters.
        offset: Offset into the buffer.
        count: Number of draws (for multi-draw).
        family: Draw command family.
        vertex_or_index_limit: Vertex or index limit.
        instance_limit: Instance limit.
    """

    buffer: Any
    offset: int
    count: int
    family: Any
    vertex_or_index_limit: Optional[int]
    instance_limit: Optional[int]


@dataclass
class MultiDrawIndirectCount:
    """
    Command to draw geometry indirectly with count.
    
    Attributes:
        buffer: The buffer containing draw parameters.
        offset: Offset into the buffer.
        count_buffer: Buffer containing the draw count.
        count_buffer_offset: Offset into the count buffer.
        max_count: Maximum number of draws.
        family: Draw command family.
    """

    buffer: Any
    offset: int
    count_buffer: Any
    count_buffer_offset: int
    max_count: int
    family: Any


@dataclass
class PushDebugGroup:
    """
    Command to push a debug group.
    
    Attributes:
        color: Color for the debug group.
        len: Length of the string data.
    """

    color: int
    len: int


@dataclass
class PopDebugGroup:
    """Command to pop a debug group."""
    pass


@dataclass
class InsertDebugMarker:
    """
    Command to insert a debug marker.
    
    Attributes:
        color: Color for the marker.
        len: Length of the string data.
    """

    color: int
    len: int


@dataclass
class WriteTimestamp:
    """
    Command to write a timestamp.
    
    Attributes:
        query_set: The query set to write to.
        query_index: The query index.
    """

    query_set: Any
    query_index: int


@dataclass
class BeginOcclusionQuery:
    """
    Command to begin an occlusion query.
    
    Attributes:
        query_index: The query index.
    """

    query_index: int


@dataclass
class EndOcclusionQuery:
    """Command to end an occlusion query."""
    pass


@dataclass
class BeginPipelineStatisticsQuery:
    """
    Command to begin a pipeline statistics query.
    
    Attributes:
        query_set: The query set to use.
        query_index: The query index.
    """

    query_set: Any
    query_index: int


@dataclass
class EndPipelineStatisticsQuery:
    """Command to end a pipeline statistics query."""
    pass


@dataclass
class ExecuteBundle:
    """
    Command to execute a render bundle.
    
    Attributes:
        bundle: The render bundle to execute.
    """

    bundle: Any


@dataclass
class SetBlendConstant:
    """
    Command to set blend constant.
    
    Attributes:
        color: The blend constant color.
    """

    color: Any


@dataclass
class SetStencilReference:
    """
    Command to set stencil reference.
    
    Attributes:
        reference: The stencil reference value.
    """

    reference: int


@dataclass
class SetViewport:
    """
    Command to set viewport.
    
    Attributes:
        x: X coordinate.
        y: Y coordinate.
        w: Width.
        h: Height.
        min_depth: Minimum depth.
        max_depth: Maximum depth.
    """

    x: float
    y: float
    w: float
    h: float
    min_depth: float
    max_depth: float


@dataclass
class SetScissor:
    """
    Command to set scissor.
    
    Attributes:
        x: X coordinate.
        y: Y coordinate.
        w: Width.
        h: Height.
    """

    x: int
    y: int
    w: int
    h: int
