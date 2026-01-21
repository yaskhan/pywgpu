"""
Encoder commands for command encoding.

This module defines encoder commands that can be recorded into a command
encoder. These commands are used to encode operations for execution on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Command:
    """
    Command for encoding.

    This enum represents different types of commands that can be recorded
    into a command encoder.

    Attributes:
        copy_buffer_to_buffer: Copy data between buffers.
        copy_buffer_to_texture: Copy data from buffer to texture.
        copy_texture_to_buffer: Copy data from texture to buffer.
        copy_texture_to_texture: Copy data between textures.
        clear_buffer: Clear a buffer.
        clear_texture: Clear a texture.
        write_timestamp: Write a timestamp.
        resolve_query_set: Resolve a query set.
        push_debug_group: Push a debug group.
        pop_debug_group: Pop a debug group.
        insert_debug_marker: Insert a debug marker.
        run_compute_pass: Run a compute pass.
        run_render_pass: Run a render pass.
        build_acceleration_structures: Build acceleration structures.
        transition_resources: Transition resources.
    """

    COPY_BUFFER_TO_BUFFER = "copy_buffer_to_buffer"
    COPY_BUFFER_TO_TEXTURE = "copy_buffer_to_texture"
    COPY_TEXTURE_TO_BUFFER = "copy_texture_to_buffer"
    COPY_TEXTURE_TO_TEXTURE = "copy_texture_to_texture"
    CLEAR_BUFFER = "clear_buffer"
    CLEAR_TEXTURE = "clear_texture"
    WRITE_TIMESTAMP = "write_timestamp"
    RESOLVE_QUERY_SET = "resolve_query_set"
    PUSH_DEBUG_GROUP = "push_debug_group"
    POP_DEBUG_GROUP = "pop_debug_group"
    INSERT_DEBUG_MARKER = "insert_debug_marker"
    RUN_COMPUTE_PASS = "run_compute_pass"
    RUN_RENDER_PASS = "run_render_pass"
    BUILD_ACCELERATION_STRUCTURES = "build_acceleration_structures"
    TRANSITION_RESOURCES = "transition_resources"


@dataclass
class CopyBufferToBuffer:
    """
    Command to copy data between buffers.

    Attributes:
        src: Source buffer.
        src_offset: Offset in source buffer.
        dst: Destination buffer.
        dst_offset: Offset in destination buffer.
        size: Size of data to copy.
    """

    src: Any
    src_offset: int
    dst: Any
    dst_offset: int
    size: Optional[int]


@dataclass
class CopyBufferToTexture:
    """
    Command to copy data from buffer to texture.

    Attributes:
        src: Source buffer info.
        dst: Destination texture info.
        size: Size of data to copy.
    """

    src: Any
    dst: Any
    size: Any


@dataclass
class CopyTextureToBuffer:
    """
    Command to copy data from texture to buffer.

    Attributes:
        src: Source texture info.
        dst: Destination buffer info.
        size: Size of data to copy.
    """

    src: Any
    dst: Any
    size: Any


@dataclass
class CopyTextureToTexture:
    """
    Command to copy data between textures.

    Attributes:
        src: Source texture info.
        dst: Destination texture info.
        size: Size of data to copy.
    """

    src: Any
    dst: Any
    size: Any


@dataclass
class ClearBuffer:
    """
    Command to clear a buffer.

    Attributes:
        dst: Destination buffer.
        offset: Offset into buffer.
        size: Size of data to clear.
    """

    dst: Any
    offset: int
    size: Optional[int]


@dataclass
class ClearTexture:
    """
    Command to clear a texture.

    Attributes:
        dst: Destination texture.
        subresource_range: Subresource range to clear.
    """

    dst: Any
    subresource_range: Any


@dataclass
class WriteTimestamp:
    """
    Command to write a timestamp.

    Attributes:
        query_set: Query set to write to.
        query_index: Query index.
    """

    query_set: Any
    query_index: int


@dataclass
class ResolveQuerySet:
    """
    Command to resolve a query set.

    Attributes:
        query_set: Query set to resolve.
        start_query: Start query index.
        query_count: Number of queries to resolve.
        destination: Destination buffer.
        destination_offset: Offset in destination buffer.
    """

    query_set: Any
    start_query: int
    query_count: int
    destination: Any
    destination_offset: int


@dataclass
class PushDebugGroup:
    """
    Command to push a debug group.

    Attributes:
        label: Debug group label.
    """

    label: str


@dataclass
class PopDebugGroup:
    """Command to pop a debug group."""

    pass


@dataclass
class InsertDebugMarker:
    """
    Command to insert a debug marker.

    Attributes:
        label: Debug marker label.
    """

    label: str


@dataclass
class RunComputePass:
    """
    Command to run a compute pass.

    Attributes:
        pass_data: Compute pass data.
        timestamp_writes: Timestamp writes for the pass.
    """

    pass_data: Any
    timestamp_writes: Optional[Any]


@dataclass
class RunRenderPass:
    """
    Command to run a render pass.

    Attributes:
        pass_data: Render pass data.
        color_attachments: Color attachments.
        depth_stencil_attachment: Depth/stencil attachment.
        timestamp_writes: Timestamp writes for the pass.
        occlusion_query_set: Occlusion query set.
        multiview_mask: Multiview mask.
    """

    pass_data: Any
    color_attachments: List[Any]
    depth_stencil_attachment: Optional[Any]
    timestamp_writes: Optional[Any]
    occlusion_query_set: Optional[Any]
    multiview_mask: Optional[int]


@dataclass
class BuildAccelerationStructures:
    """
    Command to build acceleration structures.

    Attributes:
        blas: Bottom-level acceleration structures to build.
        tlas: Top-level acceleration structures to build.
    """

    blas: List[Any]
    tlas: List[Any]


@dataclass
class TransitionResources:
    """
    Command to transition resources.

    Attributes:
        buffer_transitions: Buffer transitions.
        texture_transitions: Texture transitions.
    """

    buffer_transitions: List[Any]
    texture_transitions: List[Any]
