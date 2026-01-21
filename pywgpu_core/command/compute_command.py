"""
Compute commands for command encoding.

This module defines compute commands that can be recorded into a compute pass.
These commands are used to encode compute shader operations for execution on
the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ComputeCommand:
    """
    Compute command for encoding.

    This enum represents different types of compute commands that can be
    recorded into a compute pass.

    Attributes:
        set_bind_group: Set a bind group.
        set_pipeline: Set a compute pipeline.
        set_immediate: Set immediate data.
        dispatch: Dispatch workgroups.
        dispatch_indirect: Dispatch workgroups indirectly.
        push_debug_group: Push a debug group.
        pop_debug_group: Pop a debug group.
        insert_debug_marker: Insert a debug marker.
        write_timestamp: Write a timestamp.
        begin_pipeline_statistics_query: Begin a pipeline statistics query.
        end_pipeline_statistics_query: End a pipeline statistics query.
    """

    SET_BIND_GROUP = "set_bind_group"
    SET_PIPELINE = "set_pipeline"
    SET_IMMEDIATE = "set_immediate"
    DISPATCH = "dispatch"
    DISPATCH_INDIRECT = "dispatch_indirect"
    PUSH_DEBUG_GROUP = "push_debug_group"
    POP_DEBUG_GROUP = "pop_debug_group"
    INSERT_DEBUG_MARKER = "insert_debug_marker"
    WRITE_TIMESTAMP = "write_timestamp"
    BEGIN_PIPELINE_STATISTICS_QUERY = "begin_pipeline_statistics_query"
    END_PIPELINE_STATISTICS_QUERY = "end_pipeline_statistics_query"


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
    Command to set a compute pipeline.

    Attributes:
        pipeline: The compute pipeline.
    """

    pipeline: Any


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
class Dispatch:
    """
    Command to dispatch workgroups.

    Attributes:
        workgroups: Workgroup counts for X, Y, Z.
    """

    workgroups: tuple[int, int, int]


@dataclass
class DispatchIndirect:
    """
    Command to dispatch workgroups indirectly.

    Attributes:
        buffer: The buffer containing dispatch parameters.
        offset: Offset into the buffer.
    """

    buffer: Any
    offset: int


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
