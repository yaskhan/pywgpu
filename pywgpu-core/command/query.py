"""
Query management for command encoding.

This module implements query management for wgpu-core. It provides:
- QueryUseError: Error when query is used incorrectly
- TimestampWritesError: Error related to timestamp writes
- Pipeline statistics query management

Queries are used to collect GPU timing and occlusion information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from .. import wgt
except ImportError:
    # Fallback if wgt is not available
    class wgt:
        QUERY_RESOLVE_BUFFER_ALIGNMENT = 256
        QUERY_SIZE = 8
        
        class Features:
            TIMESTAMP_QUERY_INSIDE_ENCODERS = 1 << 0
        
        class BufferUsages:
            QUERY_RESOLVE = 1 << 9
            COPY_DST = 1 << 2
        
        class BufferUses:
            COPY_DST = 1 << 2

try:
    from ..resource import DestroyedResourceError
except ImportError:
    class DestroyedResourceError(Exception):
        pass

try:
    from ..init_tracker import MemoryInitKind
except ImportError:
    class MemoryInitKind:
        ImplicitlyInitialized = "ImplicitlyInitialized"


@dataclass
class QueryUseError(Exception):
    """
    Error when query is used incorrectly.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class TimestampWritesError(Exception):
    """
    Error related to timestamp writes.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class PassTimestampWrites:
    """
    Timestamp writes for a pass.

    Attributes:
        query_set: The query set to write to.
        beginning_of_pass_write_index: Index for beginning of pass write.
        end_of_pass_write_index: Index for end of pass write.
    """

    query_set: Any
    beginning_of_pass_write_index: Optional[int]
    end_of_pass_write_index: Optional[int]


def validate_and_begin_pipeline_statistics_query(
    state: Any,
    query_set: Any,
    query_index: int,
) -> None:
    """
    Validate and begin a pipeline statistics query.

    Args:
        state: The pass state.
        query_set: The query set.
        query_index: The query index.

    Raises:
        QueryUseError: If query is used incorrectly.
    """
    query_set.same_device(state.device)

    needs_reset = state.reset_state is None
    query_set.validate_query(
        "PipelineStatistics",
        query_index,
        state.reset_state,
    )

    state.tracker.query_sets.insert_single(query_set)

    if state.active_query is not None:
        _, old_idx = state.active_query
        raise QueryUseError(
            f"Query {query_index} started while {old_idx} is still active"
        )

    state.active_query = (query_set, query_index)

    # If we don't have a reset state tracker which can defer resets, we must reset now.
    if needs_reset:
        state.raw_encoder.reset_queries(query_set.raw(), query_index, 1)

    state.raw_encoder.begin_query(query_set.raw(), query_index)


def end_pipeline_statistics_query(state: Any) -> None:
    """
    End a pipeline statistics query.

    Args:
        state: The pass state.

    Raises:
        QueryUseError: If query is used incorrectly.
    """
    if state.active_query is not None:
        query_set, query_index = state.active_query
        state.active_query = None
        state.raw_encoder.end_query(query_set.raw(), query_index)
    else:
        raise QueryUseError("No query is currently active")


def validate_and_begin_occlusion_query(
    state: Any,
    query_set: Any,
    query_index: int,
) -> None:
    """
    Validate and begin an occlusion query.

    Args:
        state: The pass state.
        query_set: The query set.
        query_index: The query index.

    Raises:
        QueryUseError: If query is used incorrectly.
    """
    needs_reset = state.reset_state is None
    query_set.validate_query("Occlusion", query_index, state.reset_state)

    state.tracker.query_sets.insert_single(query_set)

    if state.active_query is not None:
        _, old_idx = state.active_query
        raise QueryUseError(
            f"Query {query_index} started while {old_idx} is still active"
        )

    state.active_query = (query_set, query_index)

    if needs_reset:
        state.raw_encoder.reset_queries(query_set.raw(), query_index, 1)

    state.raw_encoder.begin_query(query_set.raw(), query_index)


def end_occlusion_query(state: Any) -> None:
    """
    End an occlusion query.

    Args:
        state: The pass state.

    Raises:
        QueryUseError: If query is used incorrectly.
    """
    if state.active_query is not None:
        query_set, query_index = state.active_query
        state.active_query = None
        state.raw_encoder.end_query(query_set.raw(), query_index)
    else:
        raise QueryUseError("No query is currently active")


def write_timestamp(
    state: Any,
    query_set: Any,
    query_index: int,
) -> None:
    """
    Write a timestamp.

    Args:
        state: The encoding state.
        query_set: The query set.
        query_index: The query index.
    
    Raises:
        MissingFeatures: If TIMESTAMP_QUERY_INSIDE_ENCODERS feature is not enabled.
    """
    # Require TIMESTAMP_QUERY_INSIDE_ENCODERS feature
    if hasattr(state.device, 'require_features'):
        state.device.require_features(wgt.Features.TIMESTAMP_QUERY_INSIDE_ENCODERS)
    
    query_set.same_device(state.device)
    query_set.validate_and_write_timestamp(state.raw_encoder, query_index, None)
    state.tracker.query_sets.insert_single(query_set)


def resolve_query_set(
    state: Any,
    query_set: Any,
    start_query: int,
    query_count: int,
    dst_buffer: Any,
    destination_offset: int,
) -> None:
    """
    Resolve a query set.

    Args:
        state: The encoding state.
        query_set: The query set.
        start_query: The start query.
        query_count: The query count.
        dst_buffer: The destination buffer.
        destination_offset: The destination offset.
    
    Raises:
        ValueError: If buffer offset alignment is incorrect.
        QueryUseError: If query range is out of bounds.
        BufferOverrunError: If buffer is too small.
    """
    # Validate buffer offset alignment
    if destination_offset % wgt.QUERY_RESOLVE_BUFFER_ALIGNMENT != 0:
        raise ValueError(
            f"Buffer offset {destination_offset} is not aligned to "
            f"QUERY_RESOLVE_BUFFER_ALIGNMENT ({wgt.QUERY_RESOLVE_BUFFER_ALIGNMENT})"
        )

    query_set.same_device(state.device)
    dst_buffer.same_device(state.device)
    dst_buffer.check_destroyed(state.snatch_guard)

    # Track buffer usage and get transition
    dst_pending = state.tracker.buffers.set_single(dst_buffer, wgt.BufferUses.COPY_DST)
    if dst_pending:
        dst_barrier = dst_pending.into_hal(dst_buffer, state.snatch_guard)
        if dst_barrier and hasattr(state.raw_encoder, 'transition_buffers'):
            state.raw_encoder.transition_buffers([dst_barrier])

    # Check buffer has QUERY_RESOLVE usage
    if hasattr(dst_buffer, 'check_usage'):
        dst_buffer.check_usage(wgt.BufferUsages.QUERY_RESOLVE)

    # Validate query range
    end_query = start_query + query_count
    if end_query > query_set.desc.count:
        raise QueryUseError(
            f"Resolving queries {start_query}..{end_query} would overrun "
            f"the query set of size {query_set.desc.count}"
        )

    # Calculate stride and bytes used
    # Elements per query depends on query type
    if hasattr(query_set.desc, 'ty'):
        query_type = query_set.desc.ty
        if query_type == 'Occlusion':
            elements_per_query = 1
        elif query_type == 'Timestamp':
            elements_per_query = 1
        elif hasattr(query_type, '__name__') and 'PipelineStatistics' in query_type.__name__:
            # Count bits in pipeline statistics flags
            elements_per_query = bin(query_type).count('1') if isinstance(query_type, int) else 1
        else:
            elements_per_query = 1
    else:
        elements_per_query = 1
    
    stride = elements_per_query * wgt.QUERY_SIZE
    bytes_used = stride * query_count

    # Validate buffer size
    buffer_start_offset = destination_offset
    buffer_end_offset = buffer_start_offset + bytes_used
    
    if buffer_end_offset > dst_buffer.size:
        raise ValueError(
            f"Resolving queries {start_query}..{end_query} ({stride} byte queries) "
            f"will overrun the destination buffer of size {dst_buffer.size} "
            f"using offsets {buffer_start_offset}..{buffer_end_offset} ({bytes_used} bytes used)"
        )

    # Track memory initialization
    # The buffer region will be implicitly initialized by the query resolve
    if hasattr(dst_buffer, 'initialization_status'):
        init_actions = dst_buffer.initialization_status.read().create_action(
            dst_buffer,
            range(buffer_start_offset, buffer_end_offset),
            MemoryInitKind.ImplicitlyInitialized,
        )
        if init_actions:
            state.buffer_memory_init_actions.extend(init_actions)

    # Get raw buffer handle
    raw_dst_buffer = dst_buffer.try_raw(state.snatch_guard)
    if raw_dst_buffer is None:
        raise DestroyedResourceError(dst_buffer.error_ident())

    # Issue HAL command
    if hasattr(state.raw_encoder, 'copy_query_results'):
        state.raw_encoder.copy_query_results(
            query_set.raw(),
            start_query,
            query_count,
            raw_dst_buffer,
            destination_offset,
            stride,
        )
    
    # Track query set usage
    state.tracker.query_sets.insert_single(query_set)
