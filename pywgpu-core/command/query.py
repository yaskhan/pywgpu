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
    """
    # state.device.require_features("TIMESTAMP_QUERY_INSIDE_ENCODERS")
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
    """
    # Basic validation
    if destination_offset % 8 != 0: # wgt::QUERY_RESOLVE_BUFFER_ALIGNMENT
        raise ValueError("Buffer offset alignment error")

    query_set.same_device(state.device)
    dst_buffer.same_device(state.device)
    dst_buffer.check_destroyed(state.snatch_guard)

    # Simplified tracking and transition for now
    state.tracker.buffers.set_single(dst_buffer, "COPY_DST")
    
    # dst_buffer.check_usage("QUERY_RESOLVE")
    
    # ... more validation logic from Rust could go here ...
    
    state.raw_encoder.copy_query_results(
        query_set.raw(),
        start_query,
        query_count,
        dst_buffer.raw(),
        destination_offset,
    )
