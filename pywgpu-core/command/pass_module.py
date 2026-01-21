"""
Pass management for command encoding.

This module implements pass management for wgpu-core. It provides:
- PassState: State for a pass
- PassErrorScope: Error scopes for passes
- MissingPipeline: Error when pipeline is missing
- BindGroupIndexOutOfRange: Error when bind group index is out of range
- InvalidValuesOffset: Error when values offset is invalid

Passes are used to group related commands together and provide isolation
between different rendering operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class PassState:
    """
    State for a pass.
    
    Attributes:
        base: Base encoding state.
        binder: Bind group binder.
        temp_offsets: Temporary offsets.
        dynamic_offset_count: Dynamic offset count.
        pending_discard_init_fixups: Pending discard init fixups.
        scope: Usage scope.
        string_offset: String offset.
    """

    base: Any
    binder: Any
    temp_offsets: List[int]
    dynamic_offset_count: int
    pending_discard_init_fixups: Any
    scope: Any
    string_offset: int


@dataclass
class PassErrorScope:
    """
    Error scope for passes.
    
    Attributes:
        pass_scope: The pass scope.
    """

    pass_scope: str


@dataclass
class MissingPipeline(Exception):
    """
    Error when pipeline is missing.
    
    Attributes:
        message: The error message.
    """

    message: str = "Pipeline must be set"

    def __str__(self) -> str:
        return self.message


@dataclass
class BindGroupIndexOutOfRange(Exception):
    """
    Error when bind group index is out of range.
    
    Attributes:
        index: The bind group index.
        max: The maximum allowed index.
    """

    index: int
    max: int

    def __str__(self) -> str:
        return f"Bind group index {self.index} is out of range (max: {self.max})"


@dataclass
class InvalidValuesOffset(Exception):
    """
    Error when values offset is invalid.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


def set_bind_group(
    state: Any,
    device: Any,
    dynamic_offsets: List[int],
    index: int,
    num_dynamic_offsets: int,
    bind_group: Any,
    is_compute: bool,
) -> None:
    """
    Set a bind group.
    
    Args:
        state: The pass state.
        device: The device.
        dynamic_offsets: Dynamic offsets.
        index: Bind group index.
        num_dynamic_offsets: Number of dynamic offsets.
        bind_group: The bind group.
        is_compute: Whether this is a compute pass.
    
    Raises:
        BindGroupIndexOutOfRange: If index is out of range.
    """
    # Limits validation
    max_bind_groups = 8 # device.limits.max_bind_groups
    if index >= max_bind_groups:
        raise BindGroupIndexOutOfRange(index, max_bind_groups)

    # Device validation
    if bind_group.device is not device:
        raise ValueError("Bind group is from a different device")
    
    # Validation of dynamic offsets count
    expected_dynamic_offsets = getattr(bind_group.layout, 'dynamic_offset_count', 0)
    if len(dynamic_offsets) != expected_dynamic_offsets:
        raise ValueError(f"Dynamic offsets count mismatch: expected {expected_dynamic_offsets}, got {len(dynamic_offsets)}")

    # Update binder
    state.binder.assign_group(index, bind_group, dynamic_offsets)
    
    # Resource tracking - bind groups keep resources alive
    state.scope.bind_groups.insert_single(bind_group)
    
    # In Rust, we also track buffers and textures inside the bind group here
    # but wgpu-core often defers this until flush_bindings or draw time.
    # For now, we just update the binder.


def set_immediates(
    state: Any,
    immediates_data: List[int],
    offset: int,
    size_bytes: int,
    values_offset: Optional[int],
    callback: Any,
) -> None:
    """
    Set immediates.
    
    Args:
        state: The pass state.
        immediates_data: Immediates data.
        offset: Byte offset.
        size_bytes: Size in bytes.
        values_offset: Values offset.
        callback: Callback for processing immediates.
    
    Raises:
        InvalidValuesOffset: If values offset is invalid.
    """
    if values_offset is None:
        raise InvalidValuesOffset("Values offset must be set")

    end_offset_bytes = offset + size_bytes
    values_end_offset = values_offset + size_bytes // 4 # IMMEDIATE_DATA_ALIGNMENT
    data_slice = immediates_data[values_offset:values_end_offset]

    pipeline_layout = state.binder.pipeline_layout
    if pipeline_layout is None:
        raise MissingPipeline()

    # pipeline_layout.validate_immediates_ranges(offset, end_offset_bytes)

    callback(data_slice)

    state.base.raw_encoder.set_immediates(pipeline_layout.raw(), offset, data_slice)


try:
    from ..track import BufferUses, TextureUses
except ImportError:
    # Use fallback if not available
    class BufferUses:
        UNIFORM = 1 << 6
        STORAGE_READ_ONLY = 1 << 7
        STORAGE_READ_WRITE = 1 << 8
    class TextureUses:
        RESOURCE = 1 << 4
        STORAGE_READ_ONLY = 1 << 8
        STORAGE_READ_WRITE = 1 << 10

def flush_bindings_helper(state: Any) -> None:
    """
    Flush bindings to the HAL encoder.
    
    This function:
    1. Gets the range of bind groups that need rebinding.
    2. Iterates over them and tracks their resources in the pass scope.
    3. Translates and issues HAL set_bind_group commands.
    
    Args:
        state: The pass state (PassState).
    """
    range_ = state.binder.take_rebind_range()
    if not range_:
        return

    entries = state.binder.entries(range_)
    pipeline_layout = state.binder.pipeline_layout
    if not pipeline_layout:
        raise MissingPipeline()

    for i, entry in entries:
        bind_group = entry.group
        if bind_group is None:
            continue

        # 1. Track resources in the bind group
        # Each bind group knows which resources it contains and their usages
        for bg_entry in bind_group.entries:
            resource = bg_entry.resource
            # We need to find the layout entry for this binding to get the usage
            # For now, let's assume we can determine usage from the resource type 
            # and some hint from the BGL
            
            # Simple heuristic for now:
            if hasattr(resource, 'resource_type'): # e.g. Buffer
                usage = BufferUses.UNIFORM # Default
                # TODO: Get actual usage from BGL entry
                state.scope.buffers.set_single(resource, usage)
                # Register memory init action if it's a buffer
                # state.base.buffer_memory_init_actions.extend(...)
            elif hasattr(resource, 'view'): # e.g. TextureView
                usage = TextureUses.RESOURCE # Default
                # TODO: Get actual usage from BGL entry
                state.scope.textures.set_single(resource, usage)
                # Register memory init action if it's a texture
                state.base.texture_memory_actions.register_init_action(...)

        # 2. Add bind group to stateless tracking (keep-alive)
        state.scope.bind_groups.insert_single(bind_group)

        # 3. Issue HAL command
        if hasattr(state.base, 'raw_encoder') and state.base.raw_encoder:
            state.base.raw_encoder.set_bind_group(
                pipeline_layout.raw(),
                i,
                bind_group.raw(),
                entry.dynamic_offsets,
            )
 
 
def push_debug_group(state: Any, string_data: bytearray, length: int) -> None:
    """Push a debug group."""
    label = string_data[state.string_offset : state.string_offset + length].decode("utf-8")
    state.string_offset += length

    if hasattr(state.base, "raw_encoder") and state.base.raw_encoder:
        state.base.raw_encoder.push_debug_group(label)


def pop_debug_group(state: Any) -> None:
    """Pop a debug group."""
    if hasattr(state.base, "raw_encoder") and state.base.raw_encoder:
        state.base.raw_encoder.pop_debug_group()


def insert_debug_marker(state: Any, string_data: bytearray, length: int) -> None:
    """Insert a debug marker."""
    label = string_data[state.string_offset : state.string_offset + length].decode("utf-8")
    state.string_offset += length

    if hasattr(state.base, "raw_encoder") and state.base.raw_encoder:
        state.base.raw_encoder.insert_debug_marker(label)


def write_timestamp(
    state: Any,
    device: Any,
    encoder: Any,
    query_set: Any,
    query_index: int,
) -> None:
    """Write a timestamp."""
    raw_query_set = state.base.tracker.query_sets.insert_single(query_set)
    if hasattr(state.base, "raw_encoder") and state.base.raw_encoder:
        state.base.raw_encoder.write_timestamp(raw_query_set.raw(), query_index)


def change_pipeline_layout(state: Any, layout: Any) -> None:
    """Changes the pipeline layout."""
    state.binder.pipeline_layout = layout
    # In a full implementation, this might invalidate bind groups
