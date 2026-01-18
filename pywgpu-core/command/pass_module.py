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
    # Implementation depends on command processing
    pass


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


def flush_bindings_helper(state: Any) -> None:
    """
    Flush bindings helper.
    
    Args:
        state: The pass state.
    """
    range_ = state.binder.take_rebind_range()
    if not range_:
        return

    entries = state.binder.entries(range_)

    for i, entry in entries:
        bind_group = entry.group
        if bind_group is None:
            continue

        # In Rust this extends buffer_memory_init_actions and texture_memory_actions
        # state.base.buffer_memory_init_actions.extend(...)
        # state.base.texture_memory_actions.register_init_action(action)
        pass

    pipeline_layout = state.binder.pipeline_layout
    if pipeline_layout:
        for i, entry in entries:
            if entry.group:
                state.base.raw_encoder.set_bind_group(
                    pipeline_layout.raw(),
                    i,
                    entry.group.raw(),
                    entry.dynamic_offsets,
                )
