# Additional missing structures and descriptors from Rust binding_model.rs

from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class BufferBinding:
    """
    Buffer binding with offset and optional size.
    
    Corresponds to Rust's BufferBinding struct.
    """
    buffer: Any  # Buffer or BufferId
    offset: int  # wgt::BufferAddress
    size: Optional[int] = None  # Optional size, None means to end of buffer


@dataclass
class BindGroupDescriptor:
    """
    Describes a group of bindings and the resources to be bound.
    
    Corresponds to Rust's BindGroupDescriptor.
    """
    label: Optional[str] = None
    layout: Any = None  # BindGroupLayout or BindGroupLayoutId
    entries: List[Any] = None  # List of BindGroupEntry


@dataclass
class BindGroupLayoutDescriptor:
    """
    Describes a bind group layout.
    
    Corresponds to Rust's BindGroupLayoutDescriptor.
    """
    label: Optional[str] = None
    entries: List[Any] = None  # List of wgt::BindGroupLayoutEntry


@dataclass
class PipelineLayoutDescriptor:
    """
    Describes a pipeline layout.
    
    Corresponds to Rust's PipelineLayoutDescriptor.
    """
    label: Optional[str] = None
    bind_group_layouts: List[Any] = None  # List of BindGroupLayout or BindGroupLayoutId
    immediate_size: int = 0  # Size of immediate data in bytes


@dataclass
class BindGroupDynamicBindingData:
    """
    Data for dynamic buffer bindings.
    
    Corresponds to Rust's BindGroupDynamicBindingData.
    """
    binding_idx: int  # The index of the binding
    buffer_size: int  # The size of the buffer
    binding_range: range  # The range being bound
    maximum_dynamic_offset: int  # Maximum allowed dynamic offset
    binding_type: Any  # wgt::BufferBindingType


@dataclass
class BindGroupLateBufferBindingInfo:
    """
    Information about late buffer bindings (buffers without min_binding_size in BGL).
    
    Corresponds to Rust's BindGroupLateBufferBindingInfo.
    """
    shader_size: int  # Expected size from shader
    bound_size: int  # Actual bound size


# Error classes

@dataclass
class BindError(Exception):
    """Errors that can occur when binding resources."""
    pass


@dataclass
class MismatchedDynamicOffsetCountError(BindError):
    """Dynamic offset count mismatch."""
    bind_group: Any
    group: int
    expected: int
    actual: int
    
    def __str__(self) -> str:
        s0 = "s" if self.expected >= 2 else ""
        s1 = "s" if self.actual >= 2 else ""
        return (
            f"{self.bind_group} {self.group} expects {self.expected} dynamic offset{s0}. "
            f"However {self.actual} dynamic offset{s1} were provided."
        )


@dataclass
class UnalignedDynamicBindingError(BindError):
    """Dynamic binding offset not aligned."""
    bind_group: Any
    idx: int
    group: int
    binding: int
    offset: int
    alignment: int
    limit_name: str
    
    def __str__(self) -> str:
        return (
            f"Dynamic binding index {self.idx} (targeting {self.bind_group} {self.group}, "
            f"binding {self.binding}) with value {self.offset}, does not respect device's "
            f"requested `{self.limit_name}` limit: {self.alignment}"
        )


@dataclass
class DynamicBindingOutOfBoundsError(BindError):
    """Dynamic binding offset would overrun buffer."""
    bind_group: Any
    idx: int
    group: int
    binding: int
    offset: int
    buffer_size: int
    binding_range: range
    maximum_dynamic_offset: int
    
    def __str__(self) -> str:
        return (
            f"Dynamic binding offset index {self.idx} with offset {self.offset} would overrun "
            f"the buffer bound to {self.bind_group} {self.group} -> binding {self.binding}. "
            f"Buffer size is {self.buffer_size} bytes, the binding binds bytes {self.binding_range}, "
            f"meaning the maximum the binding can be offset is {self.maximum_dynamic_offset} bytes"
        )


@dataclass
class GetBindGroupLayoutError(Exception):
    """Error getting bind group layout."""
    pass


@dataclass
class InvalidGroupIndexError(GetBindGroupLayoutError):
    """Invalid group index."""
    index: int
    
    def __str__(self) -> str:
        return f"Invalid group index {self.index}"


@dataclass
class LateMinBufferBindingSizeMismatch(Exception):
    """
    Buffer binding size mismatch between shader expectation and actual binding.
    
    Corresponds to Rust's LateMinBufferBindingSizeMismatch.
    """
    group_index: int
    binding_index: int
    shader_size: int  # wgt::BufferAddress
    bound_size: int  # wgt::BufferAddress
    
    def __str__(self) -> str:
        return (
            f"In bind group index {self.group_index}, the buffer bound at binding index "
            f"{self.binding_index} is bound with size {self.bound_size} where the shader "
            f"expects {self.shader_size}."
        )
