"""
Binding model logic (resource layouts, bind groups).

This module implements the WebGPU binding model, which includes:
- Bind group layouts: Descriptions of bindings for a bind group
- Bind groups: Collections of resources bound to shader stages
- Pipeline layouts: Descriptions of bind group layouts for a pipeline

The binding model is a core part of WebGPU's resource management system,
allowing shaders to access buffers, textures, samplers, and other resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from . import errors
from .device import Device, DeviceError
from .id import BindGroupLayoutId, BindGroupId, PipelineLayoutId
from .resource import (
    DestroyedResourceError,
    InvalidResourceError,
    Labeled,
    MissingBufferUsageError,
    MissingTextureUsageError,
    ParentDevice,
    ResourceErrorIdent,
    ResourceType,
    Trackable,
)
from .track import ResourceUsageCompatibilityError


@dataclass
class BindGroupLayoutEntryError(Exception):
    """Error related to a bind group layout entry."""
    
    def __str__(self) -> str:
        return "Bind group layout entry error"


@dataclass
class CreateBindGroupLayoutError(Exception):
    """Error creating a bind group layout."""
    
    def __str__(self) -> str:
        return "Failed to create bind group layout"


@dataclass
class BindingError(Exception):
    """Error related to a binding."""
    
    def __str__(self) -> str:
        return "Binding error"


@dataclass
class CreateBindGroupError(Exception):
    """Error creating a bind group."""
    
    def __str__(self) -> str:
        return "Failed to create bind group"


@dataclass
class BindingZone(Exception):
    """Zone for binding errors."""
    
    def __str__(self) -> str:
        return "Binding zone"


@dataclass
class BindingTypeMaxCountError(Exception):
    """Error when too many bindings of a type are used."""
    
    def __str__(self) -> str:
        return "Too many bindings of a type"


@dataclass
class BindingTypeMaxCountErrorKind:
    """Types of binding count errors."""

    DYNAMIC_UNIFORM_BUFFERS = "DynamicUniformBuffers"
    DYNAMIC_STORAGE_BUFFERS = "DynamicStorageBuffers"
    SAMPLED_TEXTURES = "SampledTextures"
    SAMPLERS = "Samplers"
    STORAGE_BUFFERS = "StorageBuffers"
    STORAGE_TEXTURES = "StorageTextures"
    UNIFORM_BUFFERS = "UniformBuffers"
    BINDING_ARRAY_ELEMENTS = "BindingArrayElements"
    BINDING_ARRAY_SAMPLER_ELEMENTS = "BindingArraySamplerElements"
    ACCELERATION_STRUCTURES = "AccelerationStructures"

    @staticmethod
    def to_config_str(kind: str) -> str:
        """Convert binding type max count error kind to config string."""
        mapping = {
            "DynamicUniformBuffers": "max_dynamic_uniform_buffers_per_pipeline_layout",
            "DynamicStorageBuffers": "max_dynamic_storage_buffers_per_pipeline_layout",
            "SampledTextures": "max_sampled_textures_per_shader_stage",
            "Samplers": "max_samplers_per_shader_stage",
            "StorageBuffers": "max_storage_buffers_per_shader_stage",
            "StorageTextures": "max_storage_textures_per_shader_stage",
            "UniformBuffers": "max_uniform_buffers_per_shader_stage",
            "BindingArrayElements": "max_binding_array_elements_per_shader_stage",
            "BindingArraySamplerElements": "max_binding_array_sampler_elements_per_shader_stage",
            "AccelerationStructures": "max_acceleration_structures_per_shader_stage",
        }
        return mapping.get(kind, "unknown_config")


class PerStageBindingTypeCounter:
    """
    Counter for binding types per shader stage.
    
    Tracks the number of bindings of each type for each shader stage
    (vertex, fragment, compute).
    """

    def __init__(self) -> None:
        """Initialize the counter with zero for all stages."""
        self.vertex: int = 0
        self.fragment: int = 0
        self.compute: int = 0

    def add(self, stage: Any, count: int) -> None:
        """
        Add bindings to the counter for the given stage.
        
        Args:
            stage: The shader stage(s) to add bindings for.
            count: The number of bindings to add.
        """
        # Check which stages are included
        # Assuming stage is a bitfield with VERTEX=1, FRAGMENT=2, COMPUTE=4
        try:
            # Try to import wgpu_types for proper stage handling
            import pywgpu_types as wgt
            if hasattr(wgt, 'ShaderStages'):
                if hasattr(stage, '__and__'):
                    # Bitfield stage
                    if stage & getattr(wgt.ShaderStages, 'VERTEX', 1):
                        self.vertex += count
                    if stage & getattr(wgt.ShaderStages, 'FRAGMENT', 2):
                        self.fragment += count
                    if stage & getattr(wgt.ShaderStages, 'COMPUTE', 4):
                        self.compute += count
                else:
                    # Single stage or unknown format
                    self.vertex += count
                    self.fragment += count
                    self.compute += count
            else:
                # No wgt available, add to all
                self.vertex += count
                self.fragment += count
                self.compute += count
        except ImportError:
            # Fallback: add to all stages
            self.vertex += count
            self.fragment += count
            self.compute += count

    def max(self) -> tuple[Any, int]:
        """
        Get the maximum binding count across all stages.
        
        Returns:
            A tuple of (stage, max_count).
        """
        max_value = max(self.vertex, self.fragment, self.compute)
        
        # Determine which stage has the maximum value
        stage_value = 0
        try:
            import pywgpu_types as wgt
            if hasattr(wgt, 'ShaderStages'):
                if max_value == self.vertex:
                    stage_value |= getattr(wgt.ShaderStages, 'VERTEX', 1)
                if max_value == self.fragment:
                    stage_value |= getattr(wgt.ShaderStages, 'FRAGMENT', 2)
                if max_value == self.compute:
                    stage_value |= getattr(wgt.ShaderStages, 'COMPUTE', 4)
        except ImportError:
            # Fallback: use simple integer values
            if max_value == self.vertex:
                stage_value |= 1
            if max_value == self.fragment:
                stage_value |= 2
            if max_value == self.compute:
                stage_value |= 4
        
        return (stage_value, max_value)

    def merge(self, other: PerStageBindingTypeCounter) -> None:
        """
        Merge another counter into this one.
        
        Args:
            other: The counter to merge.
        """
        self.vertex = max(self.vertex, other.vertex)
        self.fragment = max(self.fragment, other.fragment)
        self.compute = max(self.compute, other.compute)

    def validate(self, limit: int, kind: str) -> None:
        """
        Validate that the binding count does not exceed the limit.
        
        Args:
            limit: The maximum allowed binding count.
            kind: The type of binding being validated.
        
        Raises:
            BindingTypeMaxCountError: If the limit is exceeded.
        """
        zone, count = self.max()
        if limit < count:
            raise BindingTypeMaxCountError(
                kind=kind, zone=zone, limit=limit, count=count
            )


class BindingTypeMaxCountValidator:
    """
    Validator for binding type counts.
    
    Tracks and validates the number of bindings of each type across all stages.
    """

    def __init__(self) -> None:
        """Initialize the validator with zero counts for all binding types."""
        self.dynamic_uniform_buffers: int = 0
        self.dynamic_storage_buffers: int = 0
        self.sampled_textures: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.samplers: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.storage_buffers: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.storage_textures: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.uniform_buffers: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.acceleration_structures: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.binding_array_elements: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.binding_array_sampler_elements: PerStageBindingTypeCounter = PerStageBindingTypeCounter()
        self.has_bindless_array: bool = False

    def add_binding(self, binding: Any) -> None:
        """
        Add a binding to the validator.

        Args:
            binding: The binding to add (BindGroupLayoutEntry).
        """
        # Extract binding information
        visibility = getattr(binding, 'visibility', None)
        binding_type = getattr(binding, 'binding_type', None) or getattr(binding, 'ty', None)
        count = getattr(binding, 'count', 1) or 1
        
        if binding_type is None:
            return
        
        # Determine binding type and update counters
        # This is a simplified implementation
        binding_type_str = str(binding_type).lower() if not isinstance(binding_type, str) else binding_type.lower()
        
        if 'uniform' in binding_type_str:
            if 'dynamic' in binding_type_str:
                self.dynamic_uniform_buffers += count
            else:
                self.uniform_buffers.add(visibility, count)
        elif 'storage' in binding_type_str:
            if 'dynamic' in binding_type_str:
                self.dynamic_storage_buffers += count
            else:
                self.storage_buffers.add(visibility, count)
        elif 'texture' in binding_type_str:
            if 'storage' in binding_type_str:
                self.storage_textures.add(visibility, count)
            else:
                self.sampled_textures.add(visibility, count)
        elif 'sampler' in binding_type_str:
            self.samplers.add(visibility, count)
        elif 'acceleration' in binding_type_str:
            self.acceleration_structures.add(visibility, count)
        
        # Check for binding arrays
        if count > 1:
            self.binding_array_elements.add(visibility, count)
            if 'sampler' in binding_type_str:
                self.binding_array_sampler_elements.add(visibility, count)
            # Check for bindless arrays (very large counts)
            if count > 1000:  # Arbitrary threshold for "bindless"
                self.has_bindless_array = True

    def merge(self, other: BindingTypeMaxCountValidator) -> None:
        """
        Merge another validator into this one.
        
        Args:
            other: The validator to merge.
        """
        self.dynamic_uniform_buffers += other.dynamic_uniform_buffers
        self.dynamic_storage_buffers += other.dynamic_storage_buffers
        self.sampled_textures.merge(other.sampled_textures)
        self.samplers.merge(other.samplers)
        self.storage_buffers.merge(other.storage_buffers)
        self.storage_textures.merge(other.storage_textures)
        self.uniform_buffers.merge(other.uniform_buffers)
        self.acceleration_structures.merge(other.acceleration_structures)
        self.binding_array_elements.merge(other.binding_array_elements)
        self.binding_array_sampler_elements.merge(other.binding_array_sampler_elements)

    def validate(self, limits: Any) -> None:
        """
        Validate all binding counts against the given limits.
        
        Args:
            limits: The device limits to validate against.
        
        Raises:
            BindingTypeMaxCountError: If any limit is exceeded.
        """
        # Get limits or use defaults
        max_dynamic_uniform = getattr(limits, 'max_dynamic_uniform_buffers_per_pipeline_layout', 8)
        max_dynamic_storage = getattr(limits, 'max_dynamic_storage_buffers_per_pipeline_layout', 4)
        max_sampled_textures = getattr(limits, 'max_sampled_textures_per_shader_stage', 16)
        max_samplers = getattr(limits, 'max_samplers_per_shader_stage', 16)
        max_storage_buffers = getattr(limits, 'max_storage_buffers_per_shader_stage', 8)
        max_storage_textures = getattr(limits, 'max_storage_textures_per_shader_stage', 4)
        max_uniform_buffers = getattr(limits, 'max_uniform_buffers_per_shader_stage', 12)
        
        # Validate dynamic buffers (these are per-pipeline, not per-stage)
        if self.dynamic_uniform_buffers > max_dynamic_uniform:
            raise BindingTypeMaxCountError(
                f"Too many dynamic uniform buffers: {self.dynamic_uniform_buffers} > {max_dynamic_uniform}"
            )
        
        if self.dynamic_storage_buffers > max_dynamic_storage:
            raise BindingTypeMaxCountError(
                f"Too many dynamic storage buffers: {self.dynamic_storage_buffers} > {max_dynamic_storage}"
            )
        
        # Validate per-stage limits
        self.sampled_textures.validate(max_sampled_textures, "sampled_textures")
        self.samplers.validate(max_samplers, "samplers")
        self.storage_buffers.validate(max_storage_buffers, "storage_buffers")
        self.storage_textures.validate(max_storage_textures, "storage_textures")
        self.uniform_buffers.validate(max_uniform_buffers, "uniform_buffers")

    def validate_binding_arrays(self) -> None:
        """
        Validate that the bind group layout does not contain both a binding array and a dynamic offset array.
        
        Raises:
            CreateBindGroupLayoutError: If validation fails.
        """
        has_dynamic_offset_array = (
            self.dynamic_uniform_buffers > 0 or self.dynamic_storage_buffers > 0
        )
        has_uniform_buffer = self.uniform_buffers.max()[1] > 0
        if self.has_bindless_array and has_dynamic_offset_array:
            raise CreateBindGroupLayoutError(
                "Bind groups may not contain both a binding array and a dynamically offset buffer"
            )
        if self.has_bindless_array and has_uniform_buffer:
            raise CreateBindGroupLayoutError(
                "Bind groups may not contain both a binding array and a uniform buffer"
            )


@dataclass
class BindGroupEntry:
    """
    Bindable resource and the slot to bind it to.
    
    Attributes:
        binding: Slot for which binding provides resource.
        resource: Resource to attach to the binding.
    """
    binding: int
    resource: Any


@dataclass
class ResolvedBindGroupEntry:
    """Resolved version of BindGroupEntry with Arc resources."""
    binding: int
    resource: Any


class BindGroupLayout:
    """
    A layout describing the bindings for a bind group.
    
    Bind group layouts define the types and number of resources that can be
    bound to a bind group. They are used to create bind groups and validate
    that resources are compatible with the layout.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        entries: List of bind group layout entries.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, entries: List[Any], label: str = "") -> None:
        """Initialize the bind group layout."""
        self.device = device
        self.label = label
        self.entries = entries
        self.tracking_data = None  # Would be TrackingData

    def is_equal(self, other: Any) -> bool:
        """Check if this layout is equal to another."""
        if not isinstance(other, BindGroupLayout):
            return False
        if self is other:
            return True
        # In a real implementation, we would compare the entries
        # For now, we compare identity or some hash if available
        return id(self) == id(other)

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="BindGroupLayout",
            label=self.label
        )


class BindGroup:
    """
    A collection of resources bound to shader stages.
    
    Bind groups hold resources (buffers, textures, samplers, etc.) that are
    accessible to shaders. They are created from a bind group layout and
    must match the layout's binding types and counts.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        layout: The bind group layout.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, layout: BindGroupLayout, entries: List[BindGroupEntry], label: str = "") -> None:
        """Initialize the bind group."""
        self.device = device
        self.label = label
        self.layout = layout
        self.entries = entries
        self.tracking_data = None  # Would be TrackingData
        self.late_buffer_binding_infos = [] # Stores LateBufferBindingInfo

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="BindGroup",
            label=self.label
        )
    
    def try_raw(self, guard: Any) -> Any:
        """
        Get the raw HAL bind group, validating all resources are still alive.
        
        Args:
            guard: Snatch guard for resource validation.
            
        Returns:
            The raw HAL bind group.
            
        Raises:
            DestroyedResourceError: If any resource has been destroyed.
        """
        # In Rust: validates all buffers and textures in used_buffer_ranges and used_texture_ranges
        if hasattr(self, 'used_buffer_ranges'):
            for buffer_action in self.used_buffer_ranges:
                if hasattr(buffer_action, 'buffer'):
                    buffer_action.buffer.try_raw(guard)
        
        if hasattr(self, 'used_texture_ranges'):
            for texture_action in self.used_texture_ranges:
                if hasattr(texture_action, 'texture'):
                    texture_action.texture.try_raw(guard)
        
        # Get the raw bind group
        if hasattr(self, 'raw') and self.raw:
            if hasattr(self.raw, 'get'):
                raw = self.raw.get(guard)
                if raw is None:
                    raise DestroyedResourceError(self.error_ident())
                return raw
            return self.raw
        
        raise DestroyedResourceError(self.error_ident())
    
    def validate_dynamic_bindings(
        self,
        bind_group_index: int,
        offsets: List[int]
    ) -> None:
        """
        Validate dynamic binding offsets.
        
        Args:
            bind_group_index: Index of this bind group.
            offsets: List of dynamic offsets to validate.
            
        Raises:
            BindError: If validation fails.
        """
        # In Rust: validates dynamic_binding_info against provided offsets
        if not hasattr(self, 'dynamic_binding_info'):
            self.dynamic_binding_info = []
        
        if len(self.dynamic_binding_info) != len(offsets):
            raise MismatchedDynamicOffsetCountError(
                bind_group=self.error_ident(),
                group=bind_group_index,
                expected=len(self.dynamic_binding_info),
                actual=len(offsets)
            )
        
        for idx, (info, offset) in enumerate(zip(self.dynamic_binding_info, offsets)):
            # Get alignment requirements
            # In Rust: buffer_binding_type_alignment(&self.device.limits, info.binding_type)
            alignment = 256  # Default, would come from device limits
            limit_name = "minUniformBufferOffsetAlignment"
            
            if hasattr(info, 'binding_type'):
                # Determine alignment based on binding type
                binding_type = info.binding_type
                # Check if binding_type is a BufferBindingType enum or similar
                if hasattr(binding_type, 'Uniform') and binding_type == binding_type.Uniform:
                    alignment = self.device.limits.min_uniform_buffer_offset_alignment
                    limit_name = "min_uniform_buffer_offset_alignment"
                elif hasattr(binding_type, 'Storage'):
                    alignment = self.device.limits.min_storage_buffer_offset_alignment
                    limit_name = "min_storage_buffer_offset_alignment"
            
            # Check alignment
            if offset % alignment != 0:
                raise UnalignedDynamicBindingError(
                    bind_group=self.error_ident(),
                    idx=idx,
                    group=bind_group_index,
                    binding=info.binding_idx,
                    offset=offset,
                    alignment=alignment,
                    limit_name=limit_name
                )
            
            # Check bounds
            if offset > info.maximum_dynamic_offset:
                raise DynamicBindingOutOfBoundsError(
                    bind_group=self.error_ident(),
                    idx=idx,
                    group=bind_group_index,
                    binding=info.binding_idx,
                    offset=offset,
                    buffer_size=info.buffer_size,
                    binding_range=info.binding_range,
                    maximum_dynamic_offset=info.maximum_dynamic_offset
                )


class PipelineLayout:
    """
    A layout describing the bind groups for a pipeline.
    
    Pipeline layouts define the bind group layouts that will be used by a
    pipeline. They are used to create compute and render pipelines.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        bind_group_layouts: List of bind group layouts.
        immediate_size: Size of immediate data.
        tracking_data: Data for resource tracking.
    """

    def __init__(
        self, 
        device: Device, 
        bind_group_layouts: List[BindGroupLayout], 
        immediate_size: int = 0,
        label: str = ""
    ) -> None:
        """Initialize the pipeline layout."""
        self.device = device
        self.label = label
        self.bind_group_layouts = bind_group_layouts
        self.immediate_size = immediate_size
        self.tracking_data = None  # Would be TrackingData

    def is_equal(self, other: Any) -> bool:
        """Check if this layout is equal to another."""
        if not isinstance(other, PipelineLayout):
            return False
        if self is other:
            return True
        
        if self.immediate_size != other.immediate_size:
            return False
        
        if len(self.bind_group_layouts) != len(other.bind_group_layouts):
            return False
            
        for a, b in zip(self.bind_group_layouts, other.bind_group_layouts):
            if not a.is_equal(b):
                return False
                
        return True

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="PipelineLayout",
            label=self.label
        )

    def raw(self) -> Any:
        """Get the raw HAL layout."""
        return None
    
    def validate_immediates_ranges(self, offset: int, end_offset: int) -> None:
        """
        Validate immediates match up with expected ranges.
        
        Args:
            offset: Start offset of immediate data.
            end_offset: End offset of immediate data.
            
        Raises:
            ImmediateUploadError: If validation fails.
        """
        # In Rust: validate offset alignment and size
        try:
            import wgt
            IMMEDIATE_DATA_ALIGNMENT = wgt.IMMEDIATE_DATA_ALIGNMENT
        except:
            IMMEDIATE_DATA_ALIGNMENT = 4
        
        if offset % IMMEDIATE_DATA_ALIGNMENT != 0:
            raise ValueError(f"Immediate data offset {offset} is not aligned to {IMMEDIATE_DATA_ALIGNMENT}")
        
        if end_offset > self.immediate_size:
            raise ValueError(
                f"Immediate data range {offset}..{end_offset} exceeds pipeline layout "
                f"immediate size {self.immediate_size}"
            )


# Additional structures and descriptors

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


# Additional error classes

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
