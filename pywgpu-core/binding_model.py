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
        # Implementation depends on wgpu_types.ShaderStages
        pass

    def max(self) -> tuple[Any, int]:
        """
        Get the maximum binding count across all stages.
        
        Returns:
            A tuple of (stage, max_count).
        """
        max_value = max(self.vertex, self.fragment, self.compute)
        stage = None  # Would be ShaderStages based on which stage has max
        return (stage, max_value)

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
            binding: The binding to add.
        """
        # Implementation depends on wgpu_types.BindGroupLayoutEntry
        pass

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
        # Implementation depends on wgpu_types.Limits
        pass

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
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the bind group layout."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="BindGroupLayout",
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
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the bind group."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="BindGroup",
            label=self.label
        )


class PipelineLayout:
    """
    A layout describing the bind groups for a pipeline.
    
    Pipeline layouts define the bind group layouts that will be used by a
    pipeline. They are used to create compute and render pipelines.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the pipeline layout."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="PipelineLayout",
            label=self.label
        )
