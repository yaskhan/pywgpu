"""
Pipeline management in wgpu-core.

This module implements WebGPU pipelines, which are objects that encode
the state of the GPU pipeline, including:
- Shader stages (vertex, fragment, compute)
- Bind group layouts
- Vertex buffer layouts
- Rasterization state
- Depth/stencil state
- Multisample state
- Color target states

Pipelines are created from pipeline descriptors and are used to execute
commands on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from . import errors
from .device import Device
from .id import PipelineCacheId, PipelineLayoutId, ShaderModuleId
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


@dataclass
class ShaderModuleDescriptor:
    """
    Descriptor for creating a shader module.
    
    Attributes:
        label: A human-readable label for debugging.
        runtime_checks: Whether to perform runtime checks on shader code.
    """

    label: Optional[str] = None
    runtime_checks: bool = True


@dataclass
class ProgrammableStageDescriptor:
    """
    Describes a programmable pipeline stage.
    
    Attributes:
        module: The compiled shader module for this stage.
        entry_point: The name of the entry point in the compiled shader.
        constants: Pipeline-overridable constants.
        zero_initialize_workgroup_memory: Whether to zero-initialize workgroup memory.
    """

    module: ShaderModuleId
    entry_point: Optional[str] = None
    constants: dict[str, Any] = None
    zero_initialize_workgroup_memory: bool = True


@dataclass
class ComputePipelineDescriptor:
    """
    Describes a compute pipeline.
    
    Attributes:
        label: A human-readable label for debugging.
        layout: The layout of bind groups for this pipeline.
        stage: The compiled compute stage and its entry point.
        cache: The pipeline cache to use when creating this pipeline.
    """

    label: Optional[str] = None
    layout: Optional[PipelineLayoutId] = None
    stage: ProgrammableStageDescriptor = None
    cache: Optional[PipelineCacheId] = None


@dataclass
class VertexBufferLayout:
    """
    Describes how the vertex buffer is interpreted.
    
    Attributes:
        array_stride: The stride, in bytes, between elements of this buffer.
        step_mode: How often this vertex buffer is "stepped" forward.
        attributes: The list of attributes which comprise a single vertex.
    """

    array_stride: int
    step_mode: Any
    attributes: List[Any]


@dataclass
class VertexState:
    """
    Describes the vertex process in a render pipeline.
    
    Attributes:
        stage: The compiled vertex stage and its entry point.
        buffers: The format of any vertex buffers used with this pipeline.
    """

    stage: ProgrammableStageDescriptor
    buffers: List[VertexBufferLayout]


@dataclass
class FragmentState:
    """
    Describes fragment processing in a render pipeline.
    
    Attributes:
        stage: The compiled fragment stage and its entry point.
        targets: The effect of draw calls on the color aspect of the output target.
    """

    stage: ProgrammableStageDescriptor
    targets: List[Optional[Any]]


@dataclass
class RenderPipelineDescriptor:
    """
    Describes a render (graphics) pipeline.
    
    Attributes:
        label: A human-readable label for debugging.
        layout: The layout of bind groups for this pipeline.
        vertex: The vertex processing state for this pipeline.
        primitive: The properties of the pipeline at the primitive assembly and rasterization level.
        depth_stencil: The effect of draw calls on the depth and stencil aspects of the output target.
        multisample: The multi-sampling properties of the pipeline.
        fragment: The fragment processing state for this pipeline.
        multiview_mask: If the pipeline will be used with a multiview render pass.
        cache: The pipeline cache to use when creating this pipeline.
    """

    label: Optional[str] = None
    layout: Optional[PipelineLayoutId] = None
    vertex: VertexState = None
    primitive: Any = None
    depth_stencil: Optional[Any] = None
    multisample: Any = None
    fragment: Optional[FragmentState] = None
    multiview_mask: Optional[int] = None
    cache: Optional[PipelineCacheId] = None


@dataclass
class MeshPipelineDescriptor:
    """
    Describes a mesh shader pipeline.
    
    Attributes:
        label: A human-readable label for debugging.
        layout: The layout of bind groups for this pipeline.
        task: The task processing state for this pipeline.
        mesh: The mesh processing state for this pipeline.
        primitive: The properties of the pipeline at the primitive assembly and rasterization level.
        depth_stencil: The effect of draw calls on the depth and stencil aspects of the output target.
        multisample: The multi-sampling properties of the pipeline.
        fragment: The fragment processing state for this pipeline.
        multiview: If the pipeline will be used with a multiview render pass.
        cache: The pipeline cache to use when creating this pipeline.
    """

    label: Optional[str] = None
    layout: Optional[PipelineLayoutId] = None
    task: Optional[Any] = None
    mesh: Any = None
    primitive: Any = None
    depth_stencil: Optional[Any] = None
    multisample: Any = None
    fragment: Optional[FragmentState] = None
    multiview: Optional[int] = None
    cache: Optional[PipelineCacheId] = None


@dataclass
class ShaderModule:
    """
    A compiled shader module.
    
    A shader module contains compiled shader code that can be used in
    pipelines. It is created from shader source code or bytecode.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the shader module."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="ShaderModule",
            label=self.label
        )


@dataclass
class ComputePipeline:
    """
    A compute pipeline.
    
    A compute pipeline encodes the state of a compute shader execution,
    including the shader module, entry point, and bind group layouts.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the compute pipeline."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="ComputePipeline",
            label=self.label
        )


@dataclass
class PipelineCache:
    """
    A pipeline cache.
    
    A pipeline cache stores compiled pipeline state that can be reused
    across application runs to reduce compilation time.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the pipeline cache."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="PipelineCache",
            label=self.label
        )


@dataclass
class RenderPipeline:
    """
    A render (graphics) pipeline.
    
    A render pipeline encodes the state of a graphics rendering execution,
    including vertex processing, rasterization, and fragment processing.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the render pipeline."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="RenderPipeline",
            label=self.label
        )
