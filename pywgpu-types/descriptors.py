from typing import Optional, List, Any
from pydantic import BaseModel, Field
from enum import IntFlag, Enum
from .binding import (
    BindGroupDescriptor,
    BindGroupLayoutDescriptor,
    BindGroupEntry,
    BindGroupLayoutEntry,
)
from .texture import TextureDescriptor, TextureViewDescriptor
from .shader import ShaderModuleDescriptor
from .pipeline import PipelineLayoutDescriptor, PipelineCompilationOptions
from .command import CommandEncoderDescriptor, CommandBufferDescriptor
from .pass_desc import RenderPassDescriptor, ComputePassDescriptor
from .bundle import RenderBundleDescriptor, RenderBundleEncoderDescriptor
from .query import QuerySetDescriptor
from .rt import BlasDescriptor, TlasDescriptor


class Backends(IntFlag):
    """Backends enabled for an instance."""

    VULKAN = 1 << 0
    GL = 1 << 1
    METAL = 1 << 2
    DX12 = 1 << 3
    BROWSER_WEBGPU = 1 << 4
    ALL = VULKAN | GL | METAL | DX12 | BROWSER_WEBGPU
    PRIMARY = VULKAN | METAL | DX12 | BROWSER_WEBGPU


class DeviceDescriptor(BaseModel):
    label: Optional[str] = None
    required_features: List[str] = Field(default_factory=list)
    required_limits: dict = Field(default_factory=dict)


class BufferDescriptor(BaseModel):
    label: Optional[str] = None
    size: int
    usage: int
    mapped_at_creation: bool = False


class ExternalTextureDescriptor(BaseModel):
    label: Optional[str] = None
    source: Any
    color_space: str = "srgb"


class PipelineCacheDescriptor(BaseModel):
    label: Optional[str] = None
    data: Optional[bytes] = None
    fallback: bool = True
