from typing import List, Optional, Union
from pydantic import BaseModel, Field
from .buffer import BufferBindingType
from .texture import TextureViewDimension, TextureFormat, StorageTextureAccess


class SamplerBindingType(BaseModel):
    type: str = "sampler"
    filtering: bool = True
    comparison: bool = False


class TextureBindingType(BaseModel):
    type: str = "texture"
    sample_type: str = "float"
    view_dimension: TextureViewDimension = TextureViewDimension.D2
    multisampled: bool = False


class StorageTextureBindingType(BaseModel):
    type: str = "storage"
    access: StorageTextureAccess = StorageTextureAccess.WRITE_ONLY
    format: TextureFormat
    view_dimension: TextureViewDimension = TextureViewDimension.D2


class BufferBinding(BaseModel):
    buffer: "Buffer"
    offset: int = 0
    size: Optional[int] = None


class AccelerationStructureBindingType(BaseModel):
    type: str = "acceleration-structure"


class BindGroupLayoutEntry(BaseModel):
    """Entry in a BindGroupLayout."""

    binding: int
    visibility: int  # ShaderStage flags
    ty: Union[
        BufferBindingType,
        SamplerBindingType,
        TextureBindingType,
        StorageTextureBindingType,
        AccelerationStructureBindingType,
    ] = Field(..., discriminator="type")
    count: Optional[int] = None


class BindGroupLayoutDescriptor(BaseModel):
    label: Optional[str] = None
    entries: List[BindGroupLayoutEntry]


class BindGroupLayout(BaseModel):
    pass


class BindGroupEntry(BaseModel):
    """Entry in a BindGroup."""

    binding: int
    resource: Union[
        "BufferBinding",
        List["BufferBinding"],
        "TextureView",
        List["TextureView"],
        "Sampler",
        List["Sampler"],
    ]


class BindGroupDescriptor(BaseModel):
    label: Optional[str] = None
    layout: BindGroupLayout
    entries: List[BindGroupEntry]


class BindGroup(BaseModel):
    pass
