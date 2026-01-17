from typing import List, Optional, Union, Any
from pydantic import BaseModel
from .buffer import BufferBindingType

class BindingType(BaseModel):
    """Type of a binding (Buffer, Texture, Sampler, etc.)."""
    type: str # 'buffer', 'texture', 'sampler', 'storage_texture'
    # Additional fields would depend on the type (e.g. view_dimension for texture)

class BindGroupLayoutEntry(BaseModel):
    """Entry in a BindGroupLayout."""
    binding: int
    visibility: int # ShaderStage flags
    ty: BindingType
    count: Optional[int] = None

class BindGroupLayoutDescriptor(BaseModel):
    label: Optional[str] = None
    entries: List[BindGroupLayoutEntry]

class BindGroupEntry(BaseModel):
    """Entry in a BindGroup."""
    binding: int
    resource: Any # BufferBinding, TextureView, Sampler

class BindGroupDescriptor(BaseModel):
    label: Optional[str] = None
    layout: Any # BindGroupLayout
    entries: List[BindGroupEntry]
