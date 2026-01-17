from enum import Enum, IntFlag
from typing import Optional, List, Any
from pydantic import BaseModel
from .features import Features

class TextureUsage(IntFlag):
    COPY_SRC = 1 << 0
    COPY_DST = 1 << 1
    TEXTURE_BINDING = 1 << 2
    STORAGE_BINDING = 1 << 3
    RENDER_ATTACHMENT = 1 << 4
    # Native features
    STORAGE_ATOMIC = 1 << 16

class TextureDimension(Enum):
    D1 = "1d"
    D2 = "2d"
    D3 = "3d"

class TextureAspect(Enum):
    ALL = "all"
    STENCIL_ONLY = "stencil-only"
    DEPTH_ONLY = "depth-only"

class TextureFormat(Enum):
    # 8-bit formats
    R8UNORM = "r8unorm"
    R8SNORM = "r8snorm"
    R8UINT = "r8uint"
    R8SINT = "r8sint"
    # 16-bit formats
    R16UINT = "r16uint"
    R16SINT = "r16sint"
    R16FLOAT = "r16float"
    RG8UNORM = "rg8unorm"
    RG8SNORM = "rg8snorm"
    # 32-bit formats
    R32UINT = "r32uint"
    R32SINT = "r32sint"
    R32FLOAT = "r32float"
    RG16UINT = "rg16uint"
    RG16SINT = "rg16sint"
    RG16FLOAT = "rg16float"
    RGBA8UNORM = "rgba8unorm"
    RGBA8UNORM_SRGB = "rgba8unorm-srgb"
    RGBA8SNORM = "rgba8snorm"
    RGBA8UINT = "rgba8uint"
    RGBA8SINT = "rgba8sint"
    BGRA8UNORM = "bgra8unorm"
    BGRA8UNORM_SRGB = "bgra8unorm-srgb"
    # Packed 32-bit formats
    RGB10A2UNORM = "rgb10a2unorm"
    RG11B10UFLOAT = "rg11b10ufloat"
    # 64-bit formats
    RG32UINT = "rg32uint"
    RG32SINT = "rg32sint"
    RG32FLOAT = "rg32float"
    RGBA16UINT = "rgba16uint"
    RGBA16SINT = "rgba16sint"
    RGBA16FLOAT = "rgba16float"
    # 128-bit formats
    RGBA32UINT = "rgba32uint"
    RGBA32SINT = "rgba32sint"
    RGBA32FLOAT = "rgba32float"
    # Depth/Stencil formats
    STENCIL8 = "stencil8"
    DEPTH16UNORM = "depth16unorm"
    DEPTH24PLUS = "depth24plus"
    DEPTH24PLUS_STENCIL8 = "depth24plus-stencil8"
    DEPTH32FLOAT = "depth32float"
    DEPTH32FLOAT_STENCIL8 = "depth32float-stencil8"
    # BC compressed formats
    BC1_RGBA_UNORM = "bc1-rgba-unorm"
    BC1_RGBA_UNORM_SRGB = "bc1-rgba-unorm-srgb"
    # ... many others

class Extent3d(BaseModel):
    width: int
    height: int = 1
    depth_or_array_layers: int = 1

class TextureDescriptor(BaseModel):
    label: Optional[str] = None
    size: Extent3d
    mip_level_count: int = 1
    sample_count: int = 1
    dimension: TextureDimension = TextureDimension.D2
    format: TextureFormat
    usage: int
    view_formats: List[TextureFormat] = []

class TextureViewDimension(Enum):
    D1 = "1d"
    D2 = "2d"
    D2_ARRAY = "2d-array"
    CUBE = "cube"
    CUBE_ARRAY = "cube-array"
    D3 = "3d"

class TextureViewDescriptor(BaseModel):
    label: Optional[str] = None
    format: Optional[TextureFormat] = None
    dimension: Optional[TextureViewDimension] = None
    aspect: TextureAspect = TextureAspect.ALL
    base_mip_level: int = 0
    mip_level_count: Optional[int] = None
    base_array_layer: int = 0
    array_layer_count: Optional[int] = None
