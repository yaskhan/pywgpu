from enum import Enum, IntFlag
from typing import Optional, List
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
    RG8UINT = "rg8uint"
    RG8SINT = "rg8sint"
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
    RGB9E5UFLOAT = "rgb9e5ufloat"
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
    BC2_RGBA_UNORM = "bc2-rgba-unorm"
    BC2_RGBA_UNORM_SRGB = "bc2-rgba-unorm-srgb"
    BC3_RGBA_UNORM = "bc3-rgba-unorm"
    BC3_RGBA_UNORM_SRGB = "bc3-rgba-unorm-srgb"
    BC4_R_UNORM = "bc4-r-unorm"
    BC4_R_SNORM = "bc4-r-snorm"
    BC5_RG_UNORM = "bc5-rg-unorm"
    BC5_RG_SNORM = "bc5-rg-snorm"
    BC6H_RGB_UFLOAT = "bc6h-rgb-ufloat"
    BC6H_RGB_SFLOAT = "bc6h-rgb-sfloat"
    BC7_RGBA_UNORM = "bc7-rgba-unorm"
    BC7_RGBA_UNORM_SRGB = "bc7-rgba-unorm-srgb"
    # ETC2 compressed formats
    ETC2_RGB8UNORM = "etc2-rgb8unorm"
    ETC2_RGB8UNORM_SRGB = "etc2-rgb8unorm-srgb"
    ETC2_RGB8A1UNORM = "etc2-rgb8a1unorm"
    ETC2_RGB8A1UNORM_SRGB = "etc2-rgb8a1unorm-srgb"
    ETC2_RGBA8UNORM = "etc2-rgba8unorm"
    ETC2_RGBA8UNORM_SRGB = "etc2-rgba8unorm-srgb"
    # EAC compressed formats
    EAC_R11UNORM = "eac-r11unorm"
    EAC_R11SNORM = "eac-r11snorm"
    EAC_RG11UNORM = "eac-rg11unorm"
    EAC_RG11SNORM = "eac-rg11snorm"
    # ASTC compressed formats
    ASTC_4X4_UNORM = "astc-4x4-unorm"
    ASTC_4X4_UNORM_SRGB = "astc-4x4-unorm-srgb"
    ASTC_5X4_UNORM = "astc-5x4-unorm"
    ASTC_5X4_UNORM_SRGB = "astc-5x4-unorm-srgb"
    ASTC_5X5_UNORM = "astc-5x5-unorm"
    ASTC_5X5_UNORM_SRGB = "astc-5x5-unorm-srgb"
    ASTC_6X5_UNORM = "astc-6x5-unorm"
    ASTC_6X5_UNORM_SRGB = "astc-6x5-unorm-srgb"
    ASTC_6X6_UNORM = "astc-6x6-unorm"
    ASTC_6X6_UNORM_SRGB = "astc-6x6-unorm-srgb"
    ASTC_8X5_UNORM = "astc-8x5-unorm"
    ASTC_8X5_UNORM_SRGB = "astc-8x5-unorm-srgb"
    ASTC_8X6_UNORM = "astc-8x6-unorm"
    ASTC_8X6_UNORM_SRGB = "astc-8x6-unorm-srgb"
    ASTC_8X8_UNORM = "astc-8x8-unorm"


class TextureViewDimension(Enum):
    D1 = "1d"
    D2 = "2d"
    D2_ARRAY = "2d-array"
    CUBE = "cube"
    CUBE_ARRAY = "cube-array"
    D3 = "3d"


class StorageTextureAccess(Enum):
    WRITE_ONLY = "write-only"
    READ_ONLY = "read-only"
    READ_WRITE = "read-write"
