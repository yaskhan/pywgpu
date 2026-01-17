from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class VertexFormat(Enum):
    UINT8X2 = "uint8x2"
    UINT8X4 = "uint8x4"
    SINT8X2 = "sint8x2"
    SINT8X4 = "sint8x4"
    UNORM8X2 = "unorm8x2"
    UNORM8X4 = "unorm8x4"
    SNORM8X2 = "snorm8x2"
    SNORM8X4 = "snorm8x4"
    UINT16X2 = "uint16x2"
    UINT16X4 = "uint16x4"
    SINT16X2 = "sint16x2"
    SINT16X4 = "sint16x4"
    UNORM16X2 = "unorm16x2"
    UNORM16X4 = "unorm16x4"
    SNORM16X2 = "snorm16x2"
    SNORM16X4 = "snorm16x4"
    FLOAT16X2 = "float16x2"
    FLOAT16X4 = "float16x4"
    FLOAT32 = "float32"
    FLOAT32X2 = "float32x2"
    FLOAT32X3 = "float32x3"
    FLOAT32X4 = "float32x4"
    UINT32 = "uint32"
    UINT32X2 = "uint32x2"
    UINT32X3 = "uint32x3"
    UINT32X4 = "uint32x4"
    SINT32 = "sint32"
    SINT32X2 = "sint32x2"
    SINT32X3 = "sint32x3"
    SINT32X4 = "sint32x4"
    FLOAT64 = "float64"
    FLOAT64X2 = "float64x2"

class VertexStepMode(Enum):
    VERTEX = "vertex"
    INSTANCE = "instance"

class VertexAttribute(BaseModel):
    format: VertexFormat
    offset: int
    shader_location: int

class VertexBufferLayout(BaseModel):
    array_stride: int
    step_mode: VertexStepMode = VertexStepMode.VERTEX
    attributes: List[VertexAttribute]
