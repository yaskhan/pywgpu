from typing import Optional, List, Any, Union
from pydantic import BaseModel
from enum import IntFlag, Enum

class AccelerationStructureFlags(IntFlag):
    NONE = 0
    PREFER_FAST_BUILD = 1 << 0
    PREFER_FAST_TRACE = 1 << 1
    ALLOW_UPDATE = 1 << 2

class AccelerationStructureUpdateMode(Enum):
    BUILD = 'build'
    UPDATE = 'update'

class AccelerationStructureGeometryFlags(IntFlag):
    NONE = 0
    OPAQUE = 1 << 0
    NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1

class BlasTriangleGeometrySizeDescriptor(BaseModel):
    vertex_format: str
    vertex_count: int
    index_format: Optional[str] = None
    index_count: Optional[int] = None
    flags: AccelerationStructureGeometryFlags = AccelerationStructureGeometryFlags.NONE

class BlasDescriptor(BaseModel):
    label: Optional[str] = None
    flags: AccelerationStructureFlags = AccelerationStructureFlags.NONE
    update_mode: AccelerationStructureUpdateMode = AccelerationStructureUpdateMode.BUILD

class TlasDescriptor(BaseModel):
    label: Optional[str] = None
    max_instances: int
    flags: AccelerationStructureFlags = AccelerationStructureFlags.NONE
    update_mode: AccelerationStructureUpdateMode = AccelerationStructureUpdateMode.BUILD

class TlasInstance(BaseModel):
    blas: Any
    transform: List[float] # 3x4 matrix as 12 floats
    custom_index: int = 0
    mask: int = 0xFF

class BlasTriangleGeometry(BaseModel):
    size: BlasTriangleGeometrySizeDescriptor
    vertex_buffer: Any
    first_vertex: int = 0
    vertex_stride: int
    index_buffer: Optional[Any] = None
    first_index: Optional[int] = None
    transform_buffer: Optional[Any] = None
    transform_buffer_offset: Optional[int] = None

class BlasBuildEntry(BaseModel):
    blas: Any
    geometry: Union[List[BlasTriangleGeometry], Any] # Simplified for now
