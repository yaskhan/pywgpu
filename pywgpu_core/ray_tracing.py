"""
Ray tracing management in wgpu-core.

This module implements ray tracing functionality for wgpu-core, including:
- Bottom Level Acceleration Structures (BLAS)
- Top Level Acceleration Structures (TLAS)
- Acceleration structure building and validation
- Ray tracing pipeline support

Ray tracing allows for efficient rendering of complex scenes by tracing
rays through the scene and computing intersections with geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from . import errors
from .device import Device
from .id import BlasId, BufferId, TlasId
from .resource import (
    DestroyedResourceError,
    InvalidResourceError,
    Labeled,
    MissingBufferUsageError,
    ParentDevice,
    ResourceErrorIdent,
    ResourceType,
)


@dataclass
class CreateBlasError(Exception):
    """
    Error creating a BLAS (Bottom Level Acceleration Structure).
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class CreateTlasError(Exception):
    """
    Error creating a TLAS (Top Level Acceleration Structure).
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class BuildAccelerationStructureError(Exception):
    """
    Error building an acceleration structure.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ValidateAsActionsError(Exception):
    """
    Error validating acceleration structure actions.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class BlasTriangleGeometry:
    """
    Triangle geometry for BLAS building.
    
    Attributes:
        size: Size descriptor for the geometry.
        vertex_buffer: Buffer containing vertex data.
        index_buffer: Optional buffer containing index data.
        transform_buffer: Optional buffer containing transform data.
        first_vertex: Index of the first vertex to use.
        vertex_stride: Stride between vertices.
        first_index: Optional index of the first index to use.
        transform_buffer_offset: Optional offset in the transform buffer.
    """

    size: Any
    vertex_buffer: BufferId
    index_buffer: Optional[BufferId] = None
    transform_buffer: Optional[BufferId] = None
    first_vertex: int = 0
    vertex_stride: int = 0
    first_index: Optional[int] = None
    transform_buffer_offset: Optional[int] = None


@dataclass
class BlasBuildEntry:
    """
    Entry for building a BLAS.
    
    Attributes:
        blas_id: The BLAS ID to build.
        geometries: The geometries to build.
    """

    blas_id: BlasId
    geometries: Any


@dataclass
class TlasBuildEntry:
    """
    Entry for building a TLAS.
    
    Attributes:
        tlas_id: The TLAS ID to build.
        instance_buffer_id: Buffer containing instance data.
        instance_count: Number of instances.
    """

    tlas_id: TlasId
    instance_buffer_id: BufferId
    instance_count: int


@dataclass
class TlasInstance:
    """
    Instance for a TLAS.
    
    Attributes:
        blas_id: The BLAS to instance.
        transform: 12-element transform matrix.
        custom_data: Custom data (24 bits).
        mask: Instance mask.
    """

    blas_id: BlasId
    transform: List[float]
    custom_data: int
    mask: int


@dataclass
class TlasPackage:
    """
    Package of TLAS instances.
    
    Attributes:
        tlas_id: The TLAS ID.
        instances: Iterator of optional instances.
        lowest_unmodified: The lowest unmodified index.
    """

    tlas_id: TlasId
    instances: Any
    lowest_unmodified: int


@dataclass
class TlasBuild:
    """
    TLAS build information.
    
    Attributes:
        tlas: The TLAS being built.
        dependencies: List of BLAS dependencies.
    """

    tlas: Any
    dependencies: List[Any]


@dataclass
class AsBuild:
    """
    Acceleration structure build information.
    
    Attributes:
        blas_s_built: List of built BLAS.
        tlas_s_built: List of built TLAS.
    """

    blas_s_built: List[Any]
    tlas_s_built: List[TlasBuild]


@dataclass
class AsAction:
    """
    Acceleration structure action.
    
    Attributes:
        build: Build action, if any.
        use_tlas: TLAS usage, if any.
    """

    build: Optional[AsBuild] = None
    use_tlas: Optional[Any] = None


@dataclass
class BlasTriangleGeometryInfo:
    """
    Information about a BLAS triangle geometry (without resources).
    
    Attributes:
        size: Size descriptor for the geometry.
        first_vertex: Index of the first vertex to use.
        vertex_stride: Stride between vertices.
        first_index: Optional index of the first index to use.
        transform_buffer_offset: Optional offset in the transform buffer.
    """

    size: Any
    first_vertex: int
    vertex_stride: int
    first_index: Optional[int] = None
    transform_buffer_offset: Optional[int] = None


@dataclass
class BlasPrepareCompactError(Exception):
    """
    Error preparing a BLAS for compaction.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class CompactBlasError(Exception):
    """
    Error compacting a BLAS.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class Blas:
    """
    Bottom Level Acceleration Structure (BLAS).
    
    A BLAS represents a collection of geometry that can be used for ray
    tracing. It contains triangles or other primitives that rays can
    intersect with.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the BLAS."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="Blas",
            label=self.label
        )


@dataclass
class Tlas:
    """
    Top Level Acceleration Structure (TLAS).
    
    A TLAS represents a collection of instances of BLAS. It allows for
    efficient ray tracing of complex scenes with many objects.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the TLAS."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="Tlas",
            label=self.label
        )
