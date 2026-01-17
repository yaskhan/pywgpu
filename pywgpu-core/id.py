"""
Resource ID definitions.

This module defines identifiers (IDs) for all wgpu-core resources. IDs are used
to reference resources in the global state. Each ID contains:
- An index: Used to locate the resource in storage
- An epoch: Used to detect stale IDs (when a resource is freed and a new one
  is allocated with the same index)
- A backend: Used to identify which backend the resource belongs to

IDs are designed to be:
- Small (8 bytes on 64-bit systems)
- Copyable and hashable
- Thread-safe
- Efficient for storage and lookups
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from . import hash_utils


class Marker:
    """
    Marker trait used to determine which types uniquely identify a resource.
    
    For example, Device<A> will have the same type of identifier as Device<B>
    because Device<T> for any T defines the same marker type.
    
    This trait is implemented by empty enum types that serve as type parameters
    for Id<T>.
    """

    pass


# Define marker types for each resource
class AdapterMarker(Marker):
    """Marker for Adapter resources."""
    pass


class SurfaceMarker(Marker):
    """Marker for Surface resources."""
    pass


class DeviceMarker(Marker):
    """Marker for Device resources."""
    pass


class QueueMarker(Marker):
    """Marker for Queue resources."""
    pass


class BufferMarker(Marker):
    """Marker for Buffer resources."""
    pass


class StagingBufferMarker(Marker):
    """Marker for StagingBuffer resources."""
    pass


class TextureViewMarker(Marker):
    """Marker for TextureView resources."""
    pass


class TextureMarker(Marker):
    """Marker for Texture resources."""
    pass


class ExternalTextureMarker(Marker):
    """Marker for ExternalTexture resources."""
    pass


class SamplerMarker(Marker):
    """Marker for Sampler resources."""
    pass


class BindGroupLayoutMarker(Marker):
    """Marker for BindGroupLayout resources."""
    pass


class PipelineLayoutMarker(Marker):
    """Marker for PipelineLayout resources."""
    pass


class BindGroupMarker(Marker):
    """Marker for BindGroup resources."""
    pass


class ShaderModuleMarker(Marker):
    """Marker for ShaderModule resources."""
    pass


class RenderPipelineMarker(Marker):
    """Marker for RenderPipeline resources."""
    pass


class ComputePipelineMarker(Marker):
    """Marker for ComputePipeline resources."""
    pass


class PipelineCacheMarker(Marker):
    """Marker for PipelineCache resources."""
    pass


class CommandEncoderMarker(Marker):
    """Marker for CommandEncoder resources."""
    pass


class CommandBufferMarker(Marker):
    """Marker for CommandBuffer resources."""
    pass


class RenderPassEncoderMarker(Marker):
    """Marker for RenderPassEncoder resources."""
    pass


class ComputePassEncoderMarker(Marker):
    """Marker for ComputePassEncoder resources."""
    pass


class RenderBundleEncoderMarker(Marker):
    """Marker for RenderBundleEncoder resources."""
    pass


class RenderBundleMarker(Marker):
    """Marker for RenderBundle resources."""
    pass


class QuerySetMarker(Marker):
    """Marker for QuerySet resources."""
    pass


class BlasMarker(Marker):
    """Marker for Blas resources."""
    pass


class TlasMarker(Marker):
    """Marker for Tlas resources."""
    pass


# Type aliases for IDs
AdapterId = Id[AdapterMarker]
SurfaceId = Id[SurfaceMarker]
DeviceId = Id[DeviceMarker]
QueueId = Id[QueueMarker]
BufferId = Id[BufferMarker]
StagingBufferId = Id[StagingBufferMarker]
TextureViewId = Id[TextureViewMarker]
TextureId = Id[TextureMarker]
ExternalTextureId = Id[ExternalTextureMarker]
SamplerId = Id[SamplerMarker]
BindGroupLayoutId = Id[BindGroupLayoutMarker]
PipelineLayoutId = Id[PipelineLayoutMarker]
BindGroupId = Id[BindGroupMarker]
ShaderModuleId = Id[ShaderModuleMarker]
RenderPipelineId = Id[RenderPipelineMarker]
ComputePipelineId = Id[ComputePipelineMarker]
PipelineCacheId = Id[PipelineCacheMarker]
CommandEncoderId = Id[CommandEncoderMarker]
CommandBufferId = Id[CommandBufferMarker]
RenderPassEncoderId = Id[RenderPassEncoderMarker]
ComputePassEncoderId = Id[ComputePassEncoderMarker]
RenderBundleEncoderId = Id[RenderBundleEncoderMarker]
RenderBundleId = Id[RenderBundleMarker]
QuerySetId = Id[QuerySetMarker]
BlasId = Id[BlasMarker]
TlasId = Id[TlasMarker]


@dataclass
class Id(Generic[Marker]):
    """
    An identifier for a wgpu object.
    
    An Id<T> value identifies a value stored in a Global's Hub.
    
    The Id contains:
    - An index: Used to locate the resource in storage
    - An epoch: Used to detect stale IDs
    - A backend: Used to identify which backend the resource belongs to
    
    Ids are small (8 bytes), copyable, hashable, and thread-safe.
    
    Attributes:
        _raw: The raw underlying representation of the identifier.
    """

    _raw: int

    @classmethod
    def zip(cls, index: int, epoch: int) -> Id[Marker]:
        """
        Zip together an identifier and return its raw underlying representation.
        
        Args:
            index: The index component of the ID.
            epoch: The epoch component of the ID.
        
        Returns:
            A new Id.
        
        Raises:
            ValueError: If both ID components are zero.
        """
        if index == 0 and epoch == 0:
            raise ValueError("IDs may not be zero")
        v = index | (epoch << 32)
        return cls(v)

    def unzip(self) -> tuple[int, int]:
        """
        Unzip a raw identifier into its components.
        
        Returns:
            A tuple of (index, epoch).
        """
        index = self._raw & 0xFFFFFFFF
        epoch = (self._raw >> 32) & 0xFFFFFFFF
        return (index, epoch)

    def __eq__(self, other: Any) -> bool:
        """Compare two IDs for equality."""
        if not isinstance(other, Id):
            return False
        return self._raw == other._raw

    def __hash__(self) -> int:
        """Hash the ID."""
        return hash(self._raw)

    def __repr__(self) -> str:
        """Return a string representation of the ID."""
        index, epoch = self.unzip()
        return f"Id({index}, {epoch})"


@dataclass
class PointerId(Generic[Marker]):
    """
    Identify an object by the pointer returned by Arc::as_ptr.
    
    This is used for tracing. See IDs and tracing in the hub module.
    
    Attributes:
        pointer: The pointer value.
    """

    pointer: int

    def __eq__(self, other: Any) -> bool:
        """Compare two pointer IDs for equality."""
        if not isinstance(other, PointerId):
            return False
        return self.pointer == other.pointer

    def __hash__(self) -> int:
        """Hash the pointer ID."""
        return hash(self.pointer)

    def __repr__(self) -> str:
        """Return a string representation of the pointer ID."""
        return f"PointerId({self.pointer})"
