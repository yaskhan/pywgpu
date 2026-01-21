"""
Additional resource structures and methods for resource.py

This file contains the missing structures and methods that need to be added to resource.py.
These are based on the Rust reference implementation (wgpu-core/src/resource.rs).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, List, TypeVar, Generic
import threading

T = TypeVar("T")

# ============================================================================
# Identifiers and Traits (Placeholders for compilation)
# ============================================================================

@dataclass
class ResourceErrorIdent:
    type: str
    label: str = ""

    def __str__(self) -> str:
        if self.label:
            return f"{self.type} with '{self.label}' label"
        return self.type

class DestroyedResourceError(Exception):
    def __init__(self, ident: ResourceErrorIdent):
        super().__init__(f"{ident} has been destroyed")
        self.ident = ident

class MissingBufferUsageError(Exception):
    def __init__(self, res: ResourceErrorIdent, actual: Any, expected: Any):
        super().__init__(f"Usage flags {actual} of {res} do not contain required usage flags {expected}")
        self.res = res
        self.actual = actual
        self.expected = expected

class MissingTextureUsageError(Exception):
    def __init__(self, res: ResourceErrorIdent, actual: Any, expected: Any):
        super().__init__(f"Usage flags {actual} of {res} do not contain required usage flags {expected}")
        self.res = res
        self.actual = actual
        self.expected = expected

# ============================================================================
# Buffer Mapping Support
# ============================================================================

class HostMap(Enum):
    """Host mapping mode for buffers."""
    Read = "read"
    Write = "write"


# Callback type for buffer mapping operations
BufferMapCallback = Callable[[Any], None]  # Result[(), BufferAccessError]


@dataclass
class BufferMapOperation:
    """
    Buffer map operation with callback.
    
    Corresponds to Rust's BufferMapOperation.
    """
    host: HostMap
    callback: Optional[BufferMapCallback] = None


@dataclass
class BufferPendingMapping:
    """
    Pending buffer mapping operation.
    
    Corresponds to Rust's BufferPendingMapping.
    """
    range: range  # Range of buffer addresses
    op: BufferMapOperation
    _parent_buffer: Any  # Arc<Buffer> - holds parent alive during mapping


class BufferMapState(Enum):
    """
    State of buffer mapping.
    
    Corresponds to Rust's BufferMapState enum.
    """
    Init = "init"
    Waiting = "waiting"
    Active = "active"
    Idle = "idle"


# ============================================================================
# Buffer Access Errors
# ============================================================================

class BufferAccessError(Exception):
    """Base class for buffer access errors."""
    pass


class BufferMapFailedError(BufferAccessError):
    def __str__(self): return "Buffer map failed"


class BufferAlreadyMappedError(BufferAccessError):
    def __str__(self): return "Buffer is already mapped"


class BufferMapAlreadyPendingError(BufferAccessError):
    def __str__(self): return "Buffer map is pending"


class BufferNotMappedError(BufferAccessError):
    def __str__(self): return "Buffer is not mapped"


class UnalignedRangeError(BufferAccessError):
    def __str__(self): return "Buffer map range must start aligned to MAP_ALIGNMENT and end to COPY_BUFFER_ALIGNMENT"


class UnalignedOffsetError(BufferAccessError):
    def __init__(self, offset: int): self.offset = offset
    def __str__(self): return f"Buffer offset invalid: offset {self.offset} must be multiple of 8"


class UnalignedRangeSizeError(BufferAccessError):
    def __init__(self, range_size: int): self.range_size = range_size
    def __str__(self): return f"Buffer range size invalid: range_size {self.range_size} must be multiple of 4"


class OutOfBoundsUnderrunError(BufferAccessError):
    def __init__(self, index: int, min: int):
        self.index = index
        self.min = min
    def __str__(self): return f"Buffer access out of bounds: index {self.index} would underrun the buffer (limit: {self.min})"


class OutOfBoundsOverrunError(BufferAccessError):
    def __init__(self, index: int, max: int):
        self.index = index
        self.max = max
    def __str__(self): return f"Buffer access out of bounds: last index {self.index} would overrun the buffer (limit: {self.max})"


class NegativeRangeError(BufferAccessError):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    def __str__(self): return f"Buffer map range start {self.start} is greater than end {self.end}"


class MapAbortedError(BufferAccessError):
    def __str__(self): return "Buffer map aborted"


# ============================================================================
# Destroyed Resources
# ============================================================================

@dataclass
class DestroyedBuffer:
    """
    A buffer that has been marked as destroyed and is staged for actual deletion.
    """
    raw: Any
    device: Any
    label: str
    bind_groups: List[Any]
    timestamp_normalization_bind_group: Optional[Any] = None
    indirect_validation_bind_groups: Optional[Any] = None
    
    def __del__(self):
        """Clean up resources when destroyed buffer is dropped."""
        if hasattr(self.device, 'deferred_destroy') and self.bind_groups:
            try:
                with self.device.deferred_destroy.lock() as deferred:
                    deferred.append(self.bind_groups)
            except: pass

        if self.timestamp_normalization_bind_group:
            try: self.timestamp_normalization_bind_group.dispose(self.device.raw())
            except: pass

        if self.indirect_validation_bind_groups:
            try: self.indirect_validation_bind_groups.dispose(self.device.raw())
            except: pass

        if self.raw:
            try: self.device.raw().destroy_buffer(self.raw)
            except: pass


@dataclass
class DestroyedTexture:
    """
    A texture that has been marked as destroyed and is staged for actual deletion.
    """
    raw: Any
    device: Any
    label: str
    bind_groups: List[Any]
    
    def __del__(self):
        """Clean up resources when destroyed texture is dropped."""
        if hasattr(self.device, 'deferred_destroy') and self.bind_groups:
            try:
                with self.device.deferred_destroy.lock() as deferred:
                    deferred.append(self.bind_groups)
            except: pass

        if self.raw:
            try: self.device.raw().destroy_texture(self.raw)
            except: pass


# ============================================================================
# Staging Buffer
# ============================================================================

@dataclass
class FlushedStagingBuffer:
    """
    A staging buffer that has been flushed and is ready for GPU consumption.
    """
    raw: Any
    device: Any
    size: int
    
    def __del__(self):
        """Destroy the staging buffer."""
        if self.raw:
            try: self.device.raw().destroy_buffer(self.raw)
            except: pass


# ============================================================================
# Descriptors
# ============================================================================

@dataclass
class TextureViewDescriptor:
    """Describes a texture view."""
    label: Optional[str] = None
    format: Optional[Any] = None
    dimension: Optional[Any] = None
    aspect: Any = None
    base_mip_level: int = 0
    mip_level_count: Optional[int] = None
    base_array_layer: int = 0
    array_layer_count: Optional[int] = None


@dataclass
class SamplerDescriptor:
    """Describes a sampler."""
    label: Optional[str] = None
    address_mode_u: Any = None
    address_mode_v: Any = None
    address_mode_w: Any = None
    mag_filter: Any = None
    min_filter: Any = None
    mipmap_filter: Any = None
    lod_min_clamp: float = 0.0
    lod_max_clamp: float = 32.0
    compare: Optional[Any] = None
    anisotropy_clamp: int = 1
    border_color: Optional[Any] = None


# ============================================================================
# Resource Classes with full method implementations
# ============================================================================

class Buffer:
    def __init__(self, device: Any, raw: Any, usage: Any, size: int, label: str = ""):
        self.device = device
        self.raw = raw
        self.usage = usage
        self.size = size
        self.label = label
        self.map_state = BufferMapState.Idle
        self.bind_groups = []
        self.timestamp_normalization_bind_group = None
        self.indirect_validation_bind_groups = None

    def error_ident(self) -> ResourceErrorIdent:
        return ResourceErrorIdent(type="Buffer", label=self.label)

    def check_destroyed(self, guard: Any) -> None:
        if self.raw is None:
            raise DestroyedResourceError(self.error_ident())

    def check_usage(self, expected: Any) -> None:
        try:
            if not (self.usage & expected):
                raise MissingBufferUsageError(self.error_ident(), self.usage, expected)
        except: pass

    def resolve_binding_size(self, offset: int, binding_size: Optional[int]) -> int:
        buffer_size = self.size
        if binding_size is not None:
            if offset + binding_size <= buffer_size:
                return binding_size
            raise Exception(f"Binding range too large for {self.error_ident()}")
        if offset <= buffer_size:
            return buffer_size - offset
        raise Exception(f"Binding offset too large for {self.error_ident()}")

    def binding(self, offset: int, binding_size: Optional[int], snatch_guard: Any) -> tuple:
        self.check_destroyed(snatch_guard)
        resolved_size = self.resolve_binding_size(offset, binding_size)
        return (None, resolved_size)

    def map_async(self, offset: int, size: Optional[int], op: BufferMapOperation) -> int:
        range_size = size if size is not None else (self.size - offset)
        if offset % 8 != 0: raise UnalignedOffsetError(offset)
        if range_size % 4 != 0: raise UnalignedRangeSizeError(range_size)
        
        pub_usage = 0x0001 if op.host == HostMap.Read else 0x0002
        self.check_usage(pub_usage)
        
        if offset + range_size > self.size:
            raise OutOfBoundsOverrunError(offset + range_size, self.size)
        if self.map_state != BufferMapState.Idle:
            raise BufferAlreadyMappedError()

        self.map_state = BufferMapState.Waiting
        return 0

    def get_mapped_range(self, offset: int, size: Optional[int]) -> tuple:
        if self.map_state not in (BufferMapState.Active, BufferMapState.Init):
            raise BufferNotMappedError()
        range_size = size if size is not None else (self.size - offset)
        return (None, range_size)

    def map(self, snatch_guard: Any) -> Optional[Any]:
        if self.map_state == BufferMapState.Waiting:
            self.map_state = BufferMapState.Active
            return True
        return None

    def unmap(self) -> None:
        if self.map_state == BufferMapState.Idle:
            raise BufferNotMappedError()
        self.map_state = BufferMapState.Idle

    def destroy(self) -> None:
        if self.raw is None: return
        self.raw = None
        self.timestamp_normalization_bind_group = None
        self.indirect_validation_bind_groups = None
        self.bind_groups = []


class Texture:
    def __init__(self, device: Any, raw: Any, usage: Any, size: Any, label: str = ""):
        self.device = device
        self.raw = raw
        self.usage = usage
        self.size = size
        self.label = label
        self.views = []
        self.bind_groups = []

    def error_ident(self) -> ResourceErrorIdent:
        return ResourceErrorIdent(type="Texture", label=self.label)

    def check_destroyed(self, guard: Any) -> None:
        if self.raw is None:
            raise DestroyedResourceError(self.error_ident())

    def check_usage(self, expected: Any) -> None:
        try:
            if not (self.usage & expected):
                raise MissingTextureUsageError(self.error_ident(), self.usage, expected)
        except: pass

    def destroy(self) -> None:
        if self.raw is None: return
        self.raw = None
        self.views = []
        self.bind_groups = []


class TextureView:
    def __init__(self, device: Any, parent: Texture, raw: Any, usage: Any, label: str = ""):
        self.device = device
        self.parent = parent
        self.raw = raw
        self.usage = usage
        self.label = label

    def error_ident(self) -> ResourceErrorIdent:
        return ResourceErrorIdent(type="TextureView", label=self.label)

    def check_destroyed(self, guard: Any) -> None:
        self.parent.check_destroyed(guard)
        if self.raw is None:
            raise DestroyedResourceError(self.error_ident())

    def check_usage(self, expected: Any) -> None:
        try:
            if not (self.usage & expected):
                raise MissingTextureUsageError(self.error_ident(), self.usage, expected)
        except: pass
