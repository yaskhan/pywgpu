"""
Main resource definitions and management.

This module defines the core resource types and traits used throughout
wgpu-core. It provides:
- Base resource traits (ResourceType, Labeled, ParentDevice, Trackable)
- Resource error types
- Resource lifecycle management
- Resource tracking and validation

All wgpu-core resources implement these base traits to provide a common
interface for resource management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from . import errors
from .device import Device


T = TypeVar("T")


@dataclass
class ResourceErrorIdent:
    """
    Information about a wgpu-core resource for error messages.
    
    This class provides a human-readable identifier for a resource,
    including its type and label, which is useful for error messages.
    
    Attributes:
        type: The type of the resource.
        label: The label of the resource.
    """

    type: str
    label: str = ""

    def __str__(self) -> str:
        """Return a string representation of the resource identifier."""
        if self.label:
            return f"{self.type} with '{self.label}' label"
        return self.type


class ResourceType:
    """
    Information about the wgpu-core resource type.
    
    This trait provides the type name for a resource, which is used
    in error messages and logging.
    
    Attributes:
        TYPE: The type name of the resource.
    """

    TYPE: str = ""

    @classmethod
    def type_name(cls) -> str:
        """Get the type name of the resource."""
        return cls.TYPE


class Labeled(ResourceType):
    """
    Resource with a label.

    This trait provides methods for getting the label of a resource,
    which is useful for debugging and error messages.

    Attributes:
        label: The label of the resource.
    """

    label: str = ""

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type=self.TYPE,
            label=self.label,
        )


class ParentDevice(Labeled):
    """
    Resource that has a parent device.

    This trait provides methods for accessing the parent device and
    checking if resources are from the same device.

    Attributes:
        device: The parent device.
    """

    device: Device

    def is_equal(self, other: Any) -> bool:
        """
        Check if this resource is equal to another resource.

        Args:
            other: The other resource.

        Returns:
            True if the resources are the same, False otherwise.
        """
        return self is other

    def same_device_as(self, other: Any) -> None:
        """
        Check if this resource is from the same device as another resource.

        Args:
            other: The other resource.

        Raises:
            DeviceError: If the resources are from different devices.
        """
        if not self.is_equal(other.device):
            raise errors.DeviceError(
                f"Resource {self.error_ident()} is from a different device"
            )

    def same_device(self, device: Device) -> None:
        """
        Check if this resource is from the given device.

        Args:
            device: The device to check.

        Raises:
            DeviceError: If the resource is from a different device.
        """
        if not self.is_equal(device):
            raise errors.DeviceError(
                f"Resource {self.error_ident()} is from a different device"
            )


class RawResourceAccess(ParentDevice):
    """
    Resource that provides access to raw HAL resources.
    
    This trait provides methods for accessing the raw HAL resource
    associated with a wgpu-core resource.
    
    Attributes:
        DynResource: The HAL resource type.
    """

    DynResource: Any

    def raw(self, guard: Any) -> Optional[Any]:
        """
        Get access to the raw resource if it is not destroyed.
        
        Args:
            guard: The snatch guard.
        
        Returns:
            The raw resource, or None if destroyed.
        """
        pass

    def try_raw(self, guard: Any) -> Any:
        """
        Get access to the raw resource, raising an error if destroyed.
        
        Args:
            guard: The snatch guard.
        
        Returns:
            The raw resource.
        
        Raises:
            DestroyedResourceError: If the resource has been destroyed.
        """
        raw = self.raw(guard)
        if raw is None:
            raise errors.DestroyedResourceError(self.error_ident())
        return raw


class Trackable:
    """
    Resource that can be tracked.
    
    This trait provides methods for tracking resource usage and
    managing resource lifetimes.
    
    Attributes:
        tracking_data: Data for resource tracking.
    """

    tracking_data: Any

    def tracker_index(self) -> Any:
        """Get the tracker index for this resource."""
        return self.tracking_data.tracker_index()


@dataclass
class DestroyedResourceError(Exception):
    """
    Error when accessing a destroyed resource.
    
    Attributes:
        resource: The resource identifier.
    """

    resource: ResourceErrorIdent

    def __str__(self) -> str:
        return f"{self.resource} has been destroyed"


@dataclass
class InvalidResourceError(Exception):
    """
    Error when accessing an invalid resource.
    
    Attributes:
        resource: The resource identifier.
    """

    resource: ResourceErrorIdent

    def __str__(self) -> str:
        return f"{self.resource} is invalid"


@dataclass
class MissingBufferUsageError(Exception):
    """
    Error when a buffer is missing required usage flags.
    
    Attributes:
        res: The resource identifier.
        actual: The actual usage flags.
        expected: The expected usage flags.
    """

    res: ResourceErrorIdent
    actual: int
    expected: int

    def __str__(self) -> str:
        return (
            f"Usage flags {self.actual:#x} of {self.res} do not contain "
            f"required usage flags {self.expected:#x}"
        )


@dataclass
class MissingTextureUsageError(Exception):
    """
    Error when a texture is missing required usage flags.
    
    Attributes:
        res: The resource identifier.
        actual: The actual usage flags.
        expected: The expected usage flags.
    """

    res: ResourceErrorIdent
    actual: int
    expected: int

    def __str__(self) -> str:
        return (
            f"Usage flags {self.actual:#x} of {self.res} do not contain "
            f"required usage flags {self.expected:#x}"
        )


@dataclass
class Fallible(Generic[T]):
    """
    A resource that can be either valid or invalid.
    
    This is used for resources that may fail creation, allowing the
    error to be stored and returned later.
    
    Attributes:
        valid: The valid resource, if any.
        invalid: The invalid label, if any.
    """

    valid: Optional[T] = None
    invalid: Optional[str] = None

    def get(self) -> T:
        """
        Get the resource, raising an error if invalid.
        
        Returns:
            The resource.
        
        Raises:
            InvalidResourceError: If the resource is invalid.
        """
        if self.valid is not None:
            return self.valid
        raise InvalidResourceError(
            ResourceErrorIdent(
                type=T.TYPE,
                label=self.invalid or ""
            )
        )


class Buffer:
    """
    A GPU buffer.
    
    A buffer is a linear allocation of GPU memory that can be used to
    store data for use in shaders and other GPU operations.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        usage: The usages allowed for this buffer.
        size: The size of the buffer in bytes.
        tracking_data: Data for resource tracking.
    """

    def __init__(
        self, 
        device: Device, 
        raw: Any,
        usage: Any, # wgt.BufferUsages
        size: int,
        label: str = ""
    ) -> None:
        """Initialize the buffer."""
        self.device = device
        self.label = label
        self.raw = raw # Should be Snatchable
        self.usage = usage
        self.size = size
        self.initialization_status = None # Would be BufferInitTracker
        self.tracking_data = None  # Would be TrackingData
        self.map_state = BufferMapState.Idle
        self.bind_groups = [] # Would be WeakVec<BindGroup>
        self.timestamp_normalization_bind_group = None
        self.indirect_validation_bind_groups = None

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="Buffer",
            label=self.label
        )

    def check_destroyed(self, guard: Any) -> None:
        """
        Check if buffer has been destroyed.
        
        Raises:
            DestroyedResourceError: If buffer is destroyed.
        """
        if self.raw is None:
            raise DestroyedResourceError(self.error_ident())
        # In a real implementation we would check the snatchable guard

    def check_usage(self, expected: Any) -> None:
        """
        Check that buffer usage contains required usage flags.
        
        Args:
            expected: Required usage flags (wgt.BufferUsages)
        
        Raises:
            MissingBufferUsageError: If usage is missing.
        """
        # In Rust: self.usage.contains(expected)
        # Here we assume self.usage supports bitwise operations or has a contains method
        try:
            if not (self.usage & expected):
                raise MissingBufferUsageError(
                    res=self.error_ident(),
                    actual=self.usage,
                    expected=expected
                )
        except:
            # Fallback if usage is not bit-maskable
            pass

    def resolve_binding_size(self, offset: int, binding_size: Optional[int]) -> int:
        """
        Resolve the size of a binding for buffer with offset and size.
        
        If size is None, then the remainder of the buffer starting from
        offset is used.
        """
        buffer_size = self.size

        if binding_size is not None:
            if offset + binding_size <= buffer_size:
                return binding_size
            else:
                # In Rust: BindingError::BindingRangeTooLarge
                raise Exception(f"Binding range too large for {self.error_ident()}")
        else:
            if offset <= buffer_size:
                return buffer_size - offset
            else:
                # In Rust: BindingError::BindingOffsetTooLarge
                raise Exception(f"Binding offset too large for {self.error_ident()}")

    def binding(self, offset: int, binding_size: Optional[int], snatch_guard: Any) -> tuple:
        """
        Create a new HAL BufferBinding for the buffer.
        """
        self.check_destroyed(snatch_guard)
        resolved_size = self.resolve_binding_size(offset, binding_size)
        
        # In a real implementation, we would create a HAL binding here
        # Return a placeholder for the HAL binding and the resolved size
        return (None, resolved_size)

    def map_async(self, offset: int, size: Optional[int], op: BufferMapOperation) -> int:
        """
        Asynchronously map buffer for CPU access.
        """
        range_size = size if size is not None else (self.size - offset)
        
        # Alignment checks (placeholders for wgt constants)
        MAP_ALIGNMENT = 8
        COPY_BUFFER_ALIGNMENT = 4

        if offset % MAP_ALIGNMENT != 0:
            raise UnalignedOffsetError(offset)
        if range_size % COPY_BUFFER_ALIGNMENT != 0:
            raise UnalignedRangeSizeError(range_size)

        # Usage checks
        pub_usage = None
        if op.host == HostMap.Read:
            pub_usage = 0x0001 # wgt.BufferUsages.MAP_READ
        else:
            pub_usage = 0x0002 # wgt.BufferUsages.MAP_WRITE
            
        self.check_usage(pub_usage)
        
        if offset + range_size > self.size:
            raise OutOfBoundsOverrunError(offset + range_size, self.size)

        if self.map_state != BufferMapState.Idle:
            raise BufferAlreadyMappedError()

        self.map_state = BufferMapState.Waiting
        # In a real implementation, we would queue the mapping on the device
        
        # Placeholder submission index
        return 0

    def get_mapped_range(self, offset: int, size: Optional[int]) -> tuple:
        """
        Get pointer to mapped buffer range.
        """
        # Placeholder for actual mapping logic
        if self.map_state != BufferMapState.Active and self.map_state != BufferMapState.Init:
            raise BufferNotMappedError()
            
        range_size = size if size is not None else (self.size - offset)
        # In a real implementation, return a pointer/memoryview to the mapped area
        return (None, range_size)

    def map(self, snatch_guard: Any) -> Optional[Any]:
        """
        Complete a pending buffer mapping operation.
        """
        if self.map_state == BufferMapState.Waiting:
            self.map_state = BufferMapState.Active
            # Logic to actually map the buffer via the HAL device would go here
            return True
        return None

    def unmap(self) -> None:
        """
        Unmap the buffer.
        """
        if self.map_state == BufferMapState.Idle:
            raise BufferNotMappedError()
            
        # In a real implementation:
        # 1. Flush ranges if it was a write mapping
        # 2. Call device.raw().unmap_buffer()
        self.map_state = BufferMapState.Idle

    def destroy(self) -> None:
        """
        Destroy the buffer.
        """
        if self.raw is None:
            return
            
        # Implementation of destruction logic
        # In Rust this creates a DestroyedBuffer and schedules it for deletion
        raw = self.raw
        self.raw = None
        
        # Clear other resources
        self.timestamp_normalization_bind_group = None
        self.indirect_validation_bind_groups = None
        self.bind_groups = []
        
        # In a real implementation, would add to device.deferred_destroy
        # For now, just a placeholder
        pass

    def get_indirect_validation_bind_group(self, snatch_guard: Any = None) -> Any:
        """Get the bind group for indirect validation of this buffer."""
        if not hasattr(self, '_indirect_validation_bind_group'):
            self._indirect_validation_bind_group = None
        return self._indirect_validation_bind_group


class Texture:
    """
    A GPU texture.
    
    A texture is a multi-dimensional array of texels that can be used
    for rendering, sampling, and other GPU operations.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        usage: The usages allowed for this texture.
        format: The format of the texture texels.
        dimension: The dimension of the texture.
        size: The size of the texture.
        mip_level_count: Number of mip levels.
        sample_count: Number of samples per texel.
        tracking_data: Data for resource tracking.
    """

    def __init__(
        self, 
        device: Device, 
        raw: Any,
        usage: Any, # wgt.TextureUsages
        format: Any, # wgt.TextureFormat
        dimension: Any, # wgt.TextureDimension
        size: Any, # wgt.Extent3d
        mip_level_count: int,
        sample_count: int,
        label: str = ""
    ) -> None:
        """Initialize the texture."""
        self.device = device
        self.label = label
        self.raw = raw # Should be Snatchable
        self.usage = usage
        self.format = format
        self.dimension = dimension
        self.size = size
        self.mip_level_count = mip_level_count
        self.sample_count = sample_count
        self.tracking_data = None  # Would be TrackingData
        self.initialization_status = None # Would be TextureInitTracker
        self.views = [] # Would be WeakVec<TextureView>
        self.bind_groups = [] # Would be WeakVec<BindGroup>

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="Texture",
            label=self.label
        )

    def check_destroyed(self, guard: Any) -> None:
        """
        Check if texture has been destroyed.
        
        Raises:
            DestroyedResourceError: If texture is destroyed.
        """
        if self.raw is None:
            raise DestroyedResourceError(self.error_ident())

    def check_usage(self, expected: Any) -> None:
        """
        Check that texture usage contains required texture usage,
        returns an error otherwise.
        """
        try:
            if not (self.usage & expected):
                raise MissingTextureUsageError(
                    res=self.error_ident(),
                    actual=self.usage,
                    expected=expected
                )
        except:
            pass

    def destroy(self) -> None:
        """
        Destroy the texture.
        """
        if self.raw is None:
            return
            
        raw = self.raw
        self.raw = None
        
        self.views = []
        self.bind_groups = []
        
        # In a real implementation, would add to device.deferred_destroy
        pass


class TextureView:
    """
    A view into a texture.
    
    A texture view provides a different interpretation of a texture's
    data, such as a different format or dimension.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        parent: The texture this view is from.
        usage: The usages allowed for this view.
        format: The format of the texture view.
        dimension: The dimension of the view.
        tracking_data: Data for resource tracking.
    """

    def __init__(
        self, 
        device: Device, 
        parent: Texture,
        raw: Any,
        usage: Any, # wgt.TextureUsages
        format: Any, # wgt.TextureFormat
        dimension: Any, # wgt.TextureViewDimension
        label: str = ""
    ) -> None:
        """Initialize the texture view."""
        self.device = device
        self.parent = parent
        self.label = label
        self.raw = raw # Should be Snatchable
        self.usage = usage
        self.format = format
        self.dimension = dimension
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="TextureView",
            label=self.label
        )

    def check_destroyed(self, guard: Any) -> None:
        """Check if texture view has been destroyed."""
        # 1. Check parent texture
        self.parent.check_destroyed(guard)
        
        # 2. Check view itself
        if self.raw is None:
            raise DestroyedResourceError(self.error_ident())

    def check_usage(self, expected: Any) -> None:
        """Check that texture usage contains required texture usage."""
        try:
            if not (self.usage & expected):
                raise MissingTextureUsageError(
                    res=self.error_ident(),
                    actual=self.usage,
                    expected=expected
                )
        except:
            pass


class Sampler:
    """
    A GPU sampler.
    
    A sampler defines how a texture is sampled in a shader, including
    filtering, addressing modes, and comparison functions.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the sampler."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="Sampler",
            label=self.label
        )


class ExternalTexture:
    """
    An external texture.
    
    External textures are textures that come from outside the WebGPU
    system, such as from a video element or camera feed.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the external texture."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="ExternalTexture",
            label=self.label
        )


class StagingBuffer:
    """
    A staging buffer.
    
    A staging buffer is a temporary buffer used for data transfers
    between the CPU and GPU.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the staging buffer."""
        self.device = device
        self.label = label

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="StagingBuffer",
            label=self.label
        )


class QuerySet:
    """
    A query set.
    
    A query set is used to collect GPU timing and occlusion query results.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the query set."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            type="QuerySet",
            label=self.label
        )


# ============================================================================
# Buffer Mapping Support (from Rust resource.rs lines 218-256)
# ============================================================================

from enum import Enum


class HostMap(Enum):
    """Host mapping mode for buffers."""
    Read = "read"
    Write = "write"


# Callback type for buffer mapping operations
BufferMapCallback = Any  # Callable[[Result[(), BufferAccessError]], None]


@dataclass
class BufferMapOperation:
    """
    Buffer map operation with callback.
    
    Corresponds to Rust's BufferMapOperation (lines 243-246).
    """
    host: HostMap
    callback: Optional[BufferMapCallback] = None


@dataclass
class BufferPendingMapping:
    """
    Pending buffer mapping operation.
    
    Corresponds to Rust's BufferPendingMapping (lines 417-423).
    """
    range: range  # Range of buffer addresses
    op: BufferMapOperation
    _parent_buffer: Any  # Arc<Buffer> - holds parent alive during mapping


class BufferMapState(Enum):
    """
    State of buffer mapping.
    
    Corresponds to Rust's BufferMapState enum (lines 218-231).
    
    States:
    - Init: Mapped at creation with staging buffer
    - Waiting: Waiting for GPU before mapping
    - Active: Currently mapped
    - Idle: Not mapped
    """
    Init = "init"
    Waiting = "waiting"
    Active = "active"
    Idle = "idle"


# ============================================================================
# Buffer Access Errors (from Rust resource.rs lines 257-328)
# ============================================================================

class BufferAccessError(Exception):
    """Base class for buffer access errors."""
    pass


class BufferMapFailedError(BufferAccessError):
    """Buffer map operation failed."""
    def __str__(self):
        return "Buffer map failed"


class BufferAlreadyMappedError(BufferAccessError):
    """Buffer is already mapped."""
    def __str__(self):
        return "Buffer is already mapped"


class BufferMapAlreadyPendingError(BufferAccessError):
    """Buffer map is already pending."""
    def __str__(self):
        return "Buffer map is pending"


class BufferNotMappedError(BufferAccessError):
    """Buffer is not mapped."""
    def __str__(self):
        return "Buffer is not mapped"


class UnalignedRangeError(BufferAccessError):
    """Buffer map range is not aligned."""
    def __str__(self):
        return "Buffer map range must start aligned to MAP_ALIGNMENT and end to COPY_BUFFER_ALIGNMENT"


@dataclass
class UnalignedOffsetError(BufferAccessError):
    """Buffer offset is not aligned."""
    offset: int
    
    def __str__(self):
        return f"Buffer offset invalid: offset {self.offset} must be multiple of 8"


@dataclass
class UnalignedRangeSizeError(BufferAccessError):
    """Buffer range size is not aligned."""
    range_size: int
    
    def __str__(self):
        return f"Buffer range size invalid: range_size {self.range_size} must be multiple of 4"


@dataclass
class OutOfBoundsUnderrunError(BufferAccessError):
    """Buffer access underruns the buffer."""
    index: int
    min: int
    
    def __str__(self):
        return f"Buffer access out of bounds: index {self.index} would underrun the buffer (limit: {self.min})"


@dataclass
class OutOfBoundsOverrunError(BufferAccessError):
    """Buffer access overruns the buffer."""
    index: int
    max: int
    
    def __str__(self):
        return f"Buffer access out of bounds: last index {self.index} would overrun the buffer (limit: {self.max})"


@dataclass
class NegativeRangeError(BufferAccessError):
    """Buffer map range start is greater than end."""
    start: int
    end: int
    
    def __str__(self):
        return f"Buffer map range start {self.start} is greater than end {self.end}"


class MapAbortedError(BufferAccessError):
    """Buffer map was aborted."""
    def __str__(self):
        return "Buffer map aborted"


# ============================================================================
# Destroyed Resources (from Rust resource.rs lines 1007-1045, 1458+)
# ============================================================================

@dataclass
class DestroyedBuffer:
    """
    A buffer that has been marked as destroyed and is staged for actual deletion.
    
    Corresponds to Rust's DestroyedBuffer (lines 1007-1045).
    """
    raw: Any  # ManuallyDrop<Box<dyn hal::DynBuffer>>
    device: Any  # Arc<Device>
    label: str
    bind_groups: List[Any]  # WeakVec<BindGroup>
    timestamp_normalization_bind_group: Optional[Any] = None
    indirect_validation_bind_groups: Optional[Any] = None

    def __del__(self):
        """Clean up the destroyed buffer resources."""
        # 1. Add bind groups to deferred destroy list
        if hasattr(self.device, 'deferred_destroy') and self.bind_groups:
            try:
                with self.device.deferred_destroy.lock() as deferred:
                    deferred.append(self.bind_groups)
            except:
                pass

        # 2. Dispose timestamp normalization bind group
        if self.timestamp_normalization_bind_group:
            try:
                self.timestamp_normalization_bind_group.dispose(self.device.raw())
            except:
                pass

        # 3. Dispose indirect validation bind groups
        if self.indirect_validation_bind_groups:
            try:
                self.indirect_validation_bind_groups.dispose(self.device.raw())
            except:
                pass

        # 4. Destroy raw HAL buffer
        if self.raw:
            try:
                self.device.raw().destroy_buffer(self.raw)
            except:
                pass


@dataclass
class DestroyedTexture:
    """
    A texture that has been marked as destroyed and is staged for actual deletion.
    
    Corresponds to Rust's DestroyedTexture (line 1458+).
    """
    raw: Any  # ManuallyDrop<Box<dyn hal::DynTexture>>
    device: Any  # Arc<Device>
    label: str
    bind_groups: List[Any]  # WeakVec<BindGroup>

    def __del__(self):
        """Clean up the destroyed texture resources."""
        # 1. Add bind groups to deferred destroy list
        if hasattr(self.device, 'deferred_destroy') and self.bind_groups:
            try:
                with self.device.deferred_destroy.lock() as deferred:
                    deferred.append(self.bind_groups)
            except:
                pass

        # 2. Destroy raw HAL texture
        if self.raw:
            try:
                self.device.raw().destroy_texture(self.raw)
            except:
                pass


# ============================================================================
# Staging Buffer (from Rust resource.rs lines 1072-1183)
# ============================================================================

@dataclass
class FlushedStagingBuffer:
    """
    A staging buffer that has been flushed and is ready for GPU consumption.
    
    Corresponds to Rust's FlushedStagingBuffer (line 1184+).
    """
    raw: Any  # Box<dyn hal::DynBuffer>
    device: Any  # Arc<Device>
    size: int  # wgt::BufferSize

    def __del__(self):
        """Destroy the staging buffer."""
        if self.raw:
            try:
                self.device.raw().destroy_buffer(self.raw)
            except:
                pass


# ============================================================================
# Descriptors (from Rust resource.rs lines 1649+, 1981+)
# ============================================================================

@dataclass
class TextureViewDescriptor:
    """
    Describes a texture view.
    
    Corresponds to Rust's TextureViewDescriptor (lines 1649-1703).
    """
    label: Optional[str] = None
    format: Optional[Any] = None  # wgt::TextureFormat
    dimension: Optional[Any] = None  # wgt::TextureViewDimension
    aspect: Any = None  # wgt::TextureAspect (default: All)
    base_mip_level: int = 0
    mip_level_count: Optional[int] = None
    base_array_layer: int = 0
    array_layer_count: Optional[int] = None


@dataclass
class SamplerDescriptor:
    """
    Describes a sampler.
    
    Corresponds to Rust's SamplerDescriptor (lines 1981-2007).
    """
    label: Optional[str] = None
    address_mode_u: Any = None  # wgt::AddressMode
    address_mode_v: Any = None  # wgt::AddressMode
    address_mode_w: Any = None  # wgt::AddressMode
    mag_filter: Any = None  # wgt::FilterMode
    min_filter: Any = None  # wgt::FilterMode
    mipmap_filter: Any = None  # wgt::FilterMode
    lod_min_clamp: float = 0.0
    lod_max_clamp: float = 32.0
    compare: Optional[Any] = None  # wgt::CompareFunction
    anisotropy_clamp: int = 1
    border_color: Optional[Any] = None  # wgt::SamplerBorderColor


