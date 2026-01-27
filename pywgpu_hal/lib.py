"""
A cross-platform unsafe graphics abstraction.

This module defines a set of protocols abstracting over modern graphics APIs,
with implementations ("backends") for Vulkan, Metal, Direct3D, and GL.

pywgpu-hal is a Python translation of wgpu-hal, oriented towards WebGPU
implementation goals. It has minimal overhead for validation or tracking,
and the API translation overhead is kept to the bare minimum.

The pywgpu-hal module's main design choices:

- Our protocols are meant to be *portable*: proper use should get equivalent
  results regardless of the backend.

- Our protocols' contracts are *unsafe*: implementations perform minimal
  validation, if any, and incorrect use will often cause undefined behavior.
  This allows us to minimize the overhead we impose over the underlying
  graphics system.

- In the same vein, returned errors *only cover cases the user can't
  anticipate*, like running out of memory or losing the device.

- We use *static dispatch* via Protocols. You must select a specific backend
  type and then use that according to the main protocols.

- We map buffer contents *persistently*. This means that the buffer can
  remain mapped on the CPU while the GPU reads or writes to it.

- You must record *explicit barriers* between different usages of a resource.

- Pipeline layouts are *explicitly specified* when setting bind groups.

- The API *accepts collections as iterators*, to avoid forcing the user to
  store data in particular containers.
"""

from __future__ import annotations
from typing import (
    Protocol,
    TypeVar,
    Generic,
    Optional,
    List,
    Sequence,
    Union,
    Callable,
    Any,
    Iterator,
    Tuple,
)
from dataclasses import dataclass
from enum import IntFlag, Enum
from abc import abstractmethod
import sys

# Import from pywgpu_types for WebGPU types
try:
    import pywgpu_types as wgt
except ImportError:
    # Fallback if not available
    wgt = None  # type: ignore

# Constants
MAX_CONCURRENT_SHADER_STAGES: int = 3
MAX_ANISOTROPY: int = 16
MAX_BIND_GROUPS: int = 8
MAX_VERTEX_BUFFERS: int = 16
MAX_COLOR_ATTACHMENTS: int = 8
MAX_MIP_LEVELS: int = 16
QUERY_SIZE: int = 8  # Size of a single occlusion/timestamp query in bytes

# Type Aliases
Label = Optional[str]
MemoryRange = range
FenceValue = int
DropCallback = Callable[[], None]

# ============================================================================
# Error Types
# ============================================================================


class DeviceError(Exception):
    """Error that can occur when using a device."""

    pass


class OutOfMemoryError(DeviceError):
    """Device ran out of memory."""

    def __init__(self):
        super().__init__("Out of memory")


class DeviceLostError(DeviceError):
    """Device connection was lost."""

    def __init__(self):
        super().__init__("Device is lost")


class UnexpectedError(DeviceError):
    """Unexpected error (driver implementation is at fault)."""

    def __init__(self):
        super().__init__("Unexpected error variant (driver implementation is at fault)")


class ShaderError(Exception):
    """Error that can occur when creating a shader module."""

    def __init__(self, message: str, device_error: Optional[DeviceError] = None):
        self.message = message
        self.device_error = device_error
        super().__init__(message)


class PipelineError(Exception):
    """Error that can occur when creating a pipeline."""

    def __init__(self, message: str, device_error: Optional[DeviceError] = None):
        self.message = message
        self.device_error = device_error
        super().__init__(message)


class PipelineCacheError(Exception):
    """Error that can occur when working with pipeline caches."""

    def __init__(self, device_error: Optional[DeviceError] = None):
        self.device_error = device_error
        super().__init__(str(device_error) if device_error else "Pipeline cache error")


class SurfaceError(Exception):
    """Error that can occur when working with surfaces."""

    pass


class SurfaceLostError(SurfaceError):
    """Surface is lost."""

    def __init__(self):
        super().__init__("Surface is lost")


class SurfaceOutdatedError(SurfaceError):
    """Surface is outdated, needs to be re-created."""

    def __init__(self):
        super().__init__("Surface is outdated, needs to be re-created")


class SurfaceOtherError(SurfaceError):
    """Other surface error."""

    def __init__(self, reason: str):
        super().__init__(f"Other reason: {reason}")


class InstanceError(Exception):
    """Error occurring while trying to create an instance or surface.

    These errors are very platform specific and relate to the state of
    the underlying graphics API or hardware.
    """

    def __init__(self, message: str, source: Optional[Exception] = None):
        self.message = message
        self.source = source
        super().__init__(message)


# ============================================================================
# Bitflags
# ============================================================================


class PipelineLayoutFlags(IntFlag):
    """Pipeline layout creation flags."""

    NONE = 0
    # D3D12: Add support for first_vertex and first_instance builtins
    FIRST_VERTEX_INSTANCE = 1 << 0
    # D3D12: Add support for num_workgroups builtins
    NUM_WORK_GROUPS = 1 << 1
    # D3D12: Add support for the builtins that other flags enable for indirect execution
    INDIRECT_BUILTIN_UPDATE = 1 << 2


class BindGroupLayoutFlags(IntFlag):
    """Bind group layout creation flags."""

    NONE = 0
    # Allows for bind group binding arrays to be shorter than the array in the BGL
    PARTIALLY_BOUND = 1 << 0


class TextureFormatCapabilities(IntFlag):
    """Texture format capability flags."""

    NONE = 0
    # Format can be sampled
    SAMPLED = 1 << 0
    # Format can be sampled with a linear sampler
    SAMPLED_LINEAR = 1 << 1
    # Format can be sampled with a min/max reduction sampler
    SAMPLED_MINMAX = 1 << 2
    # Format can be used as storage with read-only access
    STORAGE_READ_ONLY = 1 << 3
    # Format can be used as storage with write-only access
    STORAGE_WRITE_ONLY = 1 << 4
    # Format can be used as storage with both read and write access
    STORAGE_READ_WRITE = 1 << 5
    # Format can be used as storage with atomics
    STORAGE_ATOMIC = 1 << 6
    # Format can be used as color and input attachment
    COLOR_ATTACHMENT = 1 << 7
    # Format can be used as color (with blending) and input attachment
    COLOR_ATTACHMENT_BLEND = 1 << 8
    # Format can be used as depth-stencil and input attachment
    DEPTH_STENCIL_ATTACHMENT = 1 << 9
    # Format can be multisampled by x2
    MULTISAMPLE_X2 = 1 << 10
    # Format can be multisampled by x4
    MULTISAMPLE_X4 = 1 << 11
    # Format can be multisampled by x8
    MULTISAMPLE_X8 = 1 << 12
    # Format can be multisampled by x16
    MULTISAMPLE_X16 = 1 << 13
    # Format can be used for render pass resolve targets
    MULTISAMPLE_RESOLVE = 1 << 14
    # Format can be copied from
    COPY_SRC = 1 << 15
    # Format can be copied to
    COPY_DST = 1 << 16


class FormatAspects(IntFlag):
    """Texture format aspect flags."""

    NONE = 0
    COLOR = 1 << 0
    DEPTH = 1 << 1
    STENCIL = 1 << 2
    PLANE_0 = 1 << 3
    PLANE_1 = 1 << 4
    PLANE_2 = 1 << 5

    @property
    def DEPTH_STENCIL(self) -> "FormatAspects":
        return FormatAspects.DEPTH | FormatAspects.STENCIL

    @staticmethod
    def new(format: Any, aspect: Any) -> "FormatAspects":
        """Create FormatAspects from texture format and aspect."""
        if wgt is None:
            return FormatAspects.COLOR
        
        aspect_mask = FormatAspects.NONE
        if aspect == wgt.TextureAspect.all:
            aspect_mask = FormatAspects.all_flags()
        elif aspect == wgt.TextureAspect.depth_only:
            aspect_mask = FormatAspects.DEPTH
        elif aspect == wgt.TextureAspect.stencil_only:
            aspect_mask = FormatAspects.STENCIL
        elif aspect == wgt.TextureAspect.plane0:
            aspect_mask = FormatAspects.PLANE_0
        elif aspect == wgt.TextureAspect.plane1:
            aspect_mask = FormatAspects.PLANE_1
        elif aspect == wgt.TextureAspect.plane2:
            aspect_mask = FormatAspects.PLANE_2
            
        return FormatAspects.from_format(format) & aspect_mask

    @staticmethod
    def all_flags() -> "FormatAspects":
        return (FormatAspects.COLOR | FormatAspects.DEPTH | FormatAspects.STENCIL | 
                FormatAspects.PLANE_0 | FormatAspects.PLANE_1 | FormatAspects.PLANE_2)

    @staticmethod
    def from_format(format: Any) -> "FormatAspects":
        if format == wgt.TextureFormat.stencil8:
            return FormatAspects.STENCIL
        if format in (wgt.TextureFormat.depth16unorm, wgt.TextureFormat.depth32float, wgt.TextureFormat.depth24plus):
            return FormatAspects.DEPTH
        if format in (wgt.TextureFormat.depth32float_stencil8, wgt.TextureFormat.depth24plus_stencil8):
            return FormatAspects.DEPTH_STENCIL
        if format in (wgt.TextureFormat.nv12, wgt.TextureFormat.p010):
            return FormatAspects.PLANE_0 | FormatAspects.PLANE_1
        return FormatAspects.COLOR

    def is_one(self) -> bool:
        """Returns True if only one flag is set."""
        return self.value != 0 and (self.value & (self.value - 1)) == 0

    def map(self) -> Any:
        """Map to TextureAspect."""
        if self == FormatAspects.COLOR:
            return wgt.TextureAspect.all
        if self == FormatAspects.DEPTH:
            return wgt.TextureAspect.depth_only
        if self == FormatAspects.STENCIL:
            return wgt.TextureAspect.stencil_only
        if self == FormatAspects.PLANE_0:
            return wgt.TextureAspect.plane0
        if self == FormatAspects.PLANE_1:
            return wgt.TextureAspect.plane1
        if self == FormatAspects.PLANE_2:
            return wgt.TextureAspect.plane2
        raise ValueError(f"Cannot map multi-aspect {self} to TextureAspect")


class MemoryFlags(IntFlag):
    """Memory allocation flags."""

    NONE = 0
    TRANSIENT = 1 << 0
    PREFER_COHERENT = 1 << 1


class AttachmentOps(IntFlag):
    """Attachment load and store operations.

    There must be at least one flag from the LOAD group and one from the STORE group set.
    """

    NONE = 0
    # Load the existing contents of the attachment
    LOAD = 1 << 0
    # Clear the attachment to a specified value
    LOAD_CLEAR = 1 << 1
    # The contents of the attachment are undefined
    LOAD_DONT_CARE = 1 << 2
    # Store the contents of the attachment
    STORE = 1 << 3
    # The contents of the attachment are undefined after the pass
    STORE_DISCARD = 1 << 4


class AccelerationStructureUses(IntFlag):
    """Acceleration structure usage flags."""

    NONE = 0
    # For BLAS used as input for TLAS
    BUILD_INPUT = 1 << 0
    # Target for acceleration structure build
    BUILD_OUTPUT = 1 << 1
    # TLAS used in a shader
    SHADER_INPUT = 1 << 2
    # BLAS used to query compacted size
    QUERY_INPUT = 1 << 3
    # BLAS used as a src for a copy operation
    COPY_SRC = 1 << 4
    # BLAS used as a dst for a copy operation
    COPY_DST = 1 << 5


# ============================================================================
# Enums
# ============================================================================


class AccelerationStructureFormat(Enum):
    """Acceleration structure format."""

    TOP_LEVEL = "TopLevel"
    BOTTOM_LEVEL = "BottomLevel"


class AccelerationStructureBuildMode(Enum):
    """Acceleration structure build mode."""

    BUILD = "Build"
    UPDATE = "Update"


# ============================================================================
# Helper Functions
# ============================================================================


def hal_usage_error(txt: str) -> None:
    """Raise an error for HAL usage violations.

    # Safety
    This should only be called when wgpu-hal invariants are violated.
    """
    raise RuntimeError(f"wgpu-hal invariant was violated (usage error): {txt}")


def hal_internal_error(txt: str) -> None:
    """Raise an error for HAL internal errors.

    # Safety
    This should only be called for preventable internal errors.
    """
    raise RuntimeError(f"wgpu-hal ran into a preventable internal error: {txt}")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Alignments:
    """Alignment requirements for various operations."""

    # The alignment of the start of the buffer used as a GPU copy source
    buffer_copy_offset: int
    # The alignment of the row pitch of the texture data stored in a buffer
    buffer_copy_pitch: int
    # The finest alignment of bound range checking for uniform buffers
    uniform_bounds_check_alignment: int
    # The size of the raw TLAS instance
    raw_tlas_instance_size: int
    # What the scratch buffer for building an acceleration structure must be aligned to
    ray_tracing_scratch_buffer_alignment: int


@dataclass
class Capabilities:
    """Device capabilities."""

    limits: Any  # wgt.Limits
    alignments: Alignments
    downlevel: Any  # wgt.DownlevelCapabilities
    # Supported cooperative matrix configurations (empty if not supported)
    cooperative_matrix_properties: List[Any]  # List[wgt.CooperativeMatrixProperties]


@dataclass
class SurfaceCapabilities:
    """Surface presentation capabilities."""

    # List of supported texture formats (must be at least one)
    formats: List[Any]  # List[wgt.TextureFormat]
    # Range for the number of queued frames
    maximum_frame_latency: Tuple[int, int]  # RangeInclusive
    # Current extent of the surface, if known
    current_extent: Optional[Any]  # Optional[wgt.Extent3d]
    # Supported texture usage flags
    usage: Any  # wgt.TextureUses
    # List of supported V-sync modes (must be at least one)
    present_modes: List[Any]  # List[wgt.PresentMode]
    # List of supported alpha composition modes (must be at least one)
    composite_alpha_modes: List[Any]  # List[wgt.CompositeAlphaMode]


@dataclass
class BufferMapping:
    """Information about a mapped buffer."""

    ptr: int  # Memory address (Python int can hold pointer values)
    is_coherent: bool


@dataclass
class BufferDescriptor:
    """Buffer descriptor."""

    label: Label
    size: int  # wgt.BufferAddress
    usage: Any  # wgt.BufferUses
    memory_flags: MemoryFlags


@dataclass
class TextureDescriptor:
    """Texture descriptor."""

    label: Label
    size: Any  # wgt.Extent3d
    mip_level_count: int
    sample_count: int
    dimension: Any  # wgt.TextureDimension
    format: Any  # wgt.TextureFormat
    usage: Any  # wgt.TextureUses
    memory_flags: MemoryFlags
    # Allows views of this texture to have a different format
    view_formats: List[Any]  # List[wgt.TextureFormat]

    def copy_extent(self) -> "CopyExtent":
        """Get the copy extent for this texture."""
        return CopyExtent.map_extent_to_copy_size(self.size, self.dimension)

    def is_cube_compatible(self) -> bool:
        """Check if texture is cube-compatible."""
        return (self.dimension == wgt.TextureDimension.d2
                and self.size[2] % 6 == 0
                and self.sample_count == 1
                and self.size[0] == self.size[1])

    def array_layer_count(self) -> int:
        """Get the number of array layers."""
        if self.dimension in (wgt.TextureDimension.d1, wgt.TextureDimension.d3):
            return 1
        return self.size[2]


@dataclass
class TextureViewDescriptor:
    """TextureView descriptor.

    Valid usage:
    - format has to be the same as TextureDescriptor.format
    - dimension has to be compatible with TextureDescriptor.dimension
    - usage has to be a subset of TextureDescriptor.usage
    - range has to be a subset of parent texture
    """

    label: Label
    format: Any  # wgt.TextureFormat
    dimension: Any  # wgt.TextureViewDimension
    usage: Any  # wgt.TextureUses
    range: Any  # wgt.ImageSubresourceRange


@dataclass
class SamplerDescriptor:
    """Sampler descriptor."""

    label: Label
    address_modes: Tuple[Any, Any, Any]  # [wgt.AddressMode; 3]
    mag_filter: Any  # wgt.FilterMode
    min_filter: Any  # wgt.FilterMode
    mipmap_filter: Any  # wgt.MipmapFilterMode
    lod_clamp: Tuple[float, float]  # Range<f32>
    compare: Optional[Any]  # Optional[wgt.CompareFunction]
    # Must be in the range [1, 16]
    anisotropy_clamp: int
    border_color: Optional[Any]  # Optional[wgt.SamplerBorderColor]


@dataclass
class BindGroupLayoutDescriptor:
    """BindGroupLayout descriptor.

    Valid usage:
    - entries are sorted by ascending binding number
    """

    label: Label
    flags: BindGroupLayoutFlags
    entries: Sequence[Any]  # Sequence[wgt.BindGroupLayoutEntry]


@dataclass
class InstanceDescriptor:
    """Instance descriptor."""

    name: str
    flags: Any  # wgt.InstanceFlags
    memory_budget_thresholds: Any  # wgt.MemoryBudgetThresholds
    backend_options: Any  # wgt.BackendOptions
    telemetry: Optional[Any]  # Optional[Telemetry]
    display: Optional[Any]  # Optional[DisplayHandle]


@dataclass
class SurfaceConfiguration:
    """Surface configuration."""

    # Maximum number of queued frames
    maximum_frame_latency: int
    # Vertical synchronization mode
    present_mode: Any  # wgt.PresentMode
    # Alpha composition mode
    composite_alpha_mode: Any  # wgt.CompositeAlphaMode
    # Format of the surface textures
    format: Any  # wgt.TextureFormat
    # Requested texture extent
    extent: Any  # wgt.Extent3d
    # Allowed usage of surface textures
    usage: Any  # wgt.TextureUses
    # Allows views of swapchain texture to have a different format
    view_formats: List[Any]  # List[wgt.TextureFormat]


T = TypeVar("T")


@dataclass
class Rect(Generic[T]):
    """Rectangle with generic coordinate type."""

    x: T
    y: T
    w: T
    h: T


@dataclass
class StateTransition(Generic[T]):
    """State transition from one state to another."""

    from_: T  # 'from' is a keyword in Python
    to: T


@dataclass
class CopyExtent:
    """Copy extent."""

    width: int
    height: int
    depth: int

    @staticmethod
    def map_extent_to_copy_size(extent: Any, dim: Any) -> "CopyExtent":
        depth = 1
        if dim == wgt.TextureDimension.d3:
            depth = extent[2]
        return CopyExtent(width=extent[0], height=extent[1], depth=depth)

    def min(self, other: "CopyExtent") -> "CopyExtent":
        return CopyExtent(
            width=min(self.width, other.width),
            height=min(self.height, other.height),
            depth=min(self.depth, other.depth),
        )

    def at_mip_level(self, level: int) -> "CopyExtent":
        return CopyExtent(
            width=max(1, self.width >> level),
            height=max(1, self.height >> level),
            depth=max(1, self.depth >> level),
        )


@dataclass
class TextureCopyBase:
    """Base information for texture copy operations."""

    mip_level: int
    array_layer: int
    # Origin within a texture (for 1D and 2D textures, Z must be 0)
    origin: Any  # wgt.Origin3d
    aspect: FormatAspects

    def max_copy_size(self, full_size: "CopyExtent") -> "CopyExtent":
        mip = full_size.at_mip_level(self.mip_level)
        return CopyExtent(
            width=mip.width - self.origin[0],
            height=mip.height - self.origin[1],
            depth=mip.depth - self.origin[2],
        )


@dataclass
class BufferCopy:
    """Buffer copy operation."""

    src_offset: int  # wgt.BufferAddress
    dst_offset: int  # wgt.BufferAddress
    size: int  # wgt.BufferSize


@dataclass
class TextureCopy:
    """Texture copy operation."""

    src_base: TextureCopyBase
    dst_base: TextureCopyBase
    size: CopyExtent

    def clamp_size_to_virtual(self, full_src_size: "CopyExtent", full_dst_size: "CopyExtent") -> None:
        max_src_size = self.src_base.max_copy_size(full_src_size)
        max_dst_size = self.dst_base.max_copy_size(full_dst_size)
        self.size = self.size.min(max_src_size).min(max_dst_size)


@dataclass
class BufferTextureCopy:
    """Buffer-texture copy operation."""

    buffer_layout: Any  # wgt.TexelCopyBufferLayout
    texture_base: TextureCopyBase
    size: CopyExtent

    def clamp_size_to_virtual(self, full_size: "CopyExtent") -> None:
        max_size = self.texture_base.max_copy_size(full_size)
        self.size = self.size.min(max_size)


@dataclass
class AccelerationStructureBuildSizes:
    """Information of the required size for acceleration structure build."""

    acceleration_structure_size: int  # wgt.BufferAddress
    update_scratch_size: int  # wgt.BufferAddress
    build_scratch_size: int  # wgt.BufferAddress


@dataclass
class AccelerationStructureDescriptor:
    """Acceleration structure descriptor."""

    label: Label
    size: int  # wgt.BufferAddress
    format: AccelerationStructureFormat
    allow_compaction: bool


@dataclass
class AccelerationStructureBarrier:
    """Acceleration structure barrier."""

    usage: StateTransition[AccelerationStructureUses]


@dataclass
class TlasInstance:
    """Top-level acceleration structure instance."""

    transform: List[float]  # [f32; 12]
    custom_data: int  # u32
    mask: int  # u8
    blas_address: int  # u64


# ============================================================================
# Core Traits (Protocols)
# ============================================================================

# Type variables for generic protocols
A = TypeVar("A", bound="Api")


class Api(Protocol):
    """All the types and methods that make up an implementation on top of a backend.

    The api can either be used through generics (through use of this protocol and
    associated types) or dynamically through using the Dyn* protocols.
    """

    # Backend variant identifier
    VARIANT: Any  # wgt.Backend

    # Associated types (would be concrete types in implementations)
    Instance: type
    Surface: type
    Adapter: type
    Device: type
    Queue: type
    CommandEncoder: type
    CommandBuffer: type
    Buffer: type
    Texture: type
    SurfaceTexture: type
    TextureView: type
    Sampler: type
    QuerySet: type
    Fence: type
    BindGroupLayout: type
    BindGroup: type
    PipelineLayout: type
    ShaderModule: type
    RenderPipeline: type
    ComputePipeline: type
    PipelineCache: type
    AccelerationStructure: type


class Instance(Protocol):
    """Instance trait for backend initialization."""

    @abstractmethod
    def init(self, desc: InstanceDescriptor) -> None:
        """Initialize the instance.

        # Safety
        Unsafe - must follow platform-specific initialization requirements.
        """
        ...

    @abstractmethod
    def create_surface(
        self,
        display_handle: Any,  # RawDisplayHandle
        window_handle: Any,  # RawWindowHandle
    ) -> Any:  # Surface
        """Create a surface for the given window.

        # Safety
        Unsafe - handles must be valid for the platform.
        """
        ...

    @abstractmethod
    def enumerate_adapters(
        self, surface_hint: Optional[Any] = None  # Optional[Surface]
    ) -> List[Any]:  # List[ExposedAdapter]
        """Enumerate available adapters.

        surface_hint is only used by the GLES backend targeting WebGL2.

        # Safety
        Unsafe - must be called after init.
        """
        ...


class Surface(Protocol):
    """Surface trait for swapchain management."""

    @abstractmethod
    def configure(self, device: Any, config: SurfaceConfiguration) -> None:
        """Configure surface to use device.

        # Safety
        - All GPU work using self must have been completed
        - All AcquiredSurfaceTextures must have been destroyed
        - All TextureViews derived from AcquiredSurfaceTextures must have been destroyed
        - The surface must not currently be configured to use any other Device
        """
        ...

    @abstractmethod
    def unconfigure(self, device: Any) -> None:
        """Unconfigure surface on device.

        # Safety
        - All GPU work that uses surface must have been completed
        - All AcquiredSurfaceTextures must have been destroyed
        - All TextureViews derived from AcquiredSurfaceTextures must have been destroyed
        - The surface must have been configured on device
        """
        ...

    @abstractmethod
    def acquire_texture(
        self, timeout: Optional[float], fence: Any  # Optional[Duration]  # Fence
    ) -> Optional[Any]:  # Optional[AcquiredSurfaceTexture]
        """Return the next texture to be presented.

        # Safety
        - The surface must currently be configured on some Device
        - The fence argument must be the same Fence passed to all Queue.submit calls
        - You may only have one texture acquired from self at a time
        """
        ...

    @abstractmethod
    def discard_texture(self, texture: Any) -> None:
        """Relinquish an acquired texture without presenting it.

        # Safety
        - The surface must currently be configured on some Device
        - texture must be a SurfaceTexture returned by acquire_texture
        """
        ...


class Adapter(Protocol):
    """Adapter trait for device capabilities."""

    @abstractmethod
    def open(
        self,
        features: Any,  # wgt.Features
        limits: Any,  # wgt.Limits
        memory_hints: Any,  # wgt.MemoryHints
    ) -> Any:  # OpenDevice
        """Open a connection to the device.

        # Safety
        Unsafe - must follow device opening requirements.
        """
        ...

    @abstractmethod
    def texture_format_capabilities(
        self, format: Any  # wgt.TextureFormat
    ) -> TextureFormatCapabilities:
        """Return the set of supported capabilities for a texture format.

        # Safety
        Unsafe - must be called on a valid adapter.
        """
        ...

    @abstractmethod
    def surface_capabilities(self, surface: Any) -> Optional[SurfaceCapabilities]:
        """Returns the capabilities of working with a specified surface.

        None means presentation is not supported for it.

        # Safety
        Unsafe - surface must be valid.
        """
        ...

    @abstractmethod
    def get_presentation_timestamp(self) -> Any:  # wgt.PresentationTimestamp
        """Creates a PresentationTimestamp using the adapter's WSI.

        # Safety
        Unsafe - must be called on a valid adapter.
        """
        ...


class Device(Protocol):
    """A connection to a GPU and a pool of resources to use with it."""

    # Resource creation methods
    @abstractmethod
    def create_buffer(self, desc: BufferDescriptor) -> Any:  # Buffer
        """Creates a new buffer. Initial usage is empty."""
        ...

    @abstractmethod
    def destroy_buffer(self, buffer: Any) -> None:
        """Free buffer and any GPU resources it owns."""
        ...

    @abstractmethod
    def create_texture(self, desc: TextureDescriptor) -> Any:  # Texture
        """Creates a new texture. Initial usage is UNINITIALIZED."""
        ...

    @abstractmethod
    def destroy_texture(self, texture: Any) -> None:
        """Free texture and any GPU resources it owns."""
        ...

    @abstractmethod
    def create_texture_view(
        self, texture: Any, desc: TextureViewDescriptor  # Texture
    ) -> Any:  # TextureView
        """Create a texture view."""
        ...

    @abstractmethod
    def destroy_texture_view(self, view: Any) -> None:
        """Destroy a texture view."""
        ...

    @abstractmethod
    def create_sampler(self, desc: SamplerDescriptor) -> Any:  # Sampler
        """Create a sampler."""
        ...

    @abstractmethod
    def destroy_sampler(self, sampler: Any) -> None:
        """Destroy a sampler."""
        ...

    @abstractmethod
    def create_command_encoder(self, desc: Any) -> Any:  # CommandEncoder
        """Create a fresh CommandEncoder in the "closed" state."""
        ...

    @abstractmethod
    def create_bind_group_layout(self, desc: BindGroupLayoutDescriptor) -> Any:
        """Creates a bind group layout."""
        ...

    @abstractmethod
    def destroy_bind_group_layout(self, bg_layout: Any) -> None:
        """Destroy a bind group layout."""
        ...

    @abstractmethod
    def create_pipeline_layout(self, desc: Any) -> Any:  # PipelineLayout
        """Create a pipeline layout."""
        ...

    @abstractmethod
    def destroy_pipeline_layout(self, pipeline_layout: Any) -> None:
        """Destroy a pipeline layout."""
        ...

    @abstractmethod
    def create_bind_group(self, desc: Any) -> Any:  # BindGroup
        """Create a bind group."""
        ...

    @abstractmethod
    def destroy_bind_group(self, group: Any) -> None:
        """Destroy a bind group."""
        ...

    @abstractmethod
    def create_shader_module(self, desc: Any, shader: Any) -> Any:  # ShaderModule
        """Create a shader module."""
        ...

    @abstractmethod
    def destroy_shader_module(self, module: Any) -> None:
        """Destroy a shader module."""
        ...

    @abstractmethod
    def create_render_pipeline(self, desc: Any) -> Any:  # RenderPipeline
        """Create a render pipeline."""
        ...

    @abstractmethod
    def destroy_render_pipeline(self, pipeline: Any) -> None:
        """Destroy a render pipeline."""
        ...

    @abstractmethod
    def create_compute_pipeline(self, desc: Any) -> Any:  # ComputePipeline
        """Create a compute pipeline."""
        ...

    @abstractmethod
    def destroy_compute_pipeline(self, pipeline: Any) -> None:
        """Destroy a compute pipeline."""
        ...

    @abstractmethod
    def create_query_set(self, desc: Any) -> Any:  # QuerySet
        """Create a query set."""
        ...

    @abstractmethod
    def destroy_query_set(self, set: Any) -> None:
        """Destroy a query set."""
        ...

    @abstractmethod
    def create_fence(self) -> Any:  # Fence
        """Create a fence."""
        ...

    @abstractmethod
    def destroy_fence(self, fence: Any) -> None:
        """Destroy a fence."""
        ...

    @abstractmethod
    def wait(
        self,
        fence: Any,  # Fence
        value: FenceValue,
        timeout: Optional[float],  # Optional[Duration]
    ) -> bool:
        """Wait for fence to reach value. Returns True on success, False on timeout."""
        ...

    # Buffer mapping methods
    @abstractmethod
    def map_buffer(self, buffer: Any, range: MemoryRange) -> BufferMapping:
        """Return a pointer to CPU memory mapping the contents of buffer."""
        ...

    @abstractmethod
    def unmap_buffer(self, buffer: Any) -> None:
        """Remove the mapping established by map_buffer."""
        ...

    @abstractmethod
    def flush_mapped_ranges(self, buffer: Any, ranges: Iterator[MemoryRange]) -> None:
        """Indicate that CPU writes to mapped buffer memory should be made visible to GPU."""
        ...

    @abstractmethod
    def invalidate_mapped_ranges(
        self, buffer: Any, ranges: Iterator[MemoryRange]
    ) -> None:
        """Indicate that GPU writes to mapped buffer memory should be made visible to CPU."""
        ...


class Queue(Protocol):
    """Queue trait for command submission."""

    @abstractmethod
    def submit(
        self,
        command_buffers: Sequence[Any],  # Sequence[CommandBuffer]
        surface_textures: Sequence[Any],  # Sequence[SurfaceTexture]
        signal_fence: Tuple[Any, FenceValue],  # (Fence, FenceValue)
    ) -> None:
        """Submit command_buffers for execution on GPU.

        # Safety
        - Each CommandBuffer must have been created from a CommandEncoder from this Queue's Device
        - Each CommandBuffer must remain alive until execution is complete
        - All resources used by CommandBuffers must remain alive until execution is complete
        - Every SurfaceTexture that any command writes to must appear in surface_textures
        - No SurfaceTexture may appear in surface_textures more than once
        """
        ...

    @abstractmethod
    def present(self, surface: Any, texture: Any) -> None:
        """Present a surface texture.

        # Safety
        - surface and texture must be compatible
        - texture must have been acquired from surface
        """
        ...

    @abstractmethod
    def get_timestamp_period(self) -> float:
        """Get the timestamp period for this queue."""
        ...


class CommandEncoder(Protocol):
    """Encoder and allocation pool for CommandBuffers.

    A CommandEncoder not only constructs CommandBuffers but also acts as the
    allocation pool that owns the buffers' underlying storage.
    """

    @abstractmethod
    def begin_encoding(self, label: Label) -> None:
        """Begin encoding a new command buffer (puts encoder in "recording" state).

        # Safety
        This CommandEncoder must be in the "closed" state.
        """
        ...

    @abstractmethod
    def discard_encoding(self) -> None:
        """Discard the command list under construction (puts encoder in "closed" state).

        # Safety
        This CommandEncoder must be in the "recording" state.
        """
        ...

    @abstractmethod
    def end_encoding(self) -> Any:  # CommandBuffer
        """Return a fresh CommandBuffer holding the recorded commands.

        # Safety
        This CommandEncoder must be in the "recording" state.
        """
        ...

    @abstractmethod
    def reset_all(self, command_buffers: Iterator[Any]) -> None:
        """Reclaim all resources belonging to this CommandEncoder.

        # Safety
        - This CommandEncoder must be in the "closed" state
        - command_buffers must produce all live CommandBuffers built using this encoder
        """
        ...

    # Barrier methods
    @abstractmethod
    def transition_buffers(self, barriers: Iterator[Any]) -> None:
        """Transition buffers between usages."""
        ...

    @abstractmethod
    def transition_textures(self, barriers: Iterator[Any]) -> None:
        """Transition textures between usages."""
        ...

    # Copy operations
    @abstractmethod
    def clear_buffer(self, buffer: Any, range: MemoryRange) -> None:
        """Clear a buffer region."""
        ...

    @abstractmethod
    def copy_buffer_to_buffer(
        self, src: Any, dst: Any, regions: Iterator[BufferCopy]  # Buffer  # Buffer
    ) -> None:
        """Copy from one buffer to another."""
        ...

    @abstractmethod
    def copy_texture_to_texture(
        self,
        src: Any,  # Texture
        src_usage: Any,  # wgt.TextureUses
        dst: Any,  # Texture
        regions: Iterator[TextureCopy],
    ) -> None:
        """Copy from one texture to another."""
        ...

    @abstractmethod
    def copy_buffer_to_texture(
        self,
        src: Any,  # Buffer
        dst: Any,  # Texture
        regions: Iterator[BufferTextureCopy],
    ) -> None:
        """Copy from buffer to texture."""
        ...

    @abstractmethod
    def copy_texture_to_buffer(
        self,
        src: Any,  # Texture
        src_usage: Any,  # wgt.TextureUses
        dst: Any,  # Buffer
        regions: Iterator[BufferTextureCopy],
    ) -> None:
        """Copy from texture to buffer."""
        ...

    # Render pass methods
    @abstractmethod
    def begin_render_pass(self, desc: Any) -> None:
        """Begin a new render pass, clearing all active bindings."""
        ...

    @abstractmethod
    def end_render_pass(self) -> None:
        """End the current render pass."""
        ...

    @abstractmethod
    def set_render_pipeline(self, pipeline: Any) -> None:
        """Set the render pipeline."""
        ...

    @abstractmethod
    def set_bind_group(
        self,
        layout: Any,  # PipelineLayout
        index: int,
        group: Any,  # BindGroup
        dynamic_offsets: Sequence[int],
    ) -> None:
        """Sets the bind group at index."""
        ...

    @abstractmethod
    def draw(
        self,
        first_vertex: int,
        vertex_count: int,
        first_instance: int,
        instance_count: int,
    ) -> None:
        """Draw primitives."""
        ...

    @abstractmethod
    def draw_indexed(
        self,
        first_index: int,
        index_count: int,
        base_vertex: int,
        first_instance: int,
        instance_count: int,
    ) -> None:
        """Draw indexed primitives."""
        ...

    # Compute pass methods
    @abstractmethod
    def begin_compute_pass(self, desc: Any) -> None:
        """Begin a new compute pass, clearing all active bindings."""
        ...

    @abstractmethod
    def end_compute_pass(self) -> None:
        """End the current compute pass."""
        ...

    @abstractmethod
    def set_compute_pipeline(self, pipeline: Any) -> None:
        """Set the compute pipeline."""
        ...

    @abstractmethod
    def dispatch(self, count: Tuple[int, int, int]) -> None:
        """Dispatch compute work groups."""
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "MAX_CONCURRENT_SHADER_STAGES",
    "MAX_ANISOTROPY",
    "MAX_BIND_GROUPS",
    "MAX_VERTEX_BUFFERS",
    "MAX_COLOR_ATTACHMENTS",
    "MAX_MIP_LEVELS",
    "QUERY_SIZE",
    # Type aliases
    "Label",
    "MemoryRange",
    "FenceValue",
    "DropCallback",
    # Error types
    "DeviceError",
    "OutOfMemoryError",
    "DeviceLostError",
    "UnexpectedError",
    "ShaderError",
    "PipelineError",
    "PipelineCacheError",
    "SurfaceError",
    "SurfaceLostError",
    "SurfaceOutdatedError",
    "SurfaceOtherError",
    "InstanceError",
    # Bitflags
    "PipelineLayoutFlags",
    "BindGroupLayoutFlags",
    "TextureFormatCapabilities",
    "FormatAspects",
    "MemoryFlags",
    "AttachmentOps",
    "AccelerationStructureUses",
    # Enums
    "AccelerationStructureFormat",
    "AccelerationStructureBuildMode",
    # Helper functions
    "hal_usage_error",
    "hal_internal_error",
    # Data structures
    "Alignments",
    "Capabilities",
    "SurfaceCapabilities",
    "BufferMapping",
    "BufferDescriptor",
    "TextureDescriptor",
    "TextureViewDescriptor",
    "SamplerDescriptor",
    "BindGroupLayoutDescriptor",
    "InstanceDescriptor",
    "SurfaceConfiguration",
    "Rect",
    "StateTransition",
    "CopyExtent",
    "TextureCopyBase",
    "BufferCopy",
    "TextureCopy",
    "BufferTextureCopy",
    "AccelerationStructureBuildSizes",
    "AccelerationStructureDescriptor",
    "AccelerationStructureBarrier",
    "TlasInstance",
    # Protocols
    "Api",
    "Instance",
    "Surface",
    "Adapter",
    "Device",
    "Queue",
    "CommandEncoder",
]
