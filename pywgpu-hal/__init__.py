"""
pywgpu-hal: Hardware Abstraction Layer for pywgpu

A cross-platform unsafe graphics abstraction providing low-level access to
modern graphics APIs (Vulkan, Metal, Direct3D, OpenGL).

This module defines protocols and types for portable GPU programming with
minimal overhead. Validation and safety are the caller's responsibility.
"""

# Core types and protocols
from .lib import (
    # Constants
    MAX_CONCURRENT_SHADER_STAGES,
    MAX_ANISOTROPY,
    MAX_BIND_GROUPS,
    MAX_VERTEX_BUFFERS,
    MAX_COLOR_ATTACHMENTS,
    MAX_MIP_LEVELS,
    QUERY_SIZE,
    # Type aliases
    Label,
    MemoryRange,
    FenceValue,
    DropCallback,
    # Error types
    DeviceError,
    OutOfMemoryError,
    DeviceLostError,
    UnexpectedError,
    ShaderError,
    PipelineError,
    PipelineCacheError,
    SurfaceError,
    SurfaceLostError,
    SurfaceOutdatedError,
    SurfaceOtherError,
    InstanceError,
    # Bitflags
    PipelineLayoutFlags,
    BindGroupLayoutFlags,
    TextureFormatCapabilities,
    FormatAspects,
    MemoryFlags,
    AttachmentOps,
    AccelerationStructureUses,
    # Enums
    AccelerationStructureFormat,
    AccelerationStructureBuildMode,
    # Helper functions
    hal_usage_error,
    hal_internal_error,
    # Data structures
    Alignments,
    Capabilities,
    SurfaceCapabilities,
    BufferMapping,
    BufferDescriptor,
    TextureDescriptor,
    TextureViewDescriptor,
    SamplerDescriptor,
    BindGroupLayoutDescriptor,
    InstanceDescriptor,
    SurfaceConfiguration,
    Rect,
    StateTransition,
    CopyExtent,
    TextureCopyBase,
    BufferCopy,
    TextureCopy,
    BufferTextureCopy,
    AccelerationStructureBuildSizes,
    AccelerationStructureDescriptor,
    AccelerationStructureBarrier,
    TlasInstance,
    # Protocols
    Api,
    Instance,
    Surface,
    Adapter,
    Device,
    Queue,
    CommandEncoder,
)

# Backend APIs (conditional imports based on availability)
try:
    from . import vulkan
except ImportError:
    vulkan = None  # type: ignore

try:
    from . import metal
except ImportError:
    metal = None  # type: ignore

try:
    from . import dx12
except ImportError:
    dx12 = None  # type: ignore

try:
    from . import gles
except ImportError:
    gles = None  # type: ignore

try:
    from . import noop
except ImportError:
    noop = None  # type: ignore

# Validation canary (optional feature)
try:
    from .validation_canary import VALIDATION_CANARY, ValidationCanary
except ImportError:
    VALIDATION_CANARY = None
    ValidationCanary = None  # type: ignore

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
    # Backends
    "vulkan",
    "metal",
    "dx12",
    "gles",
    "noop",
    # Validation
    "VALIDATION_CANARY",
    "ValidationCanary",
]
