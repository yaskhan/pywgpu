"""
Validation flags and capabilities for Naga validator.

This module contains the IntFlag enums used to control validation behavior
and specify shader capabilities, mirroring the Rust implementation.
"""

from enum import IntFlag


class ValidationFlags(IntFlag):
    """
    Validation flags.
    
    If you are working with trusted shaders, then you may be able
    to save some time by skipping validation.
    
    If you do not perform full validation, invalid shaders may
    cause Naga to panic. If you do perform full validation and
    Validator.validate returns successfully, then Naga promises that
    code generation will either succeed or return an error; it
    should never panic.
    
    The default value for ValidationFlags is ValidationFlags.all().
    """
    EXPRESSIONS = 0x1  # Expressions
    BLOCKS = 0x2  # Statements and blocks of them
    CONTROL_FLOW_UNIFORMITY = 0x4  # Uniformity of control flow for operations that require it
    STRUCT_LAYOUTS = 0x8  # Host-shareable structure layouts
    CONSTANTS = 0x10  # Constants
    BINDINGS = 0x20  # Group, binding, and location attributes
    
    @classmethod
    def all(cls) -> 'ValidationFlags':
        """Get all validation flags enabled."""
        return (cls.EXPRESSIONS | cls.BLOCKS | cls.CONTROL_FLOW_UNIFORMITY | 
                cls.STRUCT_LAYOUTS | cls.CONSTANTS | cls.BINDINGS)
    
    @classmethod
    def default(cls) -> 'ValidationFlags':
        """Get default validation flags (all enabled)."""
        return cls.all()


class Capabilities(IntFlag):
    """
    Allowed IR capabilities.
    
    This controls which features are allowed in the shader module.
    The validator will reject modules that use capabilities not specified.
    """
    IMMEDIATES = 1 << 0  # Support for AddressSpace::Immediate
    FLOAT64 = 1 << 1  # Float values with width = 8
    PRIMITIVE_INDEX = 1 << 2  # Support for BuiltIn::PrimitiveIndex
    TEXTURE_AND_SAMPLER_BINDING_ARRAY = 1 << 3  # Support for binding arrays of sampled textures and samplers
    BUFFER_BINDING_ARRAY = 1 << 4  # Support for binding arrays of uniform buffers
    STORAGE_TEXTURE_BINDING_ARRAY = 1 << 5  # Support for binding arrays of storage textures
    STORAGE_BUFFER_BINDING_ARRAY = 1 << 6  # Support for binding arrays of storage buffers
    CLIP_DISTANCE = 1 << 7  # Support for BuiltIn::ClipDistance
    CULL_DISTANCE = 1 << 8  # Support for BuiltIn::CullDistance
    STORAGE_TEXTURE_16BIT_NORM_FORMATS = 1 << 9  # Support for 16-bit normalized storage texture formats
    MULTIVIEW = 1 << 10  # Support for BuiltIn::ViewIndex
    EARLY_DEPTH_TEST = 1 << 11  # Support for early_depth_test
    MULTISAMPLED_SHADING = 1 << 12  # Support for BuiltIn::SampleIndex and Sampling::Sample
    RAY_QUERY = 1 << 13  # Support for ray queries and acceleration structures
    DUAL_SOURCE_BLENDING = 1 << 14  # Support for generating two sources for blending from fragment shaders
    CUBE_ARRAY_TEXTURES = 1 << 15  # Support for arrayed cube textures
    SHADER_INT64 = 1 << 16  # Support for 64-bit signed and unsigned integers
    SUBGROUP = 1 << 17  # Support for subgroup operations (except barriers) in fragment and compute shaders
    SUBGROUP_BARRIER = 1 << 18  # Support for subgroup barriers in compute shaders
    SUBGROUP_VERTEX_STAGE = 1 << 19  # Support for subgroup operations in the vertex stage
    SHADER_INT64_ATOMIC_MIN_MAX = 1 << 20  # Support for AtomicFunction::Min and Max on 64-bit integers
    SHADER_INT64_ATOMIC_ALL_OPS = 1 << 21  # Support for all atomic operations on 64-bit integers
    SHADER_FLOAT32_ATOMIC = 1 << 22  # Support for atomic operations on 32-bit floats
    TEXTURE_ATOMIC = 1 << 23  # Support for atomic operations on images
    TEXTURE_INT64_ATOMIC = 1 << 24  # Support for atomic operations on 64-bit images
    RAY_HIT_VERTEX_POSITION = 1 << 25  # Support for ray queries returning vertex position
    SHADER_FLOAT16 = 1 << 26  # Support for 16-bit floating-point types
    TEXTURE_EXTERNAL = 1 << 27  # Support for ImageClass::External
    SHADER_FLOAT16_IN_FLOAT32 = 1 << 28  # Support for quantizeToF16, pack2x16float, and unpack2x16float
    SHADER_BARYCENTRICS = 1 << 29  # Support for fragment shader barycentric coordinates
    MESH_SHADER = 1 << 30  # Support for task shaders, mesh shaders, and per-primitive fragment inputs
    MESH_SHADER_POINT_TOPOLOGY = 1 << 31  # Support for mesh shaders which output points
    TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING = 1 << 32  # Support for non-uniform indexing
    BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING = 1 << 33  # Support for non-uniform indexing of uniform buffers
    STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING = 1 << 34  # Support for non-uniform indexing of storage textures
    STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING = 1 << 35  # Support for non-uniform indexing of storage buffers
    COOPERATIVE_MATRIX = 1 << 36  # Support for cooperative matrix types and operations
    PER_VERTEX = 1 << 37  # Support for per-vertex fragment input
    
    @classmethod
    def default(cls) -> 'Capabilities':
        """Get default capabilities (multisampled shading and cube array textures)."""
        return cls.MULTISAMPLED_SHADING | cls.CUBE_ARRAY_TEXTURES


class SubgroupOperationSet(IntFlag):
    """Supported subgroup operations."""
    BASIC = 1 << 0  # Barriers
    VOTE = 1 << 1  # Any, All
    ARITHMETIC = 1 << 2  # Reductions, scans
    BALLOT = 1 << 3  # Ballot, broadcast
    SHUFFLE = 1 << 4  # Shuffle, shuffle xor
    SHUFFLE_RELATIVE = 1 << 5  # Shuffle up, down
    QUAD_FRAGMENT_COMPUTE = 1 << 7  # Quad supported
    
    @classmethod
    def empty(cls) -> 'SubgroupOperationSet':
        """Get empty subgroup operation set."""
        return cls(0)


class ShaderStages(IntFlag):
    """Shader stages."""
    VERTEX = 0x1
    FRAGMENT = 0x2
    COMPUTE = 0x4
    MESH = 0x8
    TASK = 0x10
    COMPUTE_LIKE = COMPUTE | TASK | MESH
    
    @classmethod
    def empty(cls) -> 'ShaderStages':
        """Get empty shader stages."""
        return cls(0)


class TypeFlags(IntFlag):
    """Type capability flags."""
    CONSTRUCTIBLE = 0x1  # Type can be constructed
    HOST_SHAREABLE = 0x2  # Type can be shared with host
    IO_SHAREABLE = 0x4  # Type can be used for entry point IO
    DATA = 0x8  # Type can hold data
    SIZED = 0x10  # Type has a known size
    COPY = 0x20  # Type can be copied
    ARGUMENT = 0x40  # Type can be a function argument
    
    @classmethod
    def empty(cls) -> 'TypeFlags':
        """Get empty type flags."""
        return cls(0)
