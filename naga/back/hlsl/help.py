"""
Helpers for the HLSL backend.

Important note about Expression::ImageQuery/Expression::ArrayLength and HLSL backend:

Due to implementation of GetDimensions function in HLSL, backend can't work with it as an expression.
Instead, it generates a unique wrapped function per Expression::ImageQuery, based on texture info and query function.
See WrappedImageQuery struct that represents a unique function and will be generated before writing all statements and expressions.

This allowed to work with Expression::ImageQuery as expression and write wrapped function.

For example:
```wgsl
let dim_1d = textureDimensions(image_1d);
```

```hlsl
int NagaDimensions1D(Texture1D<float4>)
{
   uint4 ret;
   image_1d.GetDimensions(ret.x);
   return ret.x;
}

int dim_1d = NagaDimensions1D(image_1d);
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib

if TYPE_CHECKING:
    from ...ir.module import Module
    from ...ir.type import Type, Scalar
    from ...ir.expression import Expression
    from ...ir import (
        Handle,
        ImageClass,
        ImageDimension,
        ScalarKind,
        VectorSize,
        UnaryOperator,
        BinaryOperator,
        MathFunction,
    )
    from ...ir.image_query import ImageQuery
    from .. import Level, FunctionCtx


class WrappedType(Enum):
    """Types of wrapped expressions/functions."""

    ARRAY_LENGTH = 0
    IMAGE_LOAD = 1
    IMAGE_SAMPLE = 2
    IMAGE_QUERY = 3
    CONSTRUCTOR = 4
    STRUCT_MATRIX_ACCESS = 5
    MAT_CX2 = 6
    MATH = 7
    ZERO_VALUE = 8
    UNARY_OP = 9
    SATURATE = 10
    BINARY_OP = 11
    LOAD = 12
    IMAGE_GATHER = 13
    RAY_QUERY = 14


@dataclass(frozen=True)
class WrappedArrayLength:
    """Wrapper for array length queries."""

    writable: bool


@dataclass(frozen=True)
class WrappedImageLoad:
    """Wrapper for image load operations."""

    class: ImageClass


@dataclass(frozen=True)
class WrappedImageSample:
    """Wrapper for image sample operations."""

    class: ImageClass
    clamp_to_edge: bool


@dataclass(frozen=True)
class WrappedImageQuery:
    """Wrapper for image query operations."""

    dim: ImageDimension
    arrayed: bool
    class: ImageClass
    query: ImageQuery


@dataclass(frozen=True)
class WrappedConstructor:
    """Wrapper for type constructors."""

    ty: Handle[Type]


@dataclass(frozen=True)
class WrappedStructMatrixAccess:
    """Wrapper for struct matrix member access."""

    ty: Handle[Type]
    index: int


@dataclass(frozen=True)
class WrappedMatCx2:
    """Wrapper for Cx2 matrices."""

    columns: VectorSize


@dataclass(frozen=True)
class WrappedMath:
    """Wrapper for math functions."""

    fun: MathFunction
    scalar: Scalar
    components: Optional[int]


@dataclass(frozen=True)
class WrappedZeroValue:
    """Wrapper for zero value initialization."""

    ty: Handle[Type]


@dataclass(frozen=True)
class WrappedUnaryOp:
    """Wrapper for unary operations."""

    op: UnaryOperator
    ty: Tuple[Optional[VectorSize], Scalar]


@dataclass(frozen=True)
class WrappedSaturate:
    """Wrapper for saturate operation."""

    scalar: Scalar


@dataclass(frozen=True)
class WrappedBinaryOp:
    """Wrapper for binary operations."""

    op: BinaryOperator
    left_scalar: Scalar
    right_scalar: Scalar
    components: Optional[int]


@dataclass(frozen=True)
class WrappedLoad:
    """Wrapper for load operations."""

    pointer: Handle[Expression]


@dataclass(frozen=True)
class WrappedImageGather:
    """Wrapper for image gather operations."""

    dim: ImageDimension
    arrayed: bool
    class: ImageClass


@dataclass(frozen=True)
class WrappedRayQuery:
    """Wrapper for ray query operations."""

    ty: Handle[Type]
    query: str


# Constants for wrapped function names
ABS_FUNCTION = "NagaAbs"
DIV_FUNCTION = "NagaDiv"
EXTRACT_BITS_FUNCTION = "NagaExtractBits"
INSERT_BITS_FUNCTION = "NagaInsertBits"
F2I32_FUNCTION = "NagaF32toI32"
F2I64_FUNCTION = "NagaF32toI64"
F2U32_FUNCTION = "NagaF32toU32"
F2U64_FUNCTION = "NagaF32toU64"
IMAGE_LOAD_EXTERNAL_FUNCTION = "NagaImageLoad"
IMAGE_SAMPLE_BASE_CLAMP_TO_EDGE_FUNCTION = "NagaImageSampleBaseClampToEdge"
MOD_FUNCTION = "NagaMod"
NEG_FUNCTION = "NagaNeg"


def hash_wrapped_type(wrapped: any) -> str:
    """Generate a unique hash for a wrapped type to use in function names.

    Args:
        wrapped: The wrapped type to hash

    Returns:
        A unique string identifier
    """
    # Convert to string representation
    if isinstance(wrapped, WrappedArrayLength):
        s = f"AL_{wrapped.writable}"
    elif isinstance(wrapped, WrappedImageLoad):
        s = f"IL_{wrapped.class}"
    elif isinstance(wrapped, WrappedImageSample):
        s = f"IS_{wrapped.class}_{wrapped.clamp_to_edge}"
    elif isinstance(wrapped, WrappedImageQuery):
        s = f"IQ_{wrapped.dim}_{wrapped.arrayed}_{wrapped.class}_{wrapped.query}"
    elif isinstance(wrapped, WrappedConstructor):
        s = f"C_{wrapped.ty.index}"
    elif isinstance(wrapped, WrappedStructMatrixAccess):
        s = f"SMA_{wrapped.ty.index}_{wrapped.index}"
    elif isinstance(wrapped, WrappedMatCx2):
        s = f"MC2_{wrapped.columns}"
    elif isinstance(wrapped, WrappedMath):
        s = f"M_{wrapped.fun}_{wrapped.scalar}_{wrapped.components}"
    elif isinstance(wrapped, WrappedZeroValue):
        s = f"Z_{wrapped.ty.index}"
    elif isinstance(wrapped, WrappedUnaryOp):
        s = f"U_{wrapped.op}_{wrapped.ty}"
    elif isinstance(wrapped, WrappedSaturate):
        s = f"S_{wrapped.scalar}"
    elif isinstance(wrapped, WrappedBinaryOp):
        s = f"B_{wrapped.op}_{wrapped.left_scalar}_{wrapped.right_scalar}_{wrapped.components}"
    elif isinstance(wrapped, WrappedLoad):
        s = f"L_{wrapped.pointer.index}"
    elif isinstance(wrapped, WrappedImageGather):
        s = f"IG_{wrapped.dim}_{wrapped.arrayed}_{wrapped.class}"
    elif isinstance(wrapped, WrappedRayQuery):
        s = f"RQ_{wrapped.ty.index}_{wrapped.query}"
    else:
        s = str(wrapped)

    # Hash the string
    hash_obj = hashlib.md5(s.encode())
    return hash_obj.hexdigest()[:8]


def get_array_length_name(wrapped: WrappedArrayLength) -> str:
    """Get function name for wrapped array length.

    Args:
        wrapped: The wrapped array length

    Returns:
        Function name
    """
    return f"NagaArrayLength{hash_wrapped_type(wrapped)}"


def get_image_load_name(wrapped: WrappedImageLoad) -> str:
    """Get function name for wrapped image load.

    Args:
        wrapped: The wrapped image load

    Returns:
        Function name
    """
    return f"{IMAGE_LOAD_EXTERNAL_FUNCTION}{hash_wrapped_type(wrapped)}"


def get_image_sample_name(wrapped: WrappedImageSample) -> str:
    """Get function name for wrapped image sample.

    Args:
        wrapped: The wrapped image sample

    Returns:
        Function name
    """
    suffix = "Clamp" if wrapped.clamp_to_edge else ""
    return f"{IMAGE_SAMPLE_BASE_CLAMP_TO_EDGE_FUNCTION}{suffix}{hash_wrapped_type(wrapped)}"


def get_image_query_name(wrapped: WrappedImageQuery) -> str:
    """Get function name for wrapped image query.

    Args:
        wrapped: The wrapped image query

    Returns:
        Function name
    """
    return f"NagaImageQuery{hash_wrapped_type(wrapped)}"


def get_constructor_name(wrapped: WrappedConstructor) -> str:
    """Get function name for wrapped constructor.

    Args:
        wrapped: The wrapped constructor

    Returns:
        Function name
    """
    return f"NagaConstructor{hash_wrapped_type(wrapped)}"


def get_struct_matrix_access_name(wrapped: WrappedStructMatrixAccess, getter: bool) -> str:
    """Get function name for wrapped struct matrix access.

    Args:
        wrapped: The wrapped struct matrix access
        getter: True for getter, False for setter

    Returns:
        Function name
    """
    prefix = "Get" if getter else "Set"
    return f"Naga{prefix}Matrix{hash_wrapped_type(wrapped)}"


def get_mat_cx2_name(wrapped: WrappedMatCx2) -> str:
    """Get function name for wrapped Cx2 matrix.

    Args:
        wrapped: The wrapped Cx2 matrix

    Returns:
        Function name
    """
    return f"NagaMatCx2{hash_wrapped_type(wrapped)}"


def get_math_function_name(wrapped: WrappedMath) -> str:
    """Get function name for wrapped math function.

    Args:
        wrapped: The wrapped math function

    Returns:
        Function name
    """
    return f"NagaMath{hash_wrapped_type(wrapped)}"


def get_zero_value_name(wrapped: WrappedZeroValue) -> str:
    """Get function name for wrapped zero value.

    Args:
        wrapped: The wrapped zero value

    Returns:
        Function name
    """
    return f"NagaZero{hash_wrapped_type(wrapped)}"


def get_unary_op_name(wrapped: WrappedUnaryOp) -> str:
    """Get function name for wrapped unary operation.

    Args:
        wrapped: The wrapped unary operation

    Returns:
        Function name
    """
    return f"NagaUnary{hash_wrapped_type(wrapped)}"


def get_saturate_name(wrapped: WrappedSaturate) -> str:
    """Get function name for wrapped saturate.

    Args:
        wrapped: The wrapped saturate

    Returns:
        Function name
    """
    return f"NagaSaturate{hash_wrapped_type(wrapped)}"


def get_binary_op_name(wrapped: WrappedBinaryOp) -> str:
    """Get function name for wrapped binary operation.

    Args:
        wrapped: The wrapped binary operation

    Returns:
        Function name
    """
    return f"NagaBinary{hash_wrapped_type(wrapped)}"


def get_load_name(wrapped: WrappedLoad) -> str:
    """Get function name for wrapped load.

    Args:
        wrapped: The wrapped load

    Returns:
        Function name
    """
    return f"NagaLoad{hash_wrapped_type(wrapped)}"


def get_image_gather_name(wrapped: WrappedImageGather) -> str:
    """Get function name for wrapped image gather.

    Args:
        wrapped: The wrapped image gather

    Returns:
        Function name
    """
    return f"NagaImageGather{hash_wrapped_type(wrapped)}"


def get_ray_query_name(wrapped: WrappedRayQuery) -> str:
    """Get function name for wrapped ray query.

    Args:
        wrapped: The wrapped ray query

    Returns:
        Function name
    """
    return f"NagaRayQuery{hash_wrapped_type(wrapped)}"


def is_signed(scalar_kind: ScalarKind) -> bool:
    """Check if scalar kind is signed.

    Args:
        scalar_kind: The scalar kind to check

    Returns:
        True if signed, False otherwise
    """
    return scalar_kind in (ScalarKind.SINT, ScalarKind.F16, ScalarKind.F32, ScalarKind.F64)


def type_to_hlsl_scalar(scalar_kind: ScalarKind) -> str:
    """Convert scalar kind to HLSL type string.

    Args:
        scalar_kind: The scalar kind

    Returns:
        HLSL type string
    """
    match scalar_kind:
        case ScalarKind.F16:
            return "half"
        case ScalarKind.F32:
            return "float"
        case ScalarKind.F64:
            return "double"
        case ScalarKind.SINT:
            return "int"
        case ScalarKind.UINT:
            return "uint"
        case ScalarKind.BOOL:
            return "bool"
        case _:
            return "float"


def write_image_type(
    dim: "ImageDimension",
    arrayed: bool,
    class: "ImageClass",
) -> str:
    """Write HLSL image type string.
    
    Args:
        dim: Image dimension
        arrayed: Whether the image is arrayed
        class: Image class
        
    Returns:
        HLSL image type string
    """
    access_str = "RW" if hasattr(class, 'format') else ""
    dim_str = dim.to_hlsl_str() if hasattr(dim, 'to_hlsl_str') else str(dim)
    arrayed_str = "Array" if arrayed else ""
    
    base_type = f"{access_str}Texture{dim_str}{arrayed_str}"
    
    if hasattr(class, 'multi'):
        multi_str = "MS" if class.multi else ""
        if hasattr(class, 'depth') and class.depth:
            return f"{multi_str}<float>"
        else:
            # Sampled class
            return f"{multi_str}<float4>"
    elif hasattr(class, 'format'):
        # Storage class
        return f"<{class.format.to_hlsl_str() if hasattr(class.format, 'to_hlsl_str') else class.format}>"
    
    return base_type


def write_convert_yuv_to_rgb_and_return(
    level: "Level",
    y: str,
    uv: str,
    params: str,
) -> str:
    """Generate YUV to RGB conversion code.
    
    Args:
        level: Indentation level
        y: Y component variable name
        uv: UV components variable name
        params: Parameters variable name
        
    Returns:
        Generated HLSL code
    """
    l1 = level
    l2 = level.next() if hasattr(level, 'next') else f"  {l1}"
    
    return f"""{l1}float3 srcGammaRgb = mul(float4({y}, {uv}, 1.0), {params}.yuv_conversion_matrix).rgb;
{l1}float3 srcLinearRgb = srcGammaRgb < {params}.src_tf.k * {params}.src_tf.b ?
{l2}srcGammaRgb / {params}.src_tf.k :
{l2}pow((srcGammaRgb + {params}.src_tf.a - 1.0) / {params}.src_tf.a, {params}.src_tf.g);
{l1}float3 dstLinearRgb = mul(srcLinearRgb, {params}.gamut_conversion_matrix);
{l1}float3 dstGammaRgb = dstLinearRgb < {params}.dst_tf.b ?
{l2}{params}.dst_tf.k * dstLinearRgb :
{l2}{params}.dst_tf.a * pow(dstLinearRgb, 1.0 / {params}.dst_tf.g) - ({params}.dst_tf.a - 1);
{l1}return float4(dstGammaRgb, 1.0);"""


IMAGE_STORAGE_LOAD_SCALAR_WRAPPER = "LoadedStorageValueFrom"


class ImageQueryType(Enum):
    """HLSL-specific image query types."""
    
    SIZE = "Size"
    SIZE_LEVEL = "SizeLevel"
    NUM_LEVELS = "NumLevels"
    NUM_LAYERS = "NumLayers"
    NUM_SAMPLES = "NumSamples"


def convert_image_query(query: "ImageQuery") -> ImageQueryType:
    """Convert IR ImageQuery to HLSL-specific type.
    
    Args:
        query: IR image query
        
    Returns:
        HLSL image query type
    """
    if hasattr(query, 'level') and query.level is not None:
        return ImageQueryType.SIZE_LEVEL
    elif hasattr(query, 'size'):
        return ImageQueryType.SIZE
    elif hasattr(query, 'num_levels'):
        return ImageQueryType.NUM_LEVELS
    elif hasattr(query, 'num_layers'):
        return ImageQueryType.NUM_LAYERS
    elif hasattr(query, 'num_samples'):
        return ImageQueryType.NUM_SAMPLES
    return ImageQueryType.SIZE


__all__ = [
    # Wrapper types
    "WrappedArrayLength",
    "WrappedImageLoad",
    "WrappedImageSample",
    "WrappedImageQuery",
    "WrappedConstructor",
    "WrappedStructMatrixAccess",
    "WrappedMatCx2",
    "WrappedMath",
    "WrappedZeroValue",
    "WrappedUnaryOp",
    "WrappedSaturate",
    "WrappedBinaryOp",
    "WrappedLoad",
    "WrappedImageGather",
    "WrappedRayQuery",
    "WrappedType",
    # Function names
    "ABS_FUNCTION",
    "DIV_FUNCTION",
    "EXTRACT_BITS_FUNCTION",
    "INSERT_BITS_FUNCTION",
    "F2I32_FUNCTION",
    "F2I64_FUNCTION",
    "F2U32_FUNCTION",
    "F2U64_FUNCTION",
    "IMAGE_LOAD_EXTERNAL_FUNCTION",
    "IMAGE_SAMPLE_BASE_CLAMP_TO_EDGE_FUNCTION",
    "MOD_FUNCTION",
    "NEG_FUNCTION",
    # Naming functions
    "hash_wrapped_type",
    "get_array_length_name",
    "get_image_load_name",
    "get_image_sample_name",
    "get_image_query_name",
    "get_constructor_name",
    "get_struct_matrix_access_name",
    "get_mat_cx2_name",
    "get_math_function_name",
    "get_zero_value_name",
    "get_unary_op_name",
    "get_saturate_name",
    "get_binary_op_name",
    "get_load_name",
    "get_image_gather_name",
    "get_ray_query_name",
    # Utility functions
    "is_signed",
    "type_to_hlsl_scalar",
    "write_image_type",
    "write_convert_yuv_to_rgb_and_return",
    "IMAGE_STORAGE_LOAD_SCALAR_WRAPPER",
    "ImageQueryType",
    "convert_image_query",
]


# Additional wrapper types that are missing
@dataclass(frozen=True)
class WrappedSaturate:
    """Wrapper for saturate operation."""
    scalar: Scalar


@dataclass(frozen=True)
class WrappedCast:
    """Wrapper for cast operations."""
    vector_size: Optional["VectorSize"]
    src_scalar: Scalar
    dst_scalar: Scalar


@dataclass(frozen=True)
class WrappedLoad:
    """Wrapper for load operations."""
    pointer: Handle["Expression"]


@dataclass(frozen=True)
class WrappedImageGather:
    """Wrapper for image gather operations."""
    dim: "ImageDimension"
    arrayed: bool
    class_: "ImageClass"


# Writer implementation helpers
class BackendError(Exception):
    """Backend error exception."""
    pass


class WrappedSet:
    """Set to track wrapped types that have been written."""
    
    def __init__(self) -> None:
        self._written: set[str] = set()
        self.sampler_heaps = False
        self.sampler_index_buffers: dict[str, str] = {}
        self.written_committed_intersection = False
        self.written_candidate_intersection = False
    
    def insert(self, wrapped_type: "WrappedType") -> bool:
        """Insert wrapped type and return True if it was not already present."""
        type_str = str(wrapped_type)
        if type_str in self._written:
            return False
        self._written.add(type_str)
        return True
    
    def __contains__(self, item: object) -> bool:
        """Check if item is in the set."""
        return str(item) in self._written


# Writer implementation functions
def write_image_load_function(
    class_: "ImageClass",
    module: "Module" = None
) -> str:
    """Write wrapped image load function.
    
    Args:
        class_: Image class
        module: Module for external texture params lookup
        
    Returns:
        HLSL function code
    """
    if hasattr(class_, 'external') and class_.external:
        # External texture handling
        return f"""float4 {IMAGE_LOAD_EXTERNAL_FUNCTION}(Texture2D<float4> plane0, Texture2D<float4> plane1, Texture2D<float4> plane2, {module.special_types.external_texture_params if module else 'ExternalTextureParams'} params, uint2 coords) {{
    uint2 plane0_size;
    plane0.GetDimensions(plane0_size.x, plane0_size.y);
    uint2 cropped_size = any(params.size) ? params.size : plane0_size;
    coords = min(coords, cropped_size - 1);
    float3x2 load_transform = float3x2(params.load_transform_0, params.load_transform_1, params.load_transform_2);
    uint2 plane0_coords = uint2(round(mul(float3(coords, 1.0), load_transform)));
    if (params.num_planes == 1u) {{
        return plane0.Load(uint3(plane0_coords, 0u));
    }} else {{
        uint2 plane1_size;
        plane1.GetDimensions(plane1_size.x, plane1_size.y);
        uint2 plane1_coords = uint2(floor(float2(plane0_coords) * float2(plane1_size) / float2(plane0_size)));
        float y = plane0.Load(uint3(plane0_coords, 0u)).x;
        float2 uv;
        if (params.num_planes == 2u) {{
            uv = plane1.Load(uint3(plane1_coords, 0u)).xy;
        }} else {{
            uint2 plane2_size;
            plane2.GetDimensions(plane2_size.x, plane2_size.y);
            uint2 plane2_coords = uint2(floor(float2(plane0_coords) * float2(plane2_size) / float2(plane0_size)));
            uv = float2(plane1.Load(uint3(plane1_coords, 0u)).x, plane2.Load(uint3(plane2_coords, 0u)).x);
        }}
        // YUV to RGB conversion would go here
        return float4(y, uv.x, uv.y, 1.0);
    }}
}}"""
    return "// Image load function for non-external textures"


def write_image_sample_function(
    class_: "ImageClass",
    clamp_to_edge: bool,
    module: "Module" = None
) -> str:
    """Write wrapped image sample function.
    
    Args:
        class_: Image class
        clamp_to_edge: Whether to clamp to edge
        module: Module for external texture params lookup
        
    Returns:
        HLSL function code
    """
    if hasattr(class_, 'external') and class_.external and clamp_to_edge:
        return f"""float4 {IMAGE_SAMPLE_BASE_CLAMP_TO_EDGE_FUNCTION}(Texture2D<float4> plane0, Texture2D<float4> plane1, Texture2D<float4> plane2, {module.special_types.external_texture_params if module else 'ExternalTextureParams'} params, SamplerState samp, float2 coords) {{
    float2 plane0_size;
    plane0.GetDimensions(plane0_size.x, plane0_size.y);
    float3x2 sample_transform = float3x2(params.sample_transform_0, params.sample_transform_1, params.sample_transform_2);
    coords = mul(float3(coords, 1.0), sample_transform);
    float2 bounds_min = mul(float3(0.0, 0.0, 1.0), sample_transform);
    float2 bounds_max = mul(float3(1.0, 1.0, 1.0), sample_transform);
    float4 bounds = float4(min(bounds_min, bounds_max), max(bounds_min, bounds_max));
    float2 plane0_half_texel = float2(0.5, 0.5) / plane0_size;
    float2 plane0_coords = clamp(coords, bounds.xy + plane0_half_texel, bounds.zw - plane0_half_texel);
    if (params.num_planes == 1u) {{
        return plane0.SampleLevel(samp, plane0_coords, 0.0f);
    }} else {{
        // Multi-plane sampling would go here
        return plane0.SampleLevel(samp, plane0_coords, 0.0f);
    }}
}}"""
    elif (hasattr(class_, 'sampled') and class_.sampled and 
          hasattr(class_.sampled, 'kind') and 
          class_.sampled.kind == ScalarKind.FLOAT and
          not (hasattr(class_, 'multi') and class_.multi) and clamp_to_edge):
        return f"""float4 {IMAGE_SAMPLE_BASE_CLAMP_TO_EDGE_FUNCTION}(Texture2D<float4> tex, SamplerState samp, float2 coords) {{
    float2 size;
    tex.GetDimensions(size.x, size.y);
    float2 half_texel = float2(0.5, 0.5) / size;
    return tex.SampleLevel(samp, clamp(coords, half_texel, 1.0 - half_texel), 0.0);
}}"""
    return "// Image sample function for other cases"


def write_image_query_function(
    dim: "ImageDimension",
    arrayed: bool,
    class_: "ImageClass",
    query: ImageQueryType,
    module: "Module" = None
) -> str:
    """Write wrapped image query function.
    
    Args:
        dim: Image dimension
        arrayed: Whether arrayed
        class_: Image class
        query: Query type
        module: Module for type information
        
    Returns:
        HLSL function code
    """
    func_name = write_image_query_function_name(dim, arrayed, class_, query)
    
    if hasattr(class_, 'external') and class_.external:
        if query != ImageQueryType.SIZE:
            raise ValueError("External images only support Size queries")
        params_ty = module.special_types.external_texture_params if module else "ExternalTextureParams"
        return f"""uint2 {func_name}(Texture2D<float4> plane0, Texture2D<float4> plane1, Texture2D<float4> plane2, {params_ty} params) {{
    if (any(params.size)) {{
        return params.size;
    }} else {{
        uint2 ret;
        plane0.GetDimensions(ret.x, ret.y);
        return ret;
    }}
}}"""
    
    # Generate standard query function
    ret_type = "uint" if query in [ImageQueryType.NUM_LEVELS, ImageQueryType.NUM_LAYERS, ImageQueryType.NUM_SAMPLES] else "uint"
    if dim.value == 1:
        ret_type = "uint"
    elif dim.value == 2:
        ret_type = "uint2"
    elif dim.value == 3:
        ret_type = "uint3"
    elif dim.value == 4:
        ret_type = "uint2"  # Cube maps return 2D coords
    
    params = [f"{write_image_type(dim, arrayed, class_)} tex"]
    if query == ImageQueryType.SIZE_LEVEL:
        params.append("uint mip_level")
    
    param_str = ", ".join(params)
    
    return f"""{ret_type} {func_name}({param_str}) {{
    uint4 ret;
    tex.GetDimensions(""" + 
    ("mip_level, " if query == ImageQueryType.SIZE_LEVEL else 
     ("0, " if not (hasattr(class_, 'multi') and class_.multi) and 
                    not (hasattr(class_, 'storage') and class_.storage) else "")) +
    """ret.x, ret.y, ret.z, ret.w);
    """ + 
    (f"return ret.{['x', 'xy', 'xyz', 'xy', 'w'][query.value if hasattr(query, 'value') else 0]};" if query != ImageQueryType.SIZE_LEVEL else "return ret.x;") +
    """
}}"""


def write_constructor_function(
    ty: Handle["Type"],
    module: "Module"
) -> str:
    """Write wrapped constructor function.
    
    Args:
        ty: Type handle
        module: Module for type information
        
    Returns:
        HLSL function code
    """
    type_info = module.types[int(ty)]
    type_name = f"Construct{ty.index}"
    
    if type_info.inner.type == "array":
        # Array constructor
        base = type_info.inner.array.base
        size = type_info.inner.array.size
        element_type = module.types[int(base)]
        
        return f"""typedef {element_type.name} ret_{type_name};
ret_{type_name} {type_name}({element_type.name} arg0) {{
    ret_{type_name} ret = {{ arg0 }};
    return ret;
}}"""
    elif type_info.inner.type == "struct":
        # Struct constructor
        members = type_info.inner.struct.members
        struct_name = type_info.name
        
        params = []
        for i, member in enumerate(members):
            member_type = module.types[int(member.ty)]
            params.append(f"{member_type.name} arg{i}")
        
        param_str = ", ".join(params)
        member_inits = []
        for i, member in enumerate(members):
            field_name = f"field_{i}"  # Simplified field names
            member_inits.append(f"{field_name} = arg{i}")
        
        return f"""{struct_name} {type_name}({param_str}) {{
    {struct_name} ret = ({struct_name})0;
    {chr(10).join([f"    ret.{init};" for init in member_inits])}
    return ret;
}}"""
    
    return f"// Constructor for type {type_info.name}"


def write_zero_value_function(
    ty: Handle["Type"],
    module: "Module"
) -> str:
    """Write wrapped zero value function.
    
    Args:
        ty: Type handle
        module: Module for type information
        
    Returns:
        HLSL function code
    """
    type_info = module.types[int(ty)]
    func_name = f"ZeroValue{ty.index}"
    
    if type_info.inner.type == "array":
        # Array zero value
        base = type_info.inner.array.base
        size = type_info.inner.array.size
        element_type = module.types[int(base)]
        
        return f"""typedef {element_type.name} ret_{func_name};
ret_{func_name} {func_name}() {{
    ret_{func_name} ret;
    // Initialize array elements to zero
    return ret;
}}"""
    else:
        # Simple type zero value
        return f"""{type_info.name} {func_name}() {{
    return ({type_info.name})0;
}}"""


def write_sampler_heaps() -> str:
    """Write sampler heap declarations.
    
    Returns:
        HLSL sampler heap code
    """
    return """SamplerState SamplerHeap[2048]: register(s0, space0);
SamplerComparisonState ComparisonSamplerHeap[2048]: register(s2048, space0);"""


def write_sampler_index_buffer(group: int) -> str:
    """Write sampler index buffer declaration.
    
    Args:
        group: Group number
        
    Returns:
        HLSL sampler buffer code
    """
    return f"""StructuredBuffer<uint> NagaGroup{group}SamplerIndexArray : register(t{group}, space0);"""


def write_texture_coordinates(
    kind: str,
    coordinate: "Handle[Expression]",
    array_index: Optional["Handle[Expression]"],
    mip_level: Optional["Handle[Expression]"],
    module: "Module",
    func_ctx: "FunctionCtx"
) -> str:
    """Write texture coordinate handling.
    
    Args:
        kind: Texture kind
        coordinate: Coordinate expression
        array_index: Array index expression
        mip_level: Mip level expression
        module: Module
        func_ctx: Function context
        
    Returns:
        Texture coordinate expression
    """
    # This would need expression writing capability
    # For now, return a placeholder
    return f"{kind}Coords(coordinate, array_index, mip_level)"


def less_than_two_true(bools: list[str]) -> str:
    """Generate boolean expression for "less than two true".
    
    Args:
        bools: List of boolean variable names
        
    Returns:
        HLSL boolean expression
    """
    if len(bools) <= 1:
        raise ValueError("Must have multiple booleans!")
    
    final_parts = []
    remaining = list(bools)
    while remaining:
        last_bool = remaining.pop()
        for other in remaining:
            final_parts.append(f"({last_bool} && {other})")
    
    return "||".join(final_parts)


# Constants for image storage
IMAGE_STORAGE_LOAD_SCALAR_WRAPPER = "LoadedStorageValueFrom"


# Special function generators
def write_modf_function(arg_type: str) -> str:
    """Write modf function for special types.
    
    Args:
        arg_type: Argument type
        
    Returns:
        HLSL function code
    """
    return f"""{arg_type} NagaModfFunction({arg_type} arg) {{
    {arg_type} other;
    {arg_type} result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}}"""


def write_frexp_function(arg_type: str) -> str:
    """Write frexp function for special types.
    
    Args:
        arg_type: Argument type
        
    Returns:
        HLSL function code
    """
    return f"""{arg_type} NagaFrexpFunction({arg_type} arg) {{
    {arg_type} other;
    {arg_type} result;
    result.fract = frexp(arg, other);
    result.exp_ = other;
    return result;
}}"""


# Cast functions
def write_cast_function(
    src_scalar: Scalar,
    dst_scalar: Scalar,
    components: Optional[int]
) -> str:
    """Write cast function with overflow protection.
    
    Args:
        src_scalar: Source scalar type
        dst_scalar: Destination scalar type
        components: Number of vector components
        
    Returns:
        HLSL function code
    """
    if components:
        src_ty = f"{type_to_hlsl_scalar(src_scalar.kind)}{components}"
        dst_ty = f"{type_to_hlsl_scalar(dst_scalar.kind)}{components}"
    else:
        src_ty = type_to_hlsl_scalar(src_scalar.kind)
        dst_ty = type_to_hlsl_scalar(dst_scalar.kind)
    
    func_name = {
        (ScalarKind.FLOAT, ScalarKind.SINT): F2I32_FUNCTION,
        (ScalarKind.FLOAT, ScalarKind.UINT): F2U32_FUNCTION,
    }.get((src_scalar.kind, dst_scalar.kind))
    
    if not func_name:
        return f"// Cast function from {src_ty} to {dst_ty}"
    
    return f"""{dst_ty} {func_name}({src_ty} value) {{
    return ({dst_ty})clamp(value, 0.0, 1.0);
}}"""


def write_loaded_scalar_to_storage_loaded_value(scalar_type: Scalar) -> str:
    """Write conversion from scalar to storage loaded value.
    
    Args:
        scalar_type: The scalar type
        
    Returns:
        HLSL function code
    """
    if scalar_type.kind == ScalarKind.SINT:
        assert scalar_type.width == 4
        zero, one = "0", "1"
    elif scalar_type.kind == ScalarKind.UINT:
        if scalar_type.width == 4:
            zero, one = "0u", "1u"
        elif scalar_type.width == 8:
            zero, one = "0uL", "1uL"
        else:
            raise ValueError(f"Unsupported scalar width: {scalar_type.width}")
    elif scalar_type.kind == ScalarKind.FLOAT:
        assert scalar_type.width == 4
        zero, one = "0.0", "1.0"
    else:
        raise ValueError(f"Unsupported scalar kind: {scalar_type.kind}")
    
    ty = type_to_hlsl_scalar(scalar_type.kind)
    return f"""{ty}4 {IMAGE_STORAGE_LOAD_SCALAR_WRAPPER}{ty}({ty} arg) {{
    {ty}4 ret = {ty}4({zero}, {zero}, {zero}, {one});
    return ret;
}}"""


def write_wrapped_array_length_function(writable: bool) -> str:
    """Write wrapped array length function.
    
    Args:
        writable: Whether the buffer is writable
        
    Returns:
        HLSL function code
    """
    access_str = "RW" if writable else ""
    return f"""uint NagaBufferLength{access_str}({access_str}ByteAddressBuffer buffer) {{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}}"""


def write_image_query_function_name(
    dim: "ImageDimension",
    arrayed: bool,
    class_: "ImageClass",
    query: ImageQueryType
) -> str:
    """Generate image query function name.
    
    Args:
        dim: Image dimension
        arrayed: Whether arrayed
        class_: Image class
        query: Query type
        
    Returns:
        Function name
    """
    dim_str = dim.to_hlsl_str() if hasattr(dim, 'to_hlsl_str') else str(dim)
    
    if hasattr(class_, 'multi') and class_.multi:
        class_str = "MS"
    elif hasattr(class_, 'depth') and class_.depth:
        class_str = "DepthMS" if hasattr(class_, 'multi') and class_.multi else "Depth"
    elif hasattr(class_, 'multi') and not class_.multi:
        class_str = ""
    elif hasattr(class_, 'format'):
        class_str = "RW"
    else:
        class_str = "External"
    
    arrayed_str = "Array" if arrayed else ""
    
    query_str = {
        ImageQueryType.SIZE: "Dimensions",
        ImageQueryType.SIZE_LEVEL: "MipDimensions", 
        ImageQueryType.NUM_LEVELS: "NumLevels",
        ImageQueryType.NUM_LAYERS: "NumLayers",
        ImageQueryType.NUM_SAMPLES: "NumSamples",
    }[query]
    
    return f"Naga{class_str}{query_str}{dim_str}{arrayed_str}"


def write_special_functions() -> str:
    """Write special type functions.
    
    Returns:
        HLSL function code
    """
    # This would need module information to generate correctly
    # For now, return a placeholder
    return """// Special functions would be generated here based on module content
"""


def get_components() -> list[str]:
    """Get component names for vector access.
    
    Returns:
        List of component names
    """
    return ["x", "y", "z", "w"]


# Matrix and struct helpers
def write_struct_matrix_access_functions(
    struct_name: str,
    field_name: str,
    matrix_type: str,
    columns: int
) -> str:
    """Write get/set functions for struct matrix access.
    
    Args:
        struct_name: Name of the struct
        field_name: Name of the matrix field
        matrix_type: Type of matrix elements
        columns: Number of matrix columns
        
    Returns:
        HLSL function code
    """
    get_func = f"""{matrix_type} GetMat{field_name}On{struct_name}({struct_name} obj) {{
    return {matrix_type}(obj.{field_name}_0, obj.{field_name}_1);
}}"""
    
    set_func = f"""void SetMat{field_name}On{struct_name}({struct_name} obj, {matrix_type} mat) {{
    obj.{field_name}_0 = mat[0];
    obj.{field_name}_1 = mat[1];
}}"""
    
    set_vec_func = f"""void SetMatVec{field_name}On{struct_name}({struct_name} obj, float2 vec, uint mat_idx) {{
    switch(mat_idx) {{
        case 0: obj.{field_name}_0 = vec; break;
        case 1: obj.{field_name}_1 = vec; break;
    }}
}}"""
    
    set_scalar_func = f"""void SetMatScalar{field_name}On{struct_name}({struct_name} obj, float scalar, uint mat_idx, uint vec_idx) {{
    switch(mat_idx) {{
        case 0: obj.{field_name}_0[vec_idx] = scalar; break;
        case 1: obj.{field_name}_1[vec_idx] = scalar; break;
    }}
}}"""
    
    return f"""{get_func}

{set_func}

{set_vec_func}

{set_scalar_func}
"""


def write_mat_cx2_typedef_and_functions(columns: int) -> str:
    """Write typedef and helper functions for Cx2 matrices.
    
    Args:
        columns: Number of matrix columns
        
    Returns:
        HLSL typedef and function code
    """
    typedef = f"typedef struct {{ float2 _{0}; " + "".join([f"float2 _{i}; " for i in range(1, columns)]) + f"}} __mat{columns}x2;"
    
    get_col_func = f"""float2 __get_col_of_mat{columns}x2(__mat{columns}x2 mat, uint idx) {{
    switch(idx) {{
        {"".join([f'case {i}: return mat._{i}; ' for i in range(columns)])}
        default: return (float2)0;
    }}
}}"""
    
    set_col_func = f"""void __set_col_of_mat{columns}x2(__mat{columns}x2 mat, uint idx, float2 value) {{
    switch(idx) {{
        {"".join([f'case {i}: mat._{i} = value; break; ' for i in range(columns)])}
    }}
}}"""
    
    set_el_func = f"""void __set_el_of_mat{columns}x2(__mat{columns}x2 mat, uint idx, uint vec_idx, float value) {{
    switch(idx) {{
        {"".join([f'case {i}: mat._{i}[vec_idx] = value; break; ' for i in range(columns)])}
    }}
}}"""
    
    return f"""{typedef}

{get_col_func}

{set_col_func}

{set_el_func}
"""


# Math function wrappers
def write_extract_bits_function(scalar_width: int, components: Optional[int] = None) -> str:
    """Write extract bits polyfill function.
    
    Args:
        scalar_width: Width of scalar type in bytes
        components: Number of vector components (None for scalar)
        
    Returns:
        HLSL function code
    """
    width = scalar_width * 8
    if components:
        ty = f"{type_to_hlsl_scalar(ScalarKind.UINT)}{components}"
        return f"""{ty} NagaExtractBits{components}({ty} e, uint offset, uint count) {{
    uint w = {width};
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}}"""
    else:
        return f"""uint NagaExtractBits(uint e, uint offset, uint count) {{
    uint w = {width};
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}}"""


def write_insert_bits_function(scalar_width: int, components: Optional[int] = None) -> str:
    """Write insert bits polyfill function.
    
    Args:
        scalar_width: Width of scalar type in bytes
        components: Number of vector components (None for scalar)
        
    Returns:
        HLSL function code
    """
    width = scalar_width * 8
    max_val = {1: "0xFF", 2: "0xFFFF", 4: "0xFFFFFFFF", 8: "0xFFFFFFFFFFFFFFFF"}[scalar_width]
    
    if components:
        ty = f"{type_to_hlsl_scalar(ScalarKind.UINT)}{components}"
        return f"""{ty} NagaInsertBits{components}({ty} e, {ty} newbits, uint offset, uint count) {{
    uint w = {width}u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = (({max_val}u >> ({width}u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}}"""
    else:
        return f"""uint NagaInsertBits(uint e, uint newbits, uint offset, uint count) {{
    uint w = {width}u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = (({max_val}u >> ({width}u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}}"""


def write_abs_function(scalar_type: str, components: Optional[int] = None) -> str:
    """Write abs function for signed integers.
    
    Args:
        scalar_type: HLSL scalar type
        components: Number of vector components (None for scalar)
        
    Returns:
        HLSL function code
    """
    if components:
        ty = f"{scalar_type}{components}"
        return f"""{ty} NagaAbs{components}({ty} val) {{
    return val >= 0 ? val : asint(-asuint(val));
}}"""
    else:
        return f"""{scalar_type} NagaAbs({scalar_type} val) {{
    return val >= 0 ? val : asint(-asuint(val));
}}"""


def write_neg_function(scalar_type: str, components: Optional[int] = None) -> str:
    """Write negation function for signed integers.
    
    Args:
        scalar_type: HLSL scalar type
        components: Number of vector components (None for scalar)
        
    Returns:
        HLSL function code
    """
    if components:
        ty = f"{scalar_type}{components}"
        return f"""{ty} NagaNeg{components}({ty} val) {{
    return asint(-asuint(val));
}}"""
    else:
        return f"""{scalar_type} NagaNeg({scalar_type} val) {{
    return asint(-asuint(val));
}}"""


def write_div_function(scalar_type: str, components: Optional[int] = None) -> str:
    """Write division function with overflow protection.
    
    Args:
        scalar_type: HLSL scalar type
        components: Number of vector components (None for scalar)
        
    Returns:
        HLSL function code
    """
    if components:
        ty = f"{scalar_type}{components}"
        if "u" in scalar_type.lower():
            return f"""{ty} NagaDiv{components}({ty} lhs, {ty} rhs) {{
    return lhs / (rhs == 0u ? 1u : rhs);
}}"""
        else:
            return f"""{ty} NagaDiv{components}({ty} lhs, {ty} rhs) {{
    return lhs / (((lhs == {scalar_type}(-2147483648) && rhs == -1) | (rhs == 0)) ? 1 : rhs);
}}"""
    else:
        if "u" in scalar_type.lower():
            return f"""{scalar_type} NagaDiv({scalar_type} lhs, {scalar_type} rhs) {{
    return lhs / (rhs == 0u ? 1u : rhs);
}}"""
        else:
            return f"""{scalar_type} NagaDiv({scalar_type} lhs, {scalar_type} rhs) {{
    return lhs / (((lhs == {scalar_type}(-2147483648) && rhs == -1) | (rhs == 0)) ? 1 : rhs);
}}"""


def write_mod_function(scalar_type: str, components: Optional[int] = None) -> str:
    """Write modulo function with overflow protection.
    
    Args:
        scalar_type: HLSL scalar type
        components: Number of vector components (None for scalar)
        
    Returns:
        HLSL function code
    """
    if components:
        ty = f"{scalar_type}{components}"
        if "u" in scalar_type.lower():
            return f"""{ty} NagaMod{components}({ty} lhs, {ty} rhs) {{
    return lhs % (rhs == 0u ? 1u : rhs);
}}"""
        elif scalar_type.lower() == "float":
            return f"""{ty} NagaMod{components}({ty} lhs, {ty} rhs) {{
    return lhs - rhs * trunc(lhs / rhs);
}}"""
        else:
            return f"""{ty} NagaMod{components}({ty} lhs, {ty} rhs) {{
    {scalar_type} divisor = ((lhs == {scalar_type}(-2147483648) && rhs == -1) | (rhs == 0)) ? 1 : rhs;
    return lhs - (lhs / divisor) * divisor;
}}"""
    else:
        if "u" in scalar_type.lower():
            return f"""{scalar_type} NagaMod({scalar_type} lhs, {scalar_type} rhs) {{
    return lhs % (rhs == 0u ? 1u : rhs);
}}"""
        elif scalar_type.lower() == "float":
            return f"""{scalar_type} NagaMod({scalar_type} lhs, {scalar_type} rhs) {{
    return lhs - rhs * trunc(lhs / rhs);
}}"""
        else:
            return f"""{scalar_type} NagaMod({scalar_type} lhs, {scalar_type} rhs) {{
    {scalar_type} divisor = ((lhs == {scalar_type}(-2147483648) && rhs == -1) | (rhs == 0)) ? 1 : rhs;
    return lhs - (lhs / divisor) * divisor;
}}"""
