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
]
