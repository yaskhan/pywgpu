"""
Conversion functions for HLSL backend.

Contains simple 1:1 conversion functions for scalar types, built-ins,
storage formats, interpolation, sampling, and atomic operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...error import ShaderError

if TYPE_CHECKING:
    from ...ir.module import Module


class HLSLTypeString:
    """Structure for HLSL type string information."""

    def __init__(self, type_str: str):
        self.type_str = type_str


def hlsl_scalar(scalar_kind: str, width: int) -> str:
    """Helper function that returns the HLSL scalar type string.

    Args:
        scalar_kind: The scalar kind (Sint, Uint, Float, Bool)
        width: The width in bytes (2, 4, or 8)

    Returns:
        The HLSL type string

    Raises:
        ShaderError: If width is unsupported for the scalar kind
    """
    from ...ir.type import ScalarKind

    if scalar_kind == ScalarKind.SINT:
        if width == 4:
            return "int"
        elif width == 8:
            return "int64_t"
        else:
            raise ShaderError(f"Unsupported Sint width: {width}")
    elif scalar_kind == ScalarKind.UINT:
        if width == 4:
            return "uint"
        elif width == 8:
            return "uint64_t"
        else:
            raise ShaderError(f"Unsupported Uint width: {width}")
    elif scalar_kind == ScalarKind.FLOAT:
        if width == 2:
            return "half"
        elif width == 4:
            return "float"
        elif width == 8:
            return "double"
        else:
            raise ShaderError(f"Unsupported Float width: {width}")
    elif scalar_kind == ScalarKind.BOOL:
        return "bool"
    elif scalar_kind in (ScalarKind.ABSTRACT_INT, ScalarKind.ABSTRACT_FLOAT):
        raise ShaderError(f"Abstract types not supported in HLSL: {scalar_kind}")
    else:
        raise ShaderError(f"Unsupported scalar kind: {scalar_kind}")


def hlsl_cast(scalar_kind: str) -> str:
    """Return the HLSL cast function for a scalar kind.

    Args:
        scalar_kind: The scalar kind

    Returns:
        The HLSL cast function name
    """
    from ...ir.type import ScalarKind

    if scalar_kind == ScalarKind.FLOAT:
        return "asfloat"
    elif scalar_kind == ScalarKind.SINT:
        return "asint"
    elif scalar_kind == ScalarKind.UINT:
        return "asuint"
    else:
        raise ShaderError(f"Cannot cast from scalar kind: {scalar_kind}")


def hlsl_built_in(built_in: str) -> str:
    """Helper function that returns the HLSL semantic for a builtin.

    Args:
        built_in: The built-in identifier

    Returns:
        The HLSL semantic string

    Raises:
        ShaderError: If the built-in is not supported in HLSL
    """
    from ...ir import BuiltIn

    match built_in:
        # Position
        case BuiltIn.POSITION:
            return "SV_Position"

        # Vertex built-ins
        case BuiltIn.CLIP_DISTANCE:
            return "SV_ClipDistance"
        case BuiltIn.CULL_DISTANCE:
            return "SV_CullDistance"
        case BuiltIn.INSTANCE_INDEX:
            return "SV_InstanceID"
        case BuiltIn.VERTEX_INDEX:
            return "SV_VertexID"

        # Fragment built-ins
        case BuiltIn.FRAG_DEPTH:
            return "SV_Depth"
        case BuiltIn.FRONT_FACING:
            return "SV_IsFrontFace"
        case BuiltIn.PRIMITIVE_INDEX:
            return "SV_PrimitiveID"
        case BuiltIn.SAMPLE_INDEX:
            return "SV_SampleIndex"
        case BuiltIn.SAMPLE_MASK:
            return "SV_Coverage"

        # Compute built-ins
        case BuiltIn.GLOBAL_INVOCATION_ID:
            return "SV_DispatchThreadID"
        case BuiltIn.LOCAL_INVOCATION_ID:
            return "SV_GroupThreadID"
        case BuiltIn.LOCAL_INVOCATION_INDEX:
            return "SV_GroupIndex"
        case BuiltIn.WORK_GROUP_ID:
            return "SV_GroupID"
        case BuiltIn.NUM_WORK_GROUPS:
            return "SV_GroupID"  # Semantic doesn't matter, replaced in writer
        case BuiltIn.VIEW_INDEX:
            return "SV_ViewID"

        # Mesh built-ins
        case BuiltIn.CULL_PRIMITIVE:
            return "SV_CullPrimitive"

        # Unsupported built-ins
        case BuiltIn.POINT_SIZE:
            raise ShaderError("PointSize not supported in HLSL")
        case BuiltIn.POINT_COORD:
            raise ShaderError("PointCoord not supported in HLSL")
        case BuiltIn.DRAW_ID:
            raise ShaderError("DrawID not supported in HLSL")
        case BuiltIn.BASE_INSTANCE:
            raise ShaderError("BaseInstance not supported in HLSL")
        case BuiltIn.BASE_VERTEX:
            raise ShaderError("BaseVertex not supported in HLSL")
        case BuiltIn.WORK_GROUP_SIZE:
            raise ShaderError("WorkGroupSize not supported in HLSL")
        case BuiltIn.SUBGROUP_SIZE | BuiltIn.SUBGROUP_INVOCATION_ID | BuiltIn.NUM_SUBGROUPS | BuiltIn.SUBGROUP_ID:
            raise ShaderError(f"Subgroup operations not directly supported: {built_in}")
        case BuiltIn.POINT_INDEX | BuiltIn.LINE_INDICES | BuiltIn.TRIANGLE_INDICES:
            raise ShaderError(f"Mesh builtin not implemented: {built_in}")

        case _:
            raise ShaderError(f"Unknown built-in: {built_in}")


def hlsl_storage_format(format: str) -> str:
    """Helper function that returns the HLSL storage format string.

    Args:
        format: The storage format

    Returns:
        The HLSL format string
    """
    from ...ir import StorageFormat

    match format:
        case StorageFormat.R16FLOAT | StorageFormat.R32FLOAT:
            return "float"
        case StorageFormat.R8UNORM | StorageFormat.R16UNORM:
            return "unorm float"
        case StorageFormat.R8SNORM | StorageFormat.R16SNORM:
            return "snorm float"
        case StorageFormat.R8UINT | StorageFormat.R16UINT | StorageFormat.R32UINT:
            return "uint"
        case StorageFormat.R8SINT | StorageFormat.R16SINT | StorageFormat.R32SINT:
            return "int"
        case StorageFormat.R64UINT:
            return "uint64_t"

        case StorageFormat.RG16FLOAT | StorageFormat.RG32FLOAT:
            return "float4"
        case StorageFormat.RG8UNORM | StorageFormat.RG16UNORM:
            return "unorm float4"
        case StorageFormat.RG8SNORM | StorageFormat.RG16SNORM:
            return "snorm float4"
        case StorageFormat.RG8SINT | StorageFormat.RG16SINT | StorageFormat.RG32UINT:
            return "int4"
        case StorageFormat.RG8UINT | StorageFormat.RG16UINT | StorageFormat.RG32SINT:
            return "uint4"

        case StorageFormat.RG11B10UFLOAT:
            return "float4"

        case StorageFormat.RGBA16FLOAT | StorageFormat.RGBA32FLOAT:
            return "float4"
        case StorageFormat.RGBA8UNORM | StorageFormat.BGRA8UNORM | StorageFormat.RGBA16UNORM | StorageFormat.RGB10A2UNORM:
            return "unorm float4"
        case StorageFormat.RGBA8SNORM | StorageFormat.RGBA16SNORM:
            return "snorm float4"
        case StorageFormat.RGBA8UINT | StorageFormat.RGBA16UINT | StorageFormat.RGBA32UINT | StorageFormat.RGB10A2UINT:
            return "uint4"
        case StorageFormat.RGBA8SINT | StorageFormat.RGBA16SINT | StorageFormat.RGBA32SINT:
            return "int4"

        case _:
            raise ShaderError(f"Unknown storage format: {format}")


def hlsl_interpolation(interpolation: str) -> Optional[str]:
    """Helper function that returns the HLSL interpolation qualifier string.

    Args:
        interpolation: The interpolation type

    Returns:
        The HLSL qualifier string, or None if no qualifier needed
    """
    from ...ir import Interpolation

    match interpolation:
        case Interpolation.PERSPECTIVE:
            return None  # Linear is default in SM4+
        case Interpolation.LINEAR:
            return "noperspective"
        case Interpolation.FLAT:
            return "nointerpolation"
        case Interpolation.PER_VERTEX:
            raise ShaderError("PerVertex interpolation not valid here")
        case _:
            raise ShaderError(f"Unknown interpolation: {interpolation}")


def hlsl_sampling(sampling: str) -> Optional[str]:
    """Helper function that returns the HLSL auxiliary qualifier for sampling.

    Args:
        sampling: The sampling type

    Returns:
        The HLSL qualifier string, or None
    """
    from ...ir import Sampling

    match sampling:
        case Sampling.CENTER | Sampling.FIRST | Sampling.EITHER:
            return None
        case Sampling.CENTROID:
            return "centroid"
        case Sampling.SAMPLE:
            return "sample"
        case _:
            raise ShaderError(f"Unknown sampling: {sampling}")


def hlsl_atomic_suffix(atomic_fun: str) -> str:
    """Return the HLSL suffix for the InterlockedXxx method.

    Args:
        atomic_fun: The atomic function

    Returns:
        The HLSL method suffix
    """
    from ...ir import AtomicFunction

    match atomic_fun:
        case AtomicFunction.ADD | AtomicFunction.SUBTRACT:
            return "Add"
        case AtomicFunction.AND:
            return "And"
        case AtomicFunction.INCLUSIVE_OR:
            return "Or"
        case AtomicFunction.EXCLUSIVE_OR:
            return "Xor"
        case AtomicFunction.MIN:
            return "Min"
        case AtomicFunction.MAX:
            return "Max"
        case AtomicFunction.EXCHANGE:
            return "Exchange"
        case _:
            raise ShaderError(f"Unknown atomic function: {atomic_fun}")


__all__ = [
    "HLSLTypeString",
    "hlsl_scalar",
    "hlsl_cast",
    "hlsl_built_in",
    "hlsl_storage_format",
    "hlsl_interpolation",
    "hlsl_sampling",
    "hlsl_atomic_suffix",
]
