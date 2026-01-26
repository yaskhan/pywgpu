"""
Conversion functions for GLSL backend.

Contains simple 1:1 conversion functions for scalar types, built-ins,
storage qualifiers, interpolation, sampling, dimensions, and storage formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...error import ShaderError

if TYPE_CHECKING:
    from ...ir.module import Module


class ScalarString:
    """Structure returned by glsl_scalar function.

    Contains both a prefix used in other types and the full type name.
    """

    def __init__(self, prefix: str, full: str):
        self.prefix = prefix
        self.full = full


def glsl_scalar(scalar_kind: str, width: int) -> ScalarString:
    """Helper function that returns scalar related strings.

    Args:
        scalar_kind: The scalar kind (Sint, Uint, Float, Bool)
        width: The width in bytes (4 or 8)

    Returns:
        ScalarString with prefix and full type name

    Raises:
        ShaderError: If float width is not 4 or 8, or for abstract types
    """
    from ...ir.type import ScalarKind

    if scalar_kind == ScalarKind.SINT:
        return ScalarString(prefix="i", full="int")
    elif scalar_kind == ScalarKind.UINT:
        return ScalarString(prefix="u", full="uint")
    elif scalar_kind == ScalarKind.FLOAT:
        if width == 4:
            return ScalarString(prefix="", full="float")
        elif width == 8:
            return ScalarString(prefix="d", full="double")
        else:
            raise ShaderError(f"Unsupported float width: {width}")
    elif scalar_kind == ScalarKind.BOOL:
        return ScalarString(prefix="b", full="bool")
    elif scalar_kind in (ScalarKind.ABSTRACT_INT, ScalarKind.ABSTRACT_FLOAT):
        raise ShaderError(f"Abstract types not supported in GLSL: {scalar_kind}")
    else:
        raise ShaderError(f"Unsupported scalar kind: {scalar_kind}")


def glsl_built_in(built_in: str, output: bool, targeting_webgl: bool = False, draw_parameters: bool = False) -> str:
    """Helper function that returns the GLSL variable name for a builtin.

    Args:
        built_in: The built-in identifier
        output: True if outputting, False if inputting
        targeting_webgl: Whether targeting WebGL
        draw_parameters: Whether draw parameters are supported

    Returns:
        The GLSL built-in variable name
    """
    from ...ir import BuiltIn

    match built_in:
        case BuiltIn.POSITION:
            return "gl_Position" if output else "gl_FragCoord"

        case BuiltIn.VIEW_INDEX:
            if targeting_webgl:
                return "gl_ViewID_OVR"
            else:
                return "uint(gl_ViewIndex)"

        # vertex built-ins
        case BuiltIn.BASE_INSTANCE:
            return "uint(gl_BaseInstance)"
        case BuiltIn.BASE_VERTEX:
            return "uint(gl_BaseVertex)"
        case BuiltIn.CLIP_DISTANCE:
            return "gl_ClipDistance"
        case BuiltIn.CULL_DISTANCE:
            return "gl_CullDistance"
        case BuiltIn.INSTANCE_INDEX:
            if draw_parameters:
                return "(uint(gl_InstanceID) + uint(gl_BaseInstanceARB))"
            else:
                # Must match FIRST_INSTANCE_BINDING constant
                return "(uint(gl_InstanceID) + naga_vs_first_instance)"
        case BuiltIn.POINT_SIZE:
            return "gl_PointSize"
        case BuiltIn.VERTEX_INDEX:
            return "uint(gl_VertexID)"
        case BuiltIn.DRAW_ID:
            return "gl_DrawID"

        # fragment built-ins
        case BuiltIn.FRAG_DEPTH:
            return "gl_FragDepth"
        case BuiltIn.POINT_COORD:
            return "gl_PointCoord"
        case BuiltIn.FRONT_FACING:
            return "gl_FrontFacing"
        case BuiltIn.PRIMITIVE_INDEX:
            return "uint(gl_PrimitiveID)"
        case BuiltIn.SAMPLE_INDEX:
            return "gl_SampleID"
        case BuiltIn.SAMPLE_MASK:
            return "gl_SampleMask" if output else "gl_SampleMaskIn"

        # compute built-ins
        case BuiltIn.GLOBAL_INVOCATION_ID:
            return "gl_GlobalInvocationID"
        case BuiltIn.LOCAL_INVOCATION_ID:
            return "gl_LocalInvocationID"
        case BuiltIn.LOCAL_INVOCATION_INDEX:
            return "gl_LocalInvocationIndex"
        case BuiltIn.WORK_GROUP_ID:
            return "gl_WorkGroupID"
        case BuiltIn.WORK_GROUP_SIZE:
            return "gl_WorkGroupSize"
        case BuiltIn.NUM_WORK_GROUPS:
            return "gl_NumWorkGroups"

        # subgroup built-ins
        case BuiltIn.NUM_SUBGROUPS:
            return "gl_NumSubgroups"
        case BuiltIn.SUBGROUP_ID:
            return "gl_SubgroupID"
        case BuiltIn.SUBGROUP_SIZE:
            return "gl_SubgroupSize"
        case BuiltIn.SUBGROUP_INVOCATION_ID:
            return "gl_SubgroupInvocationID"

        # mesh built-ins (not directly supported in GLSL)
        case BuiltIn.CULL_PRIMITIVE | BuiltIn.POINT_INDEX | BuiltIn.LINE_INDICES | BuiltIn.TRIANGLE_INDICES | BuiltIn.MESH_TASK_SIZE | BuiltIn.VERTEX_COUNT | BuiltIn.PRIMITIVE_COUNT | BuiltIn.VERTICES | BuiltIn.PRIMITIVES:
            raise ShaderError(f"Built-in not supported in GLSL: {built_in}")

        case _:
            raise ShaderError(f"Unknown built-in: {built_in}")


def glsl_storage_qualifier(space: str) -> Optional[str]:
    """Helper function that returns the GLSL storage qualifier string.

    Args:
        space: The address space

    Returns:
        The GLSL qualifier string, or None if no qualifier needed
    """
    from ...ir import AddressSpace

    match space:
        case AddressSpace.FUNCTION | AddressSpace.PRIVATE:
            return None
        case AddressSpace.STORAGE:
            return "buffer"
        case AddressSpace.UNIFORM:
            return "uniform"
        case AddressSpace.HANDLE:
            return "uniform"
        case AddressSpace.WORK_GROUP:
            return "shared"
        case AddressSpace.IMMEDIATE:
            return "uniform"
        case AddressSpace.TASK_PAYLOAD:
            raise ShaderError("TaskPayload address space not supported in GLSL")
        case _:
            raise ShaderError(f"Unknown address space: {space}")


def glsl_interpolation(interpolation: str) -> str:
    """Helper function that returns the GLSL interpolation qualifier string.

    Args:
        interpolation: The interpolation type

    Returns:
        The GLSL qualifier string
    """
    from ...ir import Interpolation

    match interpolation:
        case Interpolation.PERSPECTIVE:
            return "smooth"
        case Interpolation.LINEAR:
            return "noperspective"
        case Interpolation.FLAT:
            return "flat"
        case Interpolation.PER_VERTEX:
            raise ShaderError("PerVertex interpolation not valid here")
        case _:
            raise ShaderError(f"Unknown interpolation: {interpolation}")


def glsl_sampling(sampling: str) -> Optional[str]:
    """Helper function that returns the GLSL auxiliary qualifier for sampling.

    Args:
        sampling: The sampling type

    Returns:
        The GLSL qualifier string, or None

    Raises:
        ShaderError: If First sampling is requested
    """
    from ...ir import Sampling

    match sampling:
        case Sampling.FIRST:
            raise ShaderError("First sampling not supported in GLSL")
        case Sampling.CENTER | Sampling.EITHER:
            return None
        case Sampling.CENTROID:
            return "centroid"
        case Sampling.SAMPLE:
            return "sample"
        case _:
            raise ShaderError(f"Unknown sampling: {sampling}")


def glsl_dimension(dim: str) -> str:
    """Helper function that returns the GLSL dimension string.

    Args:
        dim: The image dimension

    Returns:
        The GLSL dimension string
    """
    from ...ir import ImageDimension

    match dim:
        case ImageDimension.D1:
            return "1D"
        case ImageDimension.D2:
            return "2D"
        case ImageDimension.D3:
            return "3D"
        case ImageDimension.CUBE:
            return "Cube"
        case _:
            raise ShaderError(f"Unknown dimension: {dim}")


def glsl_storage_format(format: str) -> str:
    """Helper function that returns the GLSL storage format string.

    Args:
        format: The storage format

    Returns:
        The GLSL format string

    Raises:
        ShaderError: If format is not supported
    """
    from ...ir import StorageFormat

    match format:
        case StorageFormat.R8UNORM:
            return "r8"
        case StorageFormat.R8SNORM:
            return "r8_snorm"
        case StorageFormat.R8UINT:
            return "r8ui"
        case StorageFormat.R8SINT:
            return "r8i"
        case StorageFormat.R16UINT:
            return "r16ui"
        case StorageFormat.R16SINT:
            return "r16i"
        case StorageFormat.R16FLOAT:
            return "r16f"
        case StorageFormat.RG8UNORM:
            return "rg8"
        case StorageFormat.RG8SNORM:
            return "rg8_snorm"
        case StorageFormat.RG8UINT:
            return "rg8ui"
        case StorageFormat.RG8SINT:
            return "rg8i"
        case StorageFormat.R32UINT:
            return "r32ui"
        case StorageFormat.R32SINT:
            return "r32i"
        case StorageFormat.R32FLOAT:
            return "r32f"
        case StorageFormat.RG16UINT:
            return "rg16ui"
        case StorageFormat.RG16SINT:
            return "rg16i"
        case StorageFormat.RG16FLOAT:
            return "rg16f"
        case StorageFormat.RGBA8UNORM:
            return "rgba8"
        case StorageFormat.RGBA8SNORM:
            return "rgba8_snorm"
        case StorageFormat.RGBA8UINT:
            return "rgba8ui"
        case StorageFormat.RGBA8SINT:
            return "rgba8i"
        case StorageFormat.RGB10A2UINT:
            return "rgb10_a2ui"
        case StorageFormat.RGB10A2UNORM:
            return "rgb10_a2"
        case StorageFormat.RG11B10UFLOAT:
            return "r11f_g11f_b10f"
        case StorageFormat.R64UINT:
            return "r64ui"
        case StorageFormat.RG32UINT:
            return "rg32ui"
        case StorageFormat.RG32SINT:
            return "rg32i"
        case StorageFormat.RG32FLOAT:
            return "rg32f"
        case StorageFormat.RGBA16UINT:
            return "rgba16ui"
        case StorageFormat.RGBA16SINT:
            return "rgba16i"
        case StorageFormat.RGBA16FLOAT:
            return "rgba16f"
        case StorageFormat.RGBA32UINT:
            return "rgba32ui"
        case StorageFormat.RGBA32SINT:
            return "rgba32i"
        case StorageFormat.RGBA32FLOAT:
            return "rgba32f"
        case StorageFormat.R16UNORM:
            return "r16"
        case StorageFormat.R16SNORM:
            return "r16_snorm"
        case StorageFormat.RG16UNORM:
            return "rg16"
        case StorageFormat.RG16SNORM:
            return "rg16_snorm"
        case StorageFormat.RGBA16UNORM:
            return "rgba16"
        case StorageFormat.RGBA16SNORM:
            return "rgba16_snorm"
        case StorageFormat.BGRA8UNORM:
            raise ShaderError("BGRA8 format not supported in GLSL")
        case _:
            raise ShaderError(f"Unknown storage format: {format}")


__all__ = [
    "ScalarString",
    "glsl_scalar",
    "glsl_built_in",
    "glsl_storage_qualifier",
    "glsl_interpolation",
    "glsl_sampling",
    "glsl_dimension",
    "glsl_storage_format",
]
