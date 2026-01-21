"""
Backend module for shader code generation.

This module contains writers for different shader backends:
- WGSL (WebGPU Shading Language)
- GLSL (OpenGL Shading Language)
- HLSL (High-Level Shading Language)
- MSL (Metal Shading Language)
- SPIR-V (Standard Portable Intermediate Representation)
"""

from typing import Any, Tuple, Optional
from enum import Enum
from .. import Arena, Handle, Expression, BinaryOperator, TypeInner, Statement, Binding, BuiltIn
from .. import ShaderStage as ShaderStageEnum

# Names of vector components
COMPONENTS: Tuple[str, ...] = ('x', 'y', 'z', 'w')

# Indent for backends
INDENT: str = "    "

# Expressions that need baking
NeedBakeExpressions = set  # FastHashSet[Handle[Expression]]


class Baked:
    """
    A type for displaying expression handles as baking identifiers.

    Given an Expression Handle h, Baked(h) implements __str__, showing the
    handle's index prefixed by _e.
    """

    def __init__(self, handle: Handle[Expression]):
        self.handle = handle

    def __str__(self) -> str:
        return f"_e{self.handle.index}"


class RayQueryPoint(Enum):
    """
    How far through a ray query are we.
    """
    INITIALIZED = 1 << 0
    PROCEED = 1 << 1
    FINISHED_TRAVERSAL = 1 << 2


# Pipeline constants type
PipelineConstants = dict  # hashbrown::HashMap<String, f64>


class Level:
    """
    Indentation level.
    """

    def __init__(self, level: int):
        self.level = level

    def next(self) -> "Level":
        return Level(self.level + 1)

    def __str__(self) -> str:
        return INDENT * self.level


class FunctionType(Enum):
    """
    Whether we're generating an entry point or a regular function.
    """
    FUNCTION = "function"
    ENTRY_POINT = "entry_point"

    def is_compute_like_entry_point(self, module: Any) -> bool:
        """
        Returns true if the function is an entry point for a compute-like shader.
        """
        # This would need access to the module's entry points
        # For now, return False
        return False


class FunctionCtx:
    """
    Helper structure that stores data needed when writing the function.
    """

    def __init__(
        self,
        ty: FunctionType,
        info: Any,
        expressions: Arena[Expression],
        named_expressions: dict,
    ):
        self.ty = ty
        self.info = info
        self.expressions = expressions
        self.named_expressions = named_expressions

    def resolve_type(self, handle: Handle[Expression], types: Any) -> TypeInner:
        """
        Helper method that resolves a type of a given expression.
        """
        # This would need access to the type information
        # For now, return a placeholder
        return TypeInner()

    def name_key(self, local: Any) -> Any:
        """
        Helper method that generates a NameKey for a local in the current function.
        """
        # This would need access to the proc module
        # For now, return a placeholder
        return None

    def argument_key(self, arg: int) -> Any:
        """
        Helper method that generates a NameKey for a function argument.
        """
        # This would need access to the proc module
        # For now, return a placeholder
        return None

    def external_texture_argument_key(self, arg: int, external_texture_key: Any) -> Any:
        """
        Helper method that generates a NameKey for an external texture function argument.
        """
        # This would need access to the proc module
        # For now, return a placeholder
        return None

    def is_fixed_function_input(self, expression: Handle[Expression], module: Any) -> Optional[BuiltIn]:
        """
        Returns true if the given expression points to a fixed-function pipeline input.
        """
        # This would need access to the module's entry points
        # For now, return None
        return None


def binary_operation_str(op: BinaryOperator) -> str:
    """
    Helper function that returns the string corresponding to the BinaryOperator.
    """
    from .. import BinaryOperator as Bo

    if op == Bo.ADD:
        return "+"
    elif op == Bo.SUBTRACT:
        return "-"
    elif op == Bo.MULTIPLY:
        return "*"
    elif op == Bo.DIVIDE:
        return "/"
    elif op == Bo.MODULO:
        return "%"
    elif op == Bo.EQUAL:
        return "=="
    elif op == Bo.NOT_EQUAL:
        return "!="
    elif op == Bo.LESS:
        return "<"
    elif op == Bo.LESS_EQUAL:
        return "<="
    elif op == Bo.GREATER:
        return ">"
    elif op == Bo.GREATER_EQUAL:
        return ">="
    elif op == Bo.AND:
        return "&"
    elif op == Bo.EXCLUSIVE_OR:
        return "^"
    elif op == Bo.INCLUSIVE_OR:
        return "|"
    elif op == Bo.LOGICAL_AND:
        return "&&"
    elif op == Bo.LOGICAL_OR:
        return "||"
    elif op == Bo.SHIFT_LEFT:
        return "<<"
    elif op == Bo.SHIFT_RIGHT:
        return ">>"
    else:
        return "?"


class RayFlag(Enum):
    """
    Ray flags, for a RayDesc's flags field.
    """
    OPAQUE = 0x01
    NO_OPAQUE = 0x02
    TERMINATE_ON_FIRST_HIT = 0x04
    SKIP_CLOSEST_HIT_SHADER = 0x08
    CULL_BACK_FACING = 0x10
    CULL_FRONT_FACING = 0x20
    CULL_OPAQUE = 0x40
    CULL_NO_OPAQUE = 0x80
    SKIP_TRIANGLES = 0x100
    SKIP_AABBS = 0x200


class RayIntersectionType(Enum):
    """
    The intersection test to use for ray queries.
    """
    TRIANGLE = 1
    BOUNDING_BOX = 4


from .wgsl import Writer as WgslWriter, WriterFlags, write_string as write_wgsl_string
from .glsl import (
    Writer as GlslWriter,
    Version,
    Profile,
    Options as GlslOptions,
    ShaderStage,
    write_string as write_glsl_string,
)
from .hlsl import (
    Writer as HlslWriter,
    ShaderModel,
    Options as HlslOptions,
    ShaderStage as HlslShaderStage,
    write_string as write_hlsl_string,
)
from .msl import (
    Writer as MslWriter,
    Options as MslOptions,
    ShaderStage as MslShaderStage,
    write_string as write_msl_string,
)
from .spv import (
    Writer as SpvWriter,
    Options as SpvOptions,
    write_binary as write_spirv_binary,
)

__all__ = [
    # Constants
    "COMPONENTS",
    "INDENT",
    "NeedBakeExpressions",
    # Classes
    "Baked",
    "RayQueryPoint",
    "Level",
    "FunctionType",
    "FunctionCtx",
    "RayFlag",
    "RayIntersectionType",
    # Base Writer class
    "Writer",
    # WGSL backend
    "WgslWriter",
    "WriterFlags",
    "write_wgsl_string",
    # GLSL backend
    "GlslWriter",
    "Version",
    "Profile",
    "GlslOptions",
    "ShaderStage",
    "write_glsl_string",
    # HLSL backend
    "HlslWriter",
    "ShaderModel",
    "HlslOptions",
    "HlslShaderStage",
    "write_hlsl_string",
    # MSL backend
    "MslWriter",
    "MslOptions",
    "MslShaderStage",
    "write_msl_string",
    # SPIR-V backend
    "SpvWriter",
    "SpvOptions",
    "write_spirv_binary",
]


class Writer:
    """
    Base class for shader writers (SPV, MSL, HLSL, GLSL, WGSL).

    This class provides a common interface for all backend writers.
    Each backend should inherit from this class and implement the write method.
    """

    def write(self, module: "Any", info: "Any") -> "Any":
        """
        Write the shader module to the target format.

        Args:
            module: The Naga IR module to write
            info: Module validation information

        Returns:
            The generated shader code in the target format

        Raises:
            ShaderError: If writing fails
        """
        raise NotImplementedError("Subclasses must implement write method")

    def finish(self) -> str:
        """
        Finish writing and return the complete output.

        Returns:
            The complete generated shader code
        """
        return ""
