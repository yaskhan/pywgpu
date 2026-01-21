"""Component-wise operation helpers for constant evaluation.

This module provides utilities for performing component-wise operations
on vectors and scalars during constant evaluation.
Mirrors the component_wise_extractor macro in naga/src/proc/constant_evaluator.rs
"""

from __future__ import annotations

from typing import Iterable

from naga import Expression, Handle, Literal, Type
from naga.ir.operators import MathFunction

# Import generated component-wise functions
from .component_wise_impl import (
    Scalar,
    Float,
    ConcreteInt,
    Signed,
    component_wise_scalar,
    component_wise_float,
    component_wise_concrete_int,
    component_wise_signed,
)

__all__ = [
    "Scalar",
    "Float",
    "ConcreteInt",
    "Signed",
    "component_wise_scalar",
    "component_wise_float",
    "component_wise_concrete_int",
    "component_wise_signed",
    "flatten_compose_to_literals",
    "extract_vector_literals",
    "math_function_arg_count",
]


# ============================================================================
# Flattening helper
# ============================================================================

def flatten_compose_to_literals(
    ty: Handle[Type],
    components: list[Handle[Expression]],
    expressions: "Arena[Expression]",
    types: "UniqueArena[Type]",
) -> Iterable[Literal]:
    """Flatten a compose expression to its literal components.

    This function recursively expands nested compose expressions and splats
    to yield all literal components.

    Args:
        ty: Type handle of the compose expression
        components: Component handles
        expressions: Expression arena
        types: Type arena

    Yields:
        Literal values from the flattened expression
    """
    from naga.proc.type_methods import flatten_compose
    
    # Use flatten_compose to get all component handles
    flattened_handles = flatten_compose(ty, components, expressions, types)
    
    # Extract literals from each handle
    for handle in flattened_handles:
        expr = expressions[handle]
        if expr.type == ExpressionType.LITERAL:
            yield expr.literal
        else:
            raise ValueError(f"Non-literal component in flatten_compose_to_literals: {expr.type}")


# ============================================================================
# Helper for extracting vector values
# ============================================================================

def extract_vector_literals(
    expr: Handle[Expression],
    expressions: "Arena[Expression]",
    types: "UniqueArena[Type]",
) -> list[Literal]:
    """Extract all literals from a vector expression.

    Args:
        expr: Handle to the expression
        expressions: Expression arena
        types: Type arena

    Returns:
        List of literals representing the vector's components
    """
    from naga import ExpressionType
    
    expr_obj = expressions[expr]

    match expr_obj.type:
        case ExpressionType.LITERAL:
            return [expr_obj.literal]
        case ExpressionType.COMPOSE:
            return list(flatten_compose_to_literals(
                expr_obj.compose_ty,
                expr_obj.compose_components,
                expressions,
                types
            ))
        case _:
            raise ValueError(f"Cannot extract literals from expression type: {expr_obj.type}")


# ============================================================================
# Math function helpers
# ============================================================================

def math_function_arg_count(fun: MathFunction) -> int:
    """Return the expected number of arguments for a math function.

    Args:
        fun: Math function to query

    Returns:
        Expected number of arguments
    """
    from naga.ir.operators import MathFunction as MF
    
    match fun:
        # comparison - 1 argument
        case MF.ABS | MF.SATURATE:
            return 1
        # comparison - 2 arguments
        case MF.MIN | MF.MAX:
            return 2
        # comparison - 3 arguments
        case MF.CLAMP:
            return 3
            
        # trigonometry - 1 argument
        case (
            MF.COS | MF.COSH | MF.SIN | MF.SINH | MF.TAN | MF.TANH |
            MF.ACOS | MF.ASIN | MF.ATAN | MF.ASINH | MF.ACOSH | MF.ATANH |
            MF.RADIANS | MF.DEGREES
        ):
            return 1
        # trigonometry - 2 arguments
        case MF.ATAN2:
            return 2
            
        # decomposition - 1 argument
        case MF.CEIL | MF.FLOOR | MF.ROUND | MF.FRACT | MF.TRUNC | MF.MODF | MF.FREXP:
            return 1
        # decomposition - 2 arguments
        case MF.LDEXP:
            return 2
            
        # exponent - 1 argument
        case MF.EXP | MF.EXP2 | MF.LOG | MF.LOG2:
            return 1
        # exponent - 2 arguments
        case MF.POW:
            return 2
            
        # geometry - 1 argument
        case MF.LENGTH | MF.NORMALIZE:
            return 1
        # geometry - 2 arguments
        case MF.DOT | MF.CROSS | MF.DISTANCE | MF.OUTER | MF.REFLECT:
            return 2
        # geometry - 3 arguments
        case MF.FACE_FORWARD | MF.REFRACT:
            return 3
            
        # computational - 1 argument
        case MF.SIGN | MF.SQRT | MF.INVERSE_SQRT | MF.INVERSE | MF.TRANSPOSE | MF.DETERMINANT | MF.QUANTIZE_TO_F16:
            return 1
        # computational - 2 arguments
        case MF.STEP:
            return 2
        # computational - 3 arguments
        case MF.FMA | MF.MIX | MF.SMOOTH_STEP:
            return 3
            
        # bits - 1 argument
        case (
            MF.COUNT_TRAILING_ZEROS | MF.COUNT_LEADING_ZEROS | MF.COUNT_ONE_BITS |
            MF.REVERSE_BITS | MF.FIRST_TRAILING_BIT | MF.FIRST_LEADING_BIT
        ):
            return 1
        # bits - 3 arguments
        case MF.EXTRACT_BITS:
            return 3
        # bits - 4 arguments
        case MF.INSERT_BITS:
            return 4
            
        # data packing - 1 argument
        case (
            MF.PACK_4X8_SNORM | MF.PACK_4X8_UNORM | MF.PACK_2X16_SNORM | MF.PACK_2X16_UNORM |
            MF.PACK_2X16_FLOAT | MF.PACK_4X_I8 | MF.PACK_4X_U8 | MF.PACK_4X_I8_CLAMP | MF.PACK_4X_U8_CLAMP
        ):
            return 1
            
        # data unpacking - 1 argument
        case (
            MF.UNPACK_4X8_SNORM | MF.UNPACK_4X8_UNORM | MF.UNPACK_2X16_SNORM | MF.UNPACK_2X16_UNORM |
            MF.UNPACK_2X16_FLOAT | MF.UNPACK_4X_I8 | MF.UNPACK_4X_U8
        ):
            return 1
            
        # packed dot products - 2 arguments
        case MF.DOT4_I8_PACKED | MF.DOT4_U8_PACKED:
            return 2
            
        case _:
            raise ValueError(f"Unknown math function: {fun}")


# Import needed types for annotations
if __name__ != "__main__":
    from naga import Arena, UniqueArena, ExpressionType
    from .constant_evaluator import ConstantEvaluator
