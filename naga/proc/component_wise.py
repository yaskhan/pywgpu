"""Component-wise operation helpers for constant evaluation.

This module provides utilities for performing component-wise operations
on vectors and scalars during constant evaluation.
Mirrors the component_wise_extractor macro in naga/src/proc/constant_evaluator.rs
"""

from __future__ import annotations

from typing import Generic, TypeVar, Callable as CallableType, Iterable

from naga import Expression, Handle, Literal, Span, Type, TypeInner, ScalarKind
from naga import Scalar


N = TypeVar("N")
M = TypeVar("M")


# ============================================================================
# Scalar extraction types (mimicking the macro-generated enums)
# ============================================================================

class _ScalarWrapper:
    """Base wrapper for scalar values."""

    @staticmethod
    def from_literal(literal: Literal) -> _ScalarWrapper:
        """Create wrapper from a literal."""
        # TODO: Implement based on literal type
        raise NotImplementedError("_ScalarWrapper.from_literal")


def component_wise_scalar(
    eval: "ConstantEvaluator",
    span: Span,
    exprs: list[Handle[Expression]],
    handler: CallableType[[list[float | int]], Handle[Expression]],
) -> Handle[Expression]:
    """Perform component-wise scalar operation on expressions.

    If expressions are vectors of the same length, handler is called
    for each corresponding component of each vector.

    Args:
        eval: Constant evaluator instance
        span: Span for error reporting
        exprs: List of expressions to process
        handler: Function to call on component values

    Returns:
        Handle to the resulting expression

    Note:
        TODO: Implement full component-wise processing
    """
    # TODO: Implement component-wise extraction and processing
    raise NotImplementedError("component_wise_scalar")


def component_wise_float(
    eval: "ConstantEvaluator",
    span: Span,
    exprs: list[Handle[Expression]],
    handler: CallableType[[list[float]], Handle[Expression]],
) -> Handle[Expression]:
    """Perform component-wise float operation on expressions.

    Args:
        eval: Constant evaluator instance
        span: Span for error reporting
        exprs: List of expressions to process
        handler: Function to call on component values

    Returns:
        Handle to the resulting expression

    Note:
        TODO: Implement full component-wise processing
    """
    # TODO: Implement component-wise extraction and processing
    raise NotImplementedError("component_wise_float")


def component_wise_concrete_int(
    eval: "ConstantEvaluator",
    span: Span,
    exprs: list[Handle[Expression]],
    handler: CallableType[[list[int]], Handle[Expression]],
) -> Handle[Expression]:
    """Perform component-wise concrete integer operation on expressions.

    Args:
        eval: Constant evaluator instance
        span: Span for error reporting
        exprs: List of expressions to process
        handler: Function to call on component values

    Returns:
        Handle to the resulting expression

    Note:
        TODO: Implement full component-wise processing
    """
    # TODO: Implement component-wise extraction and processing
    raise NotImplementedError("component_wise_concrete_int")


def component_wise_signed(
    eval: "ConstantEvaluator",
    span: Span,
    exprs: list[Handle[Expression]],
    handler: CallableType[[list[float | int]], Handle[Expression]],
) -> Handle[Expression]:
    """Perform component-wise signed number operation on expressions.

    Args:
        eval: Constant evaluator instance
        span: Span for error reporting
        exprs: List of expressions to process
        handler: Function to call on component values

    Returns:
        Handle to the resulting expression

    Note:
        TODO: Implement full component-wise processing
    """
    # TODO: Implement component-wise extraction and processing
    raise NotImplementedError("component_wise_signed")


# ============================================================================
# Flattening helper
# ============================================================================

def flatten_compose_to_literals(
    ty: Handle[Type],
    components: list[Handle[Expression]],
    expressions: Arena[Expression],
    types: UniqueArena[Type],
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

    Note:
        TODO: Implement recursive flattening
    """
    # TODO: Implement recursive flattening
    for comp in components:
        expr = expressions[comp]
        if isinstance(expr, Expression.Literal):
            yield expr.value
        else:
            # For now, raise on non-literal components
            raise ValueError("Non-literal component in flatten_compose_to_literals")


# ============================================================================
# Helper for extracting vector values
# ============================================================================

def extract_vector_literals(
    expr: Handle[Expression],
    expressions: Arena[Expression],
    types: UniqueArena[Type],
) -> list[Literal]:
    """Extract all literals from a vector expression.

    Args:
        expr: Handle to the expression
        expressions: Expression arena
        types: Type arena

    Returns:
        List of literals representing the vector's components

    Note:
        TODO: Implement proper vector extraction
    """
    expr_obj = expressions[expr]

    match expr_obj:
        case Expression.Literal(value=literal):
            return [literal]
        case Expression.Compose(ty=ty, components=components):
            return list(flatten_compose_to_literals(ty, components, expressions, types))
        case _:
            raise ValueError("Cannot extract literals from non-vector expression")


# ============================================================================
# Math function helpers
# ============================================================================

def math_function_arg_count(fun: MathFunction) -> int:
    """Return the expected number of arguments for a math function.

    Args:
        fun: Math function to query

    Returns:
        Expected number of arguments

    Note:
        TODO: Implement based on MathFunction enum
    """
    match fun:
        # Single argument functions
        case (
            MathFunction.Abs
            | MathFunction.Acos
            | MathFunction.Asin
            | MathFunction.Atan
            | MathFunction.Ceiling
            | MathFunction.Cos
            | MathFunction.Cosh
            | MathFunction.Exp
            | MathFunction.Exp2
            | MathFunction.Floor
            | MathFunction.Fract
            | MathFunction.Length
            | MathFunction.Log
            | MathFunction.Log2
            | MathFunction.Normalize
            | MathFunction.Round
            | MathFunction.Sign
            | MathFunction.Sin
            | MathFunction.Sinh
            | MathFunction.Sqrt
            | MathFunction.Sqr
            | MathFunction.Tan
            | MathFunction.Tanh
            | MathFunction.Trunc
            | MathFunction.CountLeadingZeros
            | MathFunction.CountOneBits
            | MathFunction.CountTrailingZeros
            | MathFunction.FirstLeadingBit
            | MathFunction.FirstTrailingBit
            | MathFunction.InverseSqrt
            | MathFunction.Saturate
        ):
            return 1

        # Two argument functions
        case (
            MathFunction.Atan2
            | MathFunction.Distance
            | MathFunction.Max
            | MathFunction.Min
            | MathFunction.Pow
            | MathFunction.Step
            | MathFunction.Dot4I8Packed
            | MathFunction.Dot4U8Packed
        ):
            return 2

        # Three argument functions
        case (
            MathFunction.Clamp
            | MathFunction.Fma
            | MathFunction.Mix
            | MathFunction.SmoothStep
        ):
            return 3

        # Functions with special argument handling
        case MathFunction.Cross:
            return 2  # Two vectors of 3 components each
        case MathFunction.Dot:
            return 2  # Two vectors

        # Unknown functions
        case _:
            raise ValueError(f"Unknown math function: {fun}")


# Import needed types for annotations
if __name__ != "__main__":
    from naga import Arena, UniqueArena
    from .constant_evaluator import ConstantEvaluator
