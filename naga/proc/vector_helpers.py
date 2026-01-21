"""Vector operation helpers for constant evaluation.

This module provides utilities for working with vectors in constant expressions,
including extraction, manipulation, and common vector operations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from naga import Handle, Expression, Literal, Span
    from .constant_evaluator import ConstantEvaluator


def extract_vector_float_values(
    eval: "ConstantEvaluator",
    expr: "Handle[Expression]",
    span: "Span",
) -> list[float]:
    """Extract float values from a vector expression.
    
    Args:
        eval: Constant evaluator instance
        expr: Expression handle (must be a vector or scalar)
        span: Span for error reporting
        
    Returns:
        List of float values from the vector
        
    Raises:
        ConstantEvaluatorError: If expression is not a valid vector
    """
    from naga import Expression, ExpressionType, Literal
    from .constant_evaluator import ConstantEvaluatorError
    
    # Evaluate zero values and splats first
    expr_handle = eval.eval_zero_value_and_splat(expr, span)
    expr_obj = eval.expressions[expr_handle]
    
    match expr_obj.type:
        case ExpressionType.LITERAL:
            # Single scalar value
            lit = expr_obj.literal
            match lit:
                case Literal.AbstractFloat(value=v) | Literal.F32(value=v) | Literal.F16(value=v):
                    return [v]
                case _:
                    raise ConstantEvaluatorError("Expected float literal", {})
                    
        case ExpressionType.COMPOSE:
            # Vector composition - extract all components
            from .type_methods import flatten_compose
            
            components = flatten_compose(
                expr_obj.compose_ty,
                expr_obj.compose_components,
                eval.expressions,
                eval.types
            )
            
            values = []
            for comp_handle in components:
                comp_expr = eval.expressions[comp_handle]
                if comp_expr.type == ExpressionType.LITERAL:
                    lit = comp_expr.literal
                    match lit:
                        case Literal.AbstractFloat(value=v) | Literal.F32(value=v) | Literal.F16(value=v):
                            values.append(v)
                        case _:
                            raise ConstantEvaluatorError("Expected float component", {})
                else:
                    raise ConstantEvaluatorError("Non-literal component in vector", {})
            
            return values
            
        case _:
            raise ConstantEvaluatorError("Invalid vector expression", {})


def create_float_literal(
    eval: "ConstantEvaluator",
    value: float,
    span: "Span",
) -> "Handle[Expression]":
    """Create a float literal expression.
    
    Args:
        eval: Constant evaluator instance
        value: Float value
        span: Span for the expression
        
    Returns:
        Handle to the created literal expression
    """
    from naga import Expression, ExpressionType, Literal
    
    # Use F32 by default for computed values
    new_expr = Expression(
        type=ExpressionType.LITERAL,
        literal=Literal.F32(value),
    )
    
    return eval.register_evaluated_expr(new_expr, span)


# ============================================================================
# Vector math operations (pure Python, no numpy)
# ============================================================================

def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Compute dot product of two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Dot product (sum of element-wise products)
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector size mismatch: {len(vec1)} vs {len(vec2)}")
    
    return sum(a * b for a, b in zip(vec1, vec2))


def cross_product(vec1: list[float], vec2: list[float]) -> list[float]:
    """Compute cross product of two 3D vectors.
    
    Args:
        vec1: First 3D vector
        vec2: Second 3D vector
        
    Returns:
        Cross product vector (perpendicular to both inputs)
        
    Raises:
        ValueError: If vectors are not 3D
    """
    if len(vec1) != 3 or len(vec2) != 3:
        raise ValueError("Cross product requires 3D vectors")
    
    a, b = vec1, vec2
    return [
        a[1] * b[2] - a[2] * b[1],  # x component
        a[2] * b[0] - a[0] * b[2],  # y component
        a[0] * b[1] - a[1] * b[0],  # z component
    ]


def vector_length(vec: list[float]) -> float:
    """Compute length (magnitude) of a vector.
    
    Args:
        vec: Input vector
        
    Returns:
        Length of the vector (sqrt of sum of squares)
    """
    return math.sqrt(sum(x * x for x in vec))


def vector_distance(vec1: list[float], vec2: list[float]) -> float:
    """Compute distance between two points.
    
    Args:
        vec1: First point
        vec2: Second point
        
    Returns:
        Distance between points
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector size mismatch: {len(vec1)} vs {len(vec2)}")
    
    diff = [a - b for a, b in zip(vec1, vec2)]
    return vector_length(diff)


def vector_normalize(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector (length = 1)
    """
    length = vector_length(vec)
    if length == 0:
        raise ValueError("Cannot normalize zero vector")
    
    return [x / length for x in vec]
