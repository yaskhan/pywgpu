"""Extended constant evaluator with integrated operations.

This module extends the base constant_evaluator.py with methods that use
the implemented literal operations, zero value generation, and other helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from naga import Expression, Handle, ExpressionType, Span
from naga.ir import (
    Type, TypeInner, BinaryOperator, UnaryOperator,
    Literal, LiteralType,
)

# Import our implemented helpers
from naga.proc.literal_helpers import LiteralVector, literal_ty_inner
from naga.proc.zero_value_helpers import literal_zero, eval_zero_value_impl
from naga.proc.literal_operations import (
    apply_unary_op, apply_binary_op, LiteralOperationError
)

if TYPE_CHECKING:
    from naga.proc.constant_evaluator import ConstantEvaluator, ConstantEvaluatorError


def eval_unary_expression(
    evaluator: ConstantEvaluator,
    op: UnaryOperator,
    expr_handle: Handle,
    span: Span
) -> Handle:
    """Evaluate a unary operation on a constant expression.
    
    Args:
        evaluator: The constant evaluator instance
        op: The unary operator to apply
        expr_handle: Handle to the operand expression
        span: Source span for error reporting
        
    Returns:
        Handle to the result expression
        
    Raises:
        ConstantEvaluatorError: If evaluation fails
    """
    # Get the operand expression
    operand_expr = evaluator.expressions[expr_handle]
    
    # Only handle literal operands for now
    if operand_expr.type != ExpressionType.LITERAL:
        raise ValueError(f"Cannot evaluate unary op on non-literal: {operand_expr.type}")
    
    operand_lit = operand_expr.literal
    
    try:
        # Apply the unary operation
        result_lit = apply_unary_op(op, operand_lit)
        
        # Create result expression
        result_expr = Expression(type=ExpressionType.LITERAL, literal=result_lit)
        
        # Register and return
        return evaluator._append_expr(result_expr, span, evaluator.expression_kind_tracker.type_of_with_expr(result_expr))
        
    except LiteralOperationError as e:
        raise ValueError(f"Unary operation failed: {e}")


def eval_binary_expression(
    evaluator: ConstantEvaluator,
    op: BinaryOperator,
    left_handle: Handle,
    right_handle: Handle,
    span: Span
) -> Handle:
    """Evaluate a binary operation on constant expressions.
    
    Args:
        evaluator: The constant evaluator instance
        op: The binary operator to apply
        left_handle: Handle to the left operand expression
        right_handle: Handle to the right operand expression
        span: Source span for error reporting
        
    Returns:
        Handle to the result expression
        
    Raises:
        ConstantEvaluatorError: If evaluation fails
    """
    # Get the operand expressions
    left_expr = evaluator.expressions[left_handle]
    right_expr = evaluator.expressions[right_handle]
    
    # Only handle literal operands for now
    if left_expr.type != ExpressionType.LITERAL:
        raise ValueError(f"Cannot evaluate binary op on non-literal left: {left_expr.type}")
    if right_expr.type != ExpressionType.LITERAL:
        raise ValueError(f"Cannot evaluate binary op on non-literal right: {right_expr.type}")
    
    left_lit = left_expr.literal
    right_lit = right_expr.literal
    
    try:
        # Apply the binary operation
        result_lit = apply_binary_op(op, left_lit, right_lit)
        
        # Create result expression
        result_expr = Expression(type=ExpressionType.LITERAL, literal=result_lit)
        
        # Register and return
        return evaluator._append_expr(result_expr, span, evaluator.expression_kind_tracker.type_of_with_expr(result_expr))
        
    except LiteralOperationError as e:
        raise ValueError(f"Binary operation failed: {e}")


def eval_zero_value_expression(
    evaluator: ConstantEvaluator,
    ty_handle: Handle,
    span: Span
) -> Handle:
    """Evaluate a zero value expression for a given type.
    
    Args:
        evaluator: The constant evaluator instance
        ty_handle: Handle to the type
        span: Source span for error reporting
        
    Returns:
        Handle to the zero value expression
        
    Raises:
        ConstantEvaluatorError: If type is not constructible
    """
    try:
        return eval_zero_value_impl(evaluator, ty_handle, span)
    except ValueError as e:
        raise ValueError(f"Zero value evaluation failed: {e}")


def eval_compose_expression(
    evaluator: ConstantEvaluator,
    ty_handle: Handle,
    components: list[Handle],
    span: Span
) -> Handle:
    """Evaluate a compose expression.
    
    Args:
        evaluator: The constant evaluator instance
        ty_handle: Handle to the result type
        components: List of component expression handles
        span: Source span for error reporting
        
    Returns:
        Handle to the composed expression
        
    Raises:
        ConstantEvaluatorError: If composition fails
    """
    # Check if all components are const
    for comp in components:
        if not evaluator.expression_kind_tracker.is_const(comp):
            raise ValueError("All components must be const")
    
    # Create compose expression
    compose_expr = Expression(
        type=ExpressionType.COMPOSE,
        compose_ty=ty_handle,
        compose_components=components
    )
    
    # Register and return
    from naga.proc.constant_evaluator import ExpressionKind
    return evaluator._append_expr(compose_expr, span, ExpressionKind.CONST)


def eval_splat_expression(
    evaluator: ConstantEvaluator,
    size: int,  # VectorSize
    value_handle: Handle,
    span: Span
) -> Handle:
    """Evaluate a splat expression (scalar to vector).
    
    Args:
        evaluator: The constant evaluator instance
        size: Vector size (2, 3, or 4)
        value_handle: Handle to the scalar value expression
        span: Source span for error reporting
        
    Returns:
        Handle to the splat expression
        
    Raises:
        ConstantEvaluatorError: If value is not const or not scalar
    """
    # Check if value is const
    if not evaluator.expression_kind_tracker.is_const(value_handle):
        raise ValueError("Splat value must be const")
    
    value_expr = evaluator.expressions[value_handle]
    
    # Only handle literal values for now
    if value_expr.type != ExpressionType.LITERAL:
        raise ValueError("Splat value must be a literal")
    
    # Create splat expression
    from naga.ir import VectorSize
    vec_size = {
        2: VectorSize.BI,
        3: VectorSize.TRI,
        4: VectorSize.QUAD
    }[size]
    
    splat_expr = Expression(
        type=ExpressionType.SPLAT,
        splat_size=vec_size,
        splat_value=value_handle
    )
    
    # Register and return
    from naga.proc.constant_evaluator import ExpressionKind
    return evaluator._append_expr(splat_expr, span, ExpressionKind.CONST)


def eval_access_index_expression(
    evaluator: ConstantEvaluator,
    base_handle: Handle,
    index: int,
    span: Span
) -> Handle:
    """Evaluate an access index expression (constant index).
    
    Args:
        evaluator: The constant evaluator instance
        base_handle: Handle to the base expression (vector, matrix, array, struct)
        index: The constant index
        span: Source span for error reporting
        
    Returns:
        Handle to the accessed element
        
    Raises:
        ConstantEvaluatorError: If access fails
    """
    base_expr = evaluator.expressions[base_handle]
    
    # Handle compose expressions
    if base_expr.type == ExpressionType.COMPOSE:
        components = base_expr.compose_components
        if index >= len(components):
            raise ValueError(f"Index {index} out of bounds for compose with {len(components)} components")
        return components[index]
    
    # For other cases, create access index expression
    access_expr = Expression(
        type=ExpressionType.ACCESS_INDEX,
        access_base=base_handle,
        access_index=index
    )
    
    # Determine kind based on base
    kind = evaluator.expression_kind_tracker.type_of(base_handle)
    return evaluator._append_expr(access_expr, span, kind)


def eval_swizzle_expression(
    evaluator: ConstantEvaluator,
    size: int,  # VectorSize
    vector_handle: Handle,
    pattern: list[int],  # SwizzleComponent indices
    span: Span
) -> Handle:
    """Evaluate a swizzle expression.
    
    Args:
        evaluator: The constant evaluator instance
        size: Result vector size
        vector_handle: Handle to the source vector
        pattern: List of component indices (0=X, 1=Y, 2=Z, 3=W)
        span: Source span for error reporting
        
    Returns:
        Handle to the swizzled vector
        
    Raises:
        ConstantEvaluatorError: If swizzle fails
    """
    vector_expr = evaluator.expressions[vector_handle]
    
    # Handle compose expressions - extract components directly
    if vector_expr.type == ExpressionType.COMPOSE:
        components = vector_expr.compose_components
        swizzled_components = [components[i] for i in pattern[:size]]
        
        # Get the vector type
        ty_handle = vector_expr.compose_ty
        
        # Create new compose with swizzled components
        return eval_compose_expression(evaluator, ty_handle, swizzled_components, span)
    
    # For other cases, create swizzle expression
    from naga.ir import VectorSize, SwizzleComponent
    vec_size = {
        2: VectorSize.BI,
        3: VectorSize.TRI,
        4: VectorSize.QUAD
    }[size]
    
    swizzle_pattern = [SwizzleComponent(i) for i in pattern]
    # Pad to 4 components
    while len(swizzle_pattern) < 4:
        swizzle_pattern.append(SwizzleComponent.X)
    
    swizzle_expr = Expression(
        type=ExpressionType.SWIZZLE,
        swizzle_size=vec_size,
        swizzle_vector=vector_handle,
        swizzle_pattern=swizzle_pattern
    )
    
    # Determine kind based on vector
    kind = evaluator.expression_kind_tracker.type_of(vector_handle)
    return evaluator._append_expr(swizzle_expr, span, kind)


def eval_select_expression(
    evaluator: ConstantEvaluator,
    condition_handle: Handle,
    accept_handle: Handle,
    reject_handle: Handle,
    span: Span
) -> Handle:
    """Evaluate a select expression (ternary conditional).
    
    Args:
        evaluator: The constant evaluator instance
        condition_handle: Handle to the boolean condition
        accept_handle: Handle to the value if condition is true
        reject_handle: Handle to the value if condition is false
        span: Source span for error reporting
        
    Returns:
        Handle to the selected value
        
    Raises:
        ConstantEvaluatorError: If evaluation fails
    """
    # If all are const, try to evaluate
    if (evaluator.expression_kind_tracker.is_const(condition_handle) and
        evaluator.expression_kind_tracker.is_const(accept_handle) and
        evaluator.expression_kind_tracker.is_const(reject_handle)):
        
        condition_expr = evaluator.expressions[condition_handle]
        
        # If condition is a literal bool, select directly
        if condition_expr.type == ExpressionType.LITERAL:
            if condition_expr.literal.type == LiteralType.BOOL:
                if condition_expr.literal.bool:
                    return accept_handle
                else:
                    return reject_handle
    
    # Otherwise, create select expression
    select_expr = Expression(
        type=ExpressionType.SELECT,
        select_condition=condition_handle,
        select_accept=accept_handle,
        select_reject=reject_handle
    )
    
    # Determine kind - max of all three
    cond_kind = evaluator.expression_kind_tracker.type_of(condition_handle)
    accept_kind = evaluator.expression_kind_tracker.type_of(accept_handle)
    reject_kind = evaluator.expression_kind_tracker.type_of(reject_handle)
    kind = max(cond_kind, accept_kind, reject_kind)
    
    return evaluator._append_expr(select_expr, span, kind)
