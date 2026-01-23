"""Binary and Unary operations on Literal values.

This module implements arithmetic, logical, and bitwise operations on Literal values
for constant expression evaluation in NAGA IR.

Translated from wgpu-trunk/naga/src/proc/constant_evaluator.rs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from naga.ir import (
    Literal, LiteralType, BinaryOperator, UnaryOperator,
    ScalarKind
)

if TYPE_CHECKING:
    pass


class LiteralOperationError(Exception):
    """Error during literal operation."""
    pass


def apply_unary_op(op: UnaryOperator, operand: Literal) -> Literal:
    """Apply unary operator to a literal value.
    
    Args:
        op: The unary operator to apply
        operand: The literal operand
        
    Returns:
        Result literal
        
    Raises:
        LiteralOperationError: If operation is invalid for the operand type
    """
    match op:
        case UnaryOperator.NEGATE:
            # Negate: -x
            match operand.type:
                case LiteralType.F64:
                    return Literal(type=LiteralType.F64, f64=-operand.f64)
                case LiteralType.F32:
                    return Literal(type=LiteralType.F32, f32=-operand.f32)
                case LiteralType.F16:
                    return Literal(type=LiteralType.F16, f16=-operand.f16)
                case LiteralType.I32:
                    # Check for overflow (negating INT_MIN)
                    if operand.i32 == -(2**31):
                        raise LiteralOperationError("Overflow: negating INT_MIN")
                    return Literal(type=LiteralType.I32, i32=-operand.i32)
                case LiteralType.I64:
                    if operand.i64 == -(2**63):
                        raise LiteralOperationError("Overflow: negating INT64_MIN")
                    return Literal(type=LiteralType.I64, i64=-operand.i64)
                case LiteralType.ABSTRACT_INT:
                    return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=-operand.abstract_int)
                case LiteralType.ABSTRACT_FLOAT:
                    return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=-operand.abstract_float)
                case _:
                    raise LiteralOperationError(f"Cannot negate {operand.type}")
        
        case UnaryOperator.LOGICAL_NOT:
            # Logical NOT: !x (only for bool)
            if operand.type != LiteralType.BOOL:
                raise LiteralOperationError(f"Logical NOT requires bool, got {operand.type}")
            return Literal(type=LiteralType.BOOL, bool=not operand.bool)
        
        case UnaryOperator.BITWISE_NOT:
            # Bitwise NOT: ~x
            match operand.type:
                case LiteralType.U32:
                    return Literal(type=LiteralType.U32, u32=~operand.u32 & 0xFFFFFFFF)
                case LiteralType.I32:
                    return Literal(type=LiteralType.I32, i32=~operand.i32 & 0xFFFFFFFF)
                case LiteralType.U64:
                    return Literal(type=LiteralType.U64, u64=~operand.u64 & 0xFFFFFFFFFFFFFFFF)
                case LiteralType.I64:
                    return Literal(type=LiteralType.I64, i64=~operand.i64 & 0xFFFFFFFFFFFFFFFF)
                case LiteralType.ABSTRACT_INT:
                    # Abstract int is i64
                    return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=~operand.abstract_int)
                case _:
                    raise LiteralOperationError(f"Cannot apply bitwise NOT to {operand.type}")
        
        case _:
            raise LiteralOperationError(f"Unknown unary operator: {op}")


def apply_binary_op(op: BinaryOperator, left: Literal, right: Literal) -> Literal:
    """Apply binary operator to two literal values.
    
    Args:
        op: The binary operator to apply
        left: The left operand
        right: The right operand
        
    Returns:
        Result literal
        
    Raises:
        LiteralOperationError: If operation is invalid or types mismatch
    """
    # Type checking - operands must have same type for most operations
    if left.type != right.type:
        raise LiteralOperationError(f"Type mismatch: {left.type} vs {right.type}")
    
    lit_type = left.type
    
    match op:
        # Arithmetic operations
        case BinaryOperator.ADD:
            return _apply_add(left, right, lit_type)
        case BinaryOperator.SUBTRACT:
            return _apply_subtract(left, right, lit_type)
        case BinaryOperator.MULTIPLY:
            return _apply_multiply(left, right, lit_type)
        case BinaryOperator.DIVIDE:
            return _apply_divide(left, right, lit_type)
        case BinaryOperator.MODULO:
            return _apply_modulo(left, right, lit_type)
        
        # Comparison operations
        case BinaryOperator.EQUAL:
            return _apply_equal(left, right, lit_type)
        case BinaryOperator.NOT_EQUAL:
            result = _apply_equal(left, right, lit_type)
            return Literal(type=LiteralType.BOOL, bool=not result.bool)
        case BinaryOperator.LESS:
            return _apply_less(left, right, lit_type)
        case BinaryOperator.LESS_EQUAL:
            return _apply_less_equal(left, right, lit_type)
        case BinaryOperator.GREATER:
            return _apply_greater(left, right, lit_type)
        case BinaryOperator.GREATER_EQUAL:
            return _apply_greater_equal(left, right, lit_type)
        
        # Logical operations
        case BinaryOperator.LOGICAL_AND:
            if lit_type != LiteralType.BOOL:
                raise LiteralOperationError("Logical AND requires bool operands")
            return Literal(type=LiteralType.BOOL, bool=left.bool and right.bool)
        case BinaryOperator.LOGICAL_OR:
            if lit_type != LiteralType.BOOL:
                raise LiteralOperationError("Logical OR requires bool operands")
            return Literal(type=LiteralType.BOOL, bool=left.bool or right.bool)
        
        # Bitwise operations
        case BinaryOperator.AND:
            return _apply_bitwise_and(left, right, lit_type)
        case BinaryOperator.INCLUSIVE_OR:
            return _apply_bitwise_or(left, right, lit_type)
        case BinaryOperator.EXCLUSIVE_OR:
            return _apply_bitwise_xor(left, right, lit_type)
        case BinaryOperator.SHIFT_LEFT:
            return _apply_shift_left(left, right, lit_type)
        case BinaryOperator.SHIFT_RIGHT:
            return _apply_shift_right(left, right, lit_type)
        
        case _:
            raise LiteralOperationError(f"Unknown binary operator: {op}")


# Helper functions for arithmetic operations

def _apply_add(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply addition."""
    match lit_type:
        case LiteralType.F64:
            return Literal(type=LiteralType.F64, f64=left.f64 + right.f64)
        case LiteralType.F32:
            return Literal(type=LiteralType.F32, f32=left.f32 + right.f32)
        case LiteralType.F16:
            return Literal(type=LiteralType.F16, f16=left.f16 + right.f16)
        case LiteralType.U32:
            result = (left.u32 + right.u32) & 0xFFFFFFFF
            return Literal(type=LiteralType.U32, u32=result)
        case LiteralType.I32:
            result = (left.i32 + right.i32)
            # Wrap around for signed integers
            if result > 2**31 - 1:
                result -= 2**32
            elif result < -(2**31):
                result += 2**32
            return Literal(type=LiteralType.I32, i32=result)
        case LiteralType.U64:
            result = (left.u64 + right.u64) & 0xFFFFFFFFFFFFFFFF
            return Literal(type=LiteralType.U64, u64=result)
        case LiteralType.I64:
            result = left.i64 + right.i64
            if result > 2**63 - 1:
                result -= 2**64
            elif result < -(2**63):
                result += 2**64
            return Literal(type=LiteralType.I64, i64=result)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int + right.abstract_int)
        case LiteralType.ABSTRACT_FLOAT:
            return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=left.abstract_float + right.abstract_float)
        case _:
            raise LiteralOperationError(f"Cannot add {lit_type}")


def _apply_subtract(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply subtraction."""
    match lit_type:
        case LiteralType.F64:
            return Literal(type=LiteralType.F64, f64=left.f64 - right.f64)
        case LiteralType.F32:
            return Literal(type=LiteralType.F32, f32=left.f32 - right.f32)
        case LiteralType.F16:
            return Literal(type=LiteralType.F16, f16=left.f16 - right.f16)
        case LiteralType.U32:
            result = (left.u32 - right.u32) & 0xFFFFFFFF
            return Literal(type=LiteralType.U32, u32=result)
        case LiteralType.I32:
            result = left.i32 - right.i32
            if result > 2**31 - 1:
                result -= 2**32
            elif result < -(2**31):
                result += 2**32
            return Literal(type=LiteralType.I32, i32=result)
        case LiteralType.U64:
            result = (left.u64 - right.u64) & 0xFFFFFFFFFFFFFFFF
            return Literal(type=LiteralType.U64, u64=result)
        case LiteralType.I64:
            result = left.i64 - right.i64
            if result > 2**63 - 1:
                result -= 2**64
            elif result < -(2**63):
                result += 2**64
            return Literal(type=LiteralType.I64, i64=result)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int - right.abstract_int)
        case LiteralType.ABSTRACT_FLOAT:
            return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=left.abstract_float - right.abstract_float)
        case _:
            raise LiteralOperationError(f"Cannot subtract {lit_type}")


def _apply_multiply(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply multiplication."""
    match lit_type:
        case LiteralType.F64:
            return Literal(type=LiteralType.F64, f64=left.f64 * right.f64)
        case LiteralType.F32:
            return Literal(type=LiteralType.F32, f32=left.f32 * right.f32)
        case LiteralType.F16:
            return Literal(type=LiteralType.F16, f16=left.f16 * right.f16)
        case LiteralType.U32:
            result = (left.u32 * right.u32) & 0xFFFFFFFF
            return Literal(type=LiteralType.U32, u32=result)
        case LiteralType.I32:
            result = left.i32 * right.i32
            if result > 2**31 - 1:
                result = (result + 2**31) % 2**32 - 2**31
            elif result < -(2**31):
                result = (result + 2**31) % 2**32 - 2**31
            return Literal(type=LiteralType.I32, i32=result)
        case LiteralType.U64:
            result = (left.u64 * right.u64) & 0xFFFFFFFFFFFFFFFF
            return Literal(type=LiteralType.U64, u64=result)
        case LiteralType.I64:
            result = left.i64 * right.i64
            if result > 2**63 - 1:
                result = (result + 2**63) % 2**64 - 2**63
            elif result < -(2**63):
                result = (result + 2**63) % 2**64 - 2**63
            return Literal(type=LiteralType.I64, i64=result)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int * right.abstract_int)
        case LiteralType.ABSTRACT_FLOAT:
            return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=left.abstract_float * right.abstract_float)
        case _:
            raise LiteralOperationError(f"Cannot multiply {lit_type}")


def _apply_divide(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply division."""
    match lit_type:
        case LiteralType.F64:
            if right.f64 == 0.0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.F64, f64=left.f64 / right.f64)
        case LiteralType.F32:
            if right.f32 == 0.0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.F32, f32=left.f32 / right.f32)
        case LiteralType.F16:
            if right.f16 == 0.0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.F16, f16=left.f16 / right.f16)
        case LiteralType.U32:
            if right.u32 == 0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.U32, u32=left.u32 // right.u32)
        case LiteralType.I32:
            if right.i32 == 0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.I32, i32=left.i32 // right.i32)
        case LiteralType.U64:
            if right.u64 == 0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.U64, u64=left.u64 // right.u64)
        case LiteralType.I64:
            if right.i64 == 0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.I64, i64=left.i64 // right.i64)
        case LiteralType.ABSTRACT_INT:
            if right.abstract_int == 0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int // right.abstract_int)
        case LiteralType.ABSTRACT_FLOAT:
            if right.abstract_float == 0.0:
                raise LiteralOperationError("Division by zero")
            return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=left.abstract_float / right.abstract_float)
        case _:
            raise LiteralOperationError(f"Cannot divide {lit_type}")


def _apply_modulo(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply modulo/remainder operation."""
    match lit_type:
        case LiteralType.F64:
            if right.f64 == 0.0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.F64, f64=left.f64 % right.f64)
        case LiteralType.F32:
            if right.f32 == 0.0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.F32, f32=left.f32 % right.f32)
        case LiteralType.F16:
            if right.f16 == 0.0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.F16, f16=left.f16 % right.f16)
        case LiteralType.U32:
            if right.u32 == 0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.U32, u32=left.u32 % right.u32)
        case LiteralType.I32:
            if right.i32 == 0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.I32, i32=left.i32 % right.i32)
        case LiteralType.U64:
            if right.u64 == 0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.U64, u64=left.u64 % right.u64)
        case LiteralType.I64:
            if right.i64 == 0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.I64, i64=left.i64 % right.i64)
        case LiteralType.ABSTRACT_INT:
            if right.abstract_int == 0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int % right.abstract_int)
        case LiteralType.ABSTRACT_FLOAT:
            if right.abstract_float == 0.0:
                raise LiteralOperationError("Remainder by zero")
            return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=left.abstract_float % right.abstract_float)
        case _:
            raise LiteralOperationError(f"Cannot apply modulo to {lit_type}")


# Comparison operations

def _apply_equal(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply equality comparison."""
    match lit_type:
        case LiteralType.F64:
            result = left.f64 == right.f64
        case LiteralType.F32:
            result = left.f32 == right.f32
        case LiteralType.F16:
            result = left.f16 == right.f16
        case LiteralType.U32:
            result = left.u32 == right.u32
        case LiteralType.I32:
            result = left.i32 == right.i32
        case LiteralType.U64:
            result = left.u64 == right.u64
        case LiteralType.I64:
            result = left.i64 == right.i64
        case LiteralType.BOOL:
            result = left.bool == right.bool
        case LiteralType.ABSTRACT_INT:
            result = left.abstract_int == right.abstract_int
        case LiteralType.ABSTRACT_FLOAT:
            result = left.abstract_float == right.abstract_float
        case _:
            raise LiteralOperationError(f"Cannot compare {lit_type}")
    return Literal(type=LiteralType.BOOL, bool=result)


def _apply_less(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply less-than comparison."""
    match lit_type:
        case LiteralType.F64:
            result = left.f64 < right.f64
        case LiteralType.F32:
            result = left.f32 < right.f32
        case LiteralType.F16:
            result = left.f16 < right.f16
        case LiteralType.U32:
            result = left.u32 < right.u32
        case LiteralType.I32:
            result = left.i32 < right.i32
        case LiteralType.U64:
            result = left.u64 < right.u64
        case LiteralType.I64:
            result = left.i64 < right.i64
        case LiteralType.ABSTRACT_INT:
            result = left.abstract_int < right.abstract_int
        case LiteralType.ABSTRACT_FLOAT:
            result = left.abstract_float < right.abstract_float
        case _:
            raise LiteralOperationError(f"Cannot compare {lit_type}")
    return Literal(type=LiteralType.BOOL, bool=result)


def _apply_less_equal(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply less-than-or-equal comparison."""
    match lit_type:
        case LiteralType.F64:
            result = left.f64 <= right.f64
        case LiteralType.F32:
            result = left.f32 <= right.f32
        case LiteralType.F16:
            result = left.f16 <= right.f16
        case LiteralType.U32:
            result = left.u32 <= right.u32
        case LiteralType.I32:
            result = left.i32 <= right.i32
        case LiteralType.U64:
            result = left.u64 <= right.u64
        case LiteralType.I64:
            result = left.i64 <= right.i64
        case LiteralType.ABSTRACT_INT:
            result = left.abstract_int <= right.abstract_int
        case LiteralType.ABSTRACT_FLOAT:
            result = left.abstract_float <= right.abstract_float
        case _:
            raise LiteralOperationError(f"Cannot compare {lit_type}")
    return Literal(type=LiteralType.BOOL, bool=result)


def _apply_greater(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply greater-than comparison."""
    match lit_type:
        case LiteralType.F64:
            result = left.f64 > right.f64
        case LiteralType.F32:
            result = left.f32 > right.f32
        case LiteralType.F16:
            result = left.f16 > right.f16
        case LiteralType.U32:
            result = left.u32 > right.u32
        case LiteralType.I32:
            result = left.i32 > right.i32
        case LiteralType.U64:
            result = left.u64 > right.u64
        case LiteralType.I64:
            result = left.i64 > right.i64
        case LiteralType.ABSTRACT_INT:
            result = left.abstract_int > right.abstract_int
        case LiteralType.ABSTRACT_FLOAT:
            result = left.abstract_float > right.abstract_float
        case _:
            raise LiteralOperationError(f"Cannot compare {lit_type}")
    return Literal(type=LiteralType.BOOL, bool=result)


def _apply_greater_equal(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply greater-than-or-equal comparison."""
    match lit_type:
        case LiteralType.F64:
            result = left.f64 >= right.f64
        case LiteralType.F32:
            result = left.f32 >= right.f32
        case LiteralType.F16:
            result = left.f16 >= right.f16
        case LiteralType.U32:
            result = left.u32 >= right.u32
        case LiteralType.I32:
            result = left.i32 >= right.i32
        case LiteralType.U64:
            result = left.u64 >= right.u64
        case LiteralType.I64:
            result = left.i64 >= right.i64
        case LiteralType.ABSTRACT_INT:
            result = left.abstract_int >= right.abstract_int
        case LiteralType.ABSTRACT_FLOAT:
            result = left.abstract_float >= right.abstract_float
        case _:
            raise LiteralOperationError(f"Cannot compare {lit_type}")
    return Literal(type=LiteralType.BOOL, bool=result)


# Bitwise operations

def _apply_bitwise_and(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply bitwise AND."""
    match lit_type:
        case LiteralType.U32:
            return Literal(type=LiteralType.U32, u32=left.u32 & right.u32)
        case LiteralType.I32:
            return Literal(type=LiteralType.I32, i32=left.i32 & right.i32)
        case LiteralType.U64:
            return Literal(type=LiteralType.U64, u64=left.u64 & right.u64)
        case LiteralType.I64:
            return Literal(type=LiteralType.I64, i64=left.i64 & right.i64)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int & right.abstract_int)
        case _:
            raise LiteralOperationError(f"Cannot apply bitwise AND to {lit_type}")


def _apply_bitwise_or(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply bitwise OR."""
    match lit_type:
        case LiteralType.U32:
            return Literal(type=LiteralType.U32, u32=left.u32 | right.u32)
        case LiteralType.I32:
            return Literal(type=LiteralType.I32, i32=left.i32 | right.i32)
        case LiteralType.U64:
            return Literal(type=LiteralType.U64, u64=left.u64 | right.u64)
        case LiteralType.I64:
            return Literal(type=LiteralType.I64, i64=left.i64 | right.i64)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int | right.abstract_int)
        case _:
            raise LiteralOperationError(f"Cannot apply bitwise OR to {lit_type}")


def _apply_bitwise_xor(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply bitwise XOR."""
    match lit_type:
        case LiteralType.U32:
            return Literal(type=LiteralType.U32, u32=left.u32 ^ right.u32)
        case LiteralType.I32:
            return Literal(type=LiteralType.I32, i32=left.i32 ^ right.i32)
        case LiteralType.U64:
            return Literal(type=LiteralType.U64, u64=left.u64 ^ right.u64)
        case LiteralType.I64:
            return Literal(type=LiteralType.I64, i64=left.i64 ^ right.i64)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int ^ right.abstract_int)
        case _:
            raise LiteralOperationError(f"Cannot apply bitwise XOR to {lit_type}")


def _apply_shift_left(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply left shift."""
    # Get shift amount
    match right.type:
        case LiteralType.U32:
            shift = right.u32
        case LiteralType.I32:
            shift = right.i32
        case LiteralType.ABSTRACT_INT:
            shift = right.abstract_int
        case _:
            raise LiteralOperationError(f"Invalid shift amount type: {right.type}")
    
    if shift < 0:
        raise LiteralOperationError("Shift amount cannot be negative")
    if shift >= 32 and lit_type in (LiteralType.U32, LiteralType.I32):
        raise LiteralOperationError("Shifted more than 32 bits")
    if shift >= 64 and lit_type in (LiteralType.U64, LiteralType.I64, LiteralType.ABSTRACT_INT):
        raise LiteralOperationError("Shifted more than 64 bits")
    
    match lit_type:
        case LiteralType.U32:
            return Literal(type=LiteralType.U32, u32=(left.u32 << shift) & 0xFFFFFFFF)
        case LiteralType.I32:
            return Literal(type=LiteralType.I32, i32=(left.i32 << shift) & 0xFFFFFFFF)
        case LiteralType.U64:
            return Literal(type=LiteralType.U64, u64=(left.u64 << shift) & 0xFFFFFFFFFFFFFFFF)
        case LiteralType.I64:
            return Literal(type=LiteralType.I64, i64=(left.i64 << shift) & 0xFFFFFFFFFFFFFFFF)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int << shift)
        case _:
            raise LiteralOperationError(f"Cannot shift {lit_type}")


def _apply_shift_right(left: Literal, right: Literal, lit_type: LiteralType) -> Literal:
    """Apply right shift."""
    # Get shift amount
    match right.type:
        case LiteralType.U32:
            shift = right.u32
        case LiteralType.I32:
            shift = right.i32
        case LiteralType.ABSTRACT_INT:
            shift = right.abstract_int
        case _:
            raise LiteralOperationError(f"Invalid shift amount type: {right.type}")
    
    if shift < 0:
        raise LiteralOperationError("Shift amount cannot be negative")
    if shift >= 32 and lit_type in (LiteralType.U32, LiteralType.I32):
        raise LiteralOperationError("Shifted more than 32 bits")
    if shift >= 64 and lit_type in (LiteralType.U64, LiteralType.I64, LiteralType.ABSTRACT_INT):
        raise LiteralOperationError("Shifted more than 64 bits")
    
    match lit_type:
        case LiteralType.U32:
            # Unsigned: logical shift
            return Literal(type=LiteralType.U32, u32=left.u32 >> shift)
        case LiteralType.I32:
            # Signed: arithmetic shift (Python's >> does this automatically for negative numbers)
            return Literal(type=LiteralType.I32, i32=left.i32 >> shift)
        case LiteralType.U64:
            return Literal(type=LiteralType.U64, u64=left.u64 >> shift)
        case LiteralType.I64:
            return Literal(type=LiteralType.I64, i64=left.i64 >> shift)
        case LiteralType.ABSTRACT_INT:
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=left.abstract_int >> shift)
        case _:
            raise LiteralOperationError(f"Cannot shift {lit_type}")
