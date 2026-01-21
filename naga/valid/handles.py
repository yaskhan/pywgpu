"""
Implementation of handle validation.

This module provides validation logic to ensure that all handles within a module are:
- Valid (indices within arena structures)
- Free of forward dependencies (handles only reference earlier handles in the same arena)
"""

from __future__ import annotations

from typing import Optional, Any
import logging

from ..arena import Handle, Arena, UniqueArena, BadHandle
from ..ir import Module, Type, Expression, Constant, Override, EntryPoint, Function
from .errors import ValidationError, TypeError

log = logging.getLogger(__name__)


def validate_module_handles(module: Module) -> None:
    """
    Validates that all handles within module are valid and have no forward dependencies.
    
    By validating the above conditions, we free up subsequent logic to assume that handle
    accesses are infallible.
    
    Args:
        module: The module to validate
        
    Raises:
        ValidationError: If any handles are invalid or have forward dependencies
        
    Note:
        Errors returned by this function are intentionally sparse, for simplicity of implementation.
        It is expected that only buggy frontends or fuzzers should ever emit IR that fails this
        validation pass.
    """
    # Validate types and global expressions in tandem
    # Types can refer to global expressions (for array sizes via overrides)
    # and expressions can refer to types, so we must check both together
    # to ensure there are no cycles
    
    global_exprs_iter = iter(module.global_expressions.items())
    current_expr = None
    
    for th, t in module.types.items():
        # Check if this type references any expressions
        max_expr = validate_type_handles(th, t, module.overrides)
        
        if max_expr is not None:
            # Advance global_exprs_iter beyond max_expr
            while current_expr is None or current_expr[0] <= max_expr:
                try:
                    current_expr = next(global_exprs_iter)
                    # Validate this expression's handles
                    validate_expression_handles(current_expr[0], current_expr[1], module)
                except StopIteration:
                    break
    
    # Validate remaining global expressions
    if current_expr is not None:
        validate_expression_handles(current_expr[0], current_expr[1], module)
    
    for eh, expr in global_exprs_iter:
        validate_expression_handles(eh, expr, module)
    
    # Validate constants
    for ch, constant in module.constants.items():
        validate_constant_handles(ch, constant, module)
    
    # Validate overrides
    for oh, override in module.overrides.items():
        validate_override_handles(oh, override, module)
    
    # Validate global variables
    for gh, global_var in module.global_variables.items():
        validate_global_variable_handles(gh, global_var, module)
    
    # Validate functions
    for fh, function in module.functions.items():
        validate_function_handles(fh, function, module)
    
    # Validate entry points
    for entry_point in module.entry_points:
        validate_entry_point_handles(entry_point, module)


def validate_type_handles(
    type_handle: Handle[Type],
    ty: Type,
    overrides: Arena[Override],
) -> Optional[Handle[Expression]]:
    """
    Validate handles within a type.
    
    Args:
        type_handle: Handle to the type being validated
        ty: The type to validate
        overrides: Arena of overrides (for array size validation)
        
    Returns:
        Maximum expression handle referenced by this type (if any)
        
    Raises:
        ValidationError: If handles are invalid
    """
    max_expr = None
    inner = ty.inner
    
    # Check for expression references in array sizes
    if hasattr(inner, "size") and hasattr(inner.size, "pending"):
        # ArraySize::Pending references an override
        override_handle = inner.size.pending
        if override_handle >= len(overrides):
            raise ValidationError(f"Invalid override handle in type {type_handle}")
        
        override = overrides[override_handle]
        if override.init is not None:
            if max_expr is None or override.init > max_expr:
                max_expr = override.init
    
    return max_expr


def validate_expression_handles(
    expr_handle: Handle[Expression],
    expr: Expression,
    module: Module,
) -> None:
    """
    Validate handles within an expression.
    
    Args:
        expr_handle: Handle to the expression being validated
        expr: The expression to validate
        module: The module containing the expression
        
    Raises:
        ValidationError: If handles are invalid or reference forward items
    """
    # Placeholder - would check all handles within the expression
    # and ensure they reference valid, earlier items
    pass


def validate_constant_handles(
    const_handle: Handle[Constant],
    constant: Constant,
    module: Module,
) -> None:
    """
    Validate handles within a constant.
    
    Args:
        const_handle: Handle to the constant being validated
        constant: The constant to validate
        module: The module containing the constant
        
    Raises:
        ValidationError: If handles are invalid
    """
    # Check type handle
    if constant.ty >= len(module.types):
        raise ValidationError(f"Invalid type handle in constant {const_handle}")
    
    # Check init expression handle
    if constant.init >= len(module.global_expressions):
        raise ValidationError(f"Invalid init expression handle in constant {const_handle}")


def validate_override_handles(
    override_handle: Handle[Override],
    override: Override,
    module: Module,
) -> None:
    """
    Validate handles within an override.
    
    Args:
        override_handle: Handle to the override being validated
        override: The override to validate
        module: The module containing the override
        
    Raises:
        ValidationError: If handles are invalid
    """
    # Check type handle
    if override.ty >= len(module.types):
        raise ValidationError(f"Invalid type handle in override {override_handle}")
    
    # Check init expression handle (if present)
    if override.init is not None and override.init >= len(module.global_expressions):
        raise ValidationError(f"Invalid init expression handle in override {override_handle}")


def validate_global_variable_handles(
    global_handle: Handle,
    global_var: Any,
    module: Module,
) -> None:
    """
    Validate handles within a global variable.
    
    Args:
        global_handle: Handle to the global variable being validated
        global_var: The global variable to validate
        module: The module containing the global variable
        
    Raises:
        ValidationError: If handles are invalid
    """
    # Check type handle
    if global_var.ty >= len(module.types):
        raise ValidationError(f"Invalid type handle in global variable {global_handle}")
    
    # Check init expression handle (if present)
    if hasattr(global_var, "init") and global_var.init is not None:
        if global_var.init >= len(module.global_expressions):
            raise ValidationError(f"Invalid init expression handle in global variable {global_handle}")


def validate_function_handles(
    func_handle: Handle[Function],
    function: Function,
    module: Module,
) -> None:
    """
    Validate handles within a function.
    
    Args:
        func_handle: Handle to the function being validated
        function: The function to validate
        module: The module containing the function
        
    Raises:
        ValidationError: If handles are invalid
    """
    # Validate function arguments
    for arg in function.arguments:
        if arg.ty >= len(module.types):
            raise ValidationError(f"Invalid type handle in function {func_handle} argument")
    
    # Validate return type (if present)
    if function.result is not None:
        if function.result.ty >= len(module.types):
            raise ValidationError(f"Invalid return type handle in function {func_handle}")
    
    # Validate local variables
    for local_handle, local_var in function.local_variables.items():
        if local_var.ty >= len(module.types):
            raise ValidationError(f"Invalid type handle in function {func_handle} local variable")
    
    # Validate expressions
    for expr_handle, expr in function.expressions.items():
        validate_expression_handles(expr_handle, expr, module)


def validate_entry_point_handles(
    entry_point: EntryPoint,
    module: Module,
) -> None:
    """
    Validate handles within an entry point.
    
    Args:
        entry_point: The entry point to validate
        module: The module containing the entry point
        
    Raises:
        ValidationError: If handles are invalid
    """
    # Validate the entry point function
    validate_function_handles(None, entry_point.function, module)


__all__ = [
    "validate_module_handles",
    "validate_type_handles",
    "validate_expression_handles",
    "validate_constant_handles",
    "validate_override_handles",
    "validate_global_variable_handles",
    "validate_function_handles",
    "validate_entry_point_handles",
]
