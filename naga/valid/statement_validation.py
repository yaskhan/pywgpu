"""Statement validation for NAGA validator.

This module provides comprehensive statement validation including:
- Variable declarations
- Assignments and stores
- Control flow (if, loop, switch)
- Function calls
- Return statements
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..arena import Handle, Arena
from ..ir import Statement, StatementType, Expression
from .errors import ValidationError
from .module_info import ModuleInfo

if TYPE_CHECKING:
    from ..ir.module import Module
    from ..ir.function import Function


class StatementValidator:
    """Validates statements in a function body."""
    
    def __init__(
        self,
        module: Module,
        mod_info: ModuleInfo,
        function: Function,
    ) -> None:
        """Initialize statement validator.
        
        Args:
            module: The module being validated
            mod_info: Module validation info
            function: The function being validated
        """
        self.module = module
        self.mod_info = mod_info
        self.function = function
    
    def validate_statement(
        self,
        stmt: Statement,
    ) -> None:
        """Validate a single statement.
        
        Args:
            stmt: The statement to validate
            
        Raises:
            ValidationError: If the statement is invalid
        """
        match stmt.type:
            case StatementType.EMIT:
                self._validate_emit(stmt)
            case StatementType.BLOCK:
                self._validate_block(stmt)
            case StatementType.IF:
                self._validate_if(stmt)
            case StatementType.SWITCH:
                self._validate_switch(stmt)
            case StatementType.LOOP:
                self._validate_loop(stmt)
            case StatementType.BREAK:
                self._validate_break(stmt)
            case StatementType.CONTINUE:
                self._validate_continue(stmt)
            case StatementType.RETURN:
                self._validate_return(stmt)
            case StatementType.KILL:
                self._validate_kill(stmt)
            case StatementType.BARRIER:
                self._validate_barrier(stmt)
            case StatementType.STORE:
                self._validate_store(stmt)
            case StatementType.IMAGE_STORE:
                self._validate_image_store(stmt)
            case StatementType.ATOMIC:
                self._validate_atomic(stmt)
            case StatementType.WORK_GROUP_UNIFORM_LOAD:
                self._validate_work_group_uniform_load(stmt)
            case StatementType.CALL:
                self._validate_call(stmt)
            case StatementType.RAY_QUERY:
                self._validate_ray_query(stmt)
            case _:
                # Unknown statement type
                pass
    
    def _validate_emit(self, stmt: Statement) -> None:
        """Validate an emit statement."""
        if not hasattr(stmt, 'emit_range'):
            raise ValidationError("Emit statement missing range")
        
        # Validate that emitted expressions are in range
        start, end = stmt.emit_range
        if start < 0 or end > len(self.function.expressions):
            raise ValidationError(
                f"Emit range [{start}, {end}) out of bounds "
                f"(expressions={len(self.function.expressions)})"
            )
    
    def _validate_block(self, stmt: Statement) -> None:
        """Validate a block statement."""
        if not hasattr(stmt, 'block'):
            raise ValidationError("Block statement missing block")
        
        # Recursively validate all statements in block
        for sub_stmt in stmt.block:
            self.validate_statement(sub_stmt)
    
    def _validate_if(self, stmt: Statement) -> None:
        """Validate an if statement."""
        if not hasattr(stmt, 'if_condition'):
            raise ValidationError("If statement missing condition")
        
        # Validate condition expression handle
        cond_handle = stmt.if_condition
        if not isinstance(cond_handle, int):
            raise ValidationError(
                f"If condition must be int handle, got {type(cond_handle)}"
            )
        
        if cond_handle < 0 or cond_handle >= len(self.function.expressions):
            raise ValidationError(
                f"If condition handle {cond_handle} out of bounds"
            )
        
        # Validate accept block
        if hasattr(stmt, 'if_accept'):
            for sub_stmt in stmt.if_accept:
                self.validate_statement(sub_stmt)
        
        # Validate reject block if present
        if hasattr(stmt, 'if_reject') and stmt.if_reject:
            for sub_stmt in stmt.if_reject:
                self.validate_statement(sub_stmt)
    
    def _validate_switch(self, stmt: Statement) -> None:
        """Validate a switch statement."""
        if not hasattr(stmt, 'switch_selector'):
            raise ValidationError("Switch statement missing selector")
        
        selector_handle = stmt.switch_selector
        if not isinstance(selector_handle, int):
            raise ValidationError(
                f"Switch selector must be int handle, got {type(selector_handle)}"
            )
        
        if selector_handle < 0 or selector_handle >= len(self.function.expressions):
            raise ValidationError(
                f"Switch selector handle {selector_handle} out of bounds"
            )
        
        # Validate cases
        if hasattr(stmt, 'switch_cases'):
            for case in stmt.switch_cases:
                if hasattr(case, 'body'):
                    for sub_stmt in case.body:
                        self.validate_statement(sub_stmt)
    
    def _validate_loop(self, stmt: Statement) -> None:
        """Validate a loop statement."""
        # Validate loop body
        if hasattr(stmt, 'loop_body'):
            for sub_stmt in stmt.loop_body:
                self.validate_statement(sub_stmt)
        
        # Validate continuing block if present
        if hasattr(stmt, 'loop_continuing') and stmt.loop_continuing:
            for sub_stmt in stmt.loop_continuing:
                self.validate_statement(sub_stmt)
        
        # Validate break_if if present
        if hasattr(stmt, 'loop_break_if') and stmt.loop_break_if is not None:
            break_if_handle = stmt.loop_break_if
            if not isinstance(break_if_handle, int):
                raise ValidationError(
                    f"Loop break_if must be int handle, got {type(break_if_handle)}"
                )
            
            if break_if_handle < 0 or break_if_handle >= len(self.function.expressions):
                raise ValidationError(
                    f"Loop break_if handle {break_if_handle} out of bounds"
                )
    
    def _validate_break(self, stmt: Statement) -> None:
        """Validate a break statement."""
        # Break is always valid (context checking done elsewhere)
        pass
    
    def _validate_continue(self, stmt: Statement) -> None:
        """Validate a continue statement."""
        # Continue is always valid (context checking done elsewhere)
        pass
    
    def _validate_return(self, stmt: Statement) -> None:
        """Validate a return statement."""
        # Validate return value if present
        if hasattr(stmt, 'return_value') and stmt.return_value is not None:
            value_handle = stmt.return_value
            if not isinstance(value_handle, int):
                raise ValidationError(
                    f"Return value must be int handle, got {type(value_handle)}"
                )
            
            if value_handle < 0 or value_handle >= len(self.function.expressions):
                raise ValidationError(
                    f"Return value handle {value_handle} out of bounds"
                )
            
            # Check that function has a return type
            if self.function.result is None:
                raise ValidationError(
                    "Return statement with value in function with no return type"
                )
        else:
            # Check that function has no return type
            if self.function.result is not None:
                raise ValidationError(
                    "Return statement without value in function with return type"
                )
    
    def _validate_kill(self, stmt: Statement) -> None:
        """Validate a kill statement."""
        # Kill is always valid (fragment shader only, checked elsewhere)
        pass
    
    def _validate_barrier(self, stmt: Statement) -> None:
        """Validate a barrier statement."""
        # Barrier is valid (scope checking done elsewhere)
        pass
    
    def _validate_store(self, stmt: Statement) -> None:
        """Validate a store statement."""
        if not hasattr(stmt, 'store_pointer'):
            raise ValidationError("Store statement missing pointer")
        
        if not hasattr(stmt, 'store_value'):
            raise ValidationError("Store statement missing value")
        
        pointer_handle = stmt.store_pointer
        value_handle = stmt.store_value
        
        # Validate pointer handle
        if not isinstance(pointer_handle, int):
            raise ValidationError(
                f"Store pointer must be int handle, got {type(pointer_handle)}"
            )
        
        if pointer_handle < 0 or pointer_handle >= len(self.function.expressions):
            raise ValidationError(
                f"Store pointer handle {pointer_handle} out of bounds"
            )
        
        # Validate value handle
        if not isinstance(value_handle, int):
            raise ValidationError(
                f"Store value must be int handle, got {type(value_handle)}"
            )
        
        if value_handle < 0 or value_handle >= len(self.function.expressions):
            raise ValidationError(
                f"Store value handle {value_handle} out of bounds"
            )
    
    def _validate_image_store(self, stmt: Statement) -> None:
        """Validate an image store statement."""
        if not hasattr(stmt, 'image_store_image'):
            raise ValidationError("ImageStore statement missing image")
        
        image_handle = stmt.image_store_image
        if not isinstance(image_handle, int):
            raise ValidationError(
                f"ImageStore image must be int handle, got {type(image_handle)}"
            )
        
        if image_handle < 0 or image_handle >= len(self.function.expressions):
            raise ValidationError(
                f"ImageStore image handle {image_handle} out of bounds"
            )
    
    def _validate_atomic(self, stmt: Statement) -> None:
        """Validate an atomic statement."""
        if not hasattr(stmt, 'atomic_pointer'):
            raise ValidationError("Atomic statement missing pointer")
        
        pointer_handle = stmt.atomic_pointer
        if not isinstance(pointer_handle, int):
            raise ValidationError(
                f"Atomic pointer must be int handle, got {type(pointer_handle)}"
            )
        
        if pointer_handle < 0 or pointer_handle >= len(self.function.expressions):
            raise ValidationError(
                f"Atomic pointer handle {pointer_handle} out of bounds"
            )
    
    def _validate_work_group_uniform_load(self, stmt: Statement) -> None:
        """Validate a workgroup uniform load statement."""
        if not hasattr(stmt, 'wg_uniform_load_pointer'):
            raise ValidationError("WorkGroupUniformLoad statement missing pointer")
        
        pointer_handle = stmt.wg_uniform_load_pointer
        if not isinstance(pointer_handle, int):
            raise ValidationError(
                f"WorkGroupUniformLoad pointer must be int handle, got {type(pointer_handle)}"
            )
        
        if pointer_handle < 0 or pointer_handle >= len(self.function.expressions):
            raise ValidationError(
                f"WorkGroupUniformLoad pointer handle {pointer_handle} out of bounds"
            )
    
    def _validate_call(self, stmt: Statement) -> None:
        """Validate a function call statement."""
        if not hasattr(stmt, 'call_function'):
            raise ValidationError("Call statement missing function")
        
        func_handle = stmt.call_function
        if not isinstance(func_handle, int):
            raise ValidationError(
                f"Call function must be int handle, got {type(func_handle)}"
            )
        
        if func_handle < 0 or func_handle >= len(self.module.functions):
            raise ValidationError(
                f"Call function handle {func_handle} out of bounds "
                f"(functions={len(self.module.functions)})"
            )
        
        # Validate arguments
        if hasattr(stmt, 'call_arguments'):
            for i, arg_handle in enumerate(stmt.call_arguments):
                if not isinstance(arg_handle, int):
                    raise ValidationError(
                        f"Call argument {i} must be int handle, got {type(arg_handle)}"
                    )
                
                if arg_handle < 0 or arg_handle >= len(self.function.expressions):
                    raise ValidationError(
                        f"Call argument {i} handle {arg_handle} out of bounds"
                    )
    
    def _validate_ray_query(self, stmt: Statement) -> None:
        """Validate a ray query statement."""
        if not hasattr(stmt, 'ray_query_query'):
            raise ValidationError("RayQuery statement missing query")
        
        query_handle = stmt.ray_query_query
        if not isinstance(query_handle, int):
            raise ValidationError(
                f"RayQuery query must be int handle, got {type(query_handle)}"
            )
        
        if query_handle < 0 or query_handle >= len(self.function.expressions):
            raise ValidationError(
                f"RayQuery query handle {query_handle} out of bounds"
            )


__all__ = ['StatementValidator']
