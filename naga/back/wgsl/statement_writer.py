"""WGSL Statement Writer

Converts NAGA IR statements to valid WGSL code.
Handles control flow, variable declarations, and assignments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from ...ir import Statement, StatementType, Barrier, AtomicFunction
from ...error import ShaderError
from .expression_writer import WGSLExpressionWriter

if TYPE_CHECKING:
    from ...ir.module import Module
    from ...ir.function import Function


class WGSLStatementWriter:
    """Writes NAGA IR statements as WGSL code."""
    
    def __init__(self, module: Module, names: dict[str, str], expression_writer: WGSLExpressionWriter):
        """Initialize statement writer.
        
        Args:
            module: The module being written
            names: Name mapping for identifiers
            expression_writer: The expression writer to use
        """
        self.module = module
        self.names = names
        self.expression_writer = expression_writer
        self.indent_level = 0
    
    def _indent(self) -> str:
        """Get indentation string based on current level."""
        return "    " * self.indent_level
    
    def write_block(self, block: List[Statement], indent_level: int = 1) -> str:
        """Write a block of statements.
        
        Args:
            block: List of statements to write
            indent_level: Current indentation level
            
        Returns:
            WGSL code for the block
        """
        self.indent_level = indent_level
        lines = []
        for stmt in block:
            lines.append(self.write_statement(stmt))
        return "".join(lines)
    
    def write_statement(self, stmt: Statement) -> str:
        """Write a single statement.
        
        Args:
            stmt: The statement to write
            
        Returns:
            WGSL code for the statement
        """
        indent = self._indent()
        
        match stmt.type:
            case StatementType.EMIT:
                # Emit is a NAGA IR concept, usually doesn't output code directly
                return ""
            
            case StatementType.BLOCK:
                self.indent_level += 1
                body = self.write_block(stmt.block, self.indent_level)
                self.indent_level -= 1
                return f"{indent}{{\n{body}{indent}}}\n"
            
            case StatementType.IF:
                return self._write_if(stmt, indent)
            
            case StatementType.SWITCH:
                return self._write_switch(stmt, indent)
            
            case StatementType.LOOP:
                return self._write_loop(stmt, indent)
            
            case StatementType.BREAK:
                return f"{indent}break;\n"
            
            case StatementType.CONTINUE:
                return f"{indent}continue;\n"
            
            case StatementType.RETURN:
                if hasattr(stmt, 'return_value') and stmt.return_value is not None:
                    value = self.expression_writer.write_expression(stmt.return_value)
                    return f"{indent}return {value};\n"
                else:
                    return f"{indent}return;\n"
            
            case StatementType.KILL:
                return f"{indent}discard;\n"  # WGSL's kill is discard
            
            case StatementType.BARRIER:
                return self._write_barrier(stmt.barrier, indent)
            
            case StatementType.STORE:
                pointer = self.expression_writer.write_expression(stmt.store_pointer)
                value = self.expression_writer.write_expression(stmt.store_value)
                return f"{indent}{pointer} = {value};\n"

            case StatementType.IMAGE_STORE:
                return self._write_image_store(stmt, indent)

            case StatementType.ATOMIC:
                return self._write_atomic(stmt, indent)

            case StatementType.IMAGE_ATOMIC:
                return self._write_image_atomic(stmt, indent)
            
            case StatementType.CALL:
                return self._write_call(stmt, indent)
            
            case _:
                raise ShaderError(f"Unsupported WGSL statement: {stmt.type}")
    
    def _write_if(self, stmt: Statement, indent: str) -> str:
        """Write an if statement."""
        condition = self.expression_writer.write_expression(stmt.if_condition)
        
        self.indent_level += 1
        accept_body = self.write_block(stmt.if_accept, self.indent_level)
        self.indent_level -= 1
        
        result = f"{indent}if ({condition}) {{\n{accept_body}{indent}}}"
        
        if hasattr(stmt, 'if_reject') and stmt.if_reject:
            self.indent_level += 1
            reject_body = self.write_block(stmt.if_reject, self.indent_level)
            self.indent_level -= 1
            result += f" else {{\n{reject_body}{indent}}}"
            
        result += "\n"
        return result
    
    def _write_switch(self, stmt: Statement, indent: str) -> str:
        """Write a switch statement."""
        selector = self.expression_writer.write_expression(stmt.switch_selector)
        
        result = f"{indent}switch ({selector}) {{\n"
        self.indent_level += 1
        
        for case in stmt.switch_cases:
            if case.value == "default":
                result += f"{self._indent()}default: {{\n"
            else:
                result += f"{self._indent()}case {case.value}: {{\n"
            
            self.indent_level += 1
            case_body = self.write_block(case.body, self.indent_level)
            self.indent_level -= 1
            
            result += f"{case_body}{self._indent()}}}\n"
            
        self.indent_level -= 1
        result += f"{indent}}}\n"
        return result
    
    def _write_loop(self, stmt: Statement, indent: str) -> str:
        """Write a loop statement."""
        result = f"{indent}loop {{\n"
        self.indent_level += 1
        
        body = self.write_block(stmt.loop_body, self.indent_level)
        result += body
        
        if hasattr(stmt, 'loop_continuing') and stmt.loop_continuing:
            result += f"{self._indent()}continuing {{\n"
            self.indent_level += 1
            
            continuing_body = self.write_block(stmt.loop_continuing, self.indent_level)
            result += continuing_body
            
            if hasattr(stmt, 'loop_break_if') and stmt.loop_break_if is not None:
                condition = self.expression_writer.write_expression(stmt.loop_break_if)
                result += f"{self._indent()}break if {condition};\n"
                
            self.indent_level -= 1
            result += f"{self._indent()}}}\n"
            
        self.indent_level -= 1
        result += f"{indent}}}\n"
        return result
    
    def _write_call(self, stmt: Statement, indent: str) -> str:
        """Write a function call statement."""
        func = self.module.functions[stmt.call_function]
        func_name = self.names.get(f"function_{stmt.call_function}", func.name or f"func_{stmt.call_function}")
        
        args = [
            self.expression_writer.write_expression(arg)
            for arg in stmt.call_arguments
        ]
        
        result = f"{indent}{func_name}({', '.join(args)});\n"
        return result

    def _write_barrier(self, barrier: Barrier, indent: str) -> str:
        """Write a barrier statement."""
        if barrier == Barrier.WORK_GROUP:
            return f"{indent}workgroupBarrier();\n"
        elif barrier == Barrier.STORAGE:
            return f"{indent}storageBarrier();\n"
        else:
            return f"{indent}textureBarrier();\n"

    def _write_image_store(self, stmt: Statement, indent: str) -> str:
        """Write an image store statement."""
        image = self.expression_writer.write_expression(stmt.image_store_image)
        coord = self.expression_writer.write_expression(stmt.image_store_coordinate)
        value = self.expression_writer.write_expression(stmt.image_store_value)
        # textureStore(image, coord, value)
        return f"{indent}textureStore({image}, {coord}, {value});\n"

    def _write_atomic(self, stmt: Statement, indent: str) -> str:
        """Write an atomic statement."""
        pointer = self.expression_writer.write_expression(stmt.atomic_pointer)
        value = self.expression_writer.write_expression(stmt.atomic_value)

        fun_map = {
            AtomicFunction.ADD: "atomicAdd",
            AtomicFunction.SUBTRACT: "atomicSub",
            AtomicFunction.AND: "atomicAnd",
            AtomicFunction.EXCLUSIVE_OR: "atomicXor",
            AtomicFunction.INCLUSIVE_OR: "atomicOr",
            AtomicFunction.MIN: "atomicMin",
            AtomicFunction.MAX: "atomicMax",
            AtomicFunction.EXCHANGE: "atomicExchange",
        }

        fun_name = fun_map.get(stmt.atomic_fun)
        if fun_name is None:
            raise ShaderError(f"Unsupported WGSL atomic function: {stmt.atomic_fun}")

        call = f"{fun_name}({pointer}, {value})"
        if getattr(stmt, "atomic_result", None) is not None:
            res_name = f"_e{stmt.atomic_result}"
            return f"{indent}let {res_name} = {call};\n"

        return f"{indent}{call};\n"

    def _write_image_atomic(self, stmt: Statement, indent: str) -> str:
        """Write an image atomic statement."""
        image = self.expression_writer.write_expression(stmt.image_atomic_image)
        coord = self.expression_writer.write_expression(stmt.image_atomic_coordinate)
        value = self.expression_writer.write_expression(stmt.image_atomic_value)

        fun_map = {
            AtomicFunction.ADD: "Add",
            AtomicFunction.SUBTRACT: "Sub",
            AtomicFunction.AND: "And",
            AtomicFunction.EXCLUSIVE_OR: "Xor",
            AtomicFunction.INCLUSIVE_OR: "Or",
            AtomicFunction.MIN: "Min",
            AtomicFunction.MAX: "Max",
            AtomicFunction.EXCHANGE: "Exchange",
        }

        suffix = fun_map.get(stmt.image_atomic_fun)
        if suffix is None:
            raise ShaderError(f"Unsupported WGSL image atomic function: {stmt.image_atomic_fun}")

        array_index = getattr(stmt, "image_atomic_array_index", None)
        if array_index is not None:
            array_index_expr = self.expression_writer.write_expression(array_index)
            call = f"textureAtomic{suffix}({image}, {coord}, {array_index_expr}, {value})"
        else:
            call = f"textureAtomic{suffix}({image}, {coord}, {value})"

        return f"{indent}{call};\n"


__all__ = ['WGSLStatementWriter']
