"""WGSL Statement Writer

Converts NAGA IR statements to valid WGSL code.
Handles control flow, variable declarations, and assignments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from ...ir import Statement, StatementType
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
                # Basic workgroup barrier
                return f"{indent}workgroupBarrier();\n"
            
            case StatementType.STORE:
                pointer = self.expression_writer.write_expression(stmt.store_pointer)
                value = self.expression_writer.write_expression(stmt.store_value)
                return f"{indent}{pointer} = {value};\n"
            
            case StatementType.CALL:
                return self._write_call(stmt, indent)
            
            case _:
                return f"{indent}// TODO: Implement {stmt.type}\n"
    
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


__all__ = ['WGSLStatementWriter']
