"""
Lowering contexts for WGSL to IR.
"""

from typing import Any, Dict, List
from ....ir import Module, Function, Expression, Statement, Block


class ExpressionContext:
    """
    Context for lowering expressions.
    
    Tracks local variables, types, and other expression-specific state.
    """
    
    def __init__(self, module: Module):
        """
        Initialize expression context.
        
        Args:
            module: The module being built
        """
        self.module = module
        self.local_table: Dict[Any, Any] = {}
        self.expression_arena: List[Expression] = []
    
    def add_expression(self, expr: Expression) -> Any:
        """
        Add an expression to the arena.
        
        Args:
            expr: Expression to add
            
        Returns:
            Expression handle
        """
        handle = len(self.expression_arena)
        self.expression_arena.append(expr)
        return handle


class StatementContext:
    """
    Context for lowering statements.
    
    Tracks control flow, local variables, and statement-specific state.
    """
    
    def __init__(self, function: Function):
        """
        Initialize statement context.
        
        Args:
            function: The function being built
        """
        self.function = function
        self.local_table: Dict[Any, Any] = {}
        self.block_stack: List[Block] = [Block.new()]
    
    def add_expression(self, expr: Expression) -> Any:
        """Add an expression to the function's arena."""
        return self.function.add_expression(expr)
    
    def add_statement(self, stmt: Statement) -> None:
        """
        Add a statement to the current block.
        
        Args:
            stmt: Statement to add
        """
        from ....span import Span
        self.block_stack[-1].push(stmt, Span())
    
    def push_block(self) -> None:
        """Start a new block."""
        self.block_stack.append(Block.new())
    
    def pop_block(self) -> Block:
        """
        End the current block.
        
        Returns:
            The popped block
        """
        return self.block_stack.pop()
