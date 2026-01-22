"""
Lowering contexts for WGSL to IR.
"""

from typing import Any, Dict, List
from ...ir import Module, Function, Expression, Statement


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
        self.block_stack: List[List[Statement]] = [[]]
    
    def add_statement(self, stmt: Statement) -> None:
        """
        Add a statement to the current block.
        
        Args:
            stmt: Statement to add
        """
        self.block_stack[-1].append(stmt)
    
    def push_block(self) -> None:
        """Start a new block."""
        self.block_stack.append([])
    
    def pop_block(self) -> List[Statement]:
        """
        End the current block.
        
        Returns:
            Statements in the block
        """
        return self.block_stack.pop()
