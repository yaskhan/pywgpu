"""
Block terminator utilities.

This module provides utilities for ensuring blocks have proper return statements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Block, Statement


def ensure_block_returns(block: Block) -> None:
    """
    Ensure that the given block has return statements at the end of its control flow.
    
    Note: we don't want to blindly append a return statement
    to the end, because it may be either redundant or invalid,
    e.g. when the user already has returns in if/else branches.
    
    Args:
        block: The block to check and potentially modify
    """
    from ..ir import Statement
    from ..span import Span
    
    if not block:
        # Empty block - add return
        block.append((Statement.Return(value=None), Span()))
        return
    
    last_stmt = block[-1][0] if block else None
    
    if last_stmt is None:
        # Empty block
        block.append((Statement.Return(value=None), Span()))
    elif isinstance(last_stmt, Statement.Block):
        # Nested block
        ensure_block_returns(last_stmt.body)
    elif isinstance(last_stmt, Statement.If):
        # If statement - ensure both branches return
        ensure_block_returns(last_stmt.accept)
        ensure_block_returns(last_stmt.reject)
    elif isinstance(last_stmt, Statement.Switch):
        # Switch statement - ensure all non-fallthrough cases return
        for case in last_stmt.cases:
            if not case.fall_through:
                ensure_block_returns(case.body)
    elif isinstance(last_stmt, (Statement.Break, Statement.Continue, Statement.Return, Statement.Kill)):
        # Already has a terminator
        pass
    else:
        # No terminator - add return
        block.append((Statement.Return(value=None), Span()))


__all__ = [
    "ensure_block_returns",
]
