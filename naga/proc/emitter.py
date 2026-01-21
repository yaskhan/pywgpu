"""
Helper class to emit expressions.

This module provides the Emitter class for managing expression emission
in Naga IR generation.
"""

from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass

from ..arena import Arena, Range
from ..ir import Expression, Statement
from ..span import Span


@dataclass
class Emitter:
    """
    Helper class to emit expressions.
    
    The Emitter tracks a range of expressions that need to be emitted
    as a Statement.Emit when code generation is complete.
    
    Attributes:
        start_len: The length of the expression arena when emission started
    """
    start_len: Optional[int] = None
    
    def start(self, arena: Arena[Expression]) -> None:
        """
        Start emitting expressions.
        
        Args:
            arena: The expression arena
            
        Raises:
            RuntimeError: If emitting has already started
        """
        if self.start_len is not None:
            raise RuntimeError("Emitting has already started!")
        self.start_len = len(arena)
    
    def is_running(self) -> bool:
        """
        Check if emission is currently running.
        
        Returns:
            True if emission is running
        """
        return self.start_len is not None
    
    def finish(
        self,
        arena: Arena[Expression],
    ) -> Optional[Tuple[Statement, Span]]:
        """
        Finish emitting expressions.
        
        If any expressions were added to the arena since start() was called,
        returns an Emit statement covering those expressions. Otherwise returns None.
        
        Args:
            arena: The expression arena
            
        Returns:
            A tuple of (Statement.Emit, Span) if expressions were emitted, None otherwise
            
        Raises:
            RuntimeError: If emission was not started
        """
        if self.start_len is None:
            raise RuntimeError("Emitting was not started!")
        
        start_len = self.start_len
        self.start_len = None
        
        if start_len != len(arena):
            span = Span()
            expr_range = arena.range_from(start_len)
            
            for handle in expr_range:
                span.subsume(arena.get_span(handle))
            
            return (Statement.Emit(expr_range), span)
        else:
            return None


__all__ = [
    "Emitter",
]
