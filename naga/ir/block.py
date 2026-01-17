from typing import List, Any
from ..span import Span

class Block:
    """
    A basic block of statements.
    """
    def __init__(self) -> None:
        self.body: List[Any] = [] # List[Statement]
        self.span_info: List[Span] = []

    def append(self, statement: Any, span: Span) -> None:
        self.body.append(statement)
        self.span_info.append(span)

    def is_empty(self) -> bool:
        return len(self.body) == 0
