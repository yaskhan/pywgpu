from typing import List, Any, Iterator, Optional, Tuple
from ..span import Span


class Block:
    """
    A basic block of statements.
    """

    def __init__(self) -> None:
        self.body: List[Any] = []  # List[Statement]
        self.span_info: List[Span] = []

    @classmethod
    def new(cls) -> "Block":
        """Create a new empty block."""
        return cls()

    @classmethod
    def from_vec(cls, body: List[Any]) -> "Block":
        """Create a block from a vector of statements."""
        block = cls()
        block.body = body
        block.span_info = [Span() for _ in body]
        return block

    @classmethod
    def with_capacity(cls, capacity: int) -> "Block":
        """Create a block with a pre-allocated capacity."""
        _ = capacity
        block = cls()
        block.body = []
        block.span_info = []
        # Python lists don't have reserve(), but we can pre-allocate if needed.
        return block

    def push(self, statement: Any, span: Span) -> None:
        """Push a statement with its span."""
        self.body.append(statement)
        self.span_info.append(span)

    def append(self, statement: Any, span: Span) -> None:
        """Append a statement with its span."""
        self.push(statement, span)

    def extend(self, item: Optional[Tuple[Any, Span]]) -> None:
        """Extend the block with an optional statement-span pair."""
        if item is not None:
            statement, span = item
            self.push(statement, span)

    def extend_block(self, other: "Block") -> None:
        """Extend this block with another block's contents."""
        self.span_info.extend(other.span_info)
        self.body.extend(other.body)

    def append_block(self, other: "Block") -> None:
        """Append another block to this block."""
        self.span_info.extend(other.span_info)
        self.body.extend(other.body)

    def cull(self, start: int, end: int) -> None:
        """Remove statements in the given range."""
        del self.span_info[start:end]
        del self.body[start:end]

    def splice(self, start: int, end: int, other: "Block") -> None:
        """Splice another block into this block at the given range."""
        self.span_info[start:end] = other.span_info
        self.body[start:end] = other.body

    def span_into_iter(self) -> Iterator[Tuple[Any, Span]]:
        """Return an iterator over statement-span pairs."""
        return zip(self.body, self.span_info)

    def span_iter(self) -> Iterator[Tuple[Any, Span]]:
        """Return an iterator over statement-span pair references."""
        return zip(self.body, self.span_info)

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over statements."""
        return iter(self.body)

    def __len__(self) -> int:
        """Return the number of statements in the block."""
        return len(self.body)

    def is_empty(self) -> bool:
        """Check if the block is empty."""
        return len(self.body) == 0

    def __getitem__(self, index: int) -> Any:
        """Get a statement by index."""
        return self.body[index]

    def __setitem__(self, index: int, statement: Any) -> None:
        """Set a statement by index."""
        self.body[index] = statement
