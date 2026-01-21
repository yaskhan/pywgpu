from typing import Generic, TypeVar, Iterator, Tuple, Optional
from .handle import Handle, BadHandle

T = TypeVar("T")


class BadRangeError(Exception):
    """
    Error raised when a handle range is invalid.
    """

    def __init__(self, kind: str, start: int, end: int):
        self.kind = kind
        self.start = start
        self.end = end
        super().__init__(f"Handle range [{start}..{end}] of {kind} is either not present, or inaccessible yet")


class Range(Generic[T]):
    """
    A strongly typed range of handles.
    """

    def __init__(self, start: int, end: int) -> None:
        self._inner = (start, end)

    @property
    def inner(self) -> Tuple[int, int]:
        """Returns the inner range as a tuple (start, end)."""
        return self._inner

    @property
    def start(self) -> int:
        """Returns the start of the range."""
        return self._inner[0]

    @property
    def end(self) -> int:
        """Returns the end of the range (exclusive)."""
        return self._inner[1]

    def erase_type(self) -> "Range[None]":
        """Erases the type parameter."""
        return Range(self._inner[0], self._inner[1])

    @classmethod
    def new_from_bounds(cls, first: Handle[T], last: Handle[T]) -> "Range[T]":
        """Return a range enclosing handles first through last, inclusive."""
        return cls(first.index, last.index + 1)

    @classmethod
    def full_range_from_size(cls, size: int) -> "Range[T]":
        """Return a range covering all handles with indices from 0 to size."""
        return cls(0, size)

    def first_and_last(self) -> Optional[Tuple[Handle[T], Handle[T]]]:
        """
        Return the first and last handles included in self.

        If self is an empty range, there are no handles included, so return None.
        """
        if self._inner[0] < self._inner[1]:
            return (
                Handle.from_usize(self._inner[0]),
                Handle.from_usize(self._inner[1] - 1),
            )
        return None

    def index_range(self) -> Tuple[int, int]:
        """Return the index range covered by self."""
        return self._inner

    @classmethod
    def from_index_range(cls, inner: Tuple[int, int], arena_len: int) -> "Range[T]":
        """
        Construct a Range that covers the indices in inner.

        Args:
            inner: A tuple (start, end) representing the range
            arena_len: The length of the arena to validate against
        """
        start, end = inner
        if start > end:
            raise ValueError(f"Start {start} must be <= end {end}")
        if end > arena_len:
            raise ValueError(f"End {end} must be <= arena length {arena_len}")
        return cls(start, end)

    def __iter__(self) -> Iterator[Handle[T]]:
        for i in range(self._inner[0], self._inner[1]):
            yield Handle(i)

    def __repr__(self) -> str:
        return f"[{self._inner[0]}..{self._inner[1]}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Range):
            return self._inner == other._inner
        return False

    def __hash__(self) -> int:
        return hash(self._inner)
