from typing import Generic, TypeVar, Iterator
from .handle import Handle

T = TypeVar('T')

class Range(Generic[T]):
    """
    A range of handles.
    """
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    @classmethod
    def from_bounds(cls, first: Handle[T], last: Handle[T]) -> 'Range[T]':
        return cls(first.index, last.index + 1)

    def __iter__(self) -> Iterator[Handle[T]]:
        for i in range(self.start, self.end):
            yield Handle(i)
