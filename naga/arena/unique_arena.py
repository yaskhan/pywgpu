from typing import Generic, TypeVar, List, Dict, Tuple, Any
from .handle import Handle
from ..span import Span

T = TypeVar("T")


class UniqueArena(Generic[T]):
    """
    An arena that deduplicates items.
    """

    def __init__(self) -> None:
        self._data: List[T] = []
        self._span_info: List[Span] = []
        self._map: Dict[T, Handle[T]] = {}

    def fetch_or_append(self, item: T, span: Span) -> Handle[T]:
        if item in self._map:
            return self._map[item]

        index = len(self._data)
        handle = Handle(index)
        self._data.append(item)
        self._span_info.append(span)
        self._map[item] = handle
        return handle

    def get(self, handle: Handle[T]) -> T:
        return self._data[handle.index]

    def get_span(self, handle: Handle[T]) -> Span:
        return self._span_info[handle.index]
