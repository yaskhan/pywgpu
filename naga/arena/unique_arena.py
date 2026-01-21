from typing import Generic, TypeVar, List, Dict, Tuple, Any, Iterator, Optional
from .handle import Handle, BadHandle
from ..span import Span

T = TypeVar("T")


class UniqueArenaDrain(Generic[T]):
    """
    An iterator that drains elements from a UniqueArena.
    """

    def __init__(self, data: List[T], span_info: List[Span]) -> None:
        self._data = data
        self._span_info = span_info
        self._index = 0

    def __iter__(self) -> Iterator[Tuple[Handle[T], T, Span]]:
        return self

    def __next__(self) -> Tuple[Handle[T], T, Span]:
        if self._index >= len(self._data):
            raise StopIteration
        handle = Handle.from_usize(self._index)
        item = self._data[self._index]
        span = self._span_info[self._index]
        self._index += 1
        return handle, item, span


class UniqueArena(Generic[T]):
    """
    An arena whose elements are guaranteed to be unique.

    A UniqueArena holds a set of unique values of type T, each with an
    associated Span. Inserting a value returns a Handle<T>, which can be
    used to index the UniqueArena and obtain shared access to the T element.
    Access via a Handle is an array lookup - no hash lookup is necessary.

    The element type must implement __eq__ and __hash__. Insertions of
    equivalent elements, according to __eq__, all return the same Handle.
    """

    def __init__(self) -> None:
        self._data: List[T] = []
        self._span_info: List[Span] = []
        self._map: Dict[T, Handle[T]] = {}

    def __len__(self) -> int:
        """Return the current number of items stored in this arena."""
        return len(self._data)

    def is_empty(self) -> bool:
        """Return True if the arena contains no elements."""
        return len(self._data) == 0

    def clear(self) -> None:
        """Clears the arena, keeping all allocations."""
        self._data.clear()
        self._span_info.clear()
        self._map.clear()

    def get_span(self, handle: Handle[T]) -> Span:
        """
        Return the span associated with handle.

        If a value has been inserted multiple times, the span returned is the
        one provided with the first insertion.
        """
        if handle.index < len(self._span_info):
            return self._span_info[handle.index]
        return Span(0, 0)

    def drain_all(self) -> UniqueArenaDrain[T]:
        """Drain all elements from the arena."""
        data = self._data.copy()
        span_info = self._span_info.copy()
        self._data.clear()
        self._span_info.clear()
        self._map.clear()
        return UniqueArenaDrain(data, span_info)

    def iter(self) -> Iterator[Tuple[Handle[T], T]]:
        """
        Return an iterator over the items stored in this arena, returning both
        the item's handle and a reference to it.
        """
        for i, item in enumerate(self._data):
            yield Handle.from_usize(i), item

    def insert(self, value: T, span: Span) -> Handle[T]:
        """
        Insert a new value into the arena.

        Return a Handle<T>, which can be used to index this arena to get a
        shared reference to the element.

        If this arena already contains an element that is equal to value,
        return a Handle to the existing element, and drop value.

        If value is inserted into the arena, associate span with it.
        An element's span can be retrieved with the get_span method.
        """
        if value in self._map:
            return self._map[value]

        index = len(self._data)
        handle = Handle.from_usize(index)
        self._data.append(value)
        self._span_info.append(span)
        self._map[value] = handle
        return handle

    def replace(self, old: Handle[T], new: T) -> None:
        """
        Replace an old value with a new value.

        Raises:
            ValueError: if the old value is not in the arena
            ValueError: if the new value already exists in the arena
        """
        if old.index >= len(self._data):
            raise ValueError(f"Old handle {old} is not in the arena")
        if new in self._map:
            raise ValueError(f"New value {new} already exists in the arena")

        old_value = self._data[old.index]
        if old_value in self._map:
            del self._map[old_value]

        self._data[old.index] = new
        self._map[new] = old

    def get(self, value: T) -> Optional[Handle[T]]:
        """
        Return this arena's handle for value, if present.

        If this arena already contains an element equal to value,
        return its handle. Otherwise, return None.
        """
        return self._map.get(value)

    def get_handle(self, handle: Handle[T]) -> Optional[T]:
        """
        Return this arena's value at handle, if that is a valid handle.
        """
        if handle.index < len(self._data):
            return self._data[handle.index]
        return None

    def check_contains_handle(self, handle: Handle[T]) -> None:
        """
        Assert that handle is valid for this arena.

        Raises:
            BadHandle: if handle is not valid
        """
        if handle.index >= len(self._data):
            raise BadHandle(type(self).__name__, handle.index)

    def __getitem__(self, handle: Handle[T]) -> T:
        """Index the arena by handle."""
        if handle.index >= len(self._data):
            raise IndexError(f"Handle {handle} is out of bounds")
        return self._data[handle.index]

    def __repr__(self) -> str:
        return f"UniqueArena(len={len(self._data)})"
