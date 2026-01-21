"""
Arena implementation for storing shader translation components.

This module provides the Arena class, which stores a series of values indexed by
Handle values. This improves translator performance and reduces memory usage
by providing strongly-typed handles instead of direct references.
"""
from typing import Generic, TypeVar, List, Tuple, Iterator, Optional, Callable, MutableSequence
from .handle import Handle, BadHandle
from .range import Range, BadRangeError
from ..span import Span

T = TypeVar("T")


class Arena(Generic[T]):
    """
    An arena holding some kind of component (e.g., type, constant,
    instruction, etc.) that can be referenced.
    
    Adding new items to the arena produces a strongly-typed Handle.
    The arena can be indexed using the given handle to obtain
    a reference to the stored item.
    """

    def __init__(self) -> None:
        """Create a new arena with no initial capacity allocated."""
        self._data: List[T] = []
        self._span_info: List[Span] = []

    def __len__(self) -> int:
        """Return the current number of items stored in this arena."""
        return len(self._data)

    def is_empty(self) -> bool:
        """Return True if the arena contains no elements."""
        return len(self._data) == 0

    def into_inner(self) -> List[T]:
        """Extracts the inner list."""
        return self._data.copy()

    def iter(self) -> Iterator[Tuple[Handle[T], T]]:
        """
        Return an iterator over the items stored in this arena, returning both
        the item's handle and a reference to it.
        """
        for i, item in enumerate(self._data):
            yield Handle.from_usize(i), item

    def iter_mut_span(self) -> Iterator[Tuple[Handle[T], T, Span]]:
        """
        Return an iterator over the items stored in this arena, returning both
        the item's handle, a mutable reference to it, and the span.
        """
        for i, (item, span) in enumerate(zip(self._data, self._span_info)):
            yield Handle.from_usize(i), item, span

    def drain(self) -> Iterator[Tuple[Handle[T], T, Span]]:
        """
        Drain the arena, returning an iterator over the items stored.
        """
        data = self._data.copy()
        span_info = self._span_info.copy()
        self._data.clear()
        self._span_info.clear()
        for i, (item, span) in enumerate(zip(data, span_info)):
            yield Handle.from_usize(i), item, span

    def iter_mut(self) -> Iterator[Tuple[Handle[T], T]]:
        """
        Return an iterator over the items stored in this arena,
        returning both the item's handle and a mutable reference to it.
        """
        for i, item in enumerate(self._data):
            yield Handle.from_usize(i), item

    def append(self, value: T, span: Span) -> Handle[T]:
        """
        Add a new value to the arena, returning a typed handle.
        """
        index = len(self._data)
        self._data.append(value)
        self._span_info.append(span)
        return Handle.from_usize(index)

    def fetch_if(self, fun: Callable[[T], bool]) -> Optional[Handle[T]]:
        """
        Fetch a handle to an existing type.
        """
        for i, item in enumerate(self._data):
            if fun(item):
                return Handle.from_usize(i)
        return None

    def fetch_if_or_append(
        self, value: T, span: Span, fun: Callable[[T, T], bool]
    ) -> Handle[T]:
        """
        Add a value with a custom check for uniqueness:
        returns a handle pointing to an existing element if the check succeeds,
        or adds a new element otherwise.
        """
        for i, item in enumerate(self._data):
            if fun(item, value):
                return Handle.from_usize(i)
        return self.append(value, span)

    def fetch_or_append(self, value: T, span: Span) -> Handle[T]:
        """
        Add a value with a check for uniqueness, where the check is plain comparison.
        """
        return self.fetch_if_or_append(value, span, lambda a, b: a == b)

    def try_get(self, handle: Handle[T]) -> Optional[T]:
        """
        Get a reference to an element in the arena.
        """
        if handle.index < len(self._data):
            return self._data[handle.index]
        return None

    def get_mut(self, handle: Handle[T]) -> T:
        """
        Get a mutable reference to an element in the arena.
        """
        if handle.index >= len(self._data):
            raise BadHandle(type(self).__name__, handle.index)
        return self._data[handle.index]

    def range_from(self, old_length: int) -> Range[T]:
        """
        Get the range of handles from a particular number of elements to the end.
        """
        return Range(old_length, len(self._data))

    def clear(self) -> None:
        """Clears the arena keeping all allocations."""
        self._data.clear()
        self._span_info.clear()

    def get_span(self, handle: Handle[T]) -> Span:
        """
        Get the span associated with a handle.
        """
        if handle.index < len(self._span_info):
            return self._span_info[handle.index]
        return Span(0, 0)

    def check_contains_handle(self, handle: Handle[T]) -> None:
        """
        Assert that handle is valid for this arena.
        
        Raises:
            BadHandle: if handle is not valid
        """
        if handle.index >= len(self._data):
            raise BadHandle(type(self).__name__, handle.index)

    def check_contains_range(self, range: Range[T]) -> None:
        """
        Assert that range is valid for this arena.
        
        Raises:
            BadRangeError: if range is not valid
        """
        start, end = range.inner
        if start > end:
            raise BadRangeError(type(self).__name__, start, end)
        
        # Empty ranges are tolerated: they can be produced by compaction.
        if start == end:
            return
        
        if end > len(self._data):
            raise BadRangeError(type(self).__name__, start, end)

    def retain_mut(self, predicate: Callable[[Handle[T], T], bool]) -> None:
        """
        Retain only elements for which the predicate returns True.
        """
        index = 0
        retained = 0
        for i, item in enumerate(self._data):
            handle = Handle.from_usize(i)
            if predicate(handle, item):
                if retained != i:
                    self._data[retained] = self._data[i]
                    self._span_info[retained] = self._span_info[i]
                retained += 1
            index += 1
        
        self._data = self._data[:retained]
        self._span_info = self._span_info[:retained]

    def __getitem__(self, handle: Handle[T]) -> T:
        """Index the arena by handle."""
        if handle.index >= len(self._data):
            raise IndexError(f"Handle {handle} is out of bounds")
        return self._data[handle.index]

    def __setitem__(self, handle: Handle[T], value: T) -> None:
        """Set the value at handle."""
        if handle.index >= len(self._data):
            raise IndexError(f"Handle {handle} is out of bounds")
        self._data[handle.index] = value

    def __getitem__(self, range: Range[T]) -> List[T]:
        """Index the arena by range."""
        start, end = range.inner
        if start > end or end > len(self._data):
            raise IndexError(f"Range {range} is out of bounds")
        return self._data[start:end]

    def __repr__(self) -> str:
        return f"Arena(len={len(self._data)})"
