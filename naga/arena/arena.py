"""
Arena implementation for storing shader translation components.

This module provides the Arena class, which stores a series of values indexed by
Handle values. This improves translator performance and reduces memory usage
by providing strongly-typed handles instead of direct references.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Iterator, Tuple, List, Callable, Optional
from .handle import Handle
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
        """Returns the current number of items stored in this arena."""
        return len(self._data)

    def is_empty(self) -> bool:
        """Returns True if the arena contains no elements."""
        return len(self._data) == 0

    def iter(self) -> Iterator[Tuple[Handle[T], T]]:
        """
        Returns an iterator over the items stored in this arena, returning both
        the item's handle and a reference to it.
        """
        for i, item in enumerate(self._data):
            yield Handle(i), item

    def iter_mut(self) -> Iterator[Tuple[Handle[T], T]]:
        """
        Returns an iterator over the items stored in this arena, returning both
        the item's handle and a mutable reference to it.
        """
        for i, item in enumerate(self._data):
            # We need to yield a copy for now since we can't easily yield mutable refs
            # in a way that satisfies the borrow checker in pure Python
            yield Handle(i), item

    def append(self, value: T, span: Span) -> Handle[T]:
        """Adds a new value to the arena, returning a typed handle."""
        index = len(self._data)
        self._data.append(value)
        self._span_info.append(span)
        return Handle(index)

    def fetch_if(self, predicate: Callable[[T], bool]) -> Optional[Handle[T]]:
        """Fetch a handle to an existing item that matches the predicate."""
        for i, item in enumerate(self._data):
            if predicate(item):
                return Handle(i)
        return None

    def fetch_if_or_append(
        self, value: T, span: Span, predicate: Callable[[T, T], bool]
    ) -> Handle[T]:
        """
        Adds a value with a custom check for uniqueness:
        returns a handle pointing to an existing element if the check succeeds,
        or adds a new element otherwise.
        """
        for i, existing in enumerate(self._data):
            if predicate(existing, value):
                return Handle(i)
        return self.append(value, span)

    def fetch_or_append(self, value: T, span: Span) -> Handle[T]:
        """Adds a value with a check for uniqueness, using equality comparison."""
        return self.fetch_if_or_append(value, span, lambda a, b: a == b)

    def try_get(self, handle: Handle[T]) -> Optional[T]:
        """Try to get a reference to an element, returns None if handle is invalid."""
        try:
            return self._data[handle.index]
        except IndexError:
            return None

    def get(self, handle: Handle[T]) -> T:
        """Get a reference to an element in the arena."""
        return self._data[handle.index]

    def get_mut(self, handle: Handle[T]) -> T:
        """Get a mutable reference to an element in the arena."""
        return self._data[handle.index]

    def get_span(self, handle: Handle[T]) -> Span:
        """Get the source span associated with a handle."""
        try:
            return self._span_info[handle.index]
        except IndexError:
            return Span()

    def clear(self) -> None:
        """Clears the arena keeping all allocations."""
        self._data.clear()
        self._span_info.clear()

    def __getitem__(self, handle: Handle[T]) -> T:
        """Allow indexing with handles: arena[handle]"""
        return self._data[handle.index]

    def __setitem__(self, handle: Handle[T], value: T) -> None:
        """Allow setting values by handle: arena[handle] = value"""
        self._data[handle.index] = value
