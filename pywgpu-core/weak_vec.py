"""
Weak reference vector utility.

This module provides an optimized container for weak references that
minimizes reallocations by dropping older elements that no longer have
strong references to them.

The weak vector is used to store references to resources that may be
destroyed at any time, allowing for efficient cleanup of stale references.
"""

from __future__ import annotations

from typing import Any, Generic, Iterator, TypeVar

from weakref import ref, ReferenceType


T = TypeVar("T")


class WeakVec(Generic[T]):
    """
    An optimized container for weak references of T.
    
    This container minimizes reallocations by dropping older elements that
    no longer have strong references to them.
    
    Attributes:
        inner: The inner list of weak references.
    """

    def __init__(self) -> None:
        """Initialize the weak vector."""
        self.inner: list[ReferenceType[T]] = []

    def push(self, value: ReferenceType[T]) -> None:
        """
        Push a new element to this collection.
        
        If the inner list needs to be reallocated, we will first drop
        older elements that no longer have strong references to them.
        
        Args:
            value: The weak reference to push.
        """
        if len(self.inner) == self.inner.capacity():
            # Iterating backwards has the advantage that we don't do more work than we have to.
            for i in range(len(self.inner) - 1, -1, -1):
                if self.inner[i]() is None:
                    self.inner.pop(i)
            
            # Make sure our capacity is twice the number of live elements.
            # Leaving some spare capacity ensures that we won't re-scan immediately.
            self.inner = self.inner[:len(self.inner)]
            self.inner.reserve_exact(len(self.inner))
        
        self.inner.append(value)

    def __iter__(self) -> Iterator[ReferenceType[T]]:
        """Iterate over the weak references."""
        return iter(self.inner)

    def __len__(self) -> int:
        """Get the number of weak references."""
        return len(self.inner)


class WeakVecIter(Generic[T]):
    """
    Iterator for WeakVec.
    
    Attributes:
        inner: The inner iterator.
    """

    def __init__(self, inner: list[ReferenceType[T]]) -> None:
        """Initialize the iterator."""
        self.inner = iter(inner)

    def __next__(self) -> ReferenceType[T]:
        """Get the next weak reference."""
        return next(self.inner)
