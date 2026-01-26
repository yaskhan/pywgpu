"""
Lock types that observe lock acquisition order.

This module provides instrumented lock types for observing lock acquisition
patterns in wgpu-core. This is used for debugging and deadlock detection.

In the Python implementation, this is a simplified version that can be
extended with observability features as needed.
"""

from __future__ import annotations

import threading
from typing import Any, Generic, Optional, TypeVar

from .rank import LockRank

T = TypeVar("T")

# Type alias for rank data
RankData = Optional["HeldLock"]


class HeldLock:
    """
    Information about a currently held lock.

    Attributes:
        rank: The lock's rank.
        location: Where we acquired the lock (file:line).
    """

    def __init__(self, rank: LockRank, location: str) -> None:
        """
        Initialize held lock information.

        Args:
            rank: The lock's rank.
            location: The acquisition location.
        """
        self.rank = rank
        self.location = location


class Mutex(Generic[T]):
    """
    A Mutex instrumented for lock acquisition order observation.

    This is a wrapper around a threading.Lock, along with its rank in
    the wgpu_core lock ordering.

    Attributes:
        inner: The underlying lock.
        rank: The lock's rank.
    """

    def __init__(self, rank: LockRank, value: T) -> None:
        """
        Create a new mutex.

        Args:
            rank: The lock's rank.
            value: The initial value.
        """
        self.inner = threading.Lock()
        self.rank = rank
        self.value = value

    def lock(self) -> "MutexGuard[T]":
        """
        Acquire the lock.

        Returns:
            A guard for the locked mutex.
        """
        self.inner.acquire()
        return MutexGuard(self)

    def into_inner(self) -> T:
        """
        Consume the mutex and return the inner value.

        Returns:
            The inner value.
        """
        return self.value


class MutexGuard(Generic[T]):
    """
    A guard produced by locking a Mutex.

    Attributes:
        mutex: The mutex that was locked.
    """

    def __init__(self, mutex: Mutex[T]) -> None:
        """
        Initialize the guard.

        Args:
            mutex: The mutex that was locked.
        """
        self.mutex = mutex

    def __enter__(self) -> T:
        """Context manager entry."""
        return self.mutex.value

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.mutex.inner.release()

    def __del__(self) -> None:
        """Release the lock when the guard is dropped."""
        try:
            self.mutex.inner.release()
        except RuntimeError:
            pass


class RwLock(Generic[T]):
    """
    An RwLock instrumented for lock acquisition order observation.

    This is a wrapper around a threading.RLock, along with its rank in
    the wgpu_core lock ordering.

    Attributes:
        inner: The underlying lock.
        rank: The lock's rank.
    """

    def __init__(self, rank: LockRank, value: T) -> None:
        """
        Create a new RwLock.

        Args:
            rank: The lock's rank.
            value: The initial value.
        """
        self.inner = threading.RLock()
        self.rank = rank
        self.value = value

    def read(self) -> "RwLockReadGuard[T]":
        """
        Acquire a read lock.

        Returns:
            A read guard.
        """
        self.inner.acquire()
        return RwLockReadGuard(self)

    def write(self) -> "RwLockWriteGuard[T]":
        """
        Acquire a write lock.

        Returns:
            A write guard.
        """
        self.inner.acquire()
        return RwLockWriteGuard(self)

    def force_unlock_read(self, data: RankData) -> None:
        """
        Force unlock a read guard.

        This is unsafe and should only be used in very specific cases.

        Args:
            data: The rank data for the lock.
        """
        try:
            self.inner.release()
        except RuntimeError:
            pass


class RwLockReadGuard(Generic[T]):
    """
    A read guard produced by locking RwLock for reading.

    Attributes:
        lock: The RwLock that was locked.
    """

    def __init__(self, lock: RwLock[T]) -> None:
        """
        Initialize the read guard.

        Args:
            lock: The RwLock that was locked.
        """
        self.lock = lock

    @staticmethod
    def forget(guard: "RwLockReadGuard[T]") -> RankData:
        """
        Forget the guard, leaving the lock in a locked state.

        Args:
            guard: The guard to forget.

        Returns:
            The rank data for the lock.
        """
        # In Python, we can't truly "forget" without releasing
        # This is a simplified implementation
        return None

    def __enter__(self) -> T:
        """Context manager entry."""
        return self.lock.value

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.lock.inner.release()

    def __del__(self) -> None:
        """Release the lock when the guard is dropped."""
        try:
            self.lock.inner.release()
        except RuntimeError:
            pass


class RwLockWriteGuard(Generic[T]):
    """
    A write guard produced by locking RwLock for writing.

    Attributes:
        lock: The RwLock that was locked.
    """

    def __init__(self, lock: RwLock[T]) -> None:
        """
        Initialize the write guard.

        Args:
            lock: The RwLock that was locked.
        """
        self.lock = lock

    @staticmethod
    def downgrade(guard: "RwLockWriteGuard[T]") -> RwLockReadGuard[T]:
        """
        Downgrade a write guard to a read guard.

        Args:
            guard: The write guard to downgrade.

        Returns:
            A read guard.
        """
        return RwLockReadGuard(guard.lock)

    def __enter__(self) -> T:
        """Context manager entry."""
        return self.lock.value

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.lock.inner.release()

    def __del__(self) -> None:
        """Release the lock when the guard is dropped."""
        try:
            self.lock.inner.release()
        except RuntimeError:
            pass


__all__ = [
    "RankData",
    "HeldLock",
    "Mutex",
    "MutexGuard",
    "RwLock",
    "RwLockReadGuard",
    "RwLockWriteGuard",
]
