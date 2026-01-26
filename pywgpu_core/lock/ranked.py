"""
Lock types that enforce well-ranked lock acquisition order.

This module provides instrumented lock types that check that wgpu-core
acquires locks according to their rank, to prevent deadlocks.

The lock ranks form a directed acyclic graph, and threads must acquire
locks along paths through this graph to avoid deadlock.
"""

from __future__ import annotations

import threading
from typing import Any, Generic, Optional, TypeVar

from .rank import LockRank

T = TypeVar("T")


class LockState:
    """
    Per-thread state for the deadlock checker.

    Attributes:
        last_acquired: The last lock we acquired, and where.
        depth: The number of locks currently held.
    """

    INITIAL = None

    def __init__(
        self,
        last_acquired: Optional[tuple[LockRank, str]] = None,
        depth: int = 0,
    ) -> None:
        """
        Initialize lock state.

        Args:
            last_acquired: The last acquired lock and location.
            depth: The current depth.
        """
        self.last_acquired = last_acquired
        self.depth = depth


# Type alias for rank data
RankData = LockState

# Thread-local state
_lock_state = threading.local()


def _get_lock_state() -> LockState:
    """Get the current thread's lock state."""
    if not hasattr(_lock_state, "state"):
        _lock_state.state = LockState()
    return _lock_state.state


def _set_lock_state(state: LockState) -> None:
    """Set the current thread's lock state."""
    _lock_state.state = state


class Mutex(Generic[T]):
    """
    A Mutex instrumented for deadlock prevention.

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

        Raises:
            RuntimeError: If lock ordering is violated.
        """
        state = _get_lock_state()
        
        # Check lock ordering
        if state.last_acquired is not None:
            last_rank, last_loc = state.last_acquired
            if not last_rank.can_acquire(self.rank):
                raise RuntimeError(
                    f"Lock ordering violation: trying to acquire {self.rank} "
                    f"while holding {last_rank} (acquired at {last_loc})"
                )

        self.inner.acquire()
        saved_state = LockState(state.last_acquired, state.depth)
        
        # Update state
        state.last_acquired = (self.rank, "unknown location")
        state.depth += 1
        
        return MutexGuard(self, saved_state)

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
        saved: The saved lock state.
    """

    def __init__(self, mutex: Mutex[T], saved: LockState) -> None:
        """
        Initialize the guard.

        Args:
            mutex: The mutex that was locked.
            saved: The saved lock state.
        """
        self.mutex = mutex
        self.saved = saved

    def __enter__(self) -> T:
        """Context manager entry."""
        return self.mutex.value

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self._release()

    def _release(self) -> None:
        """Release the lock and restore state."""
        self.mutex.inner.release()
        state = _get_lock_state()
        state.last_acquired = self.saved.last_acquired
        state.depth = self.saved.depth

    def __del__(self) -> None:
        """Release the lock when the guard is dropped."""
        try:
            self._release()
        except RuntimeError:
            pass


class RwLock(Generic[T]):
    """
    An RwLock instrumented for deadlock prevention.

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

        Raises:
            RuntimeError: If lock ordering is violated.
        """
        state = _get_lock_state()
        
        # Check lock ordering
        if state.last_acquired is not None:
            last_rank, last_loc = state.last_acquired
            if not last_rank.can_acquire(self.rank):
                raise RuntimeError(
                    f"Lock ordering violation: trying to acquire {self.rank} "
                    f"while holding {last_rank} (acquired at {last_loc})"
                )

        self.inner.acquire()
        saved_state = LockState(state.last_acquired, state.depth)
        
        # Update state
        state.last_acquired = (self.rank, "unknown location")
        state.depth += 1
        
        return RwLockReadGuard(self, saved_state)

    def write(self) -> "RwLockWriteGuard[T]":
        """
        Acquire a write lock.

        Returns:
            A write guard.

        Raises:
            RuntimeError: If lock ordering is violated.
        """
        state = _get_lock_state()
        
        # Check lock ordering
        if state.last_acquired is not None:
            last_rank, last_loc = state.last_acquired
            if not last_rank.can_acquire(self.rank):
                raise RuntimeError(
                    f"Lock ordering violation: trying to acquire {self.rank} "
                    f"while holding {last_rank} (acquired at {last_loc})"
                )

        self.inner.acquire()
        saved_state = LockState(state.last_acquired, state.depth)
        
        # Update state
        state.last_acquired = (self.rank, "unknown location")
        state.depth += 1
        
        return RwLockWriteGuard(self, saved_state)

    def force_unlock_read(self, data: RankData) -> None:
        """
        Force unlock a read guard.

        This is unsafe and should only be used in very specific cases.

        Args:
            data: The rank data for the lock.
        """
        try:
            self.inner.release()
            _set_lock_state(data)
        except RuntimeError:
            pass


class RwLockReadGuard(Generic[T]):
    """
    A read guard produced by locking RwLock for reading.

    Attributes:
        lock: The RwLock that was locked.
        saved: The saved lock state.
    """

    def __init__(self, lock: RwLock[T], saved: LockState) -> None:
        """
        Initialize the read guard.

        Args:
            lock: The RwLock that was locked.
            saved: The saved lock state.
        """
        self.lock = lock
        self.saved = saved

    @staticmethod
    def forget(guard: "RwLockReadGuard[T]") -> RankData:
        """
        Forget the guard, leaving the lock in a locked state.

        Args:
            guard: The guard to forget.

        Returns:
            The rank data for the lock.
        """
        return guard.saved

    def __enter__(self) -> T:
        """Context manager entry."""
        return self.lock.value

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self._release()

    def _release(self) -> None:
        """Release the lock and restore state."""
        self.lock.inner.release()
        state = _get_lock_state()
        state.last_acquired = self.saved.last_acquired
        state.depth = self.saved.depth

    def __del__(self) -> None:
        """Release the lock when the guard is dropped."""
        try:
            self._release()
        except RuntimeError:
            pass


class RwLockWriteGuard(Generic[T]):
    """
    A write guard produced by locking RwLock for writing.

    Attributes:
        lock: The RwLock that was locked.
        saved: The saved lock state.
    """

    def __init__(self, lock: RwLock[T], saved: LockState) -> None:
        """
        Initialize the write guard.

        Args:
            lock: The RwLock that was locked.
            saved: The saved lock state.
        """
        self.lock = lock
        self.saved = saved

    @staticmethod
    def downgrade(guard: "RwLockWriteGuard[T]") -> RwLockReadGuard[T]:
        """
        Downgrade a write guard to a read guard.

        Args:
            guard: The write guard to downgrade.

        Returns:
            A read guard.
        """
        return RwLockReadGuard(guard.lock, guard.saved)

    def __enter__(self) -> T:
        """Context manager entry."""
        return self.lock.value

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self._release()

    def _release(self) -> None:
        """Release the lock and restore state."""
        self.lock.inner.release()
        state = _get_lock_state()
        state.last_acquired = self.saved.last_acquired
        state.depth = self.saved.depth

    def __del__(self) -> None:
        """Release the lock when the guard is dropped."""
        try:
            self._release()
        except RuntimeError:
            pass


__all__ = [
    "LockState",
    "RankData",
    "Mutex",
    "MutexGuard",
    "RwLock",
    "RwLockReadGuard",
    "RwLockWriteGuard",
]
