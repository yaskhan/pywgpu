"""
Resource snatching logic.

This module implements the "snatch" mechanism for wgpu-core resources.
A resource can be "snatched" (destroyed early) while still being accessed
by other parts of the system.

The snatch system uses a global lock to ensure thread-safe access to
snatchable resources. Resources that can be snatched include buffers,
textures, and other GPU resources.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from .lock import RankData, RwLock, RwLockReadGuard, RwLockWriteGuard

T = TypeVar("T")


class SnatchGuard:
    """
    A guard that provides read access to snatchable data.

    This guard is used to access resources that can be snatched. It
    holds a read lock on the snatchable lock, ensuring that the
    resource cannot be snatched while the guard is active.

    Attributes:
        _guard: The underlying read guard.
        _lock_trace: Optional lock trace for debugging.
    """

    def __init__(self, guard: RwLockReadGuard, lock_trace: Optional["_LockTrace"] = None) -> None:
        """Initialize the snatch guard."""
        self._guard = guard
        self._lock_trace = lock_trace

    @staticmethod
    def forget(guard: SnatchGuard) -> RankData:
        """
        Forget the guard, leaving the lock in a locked state.

        This is equivalent to std::mem::forget, but preserves the
        information about the lock rank.

        Args:
            guard: The guard to forget.

        Returns:
            The rank data for the lock.
        """
        # Cancel the drop by extracting rank data and not calling __del__
        # Extract rank data from the guard's lock
        if hasattr(guard._guard, "forget"):
            return guard._guard.forget()

        # Fallback: create empty rank data
        from .lock import RankData

        return RankData()

    def __del__(self) -> None:
        """Call lock trace exit when the guard is dropped."""
        if self._lock_trace is not None:
            self._lock_trace.exit()


class ExclusiveSnatchGuard:
    """
    A guard that allows snatching the snatchable data.

    This guard is used to snatch resources. It holds a write lock on
    the snatchable lock, providing exclusive access to the resource.

    Attributes:
        _guard: The underlying write guard.
        _lock_trace: Optional lock trace for debugging.
    """

    def __init__(self, guard: RwLockWriteGuard, lock_trace: Optional["_LockTrace"] = None) -> None:
        """Initialize the exclusive snatch guard."""
        self._guard = guard
        self._lock_trace = lock_trace

    def __del__(self) -> None:
        """Call lock trace exit when the guard is dropped."""
        if self._lock_trace is not None:
            self._lock_trace.exit()


class Snatchable(Generic[T]):
    """
    A value that is mostly immutable but can be "snatched" if needed.

    This class provides a mechanism for resources that can be destroyed
    early while still being accessed. The value can be read with a
    SnatchGuard and snatched with an ExclusiveSnatchGuard.

    Attributes:
        value: The underlying value, wrapped in a thread-safe container.
    """

    def __init__(self, val: Optional[T]) -> None:
        """Initialize the snatchable with a value."""
        self.value: Optional[T] = val

    @classmethod
    def new(cls, val: T) -> "Snatchable[T]":
        """
        Create a new snatchable with a value.

        Args:
            val: The value to store.

        Returns:
            A new snatchable containing the value.
        """
        return cls(val)

    @classmethod
    def empty(cls) -> "Snatchable[T]":
        """
        Create an empty snatchable with no value.

        Returns:
            A new empty snatchable.
        """
        return cls(None)

    def get(self, _guard: SnatchGuard) -> Optional[T]:
        """
        Get read access to the value.

        Args:
            _guard: The snatch guard.

        Returns:
            The value, or None if it has been snatched.
        """
        return self.value

    def snatch(self, _guard: ExclusiveSnatchGuard) -> Optional[T]:
        """
        Take the value.

        Args:
            _guard: The exclusive snatch guard.

        Returns:
            The value that was snatched, or None if already snatched.
        """
        old_value = self.value
        self.value = None
        return old_value

    def take(self) -> Optional[T]:
        """
        Take the value without a guard.

        This can only be used with exclusive access to self, so it does
        not require locking. Typically useful in a drop implementation.

        Returns:
            The value that was taken, or None if already taken.
        """
        old_value = self.value
        self.value = None
        return old_value

    def __repr__(self) -> str:
        """Get a debug representation."""
        return "<snatchable>"


class _LockTrace:
    """
    Lock trace for debugging snatch lock usage.

    This is only used in debug mode to detect recursive lock acquisition.
    """

    _local_storage = threading.local()

    def __init__(self, purpose: str) -> None:
        """
        Initialize the lock trace.

        Args:
            purpose: The purpose of acquiring this lock.
        """
        self.purpose = purpose
        self._previous: Optional[_LockTrace] = None

    def enter(self) -> None:
        """Enter the lock trace."""
        prev = getattr(_LockTrace._local_storage, "current", None)
        if prev is not None:
            import traceback

            raise RuntimeError(
                f"Attempted to acquire snatch lock recursively.\n"
                f"Currently trying to acquire a {self.purpose} lock\n"
                f"Previously acquired a {prev.purpose} lock\n"
                f"Backtrace:\n{traceback.format_stack()}"
            )
        self._previous = prev
        _LockTrace._local_storage.current = self

    def exit(self) -> None:
        """Exit the lock trace."""
        _LockTrace._local_storage.current = self._previous

    def __enter__(self) -> "_LockTrace":
        """Context manager entry."""
        self.enter()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.exit()


class SnatchLock:
    """
    A device-global lock for all snatchable data.

    This lock ensures thread-safe access to snatchable resources. All
    snatchable resources on a device share this lock.

    Attributes:
        lock: The underlying RwLock.
    """

    def __init__(self, rank: object) -> None:
        """
        Create a new snatch lock.

        Args:
            rank: The lock rank.

        Note: This is unsafe because the rank must be correct to avoid
        deadlocks. The only place this should be called is when creating
        a device.
        """
        from .lock import RwLock

        self.lock = RwLock(rank, ())

    def read(self) -> SnatchGuard:
        """
        Request read access to snatchable resources.

        Returns:
            A snatch guard.
        """
        trace = _LockTrace("read")
        trace.enter()
        return SnatchGuard(self.lock.read(), trace)

    def write(self) -> ExclusiveSnatchGuard:
        """
        Request write access to snatchable resources.

        This should only be called when a resource needs to be snatched.

        Returns:
            An exclusive snatch guard.
        """
        trace = _LockTrace("write")
        trace.enter()
        return ExclusiveSnatchGuard(self.lock.write(), trace)

    def force_unlock_read(self, data: RankData) -> None:
        """
        Force unlock a read guard.

        This is unsafe and should only be used in very specific cases,
        like when a resource needs to be snatched in a panic handler.

        Args:
            data: The rank data for the lock.
        """
        # Force unlock the underlying RwLock
        if hasattr(self.lock, "force_unlock_read"):
            self.lock.force_unlock_read(data)
        else:
            # Fallback: not implemented
            pass


__all__ = [
    "SnatchGuard",
    "ExclusiveSnatchGuard",
    "Snatchable",
    "SnatchLock",
]
