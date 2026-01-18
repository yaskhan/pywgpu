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

from typing import Any, Generic, Optional, TypeVar

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
    """

    def __init__(self, guard: RwLockReadGuard) -> None:
        """Initialize the snatch guard."""
        self._guard = guard

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
        # Implementation depends on lock implementation
        # In Python, we would typically extract rank data from the guard
        # For now, return a placeholder rank
        return RankData()


class ExclusiveSnatchGuard:
    """
    A guard that allows snatching the snatchable data.
    
    This guard is used to snatch resources. It holds a write lock on
    the snatchable lock, providing exclusive access to the resource.
    
    Attributes:
        _guard: The underlying write guard.
    """

    def __init__(self, guard: RwLockWriteGuard) -> None:
        """Initialize the exclusive snatch guard."""
        self._guard = guard


class Snatchable(Generic[T]):
    """
    A value that is mostly immutable but can be "snatched" if needed.
    
    This class provides a mechanism for resources that can be destroyed
    early while still being accessed. The value can be read with a
    SnatchGuard and snatched with an ExclusiveSnatchGuard.
    
    Attributes:
        value: The underlying value, stored in an UnsafeCell.
    """

    def __init__(self, val: T) -> None:
        """Initialize the snatchable with a value."""
        self.value: Optional[T] = val

    def get(self, guard: SnatchGuard) -> Optional[T]:
        """
        Get read access to the value.
        
        Args:
            guard: The snatch guard.
        
        Returns:
            The value, or None if it has been snatched.
        """
        return self.value

    def snatch(self, guard: ExclusiveSnatchGuard) -> Optional[T]:
        """
        Take the value.
        
        Args:
            guard: The exclusive snatch guard.
        
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


class SnatchLock:
    """
    A device-global lock for all snatchable data.
    
    This lock ensures thread-safe access to snatchable resources. All
    snatchable resources on a device share this lock.
    
    Attributes:
        lock: The underlying RwLock.
    """

    def __init__(self, rank: Any) -> None:
        """
        Create a new snatch lock.
        
        Args:
            rank: The lock rank.
        
        Note: This is unsafe because the rank must be correct to avoid
        deadlocks. The only place this should be called is when creating
        a device.
        """
        self.lock = RwLock(rank, ())

    def read(self) -> SnatchGuard:
        """
        Request read access to snatchable resources.
        
        Returns:
            A snatch guard.
        """
        return SnatchGuard(self.lock.read())

    def write(self) -> ExclusiveSnatchGuard:
        """
        Request write access to snatchable resources.
        
        This should only be called when a resource needs to be snatched.
        
        Returns:
            An exclusive snatch guard.
        """
        return ExclusiveSnatchGuard(self.lock.write())

    def force_unlock_read(self, data: RankData) -> None:
        """
        Force unlock a read guard.
        
        This is unsafe and should only be used in very specific cases,
        like when a resource needs to be snatched in a panic handler.
        
        Args:
            data: The rank data for the lock.
        """
        # Implementation depends on lock implementation
        # In Python, this would force unlock the lock
        # For now, do nothing as a placeholder
        pass
