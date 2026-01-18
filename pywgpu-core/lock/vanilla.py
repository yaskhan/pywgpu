"""
Plain, uninstrumented wrappers around threading lock types.

These definitions are used when no particular lock instrumentation is needed.
They provide a simple, no-overhead wrapper around Python's threading locks.
"""

import threading
from typing import TypeVar, Generic, Any
from .rank import LockRank


T = TypeVar('T')


class RankData:
    """
    Placeholder for rank data.
    
    In the vanilla implementation, this is empty since we don't track ranks.
    """
    pass


class MutexGuard(Generic[T]):
    """
    A guard produced by locking Mutex.
    
    This is a context manager that automatically releases the lock.
    """
    
    def __init__(self, lock: threading.Lock, value: T):
        """
        Create a mutex guard.
        
        Args:
            lock: The underlying lock.
            value: The protected value.
        """
        self._lock = lock
        self._value = value
    
    def __enter__(self) -> 'MutexGuard[T]':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
    
    @property
    def value(self) -> T:
        """Get the protected value."""
        return self._value


class Mutex(Generic[T]):
    """
    A plain wrapper around threading.Lock.
    
    This is just like threading.Lock, except that our new() method takes
    a rank, indicating where the new mutex should sit in wgpu-core's lock
    ordering. The rank is ignored in this vanilla implementation.
    """
    
    def __init__(self, rank: LockRank, value: T):
        """
        Create a new mutex.
        
        Args:
            rank: The lock rank (ignored in vanilla implementation).
            value: The value to protect.
        """
        self._lock = threading.Lock()
        self._value = value
        self._rank = rank
    
    def lock(self) -> MutexGuard[T]:
        """
        Acquire the lock and return a guard.
        
        Returns:
            A guard that provides access to the protected value.
        """
        self._lock.acquire()
        return MutexGuard(self._lock, self._value)
    
    def into_inner(self) -> T:
        """
        Consume the mutex and return the inner value.
        
        Returns:
            The protected value.
        """
        return self._value


class RwLockReadGuard(Generic[T]):
    """
    A read guard produced by locking RwLock for reading.
    """
    
    def __init__(self, lock: threading.RLock, value: T):
        """
        Create a read guard.
        
        Args:
            lock: The underlying lock.
            value: The protected value.
        """
        self._lock = lock
        self._value = value
    
    def __enter__(self) -> 'RwLockReadGuard[T]':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
    
    @property
    def value(self) -> T:
        """Get the protected value."""
        return self._value
    
    @staticmethod
    def forget(guard: 'RwLockReadGuard[T]') -> RankData:
        """
        Forget the read guard, leaving the lock in a locked state.
        
        This is equivalent to mem::forget but preserves rank information.
        
        Args:
            guard: The guard to forget.
            
        Returns:
            Rank data (empty in vanilla implementation).
        """
        # Don't release the lock
        guard._lock = None
        return RankData()


class RwLockWriteGuard(Generic[T]):
    """
    A write guard produced by locking RwLock for writing.
    """
    
    def __init__(self, lock: threading.RLock, value: T):
        """
        Create a write guard.
        
        Args:
            lock: The underlying lock.
            value: The protected value.
        """
        self._lock = lock
        self._value = value
    
    def __enter__(self) -> 'RwLockWriteGuard[T]':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock:
            self._lock.release()
    
    @property
    def value(self) -> T:
        """Get the protected value."""
        return self._value
    
    @staticmethod
    def downgrade(guard: 'RwLockWriteGuard[T]') -> RwLockReadGuard[T]:
        """
        Downgrade a write guard to a read guard.
        
        Args:
            guard: The write guard to downgrade.
            
        Returns:
            A read guard.
        """
        # In Python, we just return a read guard with the same lock
        return RwLockReadGuard(guard._lock, guard._value)


class RwLock(Generic[T]):
    """
    A plain wrapper around threading.RLock.
    
    This is just like threading.RLock, except that our new() method takes
    a rank, indicating where the new lock should sit in wgpu-core's lock
    ordering. The rank is ignored in this vanilla implementation.
    """
    
    def __init__(self, rank: LockRank, value: T):
        """
        Create a new RwLock.
        
        Args:
            rank: The lock rank (ignored in vanilla implementation).
            value: The value to protect.
        """
        self._lock = threading.RLock()
        self._value = value
        self._rank = rank
    
    def read(self) -> RwLockReadGuard[T]:
        """
        Acquire a read lock and return a guard.
        
        Returns:
            A read guard that provides access to the protected value.
        """
        self._lock.acquire()
        return RwLockReadGuard(self._lock, self._value)
    
    def write(self) -> RwLockWriteGuard[T]:
        """
        Acquire a write lock and return a guard.
        
        Returns:
            A write guard that provides access to the protected value.
        """
        self._lock.acquire()
        return RwLockWriteGuard(self._lock, self._value)
    
    def force_unlock_read(self, data: RankData) -> None:
        """
        Force an read-unlock operation on this lock.
        
        Safety: A read lock must be held which is not held by a guard.
        
        Args:
            data: Rank data from a forgotten guard.
        """
        self._lock.release()
