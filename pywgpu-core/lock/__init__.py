"""
Instrumented lock types for wgpu-core.

This module defines instrumented wrappers for lock types used in wgpu-core
(Mutex, RwLock) that help us understand and validate synchronization.

- The `ranked` module defines lock types that perform run-time checks to ensure
  each thread acquires locks only in a specific order, to prevent deadlocks.

- The `vanilla` module defines lock types that are uninstrumented, no-overhead
  wrappers around standard lock types.

By default, we use the vanilla module's locks for simplicity.
"""

from .rank import Rank, LockRank
from .vanilla import Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, RankData

__all__ = [
    'Rank',
    'LockRank',
    'Mutex',
    'MutexGuard',
    'RwLock',
    'RwLockReadGuard',
    'RwLockWriteGuard',
    'RankData',
]
