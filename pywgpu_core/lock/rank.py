"""
Ranks for wgpu-core locks, restricting acquisition order.

Each Mutex, RwLock, and SnatchLock in wgpu-core has been assigned a rank:
a node in a directed acyclic graph (DAG). The rank of the most recently
acquired lock you are still holding determines which locks you may attempt
to acquire next.

This prevents deadlocks by ensuring threads always acquire locks in a
consistent order.
"""

from typing import Set
from dataclasses import dataclass


@dataclass(frozen=True)
class LockRank:
    """
    The rank of a lock.

    Attributes:
        name: Human-readable name of this rank.
        bit: Unique bit representing this lock.
        followers: Set of ranks that can be acquired after this one.
    """

    name: str
    bit: int
    followers: Set[int]

    def __repr__(self) -> str:
        return f"LockRank({self.name})"


# Define all lock ranks used in wgpu-core
# Each rank has a unique bit and a set of permitted followers

# Command buffer locks
COMMAND_BUFFER_DATA = LockRank(
    name="CommandBuffer::data", bit=1 << 0, followers=set()  # Will be populated below
)

# Device locks
DEVICE_SNATCHABLE_LOCK = LockRank(
    name="Device::snatchable_lock", bit=1 << 1, followers=set()
)

DEVICE_USAGE_SCOPES = LockRank(name="Device::usage_scopes", bit=1 << 2, followers=set())

DEVICE_TRACE = LockRank(name="Device::trace", bit=1 << 3, followers=set())

DEVICE_TRACKERS = LockRank(name="Device::trackers", bit=1 << 4, followers=set())

DEVICE_FENCE = LockRank(name="Device::fence", bit=1 << 5, followers=set())

DEVICE_COMMAND_INDICES = LockRank(
    name="Device::command_indices", bit=1 << 6, followers=set()
)

DEVICE_DEFERRED_DESTROY = LockRank(
    name="Device::deferred_destroy", bit=1 << 7, followers=set()
)

DEVICE_LOST_CLOSURE = LockRank(
    name="Device::device_lost_closure", bit=1 << 8, followers=set()
)

# Buffer locks
BUFFER_MAP_STATE = LockRank(name="Buffer::map_state", bit=1 << 9, followers=set())

BUFFER_BIND_GROUPS = LockRank(name="Buffer::bind_groups", bit=1 << 10, followers=set())

BUFFER_INITIALIZATION_STATUS = LockRank(
    name="Buffer::initialization_status", bit=1 << 11, followers=set()
)

# Queue locks
QUEUE_PENDING_WRITES = LockRank(
    name="Queue::pending_writes", bit=1 << 12, followers=set()
)

QUEUE_LIFE_TRACKER = LockRank(name="Queue::life_tracker", bit=1 << 13, followers=set())

# Command allocator
COMMAND_ALLOCATOR_FREE_ENCODERS = LockRank(
    name="CommandAllocator::free_encoders", bit=1 << 14, followers=set()
)

# Shared tracker
SHARED_TRACKER_INDEX_ALLOCATOR_INNER = LockRank(
    name="SharedTrackerIndexAllocator::inner", bit=1 << 15, followers=set()
)

# Registry and pools
REGISTRY_STORAGE = LockRank(name="Registry::storage", bit=1 << 16, followers=set())

RESOURCE_POOL_INNER = LockRank(name="ResourcePool::inner", bit=1 << 17, followers=set())

IDENTITY_MANAGER_VALUES = LockRank(
    name="IdentityManager::values", bit=1 << 18, followers=set()
)

# Surface
SURFACE_PRESENTATION = LockRank(
    name="Surface::presentation", bit=1 << 19, followers=set()
)

# Texture locks
TEXTURE_BIND_GROUPS = LockRank(
    name="Texture::bind_groups", bit=1 << 20, followers=set()
)

TEXTURE_INITIALIZATION_STATUS = LockRank(
    name="Texture::initialization_status", bit=1 << 21, followers=set()
)

TEXTURE_CLEAR_MODE = LockRank(name="Texture::clear_mode", bit=1 << 22, followers=set())

TEXTURE_VIEWS = LockRank(name="Texture::views", bit=1 << 23, followers=set())

# Ray tracing
BLAS_BUILT_INDEX = LockRank(name="Blas::built_index", bit=1 << 24, followers=set())

BLAS_COMPACTION_STATE = LockRank(
    name="Blas::compaction_size", bit=1 << 25, followers=set()
)

TLAS_BUILT_INDEX = LockRank(name="Tlas::built_index", bit=1 << 26, followers=set())

TLAS_DEPENDENCIES = LockRank(name="Tlas::dependencies", bit=1 << 27, followers=set())

# Buffer pool
BUFFER_POOL = LockRank(name="BufferPool::buffers", bit=1 << 28, followers=set())

# Now populate followers to create the DAG
# This defines the permitted lock acquisition order

COMMAND_BUFFER_DATA.followers.update(
    [
        DEVICE_SNATCHABLE_LOCK.bit,
        DEVICE_USAGE_SCOPES.bit,
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER.bit,
        BUFFER_MAP_STATE.bit,
    ]
)

DEVICE_SNATCHABLE_LOCK.followers.update(
    [
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER.bit,
        DEVICE_TRACE.bit,
        BUFFER_MAP_STATE.bit,
    ]
)

BUFFER_MAP_STATE.followers.update(
    [
        QUEUE_PENDING_WRITES.bit,
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER.bit,
        DEVICE_TRACE.bit,
    ]
)

QUEUE_PENDING_WRITES.followers.update(
    [
        COMMAND_ALLOCATOR_FREE_ENCODERS.bit,
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER.bit,
        QUEUE_LIFE_TRACKER.bit,
    ]
)

QUEUE_LIFE_TRACKER.followers.update(
    [
        COMMAND_ALLOCATOR_FREE_ENCODERS.bit,
        DEVICE_TRACE.bit,
    ]
)

COMMAND_ALLOCATOR_FREE_ENCODERS.followers.update(
    [
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER.bit,
    ]
)

# Leaf nodes (no followers) are already empty sets


# Convenience alias
Rank = LockRank
