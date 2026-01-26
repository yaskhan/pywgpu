"""
Ranks for wgpu-core locks, restricting acquisition order.

Each Mutex, RwLock, and SnatchLock in wgpu-core has been assigned a rank:
a node in a directed acyclic graph (DAG). The rank of the most recently
acquired lock you are still holding determines which locks you may attempt
to acquire next.

This prevents deadlocks by ensuring threads always acquire locks in a
consistent order.
"""

from typing import Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    pass


class LockRankSet:
    """
    A bitflags type representing a set of lock ranks.

    This is a Python implementation of the Rust bitflags! macro.
    """

    def __init__(self, value: int = 0) -> None:
        """Initialize the lock rank set with a bit mask."""
        self._value: int = value

    @property
    def value(self) -> int:
        """Get the underlying value."""
        return self._value

    def union(self, other: "LockRankSet") -> "LockRankSet":
        """
        Union with another set (bitwise OR).

        Args:
            other: Another LockRankSet.

        Returns:
            A new LockRankSet with bits from both.
        """
        return LockRankSet(self._value | other._value)

    def contains(self, other: "LockRankSet") -> bool:
        """
        Check if this set contains another (bitwise AND).

        Args:
            other: Another LockRankSet.

        Returns:
            True if all bits in other are set in this.
        """
        return (self._value & other._value) == other._value

    def is_empty(self) -> bool:
        """Check if the set is empty."""
        return self._value == 0

    def __or__(self, other: "LockRankSet") -> "LockRankSet":
        """Bitwise OR."""
        return LockRankSet(self._value | other._value)

    def __and__(self, other: "LockRankSet") -> "LockRankSet":
        """Bitwise AND."""
        return LockRankSet(self._value & other._value)

    def __eq__(self, other: object) -> bool:
        """Equality check."""
        if isinstance(other, LockRankSet):
            return self._value == other._value
        return False

    def __repr__(self) -> str:
        """Debug representation."""
        return f"LockRankSet({self._value:#x})"

    @staticmethod
    def empty() -> "LockRankSet":
        """Create an empty lock rank set."""
        return LockRankSet(0)


@dataclass(frozen=True)
class LockRank:
    """
    The rank of a lock.

    Attributes:
        bit: The bit representing this lock.
        followers: A bitmask of permitted successor ranks.
    """

    bit: LockRankSet
    followers: LockRankSet

    def __repr__(self) -> str:
        """Debug representation."""
        return f"LockRank({self.bit!r})"


# Define all lock ranks used in wgpu-core
# Each rank has a unique bit and a set of permitted followers
# Note: We define them first with empty followers, then set up followers after

# Command buffer locks
COMMAND_BUFFER_DATA = LockRank(
    bit=LockRankSet(1 << 0),
    followers=LockRankSet.empty(),
)

# Device locks
DEVICE_SNATCHABLE_LOCK = LockRank(
    bit=LockRankSet(1 << 1),
    followers=LockRankSet.empty(),
)

DEVICE_USAGE_SCOPES = LockRank(
    bit=LockRankSet(1 << 2),
    followers=LockRankSet.empty(),
)

DEVICE_TRACE = LockRank(
    bit=LockRankSet(1 << 3),
    followers=LockRankSet.empty(),
)

DEVICE_TRACKERS = LockRank(
    bit=LockRankSet(1 << 4),
    followers=LockRankSet.empty(),
)

DEVICE_FENCE = LockRank(
    bit=LockRankSet(1 << 5),
    followers=LockRankSet.empty(),
)

DEVICE_COMMAND_INDICES = LockRank(
    bit=LockRankSet(1 << 6),
    followers=LockRankSet.empty(),
)

DEVICE_DEFERRED_DESTROY = LockRank(
    bit=LockRankSet(1 << 7),
    followers=LockRankSet.empty(),
)

DEVICE_LOST_CLOSURE = LockRank(
    bit=LockRankSet(1 << 8),
    followers=LockRankSet.empty(),
)

# Buffer locks
BUFFER_BIND_GROUPS = LockRank(
    bit=LockRankSet(1 << 9),
    followers=LockRankSet.empty(),
)

BUFFER_INITIALIZATION_STATUS = LockRank(
    bit=LockRankSet(1 << 10),
    followers=LockRankSet.empty(),
)

BUFFER_MAP_STATE = LockRank(
    bit=LockRankSet(1 << 11),
    followers=LockRankSet.empty(),
)

# Queue locks
QUEUE_PENDING_WRITES = LockRank(
    bit=LockRankSet(1 << 12),
    followers=LockRankSet.empty(),
)

QUEUE_LIFE_TRACKER = LockRank(
    bit=LockRankSet(1 << 13),
    followers=LockRankSet.empty(),
)

# Command allocator
COMMAND_ALLOCATOR_FREE_ENCODERS = LockRank(
    bit=LockRankSet(1 << 14),
    followers=LockRankSet.empty(),
)

# Shared tracker
SHARED_TRACKER_INDEX_ALLOCATOR_INNER = LockRank(
    bit=LockRankSet(1 << 15),
    followers=LockRankSet.empty(),
)

# Registry and pools
REGISTRY_STORAGE = LockRank(
    bit=LockRankSet(1 << 16),
    followers=LockRankSet.empty(),
)

RESOURCE_POOL_INNER = LockRank(
    bit=LockRankSet(1 << 17),
    followers=LockRankSet.empty(),
)

IDENTITY_MANAGER_VALUES = LockRank(
    bit=LockRankSet(1 << 18),
    followers=LockRankSet.empty(),
)

# Surface
SURFACE_PRESENTATION = LockRank(
    bit=LockRankSet(1 << 19),
    followers=LockRankSet.empty(),
)

# Texture locks
TEXTURE_BIND_GROUPS = LockRank(
    bit=LockRankSet(1 << 20),
    followers=LockRankSet.empty(),
)

TEXTURE_INITIALIZATION_STATUS = LockRank(
    bit=LockRankSet(1 << 21),
    followers=LockRankSet.empty(),
)

TEXTURE_CLEAR_MODE = LockRank(
    bit=LockRankSet(1 << 22),
    followers=LockRankSet.empty(),
)

TEXTURE_VIEWS = LockRank(
    bit=LockRankSet(1 << 23),
    followers=LockRankSet.empty(),
)

# Ray tracing
BLAS_BUILT_INDEX = LockRank(
    bit=LockRankSet(1 << 24),
    followers=LockRankSet.empty(),
)

BLAS_COMPACTION_STATE = LockRank(
    bit=LockRankSet(1 << 25),
    followers=LockRankSet.empty(),
)

TLAS_BUILT_INDEX = LockRank(
    bit=LockRankSet(1 << 26),
    followers=LockRankSet.empty(),
)

TLAS_DEPENDENCIES = LockRank(
    bit=LockRankSet(1 << 27),
    followers=LockRankSet.empty(),
)

# Buffer pool
BUFFER_POOL = LockRank(
    bit=LockRankSet(1 << 28),
    followers=LockRankSet.empty(),
)


# Convenience alias
Rank = LockRank

__all__ = [
    "LockRankSet",
    "LockRank",
    "Rank",
    # All the lock rank constants
    "COMMAND_BUFFER_DATA",
    "DEVICE_SNATCHABLE_LOCK",
    "DEVICE_USAGE_SCOPES",
    "DEVICE_TRACE",
    "DEVICE_TRACKERS",
    "DEVICE_FENCE",
    "DEVICE_COMMAND_INDICES",
    "DEVICE_DEFERRED_DESTROY",
    "DEVICE_LOST_CLOSURE",
    "BUFFER_BIND_GROUPS",
    "BUFFER_INITIALIZATION_STATUS",
    "BUFFER_MAP_STATE",
    "QUEUE_PENDING_WRITES",
    "QUEUE_LIFE_TRACKER",
    "COMMAND_ALLOCATOR_FREE_ENCODERS",
    "SHARED_TRACKER_INDEX_ALLOCATOR_INNER",
    "REGISTRY_STORAGE",
    "RESOURCE_POOL_INNER",
    "IDENTITY_MANAGER_VALUES",
    "SURFACE_PRESENTATION",
    "TEXTURE_BIND_GROUPS",
    "TEXTURE_INITIALIZATION_STATUS",
    "TEXTURE_CLEAR_MODE",
    "TEXTURE_VIEWS",
    "BLAS_BUILT_INDEX",
    "BLAS_COMPACTION_STATE",
    "TLAS_BUILT_INDEX",
    "TLAS_DEPENDENCIES",
    "BUFFER_POOL",
]
