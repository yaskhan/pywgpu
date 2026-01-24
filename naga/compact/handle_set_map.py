from typing import Generic, TypeVar, Dict, Optional, List, Set, Iterator, Any
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class Handle(Generic[T]):
    """Generic handle for arena items."""

    index: int
    generation: int = 0

    def __hash__(self) -> int:
        return hash((self.index, self.generation))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Handle):
            return self.index == other.index and self.generation == other.generation
        return False


class HandleSet(Generic[T]):
    """
    Set of handles for tracking used items.
    """

    def __init__(self, capacity: int = 0) -> None:
        self.set: Set[Handle[T]] = set()
        self.capacity = capacity

    @classmethod
    def for_arena(cls, arena: Any) -> "HandleSet[T]":
        """Create a HandleSet sized for an arena."""
        return cls(capacity=len(arena) if hasattr(arena, '__len__') else 0)

    def insert(self, handle: Handle[T]) -> bool:
        """Insert a handle into the set. Returns True if it was newly inserted."""
        before_len = len(self.set)
        self.set.add(handle)
        return len(self.set) > before_len

    def insert_iter(self, handles: Iterator[Handle[T]]) -> None:
        """Insert multiple handles into the set."""
        for handle in handles:
            self.set.add(handle)

    def contains(self, handle: Handle[T]) -> bool:
        """Check if a handle is in the set."""
        return handle in self.set

    def used(self, handle: Handle[T]) -> bool:
        """Check if a handle is used."""
        return handle in self.set

    def clear(self) -> None:
        """Clear all handles from the set."""
        self.set.clear()

    def add_all(self) -> None:
        """Add all possible handles to this set."""
        # In the Rust implementation, this adds all possible handles up to capacity
        # For our Python implementation, we'll add handles up to capacity
        for i in range(self.capacity):
            self.set.add(Handle(index=i))

    def pop(self) -> Optional[Handle[T]]:
        """Remove and return the numerically largest handle, or None if empty."""
        if not self.set:
            return None
        # Find the handle with the largest index
        max_handle = max(self.set, key=lambda h: h.index)
        self.set.remove(max_handle)
        return max_handle

    def iter(self) -> Iterator[Handle[T]]:
        """Return an iterator over all handles in this set."""
        return iter(sorted(self.set, key=lambda h: h.index))


class HandleMap(Generic[T]):
    """
    Map old handles to new handles.
    """

    def __init__(self) -> None:
        self.map: Dict[Handle[T], Handle[T]] = {}

    def insert(self, old: Handle[T], new: Handle[T]) -> None:
        """Insert a mapping from old to new handle."""
        self.map[old] = new

    def get(self, old: Handle[T]) -> Optional[Handle[T]]:
        """Get the new handle for an old handle."""
        return self.map.get(old)

    def try_adjust(self, old: Handle[T]) -> Optional[Handle[T]]:
        """
        Return the counterpart to `old` in the compacted module.
        If we thought `old` wouldn't be used in the compacted module, return None.
        """
        return self.map.get(old)

    def adjust(self, handle: Handle[T]) -> None:
        """
        Adjust a handle in place.
        Panic if the handle is not in the map.
        """
        new_handle = self.try_adjust(handle)
        if new_handle is None:
            raise ValueError(f"Handle {handle} not found in map")
        handle.index = new_handle.index
        handle.generation = new_handle.generation

    def adjust_option(self, handle: Optional[Handle[T]]) -> None:
        """Adjust an optional handle in place."""
        if handle is not None:
            self.adjust(handle)

    def adjust_range(self, range_obj: Any, arena: Any) -> None:
        """
        Shrink `range` to include only used handles.
        
        Compaction preserves the order of elements while squeezing out unused ones.
        A contiguous range in the pre-compacted arena maps to a contiguous range
        in the post-compacted arena, so we just need to adjust the endpoints.
        """
        # Get the index range from the range object
        if not hasattr(range_obj, 'start') or not hasattr(range_obj, 'end'):
            return
        
        # Find first and last used handles in the range
        first_new = None
        last_new = None
        
        for i in range(range_obj.start, range_obj.end):
            old_handle = Handle(index=i)
            new_handle = self.try_adjust(old_handle)
            if new_handle is not None:
                if first_new is None:
                    first_new = new_handle.index
                last_new = new_handle.index
        
        # Update the range
        if first_new is not None and last_new is not None:
            range_obj.start = first_new
            range_obj.end = last_new + 1
        else:
            # Empty range
            range_obj.start = 0
            range_obj.end = 0

    def used(self, handle: Handle[T]) -> bool:
        """Check if a handle is in the map (i.e., will be used in compacted module)."""
        return handle in self.map

    @classmethod
    def from_set(cls, handle_set: HandleSet[T]) -> "HandleMap[T]":
        """
        Create a HandleMap from a HandleSet.
        Assigns sequential indices to handles in the set.
        """
        handle_map = cls()
        next_index = 0
        # Process handles in sorted order to maintain ordering
        for old_handle in sorted(handle_set.set, key=lambda h: h.index):
            new_handle = Handle(index=next_index, generation=old_handle.generation)
            handle_map.insert(old_handle, new_handle)
            next_index += 1
        return handle_map

    def __len__(self) -> int:
        return len(self.map)

    def __iter__(self) -> Iterator[Handle[T]]:
        return iter(self.map.keys())
