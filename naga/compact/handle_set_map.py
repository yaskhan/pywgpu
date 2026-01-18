from typing import Generic, TypeVar, Dict, Optional, List, Set, Iterator, Any
from dataclasses import dataclass

T = TypeVar('T')


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
    def __init__(self) -> None:
        self.set: Set[Handle[T]] = set()

    def insert(self, handle: Handle[T]) -> None:
        """Insert a handle into the set."""
        self.set.add(handle)

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

    def adjust(self, handle: Handle[T]) -> None:
        """Adjust a handle in place."""
        if handle in self.map:
            new_handle = self.map[handle]
            handle.index = new_handle.index
            handle.generation = new_handle.generation

    def adjust_option(self, handle: Optional[Handle[T]]) -> None:
        """Adjust an optional handle in place."""
        if handle is not None:
            self.adjust(handle)

    def adjust_range(self, range_handle: Handle[T], arena: Any) -> None:
        """Adjust a range of handles."""
        # Placeholder implementation
        pass

    def used(self, handle: Handle[T]) -> bool:
        """Check if a handle is in the map."""
        return handle in self.map

    @classmethod
    def from_set(cls, handle_set: HandleSet[T]) -> 'HandleMap[T]':
        """Create a HandleMap from a HandleSet."""
        handle_map = cls()
        for handle in handle_set.set:
            handle_map.insert(handle, handle)
        return handle_map

    def __len__(self) -> int:
        return len(self.map)

    def __iter__(self) -> Iterator[Handle[T]]:
        return iter(self.map.keys())
