from __future__ import annotations
from typing import Generic, TypeVar, List, Optional, Iterator, Set

T = TypeVar("T")


class ResourceMetadata(Generic[T]):
    """
    Metadata for tracked resources.

    Tracks which resources are owned and their associated reference objects.
    """

    def __init__(self) -> None:
        self.owned: Set[int] = set()
        self.resources: List[Optional[T]] = []

    def set_size(self, size: int) -> None:
        """Sets the size of the internal resource storage."""
        if size > len(self.resources):
            self.resources.extend([None] * (size - len(self.resources)))

    def clear(self) -> None:
        """Clears all tracked resources."""
        self.owned.clear()
        self.resources.clear()

    def tracker_assert_in_bounds(self, index: int) -> None:
        """Ensures the index is within bounds and the resource exists if owned."""
        assert index < len(self.resources)
        if self.contains(index):
            assert self.resources[index] is not None

    def is_empty(self) -> bool:
        """Returns True if no resources are tracked."""
        return not self.owned

    def contains(self, index: int) -> bool:
        """Returns True if the resource with the given index is tracked."""
        return index in self.owned

    def insert(self, index: int, resource: T) -> T:
        """Inserts a resource into the tracker."""
        self.owned.add(index)
        if index >= len(self.resources):
            self.set_size(index + 1)
        self.resources[index] = resource
        return resource

    def get_resource_unchecked(self, index: int) -> T:
        """Gets the resource at the given index without checking membership."""
        return self.resources[index]

    def owned_resources(self) -> Iterator[T]:
        """Iterates over all owned resources in index order."""
        for index in sorted(self.owned):
            yield self.resources[index]

    def owned_indices(self) -> Iterator[int]:
        """Iterates over indices of all owned resources in order."""
        return iter(sorted(self.owned))

    def remove(self, index: int) -> None:
        """Removes the resource at the given index."""
        if index in self.owned:
            self.owned.remove(index)
            self.resources[index] = None


class ResourceMetadataProvider(Generic[T]):
    """
    A source of resource metadata.

    Provides a way to abstract over single resources vs another metadata tracker.
    """

    def __init__(
        self,
        resource: Optional[T] = None,
        metadata: Optional[ResourceMetadata[T]] = None,
    ):
        self.resource = resource
        self.metadata = metadata

    @classmethod
    def direct(cls, resource: T) -> ResourceMetadataProvider[T]:
        return cls(resource=resource)

    @classmethod
    def indirect(cls, metadata: ResourceMetadata[T]) -> ResourceMetadataProvider[T]:
        return cls(metadata=metadata)

    def get(self, index: int) -> T:
        """Gets the resource for the given index."""
        if self.resource is not None:
            return self.resource
        if self.metadata is not None:
            return self.metadata.get_resource_unchecked(index)
        raise ValueError("No resource or metadata provider")
