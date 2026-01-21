"""
Resource storage.

This module implements the storage system for wgpu-core resources. Storage
is a table that maps resource IDs to resources, providing efficient lookups
and management of resource lifetimes.

The storage system:
- Maps IDs to resources
- Tracks resource epochs to detect stale IDs
- Provides efficient lookups by ID
- Supports iteration over all stored resources
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from .id import Id, Marker
from .resource import ResourceType


T = TypeVar("T", bound=StorageItem)


@dataclass
class Element(Generic[T]):
    """
    An entry in a storage table.

    Attributes:
        occupied: The resource and its epoch, if occupied.
        vacant: Whether the entry is vacant.
    """

    occupied: Optional[tuple[T, int]] = None

    def is_occupied(self) -> bool:
        """Check if the entry is occupied."""
        return self.occupied is not None


class StorageItem(ResourceType):
    """
    Trait for items that can be stored in storage.

    This trait is implemented by resources that can be stored in the
    storage system. It provides the marker type for the resource.

    Attributes:
        Marker: The marker type for the resource.
    """

    Marker: type[Marker]


class Storage(Generic[T]):
    """
    A table of T values indexed by ID.

    Storage implements efficient lookups by ID and manages the lifecycle
    of resources. It is represented as a vector indexed by the IDs' index
    values.

    Attributes:
        map: The vector of elements.
        kind: The type name for error messages.
    """

    def __init__(self) -> None:
        """Initialize the storage."""
        self.map: list[Element[T]] = []
        self.kind: str = T.TYPE

    def insert(self, id: Id[T.Marker], value: T) -> None:
        """
        Insert a value into storage.

        Args:
            id: The ID to insert at.
            value: The value to insert.
        """
        index, epoch = id.unzip()
        index = int(index)

        # Resize the map if necessary
        if index >= len(self.map):
            self.map.extend([Element() for _ in range(index - len(self.map) + 1)])

        # Check if the slot is already occupied
        existing = self.map[index]
        if existing.occupied is not None:
            _, storage_epoch = existing.occupied
            if epoch == storage_epoch:
                raise RuntimeError(f"Index {index} of {self.kind} is already occupied")

        # Insert the value
        self.map[index] = Element(occupied=(value, epoch))

    def remove(self, id: Id[T.Marker]) -> T:
        """
        Remove a value from storage.

        Args:
            id: The ID to remove.

        Returns:
            The removed value.

        Raises:
            RuntimeError: If the ID doesn't exist or epoch mismatch.
        """
        index, epoch = id.unzip()
        index = int(index)

        if index >= len(self.map):
            raise RuntimeError(f"{self.kind}[{id}] does not exist")

        element = self.map[index]
        if element.occupied is None:
            raise RuntimeError(f"Cannot remove a vacant resource")

        value, storage_epoch = element.occupied
        if epoch != storage_epoch:
            raise RuntimeError(f"ID epoch mismatch")

        # Remove the value
        self.map[index] = Element()
        return value

    def iter(self) -> list[tuple[Id[T.Marker], T]]:
        """
        Iterate over all stored resources.

        Returns:
            A list of (ID, resource) pairs.
        """
        result = []
        for index, element in enumerate(self.map):
            if element.occupied is not None:
                value, storage_epoch = element.occupied
                id = Id.zip(index, storage_epoch)
                result.append((id, value))
        return result

    def get(self, id: Id[T.Marker]) -> T:
        """
        Get a resource by ID.

        Args:
            id: The ID of the resource.

        Returns:
            The resource.

        Raises:
            RuntimeError: If the resource doesn't exist or ID mismatch.
        """
        index, epoch = id.unzip()
        index = int(index)

        if index >= len(self.map):
            raise RuntimeError(f"{self.kind}[{id}] does not exist")

        element = self.map[index]
        if element.occupied is None:
            raise RuntimeError(f"{self.kind}[{id}] does not exist")

        value, storage_epoch = element.occupied
        if epoch != storage_epoch:
            raise RuntimeError(f"{self.kind}[{id}] is no longer alive")

        return value

    def element_size(self) -> int:
        """Get the size of each element."""
        return 0  # Would be calculated based on T
