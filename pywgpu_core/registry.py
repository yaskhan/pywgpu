"""
Resource registry.

This module implements the resource registry, which is the primary holder
of each resource type. Every resource is stored in a registry, which
manages the lifecycle of resources.

The registry:
- Allocates and tracks resource IDs
- Stores resources in a thread-safe manner
- Provides efficient lookups by ID
- Generates reports on resource usage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from .id import Id, Marker
from .identity import IdentityManager
from .lock import RwLock
from .storage import Storage, StorageItem


T = TypeVar("T", bound=StorageItem)


@dataclass
class RegistryReport:
    """
    Report containing statistics about a registry.

    Attributes:
        num_allocated: Number of IDs allocated.
        num_kept_from_user: Number of resources kept from user.
        num_released_from_user: Number of resources released from user.
        element_size: Size of each element.
    """

    num_allocated: int = 0
    num_kept_from_user: int = 0
    num_released_from_user: int = 0
    element_size: int = 0

    def is_empty(self) -> bool:
        """Check if the report is empty."""
        return self.num_allocated == 0 and self.num_kept_from_user == 0


@dataclass
class FutureId(Generic[T]):
    """
    A future ID that can be assigned to a resource.

    This class represents an ID that will be assigned to a resource
    when it is created. It provides a method to assign the resource
    to the ID.

    Attributes:
        id: The ID to assign.
        data: The storage to assign to.
    """

    id: Id[T.Marker]
    data: RwLock[Storage[T]]

    def assign(self, value: T) -> Id[T.Marker]:
        """
        Assign a resource to this ID.

        Args:
            value: The resource to assign.

        Returns:
            The assigned ID.
        """
        with self.data.write() as storage:
            storage.insert(self.id, value)
        return self.id


class Registry(Generic[T]):
    """
    Registry for resources of a specific type.

    The registry is the primary holder of each resource type. It manages
    the allocation of IDs and the storage of resources.

    Attributes:
        identity: Identity manager for allocating IDs.
        storage: Storage for resources.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self.identity: IdentityManager[T.Marker] = IdentityManager()
        self.storage: RwLock[Storage[T]] = RwLock(Storage())

    def prepare(self, id_in: Optional[Id[T.Marker]]) -> FutureId[T]:
        """
        Prepare an ID for a resource.

        Args:
            id_in: Optional ID to use. If None, a new ID is allocated.

        Returns:
            A FutureId that can be used to assign the resource.
        """
        if id_in is not None:
            id_to_use = self.identity.mark_as_used(id_in)
        else:
            id_to_use = self.identity.process()

        return FutureId(id=id_to_use, data=self.storage)

    def read(self) -> Storage[T]:
        """
        Get read access to the storage.

        Returns:
            The storage.
        """
        return self.storage.read()

    def remove(self, id: Id[T.Marker]) -> T:
        """
        Remove a resource from the registry.

        Args:
            id: The ID of the resource to remove.

        Returns:
            The removed resource.
        """
        value = self.storage.write().remove(id)
        self.identity.free(id)
        return value

    def generate_report(self) -> RegistryReport:
        """
        Generate a report of the registry's state.

        Returns:
            A report containing statistics about the registry.
        """
        storage = self.storage.read()
        report = RegistryReport(
            element_size=storage.element_size(),
        )
        report.num_allocated = self.identity.values.lock().count()
        for element in storage.map:
            if element.is_occupied():
                report.num_kept_from_user += 1
            else:
                report.num_released_from_user += 1
        return report

    def get(self, id: Id[T.Marker]) -> T:
        """
        Get a resource by ID.

        Args:
            id: The ID of the resource.

        Returns:
            The resource.

        Raises:
            Exception: If the resource doesn't exist or ID mismatch.
        """
        return self.storage.read().get(id)
