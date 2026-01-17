"""
Identity management for resources.

This module provides identity management for wgpu-core resources. It allocates
and manages IDs for resources, ensuring that each resource has a unique
identifier that can be used to reference it.

The identity manager supports two modes of operation:
1. Internal allocation: The manager allocates new IDs automatically
2. External allocation: The manager tracks IDs allocated by external code

This dual mode is useful for scenarios like trace playback, where IDs are
pre-determined and need to be tracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .id import Id, Marker
from .lock import Mutex


@dataclass
class IdSource:
    """Source of IDs for the identity manager."""
    
    EXTERNAL = "External"
    ALLOCATED = "Allocated"
    NONE = "None"


@dataclass
class IdentityValues(Generic[Marker]):
    """
    A simple structure to allocate Id identifiers.
    
    Calling alloc returns a fresh, never-before-seen id. Calling release
    marks an id as dead; it will never be returned again by alloc.
    
    IdentityValues returns Ids whose index values are suitable for use as
    indices into a Vec<T> that holds those ids' referents:
    
    - Every live id has a distinct index value. Every live id's index
      selects a distinct element in the vector.
    
    - IdentityValues prefers low index numbers. If you size your vector to
      accommodate the indices produced here, the vector's length will reflect
      the highwater mark of actual occupancy.
    
    - IdentityValues reuses the index values of freed ids before returning
      ids with new index values. Freed vector entries get reused.
    
    - The non-reuse property is achieved by storing an `epoch` alongside the
      index in an `Id`. Index values are reused, but only with a different
      epoch.
    
    IdentityValues can also be used to track the count of IDs allocated by
    some external allocator. Combining internal and external allocation is not
    allowed; calling both alloc and mark_as_used on the same
    IdentityValues will result in a panic. The external mode is used when
    playing back a trace of wgpu operations.
    
    Attributes:
        free: List of free (index, epoch) pairs.
        next_index: The next index to allocate.
        count: The number of allocated IDs.
        id_source: The source of IDs (internal or external).
    """

    free: list[tuple[int, int]]
    next_index: int
    count: int
    id_source: str

    def alloc(self) -> Id[Marker]:
        """
        Allocate a fresh, never-before-seen ID.
        
        Returns:
            A new ID.
        
        Raises:
            RuntimeError: If mark_as_used has previously been called on this IdentityValues.
        """
        if self.id_source == IdSource.EXTERNAL:
            raise RuntimeError("Mix of internally allocated and externally provided IDs")
        self.id_source = IdSource.ALLOCATED

        self.count += 1
        if self.free:
            index, epoch = self.free.pop()
            return Id.zip(index, epoch + 1)
        else:
            index = self.next_index
            self.next_index += 1
            epoch = 1
            return Id.zip(index, epoch)

    def mark_as_used(self, id: Id[Marker]) -> Id[Marker]:
        """
        Increment the count of used IDs.
        
        Args:
            id: The ID to mark as used.
        
        Returns:
            The same ID.
        
        Raises:
            RuntimeError: If alloc has previously been called on this IdentityValues.
        """
        if self.id_source == IdSource.ALLOCATED:
            raise RuntimeError("Mix of internally allocated and externally provided IDs")
        self.id_source = IdSource.EXTERNAL

        self.count += 1
        return id

    def release(self, id: Id[Marker]) -> None:
        """
        Free an ID and/or decrement the count of used IDs.
        
        Freed IDs will never be returned from alloc again.
        
        Args:
            id: The ID to release.
        """
        if self.id_source == IdSource.ALLOCATED:
            index, epoch = id.unzip()
            self.free.append((index, epoch))
        self.count -= 1

    def count(self) -> int:
        """
        Get the number of allocated IDs.
        
        Returns:
            The count of allocated IDs.
        """
        return self.count


class IdentityManager(Generic[Marker]):
    """
    Manager for identity values.
    
    This class manages the allocation and tracking of IDs for a specific
    resource type. It ensures that each resource has a unique identifier
    and can detect when an ID is reused for a different resource.
    
    Attributes:
        values: Mutex-protected identity values.
    """

    def __init__(self) -> None:
        """Initialize the IdentityManager."""
        self.values = Mutex(
            IdentityValues(
                free=[],
                next_index=0,
                count=0,
                id_source=IdSource.NONE,
            )
        )

    def process(self) -> Id[Marker]:
        """
        Allocate a new ID.
        
        Returns:
            A new ID.
        """
        return self.values.lock().alloc()

    def mark_as_used(self, id: Id[Marker]) -> Id[Marker]:
        """
        Mark an ID as used.
        
        Args:
            id: The ID to mark as used.
        
        Returns:
            The same ID.
        """
        return self.values.lock().mark_as_used(id)

    def free(self, id: Id[Marker]) -> None:
        """
        Free an ID.
        
        Args:
            id: The ID to free.
        """
        self.values.lock().release(id)
