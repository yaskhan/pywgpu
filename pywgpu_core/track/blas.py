"""
BLAS (Bottom Level Acceleration Structure) tracker.

This module implements tracking for BLAS resources used in ray tracing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..resource import Blas

from .metadata import ResourceMetadata


class BlasTracker:
    """
    A tracker that holds tracks BLASes.

    This is mostly a safe shell around ResourceMetadata.

    Attributes:
        metadata: Resource metadata tracking.
        size: Current size of the tracker.
    """

    def __init__(self) -> None:
        """Initialize a new BLAS tracker."""
        self.metadata: ResourceMetadata[Blas] = ResourceMetadata()
        self.size: int = 0

    def insert_single(self, resource: Blas) -> Blas:
        """
        Insert a single resource into the resource tracker.

        Args:
            resource: The BLAS to insert.

        Returns:
            A reference to the newly inserted resource.
        """
        index = resource.tracker_index().as_usize()
        self._allow_index(index)
        return self.metadata.insert(index, resource)

    def set_size(self, size: int) -> None:
        """
        Set the size of all the vectors inside the tracker.

        Must be called with the highest possible BLAS ID before
        all operations are performed.

        Args:
            size: The new size.
        """
        self.size = size
        self.metadata.set_size(size)

    def _allow_index(self, index: int) -> None:
        """
        Extend the vectors to let the given index be valid.

        Args:
            index: The index to allow.
        """
        if index >= self.size:
            self.set_size(index + 1)

    def remove(self, index: int) -> Optional[Blas]:
        """
        Remove a resource from the tracker.

        Args:
            index: The tracker index.

        Returns:
            The removed resource, or None if not present.
        """
        return self.metadata.remove(index)

    def clear(self) -> None:
        """Clear all tracked resources."""
        self.metadata.clear()
        self.size = 0


__all__ = ["BlasTracker"]
