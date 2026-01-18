from typing import Any, Dict
from .metadata import ResourceMetadata
from .buffer import BufferTracker, BufferUsageScope
from .texture import TextureTracker, TextureUsageScope


class ResourceUsageCompatibilityError(Exception):
    """Raised when resource usage is incompatible."""

    def __init__(self, message: str):
        super().__init__(message)


class Tracker:
    """
    General resource tracker interface.

    Tracks resource usage and ensures compatibility between different uses.
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.buffers: BufferTracker = BufferTracker()
        self.textures: TextureTracker = TextureTracker()

    def add(
        self,
        id: Any,
        ref_count: int,
        selector: Any,
        usage: int,
        life_guard: Any,
    ) -> None:
        """
        Add a resource to the tracker.

        Args:
            id: The resource ID.
            ref_count: The reference count.
            selector: The resource selector.
            usage: The usage flags.
            life_guard: The lifetime guard.
        """
        # Implementation depends on resource type
        pass

    def remove(
        self,
        id: Any,
        selector: Any,
    ) -> None:
        """
        Remove a resource from the tracker.

        Args:
            id: The resource ID.
            selector: The resource selector.
        """
        # Implementation depends on resource type
        pass

    def set_size(self, id: Any, size: int) -> None:
        """
        Set the size of a buffer.

        Args:
            id: The buffer ID.
            size: The buffer size.
        """
        self.buffers.set_size(id, size)

    def query(
        self,
        id: Any,
        selector: Any,
    ) -> ResourceMetadata:
        """
        Query resource metadata.

        Args:
            id: The resource ID.
            selector: The resource selector.

        Returns:
            The resource metadata.
        """
        # Implementation depends on resource type
        return ResourceMetadata()

    def get_usage(self, id: Any) -> int:
        """
        Get the usage of a resource.

        Args:
            id: The resource ID.

        Returns:
            The usage flags.
        """
        # Implementation depends on resource type
        return 0
