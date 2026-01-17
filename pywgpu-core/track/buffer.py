from typing import Any, List
from .metadata import ResourceMetadata

class BufferTracker:
    """
    Buffer resource tracker.
    """
    def __init__(self) -> None:
        self.buffers: List[Any] = []

    def insert_single(self, buffer: Any, state: Any) -> None:
        self.buffers.append((buffer, state))

class BufferUsageScope:
    """
    Buffer usage scope.
    """
    def __init__(self) -> None:
        self.state: List[Any] = []
        self.metadata: ResourceMetadata = ResourceMetadata()
