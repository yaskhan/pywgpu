from __future__ import annotations
from dataclasses import dataclass
from enum import IntFlag
from typing import Any, List, Optional, Iterator, Union

from .metadata import ResourceMetadata, ResourceMetadataProvider

class BufferUses(IntFlag):
    """Flags representing the ways a buffer can be used."""
    MAP_READ = 1 << 0
    MAP_WRITE = 1 << 1
    COPY_SRC = 1 << 2
    COPY_DST = 1 << 3
    INDEX = 1 << 4
    VERTEX = 1 << 5
    UNIFORM = 1 << 6
    STORAGE_READ_ONLY = 1 << 7
    STORAGE_READ_WRITE = 1 << 8
    INDIRECT = 1 << 9
    QUERY_RESOLVE = 1 << 10
    ACCELERATION_STRUCTURE_SCRATCH = 1 << 11
    BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT = 1 << 12
    TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT = 1 << 13
    ACCELERATION_STRUCTURE_QUERY = 1 << 14

    INCLUSIVE = (
        MAP_READ | COPY_SRC | INDEX | VERTEX | UNIFORM |
        STORAGE_READ_ONLY | INDIRECT | BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT |
        TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT
    )
    EXCLUSIVE = MAP_WRITE | COPY_DST | STORAGE_READ_WRITE | ACCELERATION_STRUCTURE_SCRATCH
    ORDERED = INCLUSIVE | MAP_WRITE

# Shared types are now in .__init__

class BufferUsageScope:
    """Tracks buffer usage within a specific scope (e.g., a pass)."""
    def __init__(self) -> None:
        self.state: List[BufferUses] = []
        self.metadata: ResourceMetadata[Any] = ResourceMetadata()

    def tracker_assert_in_bounds(self, index: int) -> None:
        """Ensures the index is within bounds for the scope."""
        assert index < len(self.state)
        self.metadata.tracker_assert_in_bounds(index)

    def merge_single(self, buffer: Any, new_state: BufferUses) -> None:
        """Merges a single buffer usage into the scope."""
        index = buffer.tracker_index()
        if index >= len(self.state):
            self.state.extend([BufferUses(0)] * (index + 1 - len(self.state)))
            self.metadata.set_size(index + 1)
        
        current_state = self.state[index]
        merged_state = current_state | new_state
        
        # Validation could be added here to check for exclusive usage conflicts
        
        self.state[index] = merged_state
        self.metadata.insert(index, buffer)

class BufferTracker:
    """Tracks buffer state across commands in a command buffer."""
    def __init__(self) -> None:
        self.start: List[BufferUses] = []
        self.end: List[BufferUses] = []
        self.metadata: ResourceMetadata[Any] = ResourceMetadata()
        self.temp: List[PendingTransition] = []

    def set_size(self, size: int) -> None:
        """Resizes the internal storage to accommodate more resources."""
        if size > len(self.start):
            self.start.extend([BufferUses(0)] * (size - len(self.start)))
            self.end.extend([BufferUses(0)] * (size - len(self.end)))
            self.metadata.set_size(size)

    def set_single(self, buffer: Any, state: BufferUses) -> Optional[PendingTransition]:
        """Sets the state of a single buffer, returning a transition if needed."""
        index = buffer.tracker_index()
        if index >= len(self.start):
            self.set_size(index + 1)
        
        current_state = self.end[index]
        if current_state != state:
            # Check if we can skip the barrier (simplified)
            # In a full implementation we'd check if both are in ORDERED etc.
            transition = PendingTransition(
                id=index,
                selector=None,
                usage=StateTransition(from_state=current_state, to_state=state)
            )
            self.temp.append(transition)
            self.end[index] = state
            self.metadata.insert(index, buffer)
            return transition
        return None

    def drain_transitions(self) -> Iterator[PendingTransition]:
        """Yields and clears all pending transitions."""
        yield from self.temp
        self.temp.clear()
