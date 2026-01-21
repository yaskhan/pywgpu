"""Buffer initialization tracking."""

from typing import Optional
from dataclasses import dataclass

from . import InitTracker, MemoryInitKind, Range


@dataclass
class BufferInitTrackerAction:
    """
    Action to track buffer initialization.

    Attributes:
        buffer: The buffer being tracked.
        range: The byte range affected.
        kind: Type of initialization action.
    """

    buffer: any  # Arc<Buffer> in Rust
    range: Range[int]
    kind: MemoryInitKind


class BufferInitTracker(InitTracker[int]):
    """
    Initialization tracker for buffers.

    Tracks initialization at byte granularity.
    """

    def check_action(
        self, action: BufferInitTrackerAction
    ) -> Optional[BufferInitTrackerAction]:
        """
        Check if an action has/requires any effect on initialization status.

        Shrinks the action's range if possible.

        Args:
            action: The action to check.

        Returns:
            A potentially shrunk action, or None if no effect needed.
        """
        return self.create_action(action.buffer, action.range, action.kind)

    def create_action(
        self, buffer: any, query_range: Range[int], kind: MemoryInitKind
    ) -> Optional[BufferInitTrackerAction]:
        """
        Create an action if it would have any effect on initialization status.

        Shrinks the range if possible.

        Args:
            buffer: The buffer to track.
            query_range: The byte range to check.
            kind: Type of initialization action.

        Returns:
            An action if needed, or None if range is already in correct state.
        """
        checked_range = self.check(query_range)

        if checked_range is not None:
            return BufferInitTrackerAction(
                buffer=buffer, range=checked_range, kind=kind
            )

        return None
