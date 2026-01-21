from typing import List, Callable, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .buffer import Buffer
    from .buffer import MapMode


@dataclass
class DeferredBufferMapping:
    """
    A deferred buffer mapping request captured during encoding.
    """

    buffer: "Buffer"
    mode: "MapMode"
    offset: int
    size: int
    callback: Callable[[Optional[Exception]], None]


class DeferredCommandBufferActions:
    """
    Set of actions to take when the command buffer is submitted.
    """

    def __init__(self) -> None:
        self.buffer_mappings: List[DeferredBufferMapping] = []
        self.on_submitted_work_done_callbacks: List[Callable[[], None]] = []

    def append(self, other: "DeferredCommandBufferActions") -> None:
        """Appends actions from another container."""
        self.buffer_mappings.extend(other.buffer_mappings)
        self.on_submitted_work_done_callbacks.extend(
            other.on_submitted_work_done_callbacks
        )

    def execute(self, queue: Any) -> None:
        """
        Executes the deferred actions.

        Note: The actual execution logic for map_async would depend on the
        backend queue implementation.
        """
        for mapping in self.buffer_mappings:
            # We use an async loop or similar to trigger the mapping
            import asyncio

            asyncio.create_task(
                mapping.buffer.map_async(
                    mapping.mode,
                    mapping.offset,
                    mapping.offset + mapping.size,
                    mapping.callback,
                )
            )

        for callback in self.on_submitted_work_done_callbacks:
            # How to register work done callback on queue?
            # if hasattr(queue, 'on_submitted_work_done'):
            #     queue.on_submitted_work_done(callback)
            pass


def range_to_offset_size(bounds: slice, total_size: int) -> tuple[int, int]:
    """Helper to convert slice bounds to offset and size."""
    start = bounds.start if bounds.start is not None else 0
    stop = bounds.stop if bounds.stop is not None else total_size
    return start, stop - start
