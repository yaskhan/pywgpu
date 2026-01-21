"""
Lazy initialization tracking for texture and buffer memory.

The WebGPU specification requires all texture & buffer memory to be
zero initialized on first read. To avoid unnecessary inits, we track
the initialization status of every resource and perform inits lazily.

The granularity is different for buffers and textures:

- Buffer: Byte granularity to support usecases with large, partially
  bound buffers well.

- Texture: Mip-level per layer. That is, a 2D surface is either
  completely initialized or not, subrects are not tracked.

Every use of a buffer/texture generates a InitTrackerAction which are
recorded and later resolved at queue submit by merging them with the
current state and each other in execution order.

It is important to note that from the point of view of the memory init
system there are two kind of writes:

- **Full writes**: Any kind of memcpy operation. These cause a
  `MemoryInitKind.ImplicitlyInitialized` action.

- **(Potentially) partial writes**: For example, write use in a
  Shader. The system is not able to determine if a resource is fully
  initialized afterwards but is no longer allowed to perform any
  clears, therefore this leads to a
  `MemoryInitKind.NeedsInitializedMemory` action, exactly like a read
  would.
"""

from typing import List, Optional, Iterator, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum


class MemoryInitKind(Enum):
    """Type of memory initialization."""

    # The memory range is going to be written by an already initialized source,
    # thus doesn't need extra attention other than marking as initialized.
    IMPLICITLY_INITIALIZED = "implicitly_initialized"

    # The memory range is going to be read, therefore needs to ensure prior
    # initialization.
    NEEDS_INITIALIZED_MEMORY = "needs_initialized_memory"


T = TypeVar("T", int, float)


@dataclass
class Range(Generic[T]):
    """A range from start to end (exclusive)."""

    start: T
    end: T

    def __contains__(self, value: T) -> bool:
        return self.start <= value < self.end

    def overlaps(self, other: "Range[T]") -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return int(self.end - self.start)


class InitTracker(Generic[T]):
    """
    Tracks initialization status of a linear range from 0..size.

    Maintains a non-overlapping list of all uninitialized ranges,
    sorted by range end.

    Attributes:
        uninitialized_ranges: List of uninitialized ranges.
    """

    def __init__(self, size: T):
        """
        Create a new init tracker for a resource of given size.

        All memory starts as uninitialized.

        Args:
            size: The size of the resource to track.
        """
        self.uninitialized_ranges: List[Range[T]] = [Range(0, size)] if size > 0 else []

    def check(self, query_range: Range[T]) -> Optional[Range[T]]:
        """
        Check for uninitialized ranges within a given query range.

        If `query_range` includes any uninitialized portions of this init
        tracker's resource, return the smallest subrange of `query_range` that
        covers all uninitialized regions.

        The returned range may be larger than necessary, to keep this function
        O(log n).

        Args:
            query_range: The range to check.

        Returns:
            A range covering uninitialized portions, or None if fully initialized.
        """
        # Binary search for first range that might overlap
        index = self._partition_point(lambda r: r.end <= query_range.start)

        if index >= len(self.uninitialized_ranges):
            return None

        start_range = self.uninitialized_ranges[index]

        if start_range.start >= query_range.end:
            return None

        # Found at least one uninitialized range
        start = max(start_range.start, query_range.start)

        # Check if there's another uninitialized range in query
        if index + 1 < len(self.uninitialized_ranges):
            next_range = self.uninitialized_ranges[index + 1]
            if next_range.start < query_range.end:
                # Multiple uninitialized ranges, return conservative bound
                return Range(start, query_range.end)

        # Single uninitialized range
        end = min(start_range.end, query_range.end)
        return Range(start, end)

    def uninitialized(self, drain_range: Range[T]) -> Iterator[Range[T]]:
        """
        Iterate over uninitialized ranges within drain_range.

        Args:
            drain_range: The range to query.

        Yields:
            Uninitialized ranges within drain_range.
        """
        index = self._partition_point(lambda r: r.end <= drain_range.start)

        while index < len(self.uninitialized_ranges):
            range_item = self.uninitialized_ranges[index]
            if range_item.start >= drain_range.end:
                break

            start = max(range_item.start, drain_range.start)
            end = min(range_item.end, drain_range.end)
            yield Range(start, end)

            index += 1

    def drain(self, drain_range: Range[T]) -> Iterator[Range[T]]:
        """
        Drain (mark as initialized) and iterate over uninitialized ranges.

        This marks the drain_range as initialized and yields all previously
        uninitialized subranges.

        Args:
            drain_range: The range to drain.

        Yields:
            Previously uninitialized ranges within drain_range.
        """
        index = self._partition_point(lambda r: r.end <= drain_range.start)
        first_index = index

        # Collect ranges to yield
        ranges_to_yield = []
        while index < len(self.uninitialized_ranges):
            range_item = self.uninitialized_ranges[index]
            if range_item.start >= drain_range.end:
                break

            start = max(range_item.start, drain_range.start)
            end = min(range_item.end, drain_range.end)
            ranges_to_yield.append(Range(start, end))

            index += 1

        # Yield collected ranges
        for r in ranges_to_yield:
            yield r

        # Now modify the uninitialized_ranges list
        num_affected = index - first_index
        if num_affected == 0:
            return

        first_range = self.uninitialized_ranges[first_index]

        # Split one "big" uninitialized range?
        if (
            num_affected == 1
            and first_range.start < drain_range.start
            and first_range.end > drain_range.end
        ):
            # Split into two ranges
            old_start = first_range.start
            self.uninitialized_ranges[first_index] = Range(
                drain_range.end, first_range.end
            )
            self.uninitialized_ranges.insert(
                first_index, Range(old_start, drain_range.start)
            )
        else:
            # Adjust border ranges and delete everything in-between
            remove_start = first_index
            if first_range.start < drain_range.start:
                self.uninitialized_ranges[first_index] = Range(
                    first_range.start, drain_range.start
                )
                remove_start = first_index + 1

            last_range = self.uninitialized_ranges[index - 1]
            remove_end = index
            if last_range.end > drain_range.end:
                self.uninitialized_ranges[index - 1] = Range(
                    drain_range.end, last_range.end
                )
                remove_end = index - 1

            # Remove ranges in between
            del self.uninitialized_ranges[remove_start:remove_end]

    def discard(self, pos: T) -> None:
        """
        Mark a single position as uninitialized.

        This is used when a resource is discarded/invalidated.

        Args:
            pos: The position to mark as uninitialized.
        """
        # Find first range where end >= pos
        r_idx = self._partition_point(lambda r: r.end < pos)

        if r_idx < len(self.uninitialized_ranges):
            r = self.uninitialized_ranges[r_idx]

            # Extend range at end
            if r.end == pos:
                # Check if we can merge with next range
                if r_idx + 1 < len(self.uninitialized_ranges):
                    right = self.uninitialized_ranges[r_idx + 1]
                    if right.start == pos + 1:
                        # Merge ranges
                        self.uninitialized_ranges[r_idx] = Range(r.start, right.end)
                        del self.uninitialized_ranges[r_idx + 1]
                        return

                # Just extend
                self.uninitialized_ranges[r_idx] = Range(r.start, pos + 1)
                return

            # Extend range at beginning
            if r.start > pos:
                if r.start == pos + 1:
                    self.uninitialized_ranges[r_idx] = Range(pos, r.end)
                    return
                else:
                    # Insert new range
                    self.uninitialized_ranges.insert(r_idx, Range(pos, pos + 1))
                    return

            # pos is already in an uninitialized range
            return

        # Add new range at end
        self.uninitialized_ranges.append(Range(pos, pos + 1))

    def _partition_point(self, predicate) -> int:
        """
        Binary search for partition point.

        Returns the index of the first element for which predicate returns False.
        """
        left, right = 0, len(self.uninitialized_ranges)

        while left < right:
            mid = (left + right) // 2
            if predicate(self.uninitialized_ranges[mid]):
                left = mid + 1
            else:
                right = mid

        return left


# Export submodules
from .buffer import BufferInitTracker, BufferInitTrackerAction
from .texture import (
    TextureInitTracker,
    TextureInitTrackerAction,
    TextureInitRange,
    has_copy_partial_init_tracker_coverage,
)

__all__ = [
    "MemoryInitKind",
    "Range",
    "InitTracker",
    "BufferInitTracker",
    "BufferInitTrackerAction",
    "TextureInitTracker",
    "TextureInitTrackerAction",
    "TextureInitRange",
    "has_copy_partial_init_tracker_coverage",
]
