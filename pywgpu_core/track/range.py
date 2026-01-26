"""
Range-based state tracking.

This module implements a structure that keeps track of an I -> T mapping,
optimized for cases where keys of the same values are often grouped together
linearly.
"""

from __future__ import annotations

from typing import Generic, Iterator, List, Tuple, TypeVar

I = TypeVar("I", bound=int)
T = TypeVar("T")


class RangedStates(Generic[I, T]):
    """
    Structure that keeps track of a I -> T mapping.

    Optimized for a case where keys of the same values are often grouped
    together linearly.

    Attributes:
        ranges: List of ranges, each associated with a single value.
                Ranges of keys have to be non-intersecting and ordered.
    """

    def __init__(self, ranges: List[Tuple[range, T]]) -> None:
        """
        Initialize ranged states.

        Args:
            ranges: List of (range, value) pairs.
        """
        self.ranges: List[Tuple[range, T]] = ranges

    @classmethod
    def from_range(cls, rng: range, value: T) -> RangedStates[I, T]:
        """
        Construct a new instance from a single range.

        Args:
            rng: The range.
            value: The value for the range.

        Returns:
            A new RangedStates instance.
        """
        return cls([(rng, value)])

    @classmethod
    def from_slice(cls, values: List[Tuple[range, T]]) -> RangedStates[I, T]:
        """
        Construct a new instance from a slice of ranges.

        Args:
            values: List of (range, value) pairs.

        Returns:
            A new RangedStates instance.
        """
        return cls(list(values))

    def iter(self) -> Iterator[Tuple[range, T]]:
        """
        Iterate over all ranges.

        Returns:
            An iterator over (range, value) pairs.
        """
        return iter(self.ranges)

    def iter_mut(self) -> Iterator[Tuple[range, T]]:
        """
        Iterate mutably over all ranges.

        Returns:
            An iterator over (range, value) pairs.
        """
        return iter(self.ranges)

    def check_sanity(self) -> None:
        """
        Check that all the ranges are non-intersecting and ordered.

        Raises:
            AssertionError: If the ranges are invalid.
        """
        for rng, _ in self.ranges:
            assert rng.start < rng.stop, f"Invalid range: {rng}"

        for i in range(len(self.ranges) - 1):
            a = self.ranges[i]
            b = self.ranges[i + 1]
            assert a[0].stop <= b[0].start, f"Overlapping ranges: {a[0]} and {b[0]}"

    def coalesce(self) -> None:
        """Merge neighboring ranges together where possible."""
        if not self.ranges:
            return

        num_removed = 0
        i = 0
        while i < len(self.ranges) - 1:
            cur_range, cur_val = self.ranges[i]
            next_range, next_val = self.ranges[i + 1]

            if cur_range.stop == next_range.start and cur_val == next_val:
                # Merge the ranges
                self.ranges[i] = (
                    range(cur_range.start, next_range.stop),
                    cur_val,
                )
                # Mark next range as empty (to be removed)
                self.ranges[i + 1] = (
                    range(next_range.start, next_range.start),
                    next_val,
                )
                num_removed += 1
            else:
                i += 1

        if num_removed > 0:
            self.ranges = [(r, v) for r, v in self.ranges if r.start != r.stop]

    def iter_filter(self, rng: range) -> Iterator[Tuple[range, T]]:
        """
        Iterate over ranges that intersect with the given range.

        Args:
            rng: The range to filter by.

        Returns:
            An iterator over (range, value) pairs that intersect.
        """
        for inner_range, value in self.ranges:
            if inner_range.stop > rng.start and inner_range.start < rng.stop:
                new_start = max(inner_range.start, rng.start)
                new_stop = min(inner_range.stop, rng.stop)
                yield (range(new_start, new_stop), value)

    def isolate(self, index: range, default: T) -> List[Tuple[range, T]]:
        """
        Split the storage ranges to occupy exactly the index range.

        Gaps in the ranges are filled with the default value.

        Args:
            index: The range to isolate.
            default: The default value for gaps.

        Returns:
            A mutable slice of ranges covering the index range.
        """
        # Find the starting position
        start_pos = None
        for i, (rng, _) in enumerate(self.ranges):
            if rng.stop > index.start:
                start_pos = i
                break

        if start_pos is None:
            # No ranges overlap, insert at the end
            pos = len(self.ranges)
            self.ranges.insert(pos, (index, default))
            return [self.ranges[pos]]

        # Handle the start of the range
        if self.ranges[start_pos][0].start < index.start:
            # Split the first range
            old_range, old_val = self.ranges[start_pos]
            before = range(old_range.start, index.start)
            after = range(index.start, old_range.stop)
            self.ranges[start_pos] = (before, old_val)
            self.ranges.insert(start_pos + 1, (after, old_val))
            start_pos += 1

        # Find or create the end position
        end_pos = start_pos
        for i in range(start_pos, len(self.ranges)):
            rng, _ = self.ranges[i]
            if rng.start >= index.stop:
                break
            if rng.stop > index.stop:
                # Split this range
                old_range, old_val = self.ranges[i]
                before = range(old_range.start, index.stop)
                after = range(index.stop, old_range.stop)
                self.ranges[i] = (before, old_val)
                self.ranges.insert(i + 1, (after, old_val))
                end_pos = i + 1
                break
            end_pos = i + 1

        # Fill gaps with default value
        current = index.start
        i = start_pos
        while i < end_pos:
            rng, _ = self.ranges[i]
            if rng.start > current:
                # Insert gap
                gap = range(current, rng.start)
                self.ranges.insert(i, (gap, default))
                end_pos += 1
                i += 1
            current = rng.stop
            i += 1

        # Fill final gap if needed
        if current < index.stop:
            gap = range(current, index.stop)
            self.ranges.insert(end_pos, (gap, default))
            end_pos += 1

        return self.ranges[start_pos:end_pos]


__all__ = ["RangedStates"]
