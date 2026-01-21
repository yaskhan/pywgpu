"""
Bind group management for command encoding.

This module implements bind group management for command encoders. It provides:
- Binder: Manages bind group bindings and compatibility checking
- BindGroupStateChange: Tracks bind group state changes
- Compatibility checking between bind groups and pipelines

The binder ensures that bind groups are compatible with the current pipeline
layout and validates late buffer binding sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

from . import errors


@dataclass
class BinderError(Exception):
    """
    Error related to bind group binding.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class LateBufferBinding:
    """
    Late buffer binding information.

    Attributes:
        binding_index: The binding index.
        shader_expect_size: Expected size from shader.
        bound_size: Actual bound size.
    """

    binding_index: int
    shader_expect_size: int
    bound_size: int


@dataclass
class EntryPayload:
    """
    Payload for a bind group entry.

    Attributes:
        group: The bind group.
        dynamic_offsets: Dynamic offsets for the bind group.
        late_buffer_bindings: Late buffer binding information.
        late_bindings_effective_count: Number of effective late bindings.
    """

    group: Optional[Any] = None
    dynamic_offsets: List[int] = None
    late_buffer_bindings: List[LateBufferBinding] = None
    late_bindings_effective_count: int = 0

    def __post_init__(self):
        if self.dynamic_offsets is None:
            self.dynamic_offsets = []
        if self.late_buffer_bindings is None:
            self.late_buffer_bindings = []

    def reset(self) -> None:
        """Reset the payload."""
        self.group = None
        self.dynamic_offsets.clear()
        self.late_buffer_bindings.clear()
        self.late_bindings_effective_count = 0


class Binder:
    """
    Manages bind group bindings and compatibility checking.

    The binder tracks bind group bindings and ensures compatibility with
    the current pipeline layout. It also validates late buffer binding sizes.

    Attributes:
        pipeline_layout: The current pipeline layout.
        manager: Compatibility manager for bind group layouts.
        payloads: Payloads for each bind group slot.
    """

    def __init__(self) -> None:
        """Initialize the binder."""
        self.pipeline_layout: Optional[Any] = None
        self.manager = BoundBindGroupLayouts()
        self.payloads: List[EntryPayload] = [
            EntryPayload() for _ in range(8)  # MAX_BIND_GROUPS
        ]

    def reset(self) -> None:
        """Reset the binder."""
        self.pipeline_layout = None
        self.manager = BoundBindGroupLayouts()
        for payload in self.payloads:
            payload.reset()

    def change_pipeline_layout(
        self,
        new: Any,
        late_sized_buffer_groups: List[Any],
    ) -> bool:
        """
        Change the pipeline layout.

        Returns True if the layout was actually changed.

        Args:
            new: The new pipeline layout.
            late_sized_buffer_groups: Late sized buffer groups.

        Returns:
            True if the layout changed, False otherwise.
        """
        if self.pipeline_layout is not None:
            if self.pipeline_layout.is_equal(new):
                return False

        old = self.pipeline_layout
        self.pipeline_layout = new

        self.manager.update_expectations(new.bind_group_layouts)

        # Update buffer binding sizes required by shaders
        for payload, late_group in zip(self.payloads, late_sized_buffer_groups):
            payload.late_bindings_effective_count = len(late_group.shader_sizes)
            # Update entries that already exist
            for late_binding, shader_expect_size in zip(
                payload.late_buffer_bindings,
                late_group.shader_sizes,
            ):
                late_binding.shader_expect_size = shader_expect_size
            # Add new entries
            if len(late_group.shader_sizes) > len(payload.late_buffer_bindings):
                for shader_expect_size in late_group.shader_sizes[
                    len(payload.late_buffer_bindings) :
                ]:
                    payload.late_buffer_bindings.append(
                        LateBufferBinding(
                            binding_index=0,
                            shader_expect_size=shader_expect_size,
                            bound_size=0,
                        )
                    )

        if old is not None:
            # root constants are the base compatibility property
            if old.immediate_size != new.immediate_size:
                self.manager.update_start_index(0)

        return True

    def assign_group(
        self,
        index: int,
        bind_group: Any,
        offsets: List[int],
    ) -> None:
        """
        Assign a bind group to a slot.

        Args:
            index: The slot index.
            bind_group: The bind group to assign.
            offsets: Dynamic offsets for the bind group.
        """
        payload = self.payloads[index]
        payload.group = bind_group
        payload.dynamic_offsets.clear()
        payload.dynamic_offsets.extend(offsets)

        # Fill out actual binding sizes for buffers
        # Update entries that already exist
        for late_binding, late_info in zip(
            payload.late_buffer_bindings,
            bind_group.late_buffer_binding_infos,
        ):
            late_binding.binding_index = late_info.binding_index
            late_binding.bound_size = late_info.size.get()

        # Add new entries
        if len(bind_group.late_buffer_binding_infos) > len(
            payload.late_buffer_bindings
        ):
            for late_info in bind_group.late_buffer_binding_infos[
                len(payload.late_buffer_bindings) :
            ]:
                payload.late_buffer_bindings.append(
                    LateBufferBinding(
                        binding_index=late_info.binding_index,
                        shader_expect_size=0,
                        bound_size=late_info.size.get(),
                    )
                )

        self.manager.assign(index, bind_group.layout)

    def take_rebind_range(self) -> range:
        """
        Get the range of entries that needs to be rebound.

        Returns:
            The range of entries.
        """
        return self.manager.take_rebind_range()

    def entries(self, range_: range) -> Iterator[tuple[int, EntryPayload]]:
        """
        Get entries for a range of indices.

        Args:
            range_: The range of indices.

        Returns:
            Iterator of (index, payload) pairs.
        """
        for i in range_:
            yield i, self.payloads[i]

    def list_active(self) -> List[Any]:
        """
        List active bind groups.

        Returns:
            List of active bind groups.
        """
        return [
            self.payloads[index].group
            for index in self.manager.list_active()
            if self.payloads[index].group is not None
        ]

    def list_valid(self) -> List[tuple[int, EntryPayload]]:
        """
        List valid bind group entries.

        Returns:
            List of (index, payload) pairs.
        """
        return [
            (i, self.payloads[i])
            for i in range(len(self.payloads))
            if self.manager.entries[i].is_valid()
        ]

    def check_compatibility(self, pipeline: Any) -> None:
        """
        Check compatibility with a pipeline.

        Args:
            pipeline: The pipeline to check against.

        Raises:
            BinderError: If incompatible.
        """
        self.manager.get_invalid()

    def check_late_buffer_bindings(self) -> None:
        """
        Check late buffer bindings.

        Raises:
            LateMinBufferBindingSizeMismatch: If binding size is too small.
        """
        for group_index in self.manager.list_active():
            payload = self.payloads[group_index]
            for late_binding in payload.late_buffer_bindings[
                : payload.late_bindings_effective_count
            ]:
                if late_binding.bound_size < late_binding.shader_expect_size:
                    raise errors.LateMinBufferBindingSizeMismatch(
                        group_index=group_index,
                        binding_index=late_binding.binding_index,
                        shader_size=late_binding.shader_expect_size,
                        bound_size=late_binding.bound_size,
                    )


@dataclass
class BoundBindGroupLayouts:
    """
    Manages bound bind group layouts.

    Attributes:
        entries: List of entries.
        rebind_start: Start index for rebind.
    """

    entries: List[Any] = None
    rebind_start: int = 0

    def __post_init__(self):
        if self.entries is None:
            self.entries = [Entry() for _ in range(8)]  # MAX_BIND_GROUPS

    def num_valid_entries(self) -> int:
        """Get the number of valid entries."""
        for i, entry in enumerate(self.entries):
            if entry.is_incompatible():
                return i
        return len(self.entries)

    def take_rebind_range(self) -> range:
        """Get the range of entries that needs to be rebound."""
        end = self.num_valid_entries()
        start = self.rebind_start
        self.rebind_start = end
        return range(start, max(end, start))

    def update_start_index(self, start_index: int) -> None:
        """Update the start index."""
        self.rebind_start = min(self.rebind_start, start_index)

    def update_expectations(self, expectations: List[Any]) -> None:
        """Update expectations for bind group layouts."""
        start_index = 0
        for i, (entry, expect) in enumerate(zip(self.entries, expectations)):
            if entry.expected is None or not entry.expected.is_equal(expect):
                start_index = i
                break
        else:
            start_index = len(expectations)

        for i in range(start_index, len(expectations)):
            self.entries[i].expected = expectations[i]

        for i in range(len(expectations), len(self.entries)):
            self.entries[i].expected = None

        self.update_start_index(start_index)

    def assign(self, index: int, value: Any) -> None:
        """Assign a bind group layout."""
        self.entries[index].assigned = value
        self.update_start_index(index)

    def list_active(self) -> List[int]:
        """List active entries."""
        return [i for i, entry in enumerate(self.entries) if entry.is_active()]

    def get_invalid(self) -> None:
        """Check for invalid entries."""
        for index, entry in enumerate(self.entries):
            entry.check()


@dataclass
class Entry:
    """
    Entry for bind group layout.

    Attributes:
        assigned: Assigned bind group layout.
        expected: Expected bind group layout.
    """

    assigned: Optional[Any] = None
    expected: Optional[Any] = None

    def is_active(self) -> bool:
        """Check if entry is active."""
        return self.assigned is not None and self.expected is not None

    def is_valid(self) -> bool:
        """Check if entry is valid."""
        if self.expected is not None:
            if self.assigned is not None:
                return self.expected.is_equal(self.assigned)
            return False
        return True

    def is_incompatible(self) -> bool:
        """Check if entry is incompatible."""
        return self.expected is None or not self.is_valid()

    def check(self) -> None:
        """Check entry compatibility."""
        if self.expected is not None:
            if self.assigned is None:
                raise BinderError("Missing bind group")
            if not self.expected.is_equal(self.assigned):
                raise BinderError("Incompatible bind group")


@dataclass
class BindGroupStateChange:
    """
    Tracks bind group state changes.

    Attributes:
        current: Current bind group indices.
    """

    current: List[Optional[int]] = None

    def __post_init__(self):
        if self.current is None:
            self.current = [None] * 8  # MAX_BIND_GROUPS

    def reset(self) -> None:
        """Reset the state."""
        self.current = [None] * 8
