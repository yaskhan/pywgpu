"""
Compute pass encoding.

This module implements the compute pass for wgpu-core, which is used to
record and encode compute shader commands.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from . import base, pass_, compute_command, bind
from .errors import ComputePassError, DispatchError
import pywgpu_types as wgt
from ..id import CommandEncoderId, ComputePipelineId
from ..resource import Buffer
from ..pipeline import ComputePipeline

ZERO_BUFFER_SIZE = 4


class ComputePassDescriptor:
    """
    Descriptor for creating a compute pass.

    Attributes:
        label: Debug label for the compute pass.
        timestamp_writes: Timestamp writes for the pass.
    """

    def __init__(
        self, label: Optional[str] = None, timestamp_writes: Optional[Any] = None
    ):
        self.label = label
        self.timestamp_writes = timestamp_writes


class ComputePass:
    """
    A compute pass for recording compute commands.

    A compute pass is a sequence of compute commands that will be executed on the
    GPU. Compute passes are isolated from each other and from render passes.
    """

    def __init__(
        self, parent: "wgpu_core.command.CommandEncoder", desc: ComputePassDescriptor
    ):
        """
        Create a new compute pass.

        Args:
            parent: The parent command encoder.
            desc: The descriptor for the compute pass.
        """
        self.base = base.BasePass(label=desc.label)
        self.parent = parent
        self.timestamp_writes = desc.timestamp_writes
        self.current_bind_groups = base.BindGroupStateChange()
        self.current_pipeline = base.StateChange()

    @property
    def label(self) -> Optional[str]:
        """Get the label of the compute pass."""
        return self.base.label

    def end(self):
        """

        Ends the compute pass.

        Raises:
            RuntimeError: If the pass has already been ended.
        """
        if self.parent is None:
            raise RuntimeError("ComputePass.end() called on an ended pass.")

        self.parent.end_compute_pass(self)
        self.parent = None

    def set_pipeline(self, pipeline: ComputePipeline):
        """
        Sets the compute pipeline.

        Args:
            pipeline: The compute pipeline to set.
        """
        if self.current_pipeline.set_and_check_redundant(pipeline.id):
            return
        self.base.commands.append(compute_command.SetPipeline(pipeline=pipeline))

    def set_immediate(self, offset: int, data: bytes):
        """
        Sets immediate data.

        Args:
            offset: The byte offset within immediate data storage.
            data: The immediate data to set.
        """
        import struct

        v_offset = len(self.base.immediates_data)
        ints = list(struct.unpack(f"<{len(data)//4}I", data))
        self.base.immediates_data.extend(ints)
        self.base.commands.append(
            compute_command.SetImmediate(
                offset=offset, size_bytes=len(data), values_offset=v_offset
            )
        )

    def set_bind_group(
        self, index: int, bind_group: Any, dynamic_offsets: Optional[List[int]] = None
    ):
        """
        Sets a bind group.

        Args:
            index: The bind group index.
            bind_group: The bind group to set.
            dynamic_offsets: A list of dynamic offsets for dynamic bindings.
        """
        offsets = dynamic_offsets or []
        if self.current_bind_groups.set_and_check_redundant(
            bind_group.id if bind_group else None,
            index,
            self.base.dynamic_offsets,
            offsets,
        ):
            return

        self.base.commands.append(
            compute_command.SetBindGroup(
                index=index, num_dynamic_offsets=len(offsets), bind_group=bind_group
            )
        )

    def dispatch_workgroups(self, groups_x: int, groups_y: int, groups_z: int):
        """
        Dispatches compute workgroups.

        Args:
            groups_x: Number of workgroups in the X dimension.
            groups_y: Number of workgroups in the Y dimension.
            groups_z: Number of workgroups in the Z dimension.
        """
        self.base.commands.append(
            compute_command.Dispatch(workgroups=(groups_x, groups_y, groups_z))
        )

    def dispatch_workgroups_indirect(
        self, indirect_buffer: Buffer, indirect_offset: int
    ):
        """
        Dispatches compute workgroups using parameters from a buffer.

        Args:
            indirect_buffer: The buffer containing dispatch parameters.
            indirect_offset: The offset in bytes into the buffer.
        """
        self.base.commands.append(
            compute_command.DispatchIndirect(
                buffer=indirect_buffer, offset=indirect_offset
            )
        )

    def push_debug_group(self, label: str):
        """
        Push a debug group.

        Args:
            label: The label for the debug group.
        """
        # In a real implementation, string data would be managed by BasePass
        offset = len(self.base.string_data)
        self.base.string_data.extend(label.encode("utf-8"))
        self.base.commands.append(
            compute_command.PushDebugGroup(color=0, len=len(label))
        )

    def pop_debug_group(self):
        """Pop a debug group."""
        self.base.commands.append(compute_command.PopDebugGroup())

    def insert_debug_marker(self, label: str):
        """
        Insert a debug marker.

        Args:
            label: The label for the marker.
        """
        offset = len(self.base.string_data)
        self.base.string_data.extend(label.encode("utf-8"))
        self.base.commands.append(
            compute_command.InsertDebugMarker(color=0, len=len(label))
        )

    def write_timestamp(self, query_set: Any, query_index: int):
        """
        Write a timestamp.

        Args:
            query_set: The query set to write to.
            query_index: The query index.
        """
        self.base.commands.append(
            compute_command.WriteTimestamp(query_set=query_set, query_index=query_index)
        )

    def begin_pipeline_statistics_query(self, query_set: Any, query_index: int):
        """
        Begin a pipeline statistics query.

        Args:
            query_set: The query set to use.
            query_index: The query index.
        """
        self.base.commands.append(
            compute_command.BeginPipelineStatisticsQuery(
                query_set=query_set, query_index=query_index
            )
        )

    def end_pipeline_statistics_query(self):
        """End a pipeline statistics query."""
        self.base.commands.append(compute_command.EndPipelineStatisticsQuery())


class State:
    """Internal state for encoding a compute pass."""

    def __init__(self, pass_state: pass_.PassState):
        self.pipeline: Optional[ComputePipeline] = None
        self.pass_state = pass_state
        self.active_query: Optional[Tuple[Any, int]] = None
        self.immediates: List[int] = []
        self.intermediate_trackers = wgpu_core.track.Tracker()

    def is_ready(self):
        """Checks if the pass is ready for a dispatch call."""
        if self.pipeline is None:
            raise DispatchError("Compute pipeline must be set")
        self.pass_state.binder.check_compatibility(self.pipeline)
        self.pass_state.binder.check_late_buffer_bindings()

    def flush_bindings(
        self, indirect_buffer: Optional[Buffer], track_indirect_buffer: bool
    ):
        """
        Manages resource state transitions and barriers for a dispatch.

        For compute passes, barriers may be needed before each dispatch, so this
        is called before every dispatch command.
        """
        for bind_group in self.pass_state.binder.list_active():
            self.pass_state.scope.merge_bind_group(bind_group.used)

        if indirect_buffer:
            self.pass_state.scope.buffers.merge_single(
                indirect_buffer, wgt.BufferUsages.INDIRECT
            )

        # Merge our per-dispatch scope into the intermediate trackers
        # This will detect transitions compared to previous dispatches in this pass.
        self.intermediate_trackers.merge_scope(self.pass_state.scope)
        # Clear the scope for the next dispatch
        self.pass_state.scope.clear()

        # Drain barriers from the intermediate trackers and apply them to the command encoder.
        drain_barriers(
            self.pass_state.base.raw_encoder,
            self.intermediate_trackers,
            self.pass_state.base.snatch_guard,
        )

        pass_.flush_bindings_helper(self.pass_state)


def drain_barriers(
    raw_encoder: Any, trackers: wgpu_core.track.Tracker, snatch_guard: Any
):
    """Drains pending transitions from trackers and records them as HAL barriers."""
    transitions = trackers.drain_transitions()
    if transitions["buffers"] or transitions["textures"]:
        from .transition_resources import transition_resources

        # We need a dummy state object that has a raw_encoder
        class DummyState:
            def __init__(self, encoder):
                self.raw_encoder = encoder

        transition_resources(
            DummyState(raw_encoder), transitions["buffers"], transitions["textures"]
        )


def insert_barriers_from_tracker(
    raw_encoder: Any,
    base_tracker: wgpu_core.track.Tracker,
    head_tracker: wgpu_core.track.Tracker,
    snatch_guard: Any,
):
    """Merges a tracker into another and records the resulting barriers."""
    transitions = base_tracker.merge_scope(head_tracker)
    if transitions["buffers"] or transitions["textures"]:
        from .transition_resources import transition_resources

        class DummyState:
            def __init__(self, encoder):
                self.raw_encoder = encoder

        transition_resources(
            DummyState(raw_encoder), transitions["buffers"], transitions["textures"]
        )


def encode_compute_pass(parent_state, base_pass, timestamp_writes):
    """
    High-level function to encode a compute pass into a HAL command buffer.

    This function iterates through the recorded commands, validates them, manages
    resource state, and records the low-level HAL commands.
    """
    device = parent_state.device

    # Close any open pass if necessary
    parent_state.raw_encoder.close_if_open()

    raw_encoder = parent_state.raw_encoder.open_pass(base_pass.label)

    debug_scope_depth = [0]  # Using a list for mutable reference

    state = State(
        pass_.PassState(
            base=parent_state,
            binder=bind.Binder(),
            temp_offsets=[],
            dynamic_offset_count=0,
            pending_discard_init_fixups=[],
            scope=device.new_usage_scope(),
            string_offset=0,
        )
    )
    # Note: in a real implementation we might need to set tracker sizes

    hal_timestamp_writes = None
    if timestamp_writes:
        query_set = state.pass_state.base.tracker.query_sets.insert_single(
            timestamp_writes.query_set
        )
        # Reset queries if needed (simplified)
        raw_encoder.reset_queries(query_set.raw(), range(0, 1))  # Placeholder range

        from pywgpu_hal import lib as hal

        hal_timestamp_writes = hal.PassTimestampWrites(
            query_set=query_set.raw(),
            beginning_of_pass_write_index=timestamp_writes.beginning_of_pass_write_index,
            end_of_pass_write_index=timestamp_writes.end_of_pass_write_index,
        )

    hal_desc = wgt.pass_desc.ComputePassDescriptor(
        label=base_pass.label, timestamp_writes=hal_timestamp_writes
    )
    raw_encoder.begin_compute_pass(hal_desc)

    for command in base_pass.commands:
        if isinstance(command, compute_command.SetBindGroup):
            pass_.set_bind_group(
                state.pass_state,
                device,
                base_pass.dynamic_offsets,
                command.index,
                command.num_dynamic_offsets,
                command.bind_group,
                is_compute=True,
            )
        elif isinstance(command, compute_command.SetPipeline):
            set_pipeline(state, device, command.pipeline)
        elif isinstance(command, compute_command.SetImmediate):
            pass_.set_immediates(
                state.pass_state,
                base_pass.immediates_data,
                command.offset,
                command.size_bytes,
                command.values_offset,
                lambda data: None,
            )
        elif isinstance(command, compute_command.Dispatch):
            dispatch(state, command.workgroups)
        elif isinstance(command, compute_command.DispatchIndirect):
            dispatch_indirect(state, device, command.buffer, command.offset)
        elif isinstance(command, compute_command.PushDebugGroup):
            pass_.push_debug_group(state.pass_state, base_pass.string_data, command.len)
        elif isinstance(command, compute_command.PopDebugGroup):
            pass_.pop_debug_group(state.pass_state)
        elif isinstance(command, compute_command.InsertDebugMarker):
            pass_.insert_debug_marker(
                state.pass_state, base_pass.string_data, command.len
            )
        elif isinstance(command, compute_command.WriteTimestamp):
            pass_.write_timestamp(
                state.pass_state, device, None, command.query_set, command.query_index
            )
        elif isinstance(command, compute_command.BeginPipelineStatisticsQuery):
            from .query import validate_and_begin_pipeline_statistics_query

            validate_and_begin_pipeline_statistics_query(
                command.query_set,
                state.pass_state.base.raw_encoder,
                state.pass_state.base.tracker.query_sets,
                device,
                command.query_index,
                None,
                state.active_query,
            )
        elif isinstance(command, compute_command.EndPipelineStatisticsQuery):
            from .query import end_pipeline_statistics_query

            end_pipeline_statistics_query(
                state.pass_state.base.raw_encoder, state.active_query
            )

    raw_encoder.end_compute_pass()
    parent_state.raw_encoder.close()

    # Pre-pass barriers and surface fixups
    from .memory_init import fixup_discarded_surfaces

    transit = parent_state.raw_encoder.open_pass("(pywgpu internal) Pre Pass")

    fixup_discarded_surfaces(
        device,
        state.pass_state.pending_discard_init_fixups,
        transit,
        parent_state.tracker.textures,
        parent_state.snatch_guard,
    )

    # Insert barriers from tracker
    insert_barriers_from_tracker(
        transit,
        parent_state.tracker,
        state.intermediate_trackers,
        parent_state.snatch_guard,
    )
    parent_state.raw_encoder.close_and_swap()


def set_pipeline(state: State, device, pipeline: ComputePipeline):
    """Sets the compute pipeline in the pass state."""
    pipeline.same_device(device)
    state.pipeline = pipeline

    raw_pipeline = state.pass_state.base.tracker.compute_pipelines.insert_single(
        pipeline
    )
    state.pass_state.base.raw_encoder.set_compute_pipeline(raw_pipeline.raw())

    pass_.change_pipeline_layout(state.pass_state, pipeline.layout)


def dispatch(state: State, groups: List[int]):
    """Validates and records a dispatch command."""
    state.is_ready()
    state.flush_bindings(None, False)

    limits = state.pass_state.base.device.limits
    if any(g > limits.max_compute_workgroups_per_dimension for g in groups):
        raise DispatchError(f"Invalid group size: {groups}")

    state.pass_state.base.raw_encoder.dispatch(groups)


def dispatch_indirect(state: State, device: Any, buffer: Buffer, offset: int):
    """Validates and records an indirect dispatch command."""
    buffer.same_device(device)
    state.is_ready()

    buffer.check_usage(wgt.BufferUsages.INDIRECT)
    if offset % 4 != 0:
        raise ComputePassError("Unaligned indirect buffer offset")

    # In a full implementation, validation and barrier logic would be here.
    state.flush_bindings(buffer, True)

    raw_buffer = buffer.raw()
    state.pass_state.base.raw_encoder.dispatch_indirect(raw_buffer, offset)
