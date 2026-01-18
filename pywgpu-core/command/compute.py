"""
Compute pass encoding.

This module implements compute pass encoding for wgpu-core. It provides:
- ComputePass: A pass for recording compute commands
- ComputePassDescriptor: Descriptor for creating a compute pass
- Dispatch commands for executing compute shaders

Compute passes are used to record compute shader commands that will be
executed on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from . import errors


@dataclass
class ComputePassDescriptor:
    """
    Descriptor for creating a compute pass.
    
    Attributes:
        label: Debug label for the compute pass.
        timestamp_writes: Timestamp writes for the pass.
    """

    label: Optional[str] = None
    timestamp_writes: Optional[Any] = None


@dataclass
class ComputePass:
    """
    A compute pass for recording compute commands.
    
    A compute pass is a sequence of compute commands that will be executed
    on the GPU. Compute passes are isolated from each other and from render
    passes.
    
    Attributes:
        base: Base pass data.
        parent: Parent command encoder.
        timestamp_writes: Timestamp writes for the pass.
        current_bind_groups: Bind group state change tracking.
        current_pipeline: Pipeline state change tracking.
    """

    def __init__(
        self,
        parent: Any,
        desc: ComputePassDescriptor,
    ) -> None:
        """
        Create a new compute pass.
        
        Args:
            parent: The parent command encoder.
            desc: The descriptor for the compute pass.
        """
        self.base = BasePass()
        self.parent = parent
        self.timestamp_writes = desc.timestamp_writes
        self.current_bind_groups = BindGroupStateChange()
        self.current_pipeline = StateChange()

    def label(self) -> Optional[str]:
        """Get the label of the compute pass."""
        return self.base.label

    def end(self) -> None:
        """
        End the compute pass.
        
        Raises:
            RuntimeError: If the pass has already ended.
        """
        if self.parent is None:
            raise RuntimeError("Pass already ended")
        
        # Unlock encoder and process recorded commands
        self.parent._unlock_encoder()
        self.parent = None

    def set_pipeline(self, pipeline: Any) -> None:
        """
        Set the compute pipeline.
        
        Args:
            pipeline: The compute pipeline to set.
        """
        if self.current_pipeline.current == pipeline:
            return
        self.current_pipeline.current = pipeline
        self.base.commands.append(("SetPipeline", pipeline))

    def set_bind_group(
        self,
        index: int,
        bind_group: Any,
        dynamic_offsets: Optional[List[int]] = None,
    ) -> None:
        """
        Set the bind group.
        
        Args:
            index: The bind group index.
            bind_group: The bind group to set.
            dynamic_offsets: The dynamic offsets.
        """
        self.current_bind_groups.current[index] = bind_group
        self.base.commands.append(("SetBindGroup", index, bind_group, dynamic_offsets))

    def dispatch_workgroups(
        self,
        groups_x: int,
        groups_y: int,
        groups_z: int,
    ) -> None:
        """
        Dispatch a compute workgroup.
        
        Args:
            groups_x: Number of workgroups in X dimension.
            groups_y: Number of workgroups in Y dimension.
            groups_z: Number of workgroups in Z dimension.
        """
        self.base.commands.append(("Dispatch", groups_x, groups_y, groups_z))

    def dispatch_workgroups_indirect(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
    ) -> None:
        """
        Dispatch a compute workgroup indirectly.
        
        Args:
            indirect_buffer: The buffer containing dispatch parameters.
            indirect_offset: The offset into the buffer.
        """
        self.base.commands.append(("DispatchIndirect", indirect_buffer, indirect_offset))


@dataclass
class DispatchError:
    """
    Error related to dispatch operations.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ComputePassError:
    """
    Error encountered during compute pass encoding.
    
    Attributes:
        scope: The error scope.
        inner: The inner error.
    """

    scope: Any
    inner: Any

    def __str__(self) -> str:
        return f"{self.scope}: {self.inner}"


@dataclass
class BasePass:
    """
    Base pass data.
    
    Attributes:
        label: Debug label.
        error: Error if any.
        commands: List of commands.
        dynamic_offsets: Dynamic offsets.
        string_data: String data for debug markers.
        immediates_data: Immediates data.
    """

    label: Optional[str] = None
    error: Optional[Any] = None
    commands: List[Any] = None
    dynamic_offsets: List[int] = None
    string_data: bytes = b""
    immediates_data: List[int] = None

    def __post_init__(self):
        if self.commands is None:
            self.commands = []
        if self.dynamic_offsets is None:
            self.dynamic_offsets = []
        if self.immediates_data is None:
            self.immediates_data = []

    def take(self) -> Any:
        """Take the pass data."""
        return self


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


@dataclass
class StateChange:
    """
    Tracks state changes.
    
    Attributes:
        current: Current state.
    """

    current: Optional[Any] = None
