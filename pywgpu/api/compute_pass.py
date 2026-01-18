from typing import Any, Optional, List, TYPE_CHECKING
from pywgpu_types.pass_desc import ComputePassDescriptor

if TYPE_CHECKING:
    from .compute_pipeline import ComputePipeline
    from .bind_group import BindGroup
    from .buffer import Buffer
    from .query_set import QuerySet

class ComputePass:
    """
    In-progress recording of a compute pass.
    
    Created with :meth:`CommandEncoder.begin_compute_pass`.
    """
    
    def __init__(self, inner: Any, descriptor: ComputePassDescriptor, actions: Optional[Any] = None) -> None:
        self._inner = inner
        self._descriptor = descriptor
        self._actions = actions

    def set_pipeline(self, pipeline: 'ComputePipeline') -> None:
        """Sets the active compute pipeline."""
        if hasattr(self._inner, 'set_pipeline'):
            self._inner.set_pipeline(pipeline._inner)
        else:
            raise NotImplementedError("Backend does not support set_pipeline")

    def set_bind_group(
        self, 
        index: int, 
        bind_group: 'BindGroup', 
        offsets: List[int] = []
    ) -> None:
        """Sets the active bind group for a given index."""
        if hasattr(self._inner, 'set_bind_group'):
            self._inner.set_bind_group(index, bind_group._inner, offsets)
        else:
            raise NotImplementedError("Backend does not support set_bind_group")

    def dispatch_workgroups(
        self, 
        workgroup_count_x: int, 
        workgroup_count_y: int = 1, 
        workgroup_count_z: int = 1
    ) -> None:
        """Dispatches workgroups for execution."""
        if hasattr(self._inner, 'dispatch_workgroups'):
            self._inner.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z)
        else:
            raise NotImplementedError("Backend does not support dispatch_workgroups")

    def dispatch_workgroups_indirect(
        self, 
        indirect_buffer: 'Buffer', 
        indirect_offset: int
    ) -> None:
        """Dispatches workgroups using parameters from a buffer."""
        if hasattr(self._inner, 'dispatch_workgroups_indirect'):
            self._inner.dispatch_workgroups_indirect(indirect_buffer._inner, indirect_offset)
        else:
            raise NotImplementedError("Backend does not support dispatch_workgroups_indirect")

    def map_buffer_on_submit(
        self, 
        buffer: 'Buffer', 
        mode: int, 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Schedules a buffer mapping for after the command buffer is submitted."""
        if self._actions:
            from .command_buffer_actions import DeferredBufferMapping
            if size is None:
                size = buffer.size - offset
                
            self._actions.buffer_mappings.append(DeferredBufferMapping(
                buffer=buffer,
                mode=mode,
                offset=offset,
                size=size,
                callback=lambda e: None
            ))

    def on_submitted_work_done(self, callback: Any) -> None:
        """Registers a callback for when the submitted work is done."""
        if self._actions:
            self._actions.on_submitted_work_done_callbacks.append(callback)

    def end(self) -> None:
        """Ends the compute pass."""
        if hasattr(self._inner, 'end'):
            self._inner.end()
        else:
            raise NotImplementedError("Backend does not support end")
