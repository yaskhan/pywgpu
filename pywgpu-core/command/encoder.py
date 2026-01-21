from __future__ import annotations
from typing import Any, List, Optional
from .render import RenderPass, RenderPassDescriptor
from .compute import ComputePass, ComputePassDescriptor


@dataclass
class CommandBuffer:
    """
    A finished command buffer.
    """

    device: Any
    label: str
    commands: List[Any]


from ..track import Tracker


class CommandEncoder:
    """
    A command encoder for recording GPU commands.
    """

    def __init__(self, device: Any, label: str = ""):
        self.device = device
        self.label = label
        self._status = "Recording"
        self._commands: List[Any] = []
        self.tracker = Tracker()
        self.buffer_memory_init_actions: List[Any] = []
        self.texture_memory_actions: Any = None  # CommandBufferTextureMemoryActions()

    def begin_render_pass(self, desc: RenderPassDescriptor) -> RenderPass:
        """
        Begin a render pass.

        Args:
            desc: The descriptor for the render pass.

        Returns:
            The started render pass.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot begin render pass in status: {self._status}")
        self._status = "Locked"
        return RenderPass(self, desc)

    def begin_compute_pass(self, desc: ComputePassDescriptor) -> ComputePass:
        """
        Begin a compute pass.

        Args:
            desc: The descriptor for the compute pass.

        Returns:
            The started compute pass.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot begin compute pass in status: {self._status}")
        self._status = "Locked"
        return ComputePass(self, desc)

    def invalidate(self, error: Exception):
        """Invalidate the encoder due to an error."""
        self._status = "Invalid"
        self._error = error

    def _unlock_encoder(self):
        """Unlock the encoder (called by passes when they end)."""
        if self._status != "Locked":
            raise RuntimeError(f"Cannot unlock encoder in status: {self._status}")
        self._status = "Recording"

    def copy_buffer_to_buffer(
        self,
        src: Any,
        src_offset: int,
        dst: Any,
        dst_offset: int,
        size: int,
    ) -> None:
        """
        Copy data between buffers.

        Args:
            src: The source buffer.
            src_offset: The offset in the source buffer.
            dst: The destination buffer.
            dst_offset: The offset in the destination buffer.
            size: The size of the data to copy.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        from .transfer import copy_buffer_to_buffer

        copy_buffer_to_buffer(self, src, src_offset, dst, dst_offset, size)

    def finish(self, label: str = "") -> CommandBuffer:
        """
        Finish recording and return a CommandBuffer.

        Args:
            label: Optional label for the command buffer.

        Returns:
            The finished command buffer.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot finish encoder in status: {self._status}")
        self._status = "Finished"

        # In Rust, this is where encode_commands is called if it wasn't already.
        # It also handles memory initialization.
        from .memory_init import initialize_buffer_memory, initialize_texture_memory

        # initialize_buffer_memory(...)
        # initialize_texture_memory(...)

        return CommandBuffer(
            device=self.device, label=label or self.label, commands=self._commands
        )
