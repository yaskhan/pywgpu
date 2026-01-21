from __future__ import annotations
from typing import Any, List, Optional
from dataclasses import dataclass
from .render import RenderPass, RenderPassDescriptor
from .compute import ComputePass, ComputePassDescriptor
from .memory_init import CommandBufferTextureMemoryActions


@dataclass
class CommandBuffer:
    """
    A finished command buffer.
    """

    device: Any
    label: str
    commands: List[Any]


from ..track import Tracker


@dataclass
class EncodingState:
    """
    State applicable when encoding commands.
    
    This structure is passed to encoding functions to provide access to
    the device, encoder, tracker, and memory initialization tracking.
    """
    device: Any
    raw_encoder: Any
    tracker: Any  # Tracker
    buffer_memory_init_actions: List[Any]
    texture_memory_actions: CommandBufferTextureMemoryActions
    snatch_guard: Any
    debug_scope_depth: int


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
        self.texture_memory_actions = CommandBufferTextureMemoryActions()
        self._debug_scope_depth = 0
        # Note: In a real implementation, these would be created from device
        self._raw_encoder = None  # device.hal_device.create_command_encoder(...)
        self._snatch_guard = None  # device.create_snatch_guard()

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

        copy_buffer_to_buffer(self._get_encoding_state(), src, src_offset, dst, dst_offset, size)

    def copy_buffer_to_texture(
        self,
        source: Any,
        destination: Any,
        copy_size: Any,
    ) -> None:
        """
        Copy data from a buffer to a texture.

        Args:
            source: The source buffer info.
            destination: The destination texture info.
            copy_size: The size of the copy.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        from .transfer import copy_buffer_to_texture

        copy_buffer_to_texture(self._get_encoding_state(), source, destination, copy_size)

    def copy_texture_to_buffer(
        self,
        source: Any,
        destination: Any,
        copy_size: Any,
    ) -> None:
        """
        Copy data from a texture to a buffer.

        Args:
            source: The source texture info.
            destination: The destination buffer info.
            copy_size: The size of the copy.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        from .transfer import copy_texture_to_buffer

        copy_texture_to_buffer(self._get_encoding_state(), source, destination, copy_size)

    def copy_texture_to_texture(
        self,
        source: Any,
        destination: Any,
        copy_size: Any,
    ) -> None:
        """
        Copy data between textures.

        Args:
            source: The source texture info.
            destination: The destination texture info.
            copy_size: The size of the copy.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        from .transfer import copy_texture_to_texture

        copy_texture_to_texture(self._get_encoding_state(), source, destination, copy_size)

    def clear_buffer(
        self,
        buffer: Any,
        offset: int = 0,
        size: Optional[int] = None,
    ) -> None:
        """
        Clear a buffer to zero.

        Args:
            buffer: The buffer to clear.
            offset: The offset into the buffer.
            size: The size to clear (None = entire buffer from offset).
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        from .clear import clear_buffer

        clear_buffer(self._get_encoding_state(), buffer, offset, size)

    def clear_texture(
        self,
        texture: Any,
        subresource_range: Any,
    ) -> None:
        """
        Clear a texture.

        Args:
            texture: The texture to clear.
            subresource_range: The subresource range to clear.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        from .clear import clear_texture_cmd

        clear_texture_cmd(self._get_encoding_state(), texture, subresource_range)

    def push_debug_group(self, label: str) -> None:
        """
        Begin a debug group.

        Args:
            label: The label for the debug group.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        self._debug_scope_depth += 1
        # In a real implementation, this would call the HAL encoder
        if self._raw_encoder:
            self._raw_encoder.push_debug_group(label)

    def pop_debug_group(self) -> None:
        """
        End a debug group.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        if self._debug_scope_depth == 0:
            raise RuntimeError("No debug group to pop")
        self._debug_scope_depth -= 1
        # In a real implementation, this would call the HAL encoder
        if self._raw_encoder:
            self._raw_encoder.pop_debug_group()

    def insert_debug_marker(self, label: str) -> None:
        """
        Insert a debug marker.

        Args:
            label: The label for the debug marker.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        # In a real implementation, this would call the HAL encoder
        if self._raw_encoder:
            self._raw_encoder.insert_debug_marker(label)

    def write_timestamp(self, query_set: Any, query_index: int) -> None:
        """
        Write a timestamp.

        Args:
            query_set: The query set.
            query_index: The query index.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        # Validation and HAL call would go here
        if self._raw_encoder:
            self._raw_encoder.write_timestamp(query_set, query_index)

    def resolve_query_set(
        self,
        query_set: Any,
        first_query: int,
        query_count: int,
        destination: Any,
        destination_offset: int,
    ) -> None:
        """
        Resolve query set to buffer.

        Args:
            query_set: The query set.
            first_query: The first query index.
            query_count: The number of queries.
            destination: The destination buffer.
            destination_offset: The offset in the destination buffer.
        """
        if self._status != "Recording":
            raise RuntimeError(f"Cannot record command in status: {self._status}")
        # Validation and HAL call would go here
        if self._raw_encoder:
            self._raw_encoder.resolve_query_set(
                query_set, first_query, query_count, destination, destination_offset
            )

    def _get_encoding_state(self) -> EncodingState:
        """
        Create EncodingState for command encoding.

        Returns:
            The encoding state.
        """
        return EncodingState(
            device=self.device,
            raw_encoder=self._raw_encoder,
            tracker=self.tracker,
            buffer_memory_init_actions=self.buffer_memory_init_actions,
            texture_memory_actions=self.texture_memory_actions,
            snatch_guard=self._snatch_guard,
            debug_scope_depth=self._debug_scope_depth,
        )

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
        
        # Check debug scope depth
        if self._debug_scope_depth > 0:
            raise RuntimeError(f"Unclosed debug groups: {self._debug_scope_depth} remaining")
        
        self._status = "Finished"

        # Initialize buffer memory
        # Note: In a real implementation, this would be done during command submission
        # For now, we'll import the functions but they may not be fully functional
        from .memory_init import initialize_buffer_memory, initialize_texture_memory
        
        # These would be called with proper parameters in a real implementation
        # initialize_buffer_memory(
        #     self._raw_encoder,
        #     self.buffer_memory_init_actions,
        #     self.tracker,
        #     self._snatch_guard,
        # )
        
        # initialize_texture_memory(
        #     self._raw_encoder,
        #     self.texture_memory_actions,
        #     self.tracker,
        #     self.device,
        #     self._snatch_guard,
        # )
        
        # Close the encoder
        if self._raw_encoder:
            # In a real implementation, this would close the HAL encoder
            pass  # self._raw_encoder.close()

        return CommandBuffer(
            device=self.device,
            label=label or self.label,
            commands=self._commands,
        )
