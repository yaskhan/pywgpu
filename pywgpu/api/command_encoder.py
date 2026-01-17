from typing import Optional, TYPE_CHECKING, Any
from pywgpu_types.descriptors import CommandEncoderDescriptor

if TYPE_CHECKING:
    from .command_buffer import CommandBuffer
    from .render_pass import RenderPass
    from .compute_pass import ComputePass
    from .buffer import Buffer
    from .texture import Texture
    from .query_set import QuerySet

class CommandEncoder:
    """
    Encodes commands to a command buffer.
    
    Created with :meth:`Device.create_command_encoder`.
    """
    
    def __init__(self, inner: Any, descriptor: CommandEncoderDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def begin_render_pass(self, descriptor: Any) -> 'RenderPass':
        """Begins recording a render pass."""
        pass

    def begin_compute_pass(self, descriptor: Optional[Any] = None) -> 'ComputePass':
        """Begins recording a compute pass."""
        pass

    def copy_buffer_to_buffer(
        self, 
        source: 'Buffer', 
        source_offset: int, 
        destination: 'Buffer', 
        destination_offset: int, 
        size: int
    ) -> None:
        """Copy data from one buffer to another."""
        pass

    def copy_buffer_to_texture(
        self, 
        source: Any, 
        destination: Any, 
        copy_size: Any
    ) -> None:
        """Copy data from a buffer to a texture."""
        pass

    def copy_texture_to_buffer(
        self, 
        source: Any, 
        destination: Any, 
        copy_size: Any
    ) -> None:
        """Copy data from a texture to a buffer."""
        pass

    def copy_texture_to_texture(
        self, 
        source: Any, 
        destination: Any, 
        copy_size: Any
    ) -> None:
        """Copy data from one texture to another."""
        pass

    def clear_buffer(
        self, 
        buffer: 'Buffer', 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Clears the buffer with zeros."""
        pass

    def resolve_query_set(
        self, 
        query_set: 'QuerySet', 
        first_query: int, 
        query_count: int, 
        destination: 'Buffer', 
        destination_offset: int
    ) -> None:
        """Resolves query results to a buffer."""
        pass

    def finish(self, descriptor: Optional[Any] = None) -> 'CommandBuffer':
        """Finishes recording and returns a CommandBuffer."""
        pass
