from typing import Optional, TYPE_CHECKING, Any, Union
from pywgpu_types.descriptors import CommandEncoderDescriptor
from pywgpu_types.queue import ImageCopyBuffer, ImageCopyTexture
from pywgpu_types.texture import Extent3d

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
        from .render_pass import RenderPass
        if hasattr(self._inner, 'begin_render_pass'):
            render_pass_inner = self._inner.begin_render_pass(descriptor)
            return RenderPass(render_pass_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support begin_render_pass")

    def begin_compute_pass(self, descriptor: Optional[Any] = None) -> 'ComputePass':
        """Begins recording a compute pass."""
        from .compute_pass import ComputePass
        if hasattr(self._inner, 'begin_compute_pass'):
            compute_pass_inner = self._inner.begin_compute_pass(descriptor)
            return ComputePass(compute_pass_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support begin_compute_pass")

    def copy_buffer_to_buffer(
        self, 
        source: 'Buffer', 
        source_offset: int, 
        destination: 'Buffer', 
        destination_offset: int, 
        size: int
    ) -> None:
        """Copy data from one buffer to another."""
        if hasattr(self._inner, 'copy_buffer_to_buffer'):
            self._inner.copy_buffer_to_buffer(source._inner, source_offset, destination._inner, destination_offset, size)
        else:
            raise NotImplementedError("Backend does not support copy_buffer_to_buffer")

    def copy_buffer_to_texture(
        self, 
        source: ImageCopyBuffer, 
        destination: ImageCopyTexture, 
        copy_size: Extent3d
    ) -> None:
        """Copy data from a buffer to a texture."""
        if hasattr(self._inner, 'copy_buffer_to_texture'):
            self._inner.copy_buffer_to_texture(source, destination, copy_size)
        else:
            raise NotImplementedError("Backend does not support copy_buffer_to_texture")

    def copy_texture_to_buffer(
        self, 
        source: ImageCopyTexture, 
        destination: ImageCopyBuffer, 
        copy_size: Extent3d
    ) -> None:
        """Copy data from a texture to a buffer."""
        if hasattr(self._inner, 'copy_texture_to_buffer'):
            self._inner.copy_texture_to_buffer(source, destination, copy_size)
        else:
            raise NotImplementedError("Backend does not support copy_texture_to_buffer")

    def copy_texture_to_texture(
        self, 
        source: ImageCopyTexture, 
        destination: ImageCopyTexture, 
        copy_size: Extent3d
    ) -> None:
        """Copy data from one texture to another."""
        if hasattr(self._inner, 'copy_texture_to_texture'):
            self._inner.copy_texture_to_texture(source, destination, copy_size)
        else:
            raise NotImplementedError("Backend does not support copy_texture_to_texture")

    def clear_buffer(
        self, 
        buffer: 'Buffer', 
        offset: int = 0, 
        size: Optional[int] = None
    ) -> None:
        """Clears the buffer with zeros."""
        if hasattr(self._inner, 'clear_buffer'):
            self._inner.clear_buffer(buffer._inner, offset, size)
        else:
            raise NotImplementedError("Backend does not support clear_buffer")

    def build_acceleration_structures(
        self, 
        blas: Optional[Union['BlasBuildEntry', List['BlasBuildEntry']]] = None,
        tlas: Optional[Union['Tlas', List['Tlas']]] = None
    ) -> None:
        """Builds acceleration structures."""
        if hasattr(self._inner, 'build_acceleration_structures'):
            self._inner.build_acceleration_structures(blas, tlas)
        else:
            raise NotImplementedError("Backend does not support build_acceleration_structures")

    def resolve_query_set(
        self, 
        query_set: 'QuerySet', 
        first_query: int, 
        query_count: int, 
        destination: 'Buffer', 
        destination_offset: int
    ) -> None:
        """Resolves query results to a buffer."""
        if hasattr(self._inner, 'resolve_query_set'):
            self._inner.resolve_query_set(query_set._inner, first_query, query_count, destination._inner, destination_offset)
        else:
            raise NotImplementedError("Backend does not support resolve_query_set")

    def finish(self, descriptor: Optional[Any] = None) -> 'CommandBuffer':
        """Finishes recording and returns a CommandBuffer."""
        from .command_buffer import CommandBuffer
        if hasattr(self._inner, 'finish'):
            command_buffer_inner = self._inner.finish(descriptor)
            return CommandBuffer(command_buffer_inner)
        else:
            raise NotImplementedError("Backend does not support finish")
