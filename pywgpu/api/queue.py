from typing import Any, List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel
from pywgpu_types.texture import Extent3d

if TYPE_CHECKING:
    from .command_buffer import CommandBuffer
    from .buffer import Buffer
    from .texture import Texture
    from .query_set import QuerySet

class ImageCopyTexture(BaseModel):
    texture: Any
    mip_level: int = 0
    origin: Union[List[int], Any] = [0, 0, 0]
    aspect: str = 'all'

class ImageCopyBuffer(BaseModel):
    buffer: Any
    layout: Any

class Queue:
    """
    Handle to a command queue on a device.
    
    A Queue executes recorded :class:`CommandBuffer`s and provides methods for 
    writing to buffers and textures.
    """
    
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def submit(self, command_buffers: List['CommandBuffer']) -> None:
        """
        Submits a series of command buffers for execution.
        
        Args:
            command_buffers: List of command buffers to execute.
        """
        if hasattr(self._inner, 'submit'):
            # Convert command buffers to their inner representations
            command_buffer_inners = [cb._inner for cb in command_buffers]
            self._inner.submit(command_buffer_inners)
        else:
            raise NotImplementedError("Backend does not support submit")

    def write_buffer(self, buffer: 'Buffer', offset: int, data: bytes) -> None:
        """
        Schedule a data write to a buffer.
        
        Args:
            buffer: The buffer to write to.
            offset: Offset in the buffer to start writing at.
            data: The data to write.
        """
        if hasattr(self._inner, 'write_buffer'):
            self._inner.write_buffer(buffer._inner, offset, data)
        else:
            raise NotImplementedError("Backend does not support write_buffer")

    def write_texture(
        self, 
        texture: ImageCopyTexture, 
        data: bytes, 
        data_layout: Any, 
        size: Extent3d
    ) -> None:
        """
        Schedule a data write to a texture.
        
        Args:
            texture: Destination texture + origin/mip/aspect.
            data: The raw data.
            data_layout: Layout of the source data (stride, rows, etc.).
            size: Size of the write.
        """
        pass

    def on_submitted_work_done(self, callback: Any) -> None:
        """
        Registers a callback to be invoked when submitted work is done.
        """
        pass
