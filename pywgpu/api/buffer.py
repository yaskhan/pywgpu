from typing import Optional, TYPE_CHECKING, Any, Union
from pywgpu_types.descriptors import BufferDescriptor
from enum import Enum

if TYPE_CHECKING:
    from .device import Device

class BufferMapState(Enum):
    """State of a buffer's mapping."""
    UNMAPPED = 0
    PENDING = 1
    MAPPED = 2

class Buffer:
    """
    Handle to a GPU-accessible buffer.
    
    Created with :meth:`Device.create_buffer`.
    """
    
    def __init__(self, inner: Any, descriptor: BufferDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    @property
    def size(self) -> int:
        """The size of the buffer in bytes."""
        return self._descriptor.get('size', 0)

    @property
    def usage(self) -> int:
        """The allowed usage for this buffer."""
        return self._descriptor.get('usage', 0)

    @property
    def map_state(self) -> BufferMapState:
        """The current mapping state of the buffer."""
        pass

    def slice(self, start: int = 0, end: Optional[int] = None) -> 'BufferSlice':
        """
        Creates a slice of this buffer.
        
        Args:
            start: Start offset in bytes.
            end: End offset in bytes. If None, goes to the end of the buffer.
            
        Returns:
            A BufferSlice representing the specified range.
        """
        pass

    def map_async(self, mode: int, start: int = 0, end: Optional[int] = None, callback: Any = None) -> None:
        """
        Maps the buffer or a range of it asynchronously.
        
        Args:
            mode: Map mode (READ or WRITE).
            start: Start offset.
            end: End offset.
            callback: Function called when mapping completes.
        """
        pass

    def get_mapped_range(self, start: int = 0, end: Optional[int] = None) -> memoryview:
        """
        Returns a memoryview to the mapped range.
        
        The buffer must be mapped.
        """
        pass

    def unmap(self) -> None:
        """Unmaps the buffer, making it accessible to the GPU again."""
        pass

    def destroy(self) -> None:
        """Destroys the buffer."""
        pass

class BufferSlice:
    """
    A slice of a buffer.
    
    Used to specify a range of a buffer for operations like mapping or binding.
    """
    
    def __init__(self, buffer: Buffer, offset: int, size: int) -> None:
        self.buffer = buffer
        self.offset = offset
        self.size = size

    def map_async(self, mode: int, callback: Any = None) -> None:
        """Maps this slice asynchronously."""
        pass

    def get_mapped_range(self) -> memoryview:
        """Returns a view into the mapped range of this slice."""
        pass
