# Utility functions for Device
from typing import Any, Optional
from pywgpu_types.descriptors import BufferDescriptor

class DeviceExt:
    """Extension methods for Device."""
    
    def create_buffer_init(self, label: Optional[str], contents: bytes, usage: int) -> Any:
        """Creates a buffer initialized with data."""
        # contents is bytes/memoryview
        size = len(contents)
        
        # We need to access create_buffer which is in Device (the class using this mixin)
        # We'll use self.create_buffer
        descriptor = BufferDescriptor(
            label=label,
            size=size,
            usage=usage,
            mapped_at_creation=True
        )
        
        buffer = self.create_buffer(descriptor)
        
        # Get mapped range and copy data
        mapped_range = buffer.get_mapped_range()
        mapped_range[:size] = contents
        buffer.unmap()
        
        return buffer
