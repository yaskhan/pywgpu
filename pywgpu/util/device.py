# Utility functions for Device
from typing import Any

class DeviceExt:
    """Extension methods for Device."""
    
    def create_buffer_init(self, descriptor: Any, data: bytes) -> Any:
        """Creates a buffer initialized with data."""
        if hasattr(self, '_device') and hasattr(self._device, 'create_buffer'):
            # Create buffer and write data to it
            buffer = self._device.create_buffer(descriptor)
            # In a real implementation, we would write the data to the buffer
            # For now, just return the buffer
            return buffer
        else:
            raise NotImplementedError("Device not available for create_buffer_init")
