class WGPUError(Exception):
    """Base exception for all pywgpu errors."""
    pass

class ValidationError(WGPUError):
    """Raised when validation fails."""
    pass

class DeviceError(WGPUError):
    """Raised when a device operation fails."""
    pass
