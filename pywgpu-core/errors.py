class WGPUError(Exception):
    """Base exception for all pywgpu errors."""

    pass


class ValidationError(WGPUError):
    """Raised when validation fails."""

    pass


class DeviceError(WGPUError):
    """Raised when a device operation fails."""

    pass


class DestroyedResourceError(WGPUError):
    """Raised when accessing a destroyed resource."""

    def __init__(self, resource_ident):
        self.resource_ident = resource_ident
        super().__init__(str(self))

    def __str__(self):
        return f"{self.resource_ident} has been destroyed"


class BindingTypeMaxCountError(ValidationError):
    """Raised when binding count exceeds limits."""

    def __init__(self, kind, zone, limit, count):
        self.kind = kind
        self.zone = zone
        self.limit = limit
        self.count = count
        super().__init__(str(self))

    def __str__(self):
        return f"Too many {self.kind} in {self.zone}: {self.count} > {self.limit}"


class CreateBindGroupLayoutError(ValidationError):
    """Raised when bind group layout creation fails."""

    pass
