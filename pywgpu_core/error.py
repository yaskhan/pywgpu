"""
Core error types for wgpu-core.

This module defines the base error types used throughout wgpu-core.
All errors in wgpu-core inherit from WGPUError, which provides a common
interface for error handling and reporting.

The error types are designed to be compatible with WebGPU's error reporting
mechanism, allowing errors to be categorized and handled appropriately.
"""

from __future__ import annotations

from typing import Optional


class WGPUError(Exception):
    """
    Base exception for all pywgpu errors.
    
    All errors in wgpu-core inherit from this class. It provides a common
    interface for error handling and reporting.
    
    Attributes:
        message: The error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the error with a message."""
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class ContextError(WGPUError):
    """
    Error related to context creation or management.
    
    This error is raised when there's an issue with creating or managing
    a context, such as a device context or instance context.
    
    Attributes:
        fn_ident: The function identifier where the error occurred.
        source: The underlying error source.
        label: A label for the context.
    """

    def __init__(
        self,
        fn_ident: str,
        source: Exception,
        label: str = "",
    ) -> None:
        """Initialize the context error."""
        self.fn_ident = fn_ident
        self.source = source
        self.label = label
        message = f"In {fn_ident}"
        if label:
            message += f", label = '{label}'"
        super().__init__(message)


class MultiError(WGPUError):
    """
    Error containing multiple errors.
    
    This error is used when multiple errors occur during an operation,
    allowing all errors to be reported together.
    
    Attributes:
        errors: List of errors.
    """

    def __init__(self, errors: list[Exception]) -> None:
        """Initialize the multi error with a list of errors."""
        self.errors = errors
        if errors:
            super().__init__(str(errors[0]))
        else:
            super().__init__("Multiple errors occurred")

    def __str__(self) -> str:
        """Return a string representation of all errors."""
        if len(self.errors) == 1:
            return str(self.errors[0])
        return f"Multiple errors: {', '.join(str(e) for e in self.errors)}"


class ValidationError(WGPUError):
    """
    Error raised when validation fails.
    
    This error is raised when a WebGPU operation fails validation,
    such as when an invalid parameter is passed to a function.
    
    Attributes:
        message: The validation error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the validation error."""
        super().__init__(message)


class DeviceError(WGPUError):
    """
    Error raised when a device operation fails.
    
    This error is raised when an operation on a device fails,
    such as when creating a resource or submitting commands.
    
    Attributes:
        message: The device error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the device error."""
        super().__init__(message)


class BufferAccessError(WGPUError):
    """
    Error related to buffer access.
    
    This error is raised when a buffer access operation fails,
    such as when mapping a buffer or copying data to/from a buffer.
    
    Attributes:
        message: The buffer access error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the buffer access error."""
        super().__init__(message)


class SurfaceError(WGPUError):
    """
    Error related to surface operations.
    
    This error is raised when a surface operation fails,
    such as when acquiring or presenting a surface texture.
    
    Attributes:
        message: The surface error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the surface error."""
        super().__init__(message)


class CreateSurfaceError(WGPUError):
    """
    Error related to surface creation.
    
    This error is raised when creating a surface fails.
    
    Attributes:
        message: The surface creation error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the surface creation error."""
        super().__init__(message)


class ConfigureSurfaceError(WGPUError):
    """
    Error related to surface configuration.
    
    This error is raised when configuring a surface fails.
    
    Attributes:
        message: The surface configuration error message.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the surface configuration error."""
        super().__init__(message)


class FailedLimit(WGPUError):
    """
    Error when a requested limit exceeds the allowed limit.
    
    This error is raised when a requested limit is better than the allowed limit.
    
    Attributes:
        name: The name of the limit.
        requested: The requested limit value.
        allowed: The allowed limit value.
    """

    def __init__(self, name: str, requested: int, allowed: int) -> None:
        """Initialize the failed limit error."""
        self.name = name
        self.requested = requested
        self.allowed = allowed
        super().__init__(
            f"Limit '{name}' value {requested} is better than allowed {allowed}"
        )


class DestroyedResourceError(WGPUError):
    """
    Error when accessing a destroyed resource.
    
    This error is raised when trying to access a resource that has been destroyed.
    
    Attributes:
        resource: The resource identifier.
    """

    def __init__(self, resource: str) -> None:
        """Initialize the destroyed resource error."""
        self.resource = resource
        super().__init__(f"{resource} has been destroyed")


class InvalidResourceError(WGPUError):
    """
    Error when accessing an invalid resource.
    
    This error is raised when trying to access a resource that is invalid
    (e.g., an ID that was never created or has been freed).
    
    Attributes:
        resource: The resource identifier.
    """

    def __init__(self, resource: str) -> None:
        """Initialize the invalid resource error."""
        self.resource = resource
        super().__init__(f"{resource} is invalid")


class MissingBufferUsageError(WGPUError):
    """
    Error when a buffer is missing required usage flags.
    
    This error is raised when a buffer operation requires usage flags
    that are not present on the buffer.
    
    Attributes:
        resource: The resource identifier.
        actual: The actual usage flags.
        expected: The expected usage flags.
    """

    def __init__(self, resource: str, actual: int, expected: int) -> None:
        """Initialize the missing buffer usage error."""
        self.resource = resource
        self.actual = actual
        self.expected = expected
        super().__init__(
            f"Usage flags {actual:#x} of {resource} do not contain required usage flags {expected:#x}"
        )


class MissingTextureUsageError(WGPUError):
    """
    Error when a texture is missing required usage flags.
    
    This error is raised when a texture operation requires usage flags
    that are not present on the texture.
    
    Attributes:
        resource: The resource identifier.
        actual: The actual usage flags.
        expected: The expected usage flags.
    """

    def __init__(self, resource: str, actual: int, expected: int) -> None:
        """Initialize the missing texture usage error."""
        self.resource = resource
        self.actual = actual
        self.expected = expected
        super().__init__(
            f"Usage flags {actual:#x} of {resource} do not contain required usage flags {expected:#x}"
        )


class ResourceErrorIdent:
    """
    Information about a wgpu-core resource for error messages.
    
    This class provides a human-readable identifier for a resource,
    including its type and label, which is useful for error messages.
    
    Attributes:
        type: The type of the resource.
        label: The label of the resource.
    """

    def __init__(self, type: str, label: str = "") -> None:
        """Initialize the resource error identifier."""
        self.type = type
        self.label = label

    def __str__(self) -> str:
        """Return a string representation of the resource identifier."""
        if self.label:
            return f"{self.type} with '{self.label}' label"
        return self.type
