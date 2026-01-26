"""
Error types for wgpu-core.

This module provides error types used throughout wgpu-core, including
context errors and multi-error containers.
"""

from __future__ import annotations

from typing import Iterator, List, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class WGPUError(Exception):
    """Base exception for all pywgpu errors."""

    pass


class ValidationError(WGPUError):
    """Raised when validation fails."""

    pass


class DeviceError(WGPUError):
    """Raised when a device operation fails."""

    pass


class ContextError(WGPUError):
    """
    Error with context information.

    This error wraps another error with information about the function
    where it occurred and an optional label.

    Attributes:
        fn_ident: The name of the function where the error occurred.
        source: The underlying error that caused this error.
        label: An optional label for the resource or operation.
    """

    def __init__(self, fn_ident: str, source: Exception, label: str = "") -> None:
        """
        Create a new context error.

        Args:
            fn_ident: The name of the function where the error occurred.
            source: The underlying error.
            label: An optional label for the resource or operation.
        """
        self.fn_ident = fn_ident
        self.source = source
        self.label = label
        super().__init__(str(self))

    def __str__(self) -> str:
        """Get a string representation of the error."""
        if self.label:
            return f"In {self.fn_ident}, label = '{self.label}': {self.source}"
        return f"In {self.fn_ident}: {self.source}"


class MultiError(WGPUError):
    """
    Container for multiple errors.

    This error type stores multiple errors and presents the first one
    as the primary error. It's used when multiple validation errors
    occur and need to be reported together.

    Attributes:
        inner: List of errors.
    """

    def __init__(self, errors: List[Exception]) -> None:
        """
        Create a new multi-error.

        Args:
            errors: A non-empty list of errors.

        Raises:
            ValueError: If the errors list is empty.
        """
        if not errors:
            raise ValueError("MultiError requires at least one error")
        self.inner: List[Exception] = errors
        super().__init__(str(self))

    @staticmethod
    def from_iterable(errors: List[Exception]) -> MultiError | None:
        """
        Create a MultiError from an iterable of errors.

        Args:
            errors: A list of errors.

        Returns:
            A MultiError if there are errors, None otherwise.
        """
        if not errors:
            return None
        return MultiError(errors)

    def errors(self) -> Iterator[Exception]:
        """
        Get an iterator over all errors.

        Returns:
            An iterator over all errors.
        """
        return iter(self.inner)

    def __str__(self) -> str:
        """Get a string representation using the first error."""
        return str(self.inner[0])

    def __repr__(self) -> str:
        """Get a debug representation."""
        return f"MultiError({self.inner!r})"


class DestroyedResourceError(WGPUError):
    """Raised when accessing a destroyed resource."""

    def __init__(self, resource_ident: str) -> None:
        """
        Create a new destroyed resource error.

        Args:
            resource_ident: The identifier of the destroyed resource.
        """
        self.resource_ident = resource_ident
        super().__init__(str(self))

    def __str__(self) -> str:
        """Get a string representation."""
        return f"{self.resource_ident} has been destroyed"


class BindingTypeMaxCountError(ValidationError):
    """Raised when binding count exceeds limits."""

    def __init__(self, kind: str, zone: str, limit: int, count: int) -> None:
        """
        Create a new binding type max count error.

        Args:
            kind: The type of binding (e.g., "storage buffers").
            zone: The zone where the binding is used.
            limit: The maximum allowed count.
            count: The actual count.
        """
        self.kind = kind
        self.zone = zone
        self.limit = limit
        self.count = count
        super().__init__(str(self))

    def __str__(self) -> str:
        """Get a string representation."""
        return f"Too many {self.kind} in {self.zone}: {self.count} > {self.limit}"


class CreateBindGroupLayoutError(ValidationError):
    """Raised when bind group layout creation fails."""

    pass


__all__ = [
    "WGPUError",
    "ValidationError",
    "DeviceError",
    "ContextError",
    "MultiError",
    "DestroyedResourceError",
    "BindingTypeMaxCountError",
    "CreateBindGroupLayoutError",
]
