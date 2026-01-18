"""
Main resource definitions and management.

This module defines the core resource types and traits used throughout
wgpu-core. It provides:
- Base resource traits (ResourceType, Labeled, ParentDevice, Trackable)
- Resource error types
- Resource lifecycle management
- Resource tracking and validation

All wgpu-core resources implement these base traits to provide a common
interface for resource management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from . import errors
from .device import Device


T = TypeVar("T")


@dataclass
class ResourceErrorIdent:
    """
    Information about a wgpu-core resource for error messages.
    
    This class provides a human-readable identifier for a resource,
    including its type and label, which is useful for error messages.
    
    Attributes:
        r#type: The type of the resource.
        label: The label of the resource.
    """

    r#type: str
    label: str = ""

    def __str__(self) -> str:
        """Return a string representation of the resource identifier."""
        if self.label:
            return f"{self.r#type} with '{self.label}' label"
        return self.r#type


class ResourceType:
    """
    Information about the wgpu-core resource type.
    
    This trait provides the type name for a resource, which is used
    in error messages and logging.
    
    Attributes:
        TYPE: The type name of the resource.
    """

    TYPE: str = ""

    @classmethod
    def type_name(cls) -> str:
        """Get the type name of the resource."""
        return cls.TYPE


class Labeled(ResourceType):
    """
    Resource with a label.

    This trait provides methods for getting the label of a resource,
    which is useful for debugging and error messages.

    Attributes:
        label: The label of the resource.
    """

    label: str = ""

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type=self.TYPE,
            label=self.label,
        )


class ParentDevice(Labeled):
    """
    Resource that has a parent device.

    This trait provides methods for accessing the parent device and
    checking if resources are from the same device.

    Attributes:
        device: The parent device.
    """

    device: Device

    def is_equal(self, other: Any) -> bool:
        """
        Check if this resource is equal to another resource.

        Args:
            other: The other resource.

        Returns:
            True if the resources are the same, False otherwise.
        """
        return self is other

    def same_device_as(self, other: Any) -> None:
        """
        Check if this resource is from the same device as another resource.

        Args:
            other: The other resource.

        Raises:
            DeviceError: If the resources are from different devices.
        """
        if not self.is_equal(other.device):
            raise errors.DeviceError(
                f"Resource {self.error_ident()} is from a different device"
            )

    def same_device(self, device: Device) -> None:
        """
        Check if this resource is from the given device.

        Args:
            device: The device to check.

        Raises:
            DeviceError: If the resource is from a different device.
        """
        if not self.is_equal(device):
            raise errors.DeviceError(
                f"Resource {self.error_ident()} is from a different device"
            )


class RawResourceAccess(ParentDevice):
    """
    Resource that provides access to raw HAL resources.
    
    This trait provides methods for accessing the raw HAL resource
    associated with a wgpu-core resource.
    
    Attributes:
        DynResource: The HAL resource type.
    """

    DynResource: Any

    def raw(self, guard: Any) -> Optional[Any]:
        """
        Get access to the raw resource if it is not destroyed.
        
        Args:
            guard: The snatch guard.
        
        Returns:
            The raw resource, or None if destroyed.
        """
        pass

    def try_raw(self, guard: Any) -> Any:
        """
        Get access to the raw resource, raising an error if destroyed.
        
        Args:
            guard: The snatch guard.
        
        Returns:
            The raw resource.
        
        Raises:
            DestroyedResourceError: If the resource has been destroyed.
        """
        raw = self.raw(guard)
        if raw is None:
            raise errors.DestroyedResourceError(self.error_ident())
        return raw


class Trackable:
    """
    Resource that can be tracked.
    
    This trait provides methods for tracking resource usage and
    managing resource lifetimes.
    
    Attributes:
        tracking_data: Data for resource tracking.
    """

    tracking_data: Any

    def tracker_index(self) -> Any:
        """Get the tracker index for this resource."""
        return self.tracking_data.tracker_index()


@dataclass
class DestroyedResourceError(Exception):
    """
    Error when accessing a destroyed resource.
    
    Attributes:
        resource: The resource identifier.
    """

    resource: ResourceErrorIdent

    def __str__(self) -> str:
        return f"{self.resource} has been destroyed"


@dataclass
class InvalidResourceError(Exception):
    """
    Error when accessing an invalid resource.
    
    Attributes:
        resource: The resource identifier.
    """

    resource: ResourceErrorIdent

    def __str__(self) -> str:
        return f"{self.resource} is invalid"


@dataclass
class MissingBufferUsageError(Exception):
    """
    Error when a buffer is missing required usage flags.
    
    Attributes:
        res: The resource identifier.
        actual: The actual usage flags.
        expected: The expected usage flags.
    """

    res: ResourceErrorIdent
    actual: int
    expected: int

    def __str__(self) -> str:
        return (
            f"Usage flags {self.actual:#x} of {self.res} do not contain "
            f"required usage flags {self.expected:#x}"
        )


@dataclass
class MissingTextureUsageError(Exception):
    """
    Error when a texture is missing required usage flags.
    
    Attributes:
        res: The resource identifier.
        actual: The actual usage flags.
        expected: The expected usage flags.
    """

    res: ResourceErrorIdent
    actual: int
    expected: int

    def __str__(self) -> str:
        return (
            f"Usage flags {self.actual:#x} of {self.res} do not contain "
            f"required usage flags {self.expected:#x}"
        )


@dataclass
class Fallible(Generic[T]):
    """
    A resource that can be either valid or invalid.
    
    This is used for resources that may fail creation, allowing the
    error to be stored and returned later.
    
    Attributes:
        valid: The valid resource, if any.
        invalid: The invalid label, if any.
    """

    valid: Optional[T] = None
    invalid: Optional[str] = None

    def get(self) -> T:
        """
        Get the resource, raising an error if invalid.
        
        Returns:
            The resource.
        
        Raises:
            InvalidResourceError: If the resource is invalid.
        """
        if self.valid is not None:
            return self.valid
        raise InvalidResourceError(
            ResourceErrorIdent(
                r#type=T.TYPE,
                label=self.invalid or ""
            )
        )


class Buffer:
    """
    A GPU buffer.
    
    A buffer is a linear allocation of GPU memory that can be used to
    store data for use in shaders and other GPU operations.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the buffer."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="Buffer",
            label=self.label
        )


class Texture:
    """
    A GPU texture.
    
    A texture is a multi-dimensional array of texels that can be used
    for rendering, sampling, and other GPU operations.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the texture."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="Texture",
            label=self.label
        )


class TextureView:
    """
    A view into a texture.
    
    A texture view provides a different interpretation of a texture's
    data, such as a different format or dimension.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the texture view."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="TextureView",
            label=self.label
        )


class Sampler:
    """
    A GPU sampler.
    
    A sampler defines how a texture is sampled in a shader, including
    filtering, addressing modes, and comparison functions.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the sampler."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="Sampler",
            label=self.label
        )


class ExternalTexture:
    """
    An external texture.
    
    External textures are textures that come from outside the WebGPU
    system, such as from a video element or camera feed.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the external texture."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="ExternalTexture",
            label=self.label
        )


class StagingBuffer:
    """
    A staging buffer.
    
    A staging buffer is a temporary buffer used for data transfers
    between the CPU and GPU.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the staging buffer."""
        self.device = device
        self.label = label

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="StagingBuffer",
            label=self.label
        )


class QuerySet:
    """
    A query set.
    
    A query set is used to collect GPU timing and occlusion query results.
    
    Attributes:
        device: The device that owns this resource.
        label: A human-readable label for debugging.
        tracking_data: Data for resource tracking.
    """

    def __init__(self, device: Device, label: str = "") -> None:
        """Initialize the query set."""
        self.device = device
        self.label = label
        self.tracking_data = None  # Would be TrackingData

    def error_ident(self) -> ResourceErrorIdent:
        """Get a resource error identifier."""
        return ResourceErrorIdent(
            r#type="QuerySet",
            label=self.label
        )
