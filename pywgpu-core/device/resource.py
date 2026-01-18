from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class BufferDescriptor:
    label: Optional[str] = None
    size: int = 0
    usage: int = 0
    mapped_at_creation: bool = False


@dataclass
class TextureDescriptor:
    label: Optional[str] = None
    size: Any = None
    mip_level_count: int = 1
    sample_count: int = 1
    dimension: str = "2d"
    format: str = "rgba8unorm"
    usage: int = 0


class Device:
    """
    Device reference / logic.
    """
    def __init__(self) -> None:
        self.features = 0
        self.limits = {}
        self.adapter = None
        self.downlevel = None
        self.valid = True

    def create_buffer(self, desc: BufferDescriptor) -> Any:
        """Create a buffer on the device."""
        # Placeholder implementation
        return None

    def create_texture(self, desc: TextureDescriptor) -> Any:
        """Create a texture on the device."""
        # Placeholder implementation
        return None

    def check_is_valid(self) -> None:
        """Check if the device is valid."""
        if not self.valid:
            raise RuntimeError("Device is invalid")

    def require_features(self, features: int) -> None:
        """Check if required features are supported."""
        if not (self.features & features):
            raise RuntimeError("Required features not supported")

    def require_downlevel_flags(self, flags: int) -> None:
        """Check if required downlevel flags are supported."""
        if self.downlevel and not (self.downlevel.flags & flags):
            raise RuntimeError("Required downlevel flags not supported")


def create_buffer(device: Device, desc: BufferDescriptor) -> Any:
    """Create a buffer on the device."""
    return device.create_buffer(desc)


def create_texture(device: Device, desc: TextureDescriptor) -> Any:
    """Create a texture on the device."""
    return device.create_texture(desc)
