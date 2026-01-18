from typing import Any, Optional
from dataclasses import dataclass
from ..resource import Buffer, Texture
from pywgpu_types.texture import TextureDescriptor
from pywgpu_types.buffer import BufferDescriptor


class Device:
    """
    Device reference / logic.

    A Device represents a logical connection to a GPU, providing methods to
    create resources like buffers, textures, and pipelines.

    Attributes:
        label: Human-readable label for debugging.
        valid: Whether the device is valid (not lost).
        features: Enabled features for this device.
        limits: Limits for this device.
    """

    def __init__(self, label: str = "") -> None:
        """Initialize the device."""
        self.label = label
        self.valid = True
        self.features = 0  # Features bitmask
        self.limits = None  # Limits object

    def create_buffer(self, desc: BufferDescriptor) -> Buffer:
        """
        Create a new buffer.

        Args:
            desc: Buffer descriptor.

        Returns:
            The created buffer.

        Raises:
            RuntimeError: If the device is invalid.
        """
        if not self.valid:
            raise RuntimeError("Device is invalid")

        buffer = Buffer(device=self, label=desc.label or "")
        return buffer

    def create_texture(self, desc: TextureDescriptor) -> Texture:
        """
        Create a new texture.

        Args:
            desc: Texture descriptor.

        Returns:
            The created texture.

        Raises:
            RuntimeError: If the device is invalid.
        """
        if not self.valid:
            raise RuntimeError("Device is invalid")

        texture = Texture(device=self, label=desc.label or "")
        return texture

    def lose(self) -> None:
        """Mark the device as lost."""
        self.valid = False
