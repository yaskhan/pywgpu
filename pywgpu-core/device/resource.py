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

    A Device represents a logical connection to a GPU, providing methods to
    create resources like buffers, textures, and pipelines.

    Attributes:
        label: Human-readable label for debugging.
        valid: Whether the device is valid (not lost).
        features: Enabled features for this device.
        limits: Limits for this device.
    """
    def __init__(self) -> None:
        self.features = 0
        self.limits = {}
        self.adapter = None
        self.downlevel = None
        self.valid = True

    def create_buffer(self, desc: BufferDescriptor) -> Any:
        """Create a buffer on the device.
        
        Args:
            desc: Buffer descriptor.
            
        Returns:
            The created buffer object.
            
        Raises:
            RuntimeError: If buffer creation fails.
        """
        # Import HAL
        try:
            import sys
            import os
            _hal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pywgpu-hal')
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            raise RuntimeError("pywgpu_hal module not available for buffer creation")
        
        # Check device is valid
        self.check_is_valid()
        
        # Get HAL device
        hal_device = getattr(self, 'hal_device', None) or getattr(self, '_hal_device', None)
        if hal_device is None:
            raise RuntimeError("Device does not have HAL device")
        
        # Create HAL buffer descriptor
        hal_desc = hal.BufferDescriptor(
            label=desc.label,
            size=desc.size,
            usage=desc.usage,
            memory_flags=hal.MemoryFlags.NONE
        )
        
        try:
            # Create buffer using HAL
            buffer = hal_device.create_buffer(hal_desc)
            return buffer
        except Exception as e:
            raise RuntimeError(f"Failed to create buffer: {e}") from e

    def create_texture(self, desc: TextureDescriptor) -> Any:
        """Create a texture on the device.
        
        Args:
            desc: Texture descriptor.
            
        Returns:
            The created texture object.
            
        Raises:
            RuntimeError: If texture creation fails.
        """
        # Import HAL
        try:
            import sys
            import os
            _hal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pywgpu-hal')
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            raise RuntimeError("pywgpu_hal module not available for texture creation")
        
        # Check device is valid
        self.check_is_valid()
        
        # Get HAL device
        hal_device = getattr(self, 'hal_device', None) or getattr(self, '_hal_device', None)
        if hal_device is None:
            raise RuntimeError("Device does not have HAL device")
        
        # Create HAL texture descriptor
        # This would need proper conversion from TextureDescriptor to hal.TextureDescriptor
        hal_desc = hal.TextureDescriptor(
            label=desc.label,
            size=desc.size,
            mip_level_count=desc.mip_level_count,
            sample_count=desc.sample_count,
            dimension=desc.dimension,
            format=desc.format,
            usage=desc.usage,
            memory_flags=hal.MemoryFlags.NONE,
            view_formats=[]
        )
        
        try:
            # Create texture using HAL
            texture = hal_device.create_texture(hal_desc)
            return texture
        except Exception as e:
            raise RuntimeError(f"Failed to create texture: {e}") from e

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
