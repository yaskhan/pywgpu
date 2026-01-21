"""
Scratch buffer management.

This module implements scratch buffer management for wgpu-core. Scratch
buffers are temporary buffers used for acceleration structure building
and other GPU operations that require temporary storage.

Scratch buffers are allocated on demand and are automatically freed when
no longer needed.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .device import Device

# Import HAL types
try:
    # Try importing from pywgpu-hal directory
    import sys
    import os

    _hal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pywgpu-hal")
    if _hal_path not in sys.path:
        sys.path.insert(0, _hal_path)
    import lib as hal
except ImportError:
    # Fallback if HAL not available
    hal = None  # type: ignore


class ScratchBuffer:
    """
    A scratch buffer.

    A scratch buffer is a temporary buffer used for GPU operations that
    require temporary storage, such as acceleration structure building.

    The buffer is created with ACCELERATION_STRUCTURE_SCRATCH usage and
    is automatically destroyed when the ScratchBuffer object is garbage
    collected.

    Attributes:
        device: The device that owns this buffer.
        size: The size of the buffer in bytes.
        _raw: The raw HAL buffer (private).
    """

    def __init__(self, device: "Device", size: int) -> None:
        """
        Create a new scratch buffer.

        Args:
            device: The device to create the buffer on.
            size: The size of the buffer in bytes.

        Raises:
            DeviceError: If buffer creation fails.
            RuntimeError: If HAL is not available.
        """
        if hal is None:
            raise RuntimeError("pywgpu_hal module not available")

        self.device = device
        self.size = size
        self._raw: Any = None

        # Create the HAL buffer
        try:
            # Get the raw HAL device
            hal_device = self._get_hal_device()

            # Create buffer descriptor for scratch buffer
            desc = hal.BufferDescriptor(
                label="(wgpu) scratch buffer",
                size=size,
                usage=self._get_scratch_usage(),
                memory_flags=hal.MemoryFlags.NONE,
            )

            # Create the buffer using HAL
            self._raw = hal_device.create_buffer(desc)

        except Exception as e:
            # Convert HAL errors to appropriate exceptions
            if hal and isinstance(e, hal.DeviceError):
                raise e
            raise RuntimeError(f"Failed to create scratch buffer: {e}") from e

    def _get_hal_device(self) -> Any:
        """Get the raw HAL device from the device object.

        Returns:
            The raw HAL device.

        Raises:
            RuntimeError: If the device doesn't have a raw HAL device.
        """
        # Try to get the raw device - implementation depends on Device structure
        if hasattr(self.device, "raw"):
            return self.device.raw()
        elif hasattr(self.device, "_raw"):
            return self.device._raw
        elif hasattr(self.device, "hal_device"):
            return self.device.hal_device
        else:
            # Fallback: assume device itself implements HAL Device protocol
            return self.device

    def _get_scratch_usage(self) -> Any:
        """Get the buffer usage flags for scratch buffers.

        Returns:
            Buffer usage flags for acceleration structure scratch.
        """
        # Try to get from wgpu_types if available
        try:
            import pywgpu_types as wgt

            if hasattr(wgt, "BufferUses"):
                return wgt.BufferUses.ACCELERATION_STRUCTURE_SCRATCH
        except ImportError:
            pass

        # Fallback: use a placeholder value
        # In a real implementation, this would be the proper enum value
        return 0x1000  # Placeholder for ACCELERATION_STRUCTURE_SCRATCH

    def raw(self) -> Any:
        """Get the raw HAL buffer.

        Returns:
            The raw HAL buffer object.
        """
        return self._raw

    def __del__(self) -> None:
        """Destroy the scratch buffer when garbage collected.

        This ensures the HAL buffer is properly destroyed and resources
        are freed when the ScratchBuffer is no longer needed.
        """
        # Check if _raw exists (in case __init__ failed)
        if not hasattr(self, "_raw") or self._raw is None:
            return

        try:
            # Get the HAL device
            hal_device = self._get_hal_device()

            # Destroy the buffer
            hal_device.destroy_buffer(self._raw)

            # Log the destruction (optional, for debugging)
            # print("Destroyed raw ScratchBuffer")

        except Exception as e:
            # Suppress exceptions during cleanup to avoid issues in __del__
            # In production, this might log to a proper logging system
            import sys

            print(f"Warning: Error destroying scratch buffer: {e}", file=sys.stderr)
        finally:
            self._raw = None
