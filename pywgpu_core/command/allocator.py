"""
Command allocator for wgpu-core.

This module implements the command allocator, which manages a pool of command
encoders. The allocator is responsible for:
- Acquiring command encoders from the pool
- Releasing command encoders back to the pool

The command allocator is owned by a device and provides a pool of command
encoders that can be reused to avoid the overhead of creating new encoders.
"""

from __future__ import annotations

from typing import Any, List

from . import lock


class CommandAllocator:
    """
    A pool of free command encoders, owned by a device.

    Each encoder in this list is in the "closed" state.

    Since a raw command encoder is itself a pool for allocating command
    buffers, this is a pool of pools.

    Attributes:
        free_encoders: Mutex-protected list of free encoders.
    """

    def __init__(self) -> None:
        """Initialize the command allocator."""
        self.free_encoders = lock.Mutex(lock.rank.COMMAND_ALLOCATOR_FREE_ENCODERS, [])

    def acquire_encoder(
        self,
        device: Any,
        queue: Any,
    ) -> Any:
        """
        Return a fresh command encoder in the "closed" state.

        If we have free encoders in the pool, take one of those. Otherwise,
        create a new one on the device.

        Args:
            device: The device to create encoders on.
            queue: The queue for the encoder.

        Returns:
            A command encoder.

        Raises:
            DeviceError: If encoder creation fails.
        """
        with self.free_encoders.lock() as free_encoders:
            if free_encoders:
                return free_encoders.pop()
            else:
                # Create a new encoder using the imported CommandEncoder
                from .encoder import CommandEncoder

                return CommandEncoder(device)

    def release_encoder(self, encoder: Any) -> None:
        """
        Add an encoder back to the free pool.

        Args:
            encoder: The encoder to release.
        """
        with self.free_encoders.lock() as free_encoders:
            free_encoders.append(encoder)
