from __future__ import annotations
from typing import Any
from .buffer import BufferTracker
from .texture import TextureTracker

class Tracker:
    """
    A collection of all resource trackers.
    
    This class orchestrates tracking for different types of resources
    (buffers, textures, etc.) within a command buffer or device.
    """
    def __init__(self) -> None:
        self.buffers = BufferTracker()
        self.textures = TextureTracker()

    def set_size(self, buffer_size: int, texture_size: int) -> None:
        """Sets the size of the internal trackers."""
        self.buffers.set_size(buffer_size)
        self.textures.set_size(texture_size)
