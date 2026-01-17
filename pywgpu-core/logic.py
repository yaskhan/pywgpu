from __future__ import annotations
from typing import Optional, List
from pywgpu_types import DeviceDescriptor, BufferDescriptor
from pywgpu_hal import base as hal

class Instance:
    def __init__(self) -> None:
        self._adapters: List[hal.Adapter] = []

    def request_adapter(self) -> Optional[hal.Adapter]:
        """Core logic for finding a suitable adapter."""
        return None

class Device:
    def __init__(self, hal_device: hal.Device) -> None:
        self._hal_device = hal_device

    def create_buffer(self, descriptor: BufferDescriptor) -> Buffer:
        hal_buffer = self._hal_device.create_buffer(descriptor.size, descriptor.usage)
        return Buffer(hal_buffer)

class Buffer:
    def __init__(self, hal_buffer: hal.Buffer) -> None:
        self._hal_buffer = hal_buffer

    def destroy(self) -> None:
        self._hal_buffer.destroy()
