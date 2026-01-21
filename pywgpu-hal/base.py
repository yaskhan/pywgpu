from __future__ import annotations
from typing import Protocol, List, Optional
from pywgpu_types import DeviceDescriptor


class Api(Protocol):
    """Protocol for a graphics API backend (Vulkan, Metal, etc.)."""

    def enumerate_adapters(self) -> List[Adapter]: ...


class Adapter(Protocol):
    """Protocol for a hardware adapter."""

    def request_device(self, descriptor: DeviceDescriptor) -> Device: ...


class Device(Protocol):
    """Protocol for a logical device."""

    def create_buffer(self, size: int, usage: int) -> Buffer: ...


class Buffer(Protocol):
    """Protocol for a GPU-side buffer."""

    def destroy(self) -> None: ...
