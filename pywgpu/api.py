from __future__ import annotations
from typing import Optional
from pywgpu_core import logic as core
from pywgpu_types import DeviceDescriptor

class Instance:
    def __init__(self) -> None:
        self._core_instance = core.Instance()

    def request_adapter(self) -> Optional[Adapter]:
        core_adapter = self._core_instance.request_adapter()
        if core_adapter:
            return Adapter(core_adapter)
        return None

class Adapter:
    def __init__(self, hal_adapter: Any) -> None:
        self._hal_adapter = hal_adapter

    def request_device(self, descriptor: Optional[DeviceDescriptor] = None) -> Device:
        # Mocking device creation
        return Device()

class Device:
    def __init__(self) -> None:
        pass

def request_adapter() -> Optional[Adapter]:
    return Instance().request_adapter()
