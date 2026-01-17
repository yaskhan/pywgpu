from typing import Any, List
from pywgpu_types.device import DeviceDescriptor
from pywgpu_types.instance import InstanceDescriptor

class Instance:
    """
    Core Instance logic.
    
    Manages HAL instances and backends.
    """
    def __init__(self, name: str, descriptor: InstanceDescriptor) -> None:
        self.name = name
        self.descriptor = descriptor
        self.backends = {} # Map backend type to HAL instance

    def request_adapter(
        self, 
        desc: Any, 
        inputs: Any
    ) -> Any:
        """Finds a suitable adapter."""
        pass
