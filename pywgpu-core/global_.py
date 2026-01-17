from typing import Any
from .hub import Hub
from .registry import Registry
from .instance import Instance
from pywgpu_types.instance import InstanceDescriptor

class Global:
    """
    Entrance to the wgpu-core logic.
    """
    def __init__(self, name: str, descriptor: InstanceDescriptor) -> None:
        self.instance = Instance(name, descriptor)
        self.hub = Hub()
        self.surfaces = Registry()

    def generate_report(self) -> Any:
        pass

    # Method implementations mirroring crate::Global
    def instance_request_adapter(self, options: Any) -> Any:
        pass
    
    def adapter_request_device(self, adapter_id: Any, desc: Any, trace_path: Any) -> Any:
        pass
