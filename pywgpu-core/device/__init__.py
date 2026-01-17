from .queue import Queue
from .life import LifetimeTracker
from .resource import Device
from .ops import DeviceOps
from .bgl import EntryMap
from .ray_tracing import RayTracing

class DeviceLogic(Device):
    """
    Main Device logic container.
    """
    def __init__(self, adapter: Any, desc: Any) -> None:
        self.queue = Queue(self)
        self.life = LifetimeTracker()
