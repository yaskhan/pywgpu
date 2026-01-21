from typing import Optional, Any
from pydantic import BaseModel
from enum import Enum
from .backend import Backend


class PowerPreference(Enum):
    """Power preference for choosing an adapter."""

    UNDEFINED = 0
    LOW_POWER = 1
    HIGH_PERFORMANCE = 2


class DeviceType(Enum):
    """Type of the device (adapter)."""

    OTHER = 0
    INTEGRATED_GPU = 1
    DISCRETE_GPU = 2
    VIRTUAL_GPU = 3
    CPU = 4


class AdapterInfo(BaseModel):
    """Information about an adapter."""

    name: str
    vendor: int
    device: int
    device_type: DeviceType
    driver: str
    driver_info: str
    backend: Backend
    subgroup_min_size: int = 0
    subgroup_max_size: int = 0


class RequestAdapterOptions(BaseModel):
    """Options for requestng an adapter."""

    power_preference: PowerPreference = PowerPreference.UNDEFINED
    force_fallback_adapter: bool = False
    compatible_surface: Optional[Any] = None
