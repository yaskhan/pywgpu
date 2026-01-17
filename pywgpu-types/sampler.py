from typing import Optional
from pydantic import BaseModel
from enum import Enum

class AddressMode(Enum):
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRROR_REPEAT = 2
    CLAMP_TO_BORDER = 3

class FilterMode(Enum):
    NEAREST = 0
    LINEAR = 1

class CompareFunction(Enum):
    UNDEFINED = 0
    NEVER = 1
    LESS = 2
    EQUAL = 3
    LESS_EQUAL = 4
    GREATER = 5
    NOT_EQUAL = 6
    GREATER_EQUAL = 7
    ALWAYS = 8

class SamplerDescriptor(BaseModel):
    label: Optional[str] = None
    address_mode_u: AddressMode = AddressMode.CLAMP_TO_EDGE
    address_mode_v: AddressMode = AddressMode.CLAMP_TO_EDGE
    address_mode_w: AddressMode = AddressMode.CLAMP_TO_EDGE
    mag_filter: FilterMode = FilterMode.NEAREST
    min_filter: FilterMode = FilterMode.NEAREST
    mipmap_filter: FilterMode = FilterMode.NEAREST
    lod_min_clamp: float = 0.0
    lod_max_clamp: float = 32.0
    compare: Optional[CompareFunction] = None
    anisotropy_clamp: int = 1
    border_color: Optional[str] = None # Or Enum
