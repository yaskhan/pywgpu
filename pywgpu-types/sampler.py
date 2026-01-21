from typing import Optional
from pydantic import BaseModel
from enum import Enum


class AddressMode(Enum):
    CLAMP_TO_EDGE = "clamp-to-edge"
    REPEAT = "repeat"
    MIRROR_REPEAT = "mirror-repeat"
    CLAMP_TO_BORDER = "clamp-to-border"


class FilterMode(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"


class CompareFunction(Enum):
    NEVER = "never"
    LESS = "less"
    EQUAL = "equal"
    LESS_EQUAL = "less-equal"
    GREATER = "greater"
    NOT_EQUAL = "not-equal"
    GREATER_EQUAL = "greater-equal"
    ALWAYS = "always"


class SamplerBorderColor(Enum):
    TRANSPARENT_BLACK = "transparent-black"
    OPAQUE_BLACK = "opaque-black"
    OPAQUE_WHITE = "opaque-white"
    ZERO = "zero"


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
    border_color: Optional[SamplerBorderColor] = None


class Sampler(BaseModel):
    pass
