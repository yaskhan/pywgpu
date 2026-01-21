from typing import List, Optional, Any
from pydantic import BaseModel
from enum import Enum


class PresentMode(Enum):
    AUTO_VSCYNC = 0
    AUTO_NO_VSYNC = 1
    FIFO = 2
    FIFO_RELAXED = 3
    IMMEDIATE = 4
    MAILBOX = 5


class CompositeAlphaMode(Enum):
    AUTO = 0
    OPAQUE = 1
    PRE_MULTIPLIED = 2
    POST_MULTIPLIED = 3
    INHERIT = 4


class SurfaceConfiguration(BaseModel):
    usage: int
    format: str
    width: int
    height: int
    present_mode: PresentMode = PresentMode.FIFO
    alpha_mode: CompositeAlphaMode = CompositeAlphaMode.AUTO
    view_formats: List[str] = []


class SurfaceCapabilities(BaseModel):
    formats: List[str]
    present_modes: List[PresentMode]
    alpha_modes: List[CompositeAlphaMode]


class SurfaceStatus(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    OUTDATED = 2
    LOST = 3
