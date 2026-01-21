from enum import IntFlag, Enum
from typing import Optional
from pydantic import BaseModel


class BufferUsage(IntFlag):
    MAP_READ = 1 << 0
    MAP_WRITE = 1 << 1
    COPY_SRC = 1 << 2
    COPY_DST = 1 << 3
    INDEX = 1 << 4
    VERTEX = 1 << 5
    UNIFORM = 1 << 6
    STORAGE = 1 << 7
    INDIRECT = 1 << 8
    QUERY_RESOLVE = 1 << 9
    BLAS_INPUT = 1 << 10
    TLAS_INPUT = 1 << 11


class BufferMapState(Enum):
    UNMAPPED = "unmapped"
    PENDING = "pending"
    MAPPED = "mapped"


class BufferDescriptor(BaseModel):
    label: Optional[str] = None
    size: int
    usage: int
    mapped_at_creation: bool = False


class BufferBindingType(Enum):
    UNIFORM = "uniform"
    STORAGE = "storage"
    READ_ONLY_STORAGE = "read-only-storage"
