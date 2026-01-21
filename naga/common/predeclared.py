from typing import Any, Optional
from enum import Enum


class PredeclaredTypeKind(Enum):
    VEC = "vec"
    MAT = "mat"
    SAMPLER = "sampler"


class PredeclaredType:
    """
    Helper for handling predeclared types in IR.
    """

    def __init__(self, kind: PredeclaredTypeKind, size: int, format: Any):
        self.kind = kind
        self.size = size
        self.format = format

    def struct_name(self) -> str:
        return f"predeclared_{self.kind.value}{self.size}"
