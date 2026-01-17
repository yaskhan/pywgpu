from enum import Enum
from typing import Any, Optional

class TypeInner(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    ATOMIC = "atomic"
    POINTER = "pointer"
    VALUE_POINTER = "value-pointer"
    ARRAY = "array"
    STRUCT = "struct"
    IMAGE = "image"
    SAMPLER = "sampler"
    ACCELERATION_STRUCTURE = "acceleration-structure"
    RAY_QUERY = "ray-query"
    BINDING_ARRAY = "binding-array"

class Type:
    """
    IR Type definition.
    """
    def __init__(self, name: Optional[str], inner: TypeInner) -> None:
        self.name = name
        self.inner = inner
