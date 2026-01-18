from enum import Enum
from typing import Any, Optional, Union, List

class ScalarKind(Enum):
    SINT = "sint"
    UINT = "uint"
    FLOAT = "float"
    BOOL = "bool"

class Scalar:
    """A scalar type."""
    def __init__(self, kind: ScalarKind, width: int) -> None:
        self.kind = kind
        self.width = width

class VectorSize(Enum):
    BI = 2
    TRI = 3
    QUAD = 4

class Vector:
    """A vector type."""
    def __init__(self, size: VectorSize, kind: ScalarKind, width: int) -> None:
        self.size = size
        self.kind = kind
        self.width = width

class Matrix:
    """A matrix type."""
    def __init__(self, columns: VectorSize, rows: VectorSize, width: int) -> None:
        self.columns = columns
        self.rows = rows
        self.width = width

class ArraySize:
    """Array size specification."""
    def __init__(self, size: Union[int, str], is_dynamic: bool = False) -> None:
        self.size = size
        self.is_dynamic = is_dynamic

class Array:
    """An array type."""
    def __init__(self, base_type: int, size: Optional[int] = None) -> None:
        self.base = base_type
        self.size = size
        self.stride = None

class StructMember:
    """A struct member."""
    def __init__(self, name: Optional[str], ty: int, offset: int, binding: Optional[Any] = None) -> None:
        self.name = name
        self.ty = ty
        self.offset = offset
        self.binding = binding

class Struct:
    """A struct type."""
    def __init__(self, members: List[StructMember], span: int = 0) -> None:
        self.members = members
        self.span = span

class ImageDimension(Enum):
    D1 = "1d"
    D2 = "2d"
    D3 = "3d"
    CUBE = "cube"

class Image:
    """An image type."""
    def __init__(self, dim: ImageDimension, arrayed: bool, class_: str) -> None:
        self.dim = dim
        self.arrayed = arrayed
        self.class = class_
        self.format = None

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

    def scalar(kind: ScalarKind, width: int) -> 'Type':
        """Create a scalar type."""
        typ = Type(None, TypeInner.SCALAR)
        typ._scalar = Scalar(kind, width)
        return typ

    def vector(size: VectorSize, kind: ScalarKind, width: int) -> 'Type':
        """Create a vector type."""
        typ = Type(None, TypeInner.VECTOR)
        typ._vector = Vector(size, kind, width)
        return typ

    def matrix(columns: VectorSize, rows: VectorSize, width: int) -> 'Type':
        """Create a matrix type."""
        typ = Type(None, TypeInner.MATRIX)
        typ._matrix = Matrix(columns, rows, width)
        return typ

    def array(base_type: int, size: Optional[int] = None) -> 'Type':
        """Create an array type."""
        typ = Type(None, TypeInner.ARRAY)
        typ._array = Array(base_type, size)
        return typ

    def struct(name: Optional[str], members: List[StructMember], span: int = 0) -> 'Type':
        """Create a struct type."""
        typ = Type(name, TypeInner.STRUCT)
        typ._struct = Struct(members, span)
        return typ

    def image(dim: ImageDimension, arrayed: bool, class_: str) -> 'Type':
        """Create an image type."""
        typ = Type(None, TypeInner.IMAGE)
        typ._image = Image(dim, arrayed, class_)
        return typ
