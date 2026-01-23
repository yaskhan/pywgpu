from enum import Enum, Flag, IntFlag
from typing import Any, Optional, Union, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from . import Bytes


class ScalarKind(Enum):
    """Primitive type for a scalar."""
    SINT = "sint"
    UINT = "uint"
    FLOAT = "float"
    BOOL = "bool"
    ABSTRACT_INT = "abstract-int"  # WGSL abstract integer type
    ABSTRACT_FLOAT = "abstract-float"  # WGSL abstract floating-point type


@dataclass(frozen=True, slots=True)
class Scalar:
    """Characteristics of a scalar type."""
    kind: ScalarKind
    width: int  # Size of the value in bytes


class VectorSize(Enum):
    """Number of components in a vector."""
    BI = 2
    TRI = 3
    QUAD = 4


@dataclass(frozen=True, slots=True)
class Vector:
    """A vector of numbers."""
    size: VectorSize
    scalar: Scalar


@dataclass(frozen=True, slots=True)
class Matrix:
    """A matrix of numbers."""
    columns: VectorSize
    rows: VectorSize
    scalar: Scalar


class CooperativeSize(Enum):
    """Number of components in a cooperative vector."""
    EIGHT = 8
    SIXTEEN = 16


class CooperativeRole(Enum):
    """Role of a cooperative variable in the equation A * B + C"""
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True, slots=True)
class CooperativeMatrix:
    """Matrix that is cooperatively processed by all threads in an opaque mapping."""
    columns: CooperativeSize
    rows: CooperativeSize
    scalar: Scalar
    role: CooperativeRole


class Interpolation(Enum):
    """The interpolation qualifier of a binding or struct field."""
    PERSPECTIVE = "perspective"
    LINEAR = "linear"
    FLAT = "flat"
    PER_VERTEX = "per_vertex"


class Sampling(Enum):
    """The sampling qualifiers of a binding or struct field."""
    CENTER = "center"
    CENTROID = "centroid"
    SAMPLE = "sample"
    FIRST = "first"
    EITHER = "either"


class StorageAccess(IntFlag):
    """Flags describing storage access."""
    LOAD = 1
    STORE = 2
    ATOMIC = 4


class StorageFormat(Enum):
    """Image storage format."""
    # 8-bit formats
    R8_UNORM = "r8unorm"
    R8_SNORM = "r8snorm"
    R8_UINT = "r8uint"
    R8_SINT = "r8sint"

    # 16-bit formats
    R16_UINT = "r16uint"
    R16_SINT = "r16sint"
    R16_FLOAT = "r16float"
    RG8_UNORM = "rg8unorm"
    RG8_SNORM = "rg8snorm"
    RG8_UINT = "rg8uint"
    RG8_SINT = "rg8sint"

    # 32-bit formats
    R32_UINT = "r32uint"
    R32_SINT = "r32sint"
    R32_FLOAT = "r32float"
    RG16_UINT = "rg16uint"
    RG16_SINT = "rg16sint"
    RG16_FLOAT = "rg16float"
    RGBA8_UNORM = "rgba8unorm"
    RGBA8_SNORM = "rgba8snorm"
    RGBA8_UINT = "rgba8uint"
    RGBA8_SINT = "rgba8sint"
    BGRA8_UNORM = "bgra8unorm"

    # Packed 32-bit formats
    RGB10A2_UINT = "rgb10a2uint"
    RGB10A2_UNORM = "rgb10a2unorm"
    RG11B10_UFLOAT = "rg11b10ufloat"

    # 64-bit formats
    R64_UINT = "r64uint"
    RG32_UINT = "rg32uint"
    RG32_SINT = "rg32sint"
    RG32_FLOAT = "rg32float"
    RGBA16_UINT = "rgba16uint"
    RGBA16_SINT = "rgba16sint"
    RGBA16_FLOAT = "rgba16float"

    # 128-bit formats
    RGBA32_UINT = "rgba32uint"
    RGBA32_SINT = "rgba32sint"
    RGBA32_FLOAT = "rgba32float"

    # Normalized 16-bit per channel formats
    R16_UNORM = "r16unorm"
    R16_SNORM = "r16snorm"
    RG16_UNORM = "rg16unorm"
    RG16_SNORM = "rg16snorm"
    RGBA16_UNORM = "rgba16unorm"
    RGBA16_SNORM = "rgba16snorm"


@dataclass(frozen=True, slots=True)
class ImageClassSampled:
    """Regular sampled image."""
    kind: ScalarKind
    multi: bool  # Multi-sampled image


@dataclass(frozen=True, slots=True)
class ImageClassDepth:
    """Depth comparison image."""
    multi: bool  # Multi-sampled depth image


@dataclass(frozen=True, slots=True)
class ImageClassStorage:
    """Storage image."""
    format: StorageFormat
    access: StorageAccess


class ImageClassType(Enum):
    """Sub-class of the image type."""
    SAMPLED = "sampled"
    DEPTH = "depth"
    EXTERNAL = "external"
    STORAGE = "storage"


@dataclass(frozen=True, slots=True)
class ImageClass:
    """Image class variant."""
    type: ImageClassType
    sampled: Optional[ImageClassSampled] = None
    depth: Optional[ImageClassDepth] = None
    storage: Optional[ImageClassStorage] = None

    @classmethod
    def new_sampled(cls, kind: ScalarKind, multi: bool) -> "ImageClass":
        return cls(type=ImageClassType.SAMPLED, sampled=ImageClassSampled(kind, multi))

    @classmethod
    def new_depth(cls, multi: bool) -> "ImageClass":
        return cls(type=ImageClassType.DEPTH, depth=ImageClassDepth(multi))

    @classmethod
    def new_external(cls) -> "ImageClass":
        return cls(type=ImageClassType.EXTERNAL)

    @classmethod
    def new_storage(cls, format: StorageFormat, access: StorageAccess) -> "ImageClass":
        return cls(type=ImageClassType.STORAGE, storage=ImageClassStorage(format, access))


@dataclass(frozen=True, slots=True)
class ArraySizeConstant:
    """The array size is constant."""
    value: int  # NonZeroU32 in Rust


@dataclass(frozen=True, slots=True)
class ArraySizePending:
    """The array size is an override-expression."""
    handle: int  # Handle<Override>


class ArraySizeType(Enum):
    """Size of an array."""
    CONSTANT = "constant"
    PENDING = "pending"
    DYNAMIC = "dynamic"


@dataclass(frozen=True, slots=True)
class ArraySize:
    """Size of an array."""
    type: ArraySizeType
    constant: Optional[ArraySizeConstant] = None
    pending: Optional[ArraySizePending] = None

    @classmethod
    def constant(cls, value: int) -> "ArraySize":
        return cls(type=ArraySizeType.CONSTANT, constant=ArraySizeConstant(value))

    @classmethod
    def pending(cls, handle: int) -> "ArraySize":
        return cls(type=ArraySizeType.PENDING, pending=ArraySizePending(handle))

    @classmethod
    def dynamic(cls) -> "ArraySize":
        return cls(type=ArraySizeType.DYNAMIC)


@dataclass(frozen=True, slots=True)
class StructMember:
    """Member of a user-defined structure."""
    name: Optional[str]
    ty: int  # Handle<Type>
    binding: Optional[Any]  # Option<Binding>
    offset: int


@dataclass(frozen=True, slots=True)
class Array:
    """Homogeneous list of elements."""
    base: int  # Handle<Type>
    size: ArraySize
    stride: int


@dataclass(frozen=True, slots=True)
class Struct:
    """User-defined structure."""
    members: List[StructMember]
    span: int


class ImageDimension(Enum):
    """The number of dimensions an image has."""
    D1 = "1d"
    D2 = "2d"
    D3 = "3d"
    CUBE = "cube"


@dataclass(frozen=True, slots=True)
class Image:
    """Possibly multidimensional array of texels."""
    dim: ImageDimension
    arrayed: bool
    class_: ImageClass


@dataclass(frozen=True, slots=True)
class Sampler:
    """Can be used to sample values from images."""
    comparison: bool


@dataclass(frozen=True, slots=True)
class AccelerationStructure:
    """Opaque object representing an acceleration structure of geometry."""
    vertex_return: bool


@dataclass(frozen=True, slots=True)
class RayQuery:
    """Locally used handle for ray queries."""
    vertex_return: bool


@dataclass(frozen=True, slots=True)
class BindingArray:
    """Array of bindings."""
    base: int  # Handle<Type>
    size: ArraySize


@dataclass(frozen=True, slots=True)
class ValuePointer:
    """Pointer to a scalar or vector."""
    size: Optional[VectorSize]
    scalar: Scalar
    space: str  # AddressSpace


@dataclass(frozen=True, slots=True)
class Pointer:
    """Pointer to another type."""
    base: int  # Handle<Type>
    space: str  # AddressSpace


@dataclass(frozen=True, slots=True)
class Atomic:
    """Atomic scalar."""
    scalar: Scalar


class TypeInnerType(Enum):
    """Enum with additional information, depending on the kind of type."""
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    COOPERATIVE_MATRIX = "cooperative_matrix"
    ATOMIC = "atomic"
    POINTER = "pointer"
    VALUE_POINTER = "value_pointer"
    ARRAY = "array"
    STRUCT = "struct"
    IMAGE = "image"
    SAMPLER = "sampler"
    ACCELERATION_STRUCTURE = "acceleration_structure"
    RAY_QUERY = "ray_query"
    BINDING_ARRAY = "binding_array"


@dataclass(frozen=True, slots=True)
class TypeInner:
    """Enum with additional information, depending on the kind of type."""
    type: TypeInnerType
    scalar: Optional[Scalar] = None
    vector: Optional[Vector] = None
    matrix: Optional[Matrix] = None
    cooperative_matrix: Optional[CooperativeMatrix] = None
    atomic: Optional[Atomic] = None
    pointer: Optional[Pointer] = None
    value_pointer: Optional[ValuePointer] = None
    array: Optional[Array] = None
    struct: Optional[Struct] = None
    image: Optional[Image] = None
    sampler: Optional[Sampler] = None
    acceleration_structure: Optional[AccelerationStructure] = None
    ray_query: Optional[RayQuery] = None
    binding_array: Optional[BindingArray] = None

    @classmethod
    def new_scalar(cls, scalar: Scalar) -> "TypeInner":
        return cls(type=TypeInnerType.SCALAR, scalar=scalar)

    @classmethod
    def new_vector(cls, size: VectorSize, scalar: Scalar) -> "TypeInner":
        return cls(type=TypeInnerType.VECTOR, vector=Vector(size, scalar))

    @classmethod
    def new_matrix(cls, columns: VectorSize, rows: VectorSize, scalar: Scalar) -> "TypeInner":
        return cls(type=TypeInnerType.MATRIX, matrix=Matrix(columns, rows, scalar))

    @classmethod
    def new_cooperative_matrix(
        cls,
        columns: CooperativeSize,
        rows: CooperativeSize,
        scalar: Scalar,
        role: CooperativeRole
    ) -> "TypeInner":
        return cls(
            type=TypeInnerType.COOPERATIVE_MATRIX,
            cooperative_matrix=CooperativeMatrix(columns, rows, scalar, role)
        )

    @classmethod
    def new_atomic(cls, scalar: Scalar) -> "TypeInner":
        return cls(type=TypeInnerType.ATOMIC, atomic=Atomic(scalar))

    @classmethod
    def new_pointer(cls, base: int, space: str) -> "TypeInner":
        return cls(type=TypeInnerType.POINTER, pointer=Pointer(base, space))

    @classmethod
    def new_value_pointer(cls, size: Optional[VectorSize], scalar: Scalar, space: str) -> "TypeInner":
        return cls(type=TypeInnerType.VALUE_POINTER, value_pointer=ValuePointer(size, scalar, space))

    @classmethod
    def new_array(cls, base: int, size: ArraySize, stride: int) -> "TypeInner":
        return cls(type=TypeInnerType.ARRAY, array=Array(base, size, stride))

    @classmethod
    def new_struct(cls, members: List[StructMember], span: int) -> "TypeInner":
        return cls(type=TypeInnerType.STRUCT, struct=Struct(members, span))

    @classmethod
    def new_image(cls, dim: ImageDimension, arrayed: bool, class_: ImageClass) -> "TypeInner":
        return cls(type=TypeInnerType.IMAGE, image=Image(dim, arrayed, class_))

    @classmethod
    def new_sampler(cls, comparison: bool) -> "TypeInner":
        return cls(type=TypeInnerType.SAMPLER, sampler=Sampler(comparison))

    @classmethod
    def new_acceleration_structure(cls, vertex_return: bool) -> "TypeInner":
        return cls(type=TypeInnerType.ACCELERATION_STRUCTURE, acceleration_structure=AccelerationStructure(vertex_return))

    @classmethod
    def new_ray_query(cls, vertex_return: bool) -> "TypeInner":
        return cls(type=TypeInnerType.RAY_QUERY, ray_query=RayQuery(vertex_return))

    @classmethod
    def new_binding_array(cls, base: int, size: ArraySize) -> "TypeInner":
        return cls(type=TypeInnerType.BINDING_ARRAY, binding_array=BindingArray(base, size))


@dataclass(frozen=True, slots=True)
class Type:
    """A data type declared in the module."""
    name: Optional[str]
    inner: TypeInner
