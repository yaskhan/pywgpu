"""
Operator definitions used in expressions and statements.
Transcribed from wgpu/naga/src/ir/mod.rs
"""

from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Optional


class UnaryOperator(Enum):
    """Operation that can be applied on a single value."""
    NEGATE = "negate"
    LOGICAL_NOT = "logical_not"
    BITWISE_NOT = "bitwise_not"


class BinaryOperator(Enum):
    """
    Operation that can be applied on two values.

    Arithmetic type rules:
    - Add, Subtract, Multiply, Divide, Modulo can be applied to Scalar types
      other than Bool, or Vectors thereof. Both operands must have same type.
    - Add and Subtract can also be applied to Matrix values.
    - Multiply supports additional cases:
      * Matrix or Vector can be multiplied by scalar Float
      * Matrix on left can be multiplied by Vector on right (if columns == components)
      * Vector on left can be multiplied by Matrix on right (if components == rows)
      * Two matrices can be multiplied (columns of left == rows of right)
    """
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    MODULO = "modulo"  # WGSL's % operator or SPIR-V's OpFRem
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    AND = "and"
    EXCLUSIVE_OR = "exclusive_or"
    INCLUSIVE_OR = "inclusive_or"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"  # Right shift carries sign of signed integers


class AtomicFunction(Enum):
    """
    Function on an atomic value.

    Note: these do not include load/store, which use
    Expression::Load and Statement::Store.
    """
    ADD = "add"
    SUBTRACT = "subtract"
    AND = "and"
    EXCLUSIVE_OR = "exclusive_or"
    INCLUSIVE_OR = "inclusive_or"
    MIN = "min"
    MAX = "max"
    EXCHANGE = "exchange"


@dataclass(frozen=True, slots=True)
class AtomicFunctionExchange:
    """Exchange variant with optional comparison value."""
    compare: Optional[int]  # Option<Handle<Expression>>


class DerivativeControl(Enum):
    """Hint at which precision to compute a derivative."""
    COARSE = "coarse"
    FINE = "fine"
    NONE = "none"


class DerivativeAxis(Enum):
    """Axis on which to compute a derivative."""
    X = "x"
    Y = "y"
    WIDTH = "width"


class RelationalFunction(Enum):
    """Built-in shader function for testing relation between values."""
    ALL = "all"
    ANY = "any"
    IS_NAN = "is_nan"
    IS_INF = "is_inf"


class MathFunction(Enum):
    """Built-in shader function for math."""
    # comparison
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    CLAMP = "clamp"
    SATURATE = "saturate"
    # trigonometry
    COS = "cos"
    COSH = "cosh"
    SIN = "sin"
    SINH = "sinh"
    TAN = "tan"
    TANH = "tanh"
    ACOS = "acos"
    ASIN = "asin"
    ATAN = "atan"
    ATAN2 = "atan2"
    ASINH = "asinh"
    ACOSH = "acosh"
    ATANH = "atanh"
    RADIANS = "radians"
    DEGREES = "degrees"
    # decomposition
    CEIL = "ceil"
    FLOOR = "floor"
    ROUND = "round"
    FRACT = "fract"
    TRUNC = "trunc"
    MODF = "modf"
    FREXP = "frexp"
    LDEXP = "ldexp"
    # exponent
    EXP = "exp"
    EXP2 = "exp2"
    LOG = "log"
    LOG2 = "log2"
    POW = "pow"
    # geometry
    DOT = "dot"
    DOT4_I8_PACKED = "dot4_i8_packed"
    DOT4_U8_PACKED = "dot4_u8_packed"
    OUTER = "outer"
    CROSS = "cross"
    DISTANCE = "distance"
    LENGTH = "length"
    NORMALIZE = "normalize"
    FACE_FORWARD = "face_forward"
    REFLECT = "reflect"
    REFRACT = "refract"
    # computational
    SIGN = "sign"
    FMA = "fma"
    MIX = "mix"
    STEP = "step"
    SMOOTH_STEP = "smooth_step"
    SQRT = "sqrt"
    INVERSE_SQRT = "inverse_sqrt"
    INVERSE = "inverse"
    TRANSPOSE = "transpose"
    DETERMINANT = "determinant"
    QUANTIZE_TO_F16 = "quantize_to_f16"
    # bits
    COUNT_TRAILING_ZEROS = "count_trailing_zeros"
    COUNT_LEADING_ZEROS = "count_leading_zeros"
    COUNT_ONE_BITS = "count_one_bits"
    REVERSE_BITS = "reverse_bits"
    EXTRACT_BITS = "extract_bits"
    INSERT_BITS = "insert_bits"
    FIRST_TRAILING_BIT = "first_trailing_bit"
    FIRST_LEADING_BIT = "first_leading_bit"
    # data packing
    PACK_4X8_SNORM = "pack_4x8_snorm"
    PACK_4X8_UNORM = "pack_4x8_unorm"
    PACK_2X16_SNORM = "pack_2x16_snorm"
    PACK_2X16_UNORM = "pack_2x16_unorm"
    PACK_2X16_FLOAT = "pack_2x16_float"
    PACK_4X_I8 = "pack_4x_i8"
    PACK_4X_U8 = "pack_4x_u8"
    PACK_4X_I8_CLAMP = "pack_4x_i8_clamp"
    PACK_4X_U8_CLAMP = "pack_4x_u8_clamp"
    # data unpacking
    UNPACK_4X8_SNORM = "unpack_4x8_snorm"
    UNPACK_4X8_UNORM = "unpack_4x8_unorm"
    UNPACK_2X16_SNORM = "unpack_2x16_snorm"
    UNPACK_2X16_UNORM = "unpack_2x16_unorm"
    UNPACK_2X16_FLOAT = "unpack_2x16_float"
    UNPACK_4X_I8 = "unpack_4x_i8"
    UNPACK_4X_U8 = "unpack_4x_u8"


class SwizzleComponent(Enum):
    """Component selection for a vector swizzle."""
    X = 0
    Y = 1
    Z = 2
    W = 3


@dataclass(frozen=True, slots=True)
class SampleLevelAuto:
    """Auto level selection."""
    pass


@dataclass(frozen=True, slots=True)
class SampleLevelZero:
    """Zero level selection."""
    pass


@dataclass(frozen=True, slots=True)
class SampleLevelExact:
    """Exact level selection."""
    expr: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class SampleLevelBias:
    """Bias level selection."""
    expr: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class SampleLevelGradient:
    """Gradient level selection."""
    x: int  # Handle<Expression>
    y: int  # Handle<Expression>


class SampleLevelType(Enum):
    """Sampling modifier to control the level of detail."""
    AUTO = "auto"
    ZERO = "zero"
    EXACT = "exact"
    BIAS = "bias"
    GRADIENT = "gradient"


@dataclass(frozen=True, slots=True)
class SampleLevel:
    """Sampling modifier to control the level of detail."""
    type: SampleLevelType
    exact: Optional[SampleLevelExact] = None
    bias: Optional[SampleLevelBias] = None
    gradient: Optional[SampleLevelGradient] = None

    @classmethod
    def auto(cls) -> "SampleLevel":
        return cls(type=SampleLevelType.AUTO)

    @classmethod
    def zero(cls) -> "SampleLevel":
        return cls(type=SampleLevelType.ZERO)

    @classmethod
    def exact(cls, expr: int) -> "SampleLevel":
        return cls(type=SampleLevelType.EXACT, exact=SampleLevelExact(expr))

    @classmethod
    def bias(cls, expr: int) -> "SampleLevel":
        return cls(type=SampleLevelType.BIAS, bias=SampleLevelBias(expr))

    @classmethod
    def gradient(cls, x: int, y: int) -> "SampleLevel":
        return cls(type=SampleLevelType.GRADIENT, gradient=SampleLevelGradient(x, y))


@dataclass(frozen=True, slots=True)
class ImageQuerySize:
    """Get the size at the specified level."""
    level: Optional[int]  # Option<Handle<Expression>>


class ImageQueryType(Enum):
    """Type of an image query."""
    SIZE = "size"
    NUM_LEVELS = "num_levels"
    NUM_LAYERS = "num_layers"
    NUM_SAMPLES = "num_samples"


@dataclass(frozen=True, slots=True)
class ImageQuery:
    """Type of an image query."""
    type: ImageQueryType
    size: Optional[ImageQuerySize] = None

    @classmethod
    def new_size(cls, level: Optional[int] = None) -> "ImageQuery":
        return cls(type=ImageQueryType.SIZE, size=ImageQuerySize(level))

    @classmethod
    def new_num_levels(cls) -> "ImageQuery":
        return cls(type=ImageQueryType.NUM_LEVELS)

    @classmethod
    def new_num_layers(cls) -> "ImageQuery":
        return cls(type=ImageQueryType.NUM_LAYERS)

    @classmethod
    def new_num_samples(cls) -> "ImageQuery":
        return cls(type=ImageQueryType.NUM_SAMPLES)


@dataclass(frozen=True, slots=True)
class Direction:
    """Direction for quad swap."""
    value: int  # X=0, Y=1, Diagonal=2


@dataclass(frozen=True, slots=True)
class GatherModeBroadcastFirst:
    """All gather from the active lane with the smallest index."""
    pass


@dataclass(frozen=True, slots=True)
class GatherModeBroadcast:
    """All gather from the same lane at the given index."""
    index: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class GatherModeShuffle:
    """Each gathers from a different lane at the given index."""
    index: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class GatherModeShuffleDown:
    """Each gathers from their lane plus the given shift."""
    shift: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class GatherModeShuffleUp:
    """Each gathers from their lane minus the given shift."""
    shift: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class GatherModeShuffleXor:
    """Each gathers from their lane xored with the given by the expression."""
    xor: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class GatherModeQuadBroadcast:
    """All gather from the same quad lane at the given index."""
    index: int  # Handle<Expression>


@dataclass(frozen=True, slots=True)
class GatherModeQuadSwap:
    """Each gathers from the opposite quad lane along the given direction."""
    direction: Direction


class GatherModeType(Enum):
    """The specific behavior of a SubgroupGather statement."""
    BROADCAST_FIRST = "broadcast_first"
    BROADCAST = "broadcast"
    SHUFFLE = "shuffle"
    SHUFFLE_DOWN = "shuffle_down"
    SHUFFLE_UP = "shuffle_up"
    SHUFFLE_XOR = "shuffle_xor"
    QUAD_BROADCAST = "quad_broadcast"
    QUAD_SWAP = "quad_swap"


@dataclass(frozen=True, slots=True)
class GatherMode:
    """The specific behavior of a SubgroupGather statement."""
    type: GatherModeType
    broadcast_first: Optional[GatherModeBroadcastFirst] = None
    broadcast: Optional[GatherModeBroadcast] = None
    shuffle: Optional[GatherModeShuffle] = None
    shuffle_down: Optional[GatherModeShuffleDown] = None
    shuffle_up: Optional[GatherModeShuffleUp] = None
    shuffle_xor: Optional[GatherModeShuffleXor] = None
    quad_broadcast: Optional[GatherModeQuadBroadcast] = None
    quad_swap: Optional[GatherModeQuadSwap] = None

    @classmethod
    def broadcast_first(cls) -> "GatherMode":
        return cls(type=GatherModeType.BROADCAST_FIRST, broadcast_first=GatherModeBroadcastFirst())

    @classmethod
    def broadcast(cls, index: int) -> "GatherMode":
        return cls(type=GatherModeType.BROADCAST, broadcast=GatherModeBroadcast(index))

    @classmethod
    def shuffle(cls, index: int) -> "GatherMode":
        return cls(type=GatherModeType.SHUFFLE, shuffle=GatherModeShuffle(index))

    @classmethod
    def shuffle_down(cls, shift: int) -> "GatherMode":
        return cls(type=GatherModeType.SHUFFLE_DOWN, shuffle_down=GatherModeShuffleDown(shift))

    @classmethod
    def shuffle_up(cls, shift: int) -> "GatherMode":
        return cls(type=GatherModeType.SHUFFLE_UP, shuffle_up=GatherModeShuffleUp(shift))

    @classmethod
    def shuffle_xor(cls, xor: int) -> "GatherMode":
        return cls(type=GatherModeType.SHUFFLE_XOR, shuffle_xor=GatherModeShuffleXor(xor))

    @classmethod
    def quad_broadcast(cls, index: int) -> "GatherMode":
        return cls(type=GatherModeType.QUAD_BROADCAST, quad_broadcast=GatherModeQuadBroadcast(index))

    @classmethod
    def quad_swap(cls, direction: Direction) -> "GatherMode":
        return cls(type=GatherModeType.QUAD_SWAP, quad_swap=GatherModeQuadSwap(direction))


class SubgroupOperation(Enum):
    """Subgroup operation type."""
    ALL = 0
    ANY = 1
    ADD = 2
    MUL = 3
    MIN = 4
    MAX = 5
    AND = 6
    OR = 7
    XOR = 8


class CollectiveOperation(Enum):
    """How to combine subgroup results."""
    REDUCE = 0
    INCLUSIVE_SCAN = 1
    EXCLUSIVE_SCAN = 2


class Barrier(IntFlag):
    """Memory barrier flags."""
    STORAGE = 1 << 0
    WORK_GROUP = 1 << 1
    SUB_GROUP = 1 << 2
    TEXTURE = 1 << 3


__all__ = [
    "UnaryOperator",
    "BinaryOperator",
    "AtomicFunction",
    "AtomicFunctionExchange",
    "DerivativeControl",
    "DerivativeAxis",
    "RelationalFunction",
    "MathFunction",
    "SwizzleComponent",
    "SampleLevel",
    "SampleLevelType",
    "ImageQuery",
    "ImageQueryType",
    "ImageQuerySize",
    "GatherMode",
    "GatherModeType",
    "Direction",
    "SubgroupOperation",
    "CollectiveOperation",
    "Barrier",
]
