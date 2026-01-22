from enum import IntFlag
from typing import Iterator
from naga.ir.type import Scalar, ScalarKind

SCALARS_FOR_BITS = [
    Scalar(ScalarKind.ABSTRACT_INT, 0),    # ABSTRACT_INT
    Scalar(ScalarKind.ABSTRACT_FLOAT, 0),  # ABSTRACT_FLOAT
    Scalar(ScalarKind.SINT, 4),            # I32
    Scalar(ScalarKind.SINT, 8),            # I64
    Scalar(ScalarKind.UINT, 4),            # U32
    Scalar(ScalarKind.UINT, 8),            # U64
    Scalar(ScalarKind.FLOAT, 4),           # F32
    Scalar(ScalarKind.FLOAT, 2),           # F16
    Scalar(ScalarKind.FLOAT, 8),           # F64
    Scalar(ScalarKind.BOOL, 1),            # BOOL
]

class ScalarSet(IntFlag):
    """A set of scalar types."""
    ABSTRACT_INT = 1 << 0
    ABSTRACT_FLOAT = 1 << 1
    I32 = 1 << 2
    I64 = 1 << 3
    U32 = 1 << 4
    U64 = 1 << 5
    F32 = 1 << 6
    F16 = 1 << 7
    F64 = 1 << 8
    BOOL = 1 << 9

    @classmethod
    def all_flags(cls) -> 'ScalarSet':
        return cls((1 << len(SCALARS_FOR_BITS)) - 1)

    @classmethod
    def singleton(cls, scalar: Scalar) -> 'ScalarSet':
        for i, s in enumerate(SCALARS_FOR_BITS):
            if s == scalar:
                return cls(1 << i)
        return cls(0)

    @classmethod
    def convertible_from(cls, scalar: Scalar) -> 'ScalarSet':
        kind = scalar.kind
        if kind == ScalarKind.SINT:
            if scalar.width == 4: return cls.I32
            if scalar.width == 8: return cls.I64
        elif kind == ScalarKind.UINT:
            if scalar.width == 4: return cls.U32
            if scalar.width == 8: return cls.U64
        elif kind == ScalarKind.FLOAT:
            if scalar.width == 2: return cls.F16
            if scalar.width == 4: return cls.F32
            if scalar.width == 8: return cls.F64
        elif kind == ScalarKind.BOOL:
            return cls.BOOL
        elif kind == ScalarKind.ABSTRACT_INT:
            return cls.INTEGER | cls.FLOAT
        elif kind == ScalarKind.ABSTRACT_FLOAT:
            return cls.FLOAT
        return cls(0)

    def most_general_scalar(self) -> Scalar:
        bits = self.value
        if bits == 0:
            raise ValueError("ScalarSet is empty")
        # Find the lowest bit set
        lowest = (bits & -bits).bit_length() - 1
        return SCALARS_FOR_BITS[lowest]

    def members(self) -> Iterator['ScalarSet']:
        from .one_bits_iter import OneBitsIter
        for bit in OneBitsIter(self.value):
            yield ScalarSet(1 << bit)

    def union(self, other: 'ScalarSet') -> 'ScalarSet':
        return self | other

    def intersection(self, other: 'ScalarSet') -> 'ScalarSet':
        return self & other

    def difference(self, other: 'ScalarSet') -> 'ScalarSet':
        return self & ~other

ScalarSet.FLOAT = ScalarSet.ABSTRACT_FLOAT | ScalarSet.F16 | ScalarSet.F32 | ScalarSet.F64
ScalarSet.INTEGER = ScalarSet.ABSTRACT_INT | ScalarSet.I32 | ScalarSet.I64 | ScalarSet.U32 | ScalarSet.U64
ScalarSet.NUMERIC = ScalarSet.FLOAT | ScalarSet.INTEGER
ScalarSet.ABSTRACT = ScalarSet.ABSTRACT_INT | ScalarSet.ABSTRACT_FLOAT
ScalarSet.CONCRETE = ScalarSet.all_flags() & ~ScalarSet.ABSTRACT
ScalarSet.CONCRETE_INTEGER = ScalarSet.INTEGER & ScalarSet.CONCRETE
ScalarSet.CONCRETE_FLOAT = ScalarSet.FLOAT & ScalarSet.CONCRETE
ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED = ScalarSet.CONCRETE_FLOAT
