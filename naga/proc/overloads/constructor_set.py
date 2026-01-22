from enum import IntFlag
from dataclasses import dataclass
from typing import List, Optional, Union, Iterator
from naga import ir

@dataclass(frozen=True)
class ConstructorSizeScalar:
    def to_inner(self, scalar: ir.Scalar) -> ir.TypeInner:
        return ir.TypeInner.scalar(scalar)

@dataclass(frozen=True)
class ConstructorSizeVector:
    size: ir.VectorSize
    def to_inner(self, scalar: ir.Scalar) -> ir.TypeInner:
        return ir.TypeInner.vector(self.size, scalar)

@dataclass(frozen=True)
class ConstructorSizeMatrix:
    columns: ir.VectorSize
    rows: ir.VectorSize
    def to_inner(self, scalar: ir.Scalar) -> ir.TypeInner:
        return ir.TypeInner.matrix(self.columns, self.rows, scalar)

ConstructorSize = Union[ConstructorSizeScalar, ConstructorSizeVector, ConstructorSizeMatrix]

class ConstructorSet(IntFlag):
    """A set of type constructors, represented as a bitset."""
    SCALAR = 1 << 0
    VEC2 = 1 << 1
    VEC3 = 1 << 2
    VEC4 = 1 << 3
    MAT2X2 = 1 << 4
    MAT2X3 = 1 << 5
    MAT2X4 = 1 << 6
    MAT3X2 = 1 << 7
    MAT3X3 = 1 << 8
    MAT3X4 = 1 << 9
    MAT4X2 = 1 << 10
    MAT4X3 = 1 << 11
    MAT4X4 = 1 << 12

    VECN = VEC2 | VEC3 | VEC4
    EMPTY = 0

    @classmethod
    def singleton(cls, inner: ir.TypeInner) -> 'ConstructorSet':
        from naga.ir.type import TypeInnerType, VectorSize
        if inner.type == TypeInnerType.SCALAR:
            return cls.SCALAR
        elif inner.type == TypeInnerType.VECTOR:
            size = inner.vector.size
            if size == VectorSize.BI:
                return cls.VEC2
            elif size == VectorSize.TRI:
                return cls.VEC3
            elif size == VectorSize.QUAD:
                return cls.VEC4
        elif inner.type == TypeInnerType.MATRIX:
            columns = inner.matrix.columns
            rows = inner.matrix.rows
            if columns == VectorSize.BI:
                if rows == VectorSize.BI: return cls.MAT2X2
                if rows == VectorSize.TRI: return cls.MAT2X3
                if rows == VectorSize.QUAD: return cls.MAT2X4
            elif columns == VectorSize.TRI:
                if rows == VectorSize.BI: return cls.MAT3X2
                if rows == VectorSize.TRI: return cls.MAT3X3
                if rows == VectorSize.QUAD: return cls.MAT3X4
            elif columns == VectorSize.QUAD:
                if rows == VectorSize.BI: return cls.MAT4X2
                if rows == VectorSize.TRI: return cls.MAT4X3
                if rows == VectorSize.QUAD: return cls.MAT4X4
        return cls.EMPTY

    def is_singleton(self) -> bool:
        bits = self.value
        return bits > 0 and (bits & (bits - 1)) == 0

    def members(self) -> Iterator['ConstructorSet']:
        from .one_bits_iter import OneBitsIter
        for bit in OneBitsIter(self.value):
            yield ConstructorSet(1 << bit)

    def size(self) -> ConstructorSize:
        from naga.ir.type import VectorSize
        if self == ConstructorSet.SCALAR:
            return ConstructorSizeScalar()
        elif self == ConstructorSet.VEC2:
            return ConstructorSizeVector(VectorSize.BI)
        elif self == ConstructorSet.VEC3:
            return ConstructorSizeVector(VectorSize.TRI)
        elif self == ConstructorSet.VEC4:
            return ConstructorSizeVector(VectorSize.QUAD)
        elif self == ConstructorSet.MAT2X2:
            return ConstructorSizeMatrix(VectorSize.BI, VectorSize.BI)
        elif self == ConstructorSet.MAT2X3:
            return ConstructorSizeMatrix(VectorSize.BI, VectorSize.TRI)
        elif self == ConstructorSet.MAT2X4:
            return ConstructorSizeMatrix(VectorSize.BI, VectorSize.QUAD)
        elif self == ConstructorSet.MAT3X2:
            return ConstructorSizeMatrix(VectorSize.TRI, VectorSize.BI)
        elif self == ConstructorSet.MAT3X3:
            return ConstructorSizeMatrix(VectorSize.TRI, VectorSize.TRI)
        elif self == ConstructorSet.MAT3X4:
            return ConstructorSizeMatrix(VectorSize.TRI, VectorSize.QUAD)
        elif self == ConstructorSet.MAT4X2:
            return ConstructorSizeMatrix(VectorSize.QUAD, VectorSize.BI)
        elif self == ConstructorSet.MAT4X3:
            return ConstructorSizeMatrix(VectorSize.QUAD, VectorSize.TRI)
        elif self == ConstructorSet.MAT4X4:
            return ConstructorSizeMatrix(VectorSize.QUAD, VectorSize.QUAD)
        else:
            raise ValueError(f"ConstructorSet {self} was not a singleton")

    def union(self, other: 'ConstructorSet') -> 'ConstructorSet':
        return self | other

    def intersection(self, other: 'ConstructorSet') -> 'ConstructorSet':
        return self & other
