from __future__ import annotations
from typing import List, Optional, Any, Union
from naga.ir.type import Type, TypeInner, VectorSize

class Alignment:
    """Helper for calculating type alignments."""
    ONE: int = 1
    TWO: int = 2
    FOUR: int = 4
    EIGHT: int = 8
    SIXTEEN: int = 16
    MIN_UNIFORM: int = SIXTEEN

    @staticmethod
    def is_power_of_two(n: int) -> bool:
        """Check if n is a power of two."""
        return n > 0 and (n & (n - 1)) == 0

    @staticmethod
    def from_width(width: int) -> int:
        """Create an alignment from a width."""
        if not Alignment.is_power_of_two(width):
            return width # Should ideally handle error
        return width

    @staticmethod
    def round_up(alignment: int, n: int) -> int:
        """Round n up to the nearest alignment boundary."""
        mask = alignment - 1
        return (n + mask) & ~mask

class TypeLayout:
    """Size and alignment information for a type."""
    def __init__(self, size: int, alignment: int):
        self.size: int = size
        self.alignment: int = alignment

    def to_stride(self) -> int:
        """Produce the stride as if this type is a base of an array."""
        return Alignment.round_up(self.alignment, self.size)

class Layouter:
    """
    Naga layout logic.
    """
    def __init__(self) -> None:
        self.layouts: List[TypeLayout] = []

    def update(self, module: Any) -> None:
        """
        Extend this Layouter with layouts for any new entries in module.types.
        
        Args:
            module: The Naga IR module to update layouts for.
        """
        for i in range(len(self.layouts), len(module.types)):
            ty = module.types[i]
            inner = ty.inner
            
            size = self._try_size(ty, module)
            
            alignment = 1
            if inner == TypeInner.SCALAR or inner == TypeInner.ATOMIC:
                scalar = getattr(ty, "_scalar", None) or getattr(ty, "_atomic", None)
                if scalar:
                    alignment = scalar.width
            elif inner == TypeInner.VECTOR:
                vector = getattr(ty, "_vector", None)
                if vector:
                    alignment = vector.width
                    if vector.size == VectorSize.BI:
                        alignment *= 2
                    else:
                        alignment *= 4
            elif inner == TypeInner.MATRIX:
                matrix = getattr(ty, "_matrix", None)
                if matrix:
                    alignment = matrix.width
                    if matrix.rows == VectorSize.BI:
                        alignment *= 2
                    else:
                        alignment *= 4
            elif inner == TypeInner.POINTER or inner == TypeInner.VALUE_POINTER:
                alignment = 1
            elif inner == TypeInner.ARRAY:
                array = getattr(ty, "_array", None)
                if array and array.base < i:
                    alignment = self.layouts[array.base].alignment
                else:
                    alignment = 1
            elif inner == TypeInner.STRUCT:
                struct = getattr(ty, "_struct", None)
                alignment = 1
                if struct:
                    for member in struct.members:
                        if member.ty < i:
                            alignment = max(alignment, self.layouts[member.ty].alignment)
            else:
                alignment = 1
                
            self.layouts.append(TypeLayout(size, alignment))

    def _try_size(self, ty: Type, module: Any) -> int:
        """Attempt to calculate the size of this type."""
        inner = ty.inner
        if inner == TypeInner.SCALAR or inner == TypeInner.ATOMIC:
            scalar = getattr(ty, "_scalar", None) or getattr(ty, "_atomic", None)
            return scalar.width if scalar else 0
        elif inner == TypeInner.VECTOR:
            vector = getattr(ty, "_vector", None)
            return vector.size.value * vector.width if vector else 0
        elif inner == TypeInner.MATRIX:
            matrix = getattr(ty, "_matrix", None)
            if matrix:
                rows_alignment = 2 if matrix.rows == VectorSize.BI else 4
                return rows_alignment * matrix.width * matrix.columns.value
            return 0
        elif inner == TypeInner.POINTER or inner == TypeInner.VALUE_POINTER:
            return 4
        elif inner == TypeInner.ARRAY:
            array = getattr(ty, "_array", None)
            if array:
                count = array.size if isinstance(array.size, int) else 1
                stride = array.stride if array.stride is not None else 0
                return count * stride
            return 0
        elif inner == TypeInner.STRUCT:
            struct = getattr(ty, "_struct", None)
            return struct.span if struct else 0
        else:
            return 0

    def __getitem__(self, handle: int) -> TypeLayout:
        return self.layouts[handle]
