"""
Constant and override definitions.
Transcribed from wgpu/naga/src/ir/mod.rs
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


@dataclass(frozen=True, slots=True)
class Override:
    """
    Pipeline-overridable constant.
    """
    name: Optional[str]
    id: Optional[int]  # u16 - Pipeline Constant ID
    ty: int  # Handle<Type>
    init: Optional[int]  # Option<Handle<Expression>> - Default value from global_expressions


@dataclass(frozen=True, slots=True)
class Constant:
    """
    Constant value.
    """
    name: Optional[str]
    ty: int  # Handle<Type>
    init: int  # Handle<Expression> - The value of the constant from global_expressions


class LiteralType(Enum):
    """Literal value type."""
    F64 = "f64"
    F32 = "f32"
    F16 = "f16"
    U32 = "u32"
    I32 = "i32"
    U64 = "u64"
    I64 = "i64"
    BOOL = "bool"
    ABSTRACT_INT = "abstract-int"
    ABSTRACT_FLOAT = "abstract-float"


@dataclass(frozen=True, slots=True)
class Literal:
    """Literal value."""
    type: LiteralType
    value: int | float | bool

    @classmethod
    def f64(cls, value: float) -> "Literal":
        return cls(type=LiteralType.F64, value=value)

    @classmethod
    def f32(cls, value: float) -> "Literal":
        return cls(type=LiteralType.F32, value=value)

    @classmethod
    def f16(cls, value: float) -> "Literal":
        return cls(type=LiteralType.F16, value=value)

    @classmethod
    def u32(cls, value: int) -> "Literal":
        return cls(type=LiteralType.U32, value=value)

    @classmethod
    def i32(cls, value: int) -> "Literal":
        return cls(type=LiteralType.I32, value=value)

    @classmethod
    def u64(cls, value: int) -> "Literal":
        return cls(type=LiteralType.U64, value=value)

    @classmethod
    def i64(cls, value: int) -> "Literal":
        return cls(type=LiteralType.I64, value=value)

    @classmethod
    def bool_(cls, value: bool) -> "Literal":
        return cls(type=LiteralType.BOOL, value=value)

    @classmethod
    def abstract_int(cls, value: int) -> "Literal":
        return cls(type=LiteralType.ABSTRACT_INT, value=value)

    @classmethod
    def abstract_float(cls, value: float) -> "Literal":
        return cls(type=LiteralType.ABSTRACT_FLOAT, value=value)


__all__ = [
    "Override",
    "Constant",
    "Literal",
    "LiteralType",
]
