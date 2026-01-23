"""Helper functions for Literal type operations.

Translated from wgpu-trunk/naga/src/proc/constant_evaluator.rs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from naga.ir import Literal, LiteralType, TypeInner, Scalar, ScalarKind, VectorSize

if TYPE_CHECKING:
    pass


class LiteralVector:
    """Vectors with a concrete element type.
    
    Translated from Rust enum LiteralVector in constant_evaluator.rs
    """
    
    def __init__(self, variant: str, values: list):
        """Initialize a LiteralVector.
        
        Args:
            variant: One of 'F64', 'F32', 'F16', 'U32', 'I32', 'U64', 'I64', 'Bool', 'AbstractInt', 'AbstractFloat'
            values: List of values of the appropriate type
        """
        self.variant = variant
        self.values = values
        
    def __len__(self) -> int:
        """Return the length of the vector."""
        return len(self.values)
    
    @staticmethod
    def from_literal(literal: Literal) -> 'LiteralVector':
        """Creates LiteralVector of size 1 from single Literal.
        
        Translated from Rust:
        ```rust
        fn from_literal(literal: Literal) -> Self {
            match literal {
                Literal::F64(e) => Self::F64(ArrayVec::from_iter(iter::once(e))),
                Literal::F32(e) => Self::F32(ArrayVec::from_iter(iter::once(e))),
                ...
            }
        }
        ```
        """
        match literal.type:
            case LiteralType.F64:
                return LiteralVector('F64', [literal.f64])
            case LiteralType.F32:
                return LiteralVector('F32', [literal.f32])
            case LiteralType.F16:
                return LiteralVector('F16', [literal.f16])
            case LiteralType.U32:
                return LiteralVector('U32', [literal.u32])
            case LiteralType.I32:
                return LiteralVector('I32', [literal.i32])
            case LiteralType.U64:
                return LiteralVector('U64', [literal.u64])
            case LiteralType.I64:
                return LiteralVector('I64', [literal.i64])
            case LiteralType.BOOL:
                return LiteralVector('Bool', [literal.bool])
            case LiteralType.ABSTRACT_INT:
                return LiteralVector('AbstractInt', [literal.abstract_int])
            case LiteralType.ABSTRACT_FLOAT:
                return LiteralVector('AbstractFloat', [literal.abstract_float])
            case _:
                raise ValueError(f"Unknown literal type: {literal.type}")
    
    @staticmethod
    def from_literal_vec(components: list[Literal]) -> 'LiteralVector':
        """Creates LiteralVector from list of Literals.
        
        Returns error if component types do not match.
        Panics if vector is empty.
        
        Translated from Rust:
        ```rust
        fn from_literal_vec(
            components: ArrayVec<Literal, { crate::VectorSize::MAX }>
        ) -> Result<Self, ConstantEvaluatorError> {
            assert!(!components.is_empty());
            Ok(match components[0] {
                Literal::I32(_) => Self::I32(
                    components
                        .iter()
                        .map(|l| match l {
                            &Literal::I32(v) => Ok(v),
                            _ => Err(ConstantEvaluatorError::InvalidMathArg),
                        })
                        .collect::<Result<_, _>>()?,
                ),
                ...
            })
        }
        ```
        """
        assert components, "components list cannot be empty"
        
        first = components[0]
        
        match first.type:
            case LiteralType.I32:
                values = []
                for lit in components:
                    if lit.type != LiteralType.I32:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.i32)
                return LiteralVector('I32', values)
                
            case LiteralType.U32:
                values = []
                for lit in components:
                    if lit.type != LiteralType.U32:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.u32)
                return LiteralVector('U32', values)
                
            case LiteralType.I64:
                values = []
                for lit in components:
                    if lit.type != LiteralType.I64:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.i64)
                return LiteralVector('I64', values)
                
            case LiteralType.U64:
                values = []
                for lit in components:
                    if lit.type != LiteralType.U64:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.u64)
                return LiteralVector('U64', values)
                
            case LiteralType.F32:
                values = []
                for lit in components:
                    if lit.type != LiteralType.F32:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.f32)
                return LiteralVector('F32', values)
                
            case LiteralType.F64:
                values = []
                for lit in components:
                    if lit.type != LiteralType.F64:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.f64)
                return LiteralVector('F64', values)
                
            case LiteralType.F16:
                values = []
                for lit in components:
                    if lit.type != LiteralType.F16:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.f16)
                return LiteralVector('F16', values)
                
            case LiteralType.BOOL:
                values = []
                for lit in components:
                    if lit.type != LiteralType.BOOL:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.bool)
                return LiteralVector('Bool', values)
                
            case LiteralType.ABSTRACT_INT:
                values = []
                for lit in components:
                    if lit.type != LiteralType.ABSTRACT_INT:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.abstract_int)
                return LiteralVector('AbstractInt', values)
                
            case LiteralType.ABSTRACT_FLOAT:
                values = []
                for lit in components:
                    if lit.type != LiteralType.ABSTRACT_FLOAT:
                        raise ValueError("InvalidMathArg: mismatched literal types")
                    values.append(lit.abstract_float)
                return LiteralVector('AbstractFloat', values)
                
            case _:
                raise ValueError(f"Unknown literal type: {first.type}")
    
    def to_literal_vec(self) -> list[Literal]:
        """Returns list of Literals.
        
        Translated from Rust:
        ```rust
        fn to_literal_vec(&self) -> ArrayVec<Literal, { crate::VectorSize::MAX }> {
            match *self {
                LiteralVector::F64(ref v) => v.iter().map(|e| Literal::F64(*e)).collect(),
                ...
            }
        }
        ```
        """
        match self.variant:
            case 'F64':
                return [Literal(type=LiteralType.F64, f64=v) for v in self.values]
            case 'F32':
                return [Literal(type=LiteralType.F32, f32=v) for v in self.values]
            case 'F16':
                return [Literal(type=LiteralType.F16, f16=v) for v in self.values]
            case 'U32':
                return [Literal(type=LiteralType.U32, u32=v) for v in self.values]
            case 'I32':
                return [Literal(type=LiteralType.I32, i32=v) for v in self.values]
            case 'U64':
                return [Literal(type=LiteralType.U64, u64=v) for v in self.values]
            case 'I64':
                return [Literal(type=LiteralType.I64, i64=v) for v in self.values]
            case 'Bool':
                return [Literal(type=LiteralType.BOOL, bool=v) for v in self.values]
            case 'AbstractInt':
                return [Literal(type=LiteralType.ABSTRACT_INT, abstract_int=v) for v in self.values]
            case 'AbstractFloat':
                return [Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=v) for v in self.values]
            case _:
                raise ValueError(f"Unknown variant: {self.variant}")


def literal_ty_inner(literal: Literal) -> TypeInner:
    """Return TypeInner for a Literal value.
    
    This is used in type resolution for literal expressions.
    
    Note: In Rust this is implemented as a method on Literal via impl block,
    but in Python we implement it as a standalone function.
    """
    match literal.type:
        case LiteralType.F64:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.FLOAT, width=8))
        case LiteralType.F32:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.FLOAT, width=4))
        case LiteralType.F16:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.FLOAT, width=2))
        case LiteralType.U32:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.UINT, width=4))
        case LiteralType.I32:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.SINT, width=4))
        case LiteralType.U64:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.UINT, width=8))
        case LiteralType.I64:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.SINT, width=8))
        case LiteralType.BOOL:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.BOOL, width=1))
        case LiteralType.ABSTRACT_INT:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.ABSTRACT_INT, width=8))
        case LiteralType.ABSTRACT_FLOAT:
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.ABSTRACT_FLOAT, width=8))
        case _:
            raise ValueError(f"Unknown literal type: {literal.type}")
