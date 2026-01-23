"""Zero value evaluation for constant evaluator.

Translated from wgpu-trunk/naga/src/proc/constant_evaluator.rs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from naga.ir import (
    TypeInner, TypeInnerType, Scalar, ScalarKind, ArraySize,
    Expression, ExpressionType, Literal, LiteralType, Type
)
from naga.span import Span

if TYPE_CHECKING:
    from naga import Handle, UniqueArena, Arena
    from naga.proc.constant_evaluator import ConstantEvaluator


def literal_zero(scalar: Scalar) -> Literal | None:
    """Create a zero literal for the given scalar type.
    
    Translated from Rust:
    ```rust
    pub const fn zero(scalar: crate::Scalar) -> Option<Self> {
        Self::new(0, scalar)
    }
    ```
    
    where `new` is:
    ```rust
    pub const fn new(value: u8, scalar: crate::Scalar) -> Option<Self> {
        match (value, scalar.kind, scalar.width) {
            (value, crate::ScalarKind::Float, 8) => Some(Self::F64(value as _)),
            ...
        }
    }
    ```
    """
    match (scalar.kind, scalar.width):
        case (ScalarKind.FLOAT, 8):
            return Literal(type=LiteralType.F64, f64=0.0)
        case (ScalarKind.FLOAT, 4):
            return Literal(type=LiteralType.F32, f32=0.0)
        case (ScalarKind.FLOAT, 2):
            return Literal(type=LiteralType.F16, f16=0.0)
        case (ScalarKind.UINT, 4):
            return Literal(type=LiteralType.U32, u32=0)
        case (ScalarKind.SINT, 4):
            return Literal(type=LiteralType.I32, i32=0)
        case (ScalarKind.UINT, 8):
            return Literal(type=LiteralType.U64, u64=0)
        case (ScalarKind.SINT, 8):
            return Literal(type=LiteralType.I64, i64=0)
        case (ScalarKind.BOOL, 1):  # BOOL_WIDTH = 1
            return Literal(type=LiteralType.BOOL, bool=False)
        case (ScalarKind.ABSTRACT_INT, 8):
            return Literal(type=LiteralType.ABSTRACT_INT, abstract_int=0)
        case (ScalarKind.ABSTRACT_FLOAT, 8):
            return Literal(type=LiteralType.ABSTRACT_FLOAT, abstract_float=0.0)
        case _:
            return None


def eval_zero_value_impl(
    evaluator: ConstantEvaluator,
    ty: Handle,
    span: Span
) -> Handle:
    """Lower ZeroValue expressions to Literal and Compose expressions.
    
    Translated from Rust constant_evaluator.rs:2178-2247:
    ```rust
    fn eval_zero_value_impl(
        &mut self,
        ty: Handle<Type>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.types[ty].inner {
            TypeInner::Scalar(scalar) => {
                let expr = Expression::Literal(
                    Literal::zero(scalar).ok_or(ConstantEvaluatorError::TypeNotConstructible)?,
                );
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Vector { size, scalar } => {
                let scalar_ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Scalar(scalar),
                    },
                    span,
                );
                let el = self.eval_zero_value_impl(scalar_ty, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; size as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                let vec_ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector { size: rows, scalar },
                    },
                    span,
                );
                let el = self.eval_zero_value_impl(vec_ty, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; columns as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Array {
                base,
                size: ArraySize::Constant(size),
                ..
            } => {
                let el = self.eval_zero_value_impl(base, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; size.get() as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Struct { ref members, .. } => {
                let types: Vec<_> = members.iter().map(|m| m.ty).collect();
                let mut components = Vec::with_capacity(members.len());
                for ty in types {
                    components.push(self.eval_zero_value_impl(ty, span)?);
                }
                let expr = Expression::Compose { ty, components };
                self.register_evaluated_expr(expr, span)
            }
            _ => Err(ConstantEvaluatorError::TypeNotConstructible),
        }
    }
    ```
    """
    type_inner = evaluator.types[ty].inner
    
    match type_inner.type:
        case TypeInnerType.SCALAR:
            # Scalar type - create zero literal
            scalar = type_inner.scalar
            zero_lit = literal_zero(scalar)
            if zero_lit is None:
                raise ValueError("TypeNotConstructible")
            
            expr = Expression(type=ExpressionType.LITERAL, literal=zero_lit)
            return evaluator._append_expr(expr, span, evaluator.expression_kind_tracker.type_of_with_expr(expr))
        
        case TypeInnerType.VECTOR:
            # Vector type - compose from scalar zeros
            size = type_inner.vector_size
            scalar = type_inner.vector_scalar
            
            # Insert scalar type
            scalar_ty = evaluator.types.insert(
                Type(name=None, inner=TypeInner.new_scalar(scalar)),
                span
            )
            
            # Recursively evaluate zero for scalar
            el = eval_zero_value_impl(evaluator, scalar_ty, span)
            
            # Create compose expression with repeated element
            from naga.ir import VectorSize
            size_int = {
                VectorSize.BI: 2,
                VectorSize.TRI: 3,
                VectorSize.QUAD: 4
            }[size]
            
            expr = Expression(
                type=ExpressionType.COMPOSE,
                compose_ty=ty,
                compose_components=[el] * size_int
            )
            return evaluator._append_expr(expr, span, evaluator.expression_kind_tracker.type_of_with_expr(expr))
        
        case TypeInnerType.MATRIX:
            # Matrix type - compose from vector zeros
            columns = type_inner.matrix_columns
            rows = type_inner.matrix_rows
            scalar = type_inner.matrix_scalar
            
            # Insert vector type
            vec_ty = evaluator.types.insert(
                Type(name=None, inner=TypeInner.new_vector(rows, scalar)),
                span
            )
            
            # Recursively evaluate zero for vector
            el = eval_zero_value_impl(evaluator, vec_ty, span)
            
            # Create compose expression with repeated element
            from naga.ir import VectorSize
            columns_int = {
                VectorSize.BI: 2,
                VectorSize.TRI: 3,
                VectorSize.QUAD: 4
            }[columns]
            
            expr = Expression(
                type=ExpressionType.COMPOSE,
                compose_ty=ty,
                compose_components=[el] * columns_int
            )
            return evaluator._append_expr(expr, span, evaluator.expression_kind_tracker.type_of_with_expr(expr))
        
        case TypeInnerType.ARRAY:
            # Array type - compose from element zeros
            base = type_inner.array_base
            size = type_inner.array_size
            
            # Only handle constant-sized arrays
            if size.type != ArraySize.CONSTANT:
                raise ValueError("TypeNotConstructible: non-constant array size")
            
            # Recursively evaluate zero for element type
            el = eval_zero_value_impl(evaluator, base, span)
            
            # Create compose expression with repeated element
            count = size.constant.get()  # NonMaxU32.get() returns the value
            
            expr = Expression(
                type=ExpressionType.COMPOSE,
                compose_ty=ty,
                compose_components=[el] * count
            )
            return evaluator._append_expr(expr, span, evaluator.expression_kind_tracker.type_of_with_expr(expr))
        
        case TypeInnerType.STRUCT:
            # Struct type - compose from member zeros
            members = type_inner.struct_members
            
            # Recursively evaluate zero for each member
            components = []
            for member in members:
                member_zero = eval_zero_value_impl(evaluator, member.ty, span)
                components.append(member_zero)
            
            expr = Expression(
                type=ExpressionType.COMPOSE,
                compose_ty=ty,
                compose_components=components
            )
            return evaluator._append_expr(expr, span, evaluator.expression_kind_tracker.type_of_with_expr(expr))
        
        case _:
            raise ValueError("TypeNotConstructible")
