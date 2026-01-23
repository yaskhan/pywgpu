"""
Type resolution for NAGA IR expressions during lowering.
"""

from typing import Any, Optional
from ....ir import ExpressionType, TypeInner, TypeInnerType, Scalar, Vector, Matrix, ScalarKind


def add_type_resolver_methods(lowerer_class):
    """Add type resolution helper methods to Lowerer class."""
    
    def _resolve_type(self, handle: int, ctx: Any) -> TypeInner:
        expr = ctx.function.expressions[handle]
        
        if expr.type == ExpressionType.LITERAL:
            # Literal type depends on the value
            val = expr.literal
            if isinstance(val, bool):
                return TypeInner.new_scalar(Scalar(kind=ScalarKind.BOOL, width=1))
            elif isinstance(val, int):
                return TypeInner.new_scalar(Scalar(kind=ScalarKind.SINT, width=4))
            elif isinstance(val, float):
                return TypeInner.new_scalar(Scalar(kind=ScalarKind.FLOAT, width=4))
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.FLOAT, width=4))
            
        elif expr.type == ExpressionType.CONSTANT:
            const = self.module.constants[expr.constant]
            return self.module.types[const.ty].inner
            
        elif expr.type == ExpressionType.GLOBAL_VARIABLE:
            var = self.module.global_variables[expr.global_variable]
            return self.module.types[var.ty].inner
            
        elif expr.type == ExpressionType.LOCAL_VARIABLE:
            var = ctx.function.local_variables[expr.local_variable]
            return self.module.types[var.ty].inner
            
        elif expr.type == ExpressionType.FUNCTION_ARGUMENT:
            arg = ctx.function.arguments[expr.function_argument]
            return self.module.types[arg.ty].inner
            
        elif expr.type == ExpressionType.ACCESS_INDEX:
            base_inner = self._resolve_type(expr.access_base, ctx)
            if expr.access_index_value is not None:
                index = expr.access_index_value
                if base_inner.type == TypeInnerType.STRUCT:
                    member_ty = base_inner.struct.members[index].ty
                    return self.module.types[member_ty].inner
                elif base_inner.type == TypeInnerType.VECTOR:
                    return TypeInner.new_scalar(base_inner.vector.scalar)
                elif base_inner.type == TypeInnerType.MATRIX:
                    return TypeInner.new_vector(base_inner.matrix.rows, base_inner.matrix.scalar)
                elif base_inner.type == TypeInnerType.ARRAY:
                    return self.module.types[base_inner.array.base].inner
            return base_inner # Fallback
            
        elif expr.type == ExpressionType.COMPOSE:
            return self.module.types[expr.compose_ty].inner
            
        elif expr.type == ExpressionType.SPLAT:
            # Type is vector of the splat value type
            val_inner = self._resolve_type(expr.splat_value, ctx)
            if val_inner.type == TypeInnerType.SCALAR:
                return TypeInner.new_vector(expr.splat_size, val_inner.scalar)
            return val_inner
            
        elif expr.type == ExpressionType.LOAD:
            ptr_inner = self._resolve_type(expr.load_pointer, ctx)
            if ptr_inner.type == TypeInnerType.POINTER:
                return self.module.types[ptr_inner.pointer.base].inner
            elif ptr_inner.type == TypeInnerType.VALUE_POINTER:
                if ptr_inner.value_pointer.size:
                    return TypeInner.new_vector(ptr_inner.value_pointer.size, ptr_inner.value_pointer.scalar)
                else:
                    return TypeInner.new_scalar(ptr_inner.value_pointer.scalar)
            return ptr_inner

        elif expr.type == ExpressionType.ARRAY_LENGTH:
            # arrayLength returns u32
            return TypeInner.new_scalar(Scalar(kind=ScalarKind.UINT, width=4))

        elif expr.type == ExpressionType.ATOMIC_RESULT:
            return self.module.types[expr.atomic_result_ty].inner
            
        # Add more cases as needed for built-ins, etc.
        return TypeInner.new_scalar(Scalar(kind=ScalarKind.FLOAT, width=4)) # Defaulting to f32 for now

    lowerer_class._resolve_type = _resolve_type
    return lowerer_class
