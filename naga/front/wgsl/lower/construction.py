"""
Constructor expression handling for WGSL.

Translated from wgpu-trunk/naga/src/front/wgsl/lower/construction.rs

This module handles type construction expressions like vec3(1.0, 2.0, 3.0).
"""

from typing import Any, List, Optional, Tuple
from ....ir import Expression, Scalar, VectorSize


class ConstructorHandler:
    """
    Handles constructor expressions in WGSL.
    
    Constructor expressions create values of various types:
    - Scalars: i32(5), f32(1.5)
    - Vectors: vec3(1.0, 2.0, 3.0), vec3<f32>(1.0)
    - Matrices: mat3x3<f32>(...)
    - Arrays: array<i32, 3>(1, 2, 3)
    - Structs: MyStruct(field1, field2)
    """
    
    def __init__(self, lowerer: Any):
        """
        Initialize constructor handler.
        
        Args:
            lowerer: The lowerer instance
        """
        self.lowerer = lowerer
        self.module = lowerer.module
    
    def _vector_size_to_int(self, size: VectorSize) -> int:
        """Convert VectorSize to integer."""
        from ....ir import VectorSize
        size_map = {
            VectorSize.BI: 2,
            VectorSize.TRI: 3,
            VectorSize.QUAD: 4,
        }
        return size_map.get(size, 4)
    
    def handle_constructor(
        self,
        constructor_handle: Any,
        arguments: List[Any],
        ctx: Any
    ) -> Any:
        """
        Handle a constructor expression.
        """
        from ....ir import Expression as IRExpression, TypeInnerType
        
        if constructor_handle is None:
            # Type inference needed for partial types (vec3 vs vec3<f32>)
            return None
        
        # Look up type info
        ty = self.module.types[constructor_handle]
        inner = ty.inner
        
        if inner.type == TypeInnerType.SCALAR:
            return self.construct_scalar(inner.scalar, arguments[0], ctx)
        elif inner.type == TypeInnerType.VECTOR:
            return self.construct_vector(inner.vector.size, inner.vector.scalar, arguments, ctx, type_handle=constructor_handle)
        elif inner.type == TypeInnerType.MATRIX:
            return self.construct_matrix(inner.matrix.columns, inner.matrix.rows, inner.matrix.scalar, arguments, ctx, type_handle=constructor_handle)
        elif inner.type == TypeInnerType.ARRAY:
            return self.construct_array(constructor_handle, inner.array.size, arguments, ctx)
        elif inner.type == TypeInnerType.STRUCT:
            return self.construct_struct(constructor_handle, arguments, ctx)

        # Fallback to compose
        expr = IRExpression(
            type=None,
            compose_ty=constructor_handle,
            compose_components=arguments
        )
        from ....ir import ExpressionType
        object.__setattr__(expr, 'type', ExpressionType.COMPOSE)
        return ctx.add_expression(expr)
    
    def construct_scalar(
        self,
        scalar: Scalar,
        argument: Any,
        ctx: Any
    ) -> Any:
        """
        Construct a scalar value (type cast).
        """
        from ....ir import Expression as IRExpression, ExpressionType
        
        expr = IRExpression(
            type=ExpressionType.AS,
            as_expr=argument,
            as_kind=scalar.kind,
            as_convert=None
        )
        return ctx.add_expression(expr)
    
    def construct_vector(
        self,
        size: VectorSize,
        scalar: Scalar,
        arguments: List[Any],
        ctx: Any,
        type_handle: Any = None
    ) -> Any:
        """
        Construct a vector value.
        """
        from ....ir import Expression as IRExpression, ExpressionType, TypeInnerType
        
        # Check for splat constructor
        # Splat is only for single scalar argument
        if len(arguments) == 1:
            arg_inner = self.lowerer._resolve_type(arguments[0], ctx)
            if arg_inner.type == TypeInnerType.SCALAR:
                return self.create_splat(arguments[0], size, ctx)
        
        # Extract components
        components = self.extract_components(arguments, self._vector_size_to_int(size), ctx)
        
        # Create compose expression
        expr = IRExpression(
            type=ExpressionType.COMPOSE,
            compose_ty=type_handle,
            compose_components=components
        )
        return ctx.add_expression(expr)
    
    def construct_matrix(
        self,
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
        arguments: List[Any],
        ctx: Any,
        type_handle: Any = None
    ) -> Any:
        """
        Construct a matrix value.
        """
        from ....ir import Expression as IRExpression, ExpressionType
        
        # Extract components (columns)
        col_count = self._vector_size_to_int(columns)
        components = self.extract_components(arguments, col_count, ctx)
        
        # Create compose expression
        expr = IRExpression(
            type=ExpressionType.COMPOSE,
            compose_ty=type_handle,
            compose_components=components
        )
        return ctx.add_expression(expr)

    def construct_array(
        self,
        array_type: Any,
        size: Any,
        arguments: List[Any],
        ctx: Any
    ) -> Any:
        """
        Construct an array value.
        """
        from ....ir import Expression as IRExpression, ExpressionType
        
        expr = IRExpression(
            type=ExpressionType.COMPOSE,
            compose_ty=array_type,
            compose_components=arguments
        )
        return ctx.add_expression(expr)
    
    def construct_struct(
        self,
        struct_type: Any,
        arguments: List[Any],
        ctx: Any
    ) -> Any:
        """
        Construct a struct value.
        """
        from ....ir import Expression as IRExpression, ExpressionType
        
        expr = IRExpression(
            type=ExpressionType.COMPOSE,
            compose_ty=struct_type,
            compose_components=arguments
        )
        return ctx.add_expression(expr)
    
    def extract_components(
        self,
        arguments: List[Any],
        expected_count: int,
        ctx: Any
    ) -> List[Any]:
        """
        Extract individual components from constructor arguments.
        """
        from ....ir import Expression as IRExpression, ExpressionType, TypeInnerType
        components = []
        
        for arg in arguments:
            arg_inner = self.lowerer._resolve_type(arg, ctx)
            
            if arg_inner.type == TypeInnerType.VECTOR:
                # Extract all components of the vector
                size = self._vector_size_to_int(arg_inner.vector.size)
                for i in range(size):
                    expr = IRExpression(
                        type=ExpressionType.ACCESS_INDEX,
                        access_base=arg,
                        access_index_value=i
                    )
                    components.append(ctx.add_expression(expr))
            else:
                components.append(arg)
        
        return components
    
    def create_splat(
        self,
        value: Any,
        size: VectorSize,
        ctx: Any
    ) -> Any:
        """
        Create a splat expression.
        """
        from ....ir import Expression as IRExpression, ExpressionType
        
        expr = IRExpression(
            type=ExpressionType.SPLAT,
            splat_size=size,
            splat_value=value
        )
        return ctx.add_expression(expr)
