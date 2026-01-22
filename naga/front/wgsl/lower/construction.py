"""
Constructor expression handling for WGSL.

Translated from wgpu-trunk/naga/src/front/wgsl/lower/construction.rs

This module handles type construction expressions like vec3(1.0, 2.0, 3.0).
"""

from typing import Any, List, Optional, Tuple
from ...ir import Expression, Scalar, VectorSize


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
    
    def __init__(self, module: Any):
        """
        Initialize constructor handler.
        
        Args:
            module: The module being built
        """
        self.module = module
    
    def _vector_size_to_int(self, size: VectorSize) -> int:
        """Convert VectorSize to integer."""
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
        
        Args:
            constructor_handle: Handle to the type being constructed
            arguments: Constructor arguments (expression handles)
            ctx: Expression context
            
        Returns:
            Expression handle for the constructed value
        """
        from ...ir import Expression as IRExpression, TypeInner
        
        if constructor_handle is None:
            # Type inference needed for partial types (vec3 vs vec3<f32>)
            # For now, we assume constructor_handle is already resolved
            return None
        
        # Look up type info
        ty = self.module.types[constructor_handle]
        inner = ty.inner
        
        if isinstance(inner, TypeInner.SCALAR):
            return self.construct_scalar(inner.value, arguments[0], ctx)
        elif isinstance(inner, TypeInner.VECTOR):
            return self.construct_vector(inner.size, inner.scalar, arguments, ctx, type_handle=constructor_handle)
        elif isinstance(inner, TypeInner.MATRIX):
            return self.construct_matrix(inner.columns, inner.rows, inner.scalar, arguments, ctx, type_handle=constructor_handle)
        elif isinstance(inner, TypeInner.ARRAY):
            return self.construct_array(constructor_handle, inner.size, arguments, ctx)
        elif isinstance(inner, TypeInner.STRUCT):
            return self.construct_struct(constructor_handle, arguments, ctx)

        
        # Fallback to compose
        expr = IRExpression.COMPOSE(ty=constructor_handle, components=arguments)
        return ctx.add_expression(expr)
    
    def construct_scalar(
        self,
        scalar: Scalar,
        argument: Any,
        ctx: Any
    ) -> Any:
        """
        Construct a scalar value (type cast).
        
        Args:
            scalar: Target scalar type
            argument: Source expression
            ctx: Expression context
            
        Returns:
            Expression handle for cast
        """
        from ...ir import Expression as IRExpression
        
        # Create type cast expression
        expr = IRExpression.AS(expr=argument, kind=scalar.kind, convert=None)
        
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
        from ...ir import Expression as IRExpression
        
        # Check for splat constructor
        if self.is_splat_constructor(arguments):
            return self.create_splat(arguments[0], size, ctx)
        
        # Extract components
        components = self.extract_components(arguments, self._vector_size_to_int(size), ctx)
        
        # Create compose expression
        expr = IRExpression.COMPOSE(ty=type_handle, components=components)
        
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
        from ...ir import Expression as IRExpression
        
        # Extract components (columns or individual elements)
        col_count = self._vector_size_to_int(columns)
        row_count = self._vector_size_to_int(rows)
        total_components = col_count * row_count
        
        components = self.extract_components(arguments, total_components, ctx)
        
        # Create compose expression
        expr = IRExpression.COMPOSE(ty=type_handle, components=components)
        
        return ctx.add_expression(expr)

    
    def construct_array(
        self,
        base_type: Any,
        size: Any,
        arguments: List[Any],
        ctx: Any
    ) -> Any:
        """
        Construct an array value.
        
        Args:
            base_type: Array element type
            size: Array size
            arguments: Element expressions
            ctx: Expression context
            
        Returns:
            Expression handle for array
        """
        from ...ir import Expression as IRExpression
        
        # Validate argument count
        # TODO: Check size matches len(arguments)
        
        # Create compose expression
        expr = IRExpression.COMPOSE(ty=base_type, components=arguments)
        
        return ctx.add_expression(expr)
    
    def construct_struct(
        self,
        struct_type: Any,
        arguments: List[Any],
        ctx: Any
    ) -> Any:
        """
        Construct a struct value.
        
        Args:
            struct_type: Struct type handle
            arguments: Field expressions
            ctx: Expression context
            
        Returns:
            Expression handle for struct
        """
        from ...ir import Expression as IRExpression
        
        # Validate argument count matches struct fields
        # TODO: Get field count from struct_type
        
        # Create compose expression
        expr = IRExpression.COMPOSE(ty=struct_type, components=arguments)
        
        return ctx.add_expression(expr)
    
    def infer_constructor_type(
        self,
        partial_type: Any,
        arguments: List[Any],
        ctx: Any
    ) -> Any:
        """
        Infer the complete type for a partial constructor.
        """
        # Look at first argument to infer component type
        if arguments:
            # We need a way to get the type of an expression handle
            # This would usually involve looking up the expression in ctx.expression_arena
            # and checking its resolved type.
            # For now, we assume float32 if we can't determine it easily.
            return partial_type
        
        return partial_type
    
    def extract_components(
        self,
        arguments: List[Any],
        expected_count: int,
        ctx: Any
    ) -> List[Any]:
        """
        Extract individual components from constructor arguments.
        
        Handles cases like:
        - vec4(vec2(1, 2), 3, 4) -> [1, 2, 3, 4]
        """
        components = []
        
        for arg in arguments:
            # In NAGA logic, if an argument is a vector, we should emit 
            # ACCESS_INDEX expressions for each of its components.
            # This requires knowing the type of 'arg'.
            
            # TODO: Add logic to check type of 'arg' and flatten if it's a composite
            components.append(arg)
        
        # Validate component count
        # In WGSL, splat constructors have 1 argument but can fill multiple components
        if len(components) != expected_count and len(arguments) != 1:
            self.validate_component_count(len(components), expected_count, (0, 0))
        
        return components
    
    def validate_component_count(
        self,
        actual: int,
        expected: int,
        span: Tuple[int, int]
    ) -> None:
        """
        Validate that component count matches expected.
        
        Args:
            actual: Actual component count
            expected: Expected component count
            span: Source location for error
            
        Raises:
            ParseError: If counts don't match
        """
        if actual != expected:
            from ..error import ParseError
            raise ParseError(
                message=f"wrong number of components: expected {expected}, got {actual}",
                labels=[(span[0], span[1], "")],
                notes=[]
            )
    
    def is_splat_constructor(self, arguments: List[Any]) -> bool:
        """
        Check if this is a splat constructor (single argument).
        
        Args:
            arguments: Constructor arguments
            
        Returns:
            True if this is a splat
        """
        return len(arguments) == 1
    
    def create_splat(
        self,
        value: Any,
        size: VectorSize,
        ctx: Any
    ) -> Any:
        """
        Create a splat expression (repeat value across vector).
        
        Args:
            value: Value to splat
            size: Vector size
            ctx: Expression context
            
        Returns:
            Expression handle for splat
        """
        from ...ir import Expression as IRExpression
        
        # Create splat expression
        expr = IRExpression.SPLAT(size=size, value=value)
        
        return ctx.add_expression(expr)
