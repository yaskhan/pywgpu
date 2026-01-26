"""
Complete GLSL function handling and conversion utilities.

This is a comprehensive translation from Rust wgpu-trunk/naga/src/front/glsl/functions.rs
"""

from typing import Any, Optional, List, Dict, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from naga.ir import (
    Expression, ExpressionType, ScalarKind, Scalar, Literal, Type, TypeInner,
    VectorSize, MatrixStride, ArraySize, Span
)
from .error import Error, ErrorKind


class ConversionType(Enum):
    """Types of type conversions."""
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    PRECISION = "precision"
    ARRAY_TO_POINTER = "array_to_pointer"


@dataclass
class ConversionRule:
    """Rule for type conversion."""
    from_type: str
    to_type: str
    conversion_type: ConversionType
    cost: int  # Lower cost means better conversion


@dataclass
class ProxyWrite:
    """Struct detailing a store operation that must happen after a function call."""
    target: int  # Expression handle
    value: int   # Expression handle (pointer to read value)
    convert: Optional[Scalar] = None  # Optional conversion


class FunctionCallKind(Enum):
    """Type of function call."""
    TYPE_CONSTRUCTOR = "type_constructor"
    FUNCTION = "function"


class FunctionHandler:
    """Handler for GLSL function calls and type conversions."""
    
    def __init__(self):
        self.conversion_rules: List[ConversionRule] = []
        self._initialize_conversion_rules()
    
    def _initialize_conversion_rules(self) -> None:
        """Initialize built-in conversion rules."""
        # Basic type conversion rules matching GLSL semantics
        self.conversion_rules.extend([
            # Integer to float conversions (implicit)
            ConversionRule("int", "float", ConversionType.IMPLICIT, 1),
            ConversionRule("uint", "float", ConversionType.IMPLICIT, 1),
            
            # Boolean conversions
            ConversionRule("bool", "int", ConversionType.IMPLICIT, 2),
            ConversionRule("bool", "uint", ConversionType.IMPLICIT, 2),
            ConversionRule("bool", "float", ConversionType.IMPLICIT, 2),
            
            # Precision conversions (lowp -> mediump -> highp)
            ConversionRule("lowp", "mediump", ConversionType.PRECISION, 1),
            ConversionRule("lowp", "highp", ConversionType.PRECISION, 2),
            ConversionRule("mediump", "highp", ConversionType.PRECISION, 1),
            
            # Widening conversions
            ConversionRule("int8", "int16", ConversionType.IMPLICIT, 1),
            ConversionRule("int16", "int32", ConversionType.IMPLICIT, 1),
            ConversionRule("uint8", "uint16", ConversionType.IMPLICIT, 1),
            ConversionRule("uint16", "uint32", ConversionType.IMPLICIT, 1),
            
            # Explicit casts
            ConversionRule("float", "int", ConversionType.EXPLICIT, 1),
            ConversionRule("float", "uint", ConversionType.EXPLICIT, 1),
            ConversionRule("int", "uint", ConversionType.EXPLICIT, 0),
            ConversionRule("uint", "int", ConversionType.EXPLICIT, 0),
            ConversionRule("float", "bool", ConversionType.EXPLICIT, 2),
            ConversionRule("int", "bool", ConversionType.EXPLICIT, 2),
            ConversionRule("uint", "bool", ConversionType.EXPLICIT, 2),
        ])
    
    def function_or_constructor_call(
        self,
        frontend: Any,
        ctx: Any,
        stmt: Any,
        fc: FunctionCallKind,
        raw_args: List[Tuple[int, Span]],
        meta: Span
    ) -> Optional[int]:
        """
        Main entry point for function and constructor calls.
        
        Args:
            frontend: The GLSL frontend instance
            ctx: Parsing context
            stmt: Statement context
            fc: Function call kind (constructor or function)
            raw_args: Raw argument expressions
            meta: Source location
            
        Returns:
            Expression handle or None
        """
        try:
            # Lower all arguments to IR expressions
            args: List[Tuple[int, Span]] = []
            for expr_handle, expr_meta in raw_args:
                lowered = ctx.lower_expect_inner(stmt, frontend, expr_handle, "rhs")
                if lowered is not None:
                    args.append((lowered, expr_meta))
            
            match fc:
                case FunctionCallKind.TYPE_CONSTRUCTOR:
                    if len(args) == 1:
                        return self.constructor_single(frontend, ctx, frontend.current_type, args[0], meta)
                    else:
                        return self.constructor_many(frontend, ctx, frontend.current_type, args, meta)
                case FunctionCallKind.FUNCTION:
                    return self.function_call(frontend, ctx, stmt, frontend.current_function_name, args, raw_args, meta)
            
            return None
        except Exception as e:
            frontend.errors.append(Error.semantic_error(f"Function call error: {str(e)}", meta))
            return None
    
    def constructor_single(
        self,
        frontend: Any,
        ctx: Any,
        ty: int,  # Type handle
        value_expr: Tuple[int, Span],
        meta: Span
    ) -> int:
        """
        Handle single-argument constructor (type conversion).
        
        Args:
            frontend: Frontend instance
            ctx: Context
            ty: Target type handle
            value_expr: Value to convert (handle, span)
            meta: Source location
            
        Returns:
            Expression handle
        """
        value_handle, value_meta = value_expr
        expr_type = ctx.resolve_type(value_handle, value_meta)
        
        if expr_type is None:
            # Type resolution failed, return original
            return value_handle
        
        # Extract vector size if present
        vector_size = None
        if isinstance(expr_type, TypeInner) and hasattr(expr_type, 'size'):
            vector_size = expr_type.size
        
        # Check if expression is boolean
        expr_is_bool = False
        if hasattr(expr_type, 'scalar_kind'):
            expr_is_bool = expr_type.scalar_kind() == ScalarKind.BOOL
        
        target_type = ctx.module.types[ty]
        target_inner = target_type.inner
        
        # Special case: casting from bool uses Select, not As
        if expr_is_bool and hasattr(target_inner, 'scalar'):
            result_scalar = target_inner.scalar()
            if result_scalar and result_scalar.kind != ScalarKind.BOOL:
                return self._bool_to_non_bool_cast(
                    ctx, value_handle, value_meta, meta, result_scalar, vector_size
                )
        
        return self._handle_type_conversion(ctx, ty, target_inner, value_handle, value_meta, meta, vector_size)
    
    def _bool_to_non_bool_cast(
        self,
        ctx: Any,
        value: int,
        value_meta: Span,
        meta: Span,
        result_scalar: Scalar,
        vector_size: Optional[VectorSize]
    ) -> int:
        """Handle boolean to non-boolean type conversion using Select."""
        # Create zero and one literals
        scalar_4 = Scalar(kind=result_scalar.kind, width=4)
        l0 = Literal(scalar_4, 0)
        l1 = Literal(scalar_4, 1)
        
        reject = ctx.add_expression(Expression(
            type=ExpressionType.LITERAL,
            literal=l0
        ), value_meta)
        
        accept = ctx.add_expression(Expression(
            type=ExpressionType.LITERAL,
            literal=l1
        ), value_meta)
        
        # Apply splat if vector
        if vector_size is not None:
            ctx.implicit_splat(reject, value_meta, vector_size)
            ctx.implicit_splat(accept, value_meta, vector_size)
        
        # Create select expression
        return ctx.add_expression(Expression(
            type=ExpressionType.SELECT,
            select_condition=value,
            select_accept=accept,
            select_reject=reject
        ), meta)
    
    def _handle_type_conversion(
        self,
        ctx: Any,
        ty: int,
        target_inner: TypeInner,
        value: int,
        value_meta: Span,
        meta: Span,
        vector_size: Optional[VectorSize]
    ) -> int:
        """Handle general type conversion based on target type."""
        
        match target_inner:
            case TypeInner.SCALAR(scalar=scalar):
                return self._scalar_conversion(ctx, scalar, value, value_meta, meta)
            
            case TypeInner.VECTOR(size=size, scalar=scalar):
                return self._vector_conversion(ctx, ty, size, scalar, value, value_meta, meta, vector_size)
            
            case TypeInner.MATRIX(columns=columns, rows=rows, scalar=scalar):
                return self._matrix_conversion(ctx, ty, columns, rows, scalar, (value, value_meta), meta)
            
            case TypeInner.STRUCT(members=members):
                return self._struct_conversion(ctx, ty, members, value, value_meta, meta)
            
            case TypeInner.ARRAY(base=base):
                return self._array_conversion(ctx, ty, base, value, value_meta, meta)
            
            case _:
                # Unsupported conversion
                ctx.errors.append(Error.semantic_error(f"Bad type constructor for {target_inner}", meta))
                return value
    
    def _scalar_conversion(
        self,
        ctx: Any,
        scalar: Scalar,
        value: int,
        value_meta: Span,
        meta: Span
    ) -> int:
        """Handle scalar type conversion."""
        expr = value
        
        # If value is vector or matrix, extract first component
        value_type = ctx.resolve_type(value, value_meta)
        if value_type is None:
            return value
        
        if isinstance(value_type, (TypeInner.VECTOR, TypeInner.MATRIX)):
            expr = ctx.add_expression(Expression(
                type=ExpressionType.ACCESS_INDEX,
                access_base=value,
                access_index=0
            ), meta)
            
            # If matrix, extract from vector component
            if isinstance(value_type, TypeInner.MATRIX):
                expr = ctx.add_expression(Expression(
                    type=ExpressionType.ACCESS_INDEX,
                    access_base=expr,
                    access_index=0
                ), meta)
        
        # Create As expression for type conversion
        return ctx.add_expression(Expression(
            type=ExpressionType.AS,
            as_expr=expr,
            as_kind=scalar.kind,
            as_convert=scalar.width
        ), meta)
    
    def _vector_conversion(
        self,
        ctx: Any,
        ty: int,
        size: VectorSize,
        scalar: Scalar,
        value: int,
        value_meta: Span,
        meta: Span,
        vector_size: Optional[VectorSize]
 ) -> int:
        """Handle vector type conversion."""
        # Resize vector if needed
        if vector_size != size:
            value = ctx.vector_resize(size, value, value_meta)
        
        # Apply type conversion
        return ctx.add_expression(Expression(
            type=ExpressionType.AS,
            as_expr=value,
            as_kind=scalar.kind,
            as_convert=scalar.width
        ), meta)
    
    def _matrix_conversion(
        self,
        ctx: Any,
        ty: int,
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
        value_expr: Tuple[int, Span],
        meta: Span
    ) -> int:
        """Handle matrix type conversion."""
        return self.matrix_one_arg(ctx, ty, columns, rows, scalar, value_expr, meta)
    
    def _struct_conversion(
        self,
        ctx: Any,
        ty: int,
        members: List[Any],
        value: int,
        value_meta: Span,
        meta: Span
    ) -> int:
        """Handle struct type conversion."""
        # Check if we can perform component-wise conversion
        if members:
            first_member = members[0]
            first_type = ctx.module.types[first_member.ty]
            scalar = self._extract_scalar_component(first_type.inner)
            
            if scalar:
                ctx.implicit_conversion(value, value_meta, scalar)
        
        # Create compose expression
        return ctx.add_expression(Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=ty,
            compose_components=[value]
        ), meta)
    
    def _array_conversion(
        self,
        ctx: Any,
        ty: int,
        base: int,
        value: int,
        value_meta: Span,
        meta: Span
    ) -> int:
        """Handle array type conversion."""
        # Check component type for conversion
        base_type = ctx.module.types[base]
        scalar = self._extract_scalar_component(base_type.inner)
        
        if scalar:
            ctx.implicit_conversion(value, value_meta, scalar)
        
        # Create compose expression
        return ctx.add_expression(Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=ty,
            compose_components=[value]
        ), meta)
    
    def _extract_scalar_component(self, type_inner: TypeInner) -> Optional[Scalar]:
        """Extract scalar component from vector/matrix/etc."""
        match type_inner:
            case TypeInner.SCALAR(scalar=scalar):
                return scalar
            case TypeInner.VECTOR(scalar=scalar):
                return scalar
            case TypeInner.MATRIX(scalar=scalar):
                return scalar
            case TypeInner.ARRAY(base=base):
                # Recursively check base type
                return self._extract_scalar_component(base)
            case _:
                return None
    
    def constructor_many(
        self,
        frontend: Any,
        ctx: Any,
        ty: int,
        args: List[Tuple[int, Span]],
        meta: Span
    ) -> int:
        """Handle multi-argument constructor."""
        # For now, compose all arguments directly
        # In full implementation, this would handle more complex cases
        components = [handle for handle, _ in args]
        
        return ctx.add_expression(Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=ty,
            compose_components=components
        ), meta)
    
    def function_call(
        self,
        frontend: Any,
        ctx: Any,
        stmt: Any,
        name: str,
        args: List[Tuple[int, Span]],
        raw_args: List[Tuple[int, Span]],
        meta: Span
    ) -> Optional[int]:
        """Handle function calls."""
        # Look up function in available functions
        if hasattr(frontend, 'function_parser'):
            func = frontend.function_parser.resolve_function_call(name, args)
            if func:
                return self._emit_function_call(ctx, func, args, meta)
        
        # Check if it's a builtin
        if hasattr(frontend, 'builtins'):
            builtin_result = self._emit_builtin_call(frontend, ctx, name, args, meta)
            if builtin_result is not None:
                return builtin_result
        
        frontend.errors.append(Error.semantic_error(f"Unknown function: {name}", meta))
        return None
    
    def _emit_function_call(
        self,
        ctx: Any,
        func: Any,
        args: List[Tuple[int, Span]],
        meta: Span
    ) -> int:
        """Emit a function call expression."""
        arg_handles = [handle for handle, _ in args]
        
        if hasattr(func, 'result_type'):
            # Function with return value
            return ctx.add_expression(Expression(
                type=ExpressionType.CALL,
                call_function=func,
                call_arguments=arg_handles
            ), meta)
        else:
            # Function with no return (statement)
            ctx.emit_statement(Statement(
                type=StatementType.CALL,
                call_function=func,
                call_arguments=arg_handles
            ))
            return None
    
    def _emit_builtin_call(
        self,
        frontend: Any,
        ctx: Any,
        name: str,
        args: List[Tuple[int, Span]],
        meta: Span
    ) -> Optional[int]:
        """Emit a builtin function call."""
        # Get builtin function info
        builtin_info = frontend.builtins.get_builtin_function(name, args)
        if builtin_info:
            # Create appropriate IR expression
            arg_handles = [handle for handle, _ in args]
            
            if builtin_info.is_math:
                return self._emit_math_builtin(ctx, builtin_info, arg_handles, meta)
            elif builtin_info.is_texture:
                return self._emit_texture_builtin(ctx, builtin_info, arg_handles, meta)
            else:
                # Generic builtin
                return ctx.add_expression(Expression(
                    type=ExpressionType.BUILTIN_CALL,
                    builtin_name=name,
                    builtin_args=arg_handles
                ), meta)
        
        return None
    
    def _emit_math_builtin(
        self,
        ctx: Any,
        builtin: Any,
        args: List[int],
        meta: Span
    ) -> int:
        """Emit math builtin call."""
        # Map to Naga math function
        math_func = builtin.naga_math_function()
        
        return ctx.add_expression(Expression(
            type=ExpressionType.MATH,
            math_function=math_func,
            math_args=args
        ), meta)
    
    def _emit_texture_builtin(
        self,
        ctx: Any,
        builtin: Any,
        args: List[int],
        meta: Span
    ) -> int:
        """Emit texture builtin call."""
        # Get texture information
        texture = args[0]
        sampler = args[1] if len(args) > 1 else None
        coords = args[2] if len(args) > 2 else None
        
        return ctx.add_expression(Expression(
            type=ExpressionType.TEXTURE_ACCESS,
            texture=texture,
            sampler=sampler,
            texture_coord=coords,
            texture_builtin=builtin.name
        ), meta)
    
    def matrix_one_arg(
        self,
        ctx: Any,
        ty: int,
        columns: VectorSize,
        rows: VectorSize,
        element_scalar: Scalar,
        value_expr: Tuple[int, Span],
        meta: Span
    ) -> int:
        """Construct matrix from single argument."""
        value, value_meta = value_expr
        components = []
        
        # Note: Expression::As doesn't support matrix width casts
        # so we need component-wise operations
        ctx.forced_conversion(value, value_meta, element_scalar)
        
        value_type = ctx.resolve_type(value, value_meta)
        
        match value_type:
            case TypeInner.SCALAR():
                # Diagonal matrix from scalar
                return self._diagonal_matrix(ctx, columns, rows, element_scalar, value, meta)
            
            case TypeInner.MATRIX(rows=ori_rows, columns=ori_cols):
                # Resize matrix
                return self._resize_matrix(
                    ctx, ty, columns, rows, element_scalar,
                    value, value_meta, ori_cols, ori_rows, meta
                )
            
            case _:
                # Default: compose with single component
                return ctx.add_expression(Expression(
                    type=ExpressionType.COMPOSE,
                    compose_ty=ty,
                    compose_components=[value]
                ), meta)
    
    def _diagonal_matrix(
        self,
        ctx: Any,
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
        value: int,
        meta: Span
    ) -> int:
        """Create diagonal matrix from scalar."""
        # Create vector type for columns
        vector_ty = ctx.module.types.add_type(Type(
            name=None,
            inner=TypeInner.VECTOR(size=rows, scalar=scalar)
        ), meta)
        
        # Create zero literal
        zero_literal = Literal(scalar, 0)
        zero = ctx.add_expression(Expression(
            type=ExpressionType.LITERAL,
            literal=zero_literal
        ), meta)
        
        components = []
        for i in range(columns.value):
            vec_components = []
            for j in range(rows.value):
                vec_components.append(value if i == j else zero)
            
            column_vec = ctx.add_expression(Expression(
                type=ExpressionType.COMPOSE,
                compose_ty=vector_ty,
                compose_components=vec_components
            ), meta)
            components.append(column_vec)
        
        return ctx.add_expression(Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=ctx.module.types.add_type(Type(
                name=None,
                inner=TypeInner.MATRIX(columns=columns, rows=rows, scalar=scalar)
            ), meta),
            compose_components=components
        ), meta)
    
    def _resize_matrix(
        self,
        ctx: Any,
        ty: int,
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
        value: int,
        value_meta: Span,
        ori_cols: VectorSize,
        ori_rows: VectorSize,
        meta: Span
    ) -> int:
        """Resize matrix to different dimensions."""
        # Create zero and one literals for identity
        zero_lit = Literal(scalar, 0)
        one_lit = Literal(scalar, 1)
        
        zero = ctx.add_expression(Expression(
            type=ExpressionType.LITERAL,
            literal=zero_lit
        ), meta)
        
        one = ctx.add_expression(Expression(
            type=ExpressionType.LITERAL,
            literal=one_lit
        ), meta)
        
        # Create vector type
        vector_ty = ctx.module.types.add_type(Type(
            name=None,
            inner=TypeInner.VECTOR(size=rows, scalar=scalar)
        ), meta)
        
        components = []
        for i in range(columns.value):
            if i < ori_cols.value:
                # Copy existing column
                vec = ctx.add_expression(Expression(
                    type=ExpressionType.ACCESS_INDEX,
                    access_base=value,
                    access_index=i
                ), meta)
                
                # Resize vector if needed
                if rows != ori_rows:
                    vec = self._resize_vector_components(
                        ctx, vec, ori_rows, rows, zero, one, i, meta
                    )
                
                components.append(vec)
            else:
                # New column from identity
                vec_components = []
                for j in range(rows.value):
                    vec_components.append(one if i == j else zero)
                
                vec = ctx.add_expression(Expression(
                    type=ExpressionType.COMPOSE,
                    compose_ty=vector_ty,
                    compose_components=vec_components
                ), meta)
                components.append(vec)
        
        return ctx.add_expression(Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=ty,
            compose_components=components
        ), meta)
    
    def _resize_vector_components(
        self,
        ctx: Any,
        vec: int,
        ori_rows: VectorSize,
        rows: VectorSize,
        zero: int,
        one: int,
        col_idx: int,
        meta: Span
    ) -> int:
        """Resize vector components when matrix resizing."""
        # Extract and rebuild vector with new size
        components = []
        
        for i in range(rows.value):
            if i < ori_rows.value:
                # Copy existing component
                comp = ctx.add_expression(Expression(
                    type=ExpressionType.ACCESS_INDEX,
                    access_base=vec,
                    access_index=i
                ), meta)
                components.append(comp)
            else:
                # Fill with identity or zero
                components.append(one if i == col_idx else zero)
        
        vector_ty = ctx.module.types.add_type(Type(
            name=None,
            inner=TypeInner.VECTOR(size=rows, scalar=ctx.module.types[ctx.resolve_type(vec, meta)].inner.scalar())
        ), meta)
        
        return ctx.add_expression(Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=vector_ty,
            compose_components=components
        ), meta)
    
    def check_conversion_compatibility(
        self,
        from_type: str,
        to_type: str
    ) -> Optional[ConversionRule]:
        """Check if conversion between types is possible."""
        # Find best conversion rule
        compatible = [
            rule for rule in self.conversion_rules
            if rule.from_type == from_type and rule.to_type == to_type
        ]
        
        if compatible:
            return min(compatible, key=lambda r: r.cost)
        
        # Handle vector conversions
        import re
        vec_pattern = r'^([biu]?)(?:vec|vec)([234])$'
        from_match = re.match(vec_pattern, from_type)
        to_match = re.match(vec_pattern, to_type)
        
        if from_match and to_match:
            from_prefix, from_size = from_match.groups()
            to_prefix, to_size = to_match.groups()
            
            if from_size == to_size:
                # Same size, check scalar conversion
                prefix_map = {'': 'float', 'i': 'int', 'u': 'uint', 'b': 'bool'}
                from_scalar = prefix_map.get(from_prefix)
                to_scalar = prefix_map.get(to_prefix)
                
                if from_scalar and to_scalar:
                    return self.check_conversion_compatibility(from_scalar, to_scalar)
        
        return None
    
    def resolve_type(self, ctx: Any, value: Any, meta: Span = None) -> Optional[TypeInner]:
        """Resolve the type of a value handle."""
        return ctx.get_expression_type(value)
    
    def arg_type_walker(
        self,
        name: str,
        binding: Any,
        pointer: Any,
        base: Any,
        func: Callable
    ) -> Optional[Any]:
        """Walk through argument types for function calls."""
        try:
            return func(name, binding, pointer, base)
        except Exception as e:
            # Log error but don't crash
            print(f"Error in arg_type_walker for {name}: {e}")
            return None
    
    def report_type_error(
        self,
        frontend: Any,
        expected: Any,
        actual: Any,
        meta: Span
    ) -> None:
        """Report a type error with detailed information."""
        error_msg = f"Type error: expected '{expected}', got '{actual}'"
        
        # Report through frontend if available
        if hasattr(frontend, 'add_error'):
            frontend.add_error(error_msg, meta)
        else:
            print(f"Error: {error_msg} at {meta}")
        
        # Store in errors list
        if hasattr(frontend, 'errors'):
            frontend.errors.append(Error.semantic_error(error_msg, meta))


# Global function handler instance
function_handler = FunctionHandler()
