"""WGSL Expression Writer

Comprehensive expression writing for WGSL backend.
Handles all NAGA IR expression types and converts them to valid WGSL syntax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...ir import (
    Expression, ExpressionType, Literal, LiteralType,
    BinaryOperator, UnaryOperator, MathFunction
)

if TYPE_CHECKING:
    from ...arena import Handle, Arena
    from ...ir.module import Module


class WGSLExpressionWriter:
    """Writes NAGA IR expressions as WGSL code."""
    
    def __init__(self, module: Module, names: dict[str, str], expressions: Arena[Expression]):
        """Initialize expression writer.
        
        Args:
            module: The module being written
            names: Name mapping for identifiers
            expressions: The expression arena for the current function
        """
        self.module = module
        self.names = names
        self.expressions = expressions
    
    def write_expression(self, expr_handle: Handle[Expression]) -> str:
        """Write an expression to WGSL string.
        
        Args:
            expr_handle: Handle to the expression
            
        Returns:
            WGSL code for the expression
        """
        expr = self.expressions[expr_handle]
        
        match expr.type:
            case ExpressionType.LITERAL:
                return self._write_literal(expr.literal)
            
            case ExpressionType.CONSTANT:
                const = self.module.constants[expr.constant]
                return self.names.get(f"const_{expr.constant}", const.name or f"const_{expr.constant}")
            
            case ExpressionType.ZERO_VALUE:
                ty_name = self._type_name(expr.zero_value_ty)
                return f"{ty_name}()"
            
            case ExpressionType.COMPOSE:
                return self._write_compose(expr, expressions)
            
            case ExpressionType.SPLAT:
                size = expr.splat_size
                value = self.write_expression(expr.splat_value)
                # vec3<f32>(value) for splat
                return f"vec{self._vector_size_to_int(size)}({value})"
            
            case ExpressionType.SWIZZLE:
                vector = self.write_expression(expr.swizzle_vector)
                pattern = self._swizzle_pattern(expr.swizzle_pattern, expr.swizzle_size)
                return f"{vector}.{pattern}"
            
            case ExpressionType.ACCESS:
                base = self.write_expression(expr.access_base)
                index = self.write_expression(expr.access_index)
                return f"{base}[{index}]"
            
            case ExpressionType.ACCESS_INDEX:
                base = self.write_expression(expr.access_base)
                index = expr.access_index
                return f"{base}[{index}]"
            
            case ExpressionType.GLOBAL_VARIABLE:
                var = self.module.global_variables[expr.global_variable]
                return self.names.get(f"global_{expr.global_variable}", var.name or f"global_{expr.global_variable}")
            
            case ExpressionType.LOCAL_VARIABLE:
                return self.names.get(f"local_{expr.local_variable}", f"local_{expr.local_variable}")
            
            case ExpressionType.FUNCTION_ARGUMENT:
                return self.names.get(f"arg_{expr.function_argument}", f"arg_{expr.function_argument}")
            
            case ExpressionType.LOAD:
                pointer = self.write_expression(expr.load_pointer)
                return f"(*{pointer})"  # Dereference in WGSL
            
            case ExpressionType.UNARY:
                return self._write_unary(expr)
            
            case ExpressionType.BINARY:
                return self._write_binary(expr)
            
            case ExpressionType.SELECT:
                return self._write_select(expr)
            
            case ExpressionType.MATH:
                return self._write_math(expr)
            
            case ExpressionType.RELATIONAL:
                return self._write_relational(expr)
            
            case ExpressionType.CALL_RESULT:
                # Function call result - just reference it
                return self.names.get(f"call_{expr.call_result}", f"call_{expr.call_result}")
            
            case _:
                return f"/* TODO: {expr.type} */"
    
    def _write_literal(self, lit: Literal) -> str:
        """Write a literal value."""
        match lit.type:
            case LiteralType.F64:
                return f"{lit.f64}"
            case LiteralType.F32:
                return f"{lit.f32}f"
            case LiteralType.F16:
                return f"{lit.f16}h"
            case LiteralType.U32:
                return f"{lit.u32}u"
            case LiteralType.I32:
                return f"{lit.i32}"
            case LiteralType.U64:
                return f"{lit.u64}ul"
            case LiteralType.I64:
                return f"{lit.i64}l"
            case LiteralType.BOOL:
                return "true" if lit.bool else "false"
            case LiteralType.ABSTRACT_INT:
                return f"{lit.abstract_int}"
            case LiteralType.ABSTRACT_FLOAT:
                return f"{lit.abstract_float}"
            case _:
                return "0"
    
    def _write_compose(self, expr: Expression) -> str:
        """Write a compose expression."""
        ty_name = self._type_name(expr.compose_ty)
        components = [
            self.write_expression(comp)
            for comp in expr.compose_components
        ]
        return f"{ty_name}({', '.join(components)})"
    
    def _write_unary(self, expr: Expression) -> str:
        """Write a unary operation."""
        operand = self.write_expression(expr.unary_expr)
        
        match expr.unary_op:
            case UnaryOperator.NEGATE:
                return f"(-{operand})"
            case UnaryOperator.LOGICAL_NOT:
                return f"(!{operand})"
            case UnaryOperator.BITWISE_NOT:
                return f"(~{operand})"
            case _:
                return f"(/* unary */ {operand})"
    
    def _write_binary(self, expr: Expression) -> str:
        """Write a binary operation."""
        left = self.write_expression(expr.binary_left)
        right = self.write_expression(expr.binary_right)
        
        op_map = {
            BinaryOperator.ADD: "+",
            BinaryOperator.SUBTRACT: "-",
            BinaryOperator.MULTIPLY: "*",
            BinaryOperator.DIVIDE: "/",
            BinaryOperator.MODULO: "%",
            BinaryOperator.EQUAL: "==",
            BinaryOperator.NOT_EQUAL: "!=",
            BinaryOperator.LESS: "<",
            BinaryOperator.LESS_EQUAL: "<=",
            BinaryOperator.GREATER: ">",
            BinaryOperator.GREATER_EQUAL: ">=",
            BinaryOperator.LOGICAL_AND: "&&",
            BinaryOperator.LOGICAL_OR: "||",
            BinaryOperator.AND: "&",
            BinaryOperator.INCLUSIVE_OR: "|",
            BinaryOperator.EXCLUSIVE_OR: "^",
            BinaryOperator.SHIFT_LEFT: "<<",
            BinaryOperator.SHIFT_RIGHT: ">>",
        }
        
        op = op_map.get(expr.binary_op, "?")
        return f"({left} {op} {right})"
    
    def _write_select(self, expr: Expression) -> str:
        """Write a select (ternary) expression."""
        condition = self.write_expression(expr.select_condition)
        accept = self.write_expression(expr.select_accept)
        reject = self.write_expression(expr.select_reject)
        
        # WGSL uses select(reject, accept, condition)
        return f"select({reject}, {accept}, {condition})"
    
    def _write_math(self, expr: Expression) -> str:
        """Write a math function call."""
        func_name = self._math_function_name(expr.math_fun)
        
        args = [self.write_expression(expr.math_arg)]
        
        if hasattr(expr, 'math_arg1') and expr.math_arg1 is not None:
            args.append(self.write_expression(expr.math_arg1))
        if hasattr(expr, 'math_arg2') and expr.math_arg2 is not None:
            args.append(self.write_expression(expr.math_arg2))
        if hasattr(expr, 'math_arg3') and expr.math_arg3 is not None:
            args.append(self.write_expression(expr.math_arg3))
        
        return f"{func_name}({', '.join(args)})"
    
    def _write_relational(self, expr: Expression) -> str:
        """Write a relational function call."""
        func_name = self._relational_function_name(expr.relational_fun)
        arg = self.write_expression(expr.relational_argument)
        return f"{func_name}({arg})"
    
    def _math_function_name(self, func: MathFunction) -> str:
        """Get WGSL name for math function."""
        # Most math functions have the same name in WGSL
        func_map = {
            MathFunction.ABS: "abs",
            MathFunction.MIN: "min",
            MathFunction.MAX: "max",
            MathFunction.CLAMP: "clamp",
            MathFunction.SATURATE: "saturate",
            MathFunction.COS: "cos",
            MathFunction.COSH: "cosh",
            MathFunction.SIN: "sin",
            MathFunction.SINH: "sinh",
            MathFunction.TAN: "tan",
            MathFunction.TANH: "tanh",
            MathFunction.ACOS: "acos",
            MathFunction.ASIN: "asin",
            MathFunction.ATAN: "atan",
            MathFunction.ATAN2: "atan2",
            MathFunction.ASINH: "asinh",
            MathFunction.ACOSH: "acosh",
            MathFunction.ATANH: "atanh",
            MathFunction.RADIANS: "radians",
            MathFunction.DEGREES: "degrees",
            MathFunction.CEIL: "ceil",
            MathFunction.FLOOR: "floor",
            MathFunction.ROUND: "round",
            MathFunction.FRACT: "fract",
            MathFunction.TRUNC: "trunc",
            MathFunction.MODF: "modf",
            MathFunction.FREXP: "frexp",
            MathFunction.LDEXP: "ldexp",
            MathFunction.EXP: "exp",
            MathFunction.EXP2: "exp2",
            MathFunction.LOG: "log",
            MathFunction.LOG2: "log2",
            MathFunction.POW: "pow",
            MathFunction.DOT: "dot",
            MathFunction.OUTER: "outerProduct",
            MathFunction.CROSS: "cross",
            MathFunction.DISTANCE: "distance",
            MathFunction.LENGTH: "length",
            MathFunction.NORMALIZE: "normalize",
            MathFunction.FACE_FORWARD: "faceForward",
            MathFunction.REFLECT: "reflect",
            MathFunction.REFRACT: "refract",
            MathFunction.SIGN: "sign",
            MathFunction.FMA: "fma",
            MathFunction.MIX: "mix",
            MathFunction.STEP: "step",
            MathFunction.SMOOTH_STEP: "smoothstep",
            MathFunction.SQRT: "sqrt",
            MathFunction.INVERSE_SQRT: "inverseSqrt",
            MathFunction.INVERSE: "inverse",
            MathFunction.TRANSPOSE: "transpose",
            MathFunction.DETERMINANT: "determinant",
            MathFunction.QUANTIZE_TO_F16: "quantizeToF16",
            MathFunction.COUNT_TRAILING_ZEROS: "countTrailingZeros",
            MathFunction.COUNT_LEADING_ZEROS: "countLeadingZeros",
            MathFunction.COUNT_ONE_BITS: "countOneBits",
            MathFunction.REVERSE_BITS: "reverseBits",
            MathFunction.EXTRACT_BITS: "extractBits",
            MathFunction.INSERT_BITS: "insertBits",
            MathFunction.FIRST_TRAILING_BIT: "firstTrailingBit",
            MathFunction.FIRST_LEADING_BIT: "firstLeadingBit",
        }
        return func_map.get(func, str(func).lower())
    
    def _relational_function_name(self, func: Any) -> str:
        """Get WGSL name for relational function."""
        func_map = {
            "All": "all",
            "Any": "any",
            "IsNan": "isNan",
            "IsInf": "isInf",
            "IsFinite": "isFinite",
            "IsNormal": "isNormal",
        }
        return func_map.get(str(func), str(func).lower())
    
    def _type_name(self, ty_handle: int) -> str:
        """Get WGSL type name."""
        if ty_handle >= len(self.module.types):
            return "unknown"
        
        ty = self.module.types[ty_handle]
        
        # Simplified type name generation
        if hasattr(ty, 'name') and ty.name:
            return ty.name
        
        return f"Type{ty_handle}"
    
    def _vector_size_to_int(self, size: Any) -> int:
        """Convert VectorSize to integer."""
        size_map = {
            "BI": 2,
            "TRI": 3,
            "QUAD": 4,
        }
        return size_map.get(str(size), 4)
    
    def _swizzle_pattern(self, pattern: list, size: Any) -> str:
        """Generate swizzle pattern string."""
        component_map = {0: "x", 1: "y", 2: "z", 3: "w"}
        size_int = self._vector_size_to_int(size)
        
        components = []
        for i in range(size_int):
            if i < len(pattern):
                comp_idx = pattern[i] if isinstance(pattern[i], int) else 0
                components.append(component_map.get(comp_idx, "x"))
            else:
                components.append("x")
        
        return "".join(components)


__all__ = ['WGSLExpressionWriter']
