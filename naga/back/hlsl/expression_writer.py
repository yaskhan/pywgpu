"""HLSL Expression Writer

Comprehensive expression writing for HLSL backend.
Handles all NAGA IR expression types and converts them to valid HLSL syntax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from ...ir import (
    Expression, ExpressionType, Literal, LiteralType,
    BinaryOperator, UnaryOperator, MathFunction
)
from ...error import ShaderError

if TYPE_CHECKING:
    from ...arena import Handle, Arena
    from ...ir.module import Module


class HLSLExpressionWriter:
    """Writes NAGA IR expressions as HLSL code."""
    
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
        """Write an expression to HLSL string.
        
        Args:
            expr_handle: Handle to the expression
            
        Returns:
            HLSL code for the expression
        """
        expr = self.expressions[expr_handle]
        
        match expr.type:
            case ExpressionType.LITERAL:
                return self._write_literal(expr.literal)
            
            case ExpressionType.CONSTANT:
                const = self.module.constants[expr.constant]
                return self.names.get(f"const_{expr.constant}", const.name or f"const_{expr.constant}")
            
            case ExpressionType.ZERO_VALUE:
                return self._write_zero_value(expr.zero_value_ty)
            
            case ExpressionType.COMPOSE:
                return self._write_compose(expr)
            
            case ExpressionType.SPLAT:
                value = self.write_expression(expr.splat_value)
                return f"({self._type_name(expr.splat_type)})({value})"
            
            case ExpressionType.SWIZZLE:
                vector = self.write_expression(expr.swizzle_vector)
                pattern = self._swizzle_pattern(expr.swizzle_pattern)
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
                return pointer
            
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
                return self.names.get(f"call_{expr.call_result}", f"call_{expr.call_result}")
            
            case ExpressionType.IMAGE_SAMPLE:
                return self._write_image_sample(expr)
            
            case ExpressionType.IMAGE_LOAD:
                return self._write_image_load(expr)
            
            case ExpressionType.ATOMIC:
                return self._write_atomic(expr)
            
            case _:
                raise ShaderError(f"Unsupported HLSL expression: {expr.type}")
    
    def _write_literal(self, lit: Literal) -> str:
        """Write a literal value."""
        match lit.type:
            case LiteralType.F64:
                return f"{lit.f64}"
            case LiteralType.F32:
                # HLSL f32 literal
                res = str(lit.f32)
                if "." not in res and "e" not in res:
                    res += ".0"
                return f"{res}f"
            case LiteralType.F16:
                return f"{lit.f16}h"
            case LiteralType.U32:
                return f"{lit.u32}u"
            case LiteralType.I32:
                return f"{lit.i32}"
            case LiteralType.BOOL:
                return "true" if lit.bool else "false"
            case _:
                return "0"
    
    def _write_zero_value(self, ty_handle: int) -> str:
        """Write a zero value for a type."""
        # HLSL can use (Type)0 for zero initialization
        return f"({self._type_name(ty_handle)})0"

    def _write_compose(self, expr: Expression) -> str:
        """Write a compose expression."""
        # HLSL uses type constructors
        ty_name = self._type_name(expr.compose_ty)
        components = [
            self.write_expression(comp)
            for comp in expr.compose_components
        ]
        # For matrices, HLSL uses row-major by default but Naga is column-major.
        # This might need careful handling later.
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
        """Write a select expression."""
        condition = self.write_expression(expr.select_condition)
        accept = self.write_expression(expr.select_accept)
        reject = self.write_expression(expr.select_reject)
        
        # HLSL ternary
        return f"({condition} ? {accept} : {reject})"
    
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

    def _write_image_sample(self, expr: Expression) -> str:
        """Write an image sampling expression."""
        image = self.write_expression(expr.image_sample_image)
        sampler = self.write_expression(expr.image_sample_sampler)
        coord = self.write_expression(expr.image_sample_coordinate)
        
        # HLSL: texture.Sample(sampler, coordinate)
        return f"{image}.Sample({sampler}, {coord})"

    def _write_image_load(self, expr: Expression) -> str:
        """Write an image load expression."""
        image = self.write_expression(expr.image_load_image)
        coord = self.write_expression(expr.image_load_coordinate)
        
        # HLSL: texture.Load(coordinate)
        # Load usually takes a location with potentially mip level.
        # NAGA coordinate for Load is usually integer.
        return f"{image}.Load({coord})"

    def _write_atomic(self, expr: Expression) -> str:
        """Write an atomic operation."""
        raise ShaderError("Atomic operations must be emitted as statements in HLSL")
    
    def _math_function_name(self, func: MathFunction) -> str:
        """Get HLSL name for math function."""
        func_map = {
            MathFunction.ABS: "abs",
            MathFunction.MIN: "min",
            MathFunction.MAX: "max",
            MathFunction.CLAMP: "clamp",
            MathFunction.COS: "cos",
            MathFunction.SIN: "sin",
            MathFunction.TAN: "tan",
            MathFunction.ACOS: "acos",
            MathFunction.ASIN: "asin",
            MathFunction.ATAN: "atan",
            MathFunction.ATAN2: "atan2",
            MathFunction.EXP: "exp",
            MathFunction.EXP2: "exp2",
            MathFunction.LOG: "log",
            MathFunction.LOG2: "log2",
            MathFunction.POW: "pow",
            MathFunction.DOT: "dot",
            MathFunction.CROSS: "cross",
            MathFunction.DISTANCE: "distance",
            MathFunction.LENGTH: "length",
            MathFunction.NORMALIZE: "normalize",
            MathFunction.FACE_FORWARD: "faceforward",
            MathFunction.REFLECT: "reflect",
            MathFunction.REFRACT: "refract",
            MathFunction.SIGN: "sign",
            MathFunction.MIX: "lerp", # HLSL uses lerp instead of mix
            MathFunction.STEP: "step",
            MathFunction.SMOOTH_STEP: "smoothstep",
            MathFunction.SQRT: "sqrt",
            MathFunction.INVERSE_SQRT: "rsqrt",
            MathFunction.TRANSPOSE: "transpose",
            MathFunction.DETERMINANT: "determinant",
            MathFunction.INVERSE: "inverse", # Some versions of HLSL might not have inverse()
        }
        return func_map.get(func, str(func).lower())
    
    def _relational_function_name(self, func: Any) -> str:
        """Get HLSL name for relational function."""
        func_map = {
            "All": "all",
            "Any": "any",
            "IsNan": "isnan",
            "IsInf": "isinf",
        }
        return func_map.get(str(func), str(func).lower())
    
    def _type_name(self, ty_handle: int | Any) -> str:
        """Get HLSL type name."""
        if isinstance(ty_handle, int):
            item = self.module.types[ty_handle]
            if item.name:
                return item.name
            return f"Type{ty_handle}"
        return str(ty_handle).lower()

    def _swizzle_pattern(self, pattern: list) -> str:
        """Generate swizzle pattern string."""
        component_map = {0: "x", 1: "y", 2: "z", 3: "w"}
        return "".join(component_map.get(c, "x") for c in pattern)


__all__ = ['HLSLExpressionWriter']
