"""GLSL expression writer.

This is a lightweight translation of parts of `wgpu-trunk/naga/src/back/glsl/writer.rs`.

The full GLSL backend is not yet completely ported; this writer provides the
expression variants required by the current simplified GLSL backend driver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...error import ShaderError
from ...ir import (
    BinaryOperator,
    Expression,
    ExpressionType,
    Literal,
    LiteralType,
    MathFunction,
    RelationalFunction,
    UnaryOperator,
)

if TYPE_CHECKING:
    from ...ir.module import Module


class GLSLExpressionWriter:
    """Writes Naga IR expressions as GLSL source."""

    def __init__(self, module: Module, names: dict[str, str], expressions: list[Expression]):
        self.module = module
        self.names = names
        self.expressions = expressions

    def write_expression(self, expr_handle: int) -> str:
        expr = self.expressions[expr_handle]

        match expr.type:
            case ExpressionType.LITERAL:
                if expr.literal is None:
                    raise ShaderError("Malformed Literal expression")
                return self._write_literal(expr.literal)

            case ExpressionType.CONSTANT:
                if expr.constant is None:
                    raise ShaderError("Malformed Constant expression")
                constant = self.module.constants[expr.constant]
                if constant.name is not None:
                    return self.names.get(f"const_{expr.constant}", constant.name)
                return self.write_expression(constant.init)

            case ExpressionType.ZERO_VALUE:
                if expr.zero_value is None:
                    raise ShaderError("Malformed ZeroValue expression")
                return f"{self._type_name(expr.zero_value)}(0)"

            case ExpressionType.COMPOSE:
                if expr.compose_ty is None or expr.compose_components is None:
                    raise ShaderError("Malformed Compose expression")
                args = ", ".join(self.write_expression(h) for h in expr.compose_components)
                return f"{self._type_name(expr.compose_ty)}({args})"

            case ExpressionType.SPLAT:
                if expr.splat_size is None or expr.splat_value is None:
                    raise ShaderError("Malformed Splat expression")
                size = getattr(expr.splat_size, "value", 4)
                return f"vec{size}({self.write_expression(expr.splat_value)})"

            case ExpressionType.SWIZZLE:
                if (
                    expr.swizzle_vector is None
                    or expr.swizzle_size is None
                    or expr.swizzle_pattern is None
                ):
                    raise ShaderError("Malformed Swizzle expression")
                vector = self.write_expression(expr.swizzle_vector)
                size = getattr(expr.swizzle_size, "value", 4)
                pattern = self._swizzle_pattern(expr.swizzle_pattern, size)
                return f"{vector}.{pattern}"

            case ExpressionType.ACCESS:
                if expr.access_base is None or expr.access_index is None:
                    raise ShaderError("Malformed Access expression")
                base = self.write_expression(expr.access_base)
                index = self.write_expression(expr.access_index)
                return f"{base}[{index}]"

            case ExpressionType.ACCESS_INDEX:
                if expr.access_base is None or expr.access_index_value is None:
                    raise ShaderError("Malformed AccessIndex expression")
                base = self.write_expression(expr.access_base)
                return f"{base}[{expr.access_index_value}]"

            case ExpressionType.FUNCTION_ARGUMENT:
                if expr.function_argument is None:
                    raise ShaderError("Malformed FunctionArgument expression")
                return self.names.get(f"arg_{expr.function_argument}", f"arg_{expr.function_argument}")

            case ExpressionType.GLOBAL_VARIABLE:
                if expr.global_variable is None:
                    raise ShaderError("Malformed GlobalVariable expression")
                var = self.module.global_variables[expr.global_variable]
                return self.names.get(
                    f"global_{expr.global_variable}",
                    getattr(var, "name", None) or f"global_{expr.global_variable}",
                )

            case ExpressionType.LOCAL_VARIABLE:
                if expr.local_variable is None:
                    raise ShaderError("Malformed LocalVariable expression")
                return self.names.get(f"local_{expr.local_variable}", f"local_{expr.local_variable}")

            case ExpressionType.LOAD:
                if expr.load_pointer is None:
                    raise ShaderError("Malformed Load expression")
                return self.write_expression(expr.load_pointer)

            case ExpressionType.UNARY:
                if expr.unary_op is None or expr.unary_expr is None:
                    raise ShaderError("Malformed Unary expression")
                operand = self.write_expression(expr.unary_expr)
                return self._write_unary(expr.unary_op, operand)

            case ExpressionType.BINARY:
                if expr.binary_op is None or expr.binary_left is None or expr.binary_right is None:
                    raise ShaderError("Malformed Binary expression")
                left = self.write_expression(expr.binary_left)
                right = self.write_expression(expr.binary_right)
                return self._write_binary(expr.binary_op, left, right)

            case ExpressionType.SELECT:
                if (
                    expr.select_condition is None
                    or expr.select_accept is None
                    or expr.select_reject is None
                ):
                    raise ShaderError("Malformed Select expression")
                condition = self.write_expression(expr.select_condition)
                accept = self.write_expression(expr.select_accept)
                reject = self.write_expression(expr.select_reject)
                return f"({condition} ? {accept} : {reject})"

            case ExpressionType.RELATIONAL:
                if expr.relational_fun is None or expr.relational_argument is None:
                    raise ShaderError("Malformed Relational expression")
                fun = self._relational_function_name(expr.relational_fun)
                arg = self.write_expression(expr.relational_argument)
                return f"{fun}({arg})"

            case ExpressionType.MATH:
                if expr.math_fun is None or expr.math_arg is None:
                    raise ShaderError("Malformed Math expression")
                fun = self._math_function_name(expr.math_fun)
                args = [self.write_expression(expr.math_arg)]
                for extra in (expr.math_arg1, expr.math_arg2, expr.math_arg3):
                    if extra is not None:
                        args.append(self.write_expression(extra))
                return f"{fun}({', '.join(args)})"

            case ExpressionType.CALL_RESULT:
                if expr.call_result is None:
                    raise ShaderError("Malformed CallResult expression")
                return self.names.get(f"call_{expr.call_result}", f"call_{expr.call_result}")

            case ExpressionType.ATOMIC_RESULT | ExpressionType.WORKGROUP_UNIFORM_LOAD_RESULT:
                return f"_e{expr_handle}"

            case ExpressionType.IMAGE_SAMPLE:
                if (
                    expr.image_sample_image is None
                    or expr.image_sample_sampler is None
                    or expr.image_sample_coordinate is None
                ):
                    raise ShaderError("Malformed ImageSample expression")
                image = self.write_expression(expr.image_sample_image)
                sampler = self.write_expression(expr.image_sample_sampler)
                coord = self.write_expression(expr.image_sample_coordinate)
                return f"texture({image}, {coord})" if sampler == "" else f"texture({image}, {coord})"

            case ExpressionType.IMAGE_LOAD:
                if expr.image_load_image is None or expr.image_load_coordinate is None:
                    raise ShaderError("Malformed ImageLoad expression")
                image = self.write_expression(expr.image_load_image)
                coord = self.write_expression(expr.image_load_coordinate)
                return f"imageLoad({image}, {coord})"

            case _:
                raise ShaderError(f"Unsupported GLSL expression: {expr.type}")

    def _write_literal(self, lit: Literal) -> str:
        value = lit.value
        match lit.type:
            case LiteralType.F32:
                text = str(value)
                if "." not in text and "e" not in text and "E" not in text:
                    text += ".0"
                return text
            case LiteralType.F16:
                return str(value)
            case LiteralType.F64:
                return str(value)
            case LiteralType.I32:
                return str(int(value))
            case LiteralType.U32:
                return f"{int(value)}u"
            case LiteralType.BOOL:
                return "true" if bool(value) else "false"
            case LiteralType.I64:
                return str(int(value))
            case LiteralType.U64:
                return str(int(value))
            case LiteralType.ABSTRACT_INT | LiteralType.ABSTRACT_FLOAT:
                raise ShaderError(
                    "Abstract types should not appear in IR presented to backends"
                )

        raise ShaderError(f"Unsupported literal type: {lit.type}")

    def _write_unary(self, op: UnaryOperator, operand: str) -> str:
        match op:
            case UnaryOperator.NEGATE:
                return f"(-{operand})"
            case UnaryOperator.LOGICAL_NOT:
                return f"(!{operand})"
            case UnaryOperator.BITWISE_NOT:
                return f"(~{operand})"
        raise ShaderError(f"Unsupported unary operator: {op}")

    def _write_binary(self, op: BinaryOperator, left: str, right: str) -> str:
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
            BinaryOperator.AND: "&",
            BinaryOperator.EXCLUSIVE_OR: "^",
            BinaryOperator.INCLUSIVE_OR: "|",
            BinaryOperator.LOGICAL_AND: "&&",
            BinaryOperator.LOGICAL_OR: "||",
            BinaryOperator.SHIFT_LEFT: "<<",
            BinaryOperator.SHIFT_RIGHT: ">>",
        }
        symbol = op_map.get(op)
        if symbol is None:
            raise ShaderError(f"Unsupported binary operator: {op}")
        return f"({left} {symbol} {right})"

    def _math_function_name(self, func: MathFunction) -> str:
        return str(func.value)

    def _relational_function_name(self, func: RelationalFunction) -> str:
        return str(func.value)

    def _swizzle_pattern(self, pattern: list[object], size: int) -> str:
        component_map = {0: "x", 1: "y", 2: "z", 3: "w"}
        out: list[str] = []
        for i in range(size):
            if i < len(pattern):
                component = pattern[i]
                index = component.value if hasattr(component, "value") else int(component)
                out.append(component_map.get(index, "x"))
            else:
                out.append("x")
        return "".join(out)

    def _type_name(self, ty_handle: int) -> str:
        if hasattr(self.module, "types") and 0 <= ty_handle < len(self.module.types):
            ty = self.module.types[ty_handle]
            if getattr(ty, "name", None):
                return str(ty.name)
        return self.names.get(f"type_{ty_handle}", f"Type{ty_handle}")


__all__ = ["GLSLExpressionWriter"]
