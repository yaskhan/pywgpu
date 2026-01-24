"""WGSL expression writer.

This module provides a small subset of the full `naga` WGSL backend expression
writer, translated from `wgpu-trunk/naga/src/back/wgsl/writer.rs`.

The project-wide WGSL backend is still under active porting; this writer focuses
on the expression variants exercised by the current Python IR and backends.
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


class WGSLExpressionWriter:
    """Writes Naga IR expressions as WGSL source."""

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

            case ExpressionType.OVERRIDE:
                if expr.override is None:
                    raise ShaderError("Malformed Override expression")
                return self.names.get(f"override_{expr.override}", f"override_{expr.override}")

            case ExpressionType.ZERO_VALUE:
                if expr.zero_value is None:
                    raise ShaderError("Malformed ZeroValue expression")
                return f"{self._type_name(expr.zero_value)}()"

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
                # WGSL loads are implicit for non-pointer-typed expressions in this simplified IR.
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
                return f"select({reject}, {accept}, {condition})"

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

            case ExpressionType.AS:
                if expr.as_expr is None or expr.as_kind is None:
                    raise ShaderError("Malformed As expression")
                value = self.write_expression(expr.as_expr)
                # WGSL casts are written as `T(value)`.
                return f"{str(expr.as_kind.value)}({value})"

            case ExpressionType.CALL_RESULT:
                if expr.call_result is None:
                    raise ShaderError("Malformed CallResult expression")
                return self.names.get(f"call_{expr.call_result}", f"call_{expr.call_result}")

            case ExpressionType.ATOMIC_RESULT | ExpressionType.WORKGROUP_UNIFORM_LOAD_RESULT:
                return f"_e{expr_handle}"

            case ExpressionType.ARRAY_LENGTH:
                if expr.array_length is None:
                    raise ShaderError("Malformed ArrayLength expression")
                base = self.write_expression(expr.array_length)
                return f"arrayLength(&{base})"

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
                return f"textureSample({image}, {sampler}, {coord})"

            case ExpressionType.IMAGE_LOAD:
                if expr.image_load_image is None or expr.image_load_coordinate is None:
                    raise ShaderError("Malformed ImageLoad expression")
                image = self.write_expression(expr.image_load_image)
                coord = self.write_expression(expr.image_load_coordinate)
                level = "0"
                if expr.image_load_level is not None:
                    level = self.write_expression(expr.image_load_level)
                return f"textureLoad({image}, {coord}, {level})"

            case _:
                raise ShaderError(f"Unsupported WGSL expression: {expr.type}")

    def _write_literal(self, lit: Literal) -> str:
        value = lit.value
        match lit.type:
            case LiteralType.F16:
                return f"{value}h"
            case LiteralType.F32:
                return f"{value}f"
            case LiteralType.U32:
                return f"{value}u"
            case LiteralType.I32:
                if int(value) == -(2**31):
                    return f"i32({value})"
                return f"{value}i"
            case LiteralType.BOOL:
                return "true" if bool(value) else "false"
            case LiteralType.F64:
                return f"{value}lf"
            case LiteralType.I64:
                if int(value) == -(2**63):
                    return f"i64({int(value) + 1} - 1)"
                return f"{value}li"
            case LiteralType.U64:
                return f"{value}lu"
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
        # Most function names match WGSL.
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
        # The full backend uses the type arena and the naming pass. Keep a
        # predictable fallback for now.
        if hasattr(self.module, "types") and 0 <= ty_handle < len(self.module.types):
            ty = self.module.types[ty_handle]
            if getattr(ty, "name", None):
                return str(ty.name)
        return self.names.get(f"type_{ty_handle}", f"Type{ty_handle}")


__all__ = ["WGSLExpressionWriter"]
