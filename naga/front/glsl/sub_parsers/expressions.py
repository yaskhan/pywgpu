"""
Parser for GLSL expressions.

This module handles parsing of GLSL expressions including operators, 
function calls, and literals.
"""

from typing import Any, Optional, List, Dict
from ..token import TokenValue

from typing import Any, Optional, List, Dict
from ..token import TokenValue
from ....ir import Expression, BinaryOperator, UnaryOperator, Literal, Scalar, ScalarKind

class ExpressionParser:
    """Parser for GLSL expressions that lowers them to NAGA IR handles."""
    
    def __init__(self, lexer: Any):
        self.lexer = lexer
        self.errors: List[str] = []

    def parse_expression(self, ctx: Any, frontend: Any, translation_ctx: Optional[Any] = None) -> Any:
        """Parse an expression (lowest precedence: sequence)."""
        return self.parse_assignment(ctx, frontend, translation_ctx)

    def parse_assignment(self, ctx: Any, frontend: Any, translation_ctx: Optional[Any] = None) -> int:
        """Parse assignment expressions."""
        left = self.parse_conditional(ctx, frontend, translation_ctx)
        token = ctx.peek(frontend)
        if token and token.value in [
            TokenValue.ASSIGN, TokenValue.ADD_ASSIGN, TokenValue.SUB_ASSIGN,
            TokenValue.MUL_ASSIGN, TokenValue.DIV_ASSIGN, TokenValue.MOD_ASSIGN,
            TokenValue.LEFT_SHIFT_ASSIGN, TokenValue.RIGHT_SHIFT_ASSIGN,
            TokenValue.AND_ASSIGN, TokenValue.XOR_ASSIGN, TokenValue.OR_ASSIGN
        ]:
            if translation_ctx is None:
                raise ValueError("Assignments not allowed in constant expressions")

            op_token = ctx.bump(frontend)
            right = self.parse_assignment(ctx, frontend, translation_ctx)
            
            from ....ir import Statement
            if op_token.value == TokenValue.ASSIGN:
                translation_ctx.add_statement(Statement.new_store(pointer=left, value=right))
                return right
            else:
                # Compound assignment: left = left op right
                # 1. Load left (actually 'left' handle IS a pointer in NAGA IR for Local/Global variables)
                # Wait, if 'left' is an expression handle for a variable, using it in a binary op 
                # might mean different things depending on context. 
                # In NAGA IR, to get the VALUE of a variable, you just use the LOCAL_VARIABLE expression.
                binary_op = self._map_compound_assignment_op(op_token.value)
                expr = Expression.new_binary(op=binary_op, left=left, right=right)
                new_value = translation_ctx.add_expression(expr)
                translation_ctx.add_statement(Statement.new_store(pointer=left, value=new_value))
                return new_value
        return left

    def _map_compound_assignment_op(self, token_value: TokenValue) -> BinaryOperator:
        mapping = {
            TokenValue.ADD_ASSIGN: BinaryOperator.ADD,
            TokenValue.SUB_ASSIGN: BinaryOperator.SUBTRACT,
            TokenValue.MUL_ASSIGN: BinaryOperator.MULTIPLY,
            TokenValue.DIV_ASSIGN: BinaryOperator.DIVIDE,
            TokenValue.MOD_ASSIGN: BinaryOperator.MODULO,
            TokenValue.AND_ASSIGN: BinaryOperator.AND,
            TokenValue.OR_ASSIGN: BinaryOperator.INCLUSIVE_OR,
            TokenValue.XOR_ASSIGN: BinaryOperator.EXCLUSIVE_OR,
            TokenValue.LEFT_SHIFT_ASSIGN: BinaryOperator.SHIFT_LEFT,
            TokenValue.RIGHT_SHIFT_ASSIGN: BinaryOperator.SHIFT_RIGHT,
        }
        return mapping.get(token_value, BinaryOperator.ADD)

    def parse_conditional(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        """Parse conditional (ternary) expressions."""
        return self.parse_logical_or(ctx, frontend, translation_ctx)

    def parse_logical_or(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_logical_xor, [TokenValue.LOGICAL_OR])

    def parse_logical_xor(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_logical_and, [TokenValue.LOGICAL_XOR])

    def parse_logical_and(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_bitwise_or, [TokenValue.LOGICAL_AND])

    def parse_bitwise_or(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_bitwise_xor, [TokenValue.VERTICAL_BAR])

    def parse_bitwise_xor(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_bitwise_and, [TokenValue.CARET])

    def parse_bitwise_and(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_equality, [TokenValue.AMPERSAND])

    def parse_equality(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_relational, [TokenValue.EQUAL, TokenValue.NOT_EQUAL])

    def parse_relational(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_shift, [
            TokenValue.LEFT_ANGLE, TokenValue.RIGHT_ANGLE, TokenValue.LESS_EQUAL, TokenValue.GREATER_EQUAL
        ])

    def parse_shift(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_additive, [
            TokenValue.LEFT_SHIFT, TokenValue.RIGHT_SHIFT
        ])

    def parse_additive(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_multiplicative, [
            TokenValue.PLUS, TokenValue.DASH
        ])

    def parse_multiplicative(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        return self._parse_binary(ctx, frontend, translation_ctx, self.parse_unary, [
            TokenValue.STAR, TokenValue.SLASH, TokenValue.PERCENT
        ])

    def _parse_binary(self, ctx: Any, frontend: Any, translation_ctx: Any, next_prec: Any, ops: List[TokenValue]) -> Any:
        left = next_prec(ctx, frontend, translation_ctx)
        while True:
            token = ctx.peek(frontend)
            if token and token.value in ops:
                op_token = ctx.bump(frontend)
                right = next_prec(ctx, frontend, translation_ctx)
                
                op = self._map_binary_op(op_token.value)
                expr = Expression.new_binary(op=op, left=left, right=right)
                left = translation_ctx.add_expression(expr)
            else:
                break
        return left

    def _map_binary_op(self, token_value: TokenValue) -> BinaryOperator:
        mapping = {
            TokenValue.PLUS: BinaryOperator.ADD,
            TokenValue.DASH: BinaryOperator.SUBTRACT,
            TokenValue.STAR: BinaryOperator.MULTIPLY,
            TokenValue.SLASH: BinaryOperator.DIVIDE,
            TokenValue.PERCENT: BinaryOperator.MODULO,
            TokenValue.EQUAL: BinaryOperator.EQUAL,
            TokenValue.NOT_EQUAL: BinaryOperator.NOT_EQUAL,
            TokenValue.LEFT_ANGLE: BinaryOperator.LESS,
            TokenValue.RIGHT_ANGLE: BinaryOperator.GREATER,
            TokenValue.LESS_EQUAL: BinaryOperator.LESS_EQUAL,
            TokenValue.GREATER_EQUAL: BinaryOperator.GREATER_EQUAL,
            TokenValue.LOGICAL_AND: BinaryOperator.LOGICAL_AND,
            TokenValue.LOGICAL_OR: BinaryOperator.LOGICAL_OR,
            TokenValue.AMPERSAND: BinaryOperator.AND,
            TokenValue.VERTICAL_BAR: BinaryOperator.INCLUSIVE_OR,
            TokenValue.CARET: BinaryOperator.EXCLUSIVE_OR,
            TokenValue.LEFT_SHIFT: BinaryOperator.SHIFT_LEFT,
            TokenValue.RIGHT_SHIFT: BinaryOperator.SHIFT_RIGHT,
        }
        return mapping.get(token_value, BinaryOperator.ADD) # Default

    def parse_unary(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        token = ctx.peek(frontend)
        if token and token.value in [
            TokenValue.PLUS, TokenValue.DASH, TokenValue.BANG, TokenValue.TILDE,
            TokenValue.INCREMENT, TokenValue.DECREMENT
        ]:
            op_token = ctx.bump(frontend)
            expr_handle = self.parse_unary(ctx, frontend, translation_ctx)
            
            if op_token.value == TokenValue.PLUS:
                return expr_handle
            
            op = self._map_unary_op(op_token.value)
            expr = Expression.new_unary(op=op, expr=expr_handle)
            return translation_ctx.add_expression(expr)
            
        return self.parse_postfix(ctx, frontend, translation_ctx)

    def _map_unary_op(self, token_value: TokenValue) -> UnaryOperator:
        mapping = {
            TokenValue.DASH: UnaryOperator.NEGATE,
            TokenValue.BANG: UnaryOperator.LOGICAL_NOT,
            TokenValue.TILDE: UnaryOperator.BITWISE_NOT,
        }
        return mapping.get(token_value, UnaryOperator.NEGATE)

    def parse_postfix(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        expr_handle = self.parse_primary(ctx, frontend, translation_ctx)
        while True:
            token = ctx.peek(frontend)
            if token is None:
                break
            
            if token.value == TokenValue.LEFT_BRACKET:
                ctx.bump(frontend)
                index_handle = self.parse_expression(ctx, frontend, translation_ctx)
                ctx.expect(frontend, TokenValue.RIGHT_BRACKET)
                
                expr = Expression(type=ExpressionType.ACCESS_INDEX, access_base=expr_handle, access_index=index_handle)
                expr_handle = translation_ctx.add_expression(expr)
            elif token.value == TokenValue.DOT:
                ctx.bump(frontend)
                field, _ = ctx.expect_ident(frontend)
                # Handle swizzle
                if all(c in 'xyzwrgbastpq' for c in field) and len(field) <= 4:
                    from ....ir import SwizzleComponent, VectorSize
                    components = []
                    mapping = {'x': 0, 'y': 1, 'z': 2, 'w': 3,
                               'r': 0, 'g': 1, 'b': 2, 'a': 3,
                               's': 0, 't': 1, 'p': 2, 'q': 3}
                    for c in field:
                        components.append(mapping[c])
                    
                    size_map = {1: None, 2: VectorSize.BI, 3: VectorSize.TRI, 4: VectorSize.QUAD}
                    swizzle_expr = Expression(
                        type=ExpressionType.SWIZZLE,
                        swizzle_size=size_map[len(field)],
                        swizzle_vector=expr_handle,
                        swizzle_pattern=components
                    )
                    expr_handle = translation_ctx.add_expression(swizzle_expr)
                else:
                    # Generic field access (struct)
                    expr = Expression(type=ExpressionType.ACCESS, access_base=expr_handle, access_index_value=0) # Need real index
                    expr_handle = translation_ctx.add_expression(expr)
            elif token.value in [TokenValue.INCREMENT, TokenValue.DECREMENT]:
                op_token = ctx.bump(frontend)
                # For now, just handle as dummy
                return expr_handle
            else:
                break
        return expr_handle

    def parse_primary(self, ctx: Any, frontend: Any, translation_ctx: Any) -> int:
        from ....ir import ExpressionType
        token = ctx.bump(frontend)
        if token.value == TokenValue.IDENTIFIER:
            name = str(token.data)
            # Handle function calls
            next_token = ctx.peek(frontend)
            if next_token and next_token.value == TokenValue.LEFT_PAREN:
                return frontend.function_parser.parse_function_call(ctx, frontend, translation_ctx, name)
            
            # Resolve variable (local or argument)
            resolved = translation_ctx.resolve_variable(name)
            if resolved is not None:
                kind, handle = resolved
                if kind == 'local':
                    expr = Expression(type=ExpressionType.LOCAL_VARIABLE, local_variable=handle)
                    return translation_ctx.add_expression(expr)
                elif kind == 'arg':
                    expr = Expression(type=ExpressionType.FUNCTION_ARGUMENT, function_argument=handle)
                    return translation_ctx.add_expression(expr)
            
            # Resolve global variable
            global_handle = frontend.module.get_global_handle_by_name(name)
            if global_handle is not None:
                expr = Expression(type=ExpressionType.GLOBAL_VARIABLE, global_variable=global_handle)
                return translation_ctx.add_expression(expr)

            raise ValueError(f"Unknown identifier: {name}")
            
        elif token.value == TokenValue.INT_CONSTANT:
            expr = Expression(type=ExpressionType.LITERAL, literal=token.data.value)
            return translation_ctx.add_expression(expr)
        elif token.value == TokenValue.FLOAT_CONSTANT:
            expr = Expression(type=ExpressionType.LITERAL, literal=token.data.value)
            return translation_ctx.add_expression(expr)
        elif token.value == TokenValue.BOOL_CONSTANT:
            expr = Expression(type=ExpressionType.LITERAL, literal=token.data)
            return translation_ctx.add_expression(expr)
        elif token.value == TokenValue.LEFT_PAREN:
            expr_handle = self.parse_expression(ctx, frontend, translation_ctx)
            ctx.expect(frontend, TokenValue.RIGHT_PAREN)
            return expr_handle
        else:
            raise ValueError(f"Unexpected token in expression: {token.value}")
