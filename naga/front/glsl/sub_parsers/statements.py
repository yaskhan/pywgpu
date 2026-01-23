"""
Parser for GLSL statements.

This module handles parsing of GLSL statements including control flow,
assignments, and blocks.
"""

from typing import Any, Optional, List
from ..token import TokenValue
from ....ir import Statement, Block, Expression, ExpressionType, UnaryOperator

class StatementParser:
    """Parser for GLSL statements."""
    
    def __init__(self, lexer: Any):
        self.lexer = lexer
        self.errors: List[str] = []

    def parse_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        """Parse a statement."""
        token = ctx.peek(frontend)
        if token is None:
            return None
            
        if token.value == TokenValue.LEFT_BRACE:
            return self.parse_compound_statement(ctx, frontend, translation_ctx)
        elif token.value == TokenValue.IF:
            return self.parse_selection_statement(ctx, frontend, translation_ctx)
        elif token.value == TokenValue.FOR:
            return self.parse_for_statement(ctx, frontend, translation_ctx)
        elif token.value == TokenValue.WHILE:
            return self.parse_while_statement(ctx, frontend, translation_ctx)
        elif token.value == TokenValue.DO:
            return self.parse_do_while_statement(ctx, frontend, translation_ctx)
        elif token.value == TokenValue.BREAK:
            ctx.bump(frontend)
            ctx.expect(frontend, TokenValue.SEMICOLON)
            return Statement.new_break()
        elif token.value == TokenValue.CONTINUE:
            ctx.bump(frontend)
            ctx.expect(frontend, TokenValue.SEMICOLON)
            return Statement.new_continue()
        elif token.value == TokenValue.DISCARD:
            ctx.bump(frontend)
            ctx.expect(frontend, TokenValue.SEMICOLON)
            return Statement.new_kill()
        elif token.value == TokenValue.RETURN:
            return self.parse_return_statement(ctx, frontend, translation_ctx)
        # Check for Declaration
        # Heuristic: starts with type keyword or qualifier
        token = ctx.peek(frontend)
        if token.value in [
            TokenValue.CONST, TokenValue.IN, TokenValue.OUT, TokenValue.INOUT,
            TokenValue.UNIFORM, TokenValue.BUFFER, TokenValue.SHARED,
            TokenValue.STRUCT
        ]:
            return frontend.declaration_parser.parse_declaration(ctx, frontend, translation_ctx)
            
        is_type = False
        if token.value == TokenValue.VOID:
            is_type = True
        elif token.value == TokenValue.IDENTIFIER:
            name = str(token.data)
            # Basic GLSL types
            if name in ["float", "double", "int", "uint", "bool", "atomic_uint"]:
                is_type = True
            elif name.startswith(("vec", "ivec", "uvec", "bvec", "dvec")):
                is_type = True
            elif name.startswith(("mat", "dmat")):
                is_type = True
            elif name.startswith(("sampler", "isampler", "usampler", "image", "iimage", "uimage", "texture")):
                 is_type = True
        
        if is_type:
            # Check if it's a constructor call like vec3(1.0)
            next_token = ctx.peek(frontend, 1)
            if next_token and next_token.value == TokenValue.LEFT_PAREN:
                 # Constructor -> Expression
                 # BUT wait, simple constructor `vec3(0.0)` is expression.
                 # Function-like constructor?
                 # Declarations: `vec3 v = ...` -> Next token is Identifier.
                 # Constructor: `vec3(1)` -> Next token is `(`.
                 # So `if next_token == (` it is Expression.
                 pass
            else:
                 # Declaration -> vec3 v;
                 return frontend.declaration_parser.parse_declaration(ctx, frontend, translation_ctx)

        # Might be an expression statement
        expr_handle = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.SEMICOLON)
        return Statement.new_emit(expr_handle)

    def parse_compound_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        """Parse a block of statements { ... }."""
        ctx.expect(frontend, TokenValue.LEFT_BRACE)
        translation_ctx.push_scope()
        
        statements = []
        while ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.RIGHT_BRACE:
            stmt = self.parse_statement(ctx, frontend, translation_ctx)
            if stmt:
                statements.append(stmt)
                
        ctx.expect(frontend, TokenValue.RIGHT_BRACE)
        translation_ctx.pop_scope()
        
        # In Naga IR, a block is just a list of statements
        return Block.from_vec(statements)

    def parse_selection_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        """Parse if-else statement."""
        ctx.expect(frontend, TokenValue.IF)
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        condition_handle = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        
        true_stmt = self.parse_statement(ctx, frontend, translation_ctx)
        accept = Block.from_vec([true_stmt] if not isinstance(true_stmt, Block) else true_stmt.statements)
        
        reject = Block.from_vec([])
        token = ctx.peek(frontend)
        if token and token.value == TokenValue.ELSE:
            ctx.bump(frontend)
            false_stmt = self.parse_statement(ctx, frontend, translation_ctx)
            reject = Block.from_vec([false_stmt] if not isinstance(false_stmt, Block) else false_stmt.statements)
            
        return Statement.new_if(condition=condition_handle, accept=accept, reject=reject)

    def parse_return_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        ctx.expect(frontend, TokenValue.RETURN)
        value_handle = None
        if ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.SEMICOLON:
            value_handle = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.SEMICOLON)
        return Statement.new_return(value=value_handle)

    def parse_for_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        ctx.expect(frontend, TokenValue.FOR)
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        
        translation_ctx.push_scope()
        
        # 1. Init
        init = None
        if ctx.peek(frontend).value != TokenValue.SEMICOLON:
            # For now, let's try statement (which handles both decl and expr)
            init = self.parse_statement(ctx, frontend, translation_ctx)
        else:
            ctx.bump(frontend)
            
        # 2. Condition
        cond = None
        if ctx.peek(frontend).value != TokenValue.SEMICOLON:
            cond = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.SEMICOLON)
        
        # 3. Loop/Increment
        incr = None
        if ctx.peek(frontend).value != TokenValue.RIGHT_PAREN:
            incr = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        
        # 4. Body
        body_stmt = self.parse_statement(ctx, frontend, translation_ctx)
        
        translation_ctx.pop_scope()
        
        # Lowering: init; Loop { body; continuing: incr }
        body = Block.from_vec([body_stmt] if not isinstance(body_stmt, Block) else body_stmt.body)
        continuing = Block.from_vec([Statement.new_emit([incr])] if incr is not None else [])
        
        loop = Statement.new_loop(body=body, continuing=continuing, break_if=cond)
        
        if init:
            return Block.from_vec([init, loop])
        return loop

    def parse_while_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        ctx.expect(frontend, TokenValue.WHILE)
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        cond_handle = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        body_stmt = self.parse_statement(ctx, frontend, translation_ctx)

        #   if (!cond) break;
        #   body;
        # }
        
        not_cond = Expression(type=ExpressionType.UNARY, unary_op=UnaryOperator.LOGICAL_NOT, unary_expr=cond_handle)
        not_cond_handle = translation_ctx.add_expression(not_cond)
        
        break_stmt = Statement.new_break()
        if_break = Statement.new_if(condition=not_cond_handle, accept=Block.from_vec([break_stmt]), reject=Block.from_vec([]))
        
        body_stmts = [if_break]
        if isinstance(body_stmt, Block):
            body_stmts.extend(body_stmt.body)
        else:
            body_stmts.append(body_stmt)
            
        return Statement.new_loop(body=Block.from_vec(body_stmts), continuing=Block.from_vec([]), break_if=None)

    def parse_do_while_statement(self, ctx: Any, frontend: Any, translation_ctx: Any) -> Any:
        # do stmt while (cond);
        ctx.expect(frontend, TokenValue.DO)
        body_stmt = self.parse_statement(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.WHILE)
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        cond_handle = frontend.expression_parser.parse_expression(ctx, frontend, translation_ctx)
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        ctx.expect(frontend, TokenValue.SEMICOLON)
        
        #   body;
        # }
        
        not_cond = Expression(type=ExpressionType.UNARY, unary_op=UnaryOperator.LOGICAL_NOT, unary_expr=cond_handle)
        not_cond_handle = translation_ctx.add_expression(not_cond)
        
        body = Block.from_vec([body_stmt] if not isinstance(body_stmt, Block) else body_stmt.body)
        
        return Statement.new_loop(body=body, continuing=Block.from_vec([]), break_if=not_cond_handle)
