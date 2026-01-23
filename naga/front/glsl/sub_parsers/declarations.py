"""
Parser for GLSL declarations.

This module handles parsing of variable declarations, function declarations,
and other top-level declarations in GLSL.
"""

from typing import Any, Optional, List
from enum import Enum


class DeclarationType(Enum):
    """Types of declarations in GLSL."""
    VARIABLE = "variable"
    FUNCTION = "function"
    STRUCT = "struct"
    TYPE_DEF = "type_def"
    CONST = "const"
    ATTRIBUTE = "attribute"


class DeclarationParser:
    """Parser for GLSL declarations."""
    
    def __init__(self, lexer: Any):
        self.lexer = lexer
        self.errors: List[str] = []
    
    def parse_declaration(self, ctx: Any, frontend: Any, translation_ctx: Optional[Any] = None) -> Optional[Any]:
        """Parse a declaration statement."""
        # Check for layout qualifier
        layout = self.parse_layout_qualifier(ctx, frontend)
        
        # Check for type qualifier
        qualifiers = self.parse_type_qualifier(ctx, frontend)
        
        # Check if it's a block (uniform/buffer)
        from ..token import TokenValue
        token = ctx.peek(frontend)
        qual_kinds = [q.value for q in (qualifiers or []) if hasattr(q, 'value')]
        if (TokenValue.UNIFORM in qual_kinds or TokenValue.BUFFER in qual_kinds) and \
           token and token.value == TokenValue.IDENTIFIER:
            next_token = ctx.peek(frontend, 1)
            if next_token and next_token.value == TokenValue.LEFT_BRACE:
                return self.parse_block_declaration(ctx, frontend, layout, qualifiers)

        # Parse type specifier
        type_spec = frontend.type_parser_impl.parse_type_specifier(ctx, frontend)
        if type_spec is None:
            return None
        
        # Parse identifier
        from ..token import TokenValue
        token = ctx.peek(frontend)
        if token is None or token.value != TokenValue.IDENTIFIER:
            # Might be a struct definition without variable
            return "struct_def" # Placeholder
            
        name, meta = ctx.expect_ident(frontend)
        
        # Check for array dimensions
        from ..token import TokenValue
        if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.LEFT_BRACKET:
            type_spec = frontend.type_parser_impl.parse_array_dimensions(ctx, frontend, type_spec)
        
        # Branch between variable and function declaration
        from ..token import TokenValue
        next_token = ctx.peek(frontend)
        if next_token and next_token.value == TokenValue.LEFT_PAREN:
            # If we are inside a function (translation_ctx is not None), nested functions are invalid
            if translation_ctx is not None:
                raise ValueError("Nested functions are not allowed")
            # It's a function declaration or definition
            return frontend.function_parser.parse_function_declaration(ctx, frontend, type_spec, name, qualifiers)
        
        # Otherwise it's a variable declaration
        # Check for initializer
        initializer = None
        if next_token and next_token.value == TokenValue.ASSIGN:
            ctx.bump(frontend)
            
            # Use provided translation_ctx or create a dummy one for globals/constants
            parse_ctx = translation_ctx
            if parse_ctx is None:
                # Create dummy context for global/const initializer parsing
                # This is a temporary workaround to allow parsing to proceed.
                # In the future, we should implement proper Constant folding/parsing.
                from ....ir import Function, Block
                from ..context import Context
                dummy_func = Function(name="dummy_const_parser", result=None, body=Block.from_vec([]))
                parse_ctx = Context(frontend, dummy_func)
            
            initializer = frontend.expression_parser.parse_expression(ctx, frontend, parse_ctx)
            
        ctx.expect(frontend, TokenValue.SEMICOLON)
        
        # Add to global or local variables
        from ..variables import VariableDeclaration
        decl = VariableDeclaration(
            name=name,
            ty=type_spec,
            init=initializer,
            meta={'qualifiers': qualifiers, 'layout': layout}
        )
        
        if translation_ctx:
            # Local variable
            # Use add_local_variable which returns handle
            handle = translation_ctx.add_local_variable(name, type_spec, initializer)
            return handle
        else:
            # Global variable
            handle = frontend.add_global_var(ctx, decl)
            return handle
    
    def parse_layout_qualifier(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse layout qualifiers."""
        from ..token import TokenValue
        token = ctx.peek(frontend)
        if token is None or token.value != TokenValue.LAYOUT:
            return None
            
        ctx.bump(frontend)
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        
        # Parse layout arguments
        layout_args = {}
        while ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.RIGHT_PAREN:
            name, _ = ctx.expect_ident(frontend)
            if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.ASSIGN:
                ctx.bump(frontend)
                # For now just expect an integer
                token = ctx.bump(frontend)
                if token.value == TokenValue.INT_CONSTANT:
                    layout_args[name] = token.data.value
                else:
                    layout_args[name] = str(token.data)
            else:
                layout_args[name] = True
            
            if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.COMMA:
                ctx.bump(frontend)
                
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        return layout_args
    
    def parse_type_qualifier(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse type qualifiers."""
        from ..token import TokenValue
        qualifiers = []
        
        while True:
            token = ctx.peek(frontend)
            if token is None:
                break
                
            if token.value in [
                TokenValue.IN, TokenValue.OUT, TokenValue.INOUT,
                TokenValue.UNIFORM, TokenValue.BUFFER, TokenValue.CONST,
                TokenValue.SHARED, TokenValue.RESTRICT, TokenValue.MEMORY_QUALIFIER,
                TokenValue.INVARIANT, TokenValue.INTERPOLATION, TokenValue.SAMPLING,
                TokenValue.PRECISION_QUALIFIER
            ]:
                ctx.bump(frontend)
                qualifiers.append(token)
            else:
                break
                
        return qualifiers if qualifiers else None
    
    def parse_variable_declaration(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse variable declaration."""
        return None
    
    def parse_function_declaration(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse function declaration or definition."""
        return None
    
    def parse_struct_definition(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse struct definition."""
        return None

    def parse_block_declaration(self, ctx: Any, frontend: Any, layout: Any, qualifiers: Any) -> Any:
        """Parse a uniform or buffer block."""
        from ..token import TokenValue
        block_name, _ = ctx.expect_ident(frontend)
        ctx.expect(frontend, TokenValue.LEFT_BRACE)
        
        members = []
        while ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.RIGHT_BRACE:
            # Simple member parsing
            member_layout = self.parse_layout_qualifier(ctx, frontend)
            member_qualifiers = self.parse_type_qualifier(ctx, frontend) or []
            all_qualifiers = (qualifiers or []) + member_qualifiers
            
            type_spec = frontend.type_parser_impl.parse_type_specifier(ctx, frontend)
            if type_spec is None:
                ctx.bump(frontend)
                continue
                
            name, _ = ctx.expect_ident(frontend)
            
            # Check for array dimensions
            from ..token import TokenValue
            if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.LEFT_BRACKET:
                type_spec = frontend.type_parser_impl.parse_array_dimensions(ctx, frontend, type_spec)
                
            ctx.expect(frontend, TokenValue.SEMICOLON)
            
            from ..variables import VariableDeclaration
            decl = VariableDeclaration(
                name=name,
                ty=type_spec,
                init=None,
                meta={'qualifiers': all_qualifiers, 'layout': layout}
            )
            handle = frontend.add_global_var(ctx, decl)
            members.append(handle)
            
        ctx.expect(frontend, TokenValue.RIGHT_BRACE)
        
        if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.IDENTIFIER:
             ctx.bump(frontend) # Skip instance name
             
        ctx.expect(frontend, TokenValue.SEMICOLON)
        return members