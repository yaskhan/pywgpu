"""
Expression and type parsing for WGSL.

This module contains methods for parsing WGSL expressions and types.
"""

from typing import Any, Optional, List, Tuple
from .lexer import Token, TokenKind
from .ast import (
    Expression, ExpressionKind, Ident,
    BinaryExpression, UnaryExpression, CallExpression,
    IndexExpression, MemberExpression, ConstructExpression,
    LiteralExpression
)
from .error import ParseError
from .number import parse_number


class ExpressionParser:
    """Helper class for parsing expressions."""
    
    def __init__(self, parser: Any):
        """
        Initialize expression parser.
        
        Args:
            parser: Parent WgslRecursiveParser instance
        """
        self.parser = parser
    
    def parse_expression(self) -> Any:
        """
        Parse an expression.
        
        Returns:
            Expression AST node
        """
        return self.parse_binary_expression(0)
    
    def parse_binary_expression(self, min_precedence: int) -> Any:
        """
        Parse binary expression with operator precedence.
        
        Args:
            min_precedence: Minimum precedence level
            
        Returns:
            Expression AST node
        """
        # Parse left side
        left = self.parse_unary_expression()
        
        while True:
            token = self.parser.peek()
            if token is None:
                break
            
            # Get operator precedence
            precedence = self.get_binary_precedence(token.kind)
            if precedence < min_precedence:
                break
            
            # Consume operator
            op_token = self.parser.advance()
            
            # Parse right side
            right = self.parse_binary_expression(precedence + 1)
            
            # Create binary expression
            left = Expression(
                kind=ExpressionKind.BINARY,
                data=BinaryExpression(left=left, op=op_token.value, right=right),
                span=(left.span[0], right.span[1])
            )
        
        return left
    
    def parse_unary_expression(self) -> Any:
        """
        Parse unary expression.
        
        Returns:
            Expression AST node
        """
        token = self.parser.peek()
        if token is None:
            raise ParseError(
                message="unexpected end of file in expression",
                labels=[(0, 0, "")],
                notes=[]
            )
        
        # Check for unary operators
        if token.kind in (TokenKind.MINUS, TokenKind.BANG, TokenKind.TILDE, TokenKind.AND, TokenKind.STAR):
            op_token = self.parser.advance()
            operand = self.parse_unary_expression()
            
            return Expression(
                kind=ExpressionKind.UNARY,
                data=UnaryExpression(op=op_token.value, expr=operand),
                span=(op_token.span[0], operand.span[1])
            )
        
        # Parse postfix expression
        return self.parse_postfix_expression()
    
    def parse_postfix_expression(self) -> Any:
        """
        Parse postfix expression (member access, indexing, calls).
        
        Returns:
            Expression AST node
        """
        expr = self.parse_primary_expression()
        
        while True:
            token = self.parser.peek()
            if token is None:
                break
            
            if token.kind == TokenKind.PERIOD:
                # Member access: expr.member
                self.parser.advance()
                member_token = self.parser.expect(TokenKind.IDENT)
                
                expr = Expression(
                    kind=ExpressionKind.MEMBER,
                    data=MemberExpression(base=expr, member=Ident(name=member_token.value, span=member_token.span)),
                    span=(expr.span[0], member_token.span[1])
                )
            
            elif token.kind == TokenKind.BRACKET_LEFT:
                # Index access: expr[index]
                self.parser.advance()
                index = self.parse_expression()
                self.parser.expect(TokenKind.BRACKET_RIGHT)
                
                expr = Expression(
                    kind=ExpressionKind.INDEX,
                    data=IndexExpression(base=expr, index=index),
                    span=(expr.span[0], index.span[1])
                )
            
            elif token.kind == TokenKind.PAREN_LEFT:
                # Function call: expr(args)
                self.parser.advance()
                args = self.parse_argument_list()
                close_paren = self.parser.expect(TokenKind.PAREN_RIGHT)
                
                expr = Expression(
                    kind=ExpressionKind.CALL,
                    data=CallExpression(function=expr, arguments=args),
                    span=(expr.span[0], close_paren.span[1])
                )
            
            else:
                break
        
        return expr
    
    def parse_primary_expression(self) -> Any:
        """
        Parse primary expression (literals, identifiers, parenthesized).
        
        Returns:
            Expression AST node
        """
        token = self.parser.peek()
        if token is None:
            raise ParseError(
                message="unexpected end of file in expression",
                labels=[(0, 0, "")],
                notes=[]
            )
        
        # Literals
        if token.kind == TokenKind.TRUE:
            self.parser.advance()
            return Expression(
                kind=ExpressionKind.LITERAL,
                data=LiteralExpression(value=True),
                span=token.span
            )
        
        elif token.kind == TokenKind.FALSE:
            self.parser.advance()
            return Expression(
                kind=ExpressionKind.LITERAL,
                data=LiteralExpression(value=False),
                span=token.span
            )
        
        elif token.kind == TokenKind.NUMBER:
            self.parser.advance()
            # Parse number literal
            num = parse_number(token.value, token.span)
            return Expression(
                kind=ExpressionKind.LITERAL,
                data=LiteralExpression(value=num),
                span=token.span
            )
        
        # Identifier or constructor
        elif token.kind == TokenKind.IDENT:
            self.parser.advance()
            
            # Check if it's a constructor: vec2(1, 2) or vec2<i32>(1, 2)
            is_generic = self.parser.peek() and self.parser.peek().kind == TokenKind.LESS_THAN
            is_call = self.parser.peek() and self.parser.peek().kind == TokenKind.PAREN_LEFT
            
            if is_generic or is_call:
                # Need to use the previously advanced token as part of the type
                # But TypeParser expects to advance the token itself.
                # Since we already advanced, we can either rewind or just use his parse_generic if it's LESS_THAN
                
                type_node = {"name": token.value, "span": token.span}
                if is_generic:
                    full_type = self.parser.type_parser.parse_generic_type(token.value, token.span)
                    type_node = full_type
                
                if self.parser.peek() and self.parser.peek().kind == TokenKind.PAREN_LEFT:
                    # Constructor: Type(args)
                    self.parser.advance()
                    args = self.parse_argument_list()
                    close_paren = self.parser.expect(TokenKind.PAREN_RIGHT)
                    
                    return Expression(
                        kind=ExpressionKind.CONSTRUCT,
                        data=ConstructExpression(ty=type_node, arguments=args),
                        span=(token.span[0], close_paren.span[1])
                    )
                elif is_generic:
                    # Type name with generics but no parentheses? 
                    # In expression context this is likely an error but let's return it as an IDENT-like or error out
                    raise ParseError(
                        message=f"type name {token.value}<...> must be followed by '(' in expression",
                        labels=[(token.span[0], token.span[1], "")],
                        notes=[]
                    )
            
            # Simple identifier
            return Expression(
                kind=ExpressionKind.IDENT,
                data=Ident(name=token.value, span=token.span),
                span=token.span
            )
        # Parenthesized expression
        elif token.kind == TokenKind.PAREN_LEFT:
            self.parser.advance()
            expr = self.parse_expression()
            close_paren = self.parser.expect(TokenKind.PAREN_RIGHT)
            
            # Parentheses don't change the expression kind or data, just the span
            expr.span = (token.span[0], close_paren.span[1])
            return expr
        
        else:
            raise ParseError(
                message=f"unexpected token in expression: {token.kind.value}",
                labels=[(token.span[0], token.span[1], "")],
                notes=[]
            )
    
    def parse_argument_list(self) -> List[Any]:
        """
        Parse function argument list.
        
        Returns:
            List of argument expressions
        """
        args = []
        
        while True:
            token = self.parser.peek()
            if token is None or token.kind == TokenKind.PAREN_RIGHT:
                break
            
            # Parse argument expression
            arg = self.parse_expression()
            args.append(arg)
            
            # Check for comma
            if self.parser.peek() and self.parser.peek().kind == TokenKind.COMMA:
                self.parser.advance()
            else:
                break
        
        return args
    
    def get_binary_precedence(self, kind: TokenKind) -> int:
        """
        Get operator precedence for binary operators.
        
        Args:
            kind: Token kind
            
        Returns:
            Precedence level (higher = tighter binding)
        """
        precedence_map = {
            # Multiplicative: *, /, %
            TokenKind.STAR: 12,
            TokenKind.FORWARD_SLASH: 12,
            TokenKind.MODULO: 12,
            
            # Additive: +, -
            TokenKind.PLUS: 11,
            TokenKind.MINUS: 11,
            
            # Shift: <<, >>
            TokenKind.SHIFT_LEFT: 10,
            TokenKind.SHIFT_RIGHT: 10,
            
            # Bitwise AND: &
            TokenKind.AND: 9,
            
            # Bitwise XOR: ^
            TokenKind.XOR: 8,
            
            # Bitwise OR: |
            TokenKind.OR: 7,
            
            # Relational: <, >, <=, >=
            TokenKind.LESS_THAN: 6,
            TokenKind.GREATER_THAN: 6,
            TokenKind.LESS_EQUAL: 6,
            TokenKind.GREATER_EQUAL: 6,
            
            # Equality: ==, !=
            TokenKind.EQUAL_EQUAL: 5,
            TokenKind.NOT_EQUAL: 5,
            
            # Logical AND: &&
            TokenKind.AND_AND: 4,
            
            # Logical OR: ||
            TokenKind.OR_OR: 3,
        }
        
        return precedence_map.get(kind, -1)


class TypeParser:
    """Helper class for parsing types."""
    
    def __init__(self, parser: Any):
        """
        Initialize type parser.
        
        Args:
            parser: Parent WgslRecursiveParser instance
        """
        self.parser = parser
    
    def parse_type(self) -> Any:
        """
        Parse a type.
        
        Returns:
            Type AST node
        """
        token = self.parser.peek()
        if token is None:
            raise ParseError(
                message="expected type",
                labels=[(0, 0, "")],
                notes=[]
            )
        
        if token.kind != TokenKind.IDENT:
            raise ParseError(
                message=f"expected type name, got {token.kind.value}",
                labels=[(token.span[0], token.span[1], "")],
                notes=[]
            )
        
        type_name_token = self.parser.advance()
        type_name = type_name_token.value
        
        # Check for generic parameters
        if self.parser.peek() and self.parser.peek().kind == TokenKind.LESS_THAN:
            return self.parse_generic_type(type_name, type_name_token.span)
        
        # Simple type
        return {"name": type_name, "span": type_name_token.span}
    
    def parse_generic_type(self, type_name: str, name_span: Tuple[int, int]) -> Any:
        """
        Parse generic type with parameters.
        
        Args:
            type_name: Base type name
            name_span: Span of type name
            
        Returns:
            Type AST node
        """
        self.parser.expect(TokenKind.LESS_THAN)
        
        # Parse type parameters
        params = []
        while True:
            token = self.parser.peek()
            if token is None or token.kind == TokenKind.GREATER_THAN:
                break
            
            # Parse type parameter
            param = self.parse_type()
            params.append(param)
            
            # Check for comma
            if self.parser.peek() and self.parser.peek().kind == TokenKind.COMMA:
                self.parser.advance()
            else:
                break
        
        close_bracket = self.parser.expect(TokenKind.GREATER_THAN)
        
        return {
            "name": type_name,
            "params": params,
            "span": (name_span[0], close_bracket.span[1])
        }
