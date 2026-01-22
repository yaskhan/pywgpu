"""
Additional parsing methods for WGSL recursive parser.

This module extends the WgslRecursiveParser with complete implementations.
"""

from typing import Any, List
from .lexer import TokenKind
from .ast import *
from .error import ParseError


def add_parsing_methods(parser_class):
    """Add parsing methods to WgslRecursiveParser class."""
    
    def parse_function_parameters(self) -> List[Any]:
        """
        Parse function parameters.
        
        Returns:
            List of parameter declarations
        """
        parameters = []
        
        while True:
            token = self.peek()
            if token is None or token.kind == TokenKind.PAREN_RIGHT:
                break
            
            # Parse parameter attributes
            param_attrs = self.parse_attributes()
            
            # Parameter name
            name_token = self.expect(TokenKind.IDENT)
            name = Ident(name=name_token.value, span=name_token.span)
            
            # Type annotation (required)
            self.expect(TokenKind.COLON)
            param_type = self.type_parser.parse_type() if self.type_parser else None
            
            parameters.append({
                'name': name,
                'type': param_type,
                'attributes': param_attrs
            })
            
            # Check for comma
            if self.peek() and self.peek().kind == TokenKind.COMMA:
                self.advance()
            else:
                break
        
        return parameters
    
    def parse_struct_members(self) -> List[StructMember]:
        """
        Parse struct members.
        
        Returns:
            List of struct members
        """
        members = []
        
        while True:
            token = self.peek()
            if token is None or token.kind == TokenKind.BRACE_RIGHT:
                break
            
            # Parse member attributes
            member_attrs = self.parse_attributes()
            
            # Member name
            name_token = self.expect(TokenKind.IDENT)
            name = Ident(name=name_token.value, span=name_token.span)
            
            # Type annotation (required)
            self.expect(TokenKind.COLON)
            member_type = self.type_parser.parse_type() if self.type_parser else None
            
            # Optional comma or semicolon
            if self.peek() and self.peek().kind in (TokenKind.COMMA, TokenKind.SEMICOLON):
                self.advance()
            
            members.append(StructMember(
                name=name,
                type_=member_type,
                attributes=member_attrs
            ))
        
        return members
    
    def parse_override_decl_impl(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse override declaration."""
        self.expect(TokenKind.OVERRIDE)
        
        # Override name
        name_token = self.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Type annotation (optional)
        type_ = None
        if self.peek() and self.peek().kind == TokenKind.COLON:
            self.advance()
            type_ = self.type_parser.parse_type() if self.type_parser else None
        
        # Initializer (optional)
        initializer = None
        if self.peek() and self.peek().kind == TokenKind.EQUAL:
            self.advance()
            initializer = self.expr_parser.parse_expression() if self.expr_parser else None
        
        # Optional ID
        override_id = None
        if self.peek() and self.peek().kind == TokenKind.ATTR:
            self.advance()
            if self.peek() and self.peek().value == "id":
                self.advance()
                self.expect(TokenKind.PAREN_LEFT)
                id_token = self.expect(TokenKind.NUMBER)
                override_id = int(id_token.value)
                self.expect(TokenKind.PAREN_RIGHT)
        
        self.expect(TokenKind.SEMICOLON)
        
        override_decl = OverrideDecl(
            name=name,
            type_=type_,
            initializer=initializer,
            id=override_id
        )
        
        return GlobalDecl(kind=override_decl, dependencies=[])
    
    def parse_type_alias_impl(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse type alias."""
        self.expect(TokenKind.ALIAS)
        
        # Alias name
        name_token = self.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # = type
        self.expect(TokenKind.EQUAL)
        aliased_type = self.type_parser.parse_type() if self.type_parser else None
        
        self.expect(TokenKind.SEMICOLON)
        
        type_alias = TypeAlias(
            name=name,
            type_=aliased_type
        )
        
        return GlobalDecl(kind=type_alias, dependencies=[])
    
    def parse_const_assert_impl(self) -> GlobalDecl:
        """Parse const assertion."""
        self.expect(TokenKind.CONST_ASSERT)
        
        # Condition expression
        condition = self.expr_parser.parse_expression() if self.expr_parser else None
        
        self.expect(TokenKind.SEMICOLON)
        
        const_assert = ConstAssert(condition=condition)
        
        return GlobalDecl(kind=const_assert, dependencies=[])
    
    # Add methods to class
    parser_class.parse_function_parameters = parse_function_parameters
    parser_class.parse_struct_members = parse_struct_members
    parser_class.parse_override_decl_impl = parse_override_decl_impl
    parser_class.parse_type_alias_impl = parse_type_alias_impl
    parser_class.parse_const_assert_impl = parse_const_assert_impl
    
    return parser_class
