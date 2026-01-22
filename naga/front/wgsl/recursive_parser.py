"""
Recursive descent parser for WGSL.

This module implements the actual parsing logic for WGSL syntax.
"""

from typing import Any, Optional, List
from .lexer import Lexer, Token, TokenKind
from .ast import *
from .enable_extension import EnableExtensionSet
from .language_extension import LanguageExtensionSet
from .error import ParseError
from .expression_parser import ExpressionParser, TypeParser
from .statement_parser import StatementParser
from .parser_extensions import add_parsing_methods


@add_parsing_methods
class WgslRecursiveParser:
    """
    Recursive descent parser for WGSL.
    
    This implements the core parsing logic, converting tokens into AST nodes.
    """
    
    def __init__(
        self,
        lexer: Lexer,
        enable_extensions: EnableExtensionSet,
        language_extensions: LanguageExtensionSet,
        options: Any
    ):
        """
        Initialize the parser.
        
        Args:
            lexer: Token stream
            enable_extensions: Enabled extensions
            language_extensions: Active language extensions
            options: Parsing options
        """
        self.lexer = lexer
        self.enable_extensions = enable_extensions
        self.language_extensions = language_extensions
        self.options = options
        self.current_token: Optional[Token] = None
        self.peeked_token: Optional[Token] = None
        
        # Initialize helper parsers
        self.expr_parser = ExpressionParser(self)
        self.type_parser = TypeParser(self)
        self.stmt_parser = StatementParser(self)
        self.stmt_parser.expr_parser = self.expr_parser
    
    def peek(self) -> Optional[Token]:
        """
        Peek at the next token without consuming it.
        
        Returns:
            Next token or None if EOF
        """
        if self.peeked_token is None:
            self.peeked_token = next(self.lexer, None)
        return self.peeked_token
    
    def advance(self) -> Optional[Token]:
        """
        Consume and return the next token.
        
        Returns:
            Consumed token or None if EOF
        """
        if self.peeked_token is not None:
            token = self.peeked_token
            self.peeked_token = None
            self.current_token = token
            return token
        
        token = next(self.lexer, None)
        self.current_token = token
        return token
    
    def expect(self, kind: TokenKind) -> Token:
        """
        Expect a specific token kind.
        
        Args:
            kind: Expected token kind
            
        Returns:
            The token
            
        Raises:
            ParseError: If token doesn't match
        """
        token = self.advance()
        if token is None or token.kind != kind:
            raise ParseError(
                message=f"expected {kind.value}, got {token.kind.value if token else 'EOF'}",
                labels=[(token.span[0] if token else 0, token.span[1] if token else 0, "")],
                notes=[]
            )
        return token
    
    def is_directive(self) -> bool:
        """
        Check if next token starts a directive.
        
        Returns:
            True if directive
        """
        token = self.peek()
        if token is None:
            return False
        
        return token.kind in (
            TokenKind.ENABLE,
            TokenKind.REQUIRES,
            TokenKind.DIAGNOSTIC
        )
    
    def parse_directive(self) -> Any:
        """
        Parse a directive (enable, requires, diagnostic).
        
        Returns:
            Directive AST node
        """
        token = self.advance()
        
        if token.kind == TokenKind.ENABLE:
            return self.parse_enable_directive()
        elif token.kind == TokenKind.REQUIRES:
            return self.parse_requires_directive()
        elif token.kind == TokenKind.DIAGNOSTIC:
            return self.parse_diagnostic_directive()
        else:
            raise ParseError(
                message=f"unexpected directive: {token.kind.value}",
                labels=[(token.span[0], token.span[1], "")],
                notes=[]
            )
    
    def parse_enable_directive(self) -> Any:
        """Parse enable directive."""
        from .directive import parse_enable_directive
        
        # Parse extension names
        extensions = []
        while True:
            token = self.peek()
            if token is None or token.kind == TokenKind.SEMICOLON:
                break
            
            if token.kind == TokenKind.IDENT:
                self.advance()
                extensions.append(token.value)
            
            # Check for comma
            if self.peek() and self.peek().kind == TokenKind.COMMA:
                self.advance()
            else:
                break
        
        self.expect(TokenKind.SEMICOLON)
        
        return parse_enable_directive(extensions)
    
    def parse_requires_directive(self) -> Any:
        """Parse requires directive."""
        from .directive import parse_requires_directive
        
        # Parse feature names
        features = []
        while True:
            token = self.peek()
            if token is None or token.kind == TokenKind.SEMICOLON:
                break
            
            if token.kind == TokenKind.IDENT:
                self.advance()
                features.append(token.value)
            
            # Check for comma
            if self.peek() and self.peek().kind == TokenKind.COMMA:
                self.advance()
            else:
                break
        
        self.expect(TokenKind.SEMICOLON)
        
        return parse_requires_directive(features)
    
    def parse_diagnostic_directive(self) -> Any:
        """Parse diagnostic directive."""
        from .directive import parse_diagnostic_directive
        
        self.expect(TokenKind.PAREN_LEFT)
        
        # Parse severity
        severity_token = self.expect(TokenKind.IDENT)
        severity = severity_token.value
        
        self.expect(TokenKind.COMMA)
        
        # Parse rule
        rule_token = self.expect(TokenKind.IDENT)
        rule = rule_token.value
        
        self.expect(TokenKind.PAREN_RIGHT)
        self.expect(TokenKind.SEMICOLON)
        
        return parse_diagnostic_directive(severity, rule)
    
    def parse_global_decl(self) -> Optional[GlobalDecl]:
        """
        Parse a global declaration.
        
        Returns:
            GlobalDecl AST node or None
        """
        token = self.peek()
        if token is None:
            return None
        
        # Parse attributes first
        attributes = self.parse_attributes()
        
        token = self.peek()
        if token is None:
            return None
        
        # Determine declaration type
        if token.kind == TokenKind.FN:
            return self.parse_function_decl(attributes)
        elif token.kind == TokenKind.VAR:
            return self.parse_var_decl(attributes, is_global=True)
        elif token.kind == TokenKind.CONST:
            return self.parse_const_decl(attributes)
        elif token.kind == TokenKind.OVERRIDE:
            return self.parse_override_decl(attributes)
        elif token.kind == TokenKind.STRUCT:
            return self.parse_struct_decl(attributes)
        elif token.kind == TokenKind.ALIAS:
            return self.parse_type_alias(attributes)
        elif token.kind == TokenKind.CONST_ASSERT:
            return self.parse_const_assert()
        else:
            raise ParseError(
                message=f"unexpected token in global scope: {token.kind.value}",
                labels=[(token.span[0], token.span[1], "")],
                notes=[]
            )
    
    def parse_attributes(self) -> List[Attribute]:
        """
        Parse attributes (@vertex, @group(0), etc.).
        
        Returns:
            List of attributes
        """
        attributes = []
        
        while True:
            token = self.peek()
            if token is None or token.kind != TokenKind.ATTR:
                break
            
            self.advance()  # consume @
            
            # Get attribute name
            name_token = self.expect(TokenKind.IDENT)
            name = name_token.value
            
            # Check for arguments
            args = []
            if self.peek() and self.peek().kind == TokenKind.PAREN_LEFT:
                self.advance()
                
                # Parse arguments
                while True:
                    if self.peek() and self.peek().kind == TokenKind.PAREN_RIGHT:
                        break
                    
                    # TODO: Parse expression for argument
                    arg_token = self.advance()
                    if arg_token:
                        args.append(arg_token.value)
                    
                    if self.peek() and self.peek().kind == TokenKind.COMMA:
                        self.advance()
                    else:
                        break
                
                self.expect(TokenKind.PAREN_RIGHT)
            
            attributes.append(Attribute(
                name=name,
                arguments=args,
                span=name_token.span
            ))
        
        return attributes
    
    def parse_function_decl(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse function declaration."""
        self.expect(TokenKind.FN)
        
        # Function name
        name_token = self.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Parameters
        self.expect(TokenKind.PAREN_LEFT)
        parameters = self.parse_function_parameters()
        self.expect(TokenKind.PAREN_RIGHT)
        
        # Return type
        return_type = None
        if self.peek() and self.peek().kind == TokenKind.ARROW:
            self.advance()
            return_type = self.type_parser.parse_type()
        
        # Function body
        body = self.parse_block()
        
        func_decl = FunctionDecl(
            name=name,
            parameters=parameters,
            return_type=return_type,
            body=body,
            attributes=attributes
        )
        
        return GlobalDecl(kind=func_decl, dependencies=[])
    
    def parse_var_decl(self, attributes: List[Attribute], is_global: bool) -> GlobalDecl:
        """Parse variable declaration."""
        self.expect(TokenKind.VAR)
        
        # TODO: Parse address space and access mode
        # var<storage, read_write> or var<private>
        
        # Variable name
        name_token = self.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Type annotation
        type_ = None
        if self.peek() and self.peek().kind == TokenKind.COLON:
            self.advance()
            type_ = self.type_parser.parse_type()
        
        # Initializer
        initializer = None
        if self.peek() and self.peek().kind == TokenKind.EQUAL:
            self.advance()
            initializer = self.expr_parser.parse_expression()
        
        self.expect(TokenKind.SEMICOLON)
        
        var_decl = VarDecl(
            name=name,
            type_=type_,
            address_space=None,
            access_mode=None,
            initializer=initializer,
            attributes=attributes
        )
        
        return GlobalDecl(kind=var_decl, dependencies=[])
    
    def parse_const_decl(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse const declaration."""
        self.expect(TokenKind.CONST)
        
        # Constant name
        name_token = self.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Type annotation (optional)
        type_ = None
        if self.peek() and self.peek().kind == TokenKind.COLON:
            self.advance()
            type_ = self.type_parser.parse_type()
        
        # Initializer (required)
        self.expect(TokenKind.EQUAL)
        initializer = self.expr_parser.parse_expression()
        
        self.expect(TokenKind.SEMICOLON)
        
        const_decl = ConstDecl(
            name=name,
            type_=type_,
            initializer=initializer
        )
        
        return GlobalDecl(kind=const_decl, dependencies=[])
    
    def parse_override_decl(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse override declaration."""
        return self.parse_override_decl_impl(attributes)
    
    def parse_struct_decl(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse struct declaration."""
        self.expect(TokenKind.STRUCT)
        
        # Struct name
        name_token = self.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Members
        self.expect(TokenKind.BRACE_LEFT)
        members = self.parse_struct_members()
        self.expect(TokenKind.BRACE_RIGHT)
        
        struct_decl = StructDecl(
            name=name,
            members=members
        )
        
        return GlobalDecl(kind=struct_decl, dependencies=[])
    
    def parse_type_alias(self, attributes: List[Attribute]) -> GlobalDecl:
        """Parse type alias."""
        return self.parse_type_alias_impl(attributes)
    
    def parse_const_assert(self) -> GlobalDecl:
        """Parse const assertion."""
        return self.parse_const_assert_impl()
    
    def parse_block(self) -> List[Statement]:
        """
        Parse a statement block.
        
        Returns:
            List of statements
        """
        self.expect(TokenKind.BRACE_LEFT)
        
        statements = []
        while True:
            token = self.peek()
            if token is None or token.kind == TokenKind.BRACE_RIGHT:
                break
            
            # TODO: Parse statement
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        self.expect(TokenKind.BRACE_RIGHT)
        
        return statements
    
    def parse_statement(self) -> Optional[Statement]:
        """
        Parse a statement.
        
        Returns:
            Statement AST node
        """
        return self.stmt_parser.parse_statement()
