"""
Statement parsing for WGSL.

This module contains methods for parsing WGSL statements.
"""

from typing import Any, Optional, List
from .lexer import TokenKind
from .ast import (
    Statement, StatementKind, Ident, VarDecl,
    IfStatement, SwitchStatement, LoopStatement,
    ReturnStatement, AssignmentStatement, BlockStatement
)
from .error import ParseError


class StatementParser:
    """Helper class for parsing statements."""
    
    def __init__(self, parser: Any):
        """
        Initialize statement parser.
        
        Args:
            parser: Parent WgslRecursiveParser instance
        """
        self.parser = parser
        self.expr_parser = None  # Will be set by parser
    
    def parse_statement(self) -> Optional[Statement]:
        """
        Parse a statement.
        
        Returns:
            Statement AST node or None
        """
        token = self.parser.peek()
        if token is None:
            return None
        
        # Variable declaration
        if token.kind == TokenKind.VAR:
            return self.parse_var_statement()
        
        # Const declaration (local)
        elif token.kind == TokenKind.CONST:
            return self.parse_const_statement()
        
        # Let declaration
        elif token.kind in (TokenKind.IDENT,):  # 'let' would be an ident
            # Check if it's actually 'let'
            if token.value == "let":
                return self.parse_let_statement()
        
        # If statement
        if token.kind == TokenKind.IF:
            return self.parse_if_statement()
        
        # Switch statement
        elif token.kind == TokenKind.SWITCH:
            return self.parse_switch_statement()
        
        # Loop statement
        elif token.kind == TokenKind.LOOP:
            return self.parse_loop_statement()
        
        # While loop
        elif token.kind == TokenKind.WHILE:
            return self.parse_while_statement()
        
        # For loop
        elif token.kind == TokenKind.FOR:
            return self.parse_for_statement()
        
        # Break
        elif token.kind == TokenKind.BREAK:
            # Consume break
            start_token = self.parser.advance()
            end_token = self.parser.expect(TokenKind.SEMICOLON)
            return Statement(
                kind=StatementKind.BREAK,
                data=None,
                span=(start_token.span[0], end_token.span[1])
            )
        
        # Continue
        elif token.kind == TokenKind.CONTINUE:
            # Consume continue
            start_token = self.parser.advance()
            end_token = self.parser.expect(TokenKind.SEMICOLON)
            return Statement(
                kind=StatementKind.CONTINUE,
                data=None,
                span=(start_token.span[0], end_token.span[1])
            )
        
        # Return
        elif token.kind == TokenKind.RETURN:
            return self.parse_return_statement()
        
        # Discard
        elif token.kind == TokenKind.DISCARD:
            # Consume discard
            start_token = self.parser.advance()
            end_token = self.parser.expect(TokenKind.SEMICOLON)
            return Statement(
                kind=StatementKind.DISCARD,
                data=None,
                span=(start_token.span[0], end_token.span[1])
            )
        
        # Block
        elif token.kind == TokenKind.BRACE_LEFT:
            start_token = self.parser.peek()
            statements = self.parser.parse_block()
            # parse_block consumes both braces and returns a list of statements
            # Wait, WgslRecursiveParser.parse_block usually consumes braces.
            # Let's assume it does.
            return Statement(
                kind=StatementKind.BLOCK,
                data=BlockStatement(statements=statements),
                span=(start_token.span[0], statements[-1].span[1] if statements else start_token.span[1])
            )
        
        # Expression statement or assignment
        else:
            return self.parse_expression_statement()
    
    def parse_var_statement(self) -> Statement:
        """Parse variable declaration statement."""
        start_token = self.parser.expect(TokenKind.VAR)
        
        # Variable name
        name_token = self.parser.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Type annotation (optional)
        type_ = None
        if self.parser.peek() and self.parser.peek().kind == TokenKind.COLON:
            self.parser.advance()
            if self.parser.type_parser:
                type_ = self.parser.type_parser.parse_type()
        
        # Initializer (optional)
        initializer = None
        if self.parser.peek() and self.parser.peek().kind == TokenKind.EQUAL:
            self.parser.advance()
            if self.expr_parser:
                initializer = self.expr_parser.parse_expression()
        
        end_token = self.parser.expect(TokenKind.SEMICOLON)
        
        var_decl = VarDecl(
            name=name,
            type_=type_,
            address_space=None,
            access_mode=None,
            initializer=initializer,
            attributes=[]
        )
        
        return Statement(
            kind=StatementKind.VAR_DECL,
            data=var_decl,
            span=(start_token.span[0], end_token.span[1])
        )
    
    def parse_const_statement(self) -> Statement:
        """Parse const declaration statement."""
        start_token = self.parser.expect(TokenKind.CONST)
        
        # Constant name
        name_token = self.parser.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Type annotation (optional)
        type_ = None
        if self.parser.peek() and self.parser.peek().kind == TokenKind.COLON:
            self.parser.advance()
            if self.parser.type_parser:
                type_ = self.parser.type_parser.parse_type()
        
        # Initializer (required)
        self.parser.expect(TokenKind.EQUAL)
        initializer = None
        if self.expr_parser:
            initializer = self.expr_parser.parse_expression()
        
        end_token = self.parser.expect(TokenKind.SEMICOLON)
        
        # In AST, we use VarDecl for const too (kind determines usage)
        # Actually we have ConstDecl, but for local scope maybe VarDecl is fine or we should use it?
        # NAGA uses Handle<Expression> for constant in local scope if it's 'const'.
        # Let's use VarDecl for now but we might want a LocalConst variant.
        var_decl = VarDecl(
            name=name,
            type_=type_,
            address_space=None,
            access_mode=None,
            initializer=initializer,
            attributes=[]
        )
        
        return Statement(
            kind=StatementKind.VAR_DECL,
            data=var_decl,
            span=(start_token.span[0], end_token.span[1])
        )
    
    def parse_let_statement(self) -> Statement:
        """Parse let declaration statement."""
        start_token = self.parser.advance()  # consume 'let'
        
        # Variable name
        name_token = self.parser.expect(TokenKind.IDENT)
        name = Ident(name=name_token.value, span=name_token.span)
        
        # Type annotation (optional)
        type_ = None
        if self.parser.peek() and self.parser.peek().kind == TokenKind.COLON:
            self.parser.advance()
            if self.parser.type_parser:
                type_ = self.parser.type_parser.parse_type()
        
        # Initializer (required)
        self.parser.expect(TokenKind.EQUAL)
        initializer = None
        if self.expr_parser:
            initializer = self.expr_parser.parse_expression()
        
        end_token = self.parser.expect(TokenKind.SEMICOLON)
        
        var_decl = VarDecl(
            name=name,
            type_=type_,
            address_space=None,
            access_mode=None,
            initializer=initializer,
            attributes=[]
        )
        
        return Statement(
            kind=StatementKind.VAR_DECL,
            data=var_decl,
            span=(start_token.span[0], end_token.span[1])
        )
    
    def parse_if_statement(self) -> Statement:
        """Parse if statement."""
        start_token = self.parser.expect(TokenKind.IF)
        
        # Condition
        condition = None
        if self.expr_parser:
            condition = self.expr_parser.parse_expression()
        
        # Then block
        then_block = self.parser.parse_block()
        
        # Else block (optional)
        else_block = []
        if self.parser.peek() and self.parser.peek().kind == TokenKind.ELSE:
            self.parser.advance()
            
            # Check for else if
            if self.parser.peek() and self.parser.peek().kind == TokenKind.IF:
                else_block = [self.parse_if_statement()]
            else:
                else_block = self.parser.parse_block()
        
        end_span = else_block[-1].span[1] if else_block else (then_block[-1].span[1] if then_block else start_token.span[1])
        
        return Statement(
            kind=StatementKind.IF,
            data=IfStatement(condition=condition, accept=then_block, reject=else_block),
            span=(start_token.span[0], end_span)
        )
    
    def parse_switch_statement(self) -> Statement:
        """Parse switch statement."""
        start_token = self.parser.expect(TokenKind.SWITCH)
        
        # Selector expression
        selector = None
        if self.expr_parser:
            selector = self.expr_parser.parse_expression()
        
        # Cases
        self.parser.expect(TokenKind.BRACE_LEFT)
        cases = []
        
        while True:
            token = self.parser.peek()
            if token is None or token.kind == TokenKind.BRACE_RIGHT:
                break
            
            # Parse case or default
            if token.kind == TokenKind.CASE:
                self.parser.advance()
                # Parse case values
                case_value = None
                if self.expr_parser:
                    case_value = self.expr_parser.parse_expression()
                self.parser.expect(TokenKind.COLON)
                # Parse case body
                case_body = self.parser.parse_block()
                cases.append({"kind": "case", "value": case_value, "body": case_body})
            
            elif token.kind == TokenKind.DEFAULT:
                self.parser.advance()
                self.parser.expect(TokenKind.COLON)
                # Parse default body
                default_body = self.parser.parse_block()
                cases.append({"kind": "default", "body": default_body})
            
            else:
                break
        
        end_token = self.parser.expect(TokenKind.BRACE_RIGHT)
        
        return Statement(
            kind=StatementKind.SWITCH,
            data=SwitchStatement(selector=selector, cases=cases),
            span=(start_token.span[0], end_token.span[1])
        )
    
    def parse_loop_statement(self) -> Statement:
        """Parse loop statement."""
        start_token = self.parser.expect(TokenKind.LOOP)
        
        # Loop body
        body = self.parser.parse_block()
        
        end_span = body[-1].span[1] if body else start_token.span[1]
        
        return Statement(
            kind=StatementKind.LOOP,
            data=LoopStatement(body=body),
            span=(start_token.span[0], end_span)
        )
    
    def parse_while_statement(self) -> Statement:
        """Parse while statement."""
        start_token = self.parser.expect(TokenKind.WHILE)
        
        # Condition
        condition = None
        if self.expr_parser:
            condition = self.expr_parser.parse_expression()
        
        # Body
        body = self.parser.parse_block()
        
        end_span = body[-1].span[1] if body else start_token.span[1]
        
        # 'while' is transformed into 'loop' with a conditional break internally by naga
        # but for AST we can keep it as is or transform it.
        # Let's keep it as StatementKind.WHILE for now.
        return Statement(
            kind=StatementKind.WHILE,
            data={"condition": condition, "body": body},
            span=(start_token.span[0], end_span)
        )
    
    def parse_for_statement(self) -> Statement:
        """Parse for statement."""
        start_token = self.parser.expect(TokenKind.FOR)
        
        self.parser.expect(TokenKind.PAREN_LEFT)
        
        # Initializer
        initializer = None
        if self.parser.peek() and self.parser.peek().kind != TokenKind.SEMICOLON:
            initializer = self.parse_statement()
        else:
            self.parser.expect(TokenKind.SEMICOLON)
        
        # Condition
        condition = None
        if self.parser.peek() and self.parser.peek().kind != TokenKind.SEMICOLON:
            if self.expr_parser:
                condition = self.expr_parser.parse_expression()
        self.parser.expect(TokenKind.SEMICOLON)
        
        # Update
        update = None
        if self.parser.peek() and self.parser.peek().kind != TokenKind.PAREN_RIGHT:
            if self.expr_parser:
                update = self.expr_parser.parse_expression()
        
        self.parser.expect(TokenKind.PAREN_RIGHT)
        
        # Body
        body = self.parser.parse_block()
        
        end_span = body[-1].span[1] if body else start_token.span[1]
        
        return Statement(
            kind=StatementKind.FOR,
            data={"initializer": initializer, "condition": condition, "update": update, "body": body},
            span=(start_token.span[0], end_span)
        )
    
    def parse_return_statement(self) -> Statement:
        """Parse return statement."""
        start_token = self.parser.expect(TokenKind.RETURN)
        
        # Return value (optional)
        value = None
        if self.parser.peek() and self.parser.peek().kind != TokenKind.SEMICOLON:
            if self.expr_parser:
                value = self.expr_parser.parse_expression()
        
        end_token = self.parser.expect(TokenKind.SEMICOLON)
        
        return Statement(
            kind=StatementKind.RETURN,
            data=ReturnStatement(value=value),
            span=(start_token.span[0], end_token.span[1])
        )
    
    def parse_expression_statement(self) -> Statement:
        """Parse expression statement or assignment."""
        start_span = self.parser.peek().span[0] if self.parser.peek() else 0
        
        # Parse left side
        lhs = None
        if self.expr_parser:
            lhs = self.expr_parser.parse_expression()
        
        # Check for assignment operators
        token = self.parser.peek()
        if token and token.kind in (
            TokenKind.EQUAL,
            TokenKind.PLUS_EQUAL,
            TokenKind.MINUS_EQUAL,
            TokenKind.TIMES_EQUAL,
            TokenKind.DIVISION_EQUAL,
            TokenKind.MODULO_EQUAL,
            TokenKind.AND_EQUAL,
            TokenKind.OR_EQUAL,
            TokenKind.XOR_EQUAL,
            TokenKind.SHIFT_LEFT_EQUAL,
            TokenKind.SHIFT_RIGHT_EQUAL
        ):
            # Assignment
            op_token = self.parser.advance()
            
            # Parse right side
            rhs = None
            if self.expr_parser:
                rhs = self.expr_parser.parse_expression()
            
            end_token = self.parser.expect(TokenKind.SEMICOLON)
            
            return Statement(
                kind=StatementKind.ASSIGNMENT,
                data=AssignmentStatement(lhs=lhs, op=op_token.value, rhs=rhs),
                span=(start_span, end_token.span[1])
            )
        else:
            # Expression statement (function call)
            end_token = self.parser.expect(TokenKind.SEMICOLON)
            
            return Statement(
                kind=StatementKind.CALL,
                data=lhs, # The call expression itself
                span=(start_span, end_token.span[1])
            )
