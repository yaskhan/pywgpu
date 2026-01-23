"""GLSL lexer implementation.

This module provides the Lexer class for tokenizing GLSL source code.
It handles identifiers, keywords, numbers, operators, and preprocessor directives.
"""

from typing import List, Optional, Dict, Any, Tuple
import re

from ...span import Span
from .token import Token, TokenValue, Float, Integer, Directive, DirectiveKind
from .ast import Precision
from naga.ir import Interpolation, Sampling, StorageAccess

class Lexer:
    """GLSL lexer and basic preprocessor."""

    # Keyword mappings
    KEYWORDS: Dict[str, TokenValue] = {
        "layout": TokenValue.LAYOUT,
        "in": TokenValue.IN,
        "out": TokenValue.OUT,
        "inout": TokenValue.INOUT,
        "uniform": TokenValue.UNIFORM,
        "buffer": TokenValue.BUFFER,
        "const": TokenValue.CONST,
        "shared": TokenValue.SHARED,
        "restrict": TokenValue.RESTRICT,
        "readonly": TokenValue.MEMORY_QUALIFIER,
        "writeonly": TokenValue.MEMORY_QUALIFIER,
        "invariant": TokenValue.INVARIANT,
        "flat": TokenValue.INTERPOLATION,
        "smooth": TokenValue.INTERPOLATION,
        "noperspective": TokenValue.INTERPOLATION,
        "centroid": TokenValue.SAMPLING,
        "sample": TokenValue.SAMPLING,
        "precision": TokenValue.PRECISION,
        "lowp": TokenValue.PRECISION_QUALIFIER,
        "mediump": TokenValue.PRECISION_QUALIFIER,
        "highp": TokenValue.PRECISION_QUALIFIER,
        "continue": TokenValue.CONTINUE,
        "break": TokenValue.BREAK,
        "return": TokenValue.RETURN,
        "discard": TokenValue.DISCARD,
        "if": TokenValue.IF,
        "else": TokenValue.ELSE,
        "switch": TokenValue.SWITCH,
        "case": TokenValue.CASE,
        "default": TokenValue.DEFAULT,
        "while": TokenValue.WHILE,
        "do": TokenValue.DO,
        "for": TokenValue.FOR,
        "void": TokenValue.VOID,
        "struct": TokenValue.STRUCT,
        "true": TokenValue.BOOL_CONSTANT,
        "false": TokenValue.BOOL_CONSTANT,
    }

    # Operator mappings (longest first for matching)
    OPERATORS: List[Tuple[str, TokenValue]] = [
        ("<<=", TokenValue.LEFT_SHIFT_ASSIGN),
        (">>=", TokenValue.RIGHT_SHIFT_ASSIGN),
        ("++", TokenValue.INCREMENT),
        ("--", TokenValue.DECREMENT),
        ("||", TokenValue.LOGICAL_OR),
        ("&&", TokenValue.LOGICAL_AND),
        ("^^", TokenValue.LOGICAL_XOR),
        ("<=", TokenValue.LESS_EQUAL),
        (">=", TokenValue.GREATER_EQUAL),
        ("==", TokenValue.EQUAL),
        ("!=", TokenValue.NOT_EQUAL),
        ("<<", TokenValue.LEFT_SHIFT),
        (">>", TokenValue.RIGHT_SHIFT),
        ("+=", TokenValue.ADD_ASSIGN),
        ("-=", TokenValue.SUB_ASSIGN),
        ("*=", TokenValue.MUL_ASSIGN),
        ("/=", TokenValue.DIV_ASSIGN),
        ("%=", TokenValue.MOD_ASSIGN),
        ("&=", TokenValue.AND_ASSIGN),
        ("^=", TokenValue.XOR_ASSIGN),
        ("|=", TokenValue.OR_ASSIGN),
        ("=", TokenValue.ASSIGN),
        ("<", TokenValue.LEFT_ANGLE),
        (">", TokenValue.RIGHT_ANGLE),
        ("{", TokenValue.LEFT_BRACE),
        ("}", TokenValue.RIGHT_BRACE),
        ("(", TokenValue.LEFT_PAREN),
        (")", TokenValue.RIGHT_PAREN),
        ("[", TokenValue.LEFT_BRACKET),
        ("]", TokenValue.RIGHT_BRACKET),
        (",", TokenValue.COMMA),
        (";", TokenValue.SEMICOLON),
        (":", TokenValue.COLON),
        (".", TokenValue.DOT),
        ("!", TokenValue.BANG),
        ("-", TokenValue.DASH),
        ("~", TokenValue.TILDE),
        ("+", TokenValue.PLUS),
        ("*", TokenValue.STAR),
        ("/", TokenValue.SLASH),
        ("%", TokenValue.PERCENT),
        ("|", TokenValue.VERTICAL_BAR),
        ("^", TokenValue.CARET),
        ("&", TokenValue.AMPERSAND),
        ("?", TokenValue.QUESTION),
    ]

    def __init__(self, source: str, defines: Optional[Dict[str, str]] = None):
        self.source = source
        self.defines = defines or {}
        self.pos = 0
        self.line = 1
        self.column = 1
        self.length = len(source)
        self.tokens: List[Token] = []
        self._is_first_directive = True

    def tokenize(self) -> List[Token]:
        """Convert source string to token stream."""
        while not self._is_eof():
            self._skip_whitespace_and_comments()
            if self._is_eof():
                break

            if self._peek() == '#':
                self._handle_directive()
            elif self._peek().isalpha() or self._peek() == '_':
                self._handle_identifier_or_keyword()
            elif self._peek().isdigit():
                self._handle_number()
            else:
                self._handle_operator_or_delimiter()

        return self.tokens

    def _peek(self, offset: int = 0) -> str:
        """Look at the character at the current position + offset."""
        if self.pos + offset >= self.length:
            return '\0'
        return self.source[self.pos + offset]

    def _advance(self, count: int = 1) -> str:
        """Move the position forward by count characters."""
        result = ""
        for _ in range(count):
            if self._is_eof():
                break
            char = self.source[self.pos]
            result += char
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return result

    def _is_eof(self) -> bool:
        """Check if end of stream reached."""
        return self.pos >= self.length

    def _skip_whitespace_and_comments(self) -> None:
        """Skip over whitespace and GLSL comments."""
        while not self._is_eof():
            char = self._peek()
            if char.isspace():
                self._advance()
            elif char == '/' and self._peek(1) == '/':
                # Line comment
                while not self._is_eof() and self._peek() != '\n':
                    self._advance()
            elif char == '/' and self._peek(1) == '*':
                # Block comment
                self._advance(2)
                while not self._is_eof() and not (self._peek() == '*' and self._peek(1) == '/'):
                    self._advance()
                if not self._is_eof():
                    self._advance(2)
            else:
                break

    def _handle_directive(self) -> None:
        """Handle preprocessor directives."""
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        
        self._advance()  # Skip '#'
        self._skip_whitespace_and_comments()
        
        directive_name = ""
        while not self._is_eof() and self._peek().isalpha():
            directive_name += self._advance()
        
        # We only handle a subset of directives needed for NAGA
        meta = Span(start_pos, self.pos)
        
        # Collect tokens until end of line
        directive_tokens = []
        while not self._is_eof() and self._peek() != '\n':
            self._skip_whitespace_and_comments()
            if self._peek() == '\n' or self._is_eof():
                break
            # For now, just collect as strings/lexed parts
            # Real preprocessor would be more complex
            part = ""
            if self._peek().isalpha() or self._peek() == '_':
                while not self._is_eof() and (self._peek().isalnum() or self._peek() == '_'):
                    part += self._advance()
            elif self._peek().isdigit():
                while not self._is_eof() and self._peek().isdigit():
                    part += self._advance()
            else:
                part = self._advance()
            directive_tokens.append(part)

        if directive_name == "version":
            self.tokens.append(Token.simple(TokenValue.DIRECTIVE, meta)) # Placeholder
            # In a real implementation, we'd store the version directive separately
            # or handle it to set parser state. For now, let's just mark it.
            pass
        elif directive_name == "extension":
            pass
        elif directive_name == "pragma":
            pass

    def _handle_identifier_or_keyword(self) -> None:
        """Handle identifiers and keywords."""
        start_pos = self.pos
        name = ""
        while not self._is_eof() and (self._peek().isalnum() or self._peek() == '_'):
            name += self._advance()
        
        meta = Span(start_pos, self.pos)
        
        if name in self.KEYWORDS:
            value = self.KEYWORDS[name]
            if value == TokenValue.BOOL_CONSTANT:
                self.tokens.append(Token.bool_constant(name == "true", meta))
            elif value == TokenValue.INTERPOLATION:
                interp = Interpolation.FLAT
                if name == "smooth": interp = Interpolation.SMOOTH
                elif name == "noperspective": interp = Interpolation.PERSPECTIVE
                self.tokens.append(Token.interpolation(interp, meta))
            elif value == TokenValue.SAMPLING:
                samp = Sampling.CENTROID
                if name == "sample": samp = Sampling.SAMPLE
                self.tokens.append(Token.sampling(samp, meta))
            elif value == TokenValue.PRECISION_QUALIFIER:
                prec = Precision.MEDIUM
                if name == "lowp": prec = Precision.LOW
                elif name == "highp": prec = Precision.HIGH
                self.tokens.append(Token.precision_qualifier(prec, meta))
            elif name == "readonly":
                self.tokens.append(Token.memory_qualifier(StorageAccess.LOAD, meta))
            elif name == "writeonly":
                self.tokens.append(Token.memory_qualifier(StorageAccess.STORE, meta))
            else:
                self.tokens.append(Token.simple(value, meta))
        else:
            self.tokens.append(Token.identifier(name, meta))

    def _handle_number(self) -> None:
        """Handle numeric literals (integers and floats)."""
        start_pos = self.pos
        content = ""
        is_float = False
        
        # Handle hex/octal if needed, but keeping it simple for now
        while not self._is_eof() and (self._peek().isdigit() or self._peek() == '.'):
            if self._peek() == '.':
                is_float = True
            content += self._advance()
            
        # Handle suffix (f, u, lf, etc.)
        suffix = ""
        while not self._is_eof() and self._peek().lower() in "ful":
            suffix += self._advance().lower()
            
        meta = Span(start_pos, self.pos)
        
        if is_float or 'f' in suffix:
            val = float(content)
            width = 64 if "lf" in suffix else 32
            self.tokens.append(Token.float_constant(Float(val, width), meta))
        else:
            val = int(content)
            signed = 'u' not in suffix
            width = 32 # Default
            self.tokens.append(Token.int_constant(Integer(val, signed, width), meta))

    def _handle_operator_or_delimiter(self) -> None:
        """Handle multi-character and single-character operators."""
        start_pos = self.pos
        
        for op_str, token_val in self.OPERATORS:
            match = True
            for i, char in enumerate(op_str):
                if self._peek(i) != char:
                    match = False
                    break
            
            if match:
                self._advance(len(op_str))
                meta = Span(start_pos, self.pos)
                self.tokens.append(Token.simple(token_val, meta))
                return
        
        # If no operator matched, it might be an illegal character
        char = self._advance()
        # For now just ignore or we could raise an error
