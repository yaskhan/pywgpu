"""
WGSL lexer for tokenization.

Translated from wgpu-trunk/naga/src/front/wgsl/parse/lexer.rs

This module handles lexical analysis of WGSL source code.
"""

from typing import Iterator, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class TokenKind(Enum):
    """Token types in WGSL."""
    
    # Keywords
    ALIAS = "alias"
    BREAK = "break"
    CASE = "case"
    CONST = "const"
    CONST_ASSERT = "const_assert"
    CONTINUE = "continue"
    CONTINUING = "continuing"
    DEFAULT = "default"
    DIAGNOSTIC = "diagnostic"
    DISCARD = "discard"
    ELSE = "else"
    ENABLE = "enable"
    FALSE = "false"
    FN = "fn"
    FOR = "for"
    IF = "if"
    LOOP = "loop"
    OVERRIDE = "override"
    REQUIRES = "requires"
    RETURN = "return"
    STRUCT = "struct"
    SWITCH = "switch"
    TRUE = "true"
    VAR = "var"
    WHILE = "while"
    
    # Punctuation
    ARROW = "->"
    ATTR = "@"
    FORWARD_SLASH = "/"
    BANG = "!"
    BRACKET_LEFT = "["
    BRACKET_RIGHT = "]"
    BRACE_LEFT = "{"
    BRACE_RIGHT = "}"
    COLON = ":"
    COMMA = ","
    EQUAL = "="
    EQUAL_EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    MODULO = "%"
    MINUS = "-"
    MINUS_MINUS = "--"
    PERIOD = "."
    PLUS = "+"
    PLUS_PLUS = "++"
    OR = "|"
    OR_OR = "||"
    PAREN_LEFT = "("
    PAREN_RIGHT = ")"
    SEMICOLON = ";"
    STAR = "*"
    TILDE = "~"
    UNDERSCORE = "_"
    XOR = "^"
    PLUS_EQUAL = "+="
    MINUS_EQUAL = "-="
    TIMES_EQUAL = "*="
    DIVISION_EQUAL = "/="
    MODULO_EQUAL = "%="
    AND = "&"
    AND_AND = "&&"
    AND_EQUAL = "&="
    OR_EQUAL = "|="
    XOR_EQUAL = "^="
    SHIFT_LEFT = "<<"
    SHIFT_RIGHT = ">>"
    SHIFT_LEFT_EQUAL = "<<="
    SHIFT_RIGHT_EQUAL = ">>="
    
    # Literals
    IDENT = "ident"
    NUMBER = "number"
    
    # Special
    EOF = "eof"
    ERROR = "error"


@dataclass
class Token:
    """
    A token with its kind, value, and source location.
    
    Attributes:
        kind: Type of token
        value: Token value (for identifiers and literals)
        span: Source location (start, end)
    """
    kind: TokenKind
    value: Optional[str]
    span: Tuple[int, int]


class Lexer:
    """
    WGSL lexer for tokenizing source code.
    
    This converts WGSL source text into a stream of tokens.
    """
    
    def __init__(self, source: str):
        """
        Initialize lexer with source code.
        
        Args:
            source: WGSL source code to tokenize
        """
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
    
    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens."""
        while True:
            token = self.next_token()
            yield token
            if token.kind == TokenKind.EOF:
                break
    
    def next_token(self) -> Token:
        """
        Get the next token from the source.
        
        Returns:
            Next token
        """
        # Skip whitespace and comments
        self._skip_whitespace_and_comments()
        
        if self.position >= len(self.source):
            return Token(TokenKind.EOF, None, (self.position, self.position))
        
        start = self.position
        char = self.source[self.position]
        
        # Identifiers and keywords
        if char.isalpha() or char == '_':
            return self._read_ident_or_keyword(start)
        
        # Numbers
        if char.isdigit():
            return self._read_number(start)
        
        # Punctuation and operators
        return self._read_punctuation(start)
    
    def _skip_whitespace_and_comments(self) -> None:
        """Skip whitespace and comments."""
        while self.position < len(self.source):
            char = self.source[self.position]
            
            # Whitespace
            if char in ' \t\r\n':
                if char == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.position += 1
                continue
            
            # Line comment
            if char == '/' and self.position + 1 < len(self.source):
                if self.source[self.position + 1] == '/':
                    # Skip until end of line
                    while self.position < len(self.source) and self.source[self.position] != '\n':
                        self.position += 1
                    continue
                
                # Block comment
                if self.source[self.position + 1] == '*':
                    self.position += 2
                    while self.position + 1 < len(self.source):
                        if self.source[self.position] == '*' and self.source[self.position + 1] == '/':
                            self.position += 2
                            break
                        if self.source[self.position] == '\n':
                            self.line += 1
                            self.column = 1
                        self.position += 1
                    continue
            
            break
    
    def _read_ident_or_keyword(self, start: int) -> Token:
        """Read identifier or keyword."""
        while self.position < len(self.source):
            char = self.source[self.position]
            if char.isalnum() or char == '_':
                self.position += 1
            else:
                break
        
        value = self.source[start:self.position]
        
        # Check if it's a keyword
        keyword_map = {
            'alias': TokenKind.ALIAS,
            'break': TokenKind.BREAK,
            'case': TokenKind.CASE,
            'const': TokenKind.CONST,
            'const_assert': TokenKind.CONST_ASSERT,
            'continue': TokenKind.CONTINUE,
            'continuing': TokenKind.CONTINUING,
            'default': TokenKind.DEFAULT,
            'diagnostic': TokenKind.DIAGNOSTIC,
            'discard': TokenKind.DISCARD,
            'else': TokenKind.ELSE,
            'enable': TokenKind.ENABLE,
            'false': TokenKind.FALSE,
            'fn': TokenKind.FN,
            'for': TokenKind.FOR,
            'if': TokenKind.IF,
            'loop': TokenKind.LOOP,
            'override': TokenKind.OVERRIDE,
            'requires': TokenKind.REQUIRES,
            'return': TokenKind.RETURN,
            'struct': TokenKind.STRUCT,
            'switch': TokenKind.SWITCH,
            'true': TokenKind.TRUE,
            'var': TokenKind.VAR,
            'while': TokenKind.WHILE,
        }
        
        kind = keyword_map.get(value, TokenKind.IDENT)
        return Token(kind, value, (start, self.position))
    
    def _read_number(self, start: int) -> Token:
        """Read number literal."""
        # TODO: Implement full number parsing (hex, float, etc.)
        while self.position < len(self.source):
            char = self.source[self.position]
            if char.isdigit() or char in '.eE+-xXabcdefABCDEF':
                self.position += 1
            else:
                break
        
        value = self.source[start:self.position]
        return Token(TokenKind.NUMBER, value, (start, self.position))
    
    def _read_punctuation(self, start: int) -> Token:
        """Read punctuation or operator."""
        char = self.source[self.position]
        
        # Two-character operators
        if self.position + 1 < len(self.source):
            two_char = self.source[self.position:self.position + 2]
            two_char_map = {
                '->': TokenKind.ARROW,
                '==': TokenKind.EQUAL_EQUAL,
                '!=': TokenKind.NOT_EQUAL,
                '>=': TokenKind.GREATER_EQUAL,
                '<=': TokenKind.LESS_EQUAL,
                '||': TokenKind.OR_OR,
                '&&': TokenKind.AND_AND,
                '<<': TokenKind.SHIFT_LEFT,
                '>>': TokenKind.SHIFT_RIGHT,
                '++': TokenKind.PLUS_PLUS,
                '--': TokenKind.MINUS_MINUS,
                '+=': TokenKind.PLUS_EQUAL,
                '-=': TokenKind.MINUS_EQUAL,
                '*=': TokenKind.TIMES_EQUAL,
                '/=': TokenKind.DIVISION_EQUAL,
                '%=': TokenKind.MODULO_EQUAL,
                '&=': TokenKind.AND_EQUAL,
                '|=': TokenKind.OR_EQUAL,
                '^=': TokenKind.XOR_EQUAL,
            }
            
            if two_char in two_char_map:
                self.position += 2
                return Token(two_char_map[two_char], two_char, (start, self.position))
            
            # Three-character operators
            if self.position + 2 < len(self.source):
                three_char = self.source[self.position:self.position + 3]
                if three_char == '<<=':
                    self.position += 3
                    return Token(TokenKind.SHIFT_LEFT_EQUAL, three_char, (start, self.position))
                if three_char == '>>=':
                    self.position += 3
                    return Token(TokenKind.SHIFT_RIGHT_EQUAL, three_char, (start, self.position))
        
        # Single-character operators
        single_char_map = {
            '@': TokenKind.ATTR,
            '/': TokenKind.FORWARD_SLASH,
            '!': TokenKind.BANG,
            '[': TokenKind.BRACKET_LEFT,
            ']': TokenKind.BRACKET_RIGHT,
            '{': TokenKind.BRACE_LEFT,
            '}': TokenKind.BRACE_RIGHT,
            ':': TokenKind.COLON,
            ',': TokenKind.COMMA,
            '=': TokenKind.EQUAL,
            '>': TokenKind.GREATER_THAN,
            '<': TokenKind.LESS_THAN,
            '%': TokenKind.MODULO,
            '-': TokenKind.MINUS,
            '.': TokenKind.PERIOD,
            '+': TokenKind.PLUS,
            '|': TokenKind.OR,
            '(': TokenKind.PAREN_LEFT,
            ')': TokenKind.PAREN_RIGHT,
            ';': TokenKind.SEMICOLON,
            '*': TokenKind.STAR,
            '~': TokenKind.TILDE,
            '_': TokenKind.UNDERSCORE,
            '^': TokenKind.XOR,
            '&': TokenKind.AND,
        }
        
        if char in single_char_map:
            self.position += 1
            return Token(single_char_map[char], char, (start, self.position))
        
        # Unknown character
        self.position += 1
        return Token(TokenKind.ERROR, char, (start, self.position))
