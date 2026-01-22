"""GLSL token definitions.

This module is a Python translation of `wgpu-trunk/naga/src/front/glsl/token.rs`.
It provides token types used in GLSL lexing and parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from naga.ir import Interpolation, Sampling, StorageAccess, Type
from naga.span import Span

from .ast import Precision


@dataclass(frozen=True, slots=True)
class Float:
    """Float literal token value."""
    value: float
    width: int = 32  # 32 or 64 bits


@dataclass(frozen=True, slots=True)
class Integer:
    """Integer literal token value."""
    value: int
    signed: bool = True
    width: int = 32  # 8, 16, 32, or 64 bits


class TokenValue(Enum):
    """A token value used in GLSL parsing.
    
    This enum represents all the different token types that can appear
    in GLSL source code. Each variant either carries no data or carries
    associated data like identifiers or literal values.
    """
    
    # Identifiers and literals (these need associated data, handled separately)
    IDENTIFIER = "identifier"
    FLOAT_CONSTANT = "float_constant"
    INT_CONSTANT = "int_constant"
    BOOL_CONSTANT = "bool_constant"
    
    # Keywords - qualifiers
    LAYOUT = "layout"
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    UNIFORM = "uniform"
    BUFFER = "buffer"
    CONST = "const"
    SHARED = "shared"
    
    RESTRICT = "restrict"
    MEMORY_QUALIFIER = "memory_qualifier"  # writeonly, readonly
    
    INVARIANT = "invariant"
    INTERPOLATION = "interpolation"  # flat, smooth, noperspective
    SAMPLING = "sampling"  # centroid, sample
    PRECISION = "precision"
    PRECISION_QUALIFIER = "precision_qualifier"  # lowp, mediump, highp
    
    # Keywords - control flow
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"
    DISCARD = "discard"
    
    IF = "if"
    ELSE = "else"
    SWITCH = "switch"
    CASE = "case"
    DEFAULT = "default"
    WHILE = "while"
    DO = "do"
    FOR = "for"
    
    # Keywords - types
    VOID = "void"
    STRUCT = "struct"
    TYPE_NAME = "type_name"
    
    # Operators - assignment
    ASSIGN = "="
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="
    MOD_ASSIGN = "%="
    LEFT_SHIFT_ASSIGN = "<<="
    RIGHT_SHIFT_ASSIGN = ">>="
    AND_ASSIGN = "&="
    XOR_ASSIGN = "^="
    OR_ASSIGN = "|="
    
    # Operators - increment/decrement
    INCREMENT = "++"
    DECREMENT = "--"
    
    # Operators - logical
    LOGICAL_OR = "||"
    LOGICAL_AND = "&&"
    LOGICAL_XOR = "^^"
    
    # Operators - comparison
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "=="
    NOT_EQUAL = "!="
    
    # Operators - bitwise
    LEFT_SHIFT = "<<"
    RIGHT_SHIFT = ">>"
    
    # Delimiters
    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    LEFT_BRACKET = "["
    RIGHT_BRACKET = "]"
    LEFT_ANGLE = "<"
    RIGHT_ANGLE = ">"
    
    # Punctuation
    COMMA = ","
    SEMICOLON = ";"
    COLON = ":"
    DOT = "."
    BANG = "!"
    DASH = "-"
    TILDE = "~"
    PLUS = "+"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    VERTICAL_BAR = "|"
    CARET = "^"
    AMPERSAND = "&"
    QUESTION = "?"


@dataclass(frozen=True, slots=True)
class Token:
    """A token with its value and source location.
    
    Attributes:
        value: The token type (from TokenValue enum).
        meta: Source span where this token appears.
        data: Optional associated data for tokens that carry values:
            - For IDENTIFIER: string name
            - For FLOAT_CONSTANT: Float object
            - For INT_CONSTANT: Integer object
            - For BOOL_CONSTANT: bool value
            - For TYPE_NAME: Type object
            - For INTERPOLATION: Interpolation enum
            - For SAMPLING: Sampling enum
            - For PRECISION_QUALIFIER: Precision enum
            - For MEMORY_QUALIFIER: StorageAccess flags
    """
    
    value: TokenValue
    meta: Span
    data: object = None
    
    @classmethod
    def identifier(cls, name: str, meta: Span) -> "Token":
        """Create an identifier token."""
        return cls(value=TokenValue.IDENTIFIER, meta=meta, data=name)
    
    @classmethod
    def float_constant(cls, float_val: Float, meta: Span) -> "Token":
        """Create a float constant token."""
        return cls(value=TokenValue.FLOAT_CONSTANT, meta=meta, data=float_val)
    
    @classmethod
    def int_constant(cls, int_val: Integer, meta: Span) -> "Token":
        """Create an integer constant token."""
        return cls(value=TokenValue.INT_CONSTANT, meta=meta, data=int_val)
    
    @classmethod
    def bool_constant(cls, bool_val: bool, meta: Span) -> "Token":
        """Create a boolean constant token."""
        return cls(value=TokenValue.BOOL_CONSTANT, meta=meta, data=bool_val)
    
    @classmethod
    def type_name(cls, type_obj: Type, meta: Span) -> "Token":
        """Create a type name token."""
        return cls(value=TokenValue.TYPE_NAME, meta=meta, data=type_obj)
    
    @classmethod
    def interpolation(cls, interp: Interpolation, meta: Span) -> "Token":
        """Create an interpolation qualifier token."""
        return cls(value=TokenValue.INTERPOLATION, meta=meta, data=interp)
    
    @classmethod
    def sampling(cls, samp: Sampling, meta: Span) -> "Token":
        """Create a sampling qualifier token."""
        return cls(value=TokenValue.SAMPLING, meta=meta, data=samp)
    
    @classmethod
    def precision_qualifier(cls, prec: Precision, meta: Span) -> "Token":
        """Create a precision qualifier token."""
        return cls(value=TokenValue.PRECISION_QUALIFIER, meta=meta, data=prec)
    
    @classmethod
    def memory_qualifier(cls, access: StorageAccess, meta: Span) -> "Token":
        """Create a memory qualifier token."""
        return cls(value=TokenValue.MEMORY_QUALIFIER, meta=meta, data=access)
    
    @classmethod
    def simple(cls, value: TokenValue, meta: Span) -> "Token":
        """Create a simple token without associated data."""
        return cls(value=value, meta=meta, data=None)


class DirectiveKind(Enum):
    """Type of preprocessor directive."""
    VERSION = "version"
    EXTENSION = "extension"
    PRAGMA = "pragma"


@dataclass(frozen=True, slots=True)
class Directive:
    """A preprocessor directive.
    
    Attributes:
        kind: The type of directive.
        tokens: Raw preprocessor tokens for this directive.
        is_first_directive: For VERSION, whether this is the first directive.
    """
    
    kind: DirectiveKind
    tokens: list[object]  # List of preprocessor tokens
    is_first_directive: bool = False
    
    @classmethod
    def version(cls, tokens: list[object], is_first: bool) -> "Directive":
        """Create a version directive."""
        return cls(kind=DirectiveKind.VERSION, tokens=tokens, is_first_directive=is_first)
    
    @classmethod
    def extension(cls, tokens: list[object]) -> "Directive":
        """Create an extension directive."""
        return cls(kind=DirectiveKind.EXTENSION, tokens=tokens)
    
    @classmethod
    def pragma(cls, tokens: list[object]) -> "Directive":
        """Create a pragma directive."""
        return cls(kind=DirectiveKind.PRAGMA, tokens=tokens)
