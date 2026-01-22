"""GLSL frontend error types.

This module is a Python translation of `wgpu-trunk/naga/src/front/glsl/error.rs`.
It provides error types and error reporting for GLSL parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from naga.error import replace_control_chars
from naga.span import Span, SourceLocation


def join_with_comma(expected_list: list["ExpectedToken"]) -> str:
    """Join expected tokens with commas and 'or' before the last item."""
    if not expected_list:
        return ""
    if len(expected_list) == 1:
        return str(expected_list[0])
    
    parts = [str(token) for token in expected_list]
    if len(parts) == 2:
        return f"{parts[0]} or {parts[1]}"
    
    return ", ".join(parts[:-1]) + f" or {parts[-1]}"


class ExpectedTokenType(Enum):
    """Type of expected token."""
    TOKEN = "token"
    TYPE_NAME = "type_name"
    IDENTIFIER = "identifier"
    INT_LITERAL = "int_literal"
    FLOAT_LITERAL = "float_literal"
    BOOL_LITERAL = "bool_literal"
    EOF = "eof"


@dataclass(frozen=True, slots=True)
class ExpectedToken:
    """One of the expected tokens returned in InvalidToken errors."""
    
    type: ExpectedTokenType
    token_value: Optional[object] = None  # TokenValue if type is TOKEN
    
    @classmethod
    def token(cls, token_value: object) -> "ExpectedToken":
        """Create an ExpectedToken for a specific token."""
        return cls(type=ExpectedTokenType.TOKEN, token_value=token_value)
    
    @classmethod
    def type_name(cls) -> "ExpectedToken":
        """Create an ExpectedToken for a type name."""
        return cls(type=ExpectedTokenType.TYPE_NAME)
    
    @classmethod
    def identifier(cls) -> "ExpectedToken":
        """Create an ExpectedToken for an identifier."""
        return cls(type=ExpectedTokenType.IDENTIFIER)
    
    @classmethod
    def int_literal(cls) -> "ExpectedToken":
        """Create an ExpectedToken for an integer literal."""
        return cls(type=ExpectedTokenType.INT_LITERAL)
    
    @classmethod
    def float_literal(cls) -> "ExpectedToken":
        """Create an ExpectedToken for a float literal."""
        return cls(type=ExpectedTokenType.FLOAT_LITERAL)
    
    @classmethod
    def bool_literal(cls) -> "ExpectedToken":
        """Create an ExpectedToken for a boolean literal."""
        return cls(type=ExpectedTokenType.BOOL_LITERAL)
    
    @classmethod
    def eof(cls) -> "ExpectedToken":
        """Create an ExpectedToken for end of file."""
        return cls(type=ExpectedTokenType.EOF)
    
    def __str__(self) -> str:
        match self.type:
            case ExpectedTokenType.TOKEN:
                return f"{self.token_value!r}"
            case ExpectedTokenType.TYPE_NAME:
                return "a type"
            case ExpectedTokenType.IDENTIFIER:
                return "identifier"
            case ExpectedTokenType.INT_LITERAL:
                return "integer literal"
            case ExpectedTokenType.FLOAT_LITERAL:
                return "float literal"
            case ExpectedTokenType.BOOL_LITERAL:
                return "bool literal"
            case ExpectedTokenType.EOF:
                return "end of file"


class ErrorKind(Enum):
    """Information about the cause of an error."""
    END_OF_FILE = "end_of_file"
    INVALID_PROFILE = "invalid_profile"
    INVALID_VERSION = "invalid_version"
    INVALID_TOKEN = "invalid_token"
    NOT_IMPLEMENTED = "not_implemented"
    UNKNOWN_VARIABLE = "unknown_variable"
    UNKNOWN_TYPE = "unknown_type"
    UNKNOWN_FIELD = "unknown_field"
    UNKNOWN_LAYOUT_QUALIFIER = "unknown_layout_qualifier"
    UNSUPPORTED_MATRIX_TWO_ROWS_STD140 = "unsupported_matrix_two_rows_std140"
    UNSUPPORTED_F16_MATRIX_STD140 = "unsupported_f16_matrix_std140"
    VARIABLE_ALREADY_DECLARED = "variable_already_declared"
    SEMANTIC_ERROR = "semantic_error"
    PREPROCESSOR_ERROR = "preprocessor_error"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class Error:
    """Error returned during shader parsing.
    
    Attributes:
        kind: The type of error that occurred.
        message: Human-readable error message.
        meta: Source code span where the error occurred.
        expected_tokens: For InvalidToken errors, list of expected tokens.
    """
    
    kind: ErrorKind
    message: str
    meta: Span
    expected_tokens: Optional[list[ExpectedToken]] = None
    
    @classmethod
    def end_of_file(cls, meta: Span) -> "Error":
        """Create an unexpected end of file error."""
        return cls(
            kind=ErrorKind.END_OF_FILE,
            message="Unexpected end of file",
            meta=meta,
        )
    
    @classmethod
    def invalid_profile(cls, profile: str, meta: Span) -> "Error":
        """Create an invalid profile error."""
        return cls(
            kind=ErrorKind.INVALID_PROFILE,
            message=f"Invalid profile: {profile}",
            meta=meta,
        )
    
    @classmethod
    def invalid_version(cls, version: int, meta: Span) -> "Error":
        """Create an invalid version error."""
        return cls(
            kind=ErrorKind.INVALID_VERSION,
            message=f"Invalid version: {version}",
            meta=meta,
        )
    
    @classmethod
    def invalid_token(
        cls,
        found: object,
        expected: list[ExpectedToken],
        meta: Span,
    ) -> "Error":
        """Create an invalid token error."""
        expected_str = join_with_comma(expected)
        return cls(
            kind=ErrorKind.INVALID_TOKEN,
            message=f"Expected {expected_str}, found {found!r}",
            meta=meta,
            expected_tokens=expected,
        )
    
    @classmethod
    def not_implemented(cls, feature: str, meta: Span) -> "Error":
        """Create a not implemented error."""
        return cls(
            kind=ErrorKind.NOT_IMPLEMENTED,
            message=f"Not implemented: {feature}",
            meta=meta,
        )
    
    @classmethod
    def unknown_variable(cls, name: str, meta: Span) -> "Error":
        """Create an unknown variable error."""
        return cls(
            kind=ErrorKind.UNKNOWN_VARIABLE,
            message=f"Unknown variable: {name}",
            meta=meta,
        )
    
    @classmethod
    def unknown_type(cls, name: str, meta: Span) -> "Error":
        """Create an unknown type error."""
        return cls(
            kind=ErrorKind.UNKNOWN_TYPE,
            message=f"Unknown type: {name}",
            meta=meta,
        )
    
    @classmethod
    def unknown_field(cls, name: str, meta: Span) -> "Error":
        """Create an unknown field error."""
        return cls(
            kind=ErrorKind.UNKNOWN_FIELD,
            message=f"Unknown field: {name}",
            meta=meta,
        )
    
    @classmethod
    def unknown_layout_qualifier(cls, name: str, meta: Span) -> "Error":
        """Create an unknown layout qualifier error."""
        return cls(
            kind=ErrorKind.UNKNOWN_LAYOUT_QUALIFIER,
            message=f"Unknown layout qualifier: {name}",
            meta=meta,
        )
    
    @classmethod
    def unsupported_matrix_two_rows_std140(cls, columns: int, meta: Span) -> "Error":
        """Create an unsupported matCx2 in std140 error."""
        return cls(
            kind=ErrorKind.UNSUPPORTED_MATRIX_TWO_ROWS_STD140,
            message=(
                f"unsupported matrix of the form matCx2 (in this case mat{columns}x2) "
                f"in std140 block layout. See https://github.com/gfx-rs/wgpu/issues/4375"
            ),
            meta=meta,
        )
    
    @classmethod
    def unsupported_f16_matrix_std140(cls, columns: int, rows: int, meta: Span) -> "Error":
        """Create an unsupported f16matCxR in std140 error."""
        return cls(
            kind=ErrorKind.UNSUPPORTED_F16_MATRIX_STD140,
            message=(
                f"unsupported matrix of the form f16matCxR (in this case f16mat{columns}x{rows}) "
                f"in std140 block layout. See https://github.com/gfx-rs/wgpu/issues/4375"
            ),
            meta=meta,
        )
    
    @classmethod
    def variable_already_declared(cls, name: str, meta: Span) -> "Error":
        """Create a variable already declared error."""
        return cls(
            kind=ErrorKind.VARIABLE_ALREADY_DECLARED,
            message=f"Variable already declared: {name}",
            meta=meta,
        )
    
    @classmethod
    def semantic_error(cls, message: str, meta: Span) -> "Error":
        """Create a semantic error."""
        return cls(
            kind=ErrorKind.SEMANTIC_ERROR,
            message=message,
            meta=meta,
        )
    
    @classmethod
    def preprocessor_error(cls, error: object, meta: Span) -> "Error":
        """Create a preprocessor error."""
        return cls(
            kind=ErrorKind.PREPROCESSOR_ERROR,
            message=f"{error!r}",
            meta=meta,
        )
    
    @classmethod
    def internal_error(cls, message: str, meta: Span) -> "Error":
        """Create an internal error."""
        return cls(
            kind=ErrorKind.INTERNAL_ERROR,
            message=f"Internal error: {message}",
            meta=meta,
        )
    
    def location(self, source: str) -> Optional[SourceLocation]:
        """Returns a SourceLocation for the error message."""
        if self.meta.is_defined():
            return self.meta.location(source)
        return None
    
    def __str__(self) -> str:
        return self.message


@dataclass
class ParseErrors:
    """A collection of errors returned during shader parsing."""
    
    errors: list[Error]
    
    def emit_to_string(self, source: str, path: str = "glsl") -> str:
        """Format errors as a string with source context."""
        if not self.errors:
            return ""
        
        lines = []
        source_lines = replace_control_chars(source).split("\n")
        
        for err in self.errors:
            lines.append(f"Error in {path}:")
            lines.append(f"  {err.message}")
            
            if err.meta.is_defined():
                location = err.location(source)
                if location:
                    lines.append(f"  at line {location.line_number}, column {location.line_position}")
                    
                    # Show the source line
                    if 0 <= location.line_number - 1 < len(source_lines):
                        line = source_lines[location.line_number - 1]
                        lines.append(f"  {line}")
                        
                        # Show pointer to error location
                        if location.line_position > 0:
                            pointer = " " * (location.line_position - 1) + "^"
                            if location.length > 1:
                                pointer += "~" * (min(location.length, len(line)) - 1)
                            lines.append(f"  {pointer}")
            
            lines.append("")  # Empty line between errors
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"ParseErrors({len(self.errors)} error(s))"
