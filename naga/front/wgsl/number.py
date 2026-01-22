"""
WGSL number literal parsing.

Translated from wgpu-trunk/naga/src/front/wgsl/parse/number.rs

This module handles parsing of numeric literals in WGSL including
integers, floats, hexadecimal, and scientific notation.
"""

from typing import Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class NumberType(Enum):
    """Type of number literal."""
    I32 = "i32"
    U32 = "u32"
    F32 = "f32"
    F16 = "f16"
    ABSTRACT_INT = "abstract_int"
    ABSTRACT_FLOAT = "abstract_float"


@dataclass
class Number:
    """
    Parsed number literal.
    
    Attributes:
        value: The numeric value
        type_: The type of the number
        span: Source location
    """
    value: Union[int, float]
    type_: NumberType
    span: Tuple[int, int]


def parse_number(text: str, span: Tuple[int, int]) -> Number:
    """
    Parse a WGSL number literal.
    
    Supports:
    - Decimal integers: 123, 123i, 123u
    - Hexadecimal integers: 0x1a2b, 0x1a2bi, 0x1a2bu
    - Decimal floats: 1.5, 1.5f, 1.5h
    - Scientific notation: 1.5e10, 1.5e-10
    - Abstract numbers (no suffix)
    
    Args:
        text: The number literal text
        span: Source location
        
    Returns:
        Parsed Number
        
    Raises:
        ParseError: If the number is malformed
    """
    from .error import ParseError
    
    # Remove underscores (allowed as separators)
    text = text.replace('_', '')
    
    # Check for type suffix
    suffix = None
    if text.endswith('i'):
        suffix = 'i'
        text = text[:-1]
    elif text.endswith('u'):
        suffix = 'u'
        text = text[:-1]
    elif text.endswith('f'):
        suffix = 'f'
        text = text[:-1]
    elif text.endswith('h'):
        suffix = 'h'
        text = text[:-1]
    
    # Check for hexadecimal
    if text.startswith('0x') or text.startswith('0X'):
        return _parse_hex_number(text, suffix, span)
    
    # Check if it's a float (contains . or e/E)
    if '.' in text or 'e' in text.lower():
        return _parse_float_number(text, suffix, span)
    
    # Parse as integer
    return _parse_int_number(text, suffix, span)


def _parse_hex_number(text: str, suffix: Optional[str], span: Tuple[int, int]) -> Number:
    """Parse hexadecimal number."""
    from .error import ParseError
    
    try:
        # Remove 0x prefix
        hex_text = text[2:]
        value = int(hex_text, 16)
        
        if suffix == 'i':
            # Check if it fits in i32
            if value > 0x7FFFFFFF:
                raise ParseError(
                    message="hexadecimal integer literal is too large for i32",
                    labels=[(span[0], span[1], "")],
                    notes=[]
                )
            return Number(value, NumberType.I32, span)
        elif suffix == 'u':
            # Check if it fits in u32
            if value > 0xFFFFFFFF:
                raise ParseError(
                    message="hexadecimal integer literal is too large for u32",
                    labels=[(span[0], span[1], "")],
                    notes=[]
                )
            return Number(value, NumberType.U32, span)
        elif suffix in ('f', 'h'):
            raise ParseError(
                message="hexadecimal float literals are not supported",
                labels=[(span[0], span[1], "")],
                notes=[]
            )
        else:
            # Abstract int
            return Number(value, NumberType.ABSTRACT_INT, span)
    except ValueError:
        raise ParseError(
            message=f"invalid hexadecimal number: {text}",
            labels=[(span[0], span[1], "")],
            notes=[]
        )


def _parse_int_number(text: str, suffix: Optional[str], span: Tuple[int, int]) -> Number:
    """Parse decimal integer."""
    from .error import ParseError
    
    try:
        value = int(text)
        
        if suffix == 'i':
            # Check if it fits in i32
            if value < -2147483648 or value > 2147483647:
                raise ParseError(
                    message="integer literal is out of range for i32",
                    labels=[(span[0], span[1], "")],
                    notes=[]
                )
            return Number(value, NumberType.I32, span)
        elif suffix == 'u':
            # Check if it fits in u32
            if value < 0 or value > 4294967295:
                raise ParseError(
                    message="integer literal is out of range for u32",
                    labels=[(span[0], span[1], "")],
                    notes=[]
                )
            return Number(value, NumberType.U32, span)
        elif suffix in ('f', 'h'):
            # Convert to float
            return _parse_float_number(text, suffix, span)
        else:
            # Abstract int
            return Number(value, NumberType.ABSTRACT_INT, span)
    except ValueError:
        raise ParseError(
            message=f"invalid integer: {text}",
            labels=[(span[0], span[1], "")],
            notes=[]
        )


def _parse_float_number(text: str, suffix: Optional[str], span: Tuple[int, int]) -> Number:
    """Parse floating-point number."""
    from .error import ParseError
    
    try:
        value = float(text)
        
        if suffix == 'f':
            return Number(value, NumberType.F32, span)
        elif suffix == 'h':
            # f16 - check if extension is enabled
            # TODO: Validate f16 range
            return Number(value, NumberType.F16, span)
        elif suffix in ('i', 'u'):
            raise ParseError(
                message="float literal cannot have integer suffix",
                labels=[(span[0], span[1], "")],
                notes=[]
            )
        else:
            # Abstract float
            return Number(value, NumberType.ABSTRACT_FLOAT, span)
    except ValueError:
        raise ParseError(
            message=f"invalid float: {text}",
            labels=[(span[0], span[1], "")],
            notes=[]
        )


def number_to_literal(number: Number) -> str:
    """
    Convert a Number to a literal string representation.
    
    Args:
        number: The number to convert
        
    Returns:
        String representation suitable for code generation
    """
    if number.type_ in (NumberType.I32, NumberType.ABSTRACT_INT):
        return str(int(number.value))
    elif number.type_ == NumberType.U32:
        return f"{int(number.value)}u"
    elif number.type_ == NumberType.F32:
        return f"{number.value}f"
    elif number.type_ == NumberType.F16:
        return f"{number.value}h"
    elif number.type_ == NumberType.ABSTRACT_FLOAT:
        return str(number.value)
    else:
        return str(number.value)
