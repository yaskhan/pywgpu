"""
Parser for GLSL function declarations and calls.

This module handles parsing of function declarations, definitions, and calls.
"""

from typing import Any, Optional, List, Dict
from enum import Enum


class FunctionKind(Enum):
    """Types of function declarations."""
    DECLARATION = "declaration"
    DEFINITION = "definition"


class ParameterDirection(Enum):
    """Parameter direction qualifiers."""
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    CONST = "const"


class FunctionParser:
    """Parser for GLSL function declarations and calls."""
    
    def __init__(self, lexer: Any):
        self.lexer = lexer
        self.errors: List[str] = []
        self.function_signatures: Dict[str, Any] = {}
    
    def parse_function_declaration(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """
        Parse a function declaration or definition.
        
        Args:
            ctx: Parsing context
            frontend: GLSL parser frontend
            
        Returns:
            Function declaration information or None
        """
        # TODO: Implement function declaration parsing
        # This should handle:
        # - Function return type
        # - Function name
        # - Parameter list
        # - Function body (for definitions)
        # - Function qualifiers
        
        # TODO: Implicit conversions (строка 99)
        # Implement support for implicit type conversions in function calls:
        # - Integer to float conversions
        # - Boolean to integer conversions
        # - Widening conversions (int8 -> int16, etc.)
        # - Precision conversions (lowp -> mediump -> highp)
        
        return None
    
    def parse_parameter_list(self, ctx: Any, frontend: Any) -> List[Any]:
        """
        Parse function parameter list.
        
        Args:
            ctx: Parsing context
            frontend: GLSL parser frontend
            
        Returns:
            List of parameter information
        """
        # TODO: Complete parameter list parsing
        # Should handle:
        # - Parameter type
        # - Parameter name
        # - Parameter qualifiers (in, out, inout, const)
        # - Default values
        # - Array parameters
        
        return []
    
    def parse_function_call(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """
        Parse a function call expression.
        
        Args:
            ctx: Parsing context
            frontend: GLSL parser frontend
            
        Returns:
            Function call information or None
        """
        # TODO: Implement function call parsing
        # This should handle:
        # - Function name resolution
        # - Argument parsing
        # - Type checking of arguments
        # - Implicit conversions of arguments
        
        return None
    
    def resolve_function_call(self, function_name: str, args: List[Any]) -> Optional[Any]:
        """
        Resolve a function call to a specific function overload.
        
        Args:
            function_name: Name of the function to resolve
            args: List of argument expressions
            
        Returns:
            Resolved function information or None
        """
        # TODO: Implement function resolution
        # Should handle:
        # - Overload resolution based on argument types
        # - Implicit conversion application
        # - Error reporting for no matching overload
        # - Ambiguous overload detection
        
        return None
    
    def add_function_overload(self, signature: str, func_info: Any) -> None:
        """
        Add a function overload to the lookup table.
        
        Args:
            signature: Function signature string
            func_info: Function information
        """
        if signature not in self.function_signatures:
            self.function_signatures[signature] = []
        self.function_signatures[signature].append(func_info)
    
    def check_implicit_conversions(self, expected_type: Any, actual_type: Any) -> Optional[Any]:
        """
        Check if an implicit conversion is possible between types.
        
        Args:
            expected_type: Expected type
            actual_type: Actual type
            
        Returns:
            Conversion function or None
        """
        # TODO: casts and implicit conversions
        # Implement type conversion checking:
        # - Integer to float conversions
        # - Boolean to integer conversions
        # - Precision conversions
        # - Array to pointer conversions
        # - Struct conversions
        
        # TODO: Better error reporting (строка 1415)
        # Provide detailed error messages for:
        # - No valid conversion found
        # - Ambiguous conversions
        # - Precision loss in conversions
        # - Invalid conversion combinations
        
        return None