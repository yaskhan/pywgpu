"""
GLSL function handling and type conversion utilities.

This module provides utilities for handling function calls, type conversions,
and better error reporting in GLSL parsing.
"""

from typing import Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass


class ConversionType(Enum):
    """Types of type conversions."""
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    PRECISION = "precision"
    ARRAY_TO_POINTER = "array_to_pointer"


@dataclass
class ConversionRule:
    """Rule for type conversion."""
    from_type: str
    to_type: str
    conversion_type: ConversionType
    cost: int  # Lower cost means better conversion


class FunctionHandler:
    """Handler for GLSL function calls and type conversions."""
    
    def __init__(self):
        self.conversion_rules: List[ConversionRule] = []
        self._initialize_conversion_rules()
    
    def _initialize_conversion_rules(self) -> None:
        """
        Initialize built-in conversion rules.
        
        Note: Matrix width casts require special handling.
        The Expression::As operation in Naga IR doesn't support matrix width
        casts, so conversions between matrices of different dimensions need
        to be decomposed into component-wise operations.
        """
        # TODO: casts
        # `Expression::As` doesn't support matrix width casts so we need to do some extra work for casts
        
        # Basic type conversion rules
        self.conversion_rules.extend([
            # Integer to float conversions
            ConversionRule("int", "float", ConversionType.IMPLICIT, 1),
            ConversionRule("uint", "float", ConversionType.IMPLICIT, 1),
            
            # Boolean conversions
            ConversionRule("bool", "int", ConversionType.IMPLICIT, 2),
            ConversionRule("bool", "float", ConversionType.IMPLICIT, 2),
            
            # Precision conversions (lowp -> mediump -> highp)
            ConversionRule("lowp", "mediump", ConversionType.PRECISION, 1),
            ConversionRule("lowp", "highp", ConversionType.PRECISION, 2),
            ConversionRule("mediump", "highp", ConversionType.PRECISION, 1),
            
            # Widening conversions
            ConversionRule("int8", "int16", ConversionType.IMPLICIT, 1),
            ConversionRule("int16", "int32", ConversionType.IMPLICIT, 1),
            ConversionRule("uint8", "uint16", ConversionType.IMPLICIT, 1),
            ConversionRule("uint16", "uint32", ConversionType.IMPLICIT, 1),
        ])
    
    def force_conversion(self, value: Any, target_type: str, meta: Any) -> Optional[Any]:
        """
        Force conversion of a value to a target type.
        
        Args:
            value: Value to convert
            target_type: Target type name
            meta: Metadata about the conversion location
            
        Returns:
            Converted value or None if conversion fails
        """
        # TODO: Implement forced conversion logic
        # This should handle:
        # - Matrix dimension conversions (mat2x3 -> mat3x3, etc.)
        # - Vector component conversions
        # - Scalar type conversions
        # - Precision conversions
        
        return None
    
    def resolve_type(self, value: Any, meta: Any) -> Optional[Any]:
        """
        Resolve the type of a value.
        
        Args:
            value: Value to get type for
            meta: Metadata about the value
            
        Returns:
            Type information or None
        """
        # TODO: Implement type resolution
        # This should:
        # 1. Check if value is a literal and return its type
        # 2. Check if value is a variable and return its declared type
        # 3. Check if value is an expression and compute its type
        # 4. Handle type inference for complex expressions
        
        return None
    
    def add_expression(self, expr: Any, meta: Any) -> Optional[Any]:
        """
        Add an expression to the module.
        
        Args:
            expr: Expression to add
            meta: Metadata about the expression
            
        Returns:
            Expression handle or None
        """
        # TODO: Implement expression addition
        # This should handle different expression types:
        # - Literals
        # - Variables
        # - Binary operations
        # - Unary operations
        # - Function calls
        # - Type constructors
        
        return None
    
    def check_conversion_compatibility(self, from_type: str, to_type: str) -> Optional[ConversionRule]:
        """
        Check if conversion between types is possible.
        
        Args:
            from_type: Source type
            to_type: Target type
            
        Returns:
            Conversion rule or None if not compatible
        """
        # TODO: Implement conversion compatibility checking
        # This should find the best conversion rule for the given types
        
        compatible_rules = [
            rule for rule in self.conversion_rules 
            if rule.from_type == from_type and rule.to_type == to_type
        ]
        
        if compatible_rules:
            return min(compatible_rules, key=lambda r: r.cost)
        return None
    
    def arg_type_walker(self, name: str, binding: Any, pointer: Any, base: Any, func: Any) -> Optional[Any]:
        """
        Walk through argument types for function calls.
        
        Args:
            name: Argument name
            binding: Argument binding information
            pointer: Argument pointer
            base: Base type
            func: Function to call for each type
            
        Returns:
            Result of walking or None
        """
        # TODO: Better error reporting (строка 1415)
        # The original comment mentions that currently the code doesn't walk
        # the array if the size isn't known at compile time and just lets
        # validation catch it. We need better error reporting here.
        
        # TODO: Implement proper error reporting for:
        # - Unknown array sizes
        # - Type mismatches
        # - Invalid conversions
        # - Missing function overloads
        
        # For now, just call the provided function
        try:
            return func(name, binding, pointer, base)
        except Exception as e:
            # TODO: Add proper error reporting here
            # Should report detailed information about:
            # - What went wrong
            # - Where in the source code
            # - Possible solutions
            # - Expected vs actual types
            print(f"Error in arg_type_walker: {e}")
            return None
    
    def handle_matrix_constructor(self, ctx: Any, meta: Any) -> Optional[Any]:
        """
        Handle matrix constructor expressions.
        
        Args:
            ctx: Parsing context
            meta: Metadata about the constructor
            
        Returns:
            Matrix constructor expression or None
        """
        # TODO: Implement matrix constructor handling
        # This should handle:
        # - Matrix from scalar (diagonal initialization)
        # - Matrix from vector (row-wise or column-wise)
        # - Matrix from matrix (conversion between dimensions)
        # - Matrix from mixed types
        
        return None
    
    def validate_function_call(self, func_name: str, args: List[Any]) -> Optional[Any]:
        """
        Validate a function call against available overloads.
        
        Args:
            func_name: Function name
            args: Function arguments
            
        Returns:
            Validated function call or None
        """
        # TODO: Implement function call validation
        # This should:
        # 1. Find all overloads of the function
        # 2. Check argument count
        # 3. Check argument types
        # 4. Apply implicit conversions
        # 5. Report errors for invalid calls
        
        return None
    
    def get_function_overloads(self, func_name: str) -> List[Any]:
        """
        Get all overloads of a function.
        
        Args:
            func_name: Function name
            
        Returns:
            List of function overloads
        """
        # TODO: Implement function overload retrieval
        # This should return all available overloads for the given function name
        
        return []
    
    def report_type_error(self, expected: str, actual: str, meta: Any) -> None:
        """
        Report a type error with better details.
        
        Args:
            expected: Expected type
            actual: Actual type
            meta: Metadata about the error location
        """
        # TODO: Better error reporting
        # This should provide detailed error messages that help users understand:
        # 1. What type was expected
        # 2. What type was found
        # 3. Where in the source code the error occurred
        # 4. Possible solutions or conversions
        # 5. Available overloads that might match
        
        error_msg = f"Type error: expected '{expected}', got '{actual}'"
        print(f"Error: {error_msg}")
        # TODO: Add to error list with proper formatting
        # errors.push(Error {
        #     kind: ErrorKind::TypeError(error_msg),
        #     meta: meta,
        # })