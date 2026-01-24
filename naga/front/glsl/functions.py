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
        # `Expression::As` doesn't support matrix width casts so we need to do some extra work for casts.
        # This is handled in the lowerer by decomposing matrix casts into column/component operations.
        
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
            
            # Explicit casts
            ConversionRule("float", "int", ConversionType.EXPLICIT, 1),
            ConversionRule("float", "uint", ConversionType.EXPLICIT, 1),
            ConversionRule("int", "uint", ConversionType.EXPLICIT, 0),
            ConversionRule("uint", "int", ConversionType.EXPLICIT, 0),
        ])
    
    def force_conversion(self, ctx: Any, value: Any, target_type: str, meta: Any) -> Optional[Any]:
        """
        Force conversion of a value to a target type.
        """
        from ...ir import Expression, ExpressionType, ScalarKind, Scalar
        
        # Get current type
        current_type = ctx.get_expression_type(value)
        if current_type is None:
            return value

        # Mapping target_type string to IR scalar kind
        # This is a bit simplified
        mapping = {
            "float": ScalarKind.FLOAT,
            "int": ScalarKind.SINT,
            "uint": ScalarKind.UINT,
            "bool": ScalarKind.BOOL,
        }
        
        kind = mapping.get(target_type)
        if kind is not None:
             # Create CAST expression
             # In Naga IR, As is used for bitcasting, and Cast for type conversion (e.g. float to int)
             expr = Expression(
                 type=ExpressionType.AS,
                 as_kind=kind,
                 as_expr=value,
                 as_convert=4 # Default width
             )
             return ctx.add_expression(expr)
        
        return value
    
    def resolve_type(self, ctx: Any, value: Any, meta: Any = None) -> Optional[Any]:
        """
        Resolve the type of a value handle.
        """
        return ctx.get_expression_type(value)
    
    def add_expression(self, ctx: Any, expr: Any, meta: Any = None) -> Optional[Any]:
        """
        Add an expression to the module.
        """
        from ...ir import Expression
        if isinstance(expr, Expression):
            return ctx.add_expression(expr)
        return expr
    
    def check_conversion_compatibility(self, from_type: str, to_type: str) -> Optional[ConversionRule]:
        """
        Check if conversion between types is possible.
        
        Args:
            from_type: Source type
            to_type: Target type
            
        Returns:
            Conversion rule or None if not compatible
        """
        # This finds the best conversion rule for the given types.
        # It also handles implicit conversions for vectors and matrices by checking their component types.
        
        # 1. Check direct rules
        compatible_rules = [
            rule for rule in self.conversion_rules 
            if rule.from_type == from_type and rule.to_type == to_type
        ]
        
        if compatible_rules:
            return min(compatible_rules, key=lambda r: r.cost)
            
        # 2. Check for vector/matrix matching if types are not identical strings
        if from_type != to_type:
             import re
             # Standard GLSL allows component-wise implicit conversions for vectors of same size
             vec_re = r"([biu]?)vec([234])"
             from_m = re.match(vec_re, from_type)
             to_m = re.match(vec_re, to_type)
             
             if from_m and to_m:
                  f_prefix, f_size = from_m.groups()
                  t_prefix, t_size = to_m.groups()
                  if f_size == t_size:
                       # Map prefixes to scalar types
                       prefix_map = {"": "float", "i": "int", "u": "uint", "b": "bool"}
                       f_scalar = prefix_map.get(f_prefix)
                       t_scalar = prefix_map.get(t_prefix)
                       if f_scalar and t_scalar:
                            return self.check_conversion_compatibility(f_scalar, t_scalar)

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
        # Better error reporting:
        # Currently the code doesn't walk the array if the size isn't known at compile time 
        # and just lets validation catch it.
        
        # Proper error reporting for:
        # - Unknown array sizes
        # - Type mismatches
        # - Invalid conversions
        # - Missing function overloads
        
        # For now, just call the provided function and handle errors through the frontend
        try:
            return func(name, binding, pointer, base)
        except Exception as e:
            # If we were passed a frontend, we could report the error properly
            print(f"Error in arg_type_walker: {e}")
            return None
    
    def handle_matrix_constructor(self, ctx: Any, meta: Any, ty: Any, components: List[Any]) -> Optional[Any]:
        """
        Handle matrix constructor expressions.
        """
        # Matrices can be constructed from:
        # 1. A single scalar (diagonal matrix)
        # 2. A list of components (must match matrix size)
        # 3. Another matrix (resize)
        from ...ir import Expression, ExpressionType
        
        # For now, we only support full-component construction via COMPOSE
        expr = Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=ty,
            compose_components=components
        )
        return ctx.add_expression(expr)
    
    def validate_function_call(self, frontend: Any, func_name: str, args: List[Any], overloads: List[Any]) -> Optional[Any]:
        """
        Validate a function call against available overloads.
        """
        # This is already partially handled in FunctionParser.resolve_function_call
        # We can implement more rigorous validation here if needed.
        for overload in overloads:
             if len(overload.get('parameters', [])) == len(args):
                  return overload
        return None
    
    def get_function_overloads(self, frontend: Any, func_name: str) -> List[Any]:
        """
        Get all overloads of a function.
        """
        # Return all available overloads from the frontend's function parser
        if hasattr(frontend, "function_parser"):
            return frontend.function_parser.function_signatures.get(func_name, [])
        
        return []
    
    def report_type_error(self, frontend: Any, expected: Any, actual: Any, meta: Any) -> None:
        """
        Report a type error with better details.
        """
        expected_str = str(expected)
        actual_str = str(actual)
        error_msg = f"Type error: expected '{expected_str}', got '{actual_str}'"
        
        # Report error through the frontend if available
        if hasattr(frontend, "add_error"):
             frontend.add_error(error_msg, meta)
        else:
             print(f"Error: {error_msg} at {meta}")
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