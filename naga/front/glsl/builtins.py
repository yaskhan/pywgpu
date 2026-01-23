"""
GLSL builtin function definitions and implementations.

This module provides builtin function definitions for GLSL shaders,
including texture functions, math functions, and other builtins.
"""

from typing import Any, Optional, List, Dict, Callable
from enum import Enum
from dataclasses import dataclass


class BuiltinKind(Enum):
    """Types of builtin functions."""
    TEXTURE = "texture"
    MATH = "math"
    GEOMETRY = "geometry"
    FRAGMENT = "fragment"
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass
class BuiltinFunction:
    """Information about a builtin function."""
    name: str
    kind: BuiltinKind
    parameters: List[Any]
    return_type: Any
    description: str


class Builtins:
    """GLSL builtin function definitions."""
    
    def __init__(self):
        self.builtin_functions: Dict[str, List[BuiltinFunction]] = {}
        self._initialize_builtins()
    
    def _initialize_builtins(self) -> None:
        """Initialize all builtin functions."""
        self._initialize_texture_functions()
        self._initialize_math_functions()
        self._initialize_vector_functions()
        self._initialize_matrix_functions()
    
    def _initialize_texture_functions(self) -> None:
        """Initialize texture builtin functions."""
        texture_functions = [
            # Texture sampling functions
            "texture", "textureOffset", "textureProj", "textureLod", 
            "textureLodOffset", "textureGrad", "textureGradOffset",
            "textureGather", "textureGatherOffset", "textureGatherOffsets",
            "textureSize", "texelFetch", "texelFetchOffset",
            # Image functions
            "imageLoad", "imageStore", "imageSize",
            # Query functions
            "textureQueryLevels", "textureSamples",
        ]
        
        for func_name in texture_functions:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
            
            # TODO: glsl supports using bias with depth samplers but naga doesn't
            # GLSL allows using bias parameters with depth/shadow samplers, but Naga's IR
            # doesn't currently support this combination. When bias=true and shadow=true,
            # we skip adding that variation.
            
            # Add variations for different texture types
            self._add_texture_variations(func_name)
    
    def _initialize_math_functions(self) -> None:
        """Initialize math builtin functions."""
        math_functions = [
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "exp", "exp2", "log", "log2", "log10", "sqrt", "inversesqrt",
            "abs", "sign", "floor", "ceil", "fract", "min", "max", "clamp",
            "mix", "step", "smoothstep", "length", "distance", "dot", "cross",
            "normalize", "faceforward", "reflect", "refract",
            "pow", "mod", "fmod", "remainder",
        ]
        
        for func_name in math_functions:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
            
            # Add scalar and vector variations
            self._add_math_variations(func_name)
        
        # TODO: https://github.com/gfx-rs/naga/issues/2526
        # "modf" | "frexp" => { ... }
        # These functions split floats into parts (modf: integer and fractional,
        # frexp: mantissa and exponent) but require multiple return values which
        # needs special handling. See https://github.com/gfx-rs/naga/issues/2526
        self._add_unimplemented_functions(["modf", "frexp"])
    
    def _initialize_vector_functions(self) -> None:
        """Initialize vector builtin functions."""
        vector_functions = [
            "lessThan", "greaterThan", "lessThanEqual", "greaterThanEqual",
            "equal", "notEqual", "any", "all", "not",
            " radians", "degrees",
        ]
        
        for func_name in vector_functions:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
    
    def _initialize_matrix_functions(self) -> None:
        """Initialize matrix builtin functions."""
        matrix_functions = [
            "matrixCompMult", "transpose", "determinant", "inverse",
        ]
        
        for func_name in matrix_functions:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
    
    def _add_texture_variations(self, func_name: str) -> None:
        """Add texture function variations."""
        # TODO: Add support for different texture dimensions
        # - 1D textures (sampler1D, sampler1DArray)
        # - 2D textures (sampler2D, sampler2DArray, sampler2DMS, sampler2DMSArray)
        # - 3D textures (sampler3D)
        # - Cube textures (samplerCube, samplerCubeArray)
        # - Shadow samplers (sampler2DShadow, samplerCubeShadow, etc.)
        
        # TODO: Add support for multisampled textures
        # - sampler2DMS, sampler2DMSArray
        # - Different sample counts
        
        pass
    
    def _add_math_variations(self, func_name: str) -> None:
        """Add math function variations for different types."""
        # Common scalar and vector types
        types = ["float", "vec2", "vec3", "vec4"]
        
        # Single argument functions (sin, cos, etc.)
        if func_name in ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                         "exp", "exp2", "log", "log2", "log10", "sqrt", "inversesqrt", "abs", "sign", "floor", "ceil", "fract",
                         "normalize", "length"]:
            for ty in types:
                self.add_builtin_overload(func_name, BuiltinFunction(
                    name=func_name, kind=BuiltinKind.MATH, parameters=[ty], return_type=ty if func_name not in ["length"] else "float",
                    description=f"{func_name} overload"
                ))

        # Two argument functions (min, max, pow, mod, distance, dot, cross)
        elif func_name in ["min", "max", "pow", "mod", "distance", "dot", "cross"]:
            if func_name == "dot":
                for ty in types[1:]: # vectors only
                    self.add_builtin_overload("dot", BuiltinFunction(
                        name="dot", kind=BuiltinKind.MATH, parameters=[ty, ty], return_type="float",
                        description="Dot product"
                    ))
            elif func_name == "cross":
                self.add_builtin_overload("cross", BuiltinFunction(
                    name="cross", kind=BuiltinKind.MATH, parameters=["vec3", "vec3"], return_type="vec3",
                    description="Cross product"
                ))
            else:
                for ty in types:
                    self.add_builtin_overload(func_name, BuiltinFunction(
                        name=func_name, kind=BuiltinKind.MATH, parameters=[ty, ty], return_type=ty if func_name not in ["distance"] else "float",
                        description=f"{func_name} overload"
                    ))
                    
        # Three argument functions (mix, clamp, smoothstep)
        elif func_name in ["mix", "clamp", "smoothstep"]:
            for ty in types:
                self.add_builtin_overload(func_name, BuiltinFunction(
                    name=func_name, kind=BuiltinKind.MATH, parameters=[ty, ty, ty], return_type=ty,
                    description=f"{func_name} overload"
                ))
    
    def _add_unimplemented_functions(self, func_names: List[str]) -> None:
        """Mark functions as unimplemented due to known issues."""
        for func_name in func_names:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
            
            # Mark as unimplemented with TODO marker
            self.builtin_functions[func_name].append(
                BuiltinFunction(
                    name=func_name,
                    kind=BuiltinKind.MATH,
                    parameters=[],
                    return_type=None,
                    description=f"TODO: Function {func_name} not implemented due to naga issue"
                )
            )
    
    def get_builtin_function(self, name: str, arg_types: List[Any]) -> Optional[BuiltinFunction]:
        """
        Get a builtin function by name and argument types.
        
        Args:
            name: Function name
            arg_types: List of argument types
            
        Returns:
            BuiltinFunction or None if not found
        """
        if name not in self.builtin_functions:
            return None
        
        functions = self.builtin_functions[name]
        
        # TODO: Implement function overload resolution
        # This should find the best matching overload based on:
        # - Exact type matches
        # - Implicit conversion compatibility
        # - Precision compatibility
        
        # For now, return the first function
        return functions[0] if functions else None
    
    def resolve_builtin_call(self, name: str, args: List[Any]) -> Optional[Any]:
        """
        Resolve a builtin function call to its implementation.
        
        Args:
            name: Function name
            args: Function arguments
            
        Returns:
            Resolved builtin call information or None
        """
        # TODO: Implement builtin call resolution
        # This should:
        # 1. Find the matching builtin function
        # 2. Check argument compatibility
        # 3. Apply implicit conversions if needed
        # 4. Return the resolved call
        
        return None
    
    def add_builtin_overload(self, name: str, overload: BuiltinFunction) -> None:
        """
        Add a builtin function overload.
        
        Args:
            name: Function name
            overload: Function overload information
        """
        if name not in self.builtin_functions:
            self.builtin_functions[name] = []
        self.builtin_functions[name].append(overload)


# Global builtin instance
builtins = Builtins()