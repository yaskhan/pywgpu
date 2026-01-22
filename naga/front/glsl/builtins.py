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
            
            # TODO: glsl supports using bias with depth samplers but naga doesn't (строка 183)
            # This is about adding support for bias parameter with depth samplers
            # In GLSL, depth textures can use bias parameters in texture sampling
            # but naga doesn't currently support this combination
            
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
        
        # TODO: https://github.com/gfx-rs/naga/issues/2526 (строка 1395)
        # These functions are commented out in the original code:
        # "modf", "frexp"
        # Need to investigate and implement these math functions
        
        # TODO: Implement modf and frexp functions
        # "modf" - Split float into integer and fractional parts
        # "frexp" - Split float into mantissa and exponent
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
        # Add scalar variations (float, double)
        # Add vector variations (vec2, vec3, vec4)
        # Add matrix variations (mat2, mat3, mat4, etc.)
        pass
    
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