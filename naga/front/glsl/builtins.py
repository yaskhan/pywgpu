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
        self._initialize_geometry_functions()
        self._initialize_fragment_functions()
    
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
            "radians", "degrees",
        ]
        
        for func_name in vector_functions:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
    
    def _initialize_matrix_functions(self) -> None:
        """Initialize matrix builtin functions."""
        matrix_functions = [
            "matrixCompMult", "outerProduct", "transpose", "determinant", "inverse",
        ]
        
        for func_name in matrix_functions:
            if func_name not in self.builtin_functions:
                self.builtin_functions[func_name] = []
            
            # TODO: Add variations for different matrix dimensions
            # For now, just add a placeholder or common ones
            self._add_matrix_variations(func_name)

    def _initialize_geometry_functions(self) -> None:
        """Initialize geometry builtin functions."""
        # Such as EmitVertex, EndPrimitive for geometry shaders
        geometry_functions = ["EmitVertex", "EndPrimitive"]
        for func_name in geometry_functions:
            self.builtin_functions[func_name] = [
                BuiltinFunction(name=func_name, kind=BuiltinKind.GEOMETRY, parameters=[], return_type="void", description="Geometry builtin")
            ]

    def _initialize_fragment_functions(self) -> None:
        """Initialize fragment builtin functions."""
        fragment_functions = ["dFdx", "dFdy", "fwidth", "dFdxFine", "dFdyFine", "fwidthFine", "dFdxCoarse", "dFdyCoarse", "fwidthCoarse"]
        for func_name in fragment_functions:
            self.builtin_functions[func_name] = []
            self._add_fragment_variations(func_name)
    
    def _add_texture_variations(self, func_name: str) -> None:
        """Add texture function variations."""
        # Simplified variations for common texture types
        # In a full implementation, this would iterate over all supported texture dimensions and classes
        
        texture_types = ["sampler2D", "sampler3D", "samplerCube", "sampler2DShadow", "samplerCubeShadow", "sampler2DArray", "sampler2DMS"]
        coord_types = {"sampler2D": "vec2", "sampler3D": "vec3", "samplerCube": "vec3", "sampler2DShadow": "vec3", "samplerCubeShadow": "vec4", "sampler2DArray": "vec3", "sampler2DMS": "vec2"}
        
        if func_name in ["texture", "textureLod", "textureProj"]:
             for tex in texture_types:
                 coord = coord_types.get(tex, "vec2")
                 params = [tex, coord]
                 if func_name == "textureLod":
                     params.append("float")
                 
                 self.add_builtin_overload(func_name, BuiltinFunction(
                     name=func_name, kind=BuiltinKind.TEXTURE, parameters=params, return_type="vec4",
                     description=f"{func_name} for {tex}"
                 ))
                 
        elif func_name == "textureSize":
             for tex in texture_types:
                 params = [tex, "int"]
                 self.add_builtin_overload("textureSize", BuiltinFunction(
                     name="textureSize", kind=BuiltinKind.TEXTURE, parameters=params, return_type="ivec2",
                     description="Texture size"
                 ))
        
        elif func_name == "texelFetch":
             for tex in ["sampler2D", "sampler3D", "sampler2DArray", "sampler2DMS"]:
                 coord = "ivec2" if "2D" in tex else "ivec3"
                 params = [tex, coord, "int"]
                 self.add_builtin_overload("texelFetch", BuiltinFunction(
                     name="texelFetch", kind=BuiltinKind.TEXTURE, parameters=params, return_type="vec4",
                     description=f"texelFetch for {tex}"
                 ))
                 
        elif func_name in ["imageLoad", "imageStore"]:
             # Image types: image2D, uimage2D, iimage2D
             for prefix in ["", "u", "i"]:
                 img = f"{prefix}image2D"
                 coord = "ivec2"
                 if func_name == "imageLoad":
                     self.add_builtin_overload("imageLoad", BuiltinFunction(
                         name="imageLoad", kind=BuiltinKind.TEXTURE, parameters=[img, coord], return_type="vec4",
                         description="Image load"
                     ))
                 else:
                     self.add_builtin_overload("imageStore", BuiltinFunction(
                         name="imageStore", kind=BuiltinKind.TEXTURE, parameters=[img, coord, "vec4"], return_type="void",
                         description="Image store"
                     ))
    
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

    def _add_matrix_variations(self, func_name: str) -> None:
        """Add matrix function variations."""
        matrix_types = ["mat2", "mat3", "mat4", "mat2x3", "mat2x4", "mat3x2", "mat3x4", "mat4x2", "mat4x3"]
        for mat in matrix_types:
            if func_name == "transpose":
                self.add_builtin_overload(func_name, BuiltinFunction(
                    name=func_name, kind=BuiltinKind.MATRIX, parameters=[mat], return_type=mat, # transposing might change type if non-square
                    description=f"{func_name} for {mat}"
                ))
            else:
                self.add_builtin_overload(func_name, BuiltinFunction(
                    name=func_name, kind=BuiltinKind.MATRIX, parameters=[mat, mat], return_type=mat,
                    description=f"{func_name} for {mat}"
                ))

    def _add_fragment_variations(self, func_name: str) -> None:
        """Add fragment function variations (derivatives)."""
        types = ["float", "vec2", "vec3", "vec4"]
        for ty in types:
            self.add_builtin_overload(func_name, BuiltinFunction(
                name=func_name, kind=BuiltinKind.FRAGMENT, parameters=[ty], return_type=ty,
                description=f"{func_name} for {ty}"
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
        
        # Find best matching overload
        # We assume arg_types are TypeInner objects
        from ...ir import TypeInner, ScalarKind, VectorSize
        
        best_match = None
        
        for func in functions:
            if len(func.parameters) != len(arg_types):
                continue
                
            match = True
            for i, expected_str in enumerate(func.parameters):
                actual = arg_types[i]
                if not self._match_type(actual, expected_str):
                    match = False
                    break
            
            if match:
                return func
                
        return None

    def _match_type(self, actual: any, expected: str) -> bool:
        """Check if an actual TypeInner matches a string description."""
        from ...ir import TypeInner, ScalarKind, VectorSize, TypeInnerType
        
        if expected == "float":
            return actual.scalar and actual.scalar.kind == ScalarKind.FLOAT and actual.scalar.width == 4
        elif expected == "int":
            return actual.scalar and actual.scalar.kind == ScalarKind.SINT and actual.scalar.width == 4
        elif expected == "uint":
            return actual.scalar and actual.scalar.kind == ScalarKind.UINT and actual.scalar.width == 4
        elif expected == "bool":
            return actual.scalar and actual.scalar.kind == ScalarKind.BOOL
        elif expected == "double":
            return actual.scalar and actual.scalar.kind == ScalarKind.FLOAT and actual.scalar.width == 8
            
        # Vectors
        if expected.startswith("vec"):
            # Floating point vector
            size_map = {"vec2": VectorSize.BI, "vec3": VectorSize.TRI, "vec4": VectorSize.QUAD}
            size = size_map.get(expected)
            if size and actual.vector:
                return actual.vector.size == size and actual.vector.scalar.kind == ScalarKind.FLOAT
        elif expected.startswith("ivec"):
             size_map = {"ivec2": VectorSize.BI, "ivec3": VectorSize.TRI, "ivec4": VectorSize.QUAD}
             size = size_map.get(expected)
             if size and actual.vector:
                return actual.vector.size == size and actual.vector.scalar.kind == ScalarKind.SINT
        elif expected.startswith("uvec"):
             size_map = {"uvec2": VectorSize.BI, "uvec3": VectorSize.TRI, "uvec4": VectorSize.QUAD}
             size = size_map.get(expected)
             if size and actual.vector:
                return actual.vector.size == size and actual.vector.scalar.kind == ScalarKind.UINT
        elif expected.startswith("bvec"):
             size_map = {"bvec2": VectorSize.BI, "bvec3": VectorSize.TRI, "bvec4": VectorSize.QUAD}
             size = size_map.get(expected)
             if size and actual.vector:
                return actual.vector.size == size and actual.vector.scalar.kind == ScalarKind.BOOL

        # Matrices (simplified)
        if expected.startswith("mat"):
             # e.g. mat4, mat3
             if expected == "mat4" and actual.matrix:
                 return actual.matrix.columns == 4 and actual.matrix.rows == 4
             if expected == "mat3" and actual.matrix:
                 return actual.matrix.columns == 3 and actual.matrix.rows == 3

        # Samplers/Images - simplified for now
        if "sampler" in expected and actual.image:
             return True
             
        # Any type match (generic 'genType' equivalent)
        # For now, if expected is generic placeholder like 'genType', we might need more logic
        # But our builtins.py expands 'genType' into concrete types list during initialization?
        # Let's check init logic. It iterates 'types'.
        
        return False
    
    def resolve_builtin_call(self, name: str, args: List[Any], arg_types: List[Any]) -> Optional[BuiltinFunction]:
        """
        Resolve a builtin function call to its implementation.
        
        Args:
            name: Function name
            args: Function arguments
            arg_types: Argument types
            
        Returns:
            Resolved builtin call information or None
        """
        builtin = self.get_builtin_function(name, arg_types)
        if builtin:
             # Apply implicit conversions if needed (already mostly handled by get_builtin_function's match_type)
             # but here we could return a specific structure for the caller
             return builtin
        
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