"""
Built-in function resolution for WGSL lowerer.

Translated from wgpu-trunk/naga/src/front/wgsl/lower/mod.rs (Built-in section)
"""

from typing import Any, Optional, List, Dict
from ....ir import MathFunction, RelationalFunction, AtomicFunction


class BuiltInResolver:
    """
    Resolves WGSL built-in functions to NAGA IR expressions.
    """
    
    def __init__(self):
        # Map WGSL names to MathFunction enum
        self.math_functions: Dict[str, MathFunction] = {
            "abs": MathFunction.ABS,
            "acos": MathFunction.ACOS,
            "acosh": MathFunction.ACOSH,
            "asin": MathFunction.ASIN,
            "asinh": MathFunction.ASINH,
            "atan": MathFunction.ATAN,
            "atan2": MathFunction.ATAN2,
            "atanh": MathFunction.ATANH,
            "ceil": MathFunction.CEIL,
            "clamp": MathFunction.CLAMP,
            "cos": MathFunction.COS,
            "cosh": MathFunction.COSH,
            "countLeadingZeros": MathFunction.COUNT_LEADING_ZEROS,
            "countOneBits": MathFunction.COUNT_ONE_BITS,
            "countTrailingZeros": MathFunction.COUNT_TRAILING_ZEROS,
            "cross": MathFunction.CROSS,
            "degrees": MathFunction.DEGREES,
            "determinant": MathFunction.DETERMINANT,
            "distance": MathFunction.DISTANCE,
            "dot": MathFunction.DOT,
            "exp": MathFunction.EXP,
            "exp2": MathFunction.EXP2,
            "faceForward": MathFunction.FACE_FORWARD,
            "floor": MathFunction.FLOOR,
            "fma": MathFunction.FMA,
            "fract": MathFunction.FRACT,
            "frexp": MathFunction.FREXP,
            "inverseSqrt": MathFunction.INVERSE_SQRT,
            "ldexp": MathFunction.LDEXP,
            "length": MathFunction.LENGTH,
            "log": MathFunction.LOG,
            "log2": MathFunction.LOG2,
            "max": MathFunction.MAX,
            "min": MathFunction.MIN,
            "mix": MathFunction.MIX,
            "modf": MathFunction.MODF,
            "normalize": MathFunction.NORMALIZE,
            "pow": MathFunction.POW,
            "quantizeToF16": MathFunction.QUANTIZE_TO_F16,
            "radians": MathFunction.RADIANS,
            "reflect": MathFunction.REFLECT,
            "refract": MathFunction.REFRACT,
            "reverseBits": MathFunction.REVERSE_BITS,
            "round": MathFunction.ROUND,
            "saturate": MathFunction.SATURATE,
            "sign": MathFunction.SIGN,
            "sin": MathFunction.SIN,
            "sinh": MathFunction.SINH,
            "smoothstep": MathFunction.SMOOTH_STEP,
            "sqrt": MathFunction.SQRT,
            "step": MathFunction.STEP,
            "tan": MathFunction.TAN,
            "tanh": MathFunction.TANH,
            "transpose": MathFunction.TRANSPOSE,
            "trunc": MathFunction.TRUNC,
        }
        
        # Map WGSL names to RelationalFunction enum
        self.relational_functions: Dict[str, RelationalFunction] = {
            "all": RelationalFunction.ALL,
            "any": RelationalFunction.ANY,
            "isInf": RelationalFunction.IS_INF,
            "isNaN": RelationalFunction.IS_NAN,
        }
        
        # Map WGSL names to AtomicFunction enum
        self.atomic_functions: Dict[str, AtomicFunction] = {
            "atomicAdd": AtomicFunction.ADD,
            "atomicSub": AtomicFunction.SUBTRACT,
            "atomicAnd": AtomicFunction.AND,
            "atomicOr": AtomicFunction.INCLUSIVE_OR,
            "atomicXor": AtomicFunction.EXCLUSIVE_OR,
            "atomicMin": AtomicFunction.MIN,
            "atomicMax": AtomicFunction.MAX,
            "atomicExchange": AtomicFunction.EXCHANGE,
            "atomicCompareExchangeWeak": AtomicFunction.EXCHANGE, # Handled with comparison
        }
        
    def resolve_math(self, name: str) -> Optional[MathFunction]:
        """Resolve a math built-in."""
        return self.math_functions.get(name)
        
    def resolve_relational(self, name: str) -> Optional[RelationalFunction]:
        """Resolve a relational built-in."""
        return self.relational_functions.get(name)

    def resolve_atomic(self, name: str) -> Optional[AtomicFunction]:
        """Resolve an atomic built-in."""
        return self.atomic_functions.get(name)

    def resolve_derivative(self, name: str) -> Optional[tuple]:
        """
        Resolve a derivative built-in.
        
        Returns:
            Tuple of (DerivativeAxis, DerivativeControl) or None
        """
        from ....ir import DerivativeAxis, DerivativeControl
        
        mapping = {
            "dpdx": (DerivativeAxis.X, DerivativeControl.NONE),
            "dpdxCoarse": (DerivativeAxis.X, DerivativeControl.COARSE),
            "dpdxFine": (DerivativeAxis.X, DerivativeControl.FINE),
            "dpdy": (DerivativeAxis.Y, DerivativeControl.NONE),
            "dpdyCoarse": (DerivativeAxis.Y, DerivativeControl.COARSE),
            "dpdyFine": (DerivativeAxis.Y, DerivativeControl.FINE),
            "fwidth": (DerivativeAxis.WIDTH, DerivativeControl.NONE),
            "fwidthCoarse": (DerivativeAxis.WIDTH, DerivativeControl.COARSE),
            "fwidthFine": (DerivativeAxis.WIDTH, DerivativeControl.FINE),
        }
        return mapping.get(name)

    def resolve_select(self, name: str) -> bool:
        """Check if name is 'select' built-in."""
        return name == "select"

    def resolve_texture(self, name: str) -> Optional[str]:
        """
        Resolve a texture built-in.
        
        Returns:
            String identifier for the texture built-in type.
        """
        mapping = {
            "textureSample": "sample",
            "textureSampleBias": "sample_bias",
            "textureSampleCompare": "sample_compare",
            "textureSampleCompareLevel": "sample_compare_level",
            "textureSampleGrad": "sample_grad",
            "textureSampleLevel": "sample_level",
            "textureLoad": "load",
            "textureDimensions": "query_size",
            "textureNumLevels": "query_num_levels",
            "textureNumLayers": "query_num_layers",
            "textureNumSamples": "query_num_samples",
            "textureStore": "store",
            "textureGather": "gather",
            "textureGatherCompare": "gather_compare",
        }
        return mapping.get(name)

    def resolve_array_length(self, name: str) -> bool:
        """Check if name is 'arrayLength' built-in."""
        return name == "arrayLength"
