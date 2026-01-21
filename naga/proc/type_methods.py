"""Type methods and helper functions for constant evaluation.

This module provides type-related utility functions, mirroring
naga/src/proc/type_methods.rs from the wgpu crate.
"""

from dataclasses import dataclass

from naga import (
    Literal,
    Scalar,
    ScalarKind,
    TypeInner,
)


# ============================================================================
# Helper functions for type limits
# ============================================================================

class IntFloatLimits:
    """Trait for types with min/max float representable values."""

    @staticmethod
    def min_float() -> float:
        """Minimum finite representable float value."""
        # TODO: Implement for each type
        raise NotImplementedError("IntFloatLimits.min_float")

    @staticmethod
    def max_float() -> float:
        """Maximum finite representable float value."""
        # TODO: Implement for each type
        raise NotImplementedError("IntFloatLimits.max_float")


# ============================================================================
# Cross product function
# ============================================================================

def cross_product(a: list[float], b: list[float]) -> list[float]:
    """Compute the cross product of two 3D vectors.

    Args:
        a: First vector with 3 components
        b: Second vector with 3 components

    Returns:
        Cross product of a and b
    """
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Cross product requires 3D vectors")

    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


# ============================================================================
# Bit manipulation functions
# ============================================================================

def first_trailing_bit(value: int, *, signed: bool = False) -> int:
    """Find the index of the first trailing bit.

    For signed integers, considers the absolute value for determining
    leading zeros.

    Args:
        value: Integer value to analyze
        signed: Whether the value is signed

    Returns:
        Index of first trailing bit (0 for LSB), or -1 if value is 0

    Note:
        Bit indices start at 0 for the LSB (rightmost bit).
        For example, a value of 1 means LSB is set, so returns 0.
        A value of 0x[80 00...] would return 7 (the bit at index 7).
    """
    if value == 0:
        return -1

    if signed and value < 0:
        value = abs(value)

    trailing_zeros = 0
    while (value & 1) == 0:
        value >>= 1
        trailing_zeros += 1

    return trailing_zeros


def first_leading_bit(value: int, *, signed: bool = False) -> int:
    """Find the index of the first leading (most significant) bit.

    For signed integers, considers the absolute value for determining
    leading zeros (ignoring sign bit).

    Args:
        value: Integer value to analyze
        signed: Whether the value is signed

    Returns:
        Index of first leading bit (0 for LSB), or -1 if value is 0

    Note:
        Bit indices start at 0 for the LSB (rightmost bit).
        For example, a value of 1 returns 0.
        For a value with only bit N set, returns N.
    """
    if value == 0:
        return -1

    if signed and value < 0:
        # For negative numbers, count leading ones in the sign bit pattern
        # Then invert to get the first zero (which is the actual value's MSB)
        value = abs(value)

    leading_zeros = 0
    bit_width = 32 if not signed else 32  # TODO: Use actual bit width

    while value & (1 << (bit_width - 1)) == 0:
        value <<= 1
        leading_zeros += 1

    # Convert right-to-left index to left-to-right index
    return bit_width - 1 - leading_zeros


# ============================================================================
# Component-wise extraction helpers
# ============================================================================

@dataclass
class ExtractedScalar:
    """Result of component-wise scalar extraction."""

    @dataclass
    class AbstractFloat:
        values: list[float]

    @dataclass
    class F32:
        values: list[float]

    @dataclass
    class F16:
        values: list[float]

    @dataclass
    class AbstractInt:
        values: list[int]

    @dataclass
    class U32:
        values: list[int]

    @dataclass
    class I32:
        values: list[int]

    @dataclass
    class U64:
        values: list[int]

    @dataclass
    class I64:
        values: list[int]

    value: (
        AbstractFloat | F32 | F16 | AbstractInt | U32 | I32 | U64 | I64
    )


def extract_scalar_components(
    literals: list[Literal],
) -> ExtractedScalar:
    """Extract scalar components from a list of literals.

    Args:
        literals: List of literals to extract from

    Returns:
        ExtractedScalar with the appropriate variant based on literal type

    Raises:
        ValueError: If literals are empty or types don't match
    """
    if not literals:
        raise ValueError("Cannot extract from empty literal list")

    first = literals[0]

    # Determine the variant based on the first literal type
    match first:
        case Literal.F64(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.F64)]
            return ExtractedScalar(AbstractFloat(values=values))
        case Literal.F32(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.F32)]
            return ExtractedScalar.F32(values=values)
        case Literal.F16(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.F16)]
            return ExtractedScalar.F16(values=values)
        case Literal.AbstractFloat(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.AbstractFloat)]
            return ExtractedScalar(AbstractFloat(values=values))
        case Literal.U32(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.U32)]
            return ExtractedScalar.U32(values=values)
        case Literal.I32(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.I32)]
            return ExtractedScalar.I32(values=values)
        case Literal.U64(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.U64)]
            return ExtractedScalar.U64(values=values)
        case Literal.I64(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.I64)]
            return ExtractedScalar.I64(values=values)
        case Literal.AbstractInt(value=_):
            values = [lit.value for lit in literals if isinstance(lit, Literal.AbstractInt)]
            return ExtractedScalar(AbstractInt(values=values))
        case _:
            raise ValueError(f"Unsupported literal type: {type(first)}")


# ============================================================================
# Flatten compose helper
# ============================================================================

def flatten_compose(
    ty: "Handle[Type]",
    components: list["Handle[Expression]"],
    expressions: "Arena[Expression]",
    types: "UniqueArena[Type]",
) -> list["Handle[Expression]"]:
    """Flatten a compose expression to its literal components.

    This recursively expands nested compose expressions and splats
    to produce a flat list of literal or scalar-valued expressions.

    Mirrors the flatten_compose function from naga/src/proc/mod.rs

    Args:
        ty: Type of the compose expression
        components: Component handles of the compose expression
        expressions: Expression arena
        types: Type arena

    Returns:
        Flattened list of component handles
    """
    from naga import Expression, ExpressionType, TypeInner, TypeInnerType, VectorSize

    # Determine the target size and whether we're flattening vectors
    type_obj = types[ty]
    type_inner = type_obj.inner

    if type_inner.type == TypeInnerType.VECTOR:
        size = int(type_inner.vector_size)
        is_vector = True
    else:
        size = len(components)
        is_vector = False

    def flatten_compose_inner(component: "Handle[Expression]") -> list["Handle[Expression]"]:
        """Flatten a single component if it's a Compose expression."""
        if not is_vector:
            return [component]

        expr = expressions[component]
        if expr.type == ExpressionType.COMPOSE:
            # Return the subcomponents of this Compose
            return list(expr.compose_components)
        else:
            return [component]

    def flatten_splat(component: "Handle[Expression]") -> list["Handle[Expression]"]:
        """Flatten a Splat expression if is_vector is True."""
        expr_handle = component
        count = 1

        if is_vector:
            expr = expressions[expr_handle]
            if expr.type == ExpressionType.SPLAT:
                expr_handle = expr.splat_value
                count = int(expr.splat_size)

        # Repeat the expression handle 'count' times
        return [expr_handle] * count

    # Flatten up to two levels of Compose expressions, then flatten Splats
    # This handles cases like vec4(vec3(vec2(6, 7), 8), 9)
    result = []
    for comp in components:
        # First level of flattening
        level1 = flatten_compose_inner(comp)
        for comp1 in level1:
            # Second level of flattening
            level2 = flatten_compose_inner(comp1)
            for comp2 in level2:
                # Flatten splats
                result.extend(flatten_splat(comp2))

    # Take only the required number of components
    return result[:size]


# ============================================================================
# Type resolution utilities
# ============================================================================

@dataclass
class TypeResolutionHandle:
    """Type resolution that references a type handle."""

    handle: int  # Handle[Type]


@dataclass
class TypeResolutionValue:
    """Type resolution that contains a type inner value."""

    inner: TypeInner


TypeResolution = TypeResolutionHandle | TypeResolutionValue


def resolve_type_from_expression(
    expr: "Expression",
    types: "UniqueArena[Type]",
    constants: "Arena[Constant]",
) -> TypeResolution:
    """Resolve the type of an expression.
    
    This function determines the type of a constant expression by examining
    its structure. It handles literals, constants, zero values, compositions,
    and splats.

    Args:
        expr: Expression to resolve type for
        types: Type arena
        constants: Constant arena

    Returns:
        Type resolution (either handle or value)
        
    Raises:
        ValueError: If expression type cannot be resolved (not a constant expression)
    """
    from naga import ExpressionType, TypeInner, TypeInnerType, Literal
    
    match expr.type:
        case ExpressionType.LITERAL:
            # Literal expressions have their type directly from the literal
            lit = expr.literal
            
            # Get TypeInner from literal
            match lit:
                case Literal.AbstractInt(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.ABSTRACTINT, width=ScalarWidth.ABSTRACT)
                    )
                case Literal.AbstractFloat(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.ABSTRACTFLOAT, width=ScalarWidth.ABSTRACT)
                    )
                case Literal.I32(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.SINT, width=ScalarWidth.B32)
                    )
                case Literal.U32(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.UINT, width=ScalarWidth.B32)
                    )
                case Literal.F32(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.FLOAT, width=ScalarWidth.B32)
                    )
                case Literal.F16(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.FLOAT, width=ScalarWidth.B16)
                    )
                case Literal.I64(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.SINT, width=ScalarWidth.B64)
                    )
                case Literal.U64(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.UINT, width=ScalarWidth.B64)
                    )
                case Literal.Bool(_):
                    from naga import Scalar, ScalarKind, ScalarWidth
                    inner = TypeInner(
                        type=TypeInnerType.SCALAR,
                        scalar=Scalar(kind=ScalarKind.BOOL, width=ScalarWidth.B8)
                    )
                case _:
                    raise ValueError(f"Unknown literal type: {lit}")
            
            return TypeResolution.Value(inner)
            
        case ExpressionType.CONSTANT:
            # Constants have their type stored in the constant definition
            const_handle = expr.constant
            return TypeResolution.Handle(constants[const_handle].ty)
            
        case ExpressionType.ZERO_VALUE:
            # Zero values have an explicit type
            return TypeResolution.Handle(expr.zero_value_ty)
            
        case ExpressionType.COMPOSE:
            # Compose expressions have an explicit type
            return TypeResolution.Handle(expr.compose_ty)
            
        case ExpressionType.SPLAT:
            # Splat creates a vector from a scalar
            # Need to resolve the scalar type first
            # This is a simplified version - full implementation would need
            # recursive resolution
            raise NotImplementedError("Splat type resolution requires recursive resolution")
            
        case _:
            # Other expression types are not constant expressions
            raise ValueError(f"Cannot resolve type for non-constant expression: {expr.type}")
