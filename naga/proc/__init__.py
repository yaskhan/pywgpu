"""
Module processing functionality.

This package provides various utilities for processing Naga IR modules.
"""

from .namer import Namer, NameKey, EntryPointIndex, ExternalTextureNameKey
from .typifier import Typifier
from .layouter import Layouter
from .keyword_set import KeywordSet, CaseInsensitiveKeywordSet
from .emitter import Emitter
from .terminator import ensure_block_returns
from .index import (
    BoundsCheckPolicy,
    BoundsCheckPolicies,
    IndexableLength,
    IndexableLengthError,
)
from .constant_evaluator import (
    ConstantEvaluator,
    ConstantEvaluatorError,
    ExpressionKind,
    ExpressionKindTracker,
)
from .resolve import ResolveContext, ResolveError

# Type methods and helpers
from .type_methods import (
    cross_product,
    first_leading_bit,
    first_trailing_bit,
    flatten_compose,
    min_max_float_representable_by,
    TypeResolution,
    TypeResolutionHandle,
    TypeResolutionValue,
)

# Component-wise operations
from .component_wise import (
    component_wise_scalar,
    component_wise_float,
    component_wise_concrete_int,
    component_wise_signed,
    flatten_compose_to_literals,
    extract_vector_literals,
    math_function_arg_count,
)

# Math function implementations
from .constant_evaluator_math import MathFunctionEvaluator

__all__ = [
    "Namer",
    "NameKey",
    "EntryPointIndex",
    "ExternalTextureNameKey",
    "Typifier",
    "Layouter",
    "KeywordSet",
    "CaseInsensitiveKeywordSet",
    "Emitter",
    "ensure_block_returns",
    "BoundsCheckPolicy",
    "BoundsCheckPolicies",
    "IndexableLength",
    "IndexableLengthError",
    "ConstantEvaluator",
    "ConstantEvaluatorError",
    "ExpressionKind",
    "ExpressionKindTracker",
    "cross_product",
    "first_leading_bit",
    "first_trailing_bit",
    "flatten_compose",
    "min_max_float_representable_by",
    "TypeResolution",
    "TypeResolutionHandle",
    "TypeResolutionValue",
    "component_wise_scalar",
    "component_wise_float",
    "component_wise_concrete_int",
    "component_wise_signed",
    "flatten_compose_to_literals",
    "extract_vector_literals",
    "math_function_arg_count",
    "MathFunctionEvaluator",
    "ResolveContext",
    "ResolveError",
]
