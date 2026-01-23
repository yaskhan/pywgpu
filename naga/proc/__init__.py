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

# Literal helpers and operations
from .literal_helpers import (
    LiteralVector,
    literal_ty_inner,
)

from .zero_value_helpers import (
    literal_zero,
    eval_zero_value_impl,
)

from .literal_operations import (
    apply_unary_op,
    apply_binary_op,
    LiteralOperationError,
)

# Extended constant evaluator
from .constant_evaluator_extended import (
    eval_unary_expression,
    eval_binary_expression,
    eval_zero_value_expression,
    eval_compose_expression,
    eval_splat_expression,
    eval_access_index_expression,
    eval_swizzle_expression,
    eval_select_expression,
)

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
    # Literal helpers
    "LiteralVector",
    "literal_ty_inner",
    # Zero value helpers
    "literal_zero",
    "eval_zero_value_impl",
    # Literal operations
    "apply_unary_op",
    "apply_binary_op",
    "LiteralOperationError",
    # Extended evaluator
    "eval_unary_expression",
    "eval_binary_expression",
    "eval_zero_value_expression",
    "eval_compose_expression",
    "eval_splat_expression",
    "eval_access_index_expression",
    "eval_swizzle_expression",
    "eval_select_expression",
]
