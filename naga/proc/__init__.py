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
]
