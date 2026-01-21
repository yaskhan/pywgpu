"""
Module for hashing utilities.

This module provides fast hash map and hash set implementations using
non-cryptographic hash algorithms. These are optimized for performance
in wgpu-core's internal data structures.

The hash utilities are used throughout wgpu-core for efficient lookups
and collections.
"""

from __future__ import annotations

from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class FastHashMap(dict[K, V]):
    """
    HashMap using a fast, non-cryptographic hash algorithm.

    This is a dictionary subclass optimized for performance in wgpu-core.
    It uses a fast hash algorithm (FxHash) for better performance than
    the standard Python hash.

    Note: In Python, this is just a regular dict since Python's dict
    already uses a fast hash algorithm. This class is provided for
    API compatibility with the Rust implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FastHashMap."""
        super().__init__(*args, **kwargs)


class FastHashSet(set[K]):
    """
    HashSet using a fast, non-cryptographic hash algorithm.

    This is a set subclass optimized for performance in wgpu-core.
    It uses a fast hash algorithm (FxHash) for better performance than
    the standard Python hash.

    Note: In Python, this is just a regular set since Python's set
    already uses a fast hash algorithm. This class is provided for
    API compatibility with the Rust implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FastHashSet."""
        super().__init__(*args, **kwargs)


class FastIndexMap(dict[K, V]):
    """
    IndexMap using a fast, non-cryptographic hash algorithm.

    This is a dictionary subclass that maintains insertion order while
    providing fast lookups. It uses a fast hash algorithm (FxHash) for
    better performance than the standard Python hash.

    Note: In Python, this is just a regular dict since Python's dict
    maintains insertion order (since Python 3.7). This class is provided
    for API compatibility with the Rust implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FastIndexMap."""
        super().__init__(*args, **kwargs)
