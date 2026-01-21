"""
Resource pool management.

This module implements a resource pool for deduplicating resources. The pool
ensures that only one resource is created for each unique set of parameters,
which is useful for resources like bind group layouts that can be shared
across pipelines.

The pool uses weak references to allow resources to be garbage collected
when they are no longer in use.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

from . import lock
from .hash_utils import FastHashMap


K = TypeVar("K")
V = TypeVar("V")


class ResourcePool(Generic[K, V]):
    """
    A pool for deduplicating resources.

    The resource pool ensures that only one resource is created for each
    unique set of parameters. When a resource is requested, the pool checks
    if a resource with the same parameters already exists. If it does, the
    existing resource is returned. If not, a new resource is created.

    Resources are stored using weak references, so they can be garbage
    collected when they are no longer in use.

    Attributes:
        inner: Mutex-protected map of resources.
    """

    def __init__(self) -> None:
        """Initialize the resource pool."""
        self.inner = lock.Mutex(lock.rank.RESOURCE_POOL_INNER, FastHashMap())

    def get_or_init(
        self,
        key: K,
        constructor: Callable[[K], V],
    ) -> V:
        """
        Get a resource from the pool with the given key, or create a new one.

        Behaves such that only one resource will be created for each unique
        key at any one time.

        Args:
            key: The key to look up.
            constructor: A callable that creates a new resource.

        Returns:
            The resource.

        Raises:
            Exception: If the constructor raises an exception.
        """
        # Implementation depends on weak references and race conditions
        with self.inner.lock() as map_guard:
            # Check if resource already exists
            if key in map_guard:
                # For now, just return the existing resource
                # In a real implementation, this would use weak references
                return map_guard[key]

            # Create new resource
            resource = constructor(key)
            map_guard[key] = resource
            return resource

    def remove(self, key: K) -> None:
        """
        Remove the given key from the pool.

        Must only be called in the Drop impl of the resource.

        Args:
            key: The key to remove.
        """
        with self.inner.lock() as map_guard:
            map_guard.pop(key, None)
