"""
HAL (Hardware Abstraction Layer) integration for wgpu-core.

This module provides utilities for accessing raw HAL resources from wgpu-core
resources. This is useful for low-level operations and integration with
platform-specific APIs.

The module provides guards that hold resources alive while providing access to
their underlying HAL representations.
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar
from weakref import ref

from . import lock
from .resource import RawResourceAccess
from .snatch import SnatchGuard

T = TypeVar("T")
HalType = TypeVar("HalType")


class SimpleResourceGuard(Generic[T, HalType]):
    """
    A guard which holds alive a wgpu-core resource and dereferences to the HAL type.
    
    This guard ensures that the resource remains alive while the HAL type is being
    accessed. It uses a callback to derive the HAL type from the resource.
    
    Attributes:
        _guard: The resource being held alive.
        _ptr: Pointer to the HAL resource.
    """

    def __init__(self, guard: T, callback: Any) -> None:
        """
        Create a new guard from a resource, using a callback to derive the HAL type.
        
        Args:
            guard: The resource to hold alive.
            callback: A callable that derives the HAL type from the resource.
        
        Returns:
            None if the resource is not of the expected HAL type.
        """
        self._guard = guard
        self._ptr = callback(guard)
        if self._ptr is None:
            raise ValueError("Resource is not of the expected HAL type")

    def __deref__(self) -> HalType:
        """
        Dereference to the HAL type.
        
        Returns:
            The HAL resource.
        """
        return self._ptr


class SnatchableResourceGuard(Generic[T, HalType]):
    """
    A guard which holds alive a snatchable wgpu-core resource and dereferences to the HAL type.
    
    This guard is used for resources that can be "snatched" (destroyed early) while
    still being accessed. It holds the snatchable lock while the HAL type is being
    accessed.
    
    Attributes:
        resource: The snatchable resource.
        snatch_lock_rank_data: Data for the snatchable lock.
        _ptr: Pointer to the HAL resource.
    """

    def __init__(self, resource: T) -> None:
        """
        Create a new guard from a snatchable resource.
        
        Args:
            resource: The resource to hold alive.
        
        Returns:
            None if the resource is not of the expected HAL type or has been destroyed.
        """
        from .snatch import SnatchGuard

        # Grab the snatchable lock.
        snatch_guard = resource.device.snatchable_lock.read()

        # Get the raw resource and downcast it to the expected HAL type.
        underlying = resource.raw(snatch_guard)
        if underlying is None:
            raise ValueError("Resource has been destroyed")

        # Cast the raw resource to a pointer to get rid of the lifetime
        # connecting us to the snatch guard.
        self._ptr = underlying

        # SAFETY: At this point all panicking or divergance has already happened,
        # so we can safely forget the snatch guard without causing the lock to be left open.
        self.snatch_lock_rank_data = SnatchGuard.forget(snatch_guard)

        # SAFETY: We only construct this guard while the snatchable lock is held,
        # as the `drop` implementation of this guard will unsafely release the lock.
        self.resource = resource

    def __deref__(self) -> HalType:
        """
        Dereference to the HAL type.
        
        Returns:
            The HAL resource.
        """
        return self._ptr

    def __del__(self) -> None:
        """
        Release the snatchable lock when the guard is dropped.
        """
        from .snatch import SnatchGuard

        # SAFETY: We are not going to access the rank data anymore.
        data = self.snatch_lock_rank_data

        # SAFETY: The pointer is no longer going to be accessed.
        # The snatchable lock is being held because this type was not created
        # until after the snatchable lock was forgotten.
        self.resource.device.snatchable_lock.force_unlock_read(data)


class FenceGuard(Generic[HalType]):
    """
    A guard which holds alive a device and the device's fence lock, dereferencing to the HAL type.
    
    This guard is used for accessing the device's fence while holding the fence lock.
    
    Attributes:
        device: The device being held alive.
        fence_lock_rank_data: Data for the fence lock.
        _ptr: Pointer to the HAL fence.
    """

    def __init__(self, device: Any) -> None:
        """
        Create a new guard over a device's fence.
        
        Args:
            device: The device to hold alive.
        
        Returns:
            None if the device's fence is not of the expected HAL type.
        """
        from .lock import RwLockReadGuard

        # Grab the fence lock.
        fence_guard = device.fence.read()

        # Get the raw fence and downcast it to the expected HAL type, coercing it to a pointer
        # to get rid of the lifetime connecting us to the fence guard.
        self._ptr = fence_guard.as_any().downcast_ref()
        if self._ptr is None:
            raise ValueError("Fence is not of the expected HAL type")

        # SAFETY: At this point all panicking or divergance has already happened,
        # so we can safely forget the fence guard without causing the lock to be left open.
        self.fence_lock_rank_data = RwLockReadGuard.forget(fence_guard)

        # SAFETY: We only construct this guard while the fence lock is held,
        # as the `drop` implementation of this guard will unsafely release the lock.
        self.device = device

    def __deref__(self) -> HalType:
        """
        Dereference to the HAL type.
        
        Returns:
            The HAL fence.
        """
        return self._ptr

    def __del__(self) -> None:
        """
        Release the fence lock when the guard is dropped.
        """
        # SAFETY: We are not going to access the rank data anymore.
        data = self.fence_lock_rank_data

        # SAFETY: The pointer is no longer going to be accessed.
        # The fence lock is being held because this type was not created
        # until after the fence lock was forgotten.
        self.device.fence.force_unlock_read(data)
