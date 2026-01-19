"""
Stateless resource tracking.

This module implements tracking for resources that don't require state management,
such as pipelines, bind groups, and render bundles. These trackers only need to
keep resources alive and track their usage.
"""

from __future__ import annotations

from typing import Any, Generic, List, TypeVar


T = TypeVar('T')


class StatelessTracker(Generic[T]):
    """
    A tracker that holds strong references to resources.
    
    This is only used to keep resources alive and track which resources
    are used in a command buffer or render pass. Unlike stateful trackers,
    this doesn't track resource state or generate barriers.
    
    Based on Rust's StatelessTracker in wgpu-core/src/track/stateless.rs
    
    Attributes:
        resources: List of tracked resources.
    """
    
    def __init__(self) -> None:
        """Initialize empty stateless tracker."""
        self.resources: List[T] = []
    
    def insert_single(self, resource: T) -> T:
        """
        Insert a single resource into the tracker.
        
        This keeps the resource alive for the duration of the command buffer
        or render pass. The resource is added to the internal list.
        
        Args:
            resource: The resource to track.
            
        Returns:
            The same resource (for chaining).
        """
        self.resources.append(resource)
        return resource
    
    def clear(self) -> None:
        """Clear all tracked resources."""
        self.resources.clear()
    
    def __iter__(self):
        """Iterate over tracked resources."""
        return iter(self.resources)
    
    def __len__(self) -> int:
        """Get number of tracked resources."""
        return len(self.resources)
    
    def __contains__(self, resource: T) -> bool:
        """Check if resource is tracked."""
        return resource in self.resources
