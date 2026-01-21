"""
Pipeline tracking for render and compute pipelines.

This module implements tracking for GPU pipelines to ensure proper
usage validation and resource management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar


T = TypeVar("T")


@dataclass
class PipelineState:
    """
    State of a tracked pipeline.

    Attributes:
        pipeline: The pipeline resource.
        ref_count: Reference count for usage tracking.
    """

    pipeline: Any
    ref_count: int = 1


class PipelineTracker(Generic[T]):
    """
    Tracker for render or compute pipelines.

    This tracker manages pipeline usage within a command buffer or render pass,
    ensuring pipelines are properly tracked and validated.

    Attributes:
        pipelines: Dictionary mapping pipeline ID to state.
    """

    def __init__(self) -> None:
        """Initialize the pipeline tracker."""
        self.pipelines: Dict[int, PipelineState] = {}

    def insert_single(self, pipeline: T) -> T:
        """
        Insert a single pipeline into the tracker.

        This registers the pipeline for usage tracking. If the pipeline
        is already tracked, increments its reference count.

        Args:
            pipeline: The pipeline to track.

        Returns:
            The same pipeline (for chaining).
        """
        # Get pipeline ID (would use proper ID system)
        pipeline_id = id(pipeline)

        if pipeline_id in self.pipelines:
            # Already tracked, increment ref count
            self.pipelines[pipeline_id].ref_count += 1
        else:
            # New pipeline, create state
            self.pipelines[pipeline_id] = PipelineState(
                pipeline=pipeline,
                ref_count=1,
            )

        return pipeline

    def remove(self, pipeline: T) -> None:
        """
        Remove a pipeline from tracking.

        Decrements reference count and removes if it reaches zero.

        Args:
            pipeline: The pipeline to remove.
        """
        pipeline_id = id(pipeline)

        if pipeline_id in self.pipelines:
            state = self.pipelines[pipeline_id]
            state.ref_count -= 1

            if state.ref_count <= 0:
                del self.pipelines[pipeline_id]

    def get(self, pipeline: T) -> Optional[PipelineState]:
        """
        Get the state of a tracked pipeline.

        Args:
            pipeline: The pipeline to query.

        Returns:
            Pipeline state if tracked, None otherwise.
        """
        pipeline_id = id(pipeline)
        return self.pipelines.get(pipeline_id)

    def is_tracked(self, pipeline: T) -> bool:
        """
        Check if a pipeline is currently tracked.

        Args:
            pipeline: The pipeline to check.

        Returns:
            True if tracked, False otherwise.
        """
        pipeline_id = id(pipeline)
        return pipeline_id in self.pipelines

    def clear(self) -> None:
        """Clear all tracked pipelines."""
        self.pipelines.clear()

    def set_size(self, size: int) -> None:
        """
        Set the expected size for the tracker.

        This is a hint for optimization (pre-allocation).

        Args:
            size: Expected number of pipelines.
        """
        # Python dicts auto-resize, but we could pre-allocate if needed
        pass
