from __future__ import annotations
from typing import Any
from .buffer import BufferTracker
from .texture import TextureTracker
from .pipeline import PipelineTracker
from .stateless import StatelessTracker


class Tracker:
    """
    A collection of all resource trackers.

    This class orchestrates tracking for different types of resources
    (buffers, textures, pipelines, bundles, etc.) within a command buffer or device.
    """

    def __init__(self) -> None:
        # Stateful trackers (with usage tracking)
        self.buffers = BufferTracker()
        self.textures = TextureTracker()

        # Stateless trackers (just keep-alive)
        self.render_pipelines = PipelineTracker()
        self.compute_pipelines = PipelineTracker()
        self.bundles = StatelessTracker()
        self.bind_groups = StatelessTracker()
        self.query_sets = StatelessTracker()
        self.views = StatelessTracker()

    def set_size(self, buffer_size: int, texture_size: int) -> None:
        """Sets the size of the internal trackers."""
        self.buffers.set_size(buffer_size)
        self.textures.set_size(texture_size)

    def merge_scope(
        self, scope: Tracker
    ) -> Any:  # Tracker here serves as a scope container
        """Merges a tracker scope (e.g. from a pass) into this tracker."""
        from . import PendingTransition

        buffer_transitions = list(self.buffers.merge_scope(scope.buffers))
        texture_transitions = list(self.textures.merge_scope(scope.textures))

        # Pipelines are tracked in a dict
        for state in scope.render_pipelines.pipelines.values():
            self.render_pipelines.insert_single(state.pipeline)
        for state in scope.compute_pipelines.pipelines.values():
            self.compute_pipelines.insert_single(state.pipeline)

        # Stateless trackers are tracked in a list
        for bundle in scope.bundles.resources:
            self.bundles.insert_single(bundle)
        for bg in scope.bind_groups.resources:
            self.bind_groups.insert_single(bg)
        for qs in scope.query_sets.resources:
            self.query_sets.insert_single(qs)
        for view in scope.views.resources:
            self.views.insert_single(view)

        return {
            "buffers": buffer_transitions,
            "textures": texture_transitions,
        }

    def drain_transitions(self) -> Any:
        """Yields and clears all pending transitions from stateful trackers."""
        return {
            "buffers": list(self.buffers.drain_transitions()),
            "textures": list(self.textures.drain_transitions()),
        }

    def clear(self) -> None:
        """Clears all trackers."""
        self.buffers.clear()
        self.textures.clear()
        self.render_pipelines.pipelines.clear()
        self.compute_pipelines.pipelines.clear()
        self.bundles.resources.clear()
        self.bind_groups.resources.clear()
        self.query_sets.resources.clear()
        self.views.resources.clear()
