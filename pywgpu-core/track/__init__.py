from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from .metadata import ResourceMetadata
from .buffer import BufferTracker, BufferUsageScope
from .texture import TextureTracker, TextureUsageScope
from .pipeline import PipelineTracker
from .stateless import StatelessTracker


@dataclass
class StateTransition:
    """Represents a transition from one usage state to another."""
    from_state: Any
    to_state: Any


@dataclass
class PendingTransition:
    """Represents a pending resource transition."""
    id: int
    selector: Any
    usage: StateTransition


from .tracker import Tracker

class ResourceUsageCompatibilityError(Exception):
    """Raised when resource usage is incompatible."""

    def __init__(self, message: str):
        super().__init__(message)
