"""
Timestamp writes management.

This module implements timestamp writes management for wgpu-core. It provides:
- PassTimestampWrites: Timestamp writes for a pass
- TimestampWritesError: Error related to timestamp writes

Timestamp writes are used to record GPU timing information for performance
analysis and profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PassTimestampWrites:
    """
    Timestamp writes for a pass.
    
    Attributes:
        query_set: The query set to write to.
        beginning_of_pass_write_index: Index for beginning of pass write.
        end_of_pass_write_index: Index for end of pass write.
    """

    query_set: Any
    beginning_of_pass_write_index: Optional[int]
    end_of_pass_write_index: Optional[int]


@dataclass
class TimestampWritesError(Exception):
    """
    Error related to timestamp writes.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ArcPassTimestampWrites:
    """
    Timestamp writes for a pass with Arc references.
    
    Attributes:
        query_set: The query set to write to.
        beginning_of_pass_write_index: Index for beginning of pass write.
        end_of_pass_write_index: Index for end of pass write.
    """

    query_set: Any
    beginning_of_pass_write_index: Optional[int]
    end_of_pass_write_index: Optional[int]
