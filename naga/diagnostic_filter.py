"""
Diagnostic filtering system for Naga shader validation.

This module provides severity levels, triggering rules, and filter management
for controlling diagnostic output during shader compilation.
"""

from enum import Enum
from typing import Optional, Dict, List


class Severity(Enum):
    """Severity levels for diagnostics."""
    OFF = "off"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FilterableTriggeringRule(Enum):
    """Rules that can trigger diagnostics."""
    DERIVATIVE_UNIFORMITY = "derivative_uniformity"
    UNREACHABLE_CODE = "unreachable_code"
    UNUSED_PARAMETER = "unused_parameter"
    UNUSED_IMPORT = "unused_import"
    UNUSED_VARIABLE = "unused_variable"


class DiagnosticFilter:
    """
    A filtering rule that modifies how diagnostics are emitted for shaders.
    """
    
    def __init__(self, new_severity: Severity, triggering_rule: FilterableTriggeringRule):
        self.new_severity = new_severity
        self.triggering_rule = triggering_rule


class DiagnosticFilterNode:
    """
    A tree node for diagnostic filters with parent-child relationships.
    """

    def __init__(self, parent: Optional["DiagnosticFilterNode"] = None):
        self.parent = parent
        self.children: List["DiagnosticFilterNode"] = []
        self.rules: Dict[FilterableTriggeringRule, Severity] = {}

    def search(self, rule: FilterableTriggeringRule) -> Optional[Severity]:
        """Search for a rule in this node or its ancestors."""
        if rule in self.rules:
            return self.rules[rule]
        if self.parent:
            return self.parent.search(rule)
        return None


class DiagnosticFilterMap:
    """
    A map from diagnostic filters to their severity and span.
    """
    
    def __init__(self):
        self.filters: Dict[FilterableTriggeringRule, tuple[Severity, Optional[tuple[int, int]]]] = {}
    
    def add(self, diagnostic_filter: DiagnosticFilter, span: tuple[int, int]) -> None:
        """Add a diagnostic filter to the map."""
        self.filters[diagnostic_filter.triggering_rule] = (diagnostic_filter.new_severity, span)
    
    def is_empty(self) -> bool:
        """Check if the map is empty."""
        return len(self.filters) == 0
    
    def spans(self):
        """Get all spans in the map."""
        return [span for _, span in self.filters.values()]


__all__ = [
    "Severity",
    "FilterableTriggeringRule", 
    "DiagnosticFilter",
    "DiagnosticFilterNode",
    "DiagnosticFilterMap"
]
