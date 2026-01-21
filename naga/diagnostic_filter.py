from enum import Enum
from typing import Optional, Dict, List


class Severity(Enum):
    OFF = "off"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FilterableTriggeringRule(Enum):
    DERIVATIVE_UNIFORMITY = "derivative_uniformity"
    UNREACHABLE_CODE = "unreachable_code"
    UNUSED_PARAMETER = "unused_parameter"
    UNUSED_IMPORT = "unused_import"
    UNUSED_VARIABLE = "unused_variable"


class DiagnosticFilterNode:
    """
    Filter for diagnostics.
    """

    def __init__(self, parent: Optional["DiagnosticFilterNode"] = None):
        self.parent = parent
        self.children: List["DiagnosticFilterNode"] = []
        self.rules: Dict[FilterableTriggeringRule, Severity] = {}

    def search(self, rule: FilterableTriggeringRule) -> Optional[Severity]:
        if rule in self.rules:
            return self.rules[rule]
        if self.parent:
            return self.parent.search(rule)
        return None
