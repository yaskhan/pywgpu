from enum import Enum
from typing import Optional, Dict, List, Tuple, Union
from ..span import Span


class Severity(Enum):
    OFF = "off"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    def report_diag(self, err, log_handler):
        """
        Checks whether this severity is Error.

        Naga does not yet support diagnostic items at lesser severities than
        Severity::Error. When this is implemented, this method should be deleted, and the
        severity should be used directly for reporting diagnostics.
        """
        if self == Severity.OFF:
            return None

        # NOTE: These severities are not yet reported.
        if self == Severity.INFO:
            log_level = "INFO"
        elif self == Severity.WARNING:
            log_level = "WARNING"
        else:  # ERROR
            return err

        log_handler(err, log_level)
        return None


class StandardFilterableTriggeringRule(Enum):
    """
    A filterable triggering rule in a DiagnosticFilter.
    """
    DERIVATIVE_UNIFORMITY = "derivative_uniformity"

    def default_severity(self) -> Severity:
        """
        The default severity associated with this triggering rule.
        """
        if self == StandardFilterableTriggeringRule.DERIVATIVE_UNIFORMITY:
            return Severity.ERROR
        return Severity.ERROR


class FilterableTriggeringRule:
    """
    A filterable triggering rule in a DiagnosticFilter.
    """

    def __init__(self, value: Union[StandardFilterableTriggeringRule, str, Tuple[str, str]]):
        """
        Args:
            value: Either StandardFilterableTriggeringRule, a string for Unknown,
                   or a tuple of two strings for User (namespace, rule_name)
        """
        self._value = value

    @classmethod
    def standard(cls, rule: StandardFilterableTriggeringRule) -> "FilterableTriggeringRule":
        return cls(rule)

    @classmethod
    def unknown(cls, name: str) -> "FilterableTriggeringRule":
        return cls(name)

    @classmethod
    def user(cls, namespace: str, rule_name: str) -> "FilterableTriggeringRule":
        return cls((namespace, rule_name))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FilterableTriggeringRule):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        if isinstance(self._value, StandardFilterableTriggeringRule):
            return f"FilterableTriggeringRule.Standard({self._value})"
        elif isinstance(self._value, str):
            return f"FilterableTriggeringRule.Unknown({self._value})"
        else:
            return f"FilterableTriggeringRule.User({self._value[0]}, {self._value[1]})"


class DiagnosticFilter:
    """
    A filtering rule that modifies how diagnostics are emitted for shaders.
    """

    def __init__(self, new_severity: Severity, triggering_rule: FilterableTriggeringRule):
        self.new_severity = new_severity
        self.triggering_rule = triggering_rule

    def __repr__(self) -> str:
        return f"DiagnosticFilter(severity={self.new_severity}, rule={self.triggering_rule})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DiagnosticFilter):
            return (
                self.new_severity == other.new_severity
                and self.triggering_rule == other.triggering_rule
            )
        return False

    def __hash__(self) -> int:
        return hash((self.new_severity, self.triggering_rule))


class ConflictingDiagnosticRuleError(Exception):
    """
    An error returned by DiagnosticFilterMap::add when it encounters conflicting rules.
    """

    def __init__(self, triggering_rule_spans: Tuple[Span, Span]):
        self.triggering_rule_spans = triggering_rule_spans
        super().__init__("Conflicting diagnostic rules")


class ShouldConflictOnFullDuplicate(Enum):
    """
    Determines whether DiagnosticFilterMap::add should consider full duplicates a conflict.

    In WGSL, directive position does not consider this case a conflict, while attribute
    position does.
    """
    YES = "yes"
    NO = "no"


class DiagnosticFilterMap:
    """
    A map from diagnostic filters to their severity and span.

    Front ends can use this to collect the set of filters applied to a
    particular language construct, and detect duplicate/conflicting filters.
    """

    def __init__(self):
        self._filters: Dict[FilterableTriggeringRule, Tuple[Severity, Span]] = {}

    def add(
        self,
        diagnostic_filter: DiagnosticFilter,
        span: Span,
        should_conflict_on_full_duplicate: ShouldConflictOnFullDuplicate,
    ) -> None:
        """
        Add the given diagnostic_filter parsed at the given span to this map.

        Raises:
            ConflictingDiagnosticRuleError: if there's a conflict
        """
        triggering_rule = diagnostic_filter.triggering_rule
        new_severity = diagnostic_filter.new_severity

        if triggering_rule in self._filters:
            first_severity, first_span = self._filters[triggering_rule]
            should_conflict = should_conflict_on_full_duplicate == ShouldConflictOnFullDuplicate.YES

            if first_severity != new_severity or should_conflict:
                raise ConflictingDiagnosticRuleError((first_span, span))
        else:
            self._filters[triggering_rule] = (new_severity, span)

    def is_empty(self) -> bool:
        """Were any rules specified?"""
        return len(self._filters) == 0

    def spans(self) -> List[Span]:
        """Returns the spans of all contained rules."""
        return [span for _, span in self._filters.values()]

    def __iter__(self):
        """Iterate over (FilterableTriggeringRule, (Severity, Span))"""
        return iter(self._filters.items())


class DiagnosticFilterNode:
    """
    Represents a single parent-linking node in a tree of DiagnosticFilters.

    A single element of a tree of diagnostic filter rules stored in
    Module::diagnostic_filters. When nodes are built by a front-end, module-applicable
    filter rules are chained together in runs based on parse site.
    """

    def __init__(
        self,
        inner: DiagnosticFilter,
        parent: Optional["DiagnosticFilterNode"] = None,
    ):
        self.inner = inner
        self.parent = parent

    @staticmethod
    def search(
        node: Optional["DiagnosticFilterNode"],
        triggering_rule: StandardFilterableTriggeringRule,
    ) -> Severity:
        """
        Finds the most specific filter rule applicable to triggering_rule from the chain of
        diagnostic filter rules, starting with node, and returns its severity. If none
        is found, return the value of StandardFilterableTriggeringRule::default_severity.

        When triggering_rule is not applicable to this node, its parent is consulted recursively.
        """
        current = node
        while current is not None:
            rule = current.inner.triggering_rule
            new_severity = current.inner.new_severity

            if isinstance(rule._value, StandardFilterableTriggeringRule) and rule._value == triggering_rule:
                return new_severity

            current = current.parent

        return triggering_rule.default_severity()
