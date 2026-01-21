from typing import Any


class DiagnosticDisplay:
    """
    A wrapper for displaying Naga IR terms in a human-readable format.
    """

    def __init__(self, value: Any, context: Any):
        self.value = value
        self.context = context

    def __repr__(self) -> str:
        # This is a simplified representation. A real implementation would
        # use the context to look up information about the value and format
        # it in a more human-readable way.
        return f"<{self.value!r} with context {self.context!r}>"


def as_diagnostic_display(value: Any, context: Any) -> DiagnosticDisplay:
    """
    Wraps a value in a DiagnosticDisplay instance.
    """
    return DiagnosticDisplay(value, context)
