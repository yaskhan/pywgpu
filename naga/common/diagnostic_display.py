from typing import Any


class DiagnosticDisplay:
    """
    A wrapper for displaying Naga IR terms in diagnostic output.

    For some Naga IR type T, DiagnosticDisplay implements __str__ in a way
    that displays values of type T appropriately for diagnostic messages
    presented to human readers.

    For example, the implementation of __str__ for DiagnosticDisplay<Scalar>
    formats the type represented by the given Scalar appropriately for users.

    Some types like Handle<Type> require contextual information like
    a type arena to be displayed. In such cases, we implement __str__
    for a type like DiagnosticDisplay[(Handle<Type>, GlobalCtx)], where
    the GlobalCtx type provides the necessary context.

    If you only need debugging output, DiagnosticDebug uses
    easier-to-obtain context types but still does a good enough job
    for logging or debugging.
    """

    def __init__(self, value: Any):
        self.value = value

    def __str__(self) -> str:
        """
        Format the value for diagnostic output.

        This is a simplified implementation. A full implementation would
        use the context to look up information about the value and format
        it in a more human-readable way.
        """
        if isinstance(self.value, tuple):
            if len(self.value) == 2:
                item, ctx = self.value
                # Try to format with context
                return f"{item!r}"
            elif len(self.value) == 3:
                name, rule, ctx = self.value
                return f"{name}({rule!r})"
        return repr(self.value)
