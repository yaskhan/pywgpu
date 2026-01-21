from typing import Any, Tuple


class DiagnosticDebug:
    """
    A wrapper for displaying Naga IR terms in debugging output.

    This is like DiagnosticDisplay, but requires weaker context
    and produces correspondingly lower-fidelity output.
    """

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        if isinstance(self.value, tuple):
            item, types = self.value
            return f"{item!r} with types {types!r}"
        else:
            return repr(self.value)


class ForDebug:
    """
    Trait for types that can be formatted using Debug.
    """

    def for_debug(self) -> DiagnosticDebug:
        """
        Format this type using Debug.

        Return a value that implements the Debug trait by displaying
        self in a language-appropriate way.
        """
        return DiagnosticDebug(self)


class ForDebugWithTypes:
    """
    Trait for types that can be formatted using Debug with type context.
    """

    def for_debug(self, types: Any) -> DiagnosticDebug:
        """
        Format this type using Debug with a type arena.

        Given an arena to look up type handles in, return a value that
        implements the Debug trait by displaying self in a language-appropriate way.
        """
        return DiagnosticDebug((self, types))
