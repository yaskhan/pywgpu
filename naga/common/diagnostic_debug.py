from typing import Any, Tuple

class DiagnosticDebug:
    """
    A wrapper for displaying Naga IR terms in debugging output.
    """
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        if isinstance(self.value, tuple):
            # This is a bit of a simplification, but it captures the essence
            # of the Rust implementation.
            item, types = self.value
            return f"{item!r} with types {types!r}"
        else:
            return repr(self.value)

def for_debug(value: Any) -> DiagnosticDebug:
    """
    Format this type using core::fmt::Debug.
    """
    return DiagnosticDebug(value)

def for_debug_with_types(value: Any, types: Any) -> DiagnosticDebug:
    """
    Format this type using core::fmt::Debug, with a type arena.
    """
    return DiagnosticDebug((value, types))

