class Span:
    """
    Source code span.
    """

    def __init__(self, start: int = 0, end: int = 0):
        self.start = start
        self.end = end

    @classmethod
    def new(cls) -> "Span":
        """Create a new empty span."""
        return cls(0, 0)

    def __repr__(self) -> str:
        return f"Span({self.start}, {self.end})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Span) and self.start == other.start and self.end == other.end


class WithSpan:
    """
    Value with a span.
    """

    def __init__(self, value, span: Span):
        self.value = value
        self.span = span

    def __repr__(self) -> str:
        return f"WithSpan({self.value!r}, {self.span!r})"
