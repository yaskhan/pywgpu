from typing import Tuple, Optional, List, Any
from .error import replace_control_chars


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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Span):
            return self.start == other.start and self.end == other.end
        return False

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def is_defined(self) -> bool:
        """Check whether this span was defined or is a default/unknown span."""
        return self.start != 0 or self.end != 0

    def to_range(self):
        """Convert self to a range if the span is not unknown."""
        if self.is_defined():
            return range(self.start, self.end)
        return None

    def location(self, source: str) -> "SourceLocation":
        """
        Return a SourceLocation for this span in the provided source.
        """
        prefix = source[:self.start]
        line_number = prefix.count('\n') + 1
        line_start = prefix.rfind('\n')
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        line_position = self.start - line_start + 1

        return SourceLocation(
            line_number=line_number,
            line_position=line_position,
            offset=self.start,
            length=self.end - self.start,
        )


class SourceLocation:
    """
    A human-readable representation for a span, tailored for text source.

    Roughly corresponds to the positional members of GPUCompilationMessage from
    the WebGPU specification.
    """

    def __init__(self, line_number: int, line_position: int, offset: int, length: int):
        self.line_number = line_number
        self.line_position = line_position
        self.offset = offset
        self.length = length

    def __repr__(self) -> str:
        return f"SourceLocation(line={self.line_number}, pos={self.line_position}, offset={self.offset}, len={self.length})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SourceLocation):
            return (
                self.line_number == other.line_number
                and self.line_position == other.line_position
                and self.offset == other.offset
                and self.length == other.length
            )
        return False

    def __hash__(self) -> int:
        return hash((self.line_number, self.line_position, self.offset, self.length))


# A source code span together with "context", a user-readable description of what part of the error it refers to.
SpanContext = Tuple[Span, str]


class WithSpan:
    """
    Value with a span.
    """

    def __init__(self, value, span: Span):
        self.value = value
        self.span = span

    def __repr__(self) -> str:
        return f"WithSpan({self.value!r}, {self.span!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WithSpan):
            return self.value == other.value and self.span == other.span
        return False

    def __hash__(self) -> int:
        return hash((self.value, self.span))

    def with_span(self, span: Span, description: str) -> "WithSpan":
        """
        Add a new span with description.
        """
        if span.is_defined():
            # For simplicity, we'll just return a new WithSpan with the first span
            # In a full implementation, we'd store multiple spans
            return WithSpan(self.value, span)
        return self

    def with_context(self, span_context: SpanContext) -> "WithSpan":
        """
        Add a SpanContext.
        """
        span, description = span_context
        return self.with_span(span, description)

    def location(self, source: str) -> Optional[SourceLocation]:
        """
        Return a SourceLocation for our first span, if we have one.
        """
        if not source or not self.span.is_defined():
            return None

        return self.span.location(source)

    def emit_to_string(self, source: str) -> str:
        """
        Emits a summary of the error to a string.
        """
        return self.emit_to_string_with_path(source, "wgsl")

    def emit_to_string_with_path(self, source: str, path: str) -> str:
        """
        Emits a summary of the error to a string.
        """
        if not source:
            return str(self.value)

        # Simplified error reporting
        result = [f"\nShader '{path}' parsing error:"]

        if self.span.is_defined():
            location = self.span.location(source)
            if location:
                result.append(f"  Line {location.line_number}, position {location.line_position}:")
                result.append(f"  {self.value}")

                # Show the relevant line
                lines = source.split('\n')
                if location.line_number <= len(lines):
                    line = lines[location.line_number - 1]
                    result.append(f"  {line}")
                    result.append(f"  {' ' * (location.line_position - 1)}^")
        else:
            result.append(f"  {self.value}")

        return "\n".join(result)
