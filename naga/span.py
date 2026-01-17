class Span:
    """
    Source code span.
    """
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

class WithSpan:
    """
    Value with a span.
    """
    def __init__(self, value, span: Span):
        self.value = value
        self.span = span
