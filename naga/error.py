class ShaderError(Exception):
    """
    Base class for shader errors.
    """
    def __init__(self, source: str, label: str = None, inner: Exception = None):
        self.source = source
        self.label = label
        self.inner = inner
