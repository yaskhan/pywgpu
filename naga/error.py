class ShaderError(Exception):
    """
    Base class for shader errors.
    """

    def __init__(self, source: str, label: str = None, inner: Exception = None):
        self.source = source
        self.label = label
        self.inner = inner

    def __str__(self) -> str:
        message = f"Shader error"
        if self.label:
            message += f" in '{self.label}'"
        message += f":\n{self.source}"
        if self.inner:
            message += f"\n  Caused by: {self.inner}"
        return message
