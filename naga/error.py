def replace_control_chars(s: str) -> str:
    """
    Replace control characters in a string with the Unicode replacement character.

    This is used for error reporting to avoid issues with control characters in source code.
    """
    REPLACEMENT_CHAR = "\uFFFD"

    result = []
    for char in s:
        if char.isprintable() or char in "\n\r\t":
            result.append(char)
        else:
            result.append(REPLACEMENT_CHAR)

    return "".join(result)


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
