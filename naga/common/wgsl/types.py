from typing import Any

class TypeContext:
    """
    A context for printing Naga IR types as WGSL.
    """
    def __init__(self, types: Any):
        self.types = types

    def lookup_type(self, handle: Any) -> Any:
        return self.types[handle]

    def type_name(self, handle: Any) -> str:
        return self.types[handle].name or "{anonymous type}"

class WgslType:
    """
    WGSL specific types.
    """
    def __init__(self, inner: Any):
        self.inner = inner

    def to_wgsl(self, context: TypeContext) -> str:
        # This is a simplified implementation that only handles a few
        # of the possible types. A complete implementation would be
        # much more complex.
        if self.inner.kind == "scalar":
            return self.inner.name
        elif self.inner.kind == "vector":
            return f"vec{self.inner.size}<{self.inner.format.name}>"
        elif self.inner.kind == "matrix":
            return f"mat{self.inner.columns}x{self.inner.rows}<{self.inner.format.name}>"
        elif self.inner.kind == "struct":
            return context.type_name(self.inner.handle)
        else:
            return f"{{unsupported type: {self.inner.kind}}}"

