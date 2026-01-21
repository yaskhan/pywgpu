from typing import Any
from .handle_set_map import Handle, HandleSet, HandleMap


class TypeTracer:
    """
    Traces types to determine which are used.
    """

    def __init__(
        self,
        overrides: Any,
        types_used: HandleSet,
        expressions_used: HandleSet,
        overrides_used: HandleSet,
    ) -> None:
        self.overrides = overrides
        self.types_used = types_used
        self.expressions_used = expressions_used
        self.overrides_used = overrides_used

    def trace_type(self, ty: Any) -> None:
        """Trace a type to determine usage."""
        type_inner = type(ty.inner).__name__

        # Types that do not contain handles
        if type_inner in [
            "Scalar",
            "Vector",
            "Matrix",
            "CooperativeMatrix",
            "Atomic",
            "ValuePointer",
            "Image",
            "Sampler",
            "AccelerationStructure",
            "RayQuery",
        ]:
            pass

        # Types that contain handles
        elif type_inner == "Array":
            self.types_used.insert(ty.inner.base)
            if hasattr(ty.inner.size, "handle"):
                handle = ty.inner.size.handle
                self.overrides_used.insert(handle)
                override = self.overrides.get(handle)
                if override:
                    self.types_used.insert(override.ty)
                    if override.init:
                        self.expressions_used.insert(override.init)

        elif type_inner == "BindingArray":
            self.types_used.insert(ty.inner.base)
            if hasattr(ty.inner.size, "handle"):
                handle = ty.inner.size.handle
                self.overrides_used.insert(handle)
                override = self.overrides.get(handle)
                if override:
                    self.types_used.insert(override.ty)
                    if override.init:
                        self.expressions_used.insert(override.init)

        elif type_inner == "Pointer":
            self.types_used.insert(ty.inner.base)

        elif type_inner == "Struct":
            for member in ty.inner.members:
                self.types_used.insert(member.ty)


class TypeCompactor:
    """
    Compacts types.
    """

    def __init__(self) -> None:
        pass

    def compact(self, types: Any, module: Any) -> None:
        """Compact types."""
        # Placeholder implementation
        pass

    def trace_type(self, types: Any, module: Any) -> None:
        """Trace types to determine usage."""
        # Placeholder implementation
        pass

    def adjust_type(self, ty: Any, module_map: Any) -> None:
        """Adjust handles in a type."""
        # Placeholder implementation
        pass
