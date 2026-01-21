from typing import Any, Optional, List
from .bind_group import BindingResource


class Blas:
    def __init__(self, inner: Any, descriptor: Any) -> None:
        self._inner = inner
        self._descriptor = descriptor


class Tlas:
    def __init__(self, inner: Any, descriptor: Any) -> None:
        self._inner = inner
        self._descriptor = descriptor
        self._instances = [None] * descriptor.max_instances

    def __setitem__(self, index: int, instance: Optional[Any]) -> None:
        self._instances[index] = instance
        # In a real implementation, we would sync this to the backend
        if hasattr(self._inner, "set_instance"):
            self._inner.set_instance(index, instance)

    def __getitem__(self, index: int) -> Optional[Any]:
        return self._instances[index]

    def as_entire_binding(self) -> BindingResource:
        return BindingResource(self._inner)
