from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import ShaderModuleDescriptor


class ShaderModule:
    """
    Handle to a compiled shader module.

    Created with :meth:`Device.create_shader_module`.
    """

    def __init__(self, inner: Any, descriptor: ShaderModuleDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def get_compilation_info(self) -> Any:
        """
        Returns compilation info for this shader module.

        This is useful for retrieving error messages and warnings from
        compilation.
        """
        if hasattr(self._inner, "get_compilation_info"):
            return self._inner.get_compilation_info()
        else:
            raise NotImplementedError("Backend does not support get_compilation_info")
