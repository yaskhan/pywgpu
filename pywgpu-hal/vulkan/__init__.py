from ..api import Api as ApiProtocol


class Api(ApiProtocol):
    """Vulkan Backend/Api entry point."""

    def init(self) -> None:
        pass

    # Implement Protocol properties
    Instance = None
    Adapter = None
    Device = None
    Surface = None
