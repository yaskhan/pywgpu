from ..api import Api as ApiProtocol


class Api(ApiProtocol):
    """Metal Backend/Api entry point."""

    def init(self) -> None:
        pass

    Instance = None
    Adapter = None
    Device = None
    Surface = None
