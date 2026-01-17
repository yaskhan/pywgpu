from ..api import Api as ApiProtocol

class Api(ApiProtocol):
    """DX12 Backend/Api entry point."""
    def init(self) -> None:
        pass
        
    Instance = None
    Adapter = None
    Device = None
    Surface = None
