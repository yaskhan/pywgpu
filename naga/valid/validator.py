from typing import Any, List
from ..error import ShaderError

class ValidationError(ShaderError):
    """
    Validation error.
    """
    pass

class Validator:
    """
    Naga validator.
    """
    def validate(self, module: Any) -> Any:
        pass
