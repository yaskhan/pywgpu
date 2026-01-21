from __future__ import annotations
from typing import Optional, Any, List
from .module import Module


class ValidationError(Exception):
    """
    An error that occurred during validation.
    """

    def __init__(self, message: str):
        self.message = message


class ValidationInfo:
    """
    Information about the validation of a module.
    """

    def __init__(self, errors: List[ValidationError]):
        self.errors = errors

    def is_ok(self) -> bool:
        return not self.errors


class Validator:
    def __init__(self, flags: int = 0) -> None:
        pass

    def validate(self, module: Module) -> ValidationInfo:
        """Validates the Naga module."""
        return ValidationInfo([])


class Translator:
    def translate(self, source: str, source_type: str, target_type: str) -> str:
        """Translates shader code between languages."""
        return source
