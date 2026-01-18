"""naga.valid

Shader validation for Naga IR modules.

This package provides the :class:`Validator` class and related types for
validating Naga IR modules. The validator ensures modules are structurally
correct and semantically valid according to the Naga IR specification.

Example:
    >>> from naga.ir.module import Module
    >>> from naga.valid import Validator, ValidationFlags, Capabilities
    >>>
    >>> module = Module()
    >>> validator = Validator()
    >>> info = validator.validate(module)
    >>> print(f"Validated module with {len(info.functions)} functions")
"""

from .errors import (
    ComposeError,
    ConstantError,
    EntryPointError,
    ExpressionError,
    FunctionError,
    GlobalVariableError,
    InvalidHandleError,
    LayoutError,
    OverrideError,
    TypeError,
    ValidationError,
)
from .flags import (
    Capabilities,
    ShaderStages,
    SubgroupOperationSet,
    TypeFlags,
    ValidationFlags,
)
from .module_info import ExpressionInfo, FunctionInfo, ModuleInfo
from .validator import Validator

__all__ = [
    # Main validator class
    "Validator",
    # Module info types
    "ModuleInfo",
    "FunctionInfo",
    "ExpressionInfo",
    # Flags
    "ValidationFlags",
    "Capabilities",
    "SubgroupOperationSet",
    "ShaderStages",
    "TypeFlags",
    # Error types
    "ValidationError",
    "ConstantError",
    "OverrideError",
    "TypeError",
    "GlobalVariableError",
    "FunctionError",
    "ExpressionError",
    "EntryPointError",
    "InvalidHandleError",
    "LayoutError",
    "ComposeError",
]
