"""
Validation error types for Naga validator.

This module contains all error types that can be raised during validation,
mirroring the Rust implementation.
"""

from typing import Optional
from ..error import ShaderError


class ValidationError(ShaderError):
    """
    Base validation error.
    
    This is the main error type returned by the validator when a module
    fails validation.
    """
    pass


class ConstantError(ValidationError):
    """Constant validation error."""
    
    INITIALIZER_EXPR_TYPE = "Initializer must be a const-expression"
    INVALID_TYPE = "The type doesn't match the constant"
    NON_CONSTRUCTIBLE_TYPE = "The type is not constructible"
    
    def __init__(self, message: str, handle: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Initialize a constant error.
        
        Args:
            message: Error message describing the issue
            handle: Optional handle to the constant that failed validation
            name: Optional name of the constant
        """
        super().__init__(message)
        self.handle = handle
        self.name = name


class OverrideError(ValidationError):
    """Override validation error."""
    
    MISSING_NAME_AND_ID = "Override name and ID are missing"
    DUPLICATE_ID = "Override ID must be unique"
    INITIALIZER_EXPR_TYPE = "Initializer must be a const-expression or override-expression"
    INVALID_TYPE = "The type doesn't match the override"
    NON_CONSTRUCTIBLE_TYPE = "The type is not constructible"
    TYPE_NOT_SCALAR = "The type is not a scalar"
    NOT_ALLOWED = "Override declarations are not allowed"
    UNINITIALIZED_OVERRIDE = "Override is uninitialized"
    
    def __init__(self, message: str, handle: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Initialize an override error.
        
        Args:
            message: Error message describing the issue
            handle: Optional handle to the override that failed validation
            name: Optional name of the override
        """
        super().__init__(message)
        self.handle = handle
        self.name = name


class TypeError(ValidationError):
    """Type validation error."""
    
    INVALID_WIDTH = "Invalid type width"
    INVALID_SIZE = "Invalid type size"
    DISALIGNMENT = "Type alignment is incorrect"
    IMMEDIATE_ERROR = "Error with immediate type"
    
    def __init__(self, message: str, handle: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Initialize a type error.
        
        Args:
            message: Error message describing the issue
            handle: Optional handle to the type that failed validation
            name: Optional name of the type
        """
        super().__init__(message)
        self.handle = handle
        self.name = name


class GlobalVariableError(ValidationError):
    """Global variable validation error."""
    
    INVALID_TYPE = "Global variable has invalid type"
    INVALID_ADDRESS_SPACE = "Global variable has invalid address space"
    INVALID_BINDING = "Global variable has invalid binding"
    MISSING_BINDING = "Global variable is missing required binding"
    
    def __init__(self, message: str, handle: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Initialize a global variable error.
        
        Args:
            message: Error message describing the issue
            handle: Optional handle to the variable that failed validation
            name: Optional name of the variable
        """
        super().__init__(message)
        self.handle = handle
        self.name = name


class FunctionError(ValidationError):
    """Function validation error."""
    
    INVALID_ARGUMENT = "Function has invalid argument"
    INVALID_RESULT = "Function has invalid result type"
    INVALID_LOCAL_VARIABLE = "Function has invalid local variable"
    INVALID_EXPRESSION = "Function has invalid expression"
    INVALID_STATEMENT = "Function has invalid statement"
    
    def __init__(self, message: str, handle: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Initialize a function error.
        
        Args:
            message: Error message describing the issue
            handle: Optional handle to the function that failed validation
            name: Optional name of the function
        """
        super().__init__(message)
        self.handle = handle
        self.name = name


class ExpressionError(ValidationError):
    """Expression validation error."""
    
    INVALID_OPERAND = "Expression has invalid operand"
    INVALID_TYPE = "Expression has invalid type"
    TYPE_MISMATCH = "Expression type mismatch"
    
    def __init__(self, message: str, handle: Optional[int] = None) -> None:
        """
        Initialize an expression error.
        
        Args:
            message: Error message describing the issue
            handle: Optional handle to the expression that failed validation
        """
        super().__init__(message)
        self.handle = handle


class EntryPointError(ValidationError):
    """Entry point validation error."""
    
    INVALID_STAGE = "Entry point has invalid stage"
    INVALID_WORKGROUP_SIZE = "Entry point has invalid workgroup size"
    INVALID_ARGUMENT = "Entry point has invalid argument"
    INVALID_RESULT = "Entry point has invalid result"
    CONFLICTING_LOCATION = "Entry point has conflicting location"
    
    def __init__(self, message: str, stage: Optional[str] = None, name: Optional[str] = None) -> None:
        """
        Initialize an entry point error.
        
        Args:
            message: Error message describing the issue
            stage: Optional shader stage of the entry point
            name: Optional name of the entry point
        """
        super().__init__(message)
        self.stage = stage
        self.name = name


class InvalidHandleError(ValidationError):
    """Invalid handle error."""
    
    def __init__(self, handle_type: str, index: int) -> None:
        """
        Initialize an invalid handle error.
        
        Args:
            handle_type: Type of the invalid handle (e.g., "Type", "Function")
            index: Index that was out of bounds
        """
        super().__init__(f"Invalid {handle_type} handle at index {index}")
        self.handle_type = handle_type
        self.index = index


class LayoutError(ValidationError):
    """Layout calculation error."""
    
    INVALID_ALIGNMENT = "Invalid type alignment"
    INVALID_SIZE = "Invalid type size"
    SIZE_OVERFLOW = "Type size overflow"
    
    def __init__(self, message: str) -> None:
        """
        Initialize a layout error.
        
        Args:
            message: Error message describing the layout issue
        """
        super().__init__(message)


class ComposeError(ValidationError):
    """Compose operation error."""
    
    TYPE_MISMATCH = "Component types don't match composite type"
    COMPONENT_COUNT_MISMATCH = "Component count doesn't match expected count"
    
    def __init__(self, message: str) -> None:
        """
        Initialize a compose error.
        
        Args:
            message: Error message describing the compose issue
        """
        super().__init__(message)
