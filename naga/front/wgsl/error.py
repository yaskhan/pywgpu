"""
WGSL error types and error handling.

Translated from wgpu-trunk/naga/src/front/wgsl/error.rs
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ParseError(Exception):
    """
    Error produced during WGSL parsing.
    
    This represents a single error with location information
    and can format itself with source context.
    """
    message: str
    labels: List[Tuple[int, int, str]]  # (start, end, label)
    notes: List[str]
    
    def emit_to_string(self, source: str) -> str:
        """
        Format the error with source code context.
        
        Args:
            source: The original WGSL source code
            
        Returns:
            Formatted error message with source context
        """
        lines = source.split('\n')
        result = [f"error: {self.message}"]
        
        for start, end, label in self.labels:
            # Find line and column
            line_num = 0
            char_count = 0
            
            for i, line in enumerate(lines):
                if char_count + len(line) >= start:
                    line_num = i
                    break
                char_count += len(line) + 1
            
            column = start - char_count
            error_line = lines[line_num] if line_num < len(lines) else ""
            
            result.append(f"  --> line {line_num + 1}, column {column + 1}")
            result.append(f"   |")
            result.append(f"{line_num + 1:3} | {error_line}")
            result.append(f"   | {' ' * column}{'^' * max(1, end - start)} {label}")
        
        for note in self.notes:
            result.append(f"   = note: {note}")
        
        return '\n'.join(result)


class Error:
    """
    Base class for WGSL parsing errors.
    
    Specific error types from Rust implementation:
    - Unexpected: Unexpected token
    - BadNumber: Invalid number literal
    - BadU32Constant: Invalid u32 constant
    - BadI32Constant: Invalid i32 constant
    - BadAccessor: Invalid accessor
    - BadTexture: Invalid texture type
    - BadTypeCast: Invalid type cast
    - BadTextureSampleType: Invalid texture sample type
    - BadIncrDecr: Invalid increment/decrement
    - BadAssignment: Invalid assignment
    - InvalidResolve: Invalid name resolution
    - InvalidForInitializer: Invalid for loop initializer
    - Redefinition: Redefinition of identifier
    - RecursiveDeclaration: Recursive declaration
    - CyclicDeclaration: Cyclic declaration dependency
    - InvalidConstructor: Invalid constructor
    - InvalidIdentifierUnderscore: Invalid underscore in identifier
    - ReservedKeyword: Use of reserved keyword
    - UnknownAddressSpace: Unknown address space
    - RepeatedAttribute: Repeated attribute
    - UnknownAttribute: Unknown attribute
    - UnknownBuiltin: Unknown builtin
    - UnknownAccess: Unknown access mode
    - UnknownShaderStage: Unknown shader stage
    - UnknownStorageFormat: Unknown storage format
    - UnknownConservativeDepth: Unknown conservative depth
    - UnknownType: Unknown type
    - ZeroSizeOrAlign: Zero size or alignment
    - InconsistentBinding: Inconsistent binding
    - UnknownLocalFunction: Unknown local function
    - TypeNotConstructible: Type not constructible
    - TypeNotInferrable: Type not inferrable
    - InitializationTypeMismatch: Initialization type mismatch
    - MissingType: Missing type annotation
    - MissingAttribute: Missing required attribute
    - InvalidAtomicPointer: Invalid atomic pointer
    - InvalidAtomicOperandType: Invalid atomic operand type
    - InvalidGatherComponent: Invalid gather component
    - InvalidGatherLevel: Invalid gather level
    - InvalidSampleLevel: Invalid sample level
    - InvalidBinaryOperandTypes: Invalid binary operand types
    - InvalidUnaryOperandType: Invalid unary operand type
    - InvalidWorkgroupUniformLoad: Invalid workgroup uniform load
    - NotPointer: Expected pointer type
    - NotReference: Expected reference type
    - InvalidAssignment: Invalid assignment target
    - InvalidComparisonOperandTypes: Invalid comparison operand types
    - InvalidSwitchValue: Invalid switch value
    - MultipleDefault: Multiple default cases
    - LastCaseFallThrough: Last case falls through
    - MissingDefaultCase: Missing default case
    - UnexpectedSwitchValue: Unexpected switch value
    - CalledEntryPoint: Called entry point function
    - WrongArgumentCount: Wrong argument count
    - FunctionReturnsVoid: Function returns void
    - InvalidReturnType: Invalid return type
    - MissingReturnValue: Missing return value
    - ExtraReturnValue: Extra return value
    - EntryPointNotFound: Entry point not found
    - Capability: Missing capability
    - UniformityViolation: Uniformity violation
    - Other: Other error
    """
    pass


def expected_token_error(expected: str, got: str, span: Tuple[int, int]) -> ParseError:
    """Create an error for unexpected token."""
    return ParseError(
        message=f"expected {expected}, found {got}",
        labels=[(span[0], span[1], "")],
        notes=[]
    )


def redefinition_error(name: str, previous: Tuple[int, int], current: Tuple[int, int]) -> ParseError:
    """Create an error for redefinition."""
    return ParseError(
        message=f"redefinition of '{name}'",
        labels=[
            (previous[0], previous[1], "previous definition here"),
            (current[0], current[1], "redefined here")
        ],
        notes=[]
    )


def unknown_type_error(name: str, span: Tuple[int, int]) -> ParseError:
    """Create an error for unknown type."""
    return ParseError(
        message=f"unknown type: '{name}'",
        labels=[(span[0], span[1], "")],
        notes=[]
    )
