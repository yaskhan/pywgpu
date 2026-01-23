"""
Naga Intermediate Representation (IR) module.

This module provides the core data structures for Naga's shader translation IR,
mirroring the structure from rust GPU infrastructure.
"""

# Type aliases
from typing import Final

# Core IR components
from .block import Block
from .type import (
    Type,
    TypeInner,
    TypeInnerType,
    ScalarKind,
    Scalar,
    VectorSize,
    Vector,
    Matrix,
    ArraySize,
    ArraySizeType,
    Array,
    StructMember,
    Struct,
    ImageDimension,
    Image,
    Sampler,
    AccelerationStructure,
    RayQuery,
    BindingArray,
    ValuePointer,
    Pointer,
    Atomic,
    CooperativeSize,
    CooperativeRole,
    CooperativeMatrix,
    Interpolation,
    Sampling,
    StorageAccess,
    StorageFormat,
    ImageClass,
    ImageClassType,
    ImageClassSampled,
    ImageClassDepth,
    ImageClassStorage,
)
from .composite_types import (
    ShaderStage,
    BuiltIn,
    BuiltInType,
    BuiltInPosition,
    BuiltInBarycentric,
    AddressSpace,
    Binding,
)
from .constant import Constant, Override, Literal, LiteralType
from .operators import (
    UnaryOperator,
    BinaryOperator,
    AtomicFunction,
    DerivativeControl,
    DerivativeAxis,
    RelationalFunction,
    MathFunction,
    SwizzleComponent,
    SampleLevel,
    ImageQuery,
    GatherMode,
    Direction,
    SubgroupOperation,
    CollectiveOperation,
    Barrier,
    ImageQueryType,
    ImageQuerySize,
)
from .function import Function, FunctionArgument, FunctionResult, LocalVariable
from .module import Module, EntryPoint, GlobalVariable
from .expression import Expression, ExpressionType
from .statement import Statement, StatementType

# Basic type aliases from mod.rs
Bytes: Final = int  # u8 in Rust - Number of bytes per scalar

__all__ = [
    # Type aliases
    "Bytes",
    # Type definitions
    "Type",
    "TypeInner",
    "TypeInnerType",
    "ScalarKind",
    "Scalar",
    "VectorSize",
    "Vector",
    "Matrix",
    "ArraySize",
    "ArraySizeType",
    "Array",
    "StructMember",
    "Struct",
    "ImageDimension",
    "Image",
    "Sampler",
    "AccelerationStructure",
    "RayQuery",
    "Binding",
    "BindingArray",
    "ValuePointer",
    "Pointer",
    "Atomic",
    # Cooperative matrix types
    "CooperativeSize",
    "CooperativeRole",
    "CooperativeMatrix",
    # Binding qualifiers
    "Interpolation",
    "Sampling",
    "StorageAccess",
    "StorageFormat",
    # Image class types
    "ImageClass",
    "ImageClassType",
    "ImageClassSampled",
    "ImageClassDepth",
    "ImageClassStorage",
    # Composite types
    "ShaderStage",
    "BuiltIn",
    "BuiltInType",
    "BuiltInPosition",
    "BuiltInBarycentric",
    "AddressSpace",
    # Constants
    "Constant",
    "Override",
    "Literal",
    "LiteralType",
    # Operators
    "UnaryOperator",
    "BinaryOperator",
    "AtomicFunction",
    "DerivativeControl",
    "DerivativeAxis",
    "RelationalFunction",
    "MathFunction",
    "SwizzleComponent",
    "SampleLevel",
    "ImageQuery",
    "ImageQueryType",
    "ImageQuerySize",
    "GatherMode",
    "Direction",
    "SubgroupOperation",
    "CollectiveOperation",
    "Barrier",
    # Core components
    "Block",
    "Function",
    "FunctionArgument",
    "FunctionResult",
    "LocalVariable",
    "Module",
    "EntryPoint",
    "Expression",
    "ExpressionType",
    "GlobalVariable",
    "Statement",
    "StatementType",
]
