"""
Naga Intermediate Representation (IR) module.

This module provides the core data structures for Naga's shader translation IR,
mirroring the structure from rust GPU infrastructure.
"""

# Core IR components
from .block import Block
from .type import (
    Type, TypeInner, ScalarKind, Scalar, VectorSize, Vector, Matrix,
    ArraySize, Array, StructMember, Struct, ImageDimension, Image
)
from .constant import Constant
from .function import (
    Function, FunctionArgument, FunctionResult, LocalVariable
)
from .module import Module, EntryPoint
from .expression import Expression
from .statement import Statement

__all__ = [
    'Block', 'Type', 'TypeInner', 'ScalarKind', 'Scalar', 'VectorSize', 'Vector', 'Matrix',
    'ArraySize', 'Array', 'StructMember', 'Struct', 'ImageDimension', 'Image',
    'Constant', 'Function', 'FunctionArgument', 'FunctionResult', 'LocalVariable',
    'Module', 'EntryPoint', 'Expression', 'Statement'
]