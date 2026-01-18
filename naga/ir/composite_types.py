"""
Additional composite types and helper types for the IR.
"""

from enum import Enum
from typing import Any, Optional, Union, List

# Shader stage enum (matches original Rust)
class ShaderStage(Enum):
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"
    TASK = "task"
    MESH = "mesh"

# Address space enum (matches original Rust)
class AddressSpace(Enum):
    FUNCTION = "function"
    PRIVATE = "private"
    WORKGROUP = "workgroup"
    UNIFORM = "uniform"
    STORAGE = "storage"
    HANDLE = "handle"
    IMMEDIATE = "immediate"
    TASK_PAYLOAD = "task-payload"

# Built-in variables enum
class BuiltIn(Enum):
    POSITION = "position"
    VIEW_INDEX = "view-index"
    
    # Vertex shader
    BASE_INSTANCE = "base-instance"
    BASE_VERTEX = "base-vertex"
    CLIP_DISTANCE = "clip-distance"
    CULL_DISTANCE = "cull-distance"
    INSTANCE_INDEX = "instance-index"
    POINT_SIZE = "point-size"
    VERTEX_INDEX = "vertex-index"
    DRAW_ID = "draw-id"
    
    # Fragment shader
    FRAG_DEPTH = "frag-depth"
    POINT_COORD = "point-coord"
    FRONT_FACING = "front-facing"
    PRIMITIVE_INDEX = "primitive-index"
    SAMPLE_INDEX = "sample-index"
    SAMPLE_MASK = "sample-mask"
    
    # Compute shader
    GLOBAL_INVOCATION_ID = "global-invocation-id"
    LOCAL_INVOCATION_ID = "local-invocation-id"
    LOCAL_INVOCATION_INDEX = "local-invocation-index"
    WORKGROUP_ID = "workgroup-id"
    WORKGROUP_SIZE = "workgroup-size"
    NUM_WORKGROUPS = "num-workgroups"

# Interpolation qualifier
class Interpolation(Enum):
    PERSPECTIVE = "perspective"
    LINEAR = "linear"
    FLAT = "flat"
    PER_VERTEX = "per-vertex"

# Sampling qualifier
class Sampling(Enum):
    CENTER = "center"
    CENTROID = "centroid"
    SAMPLE = "sample"
    FIRST = "first"
    EITHER = "either"

# Storage format
class StorageFormat(Enum):
    R8_UNORM = "r8unorm"
    R8_SNORM = "r8snorm"
    R8_UINT = "r8uint"
    R8_SINT = "r8sint"
    R16_UINT = "r16uint"
    R16_SINT = "r16sint"
    R16_FLOAT = "r16float"
    R32_UINT = "r32uint"
    R32_SINT = "r32sint"
    R32_FLOAT = "r32float"

# Expression evaluation time
class EvaluationTime(Enum):
    EARLY = "early"  # Literals, constants
    FUNCTION_ENTRY = "function-entry"  # Function args, local vars
    MODULE_START = "module-start"  # Global vars
    CALL_RESULT = "call-result"  # Call result expressions
    EMIT_STATEMENT = "emit-statement"  # Emit statement

__all__ = [
    'ShaderStage', 'AddressSpace', 'BuiltIn', 'Interpolation', 'Sampling',
    'StorageFormat', 'EvaluationTime'
]