"""
Additional composite types and helper types for the IR.
Transcribed from wgpu/naga/src/ir/mod.rs
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ShaderStage(Enum):
    """Stage of the programmable pipeline."""
    VERTEX = "vertex"
    TASK = "task"
    MESH = "mesh"
    FRAGMENT = "fragment"
    COMPUTE = "compute"


@dataclass(frozen=True, slots=True)
class BuiltInPosition:
    """Written in vertex/mesh shaders, read in fragment shaders."""
    invariant: bool


@dataclass(frozen=True, slots=True)
class BuiltInBarycentric:
    """Read in fragment shaders."""
    perspective: bool


class BuiltInType(Enum):
    """Built-in inputs and outputs type."""
    POSITION = "position"
    VIEW_INDEX = "view-index"
    BASE_INSTANCE = "base-instance"
    BASE_VERTEX = "base-vertex"
    CLIP_DISTANCE = "clip-distance"
    CULL_DISTANCE = "cull-distance"
    INSTANCE_INDEX = "instance-index"
    POINT_SIZE = "point-size"
    VERTEX_INDEX = "vertex-index"
    DRAW_ID = "draw-id"
    FRAG_DEPTH = "frag-depth"
    POINT_COORD = "point-coord"
    FRONT_FACING = "front-facing"
    PRIMITIVE_INDEX = "primitive-index"
    BARYCENTRIC = "barycentric"
    SAMPLE_INDEX = "sample-index"
    SAMPLE_MASK = "sample-mask"
    GLOBAL_INVOCATION_ID = "global-invocation-id"
    LOCAL_INVOCATION_ID = "local-invocation-id"
    LOCAL_INVOCATION_INDEX = "local-invocation-index"
    WORKGROUP_ID = "workgroup-id"
    WORKGROUP_SIZE = "workgroup-size"
    NUM_WORKGROUPS = "num-workgroups"
    NUM_SUBGROUPS = "num-subgroups"
    SUBGROUP_ID = "subgroup-id"
    SUBGROUP_SIZE = "subgroup-size"
    SUBGROUP_INVOCATION_ID = "subgroup-invocation-id"
    MESH_TASK_SIZE = "mesh-task-size"
    CULL_PRIMITIVE = "cull-primitive"
    POINT_INDEX = "point-index"
    LINE_INDICES = "line-indices"
    TRIANGLE_INDICES = "triangle-indices"
    VERTEX_COUNT = "vertex-count"
    VERTICES = "vertices"
    PRIMITIVE_COUNT = "primitive-count"
    PRIMITIVES = "primitives"


@dataclass(frozen=True, slots=True)
class BuiltIn:
    """Built-in inputs and outputs."""
    type: BuiltInType
    position: Optional[BuiltInPosition] = None
    barycentric: Optional[BuiltInBarycentric] = None

    @classmethod
    def position(cls, invariant: bool = False) -> "BuiltIn":
        return cls(type=BuiltInType.POSITION, position=BuiltInPosition(invariant))

    @classmethod
    def barycentric(cls, perspective: bool) -> "BuiltIn":
        return cls(type=BuiltInType.BARYCENTRIC, barycentric=BuiltInBarycentric(perspective))


class AddressSpace(Enum):
    """Addressing space of variables."""
    FUNCTION = "function"
    PRIVATE = "private"
    WORKGROUP = "workgroup"
    UNIFORM = "uniform"
    STORAGE = "storage"
    HANDLE = "handle"
    IMMEDIATE = "immediate"
    TASK_PAYLOAD = "task-payload"


__all__ = [
    "ShaderStage",
    "BuiltIn",
    "BuiltInType",
    "BuiltInPosition",
    "BuiltInBarycentric",
    "AddressSpace",
]
