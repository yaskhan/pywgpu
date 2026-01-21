import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class DataKind(str, Enum):
    BIN = "bin"
    WGSL = "wgsl"
    RON = "ron"
    SPV = "spv"
    DXIL = "dxil"
    HLSL = "hlsl"
    MSL = "msl"
    GLSL = "glsl"


class Data:
    """
    Trace data reference.
    """

    def __init__(self, value: Union[str, Dict[str, Any]]):
        self.value = value

    @classmethod
    def file(cls, name: str):
        return cls({"File": name})

    @classmethod
    def string(cls, kind: DataKind, data: str):
        return cls({"String": [kind.value, data]})

    @classmethod
    def binary(cls, kind: DataKind, data: bytes):
        return cls({"Binary": [kind.value, list(data)]})

    def to_dict(self):
        return self.value


class Trace:
    """
    Base class for trace recording.
    """

    def make_binary(self, kind: DataKind, data: bytes) -> Data:
        raise NotImplementedError()

    def make_string(self, kind: DataKind, data: str) -> Data:
        raise NotImplementedError()

    def add(self, action: Dict[str, Any]) -> None:
        raise NotImplementedError()


class DiskTrace(Trace):
    """
    Records trace to a directory on disk.
    """

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.trace_file_path = os.path.join(path, "trace.json")
        self.data_id = 0
        self.actions = []

        # Initialize the trace file with an empty list if it doesn't exist
        if not os.path.exists(self.trace_file_path):
            with open(self.trace_file_path, "w") as f:
                f.write("[]")

    def make_binary(self, kind: DataKind, data: bytes) -> Data:
        self.data_id += 1
        name = f"data{self.data_id}.{kind.value}"
        with open(os.path.join(self.path, name), "wb") as f:
            f.write(data)
        return Data.file(name)

    def make_string(self, kind: DataKind, data: str) -> Data:
        return self.make_binary(kind, data.encode("utf-8"))

    def add(self, action: Dict[str, Any]) -> None:
        # For simplicity in Python, we append to a list and write the whole thing
        # In a real high-performance system, we'd append to the file
        try:
            with open(self.trace_file_path, "r+") as f:
                f.seek(0)
                try:
                    actions = json.load(f)
                except json.JSONDecodeError:
                    actions = []
                actions.append(action)
                f.seek(0)
                json.dump(actions, f, indent=2)
                f.truncate()
        except Exception as e:
            logger.warning(f"Failed to record trace action: {e}")


class MemoryTrace(Trace):
    """
    Records trace to memory.
    """

    def __init__(self):
        self.actions_list = []

    def make_binary(self, kind: DataKind, data: bytes) -> Data:
        return Data.binary(kind, data)

    def make_string(self, kind: DataKind, data: str) -> Data:
        return Data.string(kind, data)

    def add(self, action: Dict[str, Any]) -> None:
        self.actions_list.append(action)

    def actions(self) -> List[Dict[str, Any]]:
        return self.actions_list


def to_trace_id(obj: Any) -> Union[int, str]:
    """
    Convert a resource object to a trace identifier.
    """
    if obj is None:
        return 0
    # Use individual ID if present (e.g., from our ID system)
    if hasattr(obj, "id"):
        return obj.id
    # Fallback to python id() or string label if available
    return str(id(obj))


# Helper to create specific actions
def action_init(desc: Dict[str, Any], backend: str) -> Dict[str, Any]:
    return {"action": "Init", "desc": desc, "backend": backend}


def action_create_buffer(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreateBuffer", "id": into_trace(id), "desc": desc}


def action_create_texture(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreateTexture", "id": into_trace(id), "desc": desc}


def action_create_bind_group_layout(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreateBindGroupLayout", "id": into_trace(id), "desc": desc}


def action_create_pipeline_layout(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreatePipelineLayout", "id": into_trace(id), "desc": desc}


def action_create_bind_group(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreateBindGroup", "id": into_trace(id), "desc": desc}


def action_create_shader_module(
    id: Any, desc: Dict[str, Any], data: Data
) -> Dict[str, Any]:
    return {
        "action": "CreateShaderModule",
        "id": into_trace(id),
        "desc": desc,
        "data": data.to_dict(),
    }


def action_create_compute_pipeline(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreateComputePipeline", "id": into_trace(id), "desc": desc}


def action_create_render_pipeline(id: Any, desc: Dict[str, Any]) -> Dict[str, Any]:
    return {"action": "CreateGeneralRenderPipeline", "id": into_trace(id), "desc": desc}


def action_submit(index: int, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"action": "Submit", "index": index, "commands": commands}


def action_write_buffer(
    id: Any, data: Data, range_start: int, range_end: int, queued: bool
) -> Dict[str, Any]:
    return {
        "action": "WriteBuffer",
        "id": into_trace(id),
        "data": data.to_dict(),
        "range": [range_start, range_end],
        "queued": queued,
    }


# Helper to serialize various objects to trace-friendly formats
def into_trace(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "to_trace"):
        return obj.to_trace()
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, list):
        return [into_trace(item) for item in obj]
    if isinstance(obj, dict):
        return {k: into_trace(v) for k, v in obj.items()}
    # If it's a resource object (has a 'raw' or 'id' attribute typically)
    return to_trace_id(obj)


__all__ = [
    "DataKind",
    "Data",
    "Trace",
    "DiskTrace",
    "MemoryTrace",
    "to_trace_id",
    "into_trace",
]
