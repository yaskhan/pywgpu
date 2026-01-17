from enum import Enum

class ErrorType(Enum):
    OUT_OF_MEMORY = "out-of-memory"
    VALIDATION = "validation"
    INTERNAL = "internal" # Not standard webgpu but in wgpu-core
    DEVICE_LOST = "device-lost" # Not an error per se in WebGPU but an event

class ErrorFilter(Enum):
    VALIDATION = "validation"
    OUT_OF_MEMORY = "out-of-memory"
    INTERNAL = "internal"
