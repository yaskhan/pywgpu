from enum import Enum
from typing import Optional, Any


class Statement(Enum):
    """
    IR Statement enum.
    Translated from wgpu-trunk/naga/src/ir/mod.rs
    """

    # Emit a range of expressions
    EMIT = "emit"
    
    # A block containing more statements
    BLOCK = "block"
    
    # Conditionally executes one of two blocks
    IF = "if"
    
    # Conditionally executes one of multiple blocks (switch statement)
    SWITCH = "switch"
    
    # Executes a block repeatedly (loop)
    LOOP = "loop"
    
    # Exits the innermost enclosing Loop or Switch
    BREAK = "break"
    
    # Skips to the continuing block of the innermost enclosing Loop
    CONTINUE = "continue"
    
    # Returns from the function (possibly with a value)
    RETURN = "return"
    
    # Aborts the current shader execution
    KILL = "kill"
    
    # Synchronize invocations within the work group (control barrier)
    CONTROL_BARRIER = "control-barrier"
    
    # Synchronize invocations within the work group (memory barrier)
    MEMORY_BARRIER = "memory-barrier"
    
    # Stores a value at an address
    STORE = "store"
    
    # Stores a texel value to an image
    IMAGE_STORE = "image-store"
    
    # Atomic function on memory
    ATOMIC = "atomic"
    
    # Atomic operation on image texel
    IMAGE_ATOMIC = "image-atomic"
    
    # Load uniformly from a uniform pointer in the workgroup address space
    WORKGROUP_UNIFORM_LOAD = "workgroup-uniform-load"
    
    # Calls a function
    CALL = "call"
    
    # Ray query operation
    RAY_QUERY = "ray-query"
    
    # Calculate a bitmask using a boolean from each active thread in the subgroup
    SUBGROUP_BALLOT = "subgroup-ballot"
    
    # Gather a value from another active thread in the subgroup
    SUBGROUP_GATHER = "subgroup-gather"
    
    # Compute a collective operation across all active threads in the subgroup
    SUBGROUP_COLLECTIVE_OPERATION = "subgroup-collective-operation"
    
    # Store a cooperative primitive into memory
    COOPERATIVE_STORE = "cooperative-store"
