from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any, List


class StatementType(Enum):
    """
    IR Statement variant type.
    """
    EMIT = "emit"
    BLOCK = "block"
    IF = "if"
    SWITCH = "switch"
    LOOP = "loop"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    KILL = "kill"
    BARRIER = "barrier"
    STORE = "store"
    IMAGE_STORE = "image-store"
    ATOMIC = "atomic"
    IMAGE_ATOMIC = "image-atomic"
    WORKGROUP_UNIFORM_LOAD = "workgroup-uniform-load"
    CALL = "call"
    RAY_QUERY = "ray-query"
    SUBGROUP_BALLOT = "subgroup-ballot"
    SUBGROUP_GATHER = "subgroup-gather"
    SUBGROUP_COLLECTIVE_OPERATION = "subgroup-collective-operation"
    COOPERATIVE_STORE = "cooperative-store"


@dataclass(frozen=True, slots=True)
class SwitchCase:
    """
    A single case in a switch statement.
    """
    value: Optional[int]
    body: Any  # Block
    fall_through: bool


@dataclass(frozen=True, slots=True)
class Statement:
    """
    An IR statement that can be executed.
    Translated from wgpu-trunk/naga/src/ir/mod.rs
    """
    type: StatementType

    # Emit
    emit_expressions: Optional[List[int]] = None  # List[Handle<Expression>]

    # Block/If/Switch/Loop
    block: Optional[Any] = None  # Block
    if_condition: Optional[int] = None  # Handle<Expression>
    if_accept: Optional[Any] = None  # Block
    if_reject: Optional[Any] = None  # Block
    switch_selector: Optional[int] = None  # Handle<Expression>
    switch_cases: Optional[List[Any]] = None  # List[SwitchCase]
    switch_default: Optional[Any] = None  # Block
    loop_body: Optional[Any] = None  # Block
    loop_continuing: Optional[Any] = None  # Block
    loop_break_if: Optional[int] = None  # Option<Handle<Expression>>

    # Return
    return_value: Optional[int] = None  # Option<Handle<Expression>>

    # Barrier
    barrier: Optional[Any] = None  # Barrier

    # Store
    store_pointer: Optional[int] = None  # Handle<Expression>
    store_value: Optional[int] = None  # Handle<Expression>

    # ImageStore
    image_store_image: Optional[int] = None  # Handle<Expression>
    image_store_coordinate: Optional[int] = None  # Handle<Expression>
    image_store_array_index: Optional[int] = None  # Option<Handle<Expression>>
    image_store_value: Optional[int] = None  # Handle<Expression>

    # Atomic
    atomic_pointer: Optional[int] = None  # Handle<Expression>
    atomic_fun: Optional[Any] = None  # AtomicFunction
    atomic_value: Optional[int] = None  # Handle<Expression>
    atomic_result: Optional[int] = None  # Option<Handle<Expression>>

    # ImageAtomic
    image_atomic_image: Optional[int] = None  # Handle<Expression>
    image_atomic_coordinate: Optional[int] = None  # Handle<Expression>
    image_atomic_array_index: Optional[int] = None  # Option<Handle<Expression>>
    image_atomic_fun: Optional[Any] = None  # AtomicFunction
    image_atomic_value: Optional[int] = None  # Handle<Expression>
    image_atomic_result: Optional[int] = None  # Handle<Expression>

    # WorkGroupUniformLoad
    workgroup_uniform_load_pointer: Optional[int] = None  # Handle<Expression>
    workgroup_uniform_load_result: Optional[int] = None  # Handle<Expression>

    # Call
    call_function: Optional[int] = None  # Handle<Function>
    call_arguments: Optional[List[int]] = None  # List[Handle<Expression>]
    call_result: Optional[int] = None  # Option<Handle<Expression>>

    # RayQuery
    ray_query_query: Optional[int] = None  # Handle<Expression>
    ray_query_fun: Optional[Any] = None  # RayQueryFunction

    # SubgroupBallot
    subgroup_ballot_result: Optional[int] = None  # Handle<Expression>
    subgroup_ballot_predicate: Optional[int] = None  # Option<Handle<Expression>>

    # SubgroupGather
    subgroup_gather_mode: Optional[Any] = None  # GatherMode
    subgroup_gather_argument: Optional[int] = None  # Handle<Expression>
    subgroup_gather_result: Optional[int] = None  # Handle<Expression>

    # SubgroupCollectiveOperation
    subgroup_collective_op: Optional[Any] = None  # SubgroupOperation
    subgroup_collective_type: Optional[Any] = None  # CollectiveOperation
    subgroup_collective_argument: Optional[int] = None  # Handle<Expression>
    subgroup_collective_result: Optional[int] = None  # Handle<Expression>

    # CooperativeStore
    cooperative_store_pointer: Optional[int] = None  # Handle<Expression>
    cooperative_store_value: Optional[int] = None  # Handle<Expression>
    cooperative_store_matrix_type: Optional[int] = None  # Handle<Type>

    @classmethod
    def new_emit(cls, expressions: List[int]) -> "Statement":
        return cls(type=StatementType.EMIT, emit_expressions=expressions)

    @classmethod
    def new_block(cls, body: Any) -> "Statement":
        return cls(type=StatementType.BLOCK, block=body)

    @classmethod
    def new_if(cls, condition: int, accept: Any, reject: Any) -> "Statement":
        return cls(type=StatementType.IF, if_condition=condition, if_accept=accept, if_reject=reject)

    @classmethod
    def new_store(cls, pointer: int, value: int) -> "Statement":
        return cls(type=StatementType.STORE, store_pointer=pointer, store_value=value)

    @classmethod
    def new_return(cls, value: Optional[int] = None) -> "Statement":
        return cls(type=StatementType.RETURN, return_value=value)

    @classmethod
    def new_break(cls) -> "Statement":
        return cls(type=StatementType.BREAK)

    @classmethod
    def new_continue(cls) -> "Statement":
        return cls(type=StatementType.CONTINUE)

    @classmethod
    def new_kill(cls) -> "Statement":
        return cls(type=StatementType.KILL)

    @classmethod
    def new_atomic(cls, pointer: int, fun: Any, value: int, result: Optional[int] = None) -> "Statement":
        return cls(type=StatementType.ATOMIC, atomic_pointer=pointer, atomic_fun=fun, atomic_value=value, atomic_result=result)

    @classmethod
    def new_call(cls, function: int, arguments: List[int], result: Optional[int] = None) -> "Statement":
        return cls(type=StatementType.CALL, call_function=function, call_arguments=arguments, call_result=result)

    @classmethod
    def new_loop(cls, body: Any, continuing: Any, break_if: Optional[int] = None) -> "Statement":
        return cls(type=StatementType.LOOP, loop_body=body, loop_continuing=continuing, loop_break_if=break_if)

    @classmethod
    def new_switch(cls, selector: int, cases: List[SwitchCase], default: Optional[Any] = None) -> "Statement":
        return cls(type=StatementType.SWITCH, switch_selector=selector, switch_cases=cases, switch_default=default)
