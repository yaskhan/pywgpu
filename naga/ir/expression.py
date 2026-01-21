"""
Expression definitions.
Transcribed from wgpu/naga/src/ir/mod.rs

Expression is a Single Static Assignment (SSA) scheme similar to SPIR-V.
When an Expression variant holds Handle<Expression> fields, they refer
to another expression in the same arena, unless explicitly noted otherwise.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        Type, Constant, Override, LocalVariable, Function,
        UnaryOperator, BinaryOperator, MathFunction, RelationalFunction,
        DerivativeAxis, DerivativeControl, SwizzleComponent, VectorSize,
        ScalarKind, LiteralType,
    )


class ExpressionType(Enum):
    """Expression variant type."""
    LITERAL = "literal"
    CONSTANT = "constant"
    OVERRIDE = "override"
    ZERO_VALUE = "zero_value"
    COMPOSE = "compose"
    ACCESS = "access"
    ACCESS_INDEX = "access_index"
    SPLAT = "splat"
    SWIZZLE = "swizzle"
    FUNCTION_ARGUMENT = "function_argument"
    GLOBAL_VARIABLE = "global_variable"
    LOCAL_VARIABLE = "local_variable"
    LOAD = "load"
    IMAGE_SAMPLE = "image_sample"
    IMAGE_LOAD = "image_load"
    IMAGE_QUERY = "image_query"
    UNARY = "unary"
    BINARY = "binary"
    SELECT = "select"
    DERIVATIVE = "derivative"
    RELATIONAL = "relational"
    MATH = "math"
    AS = "as"
    CALL_RESULT = "call_result"
    ATOMIC_RESULT = "atomic_result"
    WORKGROUP_UNIFORM_LOAD_RESULT = "workgroup_uniform_load_result"
    ARRAY_LENGTH = "array_length"
    RAY_QUERY_VERTEX_POSITIONS = "ray_query_vertex_positions"
    RAY_QUERY_PROCEED_RESULT = "ray_query_proceed_result"
    RAY_QUERY_GET_INTERSECTION = "ray_query_get_intersection"
    SUBGROUP_BALLOT_RESULT = "subgroup_ballot_result"
    SUBGROUP_OPERATION_RESULT = "subgroup_operation_result"
    COOPERATIVE_LOAD = "cooperative_load"
    COOPERATIVE_MULTIPLY_ADD = "cooperative_multiply_add"


@dataclass(frozen=True, slots=True)
class Expression:
    """An expression that can be evaluated to obtain a value."""
    type: ExpressionType

    # Literal
    literal: Optional[int | float | bool] = None  # Literal

    # Constant/Override/ZeroValue
    constant: Optional[int] = None  # Handle<Constant>
    override: Optional[int] = None  # Handle<Override>
    zero_value: Optional[int] = None  # Handle<Type>

    # Compose
    compose_ty: Optional[int] = None  # Handle<Type>
    compose_components: Optional[list[int]] = None  # Vec<Handle<Expression>>

    # Access/AccessIndex
    access_base: Optional[int] = None  # Handle<Expression>
    access_index: Optional[int] = None  # Handle<Expression>
    access_index_value: Optional[int] = None  # u32

    # Splat
    splat_size: Optional[object] = None  # VectorSize
    splat_value: Optional[int] = None  # Handle<Expression>

    # Swizzle
    swizzle_size: Optional[object] = None  # VectorSize
    swizzle_vector: Optional[int] = None  # Handle<Expression>
    swizzle_pattern: Optional[list[object]] = None  # [SwizzleComponent; 4]

    # FunctionArgument
    function_argument: Optional[int] = None  # u32

    # GlobalVariable/LocalVariable
    global_variable: Optional[int] = None  # Handle<GlobalVariable>
    local_variable: Optional[int] = None  # Handle<LocalVariable>

    # Load
    load_pointer: Optional[int] = None  # Handle<Expression>

    # ImageSample
    image_sample_image: Optional[int] = None  # Handle<Expression>
    image_sample_sampler: Optional[int] = None  # Handle<Expression>
    image_sample_gather: Optional[object] = None  # Option<SwizzleComponent>
    image_sample_coordinate: Optional[int] = None  # Handle<Expression>
    image_sample_array_index: Optional[int] = None  # Option<Handle<Expression>>
    image_sample_offset: Optional[int] = None  # Option<Handle<Expression>>
    image_sample_level: Optional[object] = None  # SampleLevel
    image_sample_depth_ref: Optional[int] = None  # Option<Handle<Expression>>
    image_sample_clamp_to_edge: Optional[bool] = None

    # ImageLoad
    image_load_image: Optional[int] = None  # Handle<Expression>
    image_load_coordinate: Optional[int] = None  # Handle<Expression>
    image_load_array_index: Optional[int] = None  # Option<Handle<Expression>>
    image_load_sample: Optional[int] = None  # Option<Handle<Expression>>
    image_load_level: Optional[int] = None  # Option<Handle<Expression>>

    # ImageQuery
    image_query_image: Optional[int] = None  # Handle<Expression>
    image_query_query: Optional[object] = None  # ImageQuery

    # Unary
    unary_op: Optional[object] = None  # UnaryOperator
    unary_expr: Optional[int] = None  # Handle<Expression>

    # Binary
    binary_op: Optional[object] = None  # BinaryOperator
    binary_left: Optional[int] = None  # Handle<Expression>
    binary_right: Optional[int] = None  # Handle<Expression>

    # Select
    select_condition: Optional[int] = None  # Handle<Expression>
    select_accept: Optional[int] = None  # Handle<Expression>
    select_reject: Optional[int] = None  # Handle<Expression>

    # Derivative
    derivative_axis: Optional[object] = None  # DerivativeAxis
    derivative_ctrl: Optional[object] = None  # DerivativeControl
    derivative_expr: Optional[int] = None  # Handle<Expression>

    # Relational
    relational_fun: Optional[object] = None  # RelationalFunction
    relational_argument: Optional[int] = None  # Handle<Expression>

    # Math
    math_fun: Optional[object] = None  # MathFunction
    math_arg: Optional[int] = None  # Handle<Expression>
    math_arg1: Optional[int] = None  # Option<Handle<Expression>>
    math_arg2: Optional[int] = None  # Option<Handle<Expression>>
    math_arg3: Optional[int] = None  # Option<Handle<Expression>>

    # As
    as_expr: Optional[int] = None  # Handle<Expression>
    as_kind: Optional[object] = None  # ScalarKind
    as_convert: Optional[int] = None  # Option<Bytes>

    # CallResult
    call_result: Optional[int] = None  # Handle<Function>

    # AtomicResult
    atomic_result_ty: Optional[int] = None  # Handle<Type>
    atomic_result_comparison: Optional[bool] = None

    # WorkGroupUniformLoadResult
    workgroup_uniform_load_result_ty: Optional[int] = None  # Handle<Type>

    # ArrayLength
    array_length: Optional[int] = None  # Handle<Expression>

    # RayQueryVertexPositions
    ray_query_vertex_positions_query: Optional[int] = None  # Handle<Expression>
    ray_query_vertex_positions_committed: Optional[bool] = None

    # RayQueryProceedResult
    ray_query_proceed_result: Optional[bool] = None

    # RayQueryGetIntersection
    ray_query_get_intersection_query: Optional[int] = None  # Handle<Expression>
    ray_query_get_intersection_committed: Optional[bool] = None

    # SubgroupBallotResult
    subgroup_ballot_result: Optional[bool] = None

    # SubgroupOperationResult
    subgroup_operation_result_ty: Optional[int] = None  # Handle<Type>

    # CooperativeLoad
    cooperative_load_columns: Optional[object] = None  # CooperativeSize
    cooperative_load_rows: Optional[object] = None  # CooperativeSize
    cooperative_load_role: Optional[object] = None  # CooperativeRole
    cooperative_load_data: Optional[object] = None  # CooperativeData

    # CooperativeMultiplyAdd
    cooperative_multiply_add_a: Optional[int] = None  # Handle<Expression>
    cooperative_multiply_add_b: Optional[int] = None  # Handle<Expression>
    cooperative_multiply_add_c: Optional[int] = None  # Handle<Expression>


__all__ = [
    "Expression",
    "ExpressionType",
]