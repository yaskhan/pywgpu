from enum import Enum, IntFlag
from typing import Optional
from pydantic import BaseModel

class BlendFactor(Enum):
    ZERO = "zero"
    ONE = "one"
    SRC = "src"
    ONE_MINUS_SRC = "one-minus-src"
    SRC_ALPHA = "src-alpha"
    ONE_MINUS_SRC_ALPHA = "one-minus-src-alpha"
    DST = "dst"
    ONE_MINUS_DST = "one-minus-dst"
    DST_ALPHA = "dst-alpha"
    ONE_MINUS_DST_ALPHA = "one-minus-dst-alpha"
    SRC_ALPHA_SATURATED = "src-alpha-saturated"
    CONSTANT = "constant"
    ONE_MINUS_CONSTANT = "one-minus-constant"

class BlendOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    REVERSE_SUBTRACT = "reverse-subtract"
    MIN = "min"
    MAX = "max"

class BlendComponent(BaseModel):
    src_factor: BlendFactor = BlendFactor.ONE
    dst_factor: BlendFactor = BlendFactor.ZERO
    operation: BlendOperation = BlendOperation.ADD

class BlendState(BaseModel):
    color: BlendComponent
    alpha: BlendComponent

class ColorWrite(IntFlag):
    RED = 1 << 0
    GREEN = 1 << 1
    BLUE = 1 << 2
    ALPHA = 1 << 3
    ALL = RED | GREEN | BLUE | ALPHA

class CompareFunction(Enum):
    NEVER = "never"
    LESS = "less"
    EQUAL = "equal"
    LESS_EQUAL = "less-equal"
    GREATER = "greater"
    NOT_EQUAL = "not-equal"
    GREATER_EQUAL = "greater-equal"
    ALWAYS = "always"

class StencilOperation(Enum):
    KEEP = "keep"
    ZERO = "zero"
    REPLACE = "replace"
    INVERT = "invert"
    INCREMENT_CLAMP = "increment-clamp"
    DECREMENT_CLAMP = "decrement-clamp"
    INCREMENT_WRAP = "increment-wrap"
    DECREMENT_WRAP = "decrement-wrap"

class StencilFaceState(BaseModel):
    compare: CompareFunction = CompareFunction.ALWAYS
    fail_op: StencilOperation = StencilOperation.KEEP
    depth_fail_op: StencilOperation = StencilOperation.KEEP
    pass_op: StencilOperation = StencilOperation.KEEP
    
class PrimitiveTopology(Enum):
    POINT_LIST = "point-list"
    LINE_LIST = "line-list"
    LINE_STRIP = "line-strip"
    TRIANGLE_LIST = "triangle-list"
    TRIANGLE_STRIP = "triangle-strip"

class FrontFace(Enum):
    CCW = "ccw"
    CW = "cw"

class CullMode(Enum):
    NONE = "none"
    FRONT = "front"
    BACK = "back"
