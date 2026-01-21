from enum import Enum, IntFlag
from typing import Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field


class BlendFactor(Enum):
    """Alpha blend factor."""

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
    SRC1 = "src1"
    ONE_MINUS_SRC1 = "one-minus-src1"
    SRC1_ALPHA = "src1-alpha"
    ONE_MINUS_SRC1_ALPHA = "one-minus-src1-alpha"

    def ref_second_blend_source(self) -> bool:
        """
        Returns True if the blend factor references the second blend source.

        Note that the usage of those blend factors requires DUAL_SOURCE_BLENDING feature.
        """
        return self in {
            BlendFactor.SRC1,
            BlendFactor.ONE_MINUS_SRC1,
            BlendFactor.SRC1_ALPHA,
            BlendFactor.ONE_MINUS_SRC1_ALPHA,
        }


class BlendOperation(Enum):
    """Alpha blend operation."""

    ADD = "add"
    SUBTRACT = "subtract"
    REVERSE_SUBTRACT = "reverse-subtract"
    MIN = "min"
    MAX = "max"


class BlendComponent(BaseModel):
    """Describes a blend component of a BlendState."""

    src_factor: BlendFactor = BlendFactor.ONE
    dst_factor: BlendFactor = BlendFactor.ZERO
    operation: BlendOperation = BlendOperation.ADD

    def uses_constant(self) -> bool:
        """
        Returns true if the state relies on the constant color, which is
        set independently on a render command encoder.
        """
        return self.src_factor in {
            BlendFactor.CONSTANT,
            BlendFactor.ONE_MINUS_CONSTANT,
        } or self.dst_factor in {BlendFactor.CONSTANT, BlendFactor.ONE_MINUS_CONSTANT}


class BlendState(BaseModel):
    """Describe the blend state of a render pipeline."""

    color: BlendComponent
    alpha: BlendComponent

    @classmethod
    def REPLACE(cls) -> "BlendState":
        """Blend mode that does no color blending, just overwrites the output."""
        return cls(
            color=BlendComponent(
                src_factor=BlendFactor.ONE,
                dst_factor=BlendFactor.ZERO,
                operation=BlendOperation.ADD,
            ),
            alpha=BlendComponent(
                src_factor=BlendFactor.ONE,
                dst_factor=BlendFactor.ZERO,
                operation=BlendOperation.ADD,
            ),
        )

    @classmethod
    def ALPHA_BLENDING(cls) -> "BlendState":
        """Blend mode that does standard alpha blending with non-premultiplied alpha."""
        return cls(
            color=BlendComponent(
                src_factor=BlendFactor.SRC_ALPHA,
                dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                operation=BlendOperation.ADD,
            ),
            alpha=BlendComponent(
                src_factor=BlendFactor.ONE,
                dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                operation=BlendOperation.ADD,
            ),
        )

    @classmethod
    def PREMULTIPLIED_ALPHA_BLENDING(cls) -> "BlendState":
        """Blend mode that does standard alpha blending with premultiplied alpha."""
        over_component = BlendComponent(
            src_factor=BlendFactor.ONE,
            dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
            operation=BlendOperation.ADD,
        )
        return cls(color=over_component, alpha=over_component)


class ColorWrite(IntFlag):
    """Color write mask. Disabled color channels will not be written to."""

    RED = 1 << 0
    GREEN = 1 << 1
    BLUE = 1 << 2
    ALPHA = 1 << 3
    COLOR = RED | GREEN | BLUE
    ALL = RED | GREEN | BLUE | ALPHA


class ColorTargetState(BaseModel):
    """Describes the color state of a render pipeline."""

    format: str  # TextureFormat
    blend: Optional[BlendState] = None
    write_mask: int = ColorWrite.ALL

    @classmethod
    def from_format(cls, format: str) -> "ColorTargetState":
        """Create ColorTargetState from just a format."""
        return cls(format=format, blend=None, write_mask=ColorWrite.ALL)


class PrimitiveTopology(Enum):
    """Primitive type the input mesh is composed of."""

    POINT_LIST = "point-list"
    LINE_LIST = "line-list"
    LINE_STRIP = "line-strip"
    TRIANGLE_LIST = "triangle-list"
    TRIANGLE_STRIP = "triangle-strip"

    def is_strip(self) -> bool:
        """Returns true for strip topologies."""
        return self in {PrimitiveTopology.LINE_STRIP, PrimitiveTopology.TRIANGLE_STRIP}


class FrontFace(Enum):
    """Vertex winding order which classifies the "front" face of a triangle."""

    CCW = "ccw"
    CW = "cw"


class Face(Enum):
    """Face of a vertex."""

    FRONT = "front"
    BACK = "back"


class PolygonMode(Enum):
    """Type of drawing mode for polygons"""

    FILL = "fill"
    LINE = "line"
    POINT = "point"


class PrimitiveState(BaseModel):
    """Describes the state of primitive assembly and rasterization in a render pipeline."""

    topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST
    strip_index_format: Optional[str] = None  # IndexFormat
    front_face: FrontFace = FrontFace.CCW
    cull_mode: Optional[Face] = None
    unclipped_depth: bool = False
    polygon_mode: PolygonMode = PolygonMode.FILL
    conservative: bool = False


class MultisampleState(BaseModel):
    """Describes the multi-sampling state of a render pipeline."""

    count: int = 1
    mask: int = 0xFFFFFFFFFFFFFFFF  # !0 in u64
    alpha_to_coverage_enabled: bool = False


class IndexFormat(Enum):
    """Format of indices used with pipeline."""

    UINT16 = "uint16"
    UINT32 = "uint32"

    def byte_size(self) -> int:
        """Returns the size in bytes of the index format."""
        return 2 if self == IndexFormat.UINT16 else 4


class CompareFunction(Enum):
    """Comparison function used for depth and stencil operations."""

    NEVER = "never"
    LESS = "less"
    EQUAL = "equal"
    LESS_EQUAL = "less-equal"
    GREATER = "greater"
    NOT_EQUAL = "not-equal"
    GREATER_EQUAL = "greater-equal"
    ALWAYS = "always"

    def needs_ref_value(self) -> bool:
        """Returns true if the comparison depends on the reference value."""
        return self not in {CompareFunction.NEVER, CompareFunction.ALWAYS}


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
    """Describes stencil state in a render pipeline."""

    compare: CompareFunction = CompareFunction.ALWAYS
    fail_op: StencilOperation = StencilOperation.KEEP
    depth_fail_op: StencilOperation = StencilOperation.KEEP
    pass_op: StencilOperation = StencilOperation.KEEP

    @classmethod
    def IGNORE(cls) -> "StencilFaceState":
        """Ignore the stencil state for the face."""
        return cls(
            compare=CompareFunction.ALWAYS,
            fail_op=StencilOperation.KEEP,
            depth_fail_op=StencilOperation.KEEP,
            pass_op=StencilOperation.KEEP,
        )

    def needs_ref_value(self) -> bool:
        """Returns true if the face state uses the reference value for testing or operation."""
        return (
            self.compare.needs_ref_value()
            or self.fail_op == StencilOperation.REPLACE
            or self.depth_fail_op == StencilOperation.REPLACE
            or self.pass_op == StencilOperation.REPLACE
        )

    def is_read_only(self) -> bool:
        """Returns true if the face state doesn't mutate the target values."""
        return (
            self.pass_op == StencilOperation.KEEP
            and self.depth_fail_op == StencilOperation.KEEP
            and self.fail_op == StencilOperation.KEEP
        )


class StencilState(BaseModel):
    """State of the stencil operation (fixed-pipeline stage)."""

    front: StencilFaceState = StencilFaceState.IGNORE()
    back: StencilFaceState = StencilFaceState.IGNORE()
    read_mask: int = 0xFF
    write_mask: int = 0xFF

    def is_enabled(self) -> bool:
        """Returns true if the stencil test is enabled."""
        return (
            self.front != StencilFaceState.IGNORE()
            or self.back != StencilFaceState.IGNORE()
        ) and (self.read_mask != 0 or self.write_mask != 0)

    def is_read_only(self, cull_mode: Optional[Face]) -> bool:
        """Returns true if the state doesn't mutate the target values."""
        if self.write_mask == 0:
            return True

        front_ro = cull_mode == Face.FRONT or self.front.is_read_only()
        back_ro = cull_mode == Face.BACK or self.back.is_read_only()

        return front_ro and back_ro

    def needs_ref_value(self) -> bool:
        """Returns true if the stencil state uses the reference value for testing."""
        return self.front.needs_ref_value() or self.back.needs_ref_value()


@dataclass
class DepthBiasState:
    """Describes the biasing setting for the depth target."""

    constant: int = 0
    slope_scale: float = 0.0
    clamp: float = 0.0

    def is_enabled(self) -> bool:
        """Returns true if the depth biasing is enabled."""
        return self.constant != 0 or self.slope_scale != 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DepthBiasState):
            return False
        # For float comparison, we need to handle NaN properly
        import struct

        def float_to_bits(f: float) -> int:
            return struct.unpack("<Q", struct.pack("<d", f))[0]

        return (
            self.constant == other.constant
            and float_to_bits(self.slope_scale) == float_to_bits(other.slope_scale)
            and float_to_bits(self.clamp) == float_to_bits(other.clamp)
        )

    def __hash__(self) -> int:
        import struct

        def float_to_bits(f: float) -> int:
            return struct.unpack("<Q", struct.pack("<d", f))[0]

        return hash(
            (self.constant, float_to_bits(self.slope_scale), float_to_bits(self.clamp))
        )


class LoadOp(Enum):
    """
    Operation to perform to the output attachment at the start of a render pass.
    Corresponds to WebGPU GPULoadOp, plus the corresponding clearValue.

    Note: In Python, LoadOp.CLEAR should be used with a clear_value in Operations.
    """

    CLEAR = "clear"
    LOAD = "load"
    DONT_CARE = "dont-care"

    def eq_variant(self, other: "LoadOp") -> bool:
        """Returns true if variants are same (ignoring clear value)."""
        return self == other


class StoreOp(Enum):
    """
    Operation to perform to the output attachment at the end of a render pass.
    Corresponds to WebGPU GPUStoreOp.
    """

    STORE = "store"
    DISCARD = "discard"


@dataclass
class Operations(BaseModel):
    """
    Pair of load and store operations for an attachment aspect.

    This type is unique to the Rust API of wgpu. In the WebGPU specification,
    separate loadOp and storeOp fields are used instead.

    Attributes:
        load: Load operation (CLEAR, LOAD, or DONT_CARE)
        store: Store operation (STORE or DISCARD)
    """

    load: LoadOp = LoadOp.LOAD
    store: StoreOp = StoreOp.STORE

    def get_load_op(self) -> LoadOp:
        """Get the load operation."""
        return self.load

    def get_store_op(self) -> StoreOp:
        """Get the store operation."""
        return self.store


class DepthStencilState(BaseModel):
    """Describes the depth/stencil state in a render pipeline."""

    format: str  # TextureFormat
    depth_write_enabled: bool = False
    depth_compare: CompareFunction = CompareFunction.ALWAYS
    stencil: StencilState = StencilState()
    bias: DepthBiasState = DepthBiasState()

    def is_depth_enabled(self) -> bool:
        """Returns true if the depth testing is enabled."""
        return self.depth_compare != CompareFunction.ALWAYS or self.depth_write_enabled

    def is_depth_read_only(self) -> bool:
        """Returns true if the state doesn't mutate the depth buffer."""
        return not self.depth_write_enabled

    def is_stencil_read_only(self, cull_mode: Optional[Face]) -> bool:
        """Returns true if the state doesn't mutate the stencil."""
        return self.stencil.is_read_only(cull_mode)

    def is_read_only(self, cull_mode: Optional[Face]) -> bool:
        """Returns true if the state doesn't mutate either depth or stencil of the target."""
        return self.is_depth_read_only() and self.is_stencil_read_only(cull_mode)


class RenderBundleDepthStencil(BaseModel):
    """Describes the depth/stencil attachment for render bundles."""

    format: str  # TextureFormat
    depth_read_only: bool
    stencil_read_only: bool


class RenderBundleDescriptor(BaseModel):
    """Describes a RenderBundle."""

    label: Optional[str] = None


@dataclass
class DrawIndirectArgs:
    """Argument buffer layout for draw_indirect commands."""

    vertex_count: int
    instance_count: int
    first_vertex: int
    first_instance: int

    def as_bytes(self) -> bytes:
        """Return bytes representation of struct, ready to be written in a buffer."""
        import struct

        return struct.pack(
            "<4I",
            self.vertex_count,
            self.instance_count,
            self.first_vertex,
            self.first_instance,
        )


@dataclass
class DrawIndexedIndirectArgs:
    """Argument buffer layout for draw_indexed_indirect commands."""

    index_count: int
    instance_count: int
    first_index: int
    base_vertex: int
    first_instance: int

    def as_bytes(self) -> bytes:
        """Return bytes representation of struct, ready to be written in a buffer."""
        import struct

        return struct.pack(
            "<4Ii",
            self.index_count,
            self.instance_count,
            self.first_index,
            self.base_vertex,
            self.first_instance,
        )


@dataclass
class DispatchIndirectArgs:
    """Argument buffer layout for dispatch_indirect commands."""

    x: int
    y: int
    z: int

    def as_bytes(self) -> bytes:
        """Return bytes representation of struct, ready to be written into a buffer."""
        import struct

        return struct.pack("<3I", self.x, self.y, self.z)
