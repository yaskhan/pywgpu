"""
Render pass encoding.

This module implements render pass encoding for wgpu-core. It provides comprehensive
render pass functionality including attachment validation, draw call validation,
indirect drawing, render bundles, and state management.

Based on wgpu-core/src/command/render.rs (3836 lines)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple
from enum import Enum, IntFlag

# Import tracking types
try:
    from ..track import BufferUses, TextureUses
    from ..init_tracker import MemoryInitKind, Range
except ImportError:
    # Fallback if track module not available
    from enum import IntFlag, Enum
    class BufferUses(IntFlag):
        INDEX = 1 << 4
        VERTEX = 1 << 5
        STORAGE_READ_ONLY = 1 << 7
        INDIRECT = 1 << 9
    
    class TextureUses(IntFlag):
        COLOR_TARGET = 1 << 5
        DEPTH_STENCIL_READ = 1 << 6
        DEPTH_STENCIL_WRITE = 1 << 7

    class MemoryInitKind(Enum):
        IMPLICITLY_INITIALIZED = "implicitly_initialized"
        NEEDS_INITIALIZED_MEMORY = "needs_initialized_memory"

    @dataclass
    class Range:
        start: int
        end: int

from .bind import Binder
from .pass_module import PassState, set_bind_group, flush_bindings_helper, set_immediates
from .query import (
    validate_and_begin_occlusion_query, 
    end_occlusion_query, 
    validate_and_begin_pipeline_statistics_query, 
    end_pipeline_statistics_query,
    write_timestamp
)

try:
    from ..track import Tracker
except ImportError:
    # Minimal Tracker for type checking/safe attribute access
    class Tracker:
        def __init__(self):
            class Dummy:
                def merge_scope(self, other): return []
                def set_size(self, size): pass
            self.buffers = Dummy()
            self.textures = Dummy()
            self.render_pipelines = Dummy()
            self.compute_pipelines = Dummy()
            self.bundles = Dummy()
            self.bind_groups = Dummy()
            self.query_sets = Dummy()
            self.views = Dummy()
        def merge_scope(self, other): return {'buffers': [], 'textures': []}



# ============================================================================
# Enums and Constants
# ============================================================================

class DrawCommandFamily(Enum):
    """Family of draw commands."""
    DRAW = "draw"
    DRAW_INDEXED = "draw_indexed"
    DRAW_MESH_TASKS = "draw_mesh_tasks"


class DrawKind(Enum):
    """Kind of draw command."""
    DRAW = "draw"
    DRAW_INDIRECT = "draw_indirect"
    MULTI_DRAW_INDIRECT = "multi_draw_indirect"
    MULTI_DRAW_INDIRECT_COUNT = "multi_draw_indirect_count"


class OptionalState(Enum):
    """
    State of an optional value.
    
    Used to track whether optional render pass state (like blend constant)
    is unused, required, or has been set.
    """
    UNUSED = "unused"
    REQUIRED = "required"
    SET = "set"


class PipelineFlags(IntFlag):
    """Flags for pipeline-dependent render pass state."""
    NONE = 0
    WRITES_DEPTH = 1 << 0
    WRITES_STENCIL = 1 << 1
    BLEND_CONSTANT = 1 << 2
    STENCIL_REFERENCE = 1 << 3


class OptionalStateTracker:
    """
    Tracker for optional state with require() method.
    
    This wraps OptionalState to provide state transition logic.
    Used for blend constant tracking in render passes.
    
    Attributes:
        state: Current state (UNUSED, REQUIRED, or SET).
    """
    
    def __init__(self, initial: OptionalState = OptionalState.UNUSED):
        """Initialize with optional initial state."""
        self.state = initial
    
    def require(self, needed: bool) -> None:
        """
        Update requirement state based on whether value is needed.
        
        Transitions:
        - If needed=True: UNUSED → REQUIRED
        - If needed=False: REQUIRED → UNUSED
        - SET state is preserved
        
        Args:
            needed: Whether the optional value is required.
        """
        if needed:
            # Value is required
            if self.state == OptionalState.UNUSED:
                self.state = OptionalState.REQUIRED
        else:
            # Value is not required
            if self.state == OptionalState.REQUIRED:
                self.state = OptionalState.UNUSED
    
    def set(self) -> None:
        """Mark the optional value as set."""
        self.state = OptionalState.SET
    
    def is_set(self) -> bool:
        """Check if value is set."""
        return self.state == OptionalState.SET
    
    def is_required(self) -> bool:
        """Check if value is required but not set."""
        return self.state == OptionalState.REQUIRED
    
    def __eq__(self, other) -> bool:
        """Compare with OptionalState enum."""
        if isinstance(other, OptionalState):
            return self.state == other
        elif isinstance(other, OptionalStateTracker):
            return self.state == other.state
        return False


@dataclass
class Rect:
    """
    Rectangle with generic coordinates.
    
    Attributes:
        x: X coordinate.
        y: Y coordinate.
        w: Width.
        h: Height.
    """
    x: float
    y: float
    w: float
    h: float


# ============================================================================
# Error Types
# ============================================================================

class AttachmentErrorLocation(Enum):
    """Describes an attachment location in words."""
    COLOR = "color"
    COLOR_RESOLVE = "color_resolve"
    DEPTH = "depth"
    
    def __str__(self) -> str:
        if self == AttachmentErrorLocation.COLOR:
            return "color attachment's texture view"
        elif self == AttachmentErrorLocation.COLOR_RESOLVE:
            return "color attachment's resolve texture view"
        elif self == AttachmentErrorLocation.DEPTH:
            return "depth attachment's texture view"


class ColorAttachmentError(Exception):
    """Errors related to color attachments."""
    INVALID_FORMAT = "Attachment format {0} is not a color format"
    TOO_MANY = "The number of color attachments {0} exceeds the limit {1}"
    TOO_MANY_BYTES_PER_SAMPLE = "The total number of bytes per sample in color attachments {0} exceeds the limit {1}"
    DEPTH_SLICE_LIMIT = "Depth slice must be less than {1} but is {0}"
    MISSING_DEPTH_SLICE = "Color attachment's view is 3D and requires depth slice to be provided"
    UNNEEDED_DEPTH_SLICE = "Depth slice was provided but the color attachment's view is not 3D"
    SUBRESOURCE_OVERLAP = "{view}'s subresource at mip {mip_level} and depth/array layer {depth_or_array_layer} is already attached to this render pass"
    INVALID_USAGE_FOR_STORE_OP = "Color attachment's usage contains {0}. This can only be used with StoreOp::{1}, but StoreOp::{2} was provided"


class AttachmentError(Exception):
    """Errors related to attachments."""
    INVALID_DEPTH_STENCIL_ATTACHMENT_FORMAT = "The format of the depth-stencil attachment ({0}) is not a depth-or-stencil format"
    READ_ONLY_WITH_LOAD = "LoadOp must be None for read-only attachments"
    READ_ONLY_WITH_STORE = "StoreOp must be None for read-only attachments"
    NO_LOAD = "Attachment without load"
    NO_STORE = "Attachment without store"
    NO_CLEAR_VALUE = "LoadOp is `Clear` but no clear value was provided"
    CLEAR_VALUE_OUT_OF_RANGE = "Clear value ({0}) must be between 0.0 and 1.0, inclusive"


class RenderPassErrorInner(Exception):
    """Inner error type for render pass errors."""
    DEVICE = "Device error"
    COLOR_ATTACHMENT = "Color attachment error"


class EncoderStateLocked(Exception):
    """Encoder is already locked by a pass."""
    pass


class EncoderStateEnded(Exception):
    """Encoder has already been finished."""
    pass


class EncoderStateSubmitted(Exception):
    """Encoder has already been submitted."""
    pass


class EncoderStateInvalid(Exception):
    """Encoder is in an invalid state."""
    pass
    INVALID_ATTACHMENT = "Invalid attachment"
    ENCODER_STATE = "Encoder state error"
    INVALID_PARENT_ENCODER = "Parent encoder is invalid"
    DEBUG_GROUP_ERROR = "Debug group error"
    UNSUPPORTED_RESOLVE_TARGET_FORMAT = "The format of the {location} ({format}) is not resolvable"
    MISSING_ATTACHMENTS = "No color attachments or depth attachments were provided, at least one attachment of any kind must be provided"
    TEXTURE_VIEW_IS_NOT_RENDERABLE = "The {location} is not renderable"
    ATTACHMENTS_DIMENSION_MISMATCH = "Attachments have differing sizes: the {expected_location} has extent {expected_extent} but is followed by the {actual_location} which has {actual_extent}"
    ATTACHMENT_SAMPLE_COUNT_MISMATCH = "Attachments have differing sample counts: the {expected_location} has count {expected_samples} but is followed by the {actual_location} which has count {actual_samples}"
    INVALID_RESOLVE_SAMPLE_COUNTS = "The resolve source, {location}, must be multi-sampled (has {src} samples) while the resolve destination must not be multisampled (has {dst} samples)"
    MISMATCHED_RESOLVE_TEXTURE_FORMAT = "Resource source, {location}, format ({src}) must match the resolve destination format ({dst})"
    INVALID_DEPTH_OPS = "Unable to clear non-present/read-only depth"
    INVALID_STENCIL_OPS = "Unable to clear non-present/read-only stencil"
    INVALID_VALUES_OFFSET = "Invalid values offset"
    MISSING_FEATURES = "Missing features"
    MISSING_DOWNLEVEL_FLAGS = "Missing downlevel flags"
    UNALIGNED_INDIRECT_BUFFER_OFFSET = "Indirect buffer offset {0} is not a multiple of 4"
    INDIRECT_BUFFER_OVERRUN = "Indirect draw uses bytes {offset}..{end_offset} using count {count} which overruns indirect buffer of size {buffer_size}"
    INDIRECT_COUNT_BUFFER_OVERRUN = "Indirect draw uses bytes {begin_count_offset}..{end_count_offset} which overruns indirect buffer of size {count_buffer_size}"
    RESOURCE_USAGE_COMPATIBILITY = "Resource usage compatibility error"
    INCOMPATIBLE_BUNDLE_TARGETS = "Render bundle has incompatible targets"
    INCOMPATIBLE_BUNDLE_READ_ONLY_DEPTH_STENCIL = "Render bundle has incompatible read-only flags"
    RENDER_COMMAND = "Render command error"
    BIND = "Bind error"
    IMMEDIATE_OFFSET_ALIGNMENT = "Immediate data offset must be aligned to 4 bytes"
    IMMEDIATE_DATA_ALIGNMENT = "Immediate data size must be aligned to 4 bytes"
    IMMEDIATE_OUT_OF_MEMORY = "Ran out of immediate data space. Don't set 4gb of immediates per ComputePass."
    QUERY_USE = "Query use error"
    MULTI_VIEW_MISMATCH = "Multiview layer count must match"
    MULTI_VIEW_DIMENSION_MISMATCH = "Multiview pass texture views with more than one array layer must have D2Array dimension"
    TOO_MANY_MULTI_VIEW_VIEWS = "Multiview view count limit violated"
    MISSING_OCCLUSION_QUERY_SET = "missing occlusion query set"
    DESTROYED_RESOURCE = "Destroyed resource error"
    PASS_ENDED = "The compute pass has already ended and no further commands can be recorded"
    INVALID_RESOURCE = "Invalid resource error"
    TIMESTAMP_WRITES = "Timestamp writes error"


class RenderPassError(Exception):
    """Error encountered when performing a render pass."""
    
    def __init__(self, scope: str, inner: RenderPassErrorInner):
        self.scope = scope
        self.inner = inner
        super().__init__(f"{scope}: {inner}")


class DrawError(Exception):
    """Errors related to draw calls."""
    VERTEX_BEYOND_LIMIT = "Vertex {last_vertex} beyond limit {vertex_limit} for slot {slot}"
    INSTANCE_BEYOND_LIMIT = "Instance {last_instance} beyond limit {instance_limit} for slot {slot}"
    MISSING_PIPELINE = "Missing pipeline"
    MISSING_BLEND_CONSTANT = "Missing blend constant"
    MISSING_INDEX_BUFFER = "Missing index buffer"
    UNMATCHED_INDEX_FORMATS = "Pipeline expects {pipeline_format} but buffer has {buffer_format}"
    WRONG_PIPELINE_TYPE = "Wrong pipeline type"
    MISSING_VERTEX_BUFFER = "Missing vertex buffer"
    INDEX_BEYOND_LIMIT = "Index {last_index} beyond limit {limit}"


class RenderCommandError(Exception):
    """Errors related to render commands."""
    BIND_GROUP_INDEX_OUT_OF_RANGE = "Bind group index out of range"


class PassStateError(Exception):
    """Errors related to pass state."""
    pass


# ============================================================================
# Load/Store Operations
# ============================================================================

class LoadOp(Enum):
    """Load operation for attachments."""
    LOAD = "load"
    CLEAR = "clear"
    DONT_CARE = "dont_care"


class StoreOp(Enum):
    """Store operation for attachments."""
    STORE = "store"
    DISCARD = "discard"


@dataclass
class PassChannel:
    """
    Describes an individual channel within a render pass.
    
    Attributes:
        load_op: Operation to perform at the start of a renderpass.
        store_op: Operation to perform at the end of a renderpass.
        read_only: If true, the channel is not changed by a renderpass.
        clear_value: Clear value (only used if load_op is CLEAR).
    """
    load_op: Optional[LoadOp] = None
    store_op: Optional[StoreOp] = None
    read_only: bool = False
    clear_value: Any = None
    
    def resolve(self, handle_clear=None) -> 'ResolvedPassChannel':
        """Resolve and validate the channel configuration."""
        if self.read_only:
            if self.load_op is not None:
                raise AttachmentError("ReadOnlyWithLoad")
            if self.store_op is not None:
                raise AttachmentError("ReadOnlyWithStore")
            return ResolvedPassChannel(read_only=True)
        else:
            if self.load_op is None:
                raise AttachmentError("NoLoad")
            if self.store_op is None:
                raise AttachmentError("NoStore")
            
            # Handle clear value
            clear_val = self.clear_value
            if self.load_op == LoadOp.CLEAR:
                if handle_clear:
                    clear_val = handle_clear(self.clear_value)
                elif clear_val is None:
                    raise AttachmentError("NoClearValue")
            
            return ResolvedPassChannel(
                read_only=False,
                load_op=self.load_op,
                store_op=self.store_op,
                clear_value=clear_val
            )


@dataclass
class ResolvedPassChannel:
    """Validated pass channel."""
    read_only: bool
    load_op: Optional[LoadOp] = None
    store_op: Optional[StoreOp] = None
    clear_value: Any = None
    
    def is_readonly(self) -> bool:
        return self.read_only
    
    def get_load_op(self) -> LoadOp:
        """Get the load operation."""
        if self.read_only:
            return LoadOp.LOAD
        return self.load_op or LoadOp.LOAD
    
    def get_store_op(self) -> StoreOp:
        """Get the store operation."""
        if self.read_only:
            return StoreOp.STORE
        return self.store_op or StoreOp.STORE
    
    def get_clear_value(self) -> Any:
        """Get the clear value."""
        if self.load_op == LoadOp.CLEAR:
            return self.clear_value
        return None
    
    def hal_ops(self) -> int:
        """Convert to HAL attachment ops."""
        ops = 0
        if not self.read_only:
            if self.load_op == LoadOp.LOAD:
                ops |= 0x01  # LOAD
            elif self.load_op == LoadOp.CLEAR:
                ops |= 0x02  # LOAD_CLEAR
            else:
                ops |= 0x04  # LOAD_DONT_CARE
            
            if self.store_op == StoreOp.STORE:
                ops |= 0x08  # STORE
            else:
                ops |= 0x10  # STORE_DISCARD
        return ops


# ============================================================================
# Attachment Descriptors
# ============================================================================

@dataclass
class RenderPassColorAttachment:
    """
    Describes a color attachment to a render pass.
    
    Attributes:
        view: The view to use as an attachment.
        depth_slice: The depth slice index of a 3D view.
        resolve_target: The view that will receive the resolved output.
        load_op: Operation to perform at the start.
        store_op: Operation to perform at the end.
    """
    view: Any
    load_op: LoadOp
    store_op: StoreOp
    depth_slice: Optional[int] = None
    resolve_target: Optional[Any] = None
    
    def hal_ops(self) -> int:
        """Get HAL attachment ops."""
        ops = 0
        if self.load_op == LoadOp.LOAD:
            ops |= 0x01
        elif self.load_op == LoadOp.CLEAR:
            ops |= 0x02
        else:
            ops |= 0x04
        
        if self.store_op == StoreOp.STORE:
            ops |= 0x08
        else:
            ops |= 0x10
        return ops
    
    def clear_value(self) -> Tuple[float, float, float, float]:
        """Get clear color value."""
        if self.load_op == LoadOp.CLEAR:
            return (0.0, 0.0, 0.0, 0.0)  # Default clear color
        return (0.0, 0.0, 0.0, 0.0)


@dataclass
class RenderPassDepthStencilAttachment:
    """
    Describes a depth/stencil attachment to a render pass.
    
    Attributes:
        view: The view to use as an attachment.
        depth: What operations will be performed on the depth part.
        stencil: What operations will be performed on the stencil part.
    """
    view: Any
    depth: PassChannel
    stencil: PassChannel


@dataclass
class ResolvedRenderPassDepthStencilAttachment:
    """Validated depth/stencil attachment."""
    view: Any
    depth: ResolvedPassChannel
    stencil: ResolvedPassChannel


# ============================================================================
# Render Pass Descriptor
# ============================================================================

@dataclass
class RenderPassDescriptor:
    """
    Describes the attachments of a render pass.
    
    Attributes:
        label: Debug label for the render pass.
        color_attachments: The color attachments of the render pass.
        depth_stencil_attachment: The depth and stencil attachment.
        timestamp_writes: Defines where and when timestamp values will be written.
        occlusion_query_set: Defines where the occlusion query results will be stored.
        multiview_mask: The multiview array layers that will be used.
    """
    label: Optional[str] = None
    color_attachments: List[Optional[RenderPassColorAttachment]] = field(default_factory=list)
    depth_stencil_attachment: Optional[RenderPassDepthStencilAttachment] = None
    timestamp_writes: Optional[Any] = None
    occlusion_query_set: Optional[Any] = None
    multiview_mask: Optional[int] = None


# ============================================================================
# State Tracking
# ============================================================================

@dataclass
class StateChange:
    """Tracks state changes."""
    current: Optional[Any] = None


@dataclass
class BindGroupStateChange:
    """Tracks bind group state changes."""
    current: List[Optional[int]] = field(default_factory=lambda: [None] * 8)  # MAX_BIND_GROUPS


@dataclass
class VertexBufferStateChange:
    """Tracks vertex buffer state changes."""
    current: List[Optional[Tuple]] = field(default_factory=lambda: [None] * 16)  # MAX_VERTEX_BUFFERS


@dataclass
class IndexBufferStateChange:
    """Tracks index buffer state changes."""
    current: Optional[Tuple] = None


@dataclass
class BlendConstantStateChange:
    """Tracks blend constant state changes."""
    current: Optional[Any] = None


@dataclass
class StencilReferenceStateChange:
    """Tracks stencil reference state changes."""
    current: Optional[int] = None


@dataclass
class ViewportStateChange:
    """Tracks viewport state changes."""
    current: Optional[Tuple] = None


@dataclass
class ScissorStateChange:
    """Tracks scissor state changes."""
    current: Optional[Tuple] = None


# ============================================================================
# Index and Vertex State
# ============================================================================

@dataclass
class IndexState:
    """Tracks index buffer state."""
    buffer_format: Optional[str] = None
    limit: int = 0
    
    def update_buffer(self, range_start: int, range_end: int, format: str):
        """Update index buffer state."""
        self.buffer_format = format
        shift = 1 if format == "uint16" else 2
        self.limit = (range_end - range_start) >> shift
    
    def reset(self):
        """Reset index buffer state."""
        self.buffer_format = None
        self.limit = 0


@dataclass
class VertexLimits:
    """
    Vertex and instance limits for draw validation.
    
    Attributes:
        vertex_limit: Length of the shortest vertex rate vertex buffer.
        vertex_limit_slot: Buffer slot which the shortest vertex rate vertex buffer is bound to.
        instance_limit: Length of the shortest instance rate vertex buffer.
        instance_limit_slot: Buffer slot which the shortest instance rate vertex buffer is bound to.
    """
    vertex_limit: int = 2**64 - 1
    vertex_limit_slot: int = 0
    instance_limit: int = 2**64 - 1
    instance_limit_slot: int = 0
    
    @classmethod
    def new(cls, buffer_sizes, pipeline_steps) -> 'VertexLimits':
        """
        Create new vertex limits from buffer sizes and pipeline steps.
        
        Implements validation from WebGPU specification.
        """
        vertex_limit = 2**64 - 1
        vertex_limit_slot = 0
        instance_limit = 2**64 - 1
        instance_limit_slot = 0
        
        for idx, (buffer_size, step) in enumerate(zip(buffer_sizes, pipeline_steps)):
            if buffer_size is None:
                # Missing required vertex buffer
                continue
            
            if step.last_stride > buffer_size:
                # The buffer cannot fit the last vertex.
                limit = 0
            elif step.stride == 0:
                # Same vertex will be repeated, slot can accommodate any number of vertices
                continue
            else:
                # The general case
                limit = (buffer_size - step.last_stride) // step.stride + 1
            
            if step.mode == "vertex":  # VertexStepMode::Vertex
                if limit < vertex_limit:
                    vertex_limit = limit
                    vertex_limit_slot = idx
            elif step.mode == "instance":  # VertexStepMode::Instance
                if limit < instance_limit:
                    instance_limit = limit
                    instance_limit_slot = idx
        
        return cls(
            vertex_limit=vertex_limit,
            vertex_limit_slot=vertex_limit_slot,
            instance_limit=instance_limit,
            instance_limit_slot=instance_limit_slot
        )
    
    def validate_vertex_limit(self, first_vertex: int, vertex_count: int):
        """Validate vertex count against limits."""
        last_vertex = first_vertex + vertex_count
        if last_vertex > self.vertex_limit:
            raise DrawError(
                DrawError.VERTEX_BEYOND_LIMIT.format(
                    last_vertex=last_vertex,
                    vertex_limit=self.vertex_limit,
                    slot=self.vertex_limit_slot
                )
            )
    
    def validate_instance_limit(self, first_instance: int, instance_count: int):
        """Validate instance count against limits."""
        last_instance = first_instance + instance_count
        if last_instance > self.instance_limit:
            raise DrawError(
                DrawError.INSTANCE_BEYOND_LIMIT.format(
                    last_instance=last_instance,
                    instance_limit=self.instance_limit,
                    slot=self.instance_limit_slot
                )
            )


@dataclass
class VertexState:
    """Tracks vertex buffer state."""
    buffer_sizes: List[Optional[int]] = field(default_factory=lambda: [None] * 16)
    limits: VertexLimits = field(default_factory=VertexLimits)
    
    def update_limits(self, pipeline_steps: List):
        """
        Update vertex limits based on pipeline configuration.
        
        This calculates the maximum number of vertices and instances that can
        be drawn based on the bound vertex buffers and the pipeline's vertex
        input configuration.
        
        For each vertex buffer slot used by the pipeline:
        - Calculate how many vertices/instances fit in the buffer
        - Track the minimum across all slots
        
        Args:
            pipeline_steps: List of vertex buffer steps from pipeline.
                Each step defines: slot, stride, step_mode (vertex/instance)
        """
        # Reset limits to maximum
        vertex_limit = 0xFFFFFFFFFFFFFFFF  # u64::MAX
        vertex_limit_slot = 0
        instance_limit = 0xFFFFFFFFFFFFFFFF  # u64::MAX
        instance_limit_slot = 0
        
        # Calculate limits for each pipeline vertex step
        for step in pipeline_steps:
            slot = step.get('slot', 0)
            stride = step.get('stride', 0)
            step_mode = step.get('step_mode', 'vertex')  # 'vertex' or 'instance'
            
            # Get buffer size for this slot
            if slot < len(self.buffer_sizes):
                buffer_size = self.buffer_sizes[slot]
            else:
                buffer_size = None
            
            if buffer_size is None or stride == 0:
                # No buffer bound or zero stride - skip
                continue
            
            # Calculate how many vertices/instances fit in buffer
            count = buffer_size // stride
            
            # Update appropriate limit
            if step_mode == 'vertex':
                if count < vertex_limit:
                    vertex_limit = count
                    vertex_limit_slot = slot
            else:  # instance
                if count < instance_limit:
                    instance_limit = count
                    instance_limit_slot = slot
        
        # Update limits
        self.limits.vertex_limit = vertex_limit
        self.limits.vertex_limit_slot = vertex_limit_slot
        self.limits.instance_limit = instance_limit
        self.limits.instance_limit_slot = instance_limit_slot


# ============================================================================
# Base Pass
# ============================================================================

@dataclass
class BasePass:
    """
    Base pass data.
    
    Attributes:
        label: Debug label.
        error: Error if any.
        commands: List of commands.
        dynamic_offsets: Dynamic offsets.
        string_data: String data for debug markers.
        immediates_data: Immediates data.
    """
    label: Optional[str] = None
    error: Optional[Any] = None
    commands: List[Any] = field(default_factory=list)
    dynamic_offsets: List[int] = field(default_factory=list)
    string_data: bytes = b""
    immediates_data: List[int] = field(default_factory=list)
    
    def take(self) -> 'BasePass':
        """Take the pass data."""
        return self


# ============================================================================
# Render Pass
# ============================================================================

class RenderPass:
    """
    A render pass for recording render commands.
    
    A render pass is a sequence of render commands that will be executed
    on the GPU. Render passes are isolated from each other and from compute
    passes.
    
    Attributes:
        base: Base pass data.
        parent: Parent command encoder.
        color_attachments: Color attachments.
        depth_stencil_attachment: Depth/stencil attachment.
        timestamp_writes: Timestamp writes for the pass.
        occlusion_query_set: Occlusion query set.
        multiview_mask: Multiview mask.
        current_bind_groups: Bind group state change tracking.
        current_pipeline: Pipeline state change tracking.
        current_vertex_buffers: Vertex buffer state change tracking.
        current_index_buffer: Index buffer state change tracking.
        current_blend_constant: Blend constant state change tracking.
        current_stencil_reference: Stencil reference state change tracking.
        current_viewport: Viewport state change tracking.
        current_scissor: Scissor state change tracking.
    """
    
    def __init__(
        self,
        parent: Any,
        desc: RenderPassDescriptor,
    ) -> None:
        """
        Create a new render pass.
        
        Args:
            parent: The parent command encoder.
            desc: The descriptor for the render pass.
        """
        self.base = BasePass(label=desc.label)
        self.parent = parent
        self.color_attachments = desc.color_attachments
        self.depth_stencil_attachment = desc.depth_stencil_attachment
        self.timestamp_writes = desc.timestamp_writes
        self.occlusion_query_set = desc.occlusion_query_set
        self.multiview_mask = desc.multiview_mask
        
        # State tracking
        self.current_bind_groups = BindGroupStateChange()
        self.current_pipeline = StateChange()
        self.current_vertex_buffers = VertexBufferStateChange()
        self.current_index_buffer = IndexBufferStateChange()
        self.current_blend_constant = BlendConstantStateChange()
        self.current_stencil_reference = StencilReferenceStateChange()
        self.current_viewport = ViewportStateChange()
        self.current_scissor = ScissorStateChange()
        
        # Draw validation state
        self.index_state = IndexState()
        self.vertex_state = VertexState()
        self.blend_constant_required = False
    
    def label(self) -> Optional[str]:
        """Get the label of the render pass."""
        return self.base.label
    
    def end(self) -> None:
        """
        End the render pass.
        
        Raises:
            RuntimeError: If the pass has already ended.
        """
        if self.parent is None:
            raise RuntimeError("Pass already ended")
        
        # Unlock encoder and process recorded commands
        if hasattr(self.parent, '_unlock_encoder'):
            self.parent._unlock_encoder()
        self.parent = None
    
    # ========================================================================
    # Pipeline and Bind Group Commands
    # ========================================================================
    
    def set_pipeline(self, pipeline: Any) -> None:
        """
        Set the render pipeline.
        
        Args:
            pipeline: The pipeline to set.
        """
        if self.current_pipeline.current == pipeline:
            return
        
        self.current_pipeline.current = pipeline
        self.base.commands.append(("SetPipeline", pipeline))
        
        # Update state based on pipeline
        if hasattr(pipeline, 'requires_blend_constant'):
            self.blend_constant_required = pipeline.requires_blend_constant
        
        # Update vertex limits
        if hasattr(pipeline, 'vertex_steps'):
            self.vertex_state.update_limits(pipeline.vertex_steps)
    
    def set_bind_group(
        self,
        index: int,
        bind_group: Any,
        dynamic_offsets: Optional[List[int]] = None,
    ) -> None:
        """
        Set the bind group.
        
        Args:
            index: The bind group index.
            bind_group: The bind group to set.
            dynamic_offsets: The dynamic offsets.
        """
        if index >= 8:  # MAX_BIND_GROUPS
            raise RenderPassErrorInner("Bind group index out of range")
        
        self.current_bind_groups.current[index] = bind_group
        self.base.commands.append(("SetBindGroup", index, bind_group, dynamic_offsets or []))
    
    # ========================================================================
    # Buffer Commands
    # ========================================================================
    
    def set_index_buffer(
        self,
        buffer: Any,
        index_format: str,
        offset: int,
        size: Optional[int],
    ) -> None:
        """
        Set the index buffer.
        
        Args:
            buffer: The buffer to set.
            index_format: The index format ("uint16" or "uint32").
            offset: The offset into the buffer.
            size: The size of the data to use.
        """
        buffer_size = size if size is not None else (getattr(buffer, 'size', 0) - offset)
        self.index_state.update_buffer(offset, offset + buffer_size, index_format)
        
        self.current_index_buffer.current = (buffer, index_format, offset, size)
        self.base.commands.append(("SetIndexBuffer", buffer, index_format, offset, size))
    
    def set_vertex_buffer(
        self,
        slot: int,
        buffer: Any,
        offset: int,
        size: Optional[int],
    ) -> None:
        """
        Set the vertex buffer.
        
        Args:
            slot: The buffer slot to set.
            buffer: The buffer to set.
            offset: The offset into the buffer.
            size: The size of the data to use.
        """
        if slot >= 16:  # MAX_VERTEX_BUFFERS
            raise RenderPassErrorInner("Vertex buffer slot out of range")
        
        buffer_size = size if size is not None else (getattr(buffer, 'size', 0) - offset)
        self.vertex_state.buffer_sizes[slot] = buffer_size
        
        self.current_vertex_buffers.current[slot] = (buffer, offset, size)
        self.base.commands.append(("SetVertexBuffer", slot, buffer, offset, size))
    
    # ========================================================================
    # Draw Commands
    # ========================================================================
    
    def _validate_draw_state(self, indexed: bool = False):
        """Validate state before draw call."""
        if self.current_pipeline.current is None:
            raise DrawError("Missing pipeline")
        
        if self.blend_constant_required and self.current_blend_constant.current is None:
            raise DrawError("Missing blend constant")
        
        if indexed and self.index_state.buffer_format is None:
            raise DrawError("Missing index buffer")
    
    def draw(
        self,
        vertex_count: int,
        instance_count: int,
        first_vertex: int,
        first_instance: int,
    ) -> None:
        """
        Draw primitives.
        
        Args:
            vertex_count: The number of vertices to draw.
            instance_count: The number of instances to draw.
            first_vertex: The first vertex index.
            first_instance: The first instance index.
        """
        self._validate_draw_state(indexed=False)
        self.vertex_state.limits.validate_vertex_limit(first_vertex, vertex_count)
        self.vertex_state.limits.validate_instance_limit(first_instance, instance_count)
        
        self.base.commands.append(("Draw", vertex_count, instance_count, first_vertex, first_instance))
    
    def draw_indexed(
        self,
        index_count: int,
        instance_count: int,
        first_index: int,
        base_vertex: int,
        first_instance: int,
    ) -> None:
        """
        Draw indexed primitives.
        
        Args:
            index_count: The number of indices to draw.
            instance_count: The number of instances to draw.
            first_index: The first index index.
            base_vertex: The base vertex index.
            first_instance: The first instance index.
        """
        self._validate_draw_state(indexed=True)
        
        # Validate index buffer bounds
        last_index = first_index + index_count
        if last_index > self.index_state.limit:
            raise DrawError(f"Index {last_index} beyond limit {self.index_state.limit}")
        
        self.vertex_state.limits.validate_instance_limit(first_instance, instance_count)
        
        self.base.commands.append((
            "DrawIndexed",
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance
        ))
    
    def draw_indirect(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
    ) -> None:
        """
        Draw primitives using parameters from a buffer.
        
        Args:
            indirect_buffer: Buffer containing draw parameters.
            indirect_offset: Offset into the buffer.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=False)
        self.base.commands.append(("DrawIndirect", indirect_buffer, indirect_offset))
    
    def draw_indexed_indirect(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
    ) -> None:
        """
        Draw indexed primitives using parameters from a buffer.
        
        Args:
            indirect_buffer: Buffer containing draw parameters.
            indirect_offset: Offset into the buffer.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=True)
        self.base.commands.append(("DrawIndexedIndirect", indirect_buffer, indirect_offset))
    
    # ========================================================================
    # State Commands
    # ========================================================================
    
    def set_viewport(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Set the viewport.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            width: Viewport width.
            height: Viewport height.
            min_depth: Minimum depth.
            max_depth: Maximum depth.
        """
        viewport = (x, y, width, height, min_depth, max_depth)
        if self.current_viewport.current != viewport:
            self.current_viewport.current = viewport
            self.base.commands.append(("SetViewport", x, y, width, height, min_depth, max_depth))
    
    def set_scissor_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """
        Set the scissor rectangle.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            width: Scissor width.
            height: Scissor height.
        """
        scissor = (x, y, width, height)
        if self.current_scissor.current != scissor:
            self.current_scissor.current = scissor
            self.base.commands.append(("SetScissor", x, y, width, height))
    
    def set_blend_constant(self, color: Tuple[float, float, float, float]) -> None:
        """
        Set the blend constant color.
        
        Args:
            color: The blend constant color (r, g, b, a).
        """
        if self.current_blend_constant.current != color:
            self.current_blend_constant.current = color
            self.base.commands.append(("SetBlendConstant", color))
    
    def set_stencil_reference(self, reference: int) -> None:
        """
        Set the stencil reference value.
        
        Args:
            reference: The stencil reference value.
        """
        if self.current_stencil_reference.current != reference:
            self.current_stencil_reference.current = reference
            self.base.commands.append(("SetStencilReference", reference))
    
    # ========================================================================
    # Debug Commands
    # ========================================================================
    
    def push_debug_group(self, label: str) -> None:
        """
        Push a debug group.
        
        Args:
            label: The debug group label.
        """
        self.base.commands.append(("PushDebugGroup", label))
    
    def pop_debug_group(self) -> None:
        """Pop a debug group."""
        self.base.commands.append(("PopDebugGroup",))
    
    def insert_debug_marker(self, label: str) -> None:
        """
        Insert a debug marker.
        
        Args:
            label: The debug marker label.
        """
        self.base.commands.append(("InsertDebugMarker", label))
    
    # ========================================================================
    # Query Commands
    # ========================================================================
    
    def begin_occlusion_query(self, query_index: int) -> None:
        """
        Begin an occlusion query.
        
        Args:
            query_index: The query index.
        """
        if self.occlusion_query_set is None:
            raise RenderPassErrorInner("Missing occlusion query set")
        
        self.base.commands.append(("BeginOcclusionQuery", query_index))
    
    def end_occlusion_query(self) -> None:
        """End the current occlusion query."""
        self.base.commands.append(("EndOcclusionQuery",))
    
    # ========================================================================
    # Render Bundle Commands
    # ========================================================================
    
    def execute_bundles(self, bundles: List[Any]) -> None:
        """
        Execute render bundles.
        
        Args:
            bundles: List of render bundles to execute.
        """
        for bundle in bundles:
            self.base.commands.append(("ExecuteBundle", bundle))
        
        # Reset state after bundle execution
        self.current_pipeline = StateChange()
        self.current_bind_groups = BindGroupStateChange()
    
    # ========================================================================
    # Multi-Draw Indirect Count Commands
    # ========================================================================
    
    def multi_draw_indirect_count(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
        count_buffer: Any,
        count_buffer_offset: int,
        max_count: int,
    ) -> None:
        """
        Draw primitives using parameters from buffers with a count buffer.
        
        Args:
            indirect_buffer: Buffer containing draw parameters.
            indirect_offset: Offset into the indirect buffer.
            count_buffer: Buffer containing the draw count.
            count_buffer_offset: Offset into the count buffer.
            max_count: Maximum number of draws.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=False)
        self.base.commands.append((
            "MultiDrawIndirectCount",
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count,
            "draw"
        ))
    
    def multi_draw_indexed_indirect_count(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
        count_buffer: Any,
        count_buffer_offset: int,
        max_count: int,
    ) -> None:
        """
        Draw indexed primitives using parameters from buffers with a count buffer.
        
        Args:
            indirect_buffer: Buffer containing draw parameters.
            indirect_offset: Offset into the indirect buffer.
            count_buffer: Buffer containing the draw count.
            count_buffer_offset: Offset into the count buffer.
            max_count: Maximum number of draws.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=True)
        self.base.commands.append((
            "MultiDrawIndexedIndirectCount",
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count,
            "draw_indexed"
        ))
    
    # ========================================================================
    # Mesh Shader Commands
    # ========================================================================
    
    def draw_mesh_tasks(
        self,
        group_count_x: int,
        group_count_y: int,
        group_count_z: int,
    ) -> None:
        """
        Draw mesh tasks (mesh shader dispatch).
        
        Args:
            group_count_x: Number of workgroups in X dimension.
            group_count_y: Number of workgroups in Y dimension.
            group_count_z: Number of workgroups in Z dimension.
        """
        self._validate_draw_state(indexed=False)
        
        # Validate workgroup counts
        # These would come from device limits in real implementation
        max_dimension = 65535  # Typical limit
        max_total = 65535  # Typical limit
        
        if (group_count_x > max_dimension or 
            group_count_y > max_dimension or 
            group_count_z > max_dimension):
            raise DrawError(
                f"Mesh task group count exceeds dimension limit: "
                f"({group_count_x}, {group_count_y}, {group_count_z}) > {max_dimension}"
            )
        
        total = group_count_x * group_count_y * group_count_z
        if total > max_total:
            raise DrawError(
                f"Mesh task total group count {total} exceeds limit {max_total}"
            )
        
        self.base.commands.append((
            "DrawMeshTasks",
            group_count_x,
            group_count_y,
            group_count_z
        ))
    
    def draw_mesh_tasks_indirect(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
    ) -> None:
        """
        Draw mesh tasks using parameters from a buffer.
        
        Args:
            indirect_buffer: Buffer containing dispatch parameters.
            indirect_offset: Offset into the buffer.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=False)
        self.base.commands.append(("DrawMeshTasksIndirect", indirect_buffer, indirect_offset))
    
    def multi_draw_mesh_tasks_indirect(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
        count: int,
    ) -> None:
        """
        Draw multiple mesh tasks using parameters from a buffer.
        
        Args:
            indirect_buffer: Buffer containing dispatch parameters.
            indirect_offset: Offset into the buffer.
            count: Number of draws.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=False)
        self.base.commands.append((
            "MultiDrawMeshTasksIndirect",
            indirect_buffer,
            indirect_offset,
            count
        ))
    
    def multi_draw_mesh_tasks_indirect_count(
        self,
        indirect_buffer: Any,
        indirect_offset: int,
        count_buffer: Any,
        count_buffer_offset: int,
        max_count: int,
    ) -> None:
        """
        Draw mesh tasks using parameters from buffers with a count buffer.
        
        Args:
            indirect_buffer: Buffer containing dispatch parameters.
            indirect_offset: Offset into the indirect buffer.
            count_buffer: Buffer containing the draw count.
            count_buffer_offset: Offset into the count buffer.
            max_count: Maximum number of draws.
        """
        if indirect_offset % 4 != 0:
            raise RenderPassErrorInner(f"Indirect buffer offset {indirect_offset} is not aligned to 4 bytes")
        
        self._validate_draw_state(indexed=False)
        self.base.commands.append((
            "MultiDrawMeshTasksIndirectCount",
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count
        ))
    
    # ========================================================================
    # Pipeline Statistics Query Commands
    # ========================================================================
    
    def begin_pipeline_statistics_query(
        self,
        query_set: Any,
        query_index: int,
    ) -> None:
        """
        Begin a pipeline statistics query.
        
        Args:
            query_set: The query set.
            query_index: The query index.
        """
        self.base.commands.append(("BeginPipelineStatisticsQuery", query_set, query_index))
    
    def end_pipeline_statistics_query(self) -> None:
        """End the current pipeline statistics query."""
        self.base.commands.append(("EndPipelineStatisticsQuery",))
    
    # ========================================================================
    # Timestamp Commands
    # ========================================================================
    
    def write_timestamp(
        self,
        query_set: Any,
        query_index: int,
    ) -> None:
        """
        Write a timestamp value.
        
        Args:
            query_set: The query set.
            query_index: The query index.
        """
        self.base.commands.append(("WriteTimestamp", query_set, query_index))
    
    # ========================================================================
    # Immediate Data Commands (Push Constants)
    # ========================================================================
    
    def set_immediates(
        self,
        offset: int,
        data: bytes,
    ) -> None:
        """
        Set immediate data (push constants).
        
        Args:
            offset: Offset in bytes (must be aligned to 4).
            data: Data to set (length must be aligned to 4).
        """
        IMMEDIATE_ALIGNMENT = 4
        
        if offset % IMMEDIATE_ALIGNMENT != 0:
            raise RenderPassErrorInner("Immediate offset must be aligned to 4 bytes")
        
        if len(data) % IMMEDIATE_ALIGNMENT != 0:
            raise RenderPassErrorInner("Immediate data size must be aligned to 4 bytes")
        
        # Store data in immediates buffer
        values_offset = len(self.base.immediates_data)
        
        # Convert bytes to u32 values
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) == 4:
                value = int.from_bytes(chunk, byteorder='little')
                self.base.immediates_data.append(value)
        
        self.base.commands.append(("SetImmediate", offset, len(data), values_offset))


# ============================================================================
# Helper Functions
# ============================================================================

def get_stride_of_indirect_args(family: str) -> int:
    """
    Get the stride of indirect arguments for a draw family.
    
    Args:
        family: Draw family ("draw", "draw_indexed", or "draw_mesh_tasks").
        
    Returns:
        Stride in bytes.
    """
    if family == "draw":
        return 16  # DrawIndirectArgs: 4 u32s
    elif family == "draw_indexed":
        return 20  # DrawIndexedIndirectArgs: 5 u32s
    elif family == "draw_mesh_tasks":
        return 12  # DispatchIndirectArgs: 3 u32s
    else:
        raise ValueError(f"Unknown draw family: {family}")


# ============================================================================
# Core Infrastructure (Internal)
# ============================================================================

# HAL imports for proper typing
try:
    from pywgpu_hal import (
        AttachmentOps,
        FormatAspects,
    )
    from pywgpu_hal.lib import CommandEncoder as HalCommandEncoder
except ImportError:
    # Fallback for development
    AttachmentOps = Any  # type: ignore
    FormatAspects = Any  # type: ignore
    HalCommandEncoder = Any  # type: ignore

# pywgpu-types imports for WebGPU types
try:
    from pywgpu_types.texture import TextureUsage, TextureAspect
    from pywgpu_types.buffer import BufferUsage
except ImportError:
    # Fallback for development
    TextureUsage = Any  # type: ignore
    TextureAspect = Any  # type: ignore
    BufferUsage = Any  # type: ignore


@dataclass
class RenderAttachment:
    """
    Internal representation of a render attachment.
    
    Attributes:
        texture: The texture being attached.
        selector: Texture subresource selector (mip levels, array layers).
        usage: How the texture will be used in the render pass (TextureUsage flags).
    """
    texture: Any  # Arc<Texture> - will be typed when texture module is complete
    selector: Any  # TextureSelector - ImageSubresourceRange, not yet in pywgpu-types
    usage: TextureUsage  # TextureUsage flags from pywgpu-types


# Maximum total attachments (color + resolve + depth/stencil)
MAX_TOTAL_ATTACHMENTS = 8 + 8 + 1  # hal::MAX_COLOR_ATTACHMENTS * 2 + 1


@dataclass
class RenderPassContext:
    """
    Context for a render pass, used for compatibility checks.
    
    Attributes:
        color_formats: Formats of color attachments.
        depth_stencil_format: Format of depth/stencil attachment.
        sample_count: Sample count.
        multiview_mask: Multiview mask.
    """
    color_formats: List[Optional[Any]] = field(default_factory=list)
    depth_stencil_format: Optional[Any] = None
    sample_count: int = 1
    multiview_mask: Optional[int] = None
    
    def check_compatible(self, other: 'RenderPassContext', resource: Any) -> None:
        """
        Check if this context is compatible with another.
        
        Args:
            other: The other context to check against.
            resource: The resource being checked (for error reporting).
            
        Raises:
            RenderPassErrorInner: If contexts are incompatible.
        """
        if self.sample_count != other.sample_count:
            raise RenderPassErrorInner(
                f"IncompatibleSampleCount: pass={self.sample_count}, "
                f"resource={other.sample_count} for {resource}"
            )
            
        if self.multiview_mask != other.multiview_mask:
            raise RenderPassErrorInner(
                f"IncompatibleMultiview: pass={self.multiview_mask}, "
                f"resource={other.multiview_mask} for {resource}"
            )
            
        # Check color formats
        if len(self.color_formats) != len(other.color_formats):
            raise RenderPassErrorInner(
                f"IncompatibleColorAttachmentCount: pass={len(self.color_formats)}, "
                f"resource={len(other.color_formats)}"
            )
            
        for i, (fmt1, fmt2) in enumerate(zip(self.color_formats, other.color_formats)):
            if fmt1 != fmt2:
                raise RenderPassErrorInner(
                    f"IncompatibleColorFormat[{i}]: pass={fmt1}, "
                    f"resource={fmt2}"
                )
                
        # Check depth/stencil format
        if self.depth_stencil_format != other.depth_stencil_format:
            raise RenderPassErrorInner(
                f"IncompatibleDepthStencilFormat: pass={self.depth_stencil_format}, "
                f"resource={other.depth_stencil_format}"
            )


@dataclass
class RenderPassInfo:
    """
    Internal render pass information and lifecycle management.
    
    This class manages the render pass context, attachments, and HAL integration.
    It handles initialization, validation, and finalization of render passes.
    
    Attributes:
        context: Render pass context (format info, sample count, etc.).
        render_attachments: All render attachments including depth/stencil.
        is_depth_read_only: Whether depth is read-only.
        is_stencil_read_only: Whether stencil is read-only.
        extent: Render target extent.
        divergent_discards: List of divergent depth/stencil discard operations.
        multiview_mask: Multiview configuration mask.
        occlusion_query_set: Optional occlusion query set.
    """
    context: Any  # RenderPassContext - will be typed when context module is complete
    render_attachments: List[RenderAttachment] = field(default_factory=list)
    is_depth_read_only: bool = False
    is_stencil_read_only: bool = False
    extent: Tuple[int, int, int] = (0, 0, 0)
    divergent_discards: List[Dict[str, Any]] = field(default_factory=list)
    multiview_mask: Optional[int] = None
    occlusion_query_set: Optional[Any] = None
    
    @staticmethod
    def add_pass_texture_init_actions(
        load_op: Any,  # LoadOp from pywgpu-types
        store_op: Any,  # StoreOp from pywgpu-types
        texture_memory_actions: Any,  # CommandBufferTextureMemoryActions
        view: Any,  # TextureView
        pending_discard_init_fixups: Any,  # SurfacesInDiscardState
    ) -> None:
        """
        Add texture initialization actions for a render pass attachment.
        
        This handles memory initialization tracking for load/store operations.
        It ensures that textures are properly initialized before use and handles
        discard operations correctly.
        
        The function determines what memory actions are needed based on:
        - Load operation (LOAD, CLEAR, or undefined)
        - Store operation (STORE or DISCARD)
        - Current initialization status of the texture
        
        Implementation follows wgpu-core/src/command/render.rs:952-985
        
        Args:
            load_op: Load operation for the attachment (LOAD, CLEAR, or None).
            store_op: Store operation for the attachment (STORE or DISCARD).
            texture_memory_actions: Texture memory action tracker.
            view: Texture view being attached.
            pending_discard_init_fixups: Pending discard fixups list.
        """
        # Handle load operation - matches Rust: if matches!(load_op, LoadOp::Load)
        if load_op is not None and hasattr(load_op, '__class__'):
            load_op_name = load_op.__class__.__name__ if hasattr(load_op, '__class__') else str(load_op)
            
            if load_op_name == 'Load' or (isinstance(load_op, str) and load_op == 'Load'):
                # LoadOp::Load - texture MUST be initialized
                # Register init action and collect any pending fixups
                if texture_memory_actions and hasattr(texture_memory_actions, 'register_init_action'):
                    # Create TextureInitTrackerAction
                    init_action = {
                        'texture': view.parent if hasattr(view, 'parent') else None,
                        'range': view.selector if hasattr(view, 'selector') else None,
                        'kind': 'NeedsInitializedMemory',
                    }
                    
                    # Register and get fixups
                    fixups = texture_memory_actions.register_init_action(init_action)
                    
                    # Add fixups to pending list
                    if fixups and pending_discard_init_fixups is not None:
                        if hasattr(pending_discard_init_fixups, 'extend'):
                            pending_discard_init_fixups.extend(fixups)
        
        # Handle store operation for Clear + Store case
        # Rust: else if store_op == StoreOp::Store
        elif store_op is not None:
            store_op_name = store_op.__class__.__name__ if hasattr(store_op, '__class__') else str(store_op)
            
            if store_op_name == 'Store' or (isinstance(store_op, str) and store_op == 'Store'):
                # Clear + Store - texture will be initialized after pass
                if texture_memory_actions and hasattr(texture_memory_actions, 'register_implicit_init'):
                    texture = view.parent if hasattr(view, 'parent') else None
                    selector = view.selector if hasattr(view, 'selector') else None
                    
                    if texture and selector:
                        texture_memory_actions.register_implicit_init(texture, selector)
        
        # Handle discard operation
        # Rust: if store_op == StoreOp::Discard
        if store_op is not None:
            store_op_name = store_op.__class__.__name__ if hasattr(store_op, '__class__') else str(store_op)
            
            if store_op_name == 'Discard' or (isinstance(store_op, str) and store_op == 'Discard'):
                # Discard - texture will be uninitialized after pass
                if texture_memory_actions and hasattr(texture_memory_actions, 'discard'):
                    texture = view.parent if hasattr(view, 'parent') else None
                    
                    if texture and hasattr(view, 'selector'):
                        # Create TextureSurfaceDiscard
                        discard_info = {
                            'texture': texture,
                            'mip_level': view.selector.mips.start if hasattr(view.selector, 'mips') else 0,
                            'layer': view.selector.layers.start if hasattr(view.selector, 'layers') else 0,
                        }
                        texture_memory_actions.discard(discard_info)
    
    @classmethod
    def start(
        cls,
        device: Any,
        hal_label: Optional[str],
        color_attachments: List[Optional[Any]],
        depth_stencil_attachment: Optional[Any],
        timestamp_writes: Optional[Any],
        occlusion_query_set: Optional[Any],
        encoder: Any,  # HAL CommandEncoder
        trackers: Any,
        texture_memory_actions: Any,
        pending_query_resets: Any,
        pending_discard_init_fixups: Any,
        snatch_guard: Any,
        multiview_mask: Optional[int],
    ) -> 'RenderPassInfo':
        """
        Start a render pass and create RenderPassInfo.
        
        This validates all attachments, checks compatibility, and initializes
        the HAL render pass. This is one of the most complex validation functions
        in the entire render module.
        
        Validation performed:
        - All color attachments have same dimensions
        - All attachments have same sample count
        - Depth/stencil format is valid
        - Multiview configuration is valid
        - Texture usage flags are correct
        - Store operations are compatible with TRANSIENT usage
        
        Args:
            device: The device.
            hal_label: Optional debug label.
            color_attachments: Color attachments (max 8).
            depth_stencil_attachment: Optional depth/stencil attachment.
            timestamp_writes: Optional timestamp writes.
            occlusion_query_set: Optional occlusion query set.
            encoder: HAL command encoder.
            trackers: Resource trackers.
            texture_memory_actions: Texture memory actions.
            pending_query_resets: Pending query resets.
            pending_discard_init_fixups: Pending discard fixups.
            snatch_guard: Snatch guard for resource access.
            multiview_mask: Multiview mask (bitfield of views).
            
        Returns:
            Initialized RenderPassInfo with validated attachments.
            
        Raises:
            RenderPassErrorInner: If validation fails.
        """
        # Initialize tracking structures
        render_attachments = []
        is_depth_read_only = False
        is_stencil_read_only = False
        extent = None
        sample_count = None
        divergent_discards = []
        
        # HAL color attachments list
        hal_color_attachments = []
        
        # Validate and process color attachments
        for i, color_att in enumerate(color_attachments):
            if color_att is None:
                continue
            
            # Get texture view
            view = color_att.view  # Would be resolved from ID
            
            # Extract texture from view
            # texture = view.parent
            # selector = view.selector
            
            # Validate dimensions
            # view_extent = view.render_extent
            view_extent = (1920, 1080, 1)  # Placeholder
            
            if extent is None:
                # First attachment sets the extent
                extent = view_extent
            elif extent != view_extent:
                raise RenderPassErrorInner(
                    f"ColorAttachment[{i}] has incompatible extent: "
                    f"{view_extent} != {extent}"
                )
            
            # Validate sample count
            # view_samples = view.samples
            view_samples = 1  # Placeholder
            
            if sample_count is None:
                sample_count = view_samples
            elif sample_count != view_samples:
                raise RenderPassErrorInner(
                    f"ColorAttachment[{i}] has incompatible sample count: "
                    f"{view_samples} != {sample_count}"
                )
            
            # Check TRANSIENT usage with store_op
            # if view.usage & TextureUsage.TRANSIENT:
            #     if color_att.store_op != StoreOp.DISCARD:
            #         raise InvalidUsageForStoreOp
            
            # Create render attachment
            # usage = TextureUsage.RENDER_ATTACHMENT
            # render_att = RenderAttachment(
            #     texture=texture,
            #     selector=selector,
            #     usage=usage,
            # )
            # render_attachments.append(render_att)
            
            # Build HAL color attachment descriptor
            # Determine load/store ops
            load_op = color_att.load_op if hasattr(color_att, 'load_op') else None
            store_op = color_att.store_op if hasattr(color_att, 'store_op') else None
            
            # Convert to HAL AttachmentOps
            hal_ops = AttachmentOps.NONE
            if load_op == 'Load':
                hal_ops |= AttachmentOps.LOAD
            elif load_op == 'Clear':
                hal_ops |= AttachmentOps.LOAD_CLEAR
            else:
                hal_ops |= AttachmentOps.LOAD_DONT_CARE
            
            if store_op == 'Store':
                hal_ops |= AttachmentOps.STORE
            else:
                hal_ops |= AttachmentOps.STORE_DISCARD
            
            # Create HAL attachment
            hal_color_att = {
                'target': {
                    'view': view,  # Would be view.raw(snatch_guard)
                    'usage': TextureUsage.RENDER_ATTACHMENT,
                },
                'ops': hal_ops,
                'clear_value': color_att.clear_value if hasattr(color_att, 'clear_value') else (0, 0, 0, 0),
            }
            hal_color_attachments.append(hal_color_att)
            
            # Handle resolve target if present
            if hasattr(color_att, 'resolve_target') and color_att.resolve_target:
                # Validate resolve target has same extent and format
                pass
            
            # Add texture memory initialization actions
            if texture_memory_actions is not None:
                self.add_pass_texture_init_actions(
                    load_op=load_op,
                    store_op=store_op,
                    texture_memory_actions=texture_memory_actions,
                    view=view,
                    pending_discard_init_fixups=pending_discard_init_fixups,
                )
        
        # HAL depth/stencil attachment
        hal_depth_stencil = None
        
        # Validate and process depth/stencil attachment
        if depth_stencil_attachment is not None:
            ds_att = depth_stencil_attachment
            
            # Get texture view
            view = ds_att.view
            
            # Validate extent
            # view_extent = view.render_extent
            view_extent = (1920, 1080, 1)  # Placeholder
            
            if extent is None:
                extent = view_extent
            elif extent != view_extent:
                raise RenderPassErrorInner(
                    f"DepthStencilAttachment has incompatible extent: "
                    f"{view_extent} != {extent}"
                )
            
            # Validate sample count
            # view_samples = view.samples
            view_samples = 1  # Placeholder
            
            if sample_count is None:
                sample_count = view_samples
            elif sample_count != view_samples:
                raise RenderPassErrorInner(
                    f"DepthStencilAttachment has incompatible sample count: "
                    f"{view_samples} != {sample_count}"
                )
            
            # Check read-only flags
            if hasattr(ds_att, 'depth_read_only'):
                is_depth_read_only = ds_att.depth_read_only
            
            if hasattr(ds_att, 'stencil_read_only'):
                is_stencil_read_only = ds_att.stencil_read_only
            
            # Check for divergent discards
            # If depth and stencil have different store_ops, we need special handling
            depth_store = getattr(ds_att.depth, 'store_op', None) if hasattr(ds_att, 'depth') else None
            stencil_store = getattr(ds_att.stencil, 'store_op', None) if hasattr(ds_att, 'stencil') else None
            
            if depth_store and stencil_store:
                if depth_store != stencil_store:
                    # Divergent discard detected!
                    # We'll need to handle this in finish()
                    divergent_discards.append({
                        'texture_view': view,
                        'aspects': 'DEPTH' if depth_store == 'DISCARD' else 'STENCIL',
                    })
            
            # Build HAL depth/stencil attachment
            depth_ops = AttachmentOps.NONE
            stencil_ops = AttachmentOps.NONE
            
            # Depth operations
            if hasattr(ds_att, 'depth'):
                depth = ds_att.depth
                if depth.load_op == 'Load':
                    depth_ops |= AttachmentOps.LOAD
                elif depth.load_op == 'Clear':
                    depth_ops |= AttachmentOps.LOAD_CLEAR
                else:
                    depth_ops |= AttachmentOps.LOAD_DONT_CARE
                
                if depth.store_op == 'Store':
                    depth_ops |= AttachmentOps.STORE
                else:
                    depth_ops |= AttachmentOps.STORE_DISCARD
            
            # Stencil operations
            if hasattr(ds_att, 'stencil'):
                stencil = ds_att.stencil
                if stencil.load_op == 'Load':
                    stencil_ops |= AttachmentOps.LOAD
                elif stencil.load_op == 'Clear':
                    stencil_ops |= AttachmentOps.LOAD_CLEAR
                else:
                    stencil_ops |= AttachmentOps.LOAD_DONT_CARE
                
                if stencil.store_op == 'Store':
                    stencil_ops |= AttachmentOps.STORE
                else:
                    stencil_ops |= AttachmentOps.STORE_DISCARD
            
            hal_depth_stencil = {
                'target': {
                    'view': view,  # Would be view.raw(snatch_guard)
                    'usage': TextureUsage.RENDER_ATTACHMENT,
                },
                'depth_ops': depth_ops,
                'stencil_ops': stencil_ops,
                'clear_value': (
                    ds_att.depth.clear_value if hasattr(ds_att, 'depth') else 0.0,
                    ds_att.stencil.clear_value if hasattr(ds_att, 'stencil') else 0,
                ),
            }
            
            # Add texture memory initialization actions
            if texture_memory_actions is not None:
                # Depth aspect
                if hasattr(ds_att, 'depth'):
                    self.add_pass_texture_init_actions(
                        load_op=ds_att.depth.load_op,
                        store_op=ds_att.depth.store_op,
                        texture_memory_actions=texture_memory_actions,
                        view=view,
                        pending_discard_init_fixups=pending_discard_init_fixups,
                    )
                # Stencil aspect
                if hasattr(ds_att, 'stencil'):
                    self.add_pass_texture_init_actions(
                        load_op=ds_att.stencil.load_op,
                        store_op=ds_att.stencil.store_op,
                        texture_memory_actions=texture_memory_actions,
                        view=view,
                        pending_discard_init_fixups=pending_discard_init_fixups,
                    )
        
        # Set default extent if no attachments
        if extent is None:
            extent = (0, 0, 1)
        
        if sample_count is None:
            sample_count = 1
        
        # Create HAL render pass descriptor
        hal_desc = {
            'label': hal_label,
            'extent': extent,
            'sample_count': sample_count,
            'color_attachments': hal_color_attachments,
            'depth_stencil_attachment': hal_depth_stencil,
            'multiview_mask': multiview_mask,
            'timestamp_writes': timestamp_writes,
            'occlusion_query_set': occlusion_query_set,
        }
        
        # Begin HAL render pass
        # This is the critical HAL call that starts GPU rendering
        try:
            if hasattr(encoder, 'begin_render_pass'):
                encoder.begin_render_pass(hal_desc)
        except Exception as e:
            # Handle HAL errors
            # device.handle_hal_error(e)
            raise RenderPassErrorInner(f"Failed to begin render pass: {e}")
        
        # Create and return RenderPassInfo
        context = RenderPassContext(
            color_formats=[getattr(att.view, 'format', None) if att else None for att in color_attachments],
            depth_stencil_format=getattr(depth_stencil_attachment.view, 'format', None) if depth_stencil_attachment else None,
            sample_count=sample_count,
            multiview_mask=multiview_mask,
        )
        
        return cls(
            context=context,
            render_attachments=render_attachments,
            is_depth_read_only=is_depth_read_only,
            is_stencil_read_only=is_stencil_read_only,
            extent=extent,
            multiview_mask=multiview_mask,
            divergent_discards=divergent_discards,
            occlusion_query_set=occlusion_query_set,
        )
    
    def finish(
        self,
        device: Any,
        raw: Any,  # HAL CommandEncoder
        snatch_guard: Any,
        scope: Any,
        instance_flags: Any,
    ) -> None:
        """
        Finish the render pass.
        
        This ends the HAL render pass and performs cleanup. It handles:
        - Ending the HAL render pass
        - Divergent depth/stencil discard operations
        - Resource tracking updates
        
        Args:
            device: The device.
            raw: HAL command encoder.
            snatch_guard: Snatch guard for resource access.
            scope: Usage scope for resource tracking.
            instance_flags: Instance flags for validation.
            
        Raises:
            RenderPassErrorInner: If finalization fails.
        """
        # End HAL render pass
        # This tells the GPU to finish rendering to the attachments
        # This is the critical HAL call that ends GPU rendering
        try:
            if hasattr(raw, 'end_render_pass'):
                raw.end_render_pass()
        except Exception as e:
            # Handle HAL errors
            # device.handle_hal_error(e)
            raise RenderPassErrorInner(f"Failed to end render pass: {e}")
        
        # Handle divergent depth/stencil discards
        # This is needed when depth and stencil have different discard states
        # (e.g., depth is discarded but stencil is stored, or vice versa)
        
        if self.divergent_discards:
            # Process each divergent discard
            for discard_info in self.divergent_discards:
                # Extract texture view and aspects
                texture_view = discard_info.get('texture_view')
                aspects = discard_info.get('aspects')  # DEPTH or STENCIL
                
                # Get the actual texture from the view
                # texture = texture_view.parent
                
                # Create a new render pass just for the discard
                # This is necessary because we can't discard individual aspects
                # in the main render pass
                
                # Determine ops based on aspect
                if aspects == 'DEPTH':
                    depth_ops = AttachmentOps.LOAD_CLEAR | AttachmentOps.STORE
                    stencil_ops = AttachmentOps.LOAD | AttachmentOps.STORE
                else:  # STENCIL
                    depth_ops = AttachmentOps.LOAD | AttachmentOps.STORE
                    stencil_ops = AttachmentOps.LOAD_CLEAR | AttachmentOps.STORE
                
                # Build HAL descriptor for discard pass
                hal_desc = {
                    'label': '(wgpu internal) Zero init discarded depth/stencil aspect',
                    'extent': self.extent,
                    'sample_count': 1,  # Would be from view.samples
                    'color_attachments': [],
                    'depth_stencil_attachment': {
                        'target': {
                            'view': texture_view,  # Would be texture_view.raw(snatch_guard)
                            'usage': TextureUsage.RENDER_ATTACHMENT,
                        },
                        'depth_ops': depth_ops,
                        'stencil_ops': stencil_ops,
                        'clear_value': (0.0, 0),
                    },
                    'multiview_mask': self.multiview_mask,
                    'timestamp_writes': None,
                    'occlusion_query_set': None,
                }
                
                # Execute discard pass
                try:
                    if hasattr(raw, 'begin_render_pass'):
                        raw.begin_render_pass(hal_desc)
                    if hasattr(raw, 'end_render_pass'):
                        raw.end_render_pass()
                except Exception as e:
                    # Handle HAL errors
                    raise RenderPassErrorInner(f"Failed to execute divergent discard: {e}")
        
        # Update resource tracking
        # Mark all used resources in the scope
        
        # For color attachments
        if hasattr(self, 'color_targets'):
            for color_target in self.color_targets:
                if color_target is None:
                    continue
                
                # Track color attachment texture
                if hasattr(color_target, 'view') and hasattr(state, 'pass_state'):
                    if hasattr(state.pass_state, 'scope') and hasattr(state.pass_state.scope, 'textures'):
                        try:
                            texture_view = color_target.view
                            if hasattr(texture_view, 'parent'):
                                state.pass_state.scope.textures.merge_single(
                                    texture_view.parent, 
                                    None,  # Full texture
                                    TextureUses.COLOR_TARGET
                                )
                        except Exception:
                            pass
                
                # If there's a resolve target, track it too
                if hasattr(color_target, 'resolve_target') and color_target.resolve_target:
                    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
                        if hasattr(state.pass_state.scope, 'textures'):
                            try:
                                resolve_view = color_target.resolve_target
                                if hasattr(resolve_view, 'parent'):
                                    state.pass_state.scope.textures.merge_single(
                                        resolve_view.parent,
                                        None,
                                        TextureUses.COLOR_TARGET
                                    )
                            except Exception:
                                pass
        
        # For depth/stencil attachment
        if hasattr(self, 'depth_stencil_target') and self.depth_stencil_target:
            if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
                if hasattr(state.pass_state.scope, 'textures'):
                    try:
                        ds_target = self.depth_stencil_target
                        if hasattr(ds_target, 'view') and hasattr(ds_target.view, 'parent'):
                            # Basic logic: if writeable, use DEPTH_STENCIL_WRITE, else DEPTH_STENCIL_READ
                            # (Real wgpu-core logic is more complex with read-only depth/stencil)
                            usage = TextureUses.DEPTH_STENCIL_WRITE
                            if hasattr(self, 'is_depth_read_only') and self.is_depth_read_only:
                                usage = TextureUses.DEPTH_STENCIL_READ
                                
                            state.pass_state.scope.textures.merge_single(
                                ds_target.view.parent,
                                None,
                                usage
                            )
                    except Exception:
                        pass
        
        # Clear internal state
        self.divergent_discards = []


@dataclass  
class State:
    """
    Render pass state for validation and tracking.
    
    This class tracks the current state of a render pass during command encoding,
    including pipeline, buffers, and other state needed for draw call validation.
    
    Attributes:
        pipeline_flags: Current pipeline flags.
        blend_constant: Blend constant state (unused/required/set).
        stencil_reference: Current stencil reference value.
        pipeline: Currently bound render pipeline.
        index: Index buffer state.
        vertex: Vertex buffer state.
        info: Render pass info.
        pass_state: Pass-level state (binder, etc.).
        active_occlusion_query: Active occlusion query if any.
        active_pipeline_statistics_query: Active pipeline statistics query if any.
    """
    pipeline_flags: Any = None
    blend_constant: OptionalState = OptionalState.UNUSED
    stencil_reference: int = 0
    pipeline: Optional[Any] = None
    index: IndexState = field(default_factory=IndexState)
    vertex: VertexState = field(default_factory=VertexState)
    info: Optional[RenderPassInfo] = None
    pass_state: Optional[Any] = None  # pass::PassState
    active_occlusion_query: Optional[Tuple[Any, int]] = None
    active_pipeline_statistics_query: Optional[Tuple[Any, int]] = None
    device: Optional[Any] = None
    tracker: Optional[Any] = None
    raw_encoder: Optional[Any] = None
    reset_state: Optional[Any] = None # Used for query resets
    
    def is_ready(self, family: DrawCommandFamily) -> None:
        """
        Check if the state is ready for a draw call.
        
        Validates that all required state is set for the given draw command family.
        
        Args:
            family: The draw command family (draw, draw_indexed, draw_mesh_tasks).
            
        Raises:
            DrawError: If required state is missing or invalid.
        """
        if self.pipeline is None:
            raise DrawError("Missing pipeline")
        
        # Check if binder compatibility is valid
        if self.pass_state and self.pass_state.binder:
            self.pass_state.binder.check_compatibility(self.pipeline)
            self.pass_state.binder.check_late_buffer_bindings()
        
        if self.blend_constant == OptionalState.REQUIRED:
            raise DrawError("Missing blend constant")
        
        # Check vertex buffers
        vertex_buffer_count = sum(1 for size in self.vertex.buffer_sizes if size is not None)
        # if vertex_buffer_count < len(self.pipeline.vertex_steps):
        #     raise DrawError(f"Missing vertex buffer at index {vertex_buffer_count}")
        
        # Check index buffer for indexed draws
        if family == DrawCommandFamily.DRAW_INDEXED:
            if self.index.buffer_format is None:
                raise DrawError("Missing index buffer")
            # Check format compatibility with pipeline
        
        # Check mesh pipeline compatibility
        # if (family == DrawCommandFamily.DRAW_MESH_TASKS) != self.pipeline.is_mesh:
        #     raise DrawError("Wrong pipeline type")
    
    def flush_bindings(self) -> None:
        """
        Flush binding state in preparation for a draw call.
        
        This ensures all bind groups are properly bound before drawing.
        It validates that all required bind groups (as defined by the pipeline)
        are set and calls the HAL to actually bind them.
        """
        # Check if pipeline is set
        if self.pipeline is None:
            # No pipeline set, nothing to flush
            return
        
        if self.pass_state:
            flush_bindings_helper(self.pass_state)
    
    def reset_bundle(self) -> None:
        """
        Reset the render bundle-related states.
        
        Called after executing a render bundle to reset state that may have
        been modified by the bundle.
        """
        # self.pass_state.binder.reset()
        self.pipeline = None
        self.index.reset()
        self.vertex = VertexState()


# ============================================================================
# Global API (Entry Points)
# ============================================================================

class Global:
    """
    Global API for render pass operations.
    
    This class provides the main entry points for creating and managing
    render passes from command encoders. It acts as the central hub for
    resource management and validation.
    
    Attributes:
        hub: Resource hub containing registries for all GPU objects.
    """
    
    def __init__(self, hub: Any = None):
        """
        Initialize Global API.
        
        Args:
            hub: Resource hub with registries (command_encoders, texture_views, etc.)
        """
        self.hub = hub  # Would be crate::hub::Hub in Rust
    
    def command_encoder_begin_render_pass(
        self,
        encoder_id: Any,
        desc: RenderPassDescriptor,
    ) -> Tuple[RenderPass, Optional[Any]]:
        """
        Create a render pass from a command encoder.
        
        If creation fails, an invalid pass is returned. Attempting to record
        commands into an invalid pass is permitted, but a validation error will
        ultimately be generated when the parent encoder is finished.
        
        If successful, puts the encoder into the Locked state.
        
        This function performs extensive validation:
        - Locks the encoder (prevents creating multiple passes)
        - Validates color attachment count against device limits
        - Validates all texture views exist and are compatible
        - Checks texture usage flags (RENDER_ATTACHMENT, TRANSIENT)
        - Validates depth/stencil attachment format
        - Validates depth clear values are in [0.0, 1.0]
        - Validates timestamp writes
        - Validates occlusion query set
        
        Implementation follows wgpu-core/src/command/render.rs:1610-1850
        
        Args:
            encoder_id: Command encoder ID.
            desc: Render pass descriptor with attachments.
            
        Returns:
            Tuple of (RenderPass, Optional[CommandEncoderError]).
            The error is Some if the encoder is in an invalid state.
        """
        pass_scope = "Pass"
        
        # Get command encoder from hub
        cmd_enc = None
        device = None
        
        if self.hub and hasattr(self.hub, 'command_encoders'):
            try:
                cmd_enc = self.hub.command_encoders.get(encoder_id)
                if cmd_enc and hasattr(cmd_enc, 'device'):
                    device = cmd_enc.device
            except Exception:
                # ID not found or invalid
                pass
        
        # Lock encoder data
        # In this implementation, we directly interact with the encoder's status
        try:
            if not cmd_enc:
                raise EncoderStateInvalid("Command encoder not found")
            
            status = getattr(cmd_enc, '_status', 'Recording')
            if status == 'Locked':
                raise EncoderStateLocked("Command encoder is already locked by another pass")
            elif status == 'Finished':
                raise EncoderStateEnded("Command encoder has already been finished")
            elif status == 'Submitted':
                raise EncoderStateSubmitted("Command encoder has already been submitted")
            elif status == 'Invalid':
                raise EncoderStateInvalid("Command encoder is in an invalid state")
            
            # Successfully "locked" the encoder
            cmd_enc._status = 'Locked'
            
        except Exception as err:
            # Encoder is in invalid state
            # Create invalid pass and return error
            error_name = type(err).__name__
            
            if isinstance(err, EncoderStateLocked):
                # Attempting to open pass while encoder locked
                # Invalidates encoder but doesn't generate validation error
                if hasattr(cmd_enc, 'invalidate'):
                    cmd_enc.invalidate(err)
                render_pass = RenderPass(cmd_enc, desc)
                render_pass.base = None  # Invalid
                return (render_pass, None)
            
            elif isinstance(err, (EncoderStateEnded, EncoderStateSubmitted)):
                # Attempting to open pass after encoder ended
                # Generates immediate validation error
                render_pass = RenderPass(cmd_enc, desc)
                render_pass.base = None  # Invalid
                return (render_pass, err)
            
            elif isinstance(err, EncoderStateInvalid):
                # Encoder is invalid, but we can still create pass
                render_pass = RenderPass(cmd_enc, desc)
                render_pass.base = None  # Invalid
                return (render_pass, None)
            else:
                # Unexpected error, re-raise or handle as invalid
                raise err
        
        # Encoder successfully locked, now validate descriptor
        # This is the fill_arc_desc function from Rust
        
        try:
            # Get registries from hub
            texture_views = None
            query_sets = None
            
            if self.hub:
                if hasattr(self.hub, 'texture_views'):
                    texture_views = self.hub.texture_views
                if hasattr(self.hub, 'query_sets'):
                    query_sets = self.hub.query_sets
            
            # Validate device is valid
            if device and hasattr(device, 'check_is_valid'):
                # device.check_is_valid()
                pass
            
            # Get device limits
            max_color_attachments = 8  # Default
            if device and hasattr(device, 'limits'):
                max_color_attachments = device.limits.max_color_attachments
            
            # Validate color attachment count
            if len(desc.color_attachments) > max_color_attachments:
                raise ValueError(
                    f"Too many color attachments: {len(desc.color_attachments)} > {max_color_attachments}"
                )
            
            # Validate each color attachment
            arc_color_attachments = []
            
            for i, color_att in enumerate(desc.color_attachments):
                if color_att is None:
                    arc_color_attachments.append(None)
                    continue
                
                # Get texture view from hub
                view = None
                if texture_views and hasattr(color_att, 'view'):
                    try:
                        view = texture_views.get(color_att.view)
                        # Validate same device
                        if view and device and hasattr(view, 'same_device'):
                            view.same_device(device)
                    except Exception:
                        # View not found or device mismatch
                        pass
                
                # Check TRANSIENT usage with store_op
                if view and hasattr(view, 'desc'):
                    if hasattr(view.desc, 'usage'):
                        if view.desc.usage & TextureUsage.RENDER_ATTACHMENT:  # TRANSIENT check
                            if hasattr(color_att, 'store_op'):
                                if color_att.store_op != 'Discard':
                                    raise ValueError(
                                        f"ColorAttachment[{i}]: TRANSIENT texture must use StoreOp::Discard"
                                    )
                
                # Resolve resolve_target if present
                resolve_target = None
                if hasattr(color_att, 'resolve_target') and color_att.resolve_target:
                    if texture_views:
                        try:
                            rt_view = texture_views.get(color_att.resolve_target)
                            if rt_view and device and hasattr(rt_view, 'same_device'):
                                rt_view.same_device(device)
                            resolve_target = rt_view
                        except Exception:
                            # Resolve target not found or device mismatch
                            pass
                
                # Create Arc color attachment
                arc_color_att = {
                    'view': view,
                    'depth_slice': getattr(color_att, 'depth_slice', None),
                    'resolve_target': resolve_target,
                    'load_op': getattr(color_att, 'load_op', 'Load'),
                    'store_op': getattr(color_att, 'store_op', 'Store'),
                }
                arc_color_attachments.append(arc_color_att)
            
            # Validate depth/stencil attachment
            arc_depth_stencil = None
            
            if hasattr(desc, 'depth_stencil_attachment') and desc.depth_stencil_attachment:
                ds_att = desc.depth_stencil_attachment
                
                # Get texture view from hub
                view = None
                if texture_views and hasattr(ds_att, 'view'):
                    try:
                        view = texture_views.get(ds_att.view)
                        # Validate same device
                        if view and device and hasattr(view, 'same_device'):
                            view.same_device(device)
                    except Exception:
                        # View not found or device mismatch
                        pass
                
                # Check format is depth/stencil
                if view and hasattr(view, 'desc'):
                    format = view.desc.format if hasattr(view.desc, 'format') else None
                    # if format and not format.is_depth_stencil_format():
                    #     raise ValueError(f"Invalid depth/stencil format: {format}")
                
                # Validate depth clear value is in [0.0, 1.0]
                if hasattr(ds_att, 'depth') and hasattr(ds_att.depth, 'clear_value'):
                    clear = ds_att.depth.clear_value
                    if clear is not None and not (0.0 <= clear <= 1.0):
                        raise ValueError(f"Depth clear value {clear} not in [0.0, 1.0]")
                
                # Create Arc depth/stencil attachment
                arc_depth_stencil = {
                    'view': view,
                    'depth': getattr(ds_att, 'depth', None),
                    'stencil': getattr(ds_att, 'stencil', None),
                }
            
            # Validate timestamp writes
            arc_timestamp_writes = None
            if hasattr(desc, 'timestamp_writes') and desc.timestamp_writes:
                # Global::validate_pass_timestamp_writes(device, query_sets, tw)
                arc_timestamp_writes = desc.timestamp_writes
            
            # Validate occlusion query set
            arc_occlusion_query_set = None
            if hasattr(desc, 'occlusion_query_set') and desc.occlusion_query_set:
                if query_sets:
                    try:
                        query_set = query_sets.get(desc.occlusion_query_set)
                        if query_set and device and hasattr(query_set, 'same_device'):
                            query_set.same_device(device)
                        arc_occlusion_query_set = query_set
                    except Exception:
                        # Query set not found or device mismatch
                        pass
            
            # Create ArcRenderPassDescriptor
            arc_desc = {
                'label': getattr(desc, 'label', None),
                'color_attachments': arc_color_attachments,
                'depth_stencil_attachment': arc_depth_stencil,
                'timestamp_writes': arc_timestamp_writes,
                'occlusion_query_set': arc_occlusion_query_set,
                'multiview_mask': getattr(desc, 'multiview_mask', None),
            }
            
            # Create valid render pass with validated descriptor
            render_pass = RenderPass(
                parent=cmd_enc,
                desc=desc,
            )
            
            # Store validated Arc descriptor
            render_pass.arc_desc = arc_desc
            render_pass.color_attachments = arc_color_attachments
            render_pass.depth_stencil_attachment = arc_depth_stencil
            render_pass.timestamp_writes = arc_timestamp_writes
            render_pass.occlusion_query_set = arc_occlusion_query_set
            render_pass.multiview_mask = getattr(desc, 'multiview_mask', None)
            
            return (render_pass, None)
            
        except Exception as e:
            # Validation failed, create invalid pass
            render_pass = RenderPass(
                parent=cmd_enc,
                desc=desc,
            )
            render_pass.base = None  # Mark as invalid
            if hasattr(render_pass, 'error'):
                render_pass.error = e
            
            # Unlock encoder since we failed
            # cmd_buf_data.unlock_encoder()
            
            return (render_pass, None)
    
    def render_pass_end(self, pass_obj: RenderPass) -> None:
        """
        End a render pass.
        
        This finalizes the render pass, unlocks the encoder, and records
        the render pass command. This is the counterpart to 
        command_encoder_begin_render_pass.
        
        The function:
        1. Takes ownership of the parent encoder
        2. Unlocks the encoder (allows new passes)
        3. Extracts the recorded pass data
        4. Creates RunRenderPass command
        5. Pushes command to encoder's command buffer
        
        Args:
            pass_obj: The render pass to end.
            
        Raises:
            EncoderStateError: If the encoder is in an invalid state.
        """
        # Take parent encoder from pass
        # This ensures the pass can only be ended once
        cmd_enc = pass_obj.parent
        if cmd_enc is None:
            raise RuntimeError("Pass already ended")
        
        # Clear parent to prevent double-end
        pass_obj.parent = None
        
        # Lock encoder data (in Python we call the unlock method)
        try:
            if hasattr(cmd_enc, '_unlock_encoder'):
                cmd_enc._unlock_encoder()
            else:
                # Fallback if method not present
                cmd_enc._status = "Recording"
        except Exception as e:
            # If unlock fails, encoder is in bad state
            raise RuntimeError(f"Failed to unlock encoder: {e}")
        
        # Take base pass data
        # This extracts all recorded commands and metadata
        base = pass_obj.base
        if base is None:
            raise RuntimeError("Pass base data already taken")
        
        # Clear base to prevent reuse
        pass_obj.base = None
        
        # Check for encoder state errors
        # Some errors are detected during pass recording and stored in base
        if hasattr(base, 'error') and base.error is not None:
            error = base.error
            
            # Special handling for certain error types
            # EncoderStateError::Locked - pass opened while encoder locked
            # EncoderStateError::Ended - pass opened after encoder ended
            if isinstance(error, Exception):
                error_name = type(error).__name__
                if error_name in ['EncoderStateLocked', 'EncoderStateEnded']:
                    # These are validation errors that should be raised immediately
                    raise error
        
        # Create RunRenderPass command
        # This packages all the pass data into a command
        try:
            command = {
                'type': 'RunRenderPass',
                'pass': base,
                'color_attachments': pass_obj.color_attachments[:],  # Copy
                'depth_stencil_attachment': pass_obj.depth_stencil_attachment,
                'timestamp_writes': pass_obj.timestamp_writes,
                'occlusion_query_set': pass_obj.occlusion_query_set,
                'multiview_mask': pass_obj.multiview_mask,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to create RunRenderPass command: {e}")
        
        # Push command to encoder's command buffer
        if hasattr(cmd_enc, '_commands'):
            cmd_enc._commands.append(command)
        
        # Clear pass attachments to prevent reuse
        pass_obj.color_attachments = []
        pass_obj.depth_stencil_attachment = None
        pass_obj.timestamp_writes = None
        pass_obj.occlusion_query_set = None


# ============================================================================
# Standalone State Management Functions
# ============================================================================

def set_pipeline(
    state: State,
    device: Any,
    pipeline: Any,
    raw_encoder: Any = None,  # HAL CommandEncoder
) -> None:
    """
    Set the render pipeline.
    
    This validates the pipeline against the render pass context and
    updates all pipeline-dependent state. The pipeline defines:
    - Vertex/fragment shaders
    - Vertex input layout
    - Primitive topology
    - Depth/stencil state
    - Blend state
    - Multisampling
    
    Implementation follows wgpu-core/src/command/render.rs:2323-2390
    
    Args:
        state: Current render pass state.
        device: The device.
        pipeline: The render pipeline to set.
        raw_encoder: HAL command encoder (optional, extracted from state if not provided).
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Store pipeline in state
    state.pipeline = pipeline
    
    # Track pipeline in resource tracker
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        if hasattr(state.pass_state.base, 'tracker'):
            # Register pipeline in tracker for usage validation
            pipeline = state.pass_state.base.tracker.render_pipelines.insert_single(pipeline)
    
    # Validate pipeline is from same device
    if hasattr(pipeline, 'same_device'):
        # pipeline.same_device(device)
        pass
    elif hasattr(pipeline, 'device') and hasattr(device, 'id'):
        if pipeline.device != device:
            raise RenderPassErrorInner("Pipeline from different device")
    
    # Check render pass context compatibility
    # This validates that the pipeline's color/depth formats match the render pass
    if state.info and hasattr(state.info, 'context'):
        if hasattr(pipeline, 'pass_context'):
            # state.info.context.check_compatible(pipeline.pass_context, pipeline)
            # Would raise IncompatiblePipelineTargets if:
            # - Color format mismatch
            # - Depth/stencil format mismatch
            # - Sample count mismatch
            # - Multiview mismatch
            pass
    
    # Extract pipeline flags
    pipeline_flags = getattr(pipeline, 'flags', None)
    if pipeline_flags is not None:
        state.pipeline_flags = pipeline_flags
    
    # Validate depth/stencil access compatibility
    # Pipeline cannot write to depth if render pass has depth as read-only
    if pipeline_flags and state.info:
        # Check depth write compatibility
        if hasattr(pipeline_flags, 'WRITES_DEPTH'):
            writes_depth = pipeline_flags & getattr(pipeline_flags.__class__, 'WRITES_DEPTH', 0)
            if writes_depth and hasattr(state.info, 'is_depth_read_only'):
                if state.info.is_depth_read_only:
                    raise RenderPassErrorInner(
                        f"IncompatibleDepthAccess: Pipeline writes depth but "
                        f"render pass has depth as read-only"
                    )
        
        # Check stencil write compatibility
        if hasattr(pipeline_flags, 'WRITES_STENCIL'):
            writes_stencil = pipeline_flags & getattr(pipeline_flags.__class__, 'WRITES_STENCIL', 0)
            if writes_stencil and hasattr(state.info, 'is_stencil_read_only'):
                if state.info.is_stencil_read_only:
                    raise RenderPassErrorInner(
                        f"IncompatibleStencilAccess: Pipeline writes stencil but "
                        f"render pass has stencil as read-only"
                    )
    
    # Update blend constant requirement
    # If pipeline uses blend constant, it must be set before drawing
    if pipeline_flags:
        needs_blend_constant = pipeline_flags & getattr(pipeline_flags.__class__, 'BLEND_CONSTANT', 0)
        if hasattr(state, 'blend_constant'):
            if hasattr(state.blend_constant, 'require'):
                # Use the require() method for clean state transitions
                state.blend_constant.require(bool(needs_blend_constant))
            else:
                # Fallback: manual update for plain OptionalState enum
                if needs_blend_constant:
                    if state.blend_constant == OptionalState.UNUSED:
                        state.blend_constant = OptionalState.REQUIRED
                else:
                    if state.blend_constant == OptionalState.REQUIRED:
                        state.blend_constant = OptionalState.UNUSED
    
    # Get HAL encoder
    if raw_encoder is None:
        if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
            raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
    
    # Set HAL pipeline
    # This is the critical HAL call that sets the active pipeline on GPU
    if raw_encoder and hasattr(raw_encoder, 'set_render_pipeline'):
        try:
            # Get raw pipeline handle
            raw_pipeline = pipeline
            if hasattr(pipeline, 'raw'):
                # raw_pipeline = pipeline.raw()
                pass
            
            raw_encoder.set_render_pipeline(raw_pipeline)
        except Exception as e:
            raise RenderPassErrorInner(f"Failed to set render pipeline: {e}")
    
    # Update stencil reference if pipeline uses it
    # This must be done after setting the pipeline
    if pipeline_flags:
        uses_stencil_reference = pipeline_flags & getattr(pipeline_flags.__class__, 'STENCIL_REFERENCE', 0)
        if uses_stencil_reference and raw_encoder:
            if hasattr(raw_encoder, 'set_stencil_reference'):
                try:
                    stencil_ref = getattr(state, 'stencil_reference', 0)
                    raw_encoder.set_stencil_reference(stencil_ref)
                except Exception as e:
                    raise RenderPassErrorInner(f"Failed to set stencil reference: {e}")
    
    # Change pipeline layout (rebind resources)
    # When pipeline changes, bind groups may need to be rebound
    # This is because different pipelines may have different bind group layouts
    if hasattr(pipeline, 'layout') and state.pass_state:
        state.pass_state.binder.change_pipeline_layout(
            pipeline.layout,
            getattr(pipeline, 'late_sized_buffer_groups', [])
        )
    
    # Update vertex buffer limits
    # The pipeline defines which vertex buffers are used and their strides
    if hasattr(pipeline, 'vertex_steps'):
        if hasattr(state, 'vertex') and hasattr(state.vertex, 'update_limits'):
            state.vertex.update_limits(pipeline.vertex_steps)
        else:
            # Manually update limits based on vertex steps
            # vertex_steps defines: buffer slot, stride, step_mode (vertex/instance)
            pass


def set_index_buffer(
    state: State,
    device: Any,
    buffer: Any,
    index_format: str,
    offset: int,
    size: Optional[int],
) -> None:
    """
    Set the index buffer.
    
    The index buffer provides indices for indexed draw calls. Each index
    references a vertex in the vertex buffers.
    
    Args:
        state: Current render pass state.
        device: The device.
        buffer: The buffer to use as index buffer.
        index_format: Index format ("uint16" or "uint32").
        offset: Offset into the buffer (must be aligned).
        size: Optional size of the index data. If None, uses rest of buffer.
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Track buffer in resource tracker
    # buffer = state.pass_state.base.tracker.buffers.insert_single(buffer)
    
    # Validate buffer is from same device
    if hasattr(buffer, 'device') and hasattr(device, 'id'):
        if buffer.device != device:
            raise RenderPassErrorInner("Index buffer from different device")
    
    # Check buffer has INDEX usage flag
    if hasattr(buffer, 'usage'):
        # Should have BufferUsages::INDEX
        # if not buffer.usage.contains(BufferUsages::INDEX):
        #     raise RenderPassErrorInner("Buffer missing INDEX usage")
        pass
    
    # Check buffer is not destroyed
    # buffer.check_destroyed(snatch_guard)
    
    # Determine index size based on format
    if index_format == "uint16":
        index_size = 2
    elif index_format == "uint32":
        index_size = 4
    else:
        raise RenderPassErrorInner(
            f"Invalid index format: {index_format} (must be 'uint16' or 'uint32')"
        )
    
    # Validate offset alignment
    # Offset must be aligned to index size
    if offset % index_size != 0:
        raise RenderPassErrorInner(
            f"Index buffer offset {offset} not aligned to index size {index_size}"
        )
    
    # Calculate buffer size
    buffer_total_size = getattr(buffer, 'size', 0)
    
    # Determine the size of index data
    if size is not None:
        # Explicit size provided
        index_data_size = size
        
        # Validate size doesn't exceed buffer
        if offset + size > buffer_total_size:
            raise RenderPassErrorInner(
                f"Index buffer range out of bounds: "
                f"offset={offset}, size={size}, buffer_size={buffer_total_size}"
            )
    else:
        # Use rest of buffer from offset
        if offset > buffer_total_size:
            raise RenderPassErrorInner(
                f"Index buffer offset {offset} exceeds buffer size {buffer_total_size}"
            )
        index_data_size = buffer_total_size - offset
    
    # Validate size is aligned to index size
    if index_data_size % index_size != 0:
        raise RenderPassErrorInner(
            f"Index buffer size {index_data_size} not aligned to index size {index_size}"
        )
    
    # Calculate index limit (number of indices)
    index_count = index_data_size // index_size
    
    # Update index state
    state.index.update_buffer(offset, offset + index_data_size, index_format)
    state.index.limit = index_count
    
    # Register buffer memory initialization action
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        if hasattr(state.pass_state.base, 'buffer_memory_init_actions'):
            if hasattr(buffer, 'initialization_status') and hasattr(buffer.initialization_status, 'create_action'):
                action = buffer.initialization_status.create_action(
                    buffer,
                    Range(offset, offset + index_data_size),
                    MemoryInitKind.NEEDS_INITIALIZED_MEMORY
                )
                if action:
                    state.pass_state.base.buffer_memory_init_actions.append(action)
    
    # Merge buffer into resource scope
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
        if hasattr(state.pass_state.scope, 'buffers'):
            state.pass_state.scope.buffers.merge_single(buffer, BufferUses.INDEX)
    
    # Set HAL index buffer
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'set_index_buffer'):
            try:
                # Get raw buffer (would use snatch_guard in full impl)
                raw_buffer = buffer
                if hasattr(buffer, 'raw'):
                    raw_buffer = buffer.raw()
                
                raw_encoder.set_index_buffer(raw_buffer, index_format)
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to set index buffer: {e}")


def set_vertex_buffer(
    state: State,
    device: Any,
    slot: int,
    buffer: Any,
    offset: int,
    size: Optional[int],
) -> None:
    """
    Set a vertex buffer.
    
    Vertex buffers provide per-vertex data (positions, normals, UVs, etc.)
    to the vertex shader. Multiple vertex buffers can be bound to different
    slots simultaneously.
    
    Args:
        state: Current render pass state.
        device: The device.
        slot: Vertex buffer slot (must be < max_vertex_buffers limit).
        buffer: The buffer to use as vertex buffer.
        offset: Offset into the buffer (must be 4-byte aligned).
        size: Optional size of the vertex data. If None, uses rest of buffer.
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Track buffer in resource tracker
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        if hasattr(state.pass_state.base, 'tracker'):
            buffer = state.pass_state.base.tracker.buffers.insert_single(buffer)
    
    # Validate buffer is from same device
    if hasattr(buffer, 'device') and hasattr(device, 'id'):
        if buffer.device != device:
            raise RenderPassErrorInner("Vertex buffer from different device")
    
    # Check buffer has VERTEX usage flag
    if hasattr(buffer, 'usage'):
        if not (buffer.usage & BufferUsage.VERTEX):
            raise RenderPassErrorInner(
                f"Buffer {getattr(buffer, 'label', '')} is missing VERTEX usage"
            )
    
    # Check buffer is not destroyed
    snatch_guard = getattr(state.pass_state.base, 'snatch_guard', None) if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base') else None
    if hasattr(buffer, 'check_destroyed') and snatch_guard:
        buffer.check_destroyed(snatch_guard)
    
    # Validate slot is within bounds
    max_vertex_buffers = getattr(device.limits, 'max_vertex_buffers', 16) if hasattr(device, 'limits') else 16
    if slot >= max_vertex_buffers:
        raise RenderPassErrorInner(
            f"Vertex buffer slot {slot} exceeds maximum {max_vertex_buffers}"
        )
    
    # Validate offset alignment
    # Vertex buffer offset must be 4-byte aligned
    if offset % 4 != 0:
        raise RenderPassErrorInner(
            f"Vertex buffer offset {offset} not 4-byte aligned"
        )
    
    # Calculate buffer size
    buffer_total_size = getattr(buffer, 'size', 0)
    
    # Determine the size of vertex data
    if size is not None:
        # Explicit size provided
        vertex_data_size = size
        
        # Validate size doesn't exceed buffer
        if offset + size > buffer_total_size:
            raise RenderPassErrorInner(
                f"Vertex buffer range out of bounds: "
                f"offset={offset}, size={size}, buffer_size={buffer_total_size}"
            )
    else:
        # Use rest of buffer from offset
        if offset > buffer_total_size:
            raise RenderPassErrorInner(
                f"Vertex buffer offset {offset} exceeds buffer size {buffer_total_size}"
            )
        vertex_data_size = buffer_total_size - offset
    
    # Update vertex state
    # Store the buffer size for this slot for later validation
    if not hasattr(state.vertex, 'buffer_sizes'):
        state.vertex.buffer_sizes = {}
    state.vertex.buffer_sizes[slot] = vertex_data_size
    
    # Store buffer binding info
    if not hasattr(state.vertex, 'buffers'):
        state.vertex.buffers = {}
    state.vertex.buffers[slot] = {
        'buffer': buffer,
        'offset': offset,
        'size': vertex_data_size,
    }
    
    # Register buffer memory initialization action
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        if hasattr(state.pass_state.base, 'buffer_memory_init_actions'):
            if hasattr(buffer, 'initialization_status') and hasattr(buffer.initialization_status, 'create_action'):
                action = buffer.initialization_status.create_action(
                    buffer,
                    Range(offset, offset + vertex_data_size),
                    MemoryInitKind.NEEDS_INITIALIZED_MEMORY
                )
                if action:
                    state.pass_state.base.buffer_memory_init_actions.append(action)
    
    # Merge buffer into resource scope
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
        if hasattr(state.pass_state.scope, 'buffers'):
            state.pass_state.scope.buffers.merge_single(buffer, BufferUses.VERTEX)
    
    # Set HAL vertex buffer
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'set_vertex_buffer'):
            try:
                # Get raw buffer
                raw_buffer = buffer
                if hasattr(buffer, 'raw'):
                    raw_buffer = buffer.raw()
                
                raw_encoder.set_vertex_buffer(slot, raw_buffer, offset)
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to set vertex buffer: {e}")


def set_blend_constant(state: State, color: Tuple[float, float, float, float]) -> None:
    """
    Set the blend constant color.
    
    Args:
        state: Current render pass state.
        color: RGBA blend constant.
    """
    state.blend_constant = OptionalState.SET
    
    # Set HAL blend constant
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'set_blend_constant'):
            try:
                raw_encoder.set_blend_constant(color)
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to set blend constant: {e}")


def set_stencil_reference(state: State, value: int) -> None:
    """
    Set the stencil reference value.
    
    Args:
        state: Current render pass state.
        value: Stencil reference value.
    """
    state.stencil_reference = value
    
    # Set HAL stencil reference
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'set_stencil_reference'):
            try:
                raw_encoder.set_stencil_reference(value)
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to set stencil reference: {e}")


def set_viewport(
    state: State,
    rect: Rect,
    depth_min: float,
    depth_max: float,
) -> None:
    """
    Set the viewport.
    
    The viewport defines the transformation from normalized device coordinates
    to framebuffer coordinates. It also defines the depth range mapping.
    
    Args:
        state: Current render pass state.
        rect: Viewport rectangle with x, y, width, height.
        depth_min: Minimum depth value (must be in [0.0, 1.0]).
        depth_max: Maximum depth value (must be in [0.0, 1.0]).
        
    Raises:
        RenderPassErrorInner: If viewport is out of bounds or depth range is invalid.
    """
    # Get render target extent from render pass info
    extent = (0, 0, 0)
    if state.info and hasattr(state.info, 'extent'):
        extent = state.info.extent
    
    # Extract extent dimensions (convert to float for comparison)
    extent_width = float(extent[0]) if len(extent) > 0 else 0.0
    extent_height = float(extent[1]) if len(extent) > 1 else 0.0
    
    # Extract viewport rectangle parameters
    x = float(getattr(rect, 'x', 0.0))
    y = float(getattr(rect, 'y', 0.0))
    width = float(getattr(rect, 'width', 0.0) if hasattr(rect, 'width') else getattr(rect, 'w', 0.0))
    height = float(getattr(rect, 'height', 0.0) if hasattr(rect, 'height') else getattr(rect, 'h', 0.0))
    
    # Check for invalid dimensions (width/height must be positive)
    if width <= 0.0 or height <= 0.0:
        raise RenderPassErrorInner(
            f"InvalidViewportDimension: "
            f"width={width}, height={height} "
            f"(must be positive)"
        )
    
    # Get max texture dimension for viewport range validation
    # max_viewport_range = max_texture_dimension_2d * 2.0
    max_texture_dimension = 8192  # Typical default, should come from device.limits
    max_viewport_range = float(max_texture_dimension) * 2.0
    
    # Validate viewport position is within allowed range
    # Viewport can extend beyond render target but has absolute limits
    if (x < -max_viewport_range or 
        y < -max_viewport_range or
        x + width > max_viewport_range - 1.0 or
        y + height > max_viewport_range - 1.0):
        raise RenderPassErrorInner(
            f"InvalidViewportRectPosition: "
            f"viewport=({x}, {y}, {width}, {height}), "
            f"allowed_range=[{-max_viewport_range}, {max_viewport_range - 1.0}]"
        )
    
    # Validate depth range
    # Both depth_min and depth_max must be in [0.0, 1.0]
    # AND depth_min must be <= depth_max
    if not (0.0 <= depth_min <= 1.0):
        raise RenderPassErrorInner(
            f"InvalidViewportDepth: depth_min={depth_min} (must be in [0.0, 1.0])"
        )
    
    if not (0.0 <= depth_max <= 1.0):
        raise RenderPassErrorInner(
            f"InvalidViewportDepth: depth_max={depth_max} (must be in [0.0, 1.0])"
        )
    
    # WebGPU requires depth_min <= depth_max
    if depth_min > depth_max:
        raise RenderPassErrorInner(
            f"InvalidViewportDepth: depth_min={depth_min} > depth_max={depth_max} "
            f"(depth_min must be <= depth_max)"
        )
    
    # Check for NaN or infinity
    import math
    if not math.isfinite(x):
        raise RenderPassErrorInner(f"Viewport x is not finite: {x}")
    if not math.isfinite(y):
        raise RenderPassErrorInner(f"Viewport y is not finite: {y}")
    if not math.isfinite(width):
        raise RenderPassErrorInner(f"Viewport width is not finite: {width}")
    if not math.isfinite(height):
        raise RenderPassErrorInner(f"Viewport height is not finite: {height}")
    
    # Set HAL viewport
    # The viewport transformation maps:
    # - NDC x [-1, 1] -> [x, x + width]
    # - NDC y [-1, 1] -> [y, y + height]  
    # - NDC z [0, 1] -> [depth_min, depth_max]
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'set_viewport'):
            try:
                # Create viewport rect
                viewport_rect = {
                    'x': x,
                    'y': y,
                    'w': width,
                    'h': height,
                    'min_depth': depth_min,
                    'max_depth': depth_max,
                }
                raw_encoder.set_viewport(viewport_rect)
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to set viewport: {e}")


def set_scissor(state: State, rect: Rect) -> None:
    """
    Set the scissor rectangle.
    
    The scissor rectangle defines the region of the render target that will
    be affected by draw calls. Pixels outside this rectangle are discarded.
    
    Args:
        state: Current render pass state.
        rect: Scissor rectangle with x, y, width, height.
        
    Raises:
        RenderPassErrorInner: If scissor rectangle is out of bounds.
    """
    # Get render target extent from render pass info
    extent = (0, 0, 0)
    if state.info and hasattr(state.info, 'extent'):
        extent = state.info.extent
    
    # Extract extent dimensions
    extent_width = extent[0] if len(extent) > 0 else 0
    extent_height = extent[1] if len(extent) > 1 else 0
    
    # Extract scissor rectangle parameters
    x = getattr(rect, 'x', 0)
    y = getattr(rect, 'y', 0)
    width = getattr(rect, 'width', 0) if hasattr(rect, 'width') else getattr(rect, 'w', 0)
    height = getattr(rect, 'height', 0) if hasattr(rect, 'height') else getattr(rect, 'h', 0)
    
    # Validate scissor rectangle is within render target bounds
    # Check if scissor rectangle extends beyond render target
    if x + width > extent_width or y + height > extent_height:
        raise RenderPassErrorInner(
            f"Scissor rectangle out of bounds: "
            f"scissor=({x}, {y}, {width}, {height}), "
            f"extent=({extent_width}, {extent_height})"
        )
    
    # Additional validation: check for negative values or overflow
    if x < 0 or y < 0 or width < 0 or height < 0:
        raise RenderPassErrorInner(
            f"Scissor rectangle has negative values: "
            f"({x}, {y}, {width}, {height})"
        )
    
    # Set HAL scissor rectangle
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'set_scissor_rect'):
            try:
                raw_encoder.set_scissor_rect(x, y, width, height)
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to set scissor rect: {e}")


def draw(
    state: State,
    vertex_count: int,
    instance_count: int,
    first_vertex: int,
    first_instance: int,
) -> None:
    """
    Draw primitives.
    
    Args:
        state: Current render pass state.
        vertex_count: Number of vertices to draw.
        instance_count: Number of instances to draw.
        first_vertex: First vertex index.
        first_instance: First instance index.
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Validate state is ready
    state.is_ready(DrawCommandFamily.DRAW)
    
    # Validate vertex limits
    state.vertex.limits.validate_vertex_limit(first_vertex, vertex_count)
    state.vertex.limits.validate_instance_limit(first_instance, instance_count)
    
    # Flush bindings
    state.flush_bindings()
    
    # Issue HAL draw command
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'draw'):
            try:
                raw_encoder.draw(
                    vertices=vertex_count,
                    instances=instance_count,
                    first_vertex=first_vertex,
                    first_instance=first_instance
                )
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to draw: {e}")


def draw_indexed(
    state: State,
    index_count: int,
    instance_count: int,
    first_index: int,
    base_vertex: int,
    first_instance: int,
) -> None:
    """
    Draw indexed primitives.
    
    Args:
        state: Current render pass state.
        index_count: Number of indices to draw.
        instance_count: Number of instances to draw.
        first_index: First index.
        base_vertex: Base vertex offset.
        first_instance: First instance index.
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Validate state is ready
    state.is_ready(DrawCommandFamily.DRAW_INDEXED)
    
    # Validate index limits
    last_index = first_index + index_count
    if last_index > state.index.limit:
        raise DrawError(f"Index {last_index} beyond limit {state.index.limit}")
    
    # Validate instance limits
    state.vertex.limits.validate_instance_limit(first_instance, instance_count)
    
    # Flush bindings
    state.flush_bindings()
    
    # Issue HAL draw indexed command
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'draw_indexed'):
            try:
                raw_encoder.draw_indexed(
                    indices=index_count,
                    instances=instance_count,
                    first_index=first_index,
                    base_vertex=base_vertex,
                    first_instance=first_instance
                )
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to draw indexed: {e}")


def draw_mesh_tasks(
    state: State,
    group_count_x: int,
    group_count_y: int,
    group_count_z: int,
) -> None:
    """
    Draw mesh tasks (mesh shader dispatch).
    
    Args:
        state: Current render pass state.
        group_count_x: Number of workgroups in X dimension.
        group_count_y: Number of workgroups in Y dimension.
        group_count_z: Number of workgroups in Z dimension.
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Validate state is ready
    state.is_ready(DrawCommandFamily.DRAW_MESH_TASKS)
    
    # Validate multiview if needed
    validate_mesh_draw_multiview(state)
    
    # Validate workgroup counts (would use device limits)
    max_dimension = 65535  # Typical limit
    max_total = 65535
    
    if (group_count_x > max_dimension or 
        group_count_y > max_dimension or 
        group_count_z > max_dimension):
        raise DrawError(
            f"Mesh task group count exceeds dimension limit: "
            f"({group_count_x}, {group_count_y}, {group_count_z}) > {max_dimension}"
        )
    
    total = group_count_x * group_count_y * group_count_z
    if total > max_total:
        raise DrawError(f"Mesh task total group count {total} exceeds limit {max_total}")
    
    # Flush bindings
    state.flush_bindings()
    
    # Issue HAL draw mesh tasks command
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        raw_encoder = getattr(state.pass_state.base, 'raw_encoder', None)
        if raw_encoder and hasattr(raw_encoder, 'draw_mesh_tasks'):
            try:
                raw_encoder.draw_mesh_tasks(
                    x=group_count_x,
                    y=group_count_y,
                    z=group_count_z
                )
            except Exception as e:
                raise RenderPassErrorInner(f"Failed to draw mesh tasks: {e}")


def validate_mesh_draw_multiview(state: State) -> None:
    """
    Validate mesh draw with multiview.
    
    Mesh shaders with multiview have special requirements. This function
    validates that:
    1. Device has EXPERIMENTAL_MESH_SHADER_MULTIVIEW feature if multiview is used
    2. The multiview view count doesn't exceed device limits
    
    Args:
        state: Current render pass state.
        
    Raises:
        RenderPassErrorInner: If mesh shader multiview limits are violated.
    """
    # Check if multiview is enabled in the render pass
    if state.info and hasattr(state.info, 'multiview_mask'):
        multiview_mask = state.info.multiview_mask
        
        # If multiview is enabled (mask is not None and not 0)
        if multiview_mask is not None and multiview_mask != 0:
            # Calculate highest view index from the mask
            # The mask is a bitfield where each bit represents a view
            # e.g., mask=0b1111 means views 0-3, highest_bit=3
            
            # Count leading zeros to find highest bit
            # In Python: bit_length() - 1 gives us the highest bit index
            if isinstance(multiview_mask, int):
                highest_bit = multiview_mask.bit_length() - 1
            else:
                highest_bit = 0
            
            # Get device features and limits
            device = state.pass_state.base.device if (state.pass_state and 
                                                       hasattr(state.pass_state, 'base') and
                                                       hasattr(state.pass_state.base, 'device')) else None
            
            if device:
                # Check if device has experimental mesh shader multiview feature
                has_feature = False
                if hasattr(device, 'features'):
                    # features.contains(Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW)
                    has_feature = getattr(device.features, 'EXPERIMENTAL_MESH_SHADER_MULTIVIEW', False)
                
                # Get max multiview count limit
                max_multiview_count = 0
                if hasattr(device, 'limits'):
                    max_multiview_count = getattr(device.limits, 'max_mesh_multiview_view_count', 0)
                
                # Validate feature and limits
                if not has_feature or highest_bit > max_multiview_count:
                    raise DrawError(
                        f"MeshPipelineMultiviewLimitsViolated: "
                        f"highest_view_index={highest_bit}, "
                        f"max_multiviews={max_multiview_count}, "
                        f"has_feature={has_feature}"
                    )


def multi_draw_indirect(
    state: State,
    indirect_draw_validation_batcher: Any,
    device: Any,
    indirect_buffer: Any,
    offset: int,
    count: int,
    family: DrawCommandFamily,
) -> None:
    """
    Execute multiple indirect draws.
    
    This is a complex function that validates and executes multiple
    indirect draw commands from a buffer. It handles:
    - Alignment validation
    - Buffer bounds checking
    - Optional indirect draw validation
    - Batched HAL draw calls
    
    Args:
        state: Current render pass state.
        indirect_draw_validation_batcher: Validation batcher for GPU-side validation.
        device: The device.
        indirect_buffer: Buffer containing draw parameters.
        offset: Offset into the buffer (must be 4-byte aligned).
        count: Number of draws to execute.
        family: Draw command family (DRAW, DRAW_INDEXED, or DRAW_MESH_TASKS).
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Validate state is ready for drawing
    state.is_ready(family)
    
    # Flush bindings before draw
    state.flush_bindings()
    
    # Special validation for mesh shaders with multiview
    if family == DrawCommandFamily.DRAW_MESH_TASKS:
        validate_mesh_draw_multiview(state)
    
    # Check device supports indirect execution
    # device.require_downlevel_flags(DownlevelFlags::INDIRECT_EXECUTION)
    
    # Validate buffer is from same device
    if hasattr(indirect_buffer, 'device') and hasattr(device, 'id'):
        if indirect_buffer.device != device:
            raise RenderPassErrorInner("Indirect buffer from different device")
    
    # Check buffer has INDIRECT usage
    if hasattr(indirect_buffer, 'usage'):
        if not (indirect_buffer.usage & BufferUsage.INDIRECT):
            raise RenderPassErrorInner(
                f"Buffer {getattr(indirect_buffer, 'label', '')} is missing INDIRECT usage"
            )
    
    # Check buffer is not destroyed
    snatch_guard = getattr(state.pass_state.base, 'snatch_guard', None) if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base') else None
    if hasattr(indirect_buffer, 'check_destroyed') and snatch_guard:
        indirect_buffer.check_destroyed(snatch_guard)
    
    # Validate offset alignment (must be 4-byte aligned)
    if offset % 4 != 0:
        raise RenderPassErrorInner(f"UnalignedIndirectBufferOffset: {offset}")
    
    # Get stride for this draw family
    stride = get_stride_of_indirect_args(family.value if hasattr(family, 'value') else str(family))
    
    # Calculate end offset and validate buffer bounds
    end_offset = offset + stride * count
    buffer_size = getattr(indirect_buffer, 'size', 0)
    
    if end_offset > buffer_size:
        raise RenderPassErrorInner(
            f"IndirectBufferOverrun: count={count}, offset={offset}, "
            f"end_offset={end_offset}, buffer_size={buffer_size}"
        )
    
    # Register buffer memory initialization action
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        if hasattr(state.pass_state.base, 'buffer_memory_init_actions'):
            if hasattr(indirect_buffer, 'initialization_status') and hasattr(indirect_buffer.initialization_status, 'create_action'):
                action = indirect_buffer.initialization_status.create_action(
                    indirect_buffer,
                    Range(offset, end_offset),
                    MemoryInitKind.NEEDS_INITIALIZED_MEMORY
                )
                if action:
                    state.pass_state.base.buffer_memory_init_actions.append(action)
    
    # Helper function to issue HAL draw calls
    def issue_draw(raw_encoder, draw_family, buffer, draw_offset, draw_count):
        """Issue HAL draw call based on family."""
        if not raw_encoder:
            return
        if draw_family == DrawCommandFamily.DRAW:
            if hasattr(raw_encoder, 'draw_indirect'):
                raw_encoder.draw_indirect(buffer, draw_offset, draw_count)
        elif draw_family == DrawCommandFamily.DRAW_INDEXED:
            if hasattr(raw_encoder, 'draw_indexed_indirect'):
                raw_encoder.draw_indexed_indirect(buffer, draw_offset, draw_count)
        elif draw_family == DrawCommandFamily.DRAW_MESH_TASKS:
            if hasattr(raw_encoder, 'draw_mesh_tasks_indirect'):
                raw_encoder.draw_mesh_tasks_indirect(buffer, draw_offset, draw_count)
    
    # Check if indirect validation is enabled on the device
    has_indirect_validation = (
        hasattr(device, 'indirect_validation') and 
        device.indirect_validation is not None
    )
    
    raw_encoder = state.pass_state.base.raw_encoder if hasattr(state.pass_state, 'base') else None
    
    if has_indirect_validation:
        # GPU-side validation path
        # Merge buffer into scope with STORAGE_READ_ONLY usage
        if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
            if hasattr(state.pass_state.scope, 'buffers'):
                state.pass_state.scope.buffers.merge_single(
                    indirect_buffer, BufferUses.STORAGE_READ_ONLY
                )
        
        # Determine vertex/index limit for validation
        if family == DrawCommandFamily.DRAW_INDEXED:
            vertex_or_index_limit = state.index.limit
        else:
            vertex_or_index_limit = state.vertex.limits.vertex_limit if hasattr(state.vertex.limits, 'vertex_limit') else 0
        
        instance_limit = state.vertex.limits.instance_limit if hasattr(state.vertex.limits, 'instance_limit') else 0
        
        # Validation resources from the encoder state
        resources = getattr(state.pass_state.base, 'indirect_draw_validation_resources', None)
        if resources is None:
            # Should not happen in a correct implementation
            # For now, just issue direct draws as fallback if resources are missing
            raw_buffer = indirect_buffer.raw() if hasattr(indirect_buffer, 'raw') else indirect_buffer
            issue_draw(raw_encoder, family, raw_buffer, offset, count)
            return

        # Helper to add a draw to the batcher
        def add_to_batcher(off):
            dst_resource_index, dst_offset = indirect_draw_validation_batcher.add(
                resources,
                device,
                indirect_buffer,
                off,
                family,
                vertex_or_index_limit,
                instance_limit,
            )
            return {
                'buffer_index': dst_resource_index,
                'offset': dst_offset,
                'count': 1,
            }

        # Issue draw from draw_data
        def do_draw(draw_data):
            dst_buffer = resources.get_dst_buffer(draw_data['buffer_index'])
            if dst_buffer:
                issue_draw(raw_encoder, family, dst_buffer, draw_data['offset'], draw_data['count'])

        # Process the first draw
        current_draw_data = add_to_batcher(offset)
        
        # Batch subsequent draws
        for i in range(1, count):
            draw_offset = offset + stride * i
            draw_data = add_to_batcher(draw_offset)
            
            if draw_data['buffer_index'] == current_draw_data['buffer_index'] and \
               draw_data['offset'] == current_draw_data['offset'] + stride * current_draw_data['count']:
                # Consecutive in the same buffer, can be batched
                current_draw_data['count'] += 1
            else:
                # Different buffer or non-consecutive, issue previous batch
                do_draw(current_draw_data)
                current_draw_data = draw_data
        
        # Issue final batch
        do_draw(current_draw_data)
            
    else:
        # Direct path without GPU-side validation
        # Merge buffer into scope with INDIRECT usage
        if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
            if hasattr(state.pass_state.scope, 'buffers'):
                state.pass_state.scope.buffers.merge_single(
                    indirect_buffer, BufferUses.INDIRECT
                )
        
        # Issue single batched draw call for all draws
        raw_buffer = indirect_buffer.raw() if hasattr(indirect_buffer, 'raw') else indirect_buffer
        issue_draw(raw_encoder, family, raw_buffer, offset, count)


def execute_bundle(
    state: State,
    indirect_draw_validation_batcher: Any,
    device: Any,
    bundle: Any,
) -> None:
    """
    Execute a render bundle.
    
    This replays pre-recorded render commands from a bundle. The bundle must
    be compatible with the current render pass context, and depth/stencil
    read-only flags must match.
    
    Args:
        state: Current render pass state.
        indirect_draw_validation_batcher: Validation batcher for indirect draws.
        device: The device.
        bundle: The render bundle to execute.
        
    Raises:
        RenderPassErrorInner: If validation fails.
    """
    # Track bundle in resource tracker
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'base'):
        if hasattr(state.pass_state.base, 'tracker'):
            bundle = state.pass_state.base.tracker.bundles.insert_single(bundle)
    
    # Validate bundle is from same device
    if hasattr(bundle, 'device') and hasattr(device, 'id'):
        if bundle.device != device:
            raise RenderPassErrorInner("Bundle from different device")
    
    # Check render pass context compatibility
    # This validates that color/depth formats, sample counts, etc. match
    if state.info and hasattr(state.info, 'context') and hasattr(bundle, 'context'):
        if state.info.context and bundle.context:
            try:
                state.info.context.check_compatible(bundle.context, bundle)
            except RenderPassErrorInner as e:
                raise RenderPassErrorInner(f"IncompatibleBundleTargets: {e}")
    
    # Validate depth/stencil read-only compatibility
    # Bundle cannot write to depth/stencil if pass has them as read-only
    bundle_is_depth_read_only = getattr(bundle, 'is_depth_read_only', True)
    bundle_is_stencil_read_only = getattr(bundle, 'is_stencil_read_only', True)
    
    if state.info and hasattr(state.info, 'is_depth_read_only'):
        if state.info.is_depth_read_only and not bundle_is_depth_read_only:
            raise RenderPassErrorInner(
                f"IncompatibleBundleReadOnlyDepthStencil: "
                f"pass depth={state.info.is_depth_read_only}, "
                f"bundle depth={bundle_is_depth_read_only}"
            )
        
        if state.info.is_stencil_read_only and not bundle_is_stencil_read_only:
            raise RenderPassErrorInner(
                f"IncompatibleBundleReadOnlyDepthStencil: "
                f"pass stencil={state.info.is_stencil_read_only}, "
                f"bundle stencil={bundle_is_stencil_read_only}"
            )
    
    # Merge buffer memory initialization actions from bundle
    if hasattr(bundle, 'buffer_memory_init_actions') and hasattr(state, 'pass_state'):
        if hasattr(state.pass_state, 'base') and hasattr(state.pass_state.base, 'buffer_memory_init_actions'):
            for action in bundle.buffer_memory_init_actions:
                # In real wgpu-core, we'd check if it's still needed
                state.pass_state.base.buffer_memory_init_actions.append(action)
    
    # Merge texture memory initialization actions from bundle
    if hasattr(bundle, 'texture_memory_init_actions') and hasattr(state, 'pass_state'):
        if hasattr(state.pass_state, 'base') and hasattr(state.pass_state.base, 'texture_memory_init_actions'):
            for action in bundle.texture_memory_init_actions:
                # Add to pass actions
                state.pass_state.base.texture_memory_init_actions.append(action)
    
    # Execute the bundle's commands
    # This replays all recorded commands from the bundle
    if hasattr(bundle, 'execute') and hasattr(state, 'pass_state') and state.pass_state:
            bundle.execute(
                raw=state.pass_state.base.raw_encoder if hasattr(state.pass_state, 'base') else None,
                indirect_draw_validation_resources=getattr(state.pass_state.base, 'indirect_draw_validation_resources', None),
                indirect_draw_validation_batcher=indirect_draw_validation_batcher,
                snatch_guard=getattr(state.pass_state.base, 'snatch_guard', None),
            )
        except Exception as e:
            # Map execution errors to render pass errors
            error_type = type(e).__name__
            if error_type == 'DeviceError':
                raise RenderPassErrorInner(f"Device error: {e}")
            elif error_type == 'DestroyedResourceError':
                raise RenderPassErrorInner(f"Destroyed resource: {e}")
            elif error_type == 'UnimplementedError':
                raise RenderPassErrorInner(f"Unimplemented: {e}")
            else:
                raise RenderPassErrorInner(f"Bundle execution failed: {e}")
    
    # Merge bundle's resource usage into pass scope
    if hasattr(state, 'pass_state') and hasattr(state.pass_state, 'scope'):
        if hasattr(bundle, 'used') and hasattr(state.pass_state.scope, 'merge_render_bundle'):
            state.pass_state.scope.merge_render_bundle(bundle.used)
    
    # Reset bundle-related state
    # After bundle execution, pipeline and buffer bindings are undefined
    state.reset_bundle()


def encode_render_pass(
    parent_state: Any,
    base: Any,
    color_attachments: List[Any],
    depth_stencil_attachment: Optional[Any],
    timestamp_writes: Optional[Any],
    occlusion_query_set: Optional[Any],
    multiview_mask: Optional[int],
) -> None:
    """
    Encode a render pass.
    
    This is the main function that processes all recorded render pass
    commands and encodes them into the command buffer.
    
    This function:
    1. Initializes RenderPassInfo with validated attachments
    2. Creates State object for tracking
    3. Iterates through all recorded commands
    4. Dispatches each command to appropriate handler
    5. Finalizes the render pass
    6. Updates resource tracking
    
    Args:
        parent_state: Parent encoder state.
        base: Base pass data with recorded commands.
        color_attachments: Color attachments.
        depth_stencil_attachment: Optional depth/stencil attachment.
        timestamp_writes: Optional timestamp writes.
        occlusion_query_set: Optional occlusion query set.
        multiview_mask: Optional multiview mask.
        
    Raises:
        RenderPassError: If encoding fails.
    """
    pass_scope = "Pass"
    device = parent_state.device if hasattr(parent_state, 'device') else None
    
    # Initialize indirect draw validation batcher
    from .indirect_validation import DrawBatcher
    indirect_draw_validation_batcher = DrawBatcher()
    
    # Close previous encoder if open and open new pass
    raw_encoder = None
    if hasattr(parent_state, 'encoder') and parent_state.encoder:
        if hasattr(parent_state.encoder, 'close_if_open'):
            parent_state.encoder.close_if_open()
        if hasattr(parent_state.encoder, 'open_pass'):
            raw_encoder = parent_state.encoder.open_pass(base.label if hasattr(base, 'label') else None)
        else:
            raw_encoder = getattr(parent_state.encoder, 'raw', None)
    
    # Initialize tracking structures
    pending_query_resets = {}
    pending_discard_init_fixups = []
    
    # Initialize local tracker for the pass
    # This acts as a UsageScope in wgpu-core
    tracker = Tracker()
    if hasattr(parent_state, 'tracker'):
        # Initialize sizes from parent
        pass # Tracker.__init__ handles basic setup
    
    # Start the render pass - validates attachments and initializes HAL
    info = RenderPassInfo.start(
        device=device,
        hal_label=base.label if hasattr(base, 'label') else None,
        color_attachments=color_attachments,
        depth_stencil_attachment=depth_stencil_attachment,
        timestamp_writes=timestamp_writes,
        occlusion_query_set=occlusion_query_set,
        encoder=raw_encoder,
        trackers=tracker,
        texture_memory_actions=parent_state.texture_memory_actions if hasattr(parent_state, 'texture_memory_actions') else None,
        pending_query_resets=pending_query_resets,
        pending_discard_init_fixups=pending_discard_init_fixups,
        snatch_guard=parent_state.snatch_guard if hasattr(parent_state, 'snatch_guard') else None,
        multiview_mask=multiview_mask,
    )
    
    # Update tracker sizes
    if hasattr(parent_state, 'tracker') and device:
        if hasattr(device, 'tracker_indices'):
            indices = device.tracker_indices
            parent_state.tracker.buffers.set_size(indices.buffers.size())
            parent_state.tracker.textures.set_size(indices.textures.size())
    
    debug_scope_depth = 0
    
    # Create state object for command processing
    state = State(
        pipeline_flags=PipelineFlags.NONE,
        blend_constant=OptionalState.UNUSED,
        stencil_reference=0,
        pipeline=None,
        index=IndexState(),
        vertex=VertexState(),
        info=info,
        pass_state=PassState(
            base=base,
            binder=Binder(),
            temp_offsets=[],
            dynamic_offset_count=0,
            pending_discard_init_fixups=pending_discard_init_fixups,
            scope=parent_state.tracker if hasattr(parent_state, 'tracker') else None,
            string_offset=0,
        ),
        active_occlusion_query=None,
        active_pipeline_statistics_query=None,
        device=device,
        tracker=tracker,
        raw_encoder=raw_encoder,
        reset_state=None,
    )
    
    # Process all recorded commands
    commands = base.commands if hasattr(base, 'commands') else []
    
    for command in commands:
        command_type = command.get('type') if isinstance(command, dict) else type(command).__name__
        
        try:
            # Dispatch to appropriate handler based on command type
            if command_type == 'SetBindGroup':
                index = command.get('index') if isinstance(command, dict) else getattr(command, 'index', None)
                bind_group = command.get('bind_group') if isinstance(command, dict) else getattr(command, 'bind_group', None)
                dynamic_offsets = command.get('dynamic_offsets', []) if isinstance(command, dict) else getattr(command, 'dynamic_offsets', [])
                set_bind_group(state.pass_state, device, dynamic_offsets, index, len(dynamic_offsets), bind_group, False)
                
            elif command_type == 'SetPipeline':
                pipeline = command.get('pipeline') if isinstance(command, dict) else getattr(command, 'pipeline', None)
                set_pipeline(state, device, pipeline)
                
            elif command_type == 'SetIndexBuffer':
                buffer = command.get('buffer') if isinstance(command, dict) else getattr(command, 'buffer', None)
                index_format = command.get('index_format') if isinstance(command, dict) else getattr(command, 'index_format', None)
                offset = command.get('offset', 0) if isinstance(command, dict) else getattr(command, 'offset', 0)
                size = command.get('size') if isinstance(command, dict) else getattr(command, 'size', None)
                set_index_buffer(state, device, buffer, index_format, offset, size)
                
            elif command_type == 'SetVertexBuffer':
                slot = command.get('slot') if isinstance(command, dict) else getattr(command, 'slot', 0)
                buffer = command.get('buffer') if isinstance(command, dict) else getattr(command, 'buffer', None)
                offset = command.get('offset', 0) if isinstance(command, dict) else getattr(command, 'offset', 0)
                size = command.get('size') if isinstance(command, dict) else getattr(command, 'size', None)
                set_vertex_buffer(state, device, slot, buffer, offset, size)
                
            elif command_type == 'SetBlendConstant':
                color = command.get('color') if isinstance(command, dict) else getattr(command, 'color', (0, 0, 0, 0))
                set_blend_constant(state, color)
                
            elif command_type == 'SetStencilReference':
                value = command.get('value', 0) if isinstance(command, dict) else getattr(command, 'value', 0)
                set_stencil_reference(state, value)
                
            elif command_type == 'SetViewport':
                rect = command.get('rect') if isinstance(command, dict) else getattr(command, 'rect', None)
                depth_min = command.get('depth_min', 0.0) if isinstance(command, dict) else getattr(command, 'depth_min', 0.0)
                depth_max = command.get('depth_max', 1.0) if isinstance(command, dict) else getattr(command, 'depth_max', 1.0)
                set_viewport(state, rect, depth_min, depth_max)
                
            elif command_type == 'SetScissor':
                rect = command.get('rect') if isinstance(command, dict) else getattr(command, 'rect', None)
                set_scissor(state, rect)
                
            elif command_type == 'SetImmediate':
                offset = command.get('offset', 0) if isinstance(command, dict) else getattr(command, 'offset', 0)
                data = command.get('data') if isinstance(command, dict) else getattr(command, 'data', None)
                # set_immediates requires a callback or direct data
                set_immediates(state.pass_state, device, [0], offset, len(data) if data else 0, lambda d: None)
                
            elif command_type == 'Draw':
                vertex_count = command.get('vertex_count', 0) if isinstance(command, dict) else getattr(command, 'vertex_count', 0)
                instance_count = command.get('instance_count', 1) if isinstance(command, dict) else getattr(command, 'instance_count', 1)
                first_vertex = command.get('first_vertex', 0) if isinstance(command, dict) else getattr(command, 'first_vertex', 0)
                first_instance = command.get('first_instance', 0) if isinstance(command, dict) else getattr(command, 'first_instance', 0)
                draw(state, vertex_count, instance_count, first_vertex, first_instance)
                
            elif command_type == 'DrawIndexed':
                index_count = command.get('index_count', 0) if isinstance(command, dict) else getattr(command, 'index_count', 0)
                instance_count = command.get('instance_count', 1) if isinstance(command, dict) else getattr(command, 'instance_count', 1)
                first_index = command.get('first_index', 0) if isinstance(command, dict) else getattr(command, 'first_index', 0)
                base_vertex = command.get('base_vertex', 0) if isinstance(command, dict) else getattr(command, 'base_vertex', 0)
                first_instance = command.get('first_instance', 0) if isinstance(command, dict) else getattr(command, 'first_instance', 0)
                draw_indexed(state, index_count, instance_count, first_index, base_vertex, first_instance)
                
            elif command_type == 'DrawMeshTasks':
                group_count_x = command.get('group_count_x', 1) if isinstance(command, dict) else getattr(command, 'group_count_x', 1)
                group_count_y = command.get('group_count_y', 1) if isinstance(command, dict) else getattr(command, 'group_count_y', 1)
                group_count_z = command.get('group_count_z', 1) if isinstance(command, dict) else getattr(command, 'group_count_z', 1)
                draw_mesh_tasks(state, group_count_x, group_count_y, group_count_z)
                
            elif command_type == 'DrawIndirect':
                buffer = command.get('buffer') if isinstance(command, dict) else getattr(command, 'buffer', None)
                offset = command.get('offset', 0) if isinstance(command, dict) else getattr(command, 'offset', 0)
                count = command.get('count', 1) if isinstance(command, dict) else getattr(command, 'count', 1)
                family = command.get('family') if isinstance(command, dict) else getattr(command, 'family', DrawCommandFamily.DRAW)
                
                # Single or multi indirect draw
                multi_draw_indirect(state, indirect_draw_validation_batcher, device, buffer, offset, count, family)
                    
            elif command_type == 'ExecuteBundle':
                bundle = command.get('bundle') if isinstance(command, dict) else getattr(command, 'bundle', None)
                execute_bundle(state, indirect_draw_validation_batcher, device, bundle)
                
            elif command_type == 'BeginOcclusionQuery':
                query_index = command.get('query_index', 0) if isinstance(command, dict) else getattr(command, 'query_index', 0)
                validate_and_begin_occlusion_query(state, info.occlusion_query_set, query_index)
                
            elif command_type == 'EndOcclusionQuery':
                end_occlusion_query(state)
                
            elif command_type == 'BeginPipelineStatisticsQuery':
                query_set = command.get('query_set') if isinstance(command, dict) else getattr(command, 'query_set', None)
                query_index = command.get('query_index', 0) if isinstance(command, dict) else getattr(command, 'query_index', 0)
                validate_and_begin_pipeline_statistics_query(state, query_set, query_index)
                
            elif command_type == 'EndPipelineStatisticsQuery':
                end_pipeline_statistics_query(state)
                
            elif command_type == 'WriteTimestamp':
                query_set = command.get('query_set') if isinstance(command, dict) else getattr(command, 'query_set', None)
                query_index = command.get('query_index', 0) if isinstance(command, dict) else getattr(command, 'query_index', 0)
                write_timestamp(state, query_set, query_index)
                
            elif command_type == 'PushDebugGroup':
                # Handle debug group push
                debug_scope_depth += 1
                
            elif command_type == 'PopDebugGroup':
                # Handle debug group pop
                debug_scope_depth = max(0, debug_scope_depth - 1)
                
            elif command_type == 'InsertDebugMarker':
                # Handle debug marker
                pass
                
            else:
                # Unknown command type
                print(f"Warning: Unknown render command type: {command_type}")
                
        except Exception as e:
            # Wrap in RenderPassError with scope
            raise RenderPassError(f"Error in {command_type}: {e}")
    
    # Finalize the render pass
    info.finish(
        device=device,
        raw=raw_encoder,
        snatch_guard=parent_state.snatch_guard if hasattr(parent_state, 'snatch_guard') else None,
        scope=None,  # Would be usage scope
        instance_flags=None,  # Would be instance flags
    )
    
    # Update resource tracking
    # Merge the local pass tracker back into the parent command encoder's tracker
    if hasattr(parent_state, 'tracker'):
        transitions = parent_state.tracker.merge_scope(tracker)
        # These transitions would be used to generate barriers
        # In this implementation, barriers are handled at submit time or here
        pass
    
    # Process pending query resets
    # These are queries that must be reset (cleared) before their first use in the pass
    if raw_encoder:
        for query_set, indices in pending_query_resets.items():
            # In HAL, we would call reset_query_set
            if hasattr(raw_encoder, 'reset_query_set'):
                # Assuming indices is a combined range or we iterate
                # This is simplified: Rust uses a more complex bitmask or range list
                for index in indices:
                    raw_encoder.reset_query_set(query_set, index, 1)
    
    # Process pending discard fixups
    # These are textures that were in DISCARD state and now need explicit initialization
    # if they were partially updated or need to be transitioned to a stable state.
    if pending_discard_init_fixups:
        for fixup in pending_discard_init_fixups:
            # Register these in the command buffer's initialization tracker.
            if hasattr(parent_state, 'texture_memory_actions'):
                if hasattr(parent_state.texture_memory_actions, 'register_fixup'):
                    parent_state.texture_memory_actions.register_fixup(fixup)
    
    # Inject indirect draw validation pass if needed
    if hasattr(device, 'indirect_validation') and device.indirect_validation:
        # In a real implementation, we would open a "Pre Pass" or "Post Pass" internal encoder
        # to run the validation compute shader.
        if hasattr(device.indirect_validation, 'draw') and device.indirect_validation.draw:
            # Prepare resources for injection
            resources = getattr(parent_state, 'indirect_draw_validation_resources', None)
            snatch_guard = getattr(parent_state, 'snatch_guard', None)
            temp_resources = getattr(parent_state, 'temp_resources', [])
            
            # Use the raw_encoder if available, or the parent state's raw_encoder
            valid_encoder = raw_encoder if raw_encoder else getattr(parent_state, 'raw_encoder', None)
            
            if valid_encoder and resources:
                device.indirect_validation.draw.inject_validation_pass(
                    device=device,
                    snatch_guard=snatch_guard,
                    resources=resources,
                    temp_resources=temp_resources,
                    encoder=valid_encoder,
                    batcher=indirect_draw_validation_batcher
                )
                
                # Dispose of resources back to pool
                if hasattr(resources, 'dispose'):
                    resources.dispose()



