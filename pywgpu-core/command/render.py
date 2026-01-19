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
from enum import Enum


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
    """State of an optional value."""
    UNUSED = "unused"
    REQUIRED = "required"
    SET = "set"


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
        """Update vertex limits based on pipeline configuration."""
        # Simplified - full implementation would calculate limits from buffer sizes and pipeline
        pass


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
