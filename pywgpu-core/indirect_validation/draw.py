from typing import Any, Optional, List, Dict, Tuple, Iterator
from dataclasses import dataclass, field
import struct
import math


def get_stride_of_indirect_args(family: str) -> int:
    """Get the stride of indirect arguments for a given family."""
    if family == "draw":
        return 16
    elif family == "draw_indexed":
        return 20
    elif family == "draw_mesh_tasks":
        return 12
    return 0


def calculate_src_buffer_binding_size(buffer_size: int, limits: dict) -> int:
    """
    Calculate binding size for source buffer.
    
    Returns the largest binding size that when combined with dynamic
    offsets can address the whole buffer.
    """
    max_storage_buffer_binding_size = limits.get('max_storage_buffer_binding_size', 2**30)
    min_storage_buffer_offset_alignment = limits.get('min_storage_buffer_offset_alignment', 256)
    
    if buffer_size <= max_storage_buffer_binding_size:
        return buffer_size
    
    # Logic from Rust to ensure coverage with dynamic offsets
    buffer_rem = buffer_size % min_storage_buffer_offset_alignment
    binding_rem = max_storage_buffer_binding_size % min_storage_buffer_offset_alignment
    
    if buffer_rem <= binding_rem:
        return max_storage_buffer_binding_size - binding_rem + buffer_rem
    else:
        return (max_storage_buffer_binding_size - binding_rem - 
                min_storage_buffer_offset_alignment + buffer_rem)


def calculate_src_offsets(buffer_size: int, limits: dict, offset: int) -> Tuple[int, int]:
    """Splits the given offset into a dynamic offset & offset."""
    max_storage_buffer_binding_size = limits.get('max_storage_buffer_binding_size', 2**30)
    min_storage_buffer_offset_alignment = limits.get('min_storage_buffer_offset_alignment', 256)

    binding_size = calculate_src_buffer_binding_size(buffer_size, limits)
    
    # Chunk adjustment logic from Rust
    if min_storage_buffer_offset_alignment == 4:
        chunk_adjustment = 0
    elif min_storage_buffer_offset_alignment == 8:
        chunk_adjustment = 2
    else:
        chunk_adjustment = 1
        
    chunks = binding_size // min_storage_buffer_offset_alignment
    dynamic_offset_stride = max(0, chunks - chunk_adjustment) * min_storage_buffer_offset_alignment
    
    if dynamic_offset_stride == 0:
        return (0, offset)
        
    max_dynamic_offset = buffer_size - binding_size
    max_dynamic_offset_index = max_dynamic_offset // dynamic_offset_stride
    
    src_dynamic_offset_index = offset // dynamic_offset_stride
    src_dynamic_offset = min(src_dynamic_offset_index, max_dynamic_offset_index) * dynamic_offset_stride
    src_offset = offset - src_dynamic_offset
    
    return (src_dynamic_offset, src_offset)


class DrawValidator:
    """Validator for indirect draw commands."""
    
    def __init__(self, device: Any):
        """
        Create draw validator.
        
        Args:
            device: The device.
        """
        self.device = device
    
    def validate(self, buffer: Any, offset: int, indexed: bool = False) -> bool:
        """
        Validate an indirect draw command.
        
        Args:
            buffer: The indirect buffer.
            offset: Offset into the buffer.
            indexed: Whether this is an indexed draw.
            
        Returns:
            True if valid, False otherwise.
        """
        # Validate buffer size and alignment
        if not hasattr(buffer, 'size'):
            return False
        
        # Draw indirect requires 16 bytes (4 u32s)
        # DrawIndexed indirect requires 20 bytes (5 u32s)
        required_size = 20 if indexed else 16
        
        if offset + required_size > buffer.size:
            return False
        
        # Offset must be 4-byte aligned
        if offset % 4 != 0:
            return False
        
        return True


@dataclass
class MetadataEntry:
    """Matches the MetadataEntry struct used by the validation shader."""
    src_offset: int  # u32
    dst_offset: int  # u32
    vertex_or_index_limit: int  # u32
    instance_limit: int  # u32

    @classmethod
    def new(
        cls,
        indexed: bool,
        src_offset: int,
        dst_offset: int,
        vertex_or_index_limit: int,
        instance_limit: int,
    ) -> 'MetadataEntry':
        # Pack indexed flag and limits according to Rust logic:
        # src_offset | (indexed << 31)
        src_u32 = (src_offset // 4) | (31 << (1 if indexed else 0)) # Wait, bit 31
        src_u32 = (src_offset // 4) | ((1 << 31) if indexed else 0)
        
        # max_limit = u32::MAX + u32::MAX
        max_limit = 0xFFFFFFFF + 0xFFFFFFFF
        
        v_limit = min(vertex_or_index_limit, max_limit)
        v_limit_bit_32 = (v_limit >> 32) & 1
        v_limit_u32 = v_limit & 0xFFFFFFFF
        
        i_limit = min(instance_limit, max_limit)
        i_limit_bit_32 = (i_limit >> 32) & 1
        i_limit_u32 = i_limit & 0xFFFFFFFF
        
        dst_u32 = (dst_offset // 4) | (v_limit_bit_32 << 30) | (i_limit_bit_32 << 31)
        
        return cls(
            src_offset=src_u32,
            dst_offset=dst_u32,
            vertex_or_index_limit=v_limit_u32,
            instance_limit=i_limit_u32
        )

    def to_bytes(self) -> bytes:
        return struct.pack("<4I", self.src_offset, self.dst_offset, self.vertex_or_index_limit, self.instance_limit)


@dataclass
class DrawIndirectValidationBatch:
    src_buffer: Any
    src_dynamic_offset: int
    dst_resource_index: int
    entries: List[MetadataEntry] = field(default_factory=list)

    staging_buffer_index: int = 0
    staging_buffer_offset: int = 0
    metadata_resource_index: int = 0
    metadata_buffer_offset: int = 0

    def metadata_bytes(self) -> bytes:
        return b"".join(e.to_bytes() for e in self.entries)


class DrawBatcher:
    """Accumulates indirect draws for validation."""
    
    def __init__(self):
        self.batches: Dict[Tuple, DrawIndirectValidationBatch] = {}
        self.current_dst_entry: Optional[Any] = None

    def add(
        self,
        indirect_draw_validation_resources: 'DrawResources',
        device: Any,
        src_buffer: Any,
        offset: int,
        family: Any, # DrawCommandFamily
        vertex_or_index_limit: int,
        instance_limit: int,
    ) -> Tuple[int, int]:
        from . import draw
        # stride = get_stride_of_indirect_args(family)
        # In a real implementation we would also check backend for DX12 special constants
        stride = get_stride_of_indirect_args(family.name if hasattr(family, 'name') else str(family))
        
        dst_resource_index, dst_offset = indirect_draw_validation_resources.get_dst_subrange(
            stride, self
        )
        
        buffer_size = getattr(src_buffer, 'size', 0)
        limits = getattr(device, 'limits', {})
        src_dynamic_offset, src_offset = calculate_src_offsets(buffer_size, limits, offset)
        
        # key = (src_buffer.id, src_dynamic_offset, dst_resource_index)
        key = (id(src_buffer), src_dynamic_offset, dst_resource_index)
        
        indexed = (family.name == "DRAW_INDEXED" if hasattr(family, 'name') else "indexed" in str(family).lower())
        entry = MetadataEntry.new(
            indexed,
            src_offset,
            dst_offset,
            vertex_or_index_limit,
            instance_limit
        )
        
        if key not in self.batches:
            self.batches[key] = DrawIndirectValidationBatch(
                src_buffer=src_buffer,
                src_dynamic_offset=src_dynamic_offset,
                dst_resource_index=dst_resource_index,
                entries=[entry]
            )
        else:
            self.batches[key].entries.append(entry)
            
        return (dst_resource_index, dst_offset)


class DrawResources:
    """Holds command buffer-level resources for indirect draw validation."""
    
    def __init__(self, device: Any):
        self.device = device
        self.dst_entries: List[Any] = []
        self.metadata_entries: List[Any] = []

    def get_dst_subrange(self, size: int, batcher: DrawBatcher) -> Tuple[int, int]:
        # Simple implementation for now
        # Rust has BUFFER_SIZE = 1,048,560
        BUFFER_SIZE = 1_048_560
        
        if batcher.current_dst_entry is None:
            batcher.current_dst_entry = {'index': 0, 'offset': 0}
            
        entry = batcher.current_dst_entry
        if entry['offset'] + size > BUFFER_SIZE:
            entry['index'] += 1
            entry['offset'] = 0
            
        res_index = entry['index']
        res_offset = entry['offset']
        entry['offset'] += size
        
        # Ensure we have enough entries
        while len(self.dst_entries) <= res_index:
            entry = self.device.indirect_validation.draw.acquire_dst_entry()
            self.dst_entries.append(entry)
            
        return (res_index, res_offset)

    def get_metadata_subrange(self, size: int, current_entry: Dict) -> Tuple[int, int]:
        BUFFER_SIZE = 1_048_560
        if current_entry['offset'] + size > BUFFER_SIZE:
            current_entry['index'] += 1
            current_entry['offset'] = 0
            
        res_index = current_entry['index']
        res_offset = current_entry['offset']
        current_entry['offset'] += size
        
        while len(self.metadata_entries) <= res_index:
            entry = self.device.indirect_validation.draw.acquire_metadata_entry()
            self.metadata_entries.append(entry)
            
        return (res_index, res_offset)

    def get_metadata_buffer(self, index: int) -> Any:
        entry = self.metadata_entries[index]
        return entry.get('buffer') if entry else None

    def get_metadata_bind_group(self, index: int) -> Any:
        entry = self.metadata_entries[index]
        return entry.get('bind_group') if entry else None

    def get_dst_buffer(self, index: int) -> Any:
        entry = self.dst_entries[index]
        return entry.get('buffer') if entry else None

    def get_dst_bind_group(self, index: int) -> Any:
        entry = self.dst_entries[index]
        return entry.get('bind_group') if entry else None

    def dispose(self) -> None:
        """Release all acquired entries back to the pool."""
        if hasattr(self.device, 'indirect_validation') and self.device.indirect_validation:
            draw = self.device.indirect_validation.draw
            if hasattr(draw, 'release_indirect_entries'):
                draw.release_indirect_entries(self.dst_entries)
            if hasattr(draw, 'release_metadata_entries'):
                draw.release_metadata_entries(self.metadata_entries)
        self.dst_entries.clear()
        self.metadata_entries.clear()


VALIDATE_DRAW_WGSL = """
override supports_indirect_first_instance: bool;
override write_d3d12_special_constants: bool;

struct MetadataEntry {
    src_offset: u32,
    dst_offset: u32,
    vertex_or_index_limit: u32,
    instance_limit: u32,
}

struct MetadataRange {
    start: u32,
    count: u32,
}
var<immediate> metadata_range: MetadataRange;

@group(0) @binding(0)
var<storage, read> metadata: array<MetadataEntry>;
@group(1) @binding(0)
var<storage, read> src: array<u32>;
@group(2) @binding(0)
var<storage, read_write> dst: array<u32>;

fn is_bit_set(data: u32, index: u32) -> bool {
    return ((data >> index) & 1u) == 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3u) {
    if global_invocation_id.x >= metadata_range.count { return; }

    let metadata = metadata[metadata_range.start + global_invocation_id.x];
    var failed = false;

    let is_indexed = is_bit_set(metadata.src_offset, 31);
    let src_base_offset = ((metadata.src_offset << 2) >> 2);
    let dst_base_offset = ((metadata.dst_offset << 2) >> 2);

    let first_vertex_or_index = src[src_base_offset + 2];
    let vertex_or_index_count = src[src_base_offset + 0];

    {
        let can_overflow = is_bit_set(metadata.dst_offset, 30);
        let sub_overflows = metadata.vertex_or_index_limit < first_vertex_or_index;
        failed |= sub_overflows && !can_overflow;
        let vertex_or_index_limit = metadata.vertex_or_index_limit - first_vertex_or_index;
        failed |= vertex_or_index_limit < vertex_or_index_count;
    }

    let first_instance = src[src_base_offset + 3 + u32(is_indexed)];
    let instance_count = src[src_base_offset + 1];

    {
        let can_overflow = is_bit_set(metadata.dst_offset, 31);
        let sub_overflows = metadata.instance_limit < first_instance;
        failed |= sub_overflows && !can_overflow;
        let instance_limit = metadata.instance_limit - first_instance;
        failed |= instance_limit < instance_count;
    }

    if !supports_indirect_first_instance {
        failed |= first_instance != 0u;
    }

    let dst_offset = select(0u, 3u, write_d3d12_special_constants);
    if failed {
        if write_d3d12_special_constants {
            dst[dst_base_offset + 0] = 0u;
            dst[dst_base_offset + 1] = 0u;
            dst[dst_base_offset + 2] = 0u;
        }
        dst[dst_base_offset + dst_offset + 0] = 0u;
        dst[dst_base_offset + dst_offset + 1] = 0u;
        dst[dst_base_offset + dst_offset + 2] = 0u;
        dst[dst_base_offset + dst_offset + 3] = 0u;
        if (is_indexed) {
            dst[dst_base_offset + dst_offset + 4] = 0u;
        }
    } else {
        if write_d3d12_special_constants {
            dst[dst_base_offset + 0] = src[src_base_offset + 2 + u32(is_indexed)];
            dst[dst_base_offset + 1] = src[src_base_offset + 3 + u32(is_indexed)];
            dst[dst_base_offset + 2] = 0u;
        }
        dst[dst_base_offset + dst_offset + 0] = src[src_base_offset + 0];
        dst[dst_base_offset + dst_offset + 1] = src[src_base_offset + 1];
        dst[dst_base_offset + dst_offset + 2] = src[src_base_offset + 2];
        dst[dst_base_offset + dst_offset + 3] = src[src_base_offset + 3];
        if (is_indexed) {
            dst[dst_base_offset + dst_offset + 4] = src[src_base_offset + 4];
        }
    }
}
"""


class Draw:
    """
    GPU-based draw command validation.
    
    Creates a compute pipeline that validates draw parameters
    against device limits before execution.
    """
    
    def __init__(self, device: Any, features: dict, backend: str):
        """
        Create draw validation pipeline.
        
        Args:
            device: The HAL device.
            features: Device features.
            backend: Backend name.
        """
        self.device = device
        self.features = features
        self.backend = backend
        
        # 1. Create Shader Module
        self.module = device.create_shader_module(
            label="Indirect Draw Validation Shader",
            source=VALIDATE_DRAW_WGSL
        )
        
        # 2. Create Bind Group Layouts
        # BUFFER_SIZE = 1_048_560
        buffer_size = 1_048_560
        
        self.metadata_bind_group_layout = self._create_bind_group_layout(
            device, read_only=True, has_dynamic_offset=False, min_binding_size=buffer_size
        )
        self.src_bind_group_layout = self._create_bind_group_layout(
            device, read_only=True, has_dynamic_offset=True, min_binding_size=16
        )
        self.dst_bind_group_layout = self._create_bind_group_layout(
            device, read_only=False, has_dynamic_offset=False, min_binding_size=buffer_size
        )
        
        # 3. Create Pipeline Layout
        self.pipeline_layout = device.create_pipeline_layout(
            label="Indirect Draw Validation Pipeline Layout",
            bind_group_layouts=[
                self.metadata_bind_group_layout,
                self.src_bind_group_layout,
                self.dst_bind_group_layout
            ],
            immediate_size=8
        )
        
        # 4. Create Compute Pipeline
        supports_indirect_first_instance = features.get("indirect-first-instance", False)
        write_d3d12_special_constants = (backend == "dx12")
        
        self.pipeline = device.create_compute_pipeline(
            label="Indirect Draw Validation Pipeline",
            layout=self.pipeline_layout,
            stage={
                "module": self.module,
                "entry_point": "main",
                "constants": {
                    "supports_indirect_first_instance": supports_indirect_first_instance,
                    "write_d3d12_special_constants": write_d3d12_special_constants
                }
            }
        )
        
        self.free_indirect_entries = []
        self.free_metadata_entries = []

    def _create_bind_group_layout(self, device, read_only, has_dynamic_offset, min_binding_size):
        # COMPUTE = 0x4
        return device.create_bind_group_layout(
            entries=[{
                "binding": 0,
                "visibility": 0x4,
                "ty": {
                    "type": "storage",
                    "read_only": read_only,
                    "has_dynamic_offset": has_dynamic_offset,
                    "min_binding_size": min_binding_size
                }
            }]
        )
    
    def acquire_dst_entry(self) -> Any:
        """Acquire a destination buffer entry from the pool."""
        if self.free_indirect_entries:
            return self.free_indirect_entries.pop()
            
        # INDIRECT | STORAGE_READ_WRITE
        usage = 0x0100 | 0x0080 # Based on wgt::BufferUsages if available
        # But let's use the BufferUses from track if that's what we have
        return self._create_buffer_and_bind_group(
            self.device, usage, self.dst_bind_group_layout
        )

    def acquire_metadata_entry(self) -> Any:
        """Acquire a metadata buffer entry from the pool."""
        if self.free_metadata_entries:
            return self.free_metadata_entries.pop()
            
        # COPY_DST | STORAGE_READ_ONLY
        usage = 0x0008 | 0x0080 
        return self._create_buffer_and_bind_group(
            self.device, usage, self.metadata_bind_group_layout
        )

    def _create_buffer_and_bind_group(
        self,
        device: Any,
        usage: int,
        layout: Any,
    ) -> dict:
        # 1. Create buffer
        # BUFFER_SIZE = 1_048_560
        buffer_size = 1_048_560
        
        buffer = device.create_buffer({
            "label": "Indirect Validation BufferPool Entry",
            "size": buffer_size,
            "usage": usage
        })
        
        # 2. Create bind group
        bind_group = device.create_bind_group({
            "label": "Indirect Validation BufferPool BindGroup",
            "layout": layout,
            "entries": [{
                "binding": 0,
                "resource": {"buffer": buffer, "offset": 0, "size": buffer_size}
            }]
        })
        
        return {"buffer": buffer, "bind_group": bind_group}

    def release_indirect_entries(self, entries: List[Any]) -> None:
        """Release destination buffer entries back to the pool."""
        self.free_indirect_entries.extend(e for e in entries if e is not None)

    def release_metadata_entries(self, entries: List[Any]) -> None:
        """Release metadata buffer entries back to the pool."""
        self.free_metadata_entries.extend(e for e in entries if e is not None)

    def create_src_bind_group(
        self,
        device: Any,
        limits: dict,
        buffer_size: int,
        buffer: Any
    ) -> Optional[Any]:
        """
        Create source bind group for validation.
        
        Returns None if buffer_size is 0.
        
        Args:
            device: The HAL device.
            limits: Device limits.
            buffer_size: Size of the buffer.
            buffer: The buffer to validate.
            
        Returns:
            Bind group or None.
        """
        if buffer_size == 0:
            return None
        
        binding_size = calculate_src_buffer_binding_size(buffer_size, limits)
        if binding_size == 0:
            return None
        
        # Placeholder - would create actual bind group
        return {"buffer": buffer, "size": binding_size}
    
    def inject_validation_pass(
        self,
        device: Any,
        snatch_guard: Any,
        resources: DrawResources,
        temp_resources: List[Any],
        encoder: Any,
        batcher: DrawBatcher
    ) -> None:
        """Inject compute pass for validation."""
        if not batcher.batches:
            return
            
        from .utils import BufferBarriers, BufferBarrierScratch, UniqueIndexScratch, unique_indices
        from ..track import BufferUses
        
        # 1. Prepare metadata bytes
        max_staging_buffer_size = 1 << 26 # ~67MiB
        staging_buffers_data = []
        current_staging_data = bytearray()
        
        for batch in batcher.batches.values():
            data = batch.metadata_bytes()
            if len(current_staging_data) + len(data) > max_staging_buffer_size:
                staging_buffers_data.append(current_staging_data)
                current_staging_data = bytearray(data)
                batch.staging_buffer_index = len(staging_buffers_data)
                batch.staging_buffer_offset = 0
            else:
                batch.staging_buffer_offset = len(current_staging_data)
                batch.staging_buffer_index = len(staging_buffers_data)
                current_staging_data.extend(data)
        
        if current_staging_data:
            staging_buffers_data.append(current_staging_data)
            
        # Create staging buffers in HAL
        # NOTE: In a real implementation, we would use a specialized HAL call
        hal_staging_buffers = []
        for data in staging_buffers_data:
            # Create a buffer with COPY_SRC usage and initialized with data
            buf = device.create_buffer_init(
                label="Indirect Validation Staging",
                contents=data,
                usage=BufferUses.COPY_SRC
            )
            hal_staging_buffers.append(buf)
            temp_resources.append(buf)
            
        # 2. Allocate subranges in metadata buffers
        current_metadata_entry = {'index': 0, 'offset': 0}
        for batch in batcher.batches.values():
            data_len = len(batch.metadata_bytes())
            metadata_resource_index, metadata_buffer_offset = resources.get_metadata_subrange(
                data_len, current_metadata_entry
            )
            batch.metadata_resource_index = metadata_resource_index
            batch.metadata_buffer_offset = metadata_buffer_offset
            
        # 3. Transitions and Copies
        barrier_scratch = BufferBarrierScratch()
        unique_scratch = UniqueIndexScratch()
        
        # Transition metadata buffers to COPY_DST
        metadata_indices = set(batch.metadata_resource_index for batch in batcher.batches.values())
        barriers = []
        for index in metadata_indices:
            barriers.append({
                'buffer': resources.get_metadata_buffer(index),
                'usage': {
                    'from': BufferUses.STORAGE_READ_ONLY,
                    'to': BufferUses.COPY_DST
                }
            })
        BufferBarriers(barrier_scratch).extend(iter(barriers)).encode(encoder)
        
        # Copy data from staging to metadata buffers
        for batch in batcher.batches.values():
            src_buf = hal_staging_buffers[batch.staging_buffer_index]
            dst_buf = resources.get_metadata_buffer(batch.metadata_resource_index)
            
            encoder.copy_buffer_to_buffer(
                src_buf,
                batch.staging_buffer_offset,
                dst_buf,
                batch.metadata_buffer_offset,
                len(batch.metadata_bytes())
            )
            
        # 4. Final transitions before compute
        barriers = []
        # Metadata: COPY_DST -> STORAGE_READ_ONLY
        for index in metadata_indices:
            barriers.append({
                'buffer': resources.get_metadata_buffer(index),
                'usage': {
                    'from': BufferUses.COPY_DST,
                    'to': BufferUses.STORAGE_READ_ONLY
                }
            })
            
        # Destination: INDIRECT -> STORAGE_READ_WRITE
        dst_indices = set(batch.dst_resource_index for batch in batcher.batches.values())
        for index in dst_indices:
            barriers.append({
                'buffer': resources.get_dst_buffer(index),
                'usage': {
                    'from': BufferUses.INDIRECT,
                    'to': BufferUses.STORAGE_READ_WRITE
                }
            })
        BufferBarriers(barrier_scratch).extend(iter(barriers)).encode(encoder)
        
        # 5. Execute compute pass
        # In wgpu-core, this is a raw HAL compute pass
        compute_encoder = encoder.begin_compute_pass(label="Indirect Validation")
        compute_encoder.set_pipeline(self.pipeline)
        
        for batch in batcher.batches.values():
            # Set immediates (metadata start and count)
            # Struct in shader has 2 u32s: metadata_start, metadata_count
            metadata_start = batch.metadata_buffer_offset // 16 # MetadataEntry is 16 bytes
            metadata_count = len(batch.entries)
            
            # Use set_push_constants or equivalent if immediates are used
            if hasattr(compute_encoder, 'set_immediates'):
                compute_encoder.set_immediates(self.pipeline_layout, 0, [metadata_start, metadata_count])
            
            # Bind group 0: Metadata
            compute_encoder.set_bind_group(0, resources.get_metadata_bind_group(batch.metadata_resource_index))
            
            # Bind group 1: Source Indirect Buffer
            # Needs to handle dynamic offset
            src_bind_group = batch.src_buffer.get_indirect_validation_bind_group(snatch_guard)
            compute_encoder.set_bind_group(
                1, 
                src_bind_group, 
                [batch.src_dynamic_offset]
            )
            
            # Bind group 2: Destination Validated Buffer
            compute_encoder.set_bind_group(2, resources.get_dst_bind_group(batch.dst_resource_index))
            
            # Dispatch
            dispatch_x = math.ceil(len(batch.entries) / 64)
            compute_encoder.dispatch(dispatch_x, 1, 1)
            
        compute_encoder.end_pass()
        
        # 6. Post-validation transitions
        # Destination: STORAGE_READ_WRITE -> INDIRECT
        barriers = []
        for index in dst_indices:
            barriers.append({
                'buffer': resources.get_dst_buffer(index),
                'usage': {
                    'from': BufferUses.STORAGE_READ_WRITE,
                    'to': BufferUses.INDIRECT
                }
            })
        BufferBarriers(barrier_scratch).extend(iter(barriers)).encode(encoder)

    def dispose(self, device: Any) -> None:
        """
        Dispose of validation resources.
        
        Args:
            device: The HAL device.
        """
        # In a real implementation we would:
        # 1. Destroy compute pipeline
        # 2. Destroy pipeline layout
        # 3. Destroy bind group layouts
        # 4. Destroy shader module
        # 5. Destroy buffer pools
        
        # For now, just clear the stubs
        self.pipeline = None
        self.pipeline_layout = None
        self.module = None
        self.src_bind_group_layout = None
        self.dst_bind_group_layout = None
        self.metadata_bind_group_layout = None
    
