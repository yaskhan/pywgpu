"""
Utility for normalizing GPU timestamp queries to have a consistent 1GHz period.

This uses a compute shader to do the normalization, so the timestamps exist
in their correct format on the GPU, as is required by the WebGPU specification.

## Algorithm

The fundamental operation is multiplying a u64 timestamp by an f32 value.
We have neither f64s nor u64s in shaders, so we need to do something more
complicated.

We first decompose the f32 into a u32 fraction where the denominator is a
power of two. We do the computation with f64 for ease of computation, as
those can store u32s losslessly.

Because the denominator is a power of two, this means the shader can evaluate
this divide by using a shift. Additionally, we always choose the largest
denominator we can, so that the fraction is as precise as possible.

To evaluate this function, we have two helper operations (both in common.wgsl):

1. `u64_mul_u32` multiplies a u64 by a u32 and returns a u96.
2. `shift_right_u96` shifts a u96 right by a given amount, returning a u96.

We then multiply the timestamp by the numerator, and shift it right by the
denominator. This gives us the normalized timestamp.
"""

from typing import Optional, Tuple
import math

from .common import COMMON_WGSL
from .shader import TIMESTAMP_NORMALIZATION_WGSL


class TimestampNormalizerInitError(Exception):
    """Error initializing timestamp normalizer."""
    pass


class TimestampNormalizationBindGroup:
    """
    Bind group for timestamp normalization.
    
    Attributes:
        raw: The HAL bind group, or None if normalization is disabled.
    """
    
    def __init__(self, raw: Optional[any] = None):
        """
        Create a timestamp normalization bind group.
        
        Args:
            raw: The HAL bind group, or None.
        """
        self.raw = raw
    
    def dispose(self, device: any) -> None:
        """
        Dispose of the bind group.
        
        Args:
            device: The HAL device.
        """
        if self.raw is not None:
            try:
                device.destroy_bind_group(self.raw)
            except Exception:
                pass


class TimestampNormalizer:
    """
    Normalizes GPU timestamps to have a consistent 1GHz period.
    
    If the device cannot support automatic timestamp normalization,
    this will be a no-op normalizer.
    
    Attributes:
        enabled: Whether normalization is enabled.
        pipeline: The compute pipeline for normalization.
        pipeline_layout: The pipeline layout.
        bind_group_layout: The bind group layout.
    """
    
    def __init__(self, device: any, timestamp_period: float):
        """
        Create a new timestamp normalizer.
        
        Args:
            device: The device.
            timestamp_period: The timestamp period in nanoseconds.
            
        Raises:
            TimestampNormalizerInitError: If initialization fails.
        """
        self.enabled = False
        self.pipeline = None
        self.pipeline_layout = None
        self.bind_group_layout = None
        
        # Check if normalization is needed
        if timestamp_period == 1.0:
            # Period is already 1ns, no normalization needed
            return
        
        # Check if automatic normalization is supported
        # (This would check device flags in real implementation)
        
        # Placeholder - full implementation would:
        # 1. Create bind group layout
        # 2. Compile WGSL shader (common.wgsl + timestamp_normalization.wgsl)
        # 3. Create pipeline layout
        # 4. Create compute pipeline with constants
        
        self.enabled = True
    
    def create_normalization_bind_group(
        self,
        device: any,
        buffer: any,
        buffer_label: Optional[str],
        buffer_size: int,
        buffer_usages: int
    ) -> TimestampNormalizationBindGroup:
        """
        Create a bind group for normalizing timestamps in buffer.
        
        Args:
            device: The device.
            buffer: The HAL buffer containing timestamps.
            buffer_label: Optional label for debugging.
            buffer_size: Size of the buffer in bytes.
            buffer_usages: Buffer usage flags.
            
        Returns:
            A bind group for normalization.
        """
        if not self.enabled:
            return TimestampNormalizationBindGroup(None)
        
        # Placeholder - would create actual bind group
        return TimestampNormalizationBindGroup(None)
    
    def normalize(
        self,
        snatch_guard: any,
        encoder: any,
        tracker: any,
        bind_group: TimestampNormalizationBindGroup,
        buffer: any,
        buffer_offset_bytes: int,
        total_timestamps: int
    ) -> None:
        """
        Normalize timestamps in a buffer.
        
        This dispatches a compute shader to normalize the timestamps.
        
        Args:
            snatch_guard: Snatch guard for buffer access.
            encoder: Command encoder.
            tracker: Buffer tracker.
            bind_group: The normalization bind group.
            buffer: The buffer containing timestamps.
            buffer_offset_bytes: Offset into the buffer in bytes.
            total_timestamps: Number of timestamps to normalize.
        """
        if not self.enabled or bind_group.raw is None:
            return
        
        # Calculate workgroups (64 timestamps per workgroup)
        needed_workgroups = (total_timestamps + 63) // 64
        buffer_offset_timestamps = buffer_offset_bytes // 8
        
        # Placeholder - would:
        # 1. Transition buffer to STORAGE_READ_WRITE
        # 2. Begin compute pass
        # 3. Set pipeline and bind group
        # 4. Set immediate data (offset, count)
        # 5. Dispatch compute shader
        # 6. End compute pass
    
    def dispose(self, device: any) -> None:
        """
        Dispose of normalization resources.
        
        Args:
            device: The HAL device.
        """
        if not self.enabled:
            return
        
        try:
            if self.pipeline:
                device.destroy_compute_pipeline(self.pipeline)
            if self.pipeline_layout:
                device.destroy_pipeline_layout(self.pipeline_layout)
            if self.bind_group_layout:
                device.destroy_bind_group_layout(self.bind_group_layout)
        except Exception:
            pass


def compute_timestamp_period(input_period: float) -> Tuple[int, int]:
    """
    Compute the multiplier and shift for timestamp normalization.
    
    Converts a floating-point timestamp period into a fraction with
    a power-of-two denominator for efficient GPU computation.
    
    Args:
        input_period: The timestamp period in nanoseconds.
        
    Returns:
        Tuple of (multiplier, shift) where:
        - multiplier: u32 numerator
        - shift: u32 denominator as power of 2 (actual denominator is 1 << shift)
    """
    # Calculate power of 2 for denominator
    pow2 = math.ceil(math.log2(input_period))
    clamped_pow2 = max(-32, min(32, pow2))
    shift = 32 - abs(clamped_pow2)
    
    # Calculate multiplier
    denominator = float(1 << shift)
    multiplier = round(input_period * denominator)
    
    # Clamp to u32 range
    multiplier = max(0, min(0xFFFFFFFF, multiplier))
    
    return (multiplier, shift)


# Export constants
TIMESTAMP_NORMALIZATION_BUFFER_USES = 0x0080  # STORAGE_READ_WRITE

__all__ = [
    'TimestampNormalizer',
    'TimestampNormalizationBindGroup',
    'TimestampNormalizerInitError',
    'compute_timestamp_period',
    'TIMESTAMP_NORMALIZATION_BUFFER_USES',
    'COMMON_WGSL',
    'TIMESTAMP_NORMALIZATION_WGSL',
]
