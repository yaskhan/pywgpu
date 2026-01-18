"""Draw command validation."""

from typing import Any, Optional


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
        
        # Placeholder - full implementation would create:
        # - Shader module with validation logic (validate_draw.wgsl)
        # - Bind group layouts (metadata, src, dst)
        # - Pipeline layout
        # - Compute pipeline
        # - Buffer pools for indirect and metadata
        
        self.module = None
        self.metadata_bind_group_layout = None
        self.src_bind_group_layout = None
        self.dst_bind_group_layout = None
        self.pipeline_layout = None
        self.pipeline = None
        self.free_indirect_entries = []
        self.free_metadata_entries = []
    
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
        
        binding_size = self._calculate_src_buffer_binding_size(buffer_size, limits)
        if binding_size == 0:
            return None
        
        # Placeholder - would create actual bind group
        return {"buffer": buffer, "size": binding_size}
    
    def dispose(self, device: Any) -> None:
        """
        Dispose of validation resources.
        
        Args:
            device: The HAL device.
        """
        # Cleanup would go here
        pass
    
    @staticmethod
    def _calculate_src_buffer_binding_size(buffer_size: int, limits: dict) -> int:
        """
        Calculate binding size for source buffer.
        
        Returns the largest binding size that when combined with dynamic
        offsets can address the whole buffer.
        
        Args:
            buffer_size: Size of the buffer.
            limits: Device limits.
            
        Returns:
            Calculated binding size.
        """
        max_storage_buffer_binding_size = limits.get('max_storage_buffer_binding_size', 2**30)
        min_storage_buffer_offset_alignment = limits.get('min_storage_buffer_offset_alignment', 256)
        
        if buffer_size <= max_storage_buffer_binding_size:
            return buffer_size
        
        buffer_rem = buffer_size % min_storage_buffer_offset_alignment
        binding_rem = max_storage_buffer_binding_size % min_storage_buffer_offset_alignment
        
        # Can the buffer remainder fit in the binding remainder?
        if buffer_rem <= binding_rem:
            return max_storage_buffer_binding_size - binding_rem + buffer_rem
        else:
            return (max_storage_buffer_binding_size - binding_rem - 
                    min_storage_buffer_offset_alignment + buffer_rem)
