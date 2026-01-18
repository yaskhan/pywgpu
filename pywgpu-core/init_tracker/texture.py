"""Texture initialization tracking."""

from typing import Optional, List
from dataclasses import dataclass

from . import InitTracker, MemoryInitKind, Range


@dataclass
class TextureInitRange:
    """
    Range of texture mip levels and layers.
    
    Attributes:
        mip_range: Range of mip levels.
        layer_range: Range of array layers (not volume slices).
    """
    mip_range: Range[int]
    layer_range: Range[int]


@dataclass
class TextureInitTrackerAction:
    """
    Action to track texture initialization.
    
    Attributes:
        texture: The texture being tracked.
        range: The mip/layer range affected.
        kind: Type of initialization action.
    """
    texture: any  # Arc<Texture> in Rust
    range: TextureInitRange
    kind: MemoryInitKind


def has_copy_partial_init_tracker_coverage(
    copy_size: dict,
    mip_level: int,
    desc: dict
) -> bool:
    """
    Check if a copy operation doesn't fully cover texture init tracking granularity.
    
    Returns true if the copy operation is partial and the target texture
    needs to be ensured to be initialized first.
    
    Args:
        copy_size: The extent being copied (width, height, depth_or_array_layers).
        mip_level: The mip level being copied to.
        desc: The texture descriptor.
        
    Returns:
        True if the copy is partial and requires prior initialization.
    """
    # Calculate target size at mip level
    width = desc.get('width', 1)
    height = desc.get('height', 1)
    depth = desc.get('depth_or_array_layers', 1)
    
    # Calculate mip level size
    target_width = max(1, width >> mip_level)
    target_height = max(1, height >> mip_level)
    target_depth = max(1, depth >> mip_level) if desc.get('dimension') == '3d' else depth
    
    # Check if copy covers entire mip level
    copy_width = copy_size.get('width', 0)
    copy_height = copy_size.get('height', 0)
    copy_depth = copy_size.get('depth_or_array_layers', 0)
    
    is_partial = (
        copy_width != target_width or
        copy_height != target_height or
        (desc.get('dimension') == '3d' and copy_depth != target_depth)
    )
    
    return is_partial


class TextureLayerInitTracker(InitTracker[int]):
    """Initialization tracker for a single texture mip level across layers."""
    pass


class TextureInitTracker:
    """
    Initialization tracker for textures.
    
    Tracks initialization at mip-level per layer granularity.
    
    Attributes:
        mips: List of init trackers, one per mip level.
    """
    
    def __init__(self, mip_level_count: int, depth_or_array_layers: int):
        """
        Create a new texture init tracker.
        
        Args:
            mip_level_count: Number of mip levels.
            depth_or_array_layers: Number of array layers (or depth slices).
        """
        self.mips: List[TextureLayerInitTracker] = [
            TextureLayerInitTracker(depth_or_array_layers)
            for _ in range(mip_level_count)
        ]
    
    def check_action(self, action: TextureInitTrackerAction) -> Optional[TextureInitTrackerAction]:
        """
        Check if an action has/requires any effect on initialization status.
        
        Shrinks the action's range if possible.
        
        Args:
            action: The action to check.
            
        Returns:
            A potentially shrunk action, or None if no effect needed.
        """
        mip_range_start = float('inf')
        mip_range_end = float('-inf')
        layer_range_start = float('inf')
        layer_range_end = float('-inf')
        
        # Check each mip level in the action's range
        for i in range(action.range.mip_range.start, action.range.mip_range.end):
            if i >= len(self.mips):
                break
            
            mip_tracker = self.mips[i]
            uninitialized_layer_range = mip_tracker.check(action.range.layer_range)
            
            if uninitialized_layer_range is not None:
                mip_range_start = min(mip_range_start, i)
                mip_range_end = max(mip_range_end, i + 1)
                layer_range_start = min(layer_range_start, uninitialized_layer_range.start)
                layer_range_end = max(layer_range_end, uninitialized_layer_range.end)
        
        # Check if we found any uninitialized ranges
        if (mip_range_start < mip_range_end and 
            layer_range_start < layer_range_end):
            return TextureInitTrackerAction(
                texture=action.texture,
                range=TextureInitRange(
                    mip_range=Range(int(mip_range_start), int(mip_range_end)),
                    layer_range=Range(int(layer_range_start), int(layer_range_end))
                ),
                kind=action.kind
            )
        
        return None
    
    def discard(self, mip_level: int, layer: int) -> None:
        """
        Mark a specific mip level and layer as uninitialized.
        
        Args:
            mip_level: The mip level to discard.
            layer: The layer to discard.
        """
        if mip_level < len(self.mips):
            self.mips[mip_level].discard(layer)
