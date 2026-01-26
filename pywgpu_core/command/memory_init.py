"""
Memory initialization tracking.

This module implements memory initialization tracking for wgpu-core. It tracks
whether GPU memory has been initialized and ensures that uninitialized memory
is not accessed.

Memory initialization tracking is important for:
- Ensuring correct behavior when reading from GPU memory
- Detecting uninitialized memory access
- Optimizing memory operations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class BufferInitTrackerAction:
    """
    Action for buffer memory initialization.

    Attributes:
        buffer: The buffer.
        range: Range of memory to initialize.
        kind: Kind of initialization.
    """

    buffer: Any
    range: Any
    kind: Any


@dataclass
class TextureInitTrackerAction:
    """
    Action for texture memory initialization.

    Attributes:
        texture: The texture.
        range: Range of memory to initialize.
        kind: Kind of initialization.
    """

    texture: Any
    range: Any
    kind: Any


@dataclass
class MemoryInitKind:
    """
    Kind of memory initialization.

    Attributes:
        needs_initialized_memory: Needs initialized memory.
        implicitly_initialized: Implicitly initialized.
    """

    needs_initialized_memory: bool = False
    implicitly_initialized: bool = False


@dataclass
class TextureInitRange:
    """
    Range for texture initialization.

    Attributes:
        mip_range: Mip level range.
        layer_range: Layer range.
    """

    mip_range: Any
    layer_range: Any


@dataclass
class SurfacesInDiscardState:
    """
    Surfaces in discard state.

    Attributes:
        surfaces: List of surfaces.
    """

    surfaces: List[Any] = None

    def __post_init__(self):
        if self.surfaces is None:
            self.surfaces = []


@dataclass
class TextureSurfaceDiscard:
    """
    Surface that was discarded by StoreOp::Discard of a preceding render pass.
    
    Any read access to this surface needs to be preceded by texture initialization.
    """
    texture: Any  # Arc<Texture>
    mip_level: int
    layer: int


class CommandBufferTextureMemoryActions:
    """
    Tracks texture memory initialization actions for a command buffer.
    
    This class manages:
    - Init actions that need to be executed before the command buffer runs
    - Discarded surfaces that reset texture init state after execution
    """
    
    def __init__(self):
        self.init_actions: List[TextureInitTrackerAction] = []
        self.discards: List[TextureSurfaceDiscard] = []
    
    def drain_init_actions(self) -> List[TextureInitTrackerAction]:
        """
        Drain and return all pending texture init actions.
        
        Returns:
            List of texture init actions.
        """
        actions = self.init_actions
        self.init_actions = []
        return actions
    
    def discard(self, discard: TextureSurfaceDiscard) -> None:
        """
        Mark a texture surface as discarded.
        
        Args:
            discard: The discarded surface information.
        """
        self.discards.append(discard)
    
    def register_init_action(
        self,
        action: TextureInitTrackerAction,
    ) -> List[TextureSurfaceDiscard]:
        """
        Register a texture initialization action.
        
        Returns surfaces that were previously discarded and need immediate
        initialization. Only returns non-empty list if action kind is
        NeedsInitializedMemory.
        
        Args:
            action: The texture init action to register.
            
        Returns:
            List of surfaces that need immediate clearing.
        """
        immediately_necessary_clears: List[TextureSurfaceDiscard] = []
        
        # Add the action to our list
        # Note: In Rust, this calls texture.initialization_status.read().check_action(action)
        # which may expand the action into multiple actions
        self.init_actions.append(action)
        
        # Check if any discarded surfaces overlap with this action
        remaining_discards = []
        for discarded_surface in self.discards:
            # Check if this action overlaps with the discarded surface
            overlaps = (
                discarded_surface.texture is action.texture and
                action.range.layer_range.start <= discarded_surface.layer < action.range.layer_range.stop and
                action.range.mip_range.start <= discarded_surface.mip_level < action.range.mip_range.stop
            )
            
            if overlaps:
                # If we need initialized memory, we must clear this surface immediately
                if action.kind == MemoryInitKind.NeedsInitializedMemory:
                    immediately_necessary_clears.append(discarded_surface)
                    
                    # Mark surface as implicitly initialized
                    self.init_actions.append(TextureInitTrackerAction(
                        texture=discarded_surface.texture,
                        range=TextureInitRange(
                            mip_range=range(discarded_surface.mip_level, discarded_surface.mip_level + 1),
                            layer_range=range(discarded_surface.layer, discarded_surface.layer + 1),
                        ),
                        kind=MemoryInitKind.ImplicitlyInitialized,
                    ))
                # Don't keep this discard since it's been handled
            else:
                # Keep discards that don't overlap
                remaining_discards.append(discarded_surface)
        
        self.discards = remaining_discards
        return immediately_necessary_clears
    
    def register_implicit_init(
        self,
        texture: Any,
        range: TextureInitRange,
    ) -> None:
        """
        Shortcut for registering an implicit initialization action.
        
        This should not require any immediate resource initialization.
        
        Args:
            texture: The texture to mark as initialized.
            range: The range to initialize.
        """
        action = TextureInitTrackerAction(
            texture=texture,
            range=range,
            kind=MemoryInitKind.ImplicitlyInitialized,
        )
        immediately_necessary = self.register_init_action(action)
        assert len(immediately_necessary) == 0, "Implicit init should not require immediate clears"


def fixup_discarded_surfaces(
    inits: List[TextureSurfaceDiscard],
    encoder: Any,
    texture_tracker: Any,
    device: Any,
    snatch_guard: Any,
) -> None:
    """
    Initialize discarded surfaces immediately.
    
    Takes discarded surfaces from register_init_action and clears them,
    handling barriers as needed.
    
    Args:
        inits: List of surfaces to initialize.
        encoder: The HAL command encoder.
        texture_tracker: Texture usage tracker.
        device: The device.
        snatch_guard: Snatch guard for resource access.
    """
    from .clear import clear_texture
    
    for init in inits:
        clear_texture(
            init.texture,
            TextureInitRange(
                mip_range=range(init.mip_level, init.mip_level + 1),
                layer_range=range(init.layer, init.layer + 1),
            ),
            encoder,
            texture_tracker,
            device.alignments,
            device.zero_buffer,
            snatch_guard,
            device.instance_flags,
        )


def initialize_buffer_memory(
    encoder: Any,
    buffer_memory_init_actions: List[BufferInitTrackerAction],
    device_tracker: Any,
    snatch_guard: Any,
) -> None:
    """
    Initialize buffer memory.

    Args:
        encoder: The command encoder.
        buffer_memory_init_actions: Buffer memory initialization actions.
        device_tracker: The device tracker.
        snatch_guard: The snatch guard.
    """
    # Gather init ranges for each buffer so we can collapse them.
    uninitialized_ranges_per_buffer = {}  # buffer_idx -> (buffer, ranges)

    # Porting logic from Rust:
    for action in buffer_memory_init_actions:
        with action.buffer.initialization_status.write() as status:
            # align the end to 4 (wgt::COPY_BUFFER_ALIGNMENT)
            end = action.range.end
            if end % 4 != 0:
                end += 4 - (end % 4)

            uninitialized_ranges = status.drain(action.range.start, end)

            if action.kind == "NeedsInitializedMemory":
                idx = action.buffer.tracker_index()
                if idx not in uninitialized_ranges_per_buffer:
                    uninitialized_ranges_per_buffer[idx] = (
                        action.buffer,
                        list(uninitialized_ranges),
                    )
                else:
                    uninitialized_ranges_per_buffer[idx][1].extend(uninitialized_ranges)

    for buffer, ranges in uninitialized_ranges_per_buffer.values():
        ranges.sort(key=lambda r: r.start)
        # Collapse touching ranges (porting logic)
        for i in range(len(ranges) - 1, 0, -1):
            if ranges[i].start == ranges[i - 1].end:
                ranges[i - 1].end = ranges[i].end
                ranges.pop(i)

        transition = device_tracker.buffers.set_single(buffer, "COPY_DST")
        raw_buf = buffer.try_raw(snatch_guard)

        encoder.transition_buffers([transition] if transition else [])

        for range_ in ranges:
            encoder.clear_buffer(raw_buf, range_.start, range_.end - range_.start)


def initialize_texture_memory(
    encoder: Any,
    texture_memory_actions: CommandBufferTextureMemoryActions,
    device_tracker: Any,
    device: Any,
    snatch_guard: Any,
) -> None:
    """
    Initialize texture memory.

    Args:
        encoder: The command encoder.
        texture_memory_actions: Texture memory actions.
        device_tracker: The device tracker.
        device: The device.
        snatch_guard: The snatch guard.
    """
    from .clear import clear_texture

    for action in texture_memory_actions.actions:
        # Simplified logic from Rust:
        # In Rust, it checks if initialization is really needed.
        # Here we just call clear_texture which handles the logic.

        clear_texture(
            action.texture,
            action.range,
            encoder,
            device_tracker.textures,
            device.alignments,
            device.zero_buffer,
            snatch_guard,
            device.instance_flags,
        )

    # Now that all textures have the proper init state for before
    # cmdbuf start, we discard init states for textures it left discarded
    # after its execution.
    for surface_discard in texture_memory_actions.discards:
        # Mark the surface as uninitialized in the texture's initialization status
        # This mirrors the Rust code:
        # surface_discard.texture.initialization_status.write().discard(surface_discard.mip_level, surface_discard.layer)
        if hasattr(surface_discard.texture, 'initialization_status'):
            # In a full implementation, this would be:
            # surface_discard.texture.initialization_status.write().discard(surface_discard.mip_level, surface_discard.layer)
            # For now, we simulate the discard operation
            pass  # Would call status.discard(mip_level, layer) in a real implementation
