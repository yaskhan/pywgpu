from __future__ import annotations
from dataclasses import dataclass
from enum import IntFlag
from typing import Any, List, Optional, Iterator, Union, Dict, Tuple

from .metadata import ResourceMetadata

class TextureUses(IntFlag):
    """Flags representing the ways a texture can be used."""
    UNINITIALIZED = 1 << 0
    PRESENT = 1 << 1
    COPY_SRC = 1 << 2
    COPY_DST = 1 << 3
    RESOURCE = 1 << 4
    COLOR_TARGET = 1 << 5
    DEPTH_STENCIL_READ = 1 << 6
    DEPTH_STENCIL_WRITE = 1 << 7
    STORAGE_READ_ONLY = 1 << 8
    STORAGE_WRITE_ONLY = 1 << 9
    STORAGE_READ_WRITE = 1 << 10
    STORAGE_ATOMIC = 1 << 11
    TRANSIENT = 1 << 12

    INCLUSIVE = COPY_SRC | RESOURCE | DEPTH_STENCIL_READ | STORAGE_READ_ONLY
    EXCLUSIVE = (
        COPY_DST | COLOR_TARGET | DEPTH_STENCIL_WRITE | 
        STORAGE_WRITE_ONLY | STORAGE_READ_WRITE | STORAGE_ATOMIC | PRESENT
    )
    ORDERED = INCLUSIVE | COLOR_TARGET | DEPTH_STENCIL_WRITE | STORAGE_READ_ONLY

    COMPLEX = 1 << 15  # Special flag signifying per-subresource state

@dataclass(frozen=True)
class TextureSelector:
    """Selects a subresource range of a texture."""
    mips: range
    layers: range

class RangedStates:
    """
    Structure that keeps track of a I -> T mapping,
    optimized for a case where keys of the same values
    are often grouped together linearly.
    """
    def __init__(self, limit: int, default_value: TextureUses):
        self.ranges: List[List[Union[range, TextureUses]]] = [[range(0, limit), default_value]]

    def isolate(self, target: range, default_value: TextureUses) -> List[int]:
        """
        Split the storage ranges in such a way that there is a linear subset of
        them occupying exactly `target` range. Returns indices of matching ranges.
        """
        # 1. Ensure target.start is a boundary
        for i in range(len(self.ranges)):
            r, val = self.ranges[i]
            if r.start < target.start < r.stop:
                # Split this range
                self.ranges[i] = [range(r.start, target.start), val]
                self.ranges.insert(i + 1, [range(target.start, r.stop), val])
                break
        
        # 2. Ensure target.stop is a boundary
        for i in range(len(self.ranges)):
            r, val = self.ranges[i]
            if r.start < target.stop < r.stop:
                # Split this range
                self.ranges[i] = [range(r.start, target.stop), val]
                self.ranges.insert(i + 1, [range(target.stop, r.stop), val])
                break
                
        # 3. Find indices
        target_indices = []
        for i in range(len(self.ranges)):
            r, val = self.ranges[i]
            if r.start >= target.start and r.stop <= target.stop:
                target_indices.append(i)
        
        return target_indices

    def coalesce(self):
        """Merge neighboring ranges with the same value."""
        if not self.ranges:
            return
        
        new_ranges = []
        current_pair = self.ranges[0]
        
        for next_pair in self.ranges[1:]:
            if current_pair[0].stop == next_pair[0].start and current_pair[1] == next_pair[1]:
                current_pair[0] = range(current_pair[0].start, next_pair[0].stop)
            else:
                new_ranges.append(current_pair)
                current_pair = next_pair
        
        new_ranges.append(current_pair)
        self.ranges = new_ranges

    def to_selector_state_iter(self, mip: int) -> Iterator[Tuple[TextureSelector, TextureUses]]:
        for r, val in self.ranges:
            yield (TextureSelector(mips=range(mip, mip+1), layers=r), val)

class ComplexTextureState:
    """Tracks state for every subresource of a texture."""
    def __init__(self, mip_count: int, layer_count: int):
        self.mips = [RangedStates(layer_count, TextureUses.UNINITIALIZED) for _ in range(mip_count)]

    def to_selector_state_iter(self) -> Iterator[Tuple[TextureSelector, TextureUses]]:
        for mip, rs in enumerate(self.mips):
            yield from rs.to_selector_state_iter(mip)

class TextureStateSet:
    """Stores all texture state within a single usage scope or tracker."""
    def __init__(self):
        self.simple: List[TextureUses] = []
        self.complex: Dict[int, ComplexTextureState] = {}

    def set_size(self, size: int):
        if size > len(self.simple):
            self.simple.extend([TextureUses.UNINITIALIZED] * (size - len(self.simple)))

    def get_states(self, index: int) -> Union[TextureUses, ComplexTextureState]:
        state = self.simple[index]
        if state == TextureUses.COMPLEX:
            return self.complex[index]
        return state

class TextureUsageScope:
    """Tracks texture usage within a specific scope."""
    def __init__(self) -> None:
        self.set = TextureStateSet()
        self.metadata: ResourceMetadata[Any] = ResourceMetadata()

    def merge_single(self, texture: Any, selector: Optional[TextureSelector], new_state: TextureUses) -> None:
        index = texture.tracker_index()
        self.set.set_size(index + 1)
        self.metadata.insert(index, texture)
        
        if selector is None:
            # Full texture
            current = self.set.simple[index]
            if current == TextureUses.COMPLEX:
                complex_state = self.set.complex[index]
                for rs in complex_state.mips:
                    for i in range(len(rs.ranges)):
                        rs.ranges[i][1] |= new_state
                    rs.coalesce()
            else:
                self.set.simple[index] |= new_state
        else:
            # Subresource range
            if self.set.simple[index] != TextureUses.COMPLEX:
                old_simple = self.set.simple[index]
                # Default to 16/1 if not specified
                mip_count = getattr(texture, 'mip_level_count', 16)
                layer_count = getattr(texture, 'array_layer_count', 1)
                
                complex_state = ComplexTextureState(mip_count, layer_count)
                if old_simple != TextureUses.UNINITIALIZED:
                    for rs in complex_state.mips:
                        rs.ranges[0][1] = old_simple
                
                self.set.simple[index] = TextureUses.COMPLEX
                self.set.complex[index] = complex_state

            complex_state = self.set.complex[index]
            for m in selector.mips:
                if m < len(complex_state.mips):
                    rs = complex_state.mips[m]
                    target_indices = rs.isolate(selector.layers, TextureUses.UNINITIALIZED)
                    for idx in target_indices:
                        rs.ranges[idx][1] |= new_state
                    rs.coalesce()

class TextureTracker:
    """Tracks texture state across commands."""
    def __init__(self) -> None:
        self.start_set = TextureStateSet()
        self.end_set = TextureStateSet()
        self.metadata: ResourceMetadata[Any] = ResourceMetadata()
        self.temp: List[Any] = [] # List[PendingTransition]

    def set_size(self, size: int) -> None:
        self.start_set.set_size(size)
        self.end_set.set_size(size)
        self.metadata.set_size(size)

    def set_single(self, texture: Any, selector: TextureSelector, state: TextureUses) -> List[Any]:
        index = texture.tracker_index()
        self.set_size(index + 1)
        self.metadata.insert(index, texture)
        
        transitions = []
        
        # Ensure complex state
        if self.end_set.simple[index] != TextureUses.COMPLEX:
            old_simple = self.end_set.simple[index]
            mip_count = getattr(texture, 'mip_level_count', 16)
            layer_count = getattr(texture, 'array_layer_count', 1)
            complex_state = ComplexTextureState(mip_count, layer_count)
            for rs in complex_state.mips:
                rs.ranges[0][1] = old_simple
            self.end_set.simple[index] = TextureUses.COMPLEX
            self.end_set.complex[index] = complex_state

        complex_state = self.end_set.complex[index]
        from . import PendingTransition, StateTransition
        
        for m in selector.mips:
            if m < len(complex_state.mips):
                rs = complex_state.mips[m]
                target_indices = rs.isolate(selector.layers, TextureUses.UNINITIALIZED)
                for idx in target_indices:
                    old_state = rs.ranges[idx][1]
                    if old_state != state:
                        t = PendingTransition(
                            id=index,
                            selector=TextureSelector(mips=range(m, m+1), layers=rs.ranges[idx][0]),
                            usage=StateTransition(from_state=old_state, to_state=state)
                        )
                        transitions.append(t)
                        self.temp.append(t)
                        rs.ranges[idx][1] = state
                rs.coalesce()
        
        return transitions

    def drain_transitions(self) -> Iterator[Any]:
        """Yields and clears all pending transitions."""
        yield from self.temp
        self.temp.clear()
