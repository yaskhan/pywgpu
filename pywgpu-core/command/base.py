from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

@dataclass
class BasePass:
    """A stream of commands for a render or compute pass."""
    label: Optional[str] = None
    error: Optional[Exception] = None
    commands: List[Any] = field(default_factory=list)
    dynamic_offsets: List[int] = field(default_factory=list)
    string_data: bytearray = field(default_factory=bytearray)
    immediates_data: List[int] = field(default_factory=list)

@dataclass
class StateChange:
    """Tracks state changes."""
    current: Optional[Any] = None
    
    def set_and_check_redundant(self, new_value: Any) -> bool:
        if self.current == new_value:
            return True
        self.current = new_value
        return False

@dataclass
class BindGroupStateChange:
    """Tracks bind group state changes."""
    current: List[Optional[Any]] = field(default_factory=lambda: [None] * 8)
    offsets: List[List[int]] = field(default_factory=lambda: [[] for _ in range(8)])
    
    def set_and_check_redundant(self, bind_group_id: Any, index: int, dynamic_offsets_list: List[int], new_offsets: List[int]) -> bool:
        if self.current[index] == bind_group_id and self.offsets[index] == new_offsets:
            return True
        
        self.current[index] = bind_group_id
        self.offsets[index] = new_offsets
        dynamic_offsets_list.extend(new_offsets)
        return False
