from dataclasses import dataclass
from typing import List, Optional, Union
from naga import ir, UniqueArena
from naga.proc import TypeResolution

@dataclass
class Rule:
    """A single type rule."""
    arguments: List[TypeResolution]
    conclusion: 'Conclusion'

class Conclusion:
    """
    The result type of a Rule.
    
    A Conclusion value represents the return type of some operation
    in the builtin function database.
    """
    # TODO: Implement Conclusion (Value or Predeclared)
    # In Rust it's an enum:
    # Value(ir::TypeInner)
    # Predeclared(ir::PredeclaredType)
    pass

class MissingSpecialType(Exception):
    """Special type is not registered within the module."""
    def __init__(self):
        super().__init__("Special type is not registered within the module")
