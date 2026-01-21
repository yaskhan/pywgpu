"""
Workarounds for platform bugs and limitations in switches and loops.

In these docs, we use CamelCase for Naga IR concepts, and ordinary
code formatting for HLSL or GLSL concepts.

## Avoiding `continue` within `switch`

As described in https://github.com/gfx-rs/wgpu/issues/4485, the FXC HLSL
compiler doesn't allow `continue` statements within `switch` statements, but
Naga IR does. We work around this by introducing synthetic boolean local
variables and branches.

Specifically:

- We generate code for Continue statements within SwitchCases that
  sets an introduced bool local to true and does a break, jumping to
  immediately after the generated switch.

- When generating code for a Switch statement, we conservatively assume
  it might contain such a Continue statement, so:

  - If it's the outermost such Switch within a Loop, we declare the
    bool local ahead of the switch, initialized to false. Immediately
    after the switch, we check the local and do a continue if it's set.

  - If the Switch is nested within other Switches, then after the
    generated switch, we check the local (which we know was declared
    before the surrounding switch) and do a break if it's set.

  - As an optimization, we only generate the check of the local if a
    Continue statement is encountered within the Switch.

# Avoiding single body switch statements

As described in https://github.com/gfx-rs/wgpu/issues/4514, some language
front ends miscompile switch statements where all cases branch to the same
body. Our HLSL and GLSL backends render Switch statements with a single
SwitchCase as do {} while(false); loops.

However, this rewriting introduces a new loop that could "capture"
continue statements in its body. To avoid doing so, we apply the
Continue-to-break transformation described above.
"""

from __future__ import annotations

from typing import Optional, List
from enum import Enum
from dataclasses import dataclass

from ..proc import Namer


class Nesting(Enum):
    """
    A summary of the code surrounding a statement.
    
    Attributes:
        LOOP: Currently nested in at least one Loop statement.
            Continue should apply to the innermost loop.
        SWITCH: Currently nested in at least one Switch that may need to forward
            Continues. This includes Switches rendered as do {} while(false) loops.
    """
    LOOP = "loop"
    SWITCH = "switch"


@dataclass
class ContinueCtx:
    """
    Context for tracking whether continue statements need forwarding.
    
    Attributes:
        nesting: Stack of nesting states
        switch_continues_var: Name of the boolean variable for forwarding continues
    """
    nesting: List[Nesting]
    switch_continues_var: Optional[str] = None
    
    def __init__(self) -> None:
        """Initialize a new ContinueCtx."""
        self.nesting = []
        self.switch_continues_var = None
    
    def is_in_loop(self) -> bool:
        """Check if we're currently in a loop."""
        return Nesting.LOOP in self.nesting
    
    def is_in_switch(self) -> bool:
        """Check if we're currently in a switch that needs continue forwarding."""
        return Nesting.SWITCH in self.nesting
    
    def enter_loop(self) -> None:
        """Enter a loop context."""
        self.nesting.append(Nesting.LOOP)
    
    def exit_loop(self) -> None:
        """Exit a loop context."""
        if self.nesting and self.nesting[-1] == Nesting.LOOP:
            self.nesting.pop()
    
    def enter_switch(self, namer: Namer) -> Optional[str]:
        """
        Enter a switch context.
        
        Args:
            namer: Namer for generating variable names
            
        Returns:
            Name of the continue forwarding variable if needed
        """
        if self.is_in_loop():
            self.nesting.append(Nesting.SWITCH)
            if self.switch_continues_var is None:
                self.switch_continues_var = namer.call("switch_continue")
            return self.switch_continues_var
        return None
    
    def exit_switch(self) -> None:
        """Exit a switch context."""
        if self.nesting and self.nesting[-1] == Nesting.SWITCH:
            self.nesting.pop()


__all__ = [
    "Nesting",
    "ContinueCtx",
]
