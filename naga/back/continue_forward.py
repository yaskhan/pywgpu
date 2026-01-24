"""Workarounds for platform bugs and limitations in switches and loops.

This is a direct translation of `wgpu-trunk/naga/src/back/continue_forward.rs`.

In these docs, we use CamelCase for Naga IR concepts, and ordinary code
formatting for HLSL or GLSL concepts.

## Avoiding `continue` within `switch`

As described in https://github.com/gfx-rs/wgpu/issues/4485, the FXC HLSL
compiler doesn't allow `continue` statements within `switch` statements, but
Naga IR does. We work around this by introducing synthetic boolean local
variables and branches.

Specifically:

- We generate code for Continue statements within SwitchCases that sets an
  introduced bool local to true and does a break, jumping to immediately after
  the generated switch.

- When generating code for a Switch statement, we conservatively assume it
  might contain such a Continue statement, so:

  - If it's the outermost such Switch within a Loop, we declare the bool local
    ahead of the switch, initialized to false. Immediately after the switch, we
    check the local and do a continue if it's set.

  - If the Switch is nested within other Switches, then after the generated
    switch, we check the local (which we know was declared before the
    surrounding switch) and do a break if it's set.

  - As an optimization, we only generate the check of the local if a Continue
    statement is encountered within the Switch.

So while we "weaken" the Continue statement by rendering it as a break
statement, we also place checks immediately at the locations to which those
break statements will jump, until we can be sure we've reached the intended
target of the original Continue.

## Avoiding single body `switch` statements

As described in https://github.com/gfx-rs/wgpu/issues/4514, some language front
ends miscompile switch statements where all cases branch to the same body. Our
HLSL and GLSL backends render Switch statements with a single SwitchCase as
`do {} while(false);` loops.

However, this rewriting introduces a new loop that could "capture" continue
statements in its body. To avoid doing so, we apply the Continue-to-break
transformation described above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from ..proc import Namer


@dataclass(slots=True)
class LoopNesting:
    """Marker indicating that we're nested in at least one Loop statement."""


@dataclass(slots=True)
class SwitchNesting:
    """State for a nested Switch that may need to forward Continues."""

    variable: str
    continue_encountered: bool


Nesting = Union[LoopNesting, SwitchNesting]


@dataclass(slots=True)
class ExitControlFlow:
    """A micro-IR for code a backend should generate after a Switch."""

    kind: str
    variable: Optional[str] = None

    @classmethod
    def none(cls) -> "ExitControlFlow":
        return cls(kind="none")

    @classmethod
    def continue_(cls, variable: str) -> "ExitControlFlow":
        return cls(kind="continue", variable=variable)

    @classmethod
    def break_(cls, variable: str) -> "ExitControlFlow":
        return cls(kind="break", variable=variable)


class ContinueCtx:
    """Utility for tracking nesting of loops and switches.

    Backends that cannot emit `continue` directly within certain `switch`
    constructs can use this to decide when to lower `continue` to a `break`
    plus a forwarding flag, and what code to emit after the `switch`.
    """

    def __init__(self) -> None:
        self._stack: List[Nesting] = []

    def clear(self) -> None:
        """Reset internal state."""

        self._stack.clear()

    def enter_loop(self) -> None:
        """Record entering a Loop statement."""

        self._stack.append(LoopNesting())

    def exit_loop(self) -> None:
        """Record exiting a Loop statement."""

        if not self._stack or not isinstance(self._stack[-1], LoopNesting):
            raise RuntimeError("ContinueCtx stack out of sync")
        self._stack.pop()

    def enter_switch(self, namer: Namer) -> Optional[str]:
        """Record entering a Switch statement.

        Returns:
            The name of a new `bool` local variable to declare above the switch,
            if this switch is the outermost switch within a loop that needs
            forwarding. If this returns None, no new variable needs to be
            declared.
        """

        if not self._stack:
            return None

        top = self._stack[-1]
        if isinstance(top, LoopNesting):
            variable = namer.call("should_continue")
            self._stack.append(SwitchNesting(variable=variable, continue_encountered=False))
            return variable

        if isinstance(top, SwitchNesting):
            self._stack.append(
                SwitchNesting(variable=top.variable, continue_encountered=False)
            )
            return None

        return None

    def exit_switch(self) -> ExitControlFlow:
        """Record leaving a Switch statement.

        Returns:
            An ExitControlFlow indicating what code should be emitted after the
            switch to forward continues.
        """

        if not self._stack:
            return ExitControlFlow.none()

        popped = self._stack.pop()
        if isinstance(popped, LoopNesting):
            raise RuntimeError("Unexpected loop state when exiting switch")

        if not popped.continue_encountered:
            return ExitControlFlow.none()

        if self._stack and isinstance(self._stack[-1], SwitchNesting):
            outer = self._stack[-1]
            outer.continue_encountered = True
            return ExitControlFlow.break_(popped.variable)

        return ExitControlFlow.continue_(popped.variable)

    def continue_encountered(self) -> Optional[str]:
        """Determine what to generate for a Continue statement.

        If we can generate an ordinary `continue`, return None.

        Otherwise, we're enclosed by a Switch that is itself enclosed by a Loop.
        Return the name of the forwarding variable that should be set to true
        and then followed by a `break`.
        """

        if self._stack and isinstance(self._stack[-1], SwitchNesting):
            top = self._stack[-1]
            top.continue_encountered = True
            return top.variable

        return None


__all__ = [
    "ContinueCtx",
    "ExitControlFlow",
]
