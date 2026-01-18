from typing import Any, Optional
from pywgpu_types.queue import CommandBufferDescriptor


class CommandBuffer:
    """
    A command buffer that can be submitted for execution.
    """

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label


class CommandEncoder:
    """
    Command encoder stub.
    """

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label
        self._is_finished = False

    def finish(self, descriptor: Optional[CommandBufferDescriptor] = None) -> CommandBuffer:
        """
        Finish recording and return a CommandBuffer.

        Args:
            descriptor: Optional command buffer descriptor.

        Returns:
            A CommandBuffer ready for submission.

        Raises:
            RuntimeError: If the encoder has already been finished.
        """
        if self._is_finished:
            raise RuntimeError("CommandEncoder has already been finished")

        self._is_finished = True
        label = descriptor.label if descriptor else None
        return CommandBuffer(label=label)
