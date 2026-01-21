"""
Indirect validation for draw and dispatch commands.

This module provides GPU-based validation of indirect commands to ensure
they don't exceed device limits before execution.
"""

from typing import Optional
from dataclasses import dataclass


class CreateIndirectValidationPipelineError(Exception):
    """Error creating indirect validation pipeline."""

    pass


@dataclass
class BindGroups:
    """
    Bind groups for indirect validation.

    Attributes:
        dispatch: Bind group for dispatch validation.
        draw: Bind group for draw validation.
    """

    dispatch: any  # Box<dyn hal::DynBindGroup>
    draw: any  # Box<dyn hal::DynBindGroup>

    @staticmethod
    def new(
        indirect_validation: "IndirectValidation",
        device: any,
        buffer_size: int,
        buffer: any,
    ) -> Optional["BindGroups"]:
        """
        Create bind groups for indirect validation.

        Returns None if buffer_size is 0.

        Args:
            indirect_validation: The indirect validation instance.
            device: The device.
            buffer_size: Size of the buffer.
            buffer: The buffer to validate.

        Returns:
            BindGroups or None.
        """
        try:
            dispatch_bg = indirect_validation.dispatch.create_src_bind_group(
                device.raw() if hasattr(device, "raw") else device,
                device.limits if hasattr(device, "limits") else {},
                buffer_size,
                buffer,
            )

            draw_bg = indirect_validation.draw.create_src_bind_group(
                device.raw() if hasattr(device, "raw") else device,
                device.limits if hasattr(device, "limits") else {},
                buffer_size,
                buffer,
            )

            if dispatch_bg is None and draw_bg is None:
                return None

            if dispatch_bg is not None and draw_bg is not None:
                return BindGroups(dispatch=dispatch_bg, draw=draw_bg)

            # Should not happen - both should be None or both should have values
            return None

        except Exception:
            return None

    def dispose(self, device: any) -> None:
        """
        Dispose of bind groups.

        Args:
            device: The HAL device.
        """
        try:
            hal_device = device.raw() if hasattr(device, "raw") else device
            if hasattr(hal_device, "destroy_bind_group"):
                hal_device.destroy_bind_group(self.dispatch)
                hal_device.destroy_bind_group(self.draw)
        except Exception:
            pass


class IndirectValidation:
    """
    Main indirect validation coordinator.

    Manages both dispatch and draw validation.

    Attributes:
        dispatch: Dispatch command validator.
        draw: Draw command validator.
    """

    def __init__(self, device: any, limits: dict, features: dict, backend: str):
        """
        Create a new indirect validation instance.

        Args:
            device: The HAL device.
            limits: Device limits.
            features: Device features.
            backend: Backend name.
        """
        from .dispatch import Dispatch
        from .draw import Draw

        try:
            self.dispatch = Dispatch(device, limits)
        except Exception as e:
            print(f"indirect-validation error: {e}")
            raise

        try:
            self.draw = Draw(device, features, backend)
        except Exception as e:
            print(f"indirect-draw-validation error: {e}")
            raise

    def dispose(self, device: any) -> None:
        """
        Dispose of all validation resources.

        Args:
            device: The HAL device.
        """
        self.dispatch.dispose(device)
        self.draw.dispose(device)


# Export submodules
from .dispatch import Dispatch, DispatchValidator
from .draw import Draw, DrawValidator, DrawResources, DrawBatcher
from .utils import UniqueIndexScratch, BufferBarrierScratch, BufferBarriers

__all__ = [
    "IndirectValidation",
    "BindGroups",
    "Dispatch",
    "DispatchValidator",
    "Draw",
    "DrawValidator",
    "DrawResources",
    "DrawBatcher",
    "UniqueIndexScratch",
    "BufferBarrierScratch",
    "BufferBarriers",
    "CreateIndirectValidationPipelineError",
]
