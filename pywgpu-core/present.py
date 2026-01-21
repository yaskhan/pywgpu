"""
Presentation and swapchain logic.

This module implements presentation and swapchain management for wgpu-core.
It provides functionality for:
- Acquiring textures from a surface for rendering
- Presenting textures to the display
- Discarding textures that weren't presented

The presentation system manages the lifecycle of surface textures and
ensures proper synchronization between the GPU and display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import errors
from .device import Device
from .id import SurfaceId, TextureId


@dataclass
class Presentation:
    """
    Presentation state for a surface.

    This struct holds the state needed for presenting textures to a surface.

    Attributes:
        device: The device used for presentation.
        config: The surface configuration.
        acquired_texture: The currently acquired texture, if any.
    """

    device: Device
    config: Any
    acquired_texture: Optional[Any] = None


@dataclass
class SurfaceError(Exception):
    """
    Error related to surface operations.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ConfigureSurfaceError(Exception):
    """
    Error related to surface configuration.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class SurfaceOutput:
    """
    Output from getting the current texture from a surface.

    Attributes:
        status: The status of the surface.
        texture: The texture ID, if any.
    """

    status: Any
    texture: Optional[TextureId]


class Surface:
    """
    A window surface for presentation.

    A surface represents a window or display surface that can be used for
    presentation. It provides methods for getting the current texture and
    presenting it to the display.

    Attributes:
        presentation: Mutex-protected presentation state.
        surface_per_backend: Map of backend to surface.
    """

    def __init__(self) -> None:
        """Initialize the surface."""
        self.presentation: Optional[Presentation] = None
        self.surface_per_backend: dict[Any, Any] = {}

    def get_current_texture(self) -> SurfaceOutput:
        """
        Get the current texture for presentation.

        This method acquires a texture from the surface that can be used
        for rendering. The texture must be presented or discarded before
        acquiring another texture.

        Returns:
            The surface output containing the texture and status.

        Raises:
            SurfaceError: If the surface is not configured or invalid.
        """
        if self.presentation is None:
            raise SurfaceError("Surface is not configured")

        # Import HAL
        try:
            import sys
            import os

            _hal_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "pywgpu-hal"
            )
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            raise RuntimeError("pywgpu_hal module not available for surface operations")

        # Get the HAL surface for the current backend
        backend = getattr(self.presentation.device, "backend", None)
        if backend not in self.surface_per_backend:
            raise SurfaceError("Surface not available for current backend")

        hal_surface = self.surface_per_backend[backend]

        # Get the device's fence for synchronization
        device = self.presentation.device
        fence = getattr(device, "fence", None)
        if fence is None:
            raise SurfaceError("Device does not have a fence for synchronization")

        # Acquire texture from HAL surface
        try:
            # timeout=None means wait indefinitely
            acquired = hal_surface.acquire_texture(timeout=None, fence=fence)

            if acquired is None:
                return SurfaceOutput(status="Timeout", texture=None)

            # Store the acquired texture
            self.presentation.acquired_texture = acquired

            # Create a texture ID for the acquired texture
            # In a real implementation, this would be registered in the hub
            texture_id = TextureId(0)  # Placeholder

            return SurfaceOutput(status="Success", texture=texture_id)

        except hal.SurfaceError as e:
            raise SurfaceError(f"Failed to acquire surface texture: {e}") from e

    def present(self) -> Any:
        """
        Present the current texture to the surface.

        This method presents the acquired texture to the display. The
        texture must have been acquired via get_current_texture.

        Returns:
            The presentation status.

        Raises:
            SurfaceError: If presentation fails.
        """
        if self.presentation is None:
            raise SurfaceError("Surface is not configured")

        if self.presentation.acquired_texture is None:
            raise SurfaceError("No texture acquired for presentation")

        # Import HAL
        try:
            import sys
            import os

            _hal_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "pywgpu-hal"
            )
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            raise RuntimeError("pywgpu_hal module not available for surface operations")

        # Get the HAL surface and queue
        backend = getattr(self.presentation.device, "backend", None)
        if backend not in self.surface_per_backend:
            raise SurfaceError("Surface not available for current backend")

        hal_surface = self.surface_per_backend[backend]
        queue = getattr(self.presentation.device, "queue", None)

        if queue is None:
            raise SurfaceError("Device does not have a queue for presentation")

        # Get the raw HAL queue
        hal_queue = getattr(queue, "raw", None) or queue

        try:
            # Present the texture using HAL queue
            hal_queue.present(hal_surface, self.presentation.acquired_texture)

            # Clear the acquired texture
            self.presentation.acquired_texture = None

            return "Success"

        except Exception as e:
            raise SurfaceError(f"Failed to present texture: {e}") from e

    def discard(self) -> None:
        """
        Discard the current texture.

        This method discards the acquired texture without presenting it.
        This is useful when the texture is not needed for presentation.

        Raises:
            SurfaceError: If the texture cannot be discarded.
        """
        if self.presentation is None:
            raise SurfaceError("Surface is not configured")

        if self.presentation.acquired_texture is None:
            # Nothing to discard
            return

        # Import HAL
        try:
            import sys
            import os

            _hal_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "pywgpu-hal"
            )
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            raise RuntimeError("pywgpu_hal module not available for surface operations")

        # Get the HAL surface
        backend = getattr(self.presentation.device, "backend", None)
        if backend not in self.surface_per_backend:
            raise SurfaceError("Surface not available for current backend")

        hal_surface = self.surface_per_backend[backend]

        try:
            # Discard the texture using HAL surface
            hal_surface.discard_texture(self.presentation.acquired_texture)

            # Clear the acquired texture
            self.presentation.acquired_texture = None

        except Exception as e:
            raise SurfaceError(f"Failed to discard texture: {e}") from e


class Global:
    """
    Global state for wgpu-core.

    This class provides methods for surface operations and maintains
    a registry of surfaces. It acts as the central hub for managing
    surfaces across the application.

    Attributes:
        surfaces: Registry of surfaces by ID.
    """

    def __init__(self):
        """Initialize the Global state."""
        self.surfaces: dict[SurfaceId, Surface] = {}

    def surface_get_current_texture(
        self,
        surface_id: SurfaceId,
        texture_id_in: Optional[TextureId],
    ) -> SurfaceOutput:
        """
        Get the current texture from a surface.

        This method retrieves the surface from the registry and calls
        its get_current_texture method to acquire a texture for rendering.

        Args:
            surface_id: The surface ID.
            texture_id_in: Optional texture ID to use (currently unused).

        Returns:
            The surface output containing texture and status.

        Raises:
            SurfaceError: If the surface is not found or not configured.
        """
        # Look up the surface in the registry
        if surface_id not in self.surfaces:
            raise SurfaceError(f"Surface {surface_id} not found in registry")

        surface = self.surfaces[surface_id]

        # Delegate to the surface's method
        try:
            return surface.get_current_texture()
        except SurfaceError:
            raise
        except Exception as e:
            raise SurfaceError(f"Failed to get current texture: {e}") from e

    def surface_present(self, surface_id: SurfaceId) -> Any:
        """
        Present the current texture from a surface.

        This method retrieves the surface from the registry and calls
        its present method to display the acquired texture.

        Args:
            surface_id: The surface ID.

        Returns:
            The presentation status.

        Raises:
            SurfaceError: If the surface is not found or presentation fails.
        """
        # Look up the surface in the registry
        if surface_id not in self.surfaces:
            raise SurfaceError(f"Surface {surface_id} not found in registry")

        surface = self.surfaces[surface_id]

        # Delegate to the surface's method
        try:
            return surface.present()
        except SurfaceError:
            raise
        except Exception as e:
            raise SurfaceError(f"Failed to present surface: {e}") from e

    def surface_texture_discard(self, surface_id: SurfaceId) -> None:
        """
        Discard the current texture from a surface.

        This method retrieves the surface from the registry and calls
        its discard method to release the acquired texture without
        presenting it.

        Args:
            surface_id: The surface ID.

        Raises:
            SurfaceError: If the surface is not found or discard fails.
        """
        # Look up the surface in the registry
        if surface_id not in self.surfaces:
            raise SurfaceError(f"Surface {surface_id} not found in registry")

        surface = self.surfaces[surface_id]

        # Delegate to the surface's method
        try:
            surface.discard()
        except SurfaceError:
            raise
        except Exception as e:
            raise SurfaceError(f"Failed to discard texture: {e}") from e

    def register_surface(self, surface_id: SurfaceId, surface: Surface) -> None:
        """
        Register a surface in the global registry.

        Args:
            surface_id: The ID to register the surface under.
            surface: The surface to register.
        """
        self.surfaces[surface_id] = surface

    def unregister_surface(self, surface_id: SurfaceId) -> None:
        """
        Unregister a surface from the global registry.

        Args:
            surface_id: The ID of the surface to unregister.
        """
        if surface_id in self.surfaces:
            del self.surfaces[surface_id]
