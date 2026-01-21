from typing import Any, Optional, List
from dataclasses import dataclass


@dataclass
class TlasDescriptor:
    """Descriptor for a top-level acceleration structure."""

    label: Optional[str] = None
    size: int = 0
    instances: int = 0
    usage: int = 0


@dataclass
class BlasDescriptor:
    """Descriptor for a bottom-level acceleration structure."""

    label: Optional[str] = None
    size: int = 0
    usage: int = 0


@dataclass
class TlasInstance:
    """Instance for a top-level acceleration structure."""

    blas_id: Any
    transform: List[float]
    instance_id: int = 0
    mask: int = 0xFF
    flags: int = 0
    acceleration_structure_address: int = 0


@dataclass
class BlasGeometry:
    """Geometry for a bottom-level acceleration structure."""

    type: str  # "triangles" or "aabbs"
    vertex_buffer: Any
    index_buffer: Optional[Any] = None
    vertex_format: Optional[str] = None
    vertex_count: int = 0
    vertex_stride: int = 0
    index_format: Optional[str] = None
    index_count: int = 0


class RayTracing:
    """
    Ray tracing logic for acceleration structure management.

    This class provides methods for creating and managing acceleration structures
    (AS) for ray tracing. It supports both top-level (TLAS) and bottom-level (BLAS)
    acceleration structures.

    Attributes:
        enabled: Whether ray tracing is enabled.
        tlas_counter: Counter for TLAS IDs.
        blas_counter: Counter for BLAS IDs.
        tlas_registry: Registry of created TLAS.
        blas_registry: Registry of created BLAS.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.tlas_counter = 0
        self.blas_counter = 0
        self.tlas_registry: dict[int, Any] = {}
        self.blas_registry: dict[int, Any] = {}

    def create_tlas(self, device: Any, desc: TlasDescriptor) -> Any:
        """Create a top-level acceleration structure.

        Args:
            device: The device to create the TLAS on.
            desc: TLAS descriptor.

        Returns:
            The created TLAS object.
        """
        # Import HAL for acceleration structures
        try:
            import sys
            import os

            _hal_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "pywgpu-hal",
            )
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            # Fallback to placeholder if HAL not available
            self.tlas_counter += 1
            tlas = {
                "id": self.tlas_counter,
                "label": desc.label,
                "size": desc.size,
                "type": "tlas",
            }
            self.tlas_registry[self.tlas_counter] = tlas
            return tlas

        # Get HAL device
        hal_device = getattr(device, "hal_device", None) or getattr(
            device, "_hal_device", None
        )
        if hal_device is None:
            raise RuntimeError("Device does not have HAL device for ray tracing")

        # Create HAL TLAS descriptor
        hal_desc = hal.AccelerationStructureDescriptor(
            label=desc.label, size=desc.size, usage=desc.usage
        )

        try:
            # Create TLAS using HAL
            tlas = hal_device.create_acceleration_structure(hal_desc)
            self.tlas_counter += 1
            self.tlas_registry[self.tlas_counter] = tlas
            return tlas
        except Exception as e:
            raise RuntimeError(f"Failed to create TLAS: {e}") from e

    def create_blas(self, device: Any, desc: BlasDescriptor) -> Any:
        """Create a bottom-level acceleration structure.

        Args:
            device: The device to create the BLAS on.
            desc: BLAS descriptor.

        Returns:
            The created BLAS object.
        """
        # Import HAL for acceleration structures
        try:
            import sys
            import os

            _hal_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "pywgpu-hal",
            )
            if _hal_path not in sys.path:
                sys.path.insert(0, _hal_path)
            import lib as hal
        except ImportError:
            # Fallback to placeholder if HAL not available
            self.blas_counter += 1
            blas = {
                "id": self.blas_counter,
                "label": desc.label,
                "size": desc.size,
                "type": "blas",
            }
            self.blas_registry[self.blas_counter] = blas
            return blas

        # Get HAL device
        hal_device = getattr(device, "hal_device", None) or getattr(
            device, "_hal_device", None
        )
        if hal_device is None:
            raise RuntimeError("Device does not have HAL device for ray tracing")

        # Create HAL BLAS descriptor
        hal_desc = hal.AccelerationStructureDescriptor(
            label=desc.label, size=desc.size, usage=desc.usage
        )

        try:
            # Create BLAS using HAL
            blas = hal_device.create_acceleration_structure(hal_desc)
            self.blas_counter += 1
            self.blas_registry[self.blas_counter] = blas
            return blas
        except Exception as e:
            raise RuntimeError(f"Failed to create BLAS: {e}") from e

    def build_tlas(self, tlas: Any, instances: List[TlasInstance]) -> None:
        """Build a top-level acceleration structure.

        Args:
            tlas: The TLAS to build.
            instances: List of instances to include in the TLAS.
        """
        # Building TLAS requires encoding build commands
        # This would typically be done via command encoder
        # For now, store the instances with the TLAS
        if isinstance(tlas, dict):
            tlas["instances"] = instances
            tlas["built"] = True
        else:
            # HAL TLAS - would need command encoder to build
            # This is handled by encode_build_acceleration_structures
            pass

    def build_blas(self, blas: Any, geometry: BlasGeometry) -> None:
        """Build a bottom-level acceleration structure.

        Args:
            blas: The BLAS to build.
            geometry: Geometry data for the BLAS.
        """
        # Building BLAS requires encoding build commands
        # This would typically be done via command encoder
        # For now, store the geometry with the BLAS
        if isinstance(blas, dict):
            blas["geometry"] = geometry
            blas["built"] = True
        else:
            # HAL BLAS - would need command encoder to build
            # This is handled by encode_build_acceleration_structures
            pass

    def query_tlas(self, tlas: Any) -> Any:
        """Query a top-level acceleration structure.

        Args:
            tlas: The TLAS to query.

        Returns:
            TLAS information.
        """
        return tlas

    def query_blas(self, blas: Any) -> Any:
        """Query a bottom-level acceleration structure.

        Args:
            blas: The BLAS to query.

        Returns:
            BLAS information.
        """
        return blas

    def get_tlas_address(self, tlas: Any) -> int:
        """Get the GPU address of a top-level acceleration structure.

        Args:
            tlas: The TLAS.

        Returns:
            The GPU address of the TLAS.
        """
        # Get address from HAL AS if available
        if hasattr(tlas, "get_device_address"):
            try:
                return tlas.get_device_address()
            except Exception:
                pass

        # Fallback for placeholder
        if isinstance(tlas, dict):
            return tlas.get("address", 0)

        return 0

    def get_blas_address(self, blas: Any) -> int:
        """Get the GPU address of a bottom-level acceleration structure.

        Args:
            blas: The BLAS.

        Returns:
            The GPU address of the BLAS.
        """
        # Get address from HAL AS if available
        if hasattr(blas, "get_device_address"):
            try:
                return blas.get_device_address()
            except Exception:
                pass

        # Fallback for placeholder
        if isinstance(blas, dict):
            return blas.get("address", 0)

        return 0

    def destroy_tlas(self, tlas: Any) -> None:
        """Destroy a top-level acceleration structure.

        Args:
            tlas: The TLAS to destroy.
        """
        # Remove from registry
        if isinstance(tlas, dict):
            tlas_id = tlas.get("id")
            if tlas_id and tlas_id in self.tlas_registry:
                del self.tlas_registry[tlas_id]

        # Destroy HAL AS if it has destroy method
        if hasattr(tlas, "destroy"):
            try:
                tlas.destroy()
            except Exception:
                pass

    def destroy_blas(self, blas: Any) -> None:
        """Destroy a bottom-level acceleration structure.

        Args:
            blas: The BLAS to destroy.
        """
        # Remove from registry
        if isinstance(blas, dict):
            blas_id = blas.get("id")
            if blas_id and blas_id in self.blas_registry:
                del self.blas_registry[blas_id]

        # Destroy HAL AS if it has destroy method
        if hasattr(blas, "destroy"):
            try:
                blas.destroy()
            except Exception:
                pass
