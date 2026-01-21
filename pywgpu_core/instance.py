from typing import Any, Dict, Optional
from pywgpu_types.adapter import (
    AdapterInfo,
    RequestAdapterOptions,
    DeviceType,
    PowerPreference,
)
from pywgpu_types.instance import InstanceDescriptor
from pywgpu_types.backend import Backend, Backends
from pywgpu_hal.api import DynInstance


class Instance:
    """
    Core Instance logic.

    Manages HAL instances and backends.
    """

    def __init__(self, name: str, descriptor: InstanceDescriptor) -> None:
        self.name = name
        self.descriptor = descriptor
        self.backends: Dict[Backend, DynInstance] = {}
        self.supported_backends = Backends.BACKEND_NONE

    def request_adapter(
        self,
        options: RequestAdapterOptions,
        backends: Optional[Backends] = None,
    ) -> AdapterInfo:
        """
        Finds a suitable adapter.

        Args:
            options: RequestAdapterOptions describing the requirements.
            backends: Optional backends to search. If None, uses descriptor backends.

        Returns:
            AdapterInfo for the best matching adapter.

        Raises:
            Exception: If no compatible adapter is found.
        """
        if backends is None:
            backends = self.descriptor.backends

        all_adapters: list[tuple[Backend, DeviceType, AdapterInfo]] = []

        for backend in Backend:
            if not backends & Backends.from_backend(backend):
                continue

            hal_instance = self.backends.get(backend)
            if hal_instance is None:
                continue

            # Enumerate adapters from HAL
            try:
                hal_adapters = hal_instance.enumerate_adapters(
                    options.compatible_surface
                )
            except Exception:
                continue

            for hal_adapter in hal_adapters:
                info = hal_adapter.info

                # Filter by force_fallback_adapter
                if options.force_fallback_adapter:
                    if info.device_type != DeviceType.CPU:
                        continue

                # Filter by compatible_surface
                if options.compatible_surface is not None:
                    if not hal_instance.is_surface_supported(
                        hal_adapter, options.compatible_surface
                    ):
                        continue

                all_adapters.append((backend, info.device_type, info))

        if not all_adapters:
            raise Exception("No compatible adapter found")

        # Sort by power preference
        def sort_key(item: tuple[Backend, DeviceType, AdapterInfo]) -> int:
            backend, device_type, _ = item
            if options.power_preference == PowerPreference.LOW_POWER:
                if device_type == DeviceType.INTEGRATED_GPU:
                    return 0
                elif device_type == DeviceType.DISCRETE_GPU:
                    return 1
                else:
                    return 2
            elif options.power_preference == PowerPreference.HIGH_PERFORMANCE:
                if device_type == DeviceType.DISCRETE_GPU:
                    return 0
                elif device_type == DeviceType.INTEGRATED_GPU:
                    return 1
                else:
                    return 2
            else:
                return 0

        all_adapters.sort(key=sort_key)
        return all_adapters[0][2]
