from typing import Any, Optional, Tuple, List, Union, Iterator
from abc import ABC, abstractmethod


class InstanceInterface(ABC):
    @abstractmethod
    def create_surface(self, target: Any) -> Any:
        pass

    @abstractmethod
    def request_adapter(self, options: Any) -> Any:
        pass

    @abstractmethod
    def poll_all_devices(self, force_wait: bool) -> bool:
        pass


class AdapterInterface(ABC):
    @abstractmethod
    def request_device(self, descriptor: Any) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def features(self) -> Any:
        pass

    @abstractmethod
    def limits(self) -> Any:
        pass

    @abstractmethod
    def get_info(self) -> Any:
        pass


class DeviceInterface(ABC):
    @abstractmethod
    def create_buffer(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_texture(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_sampler(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_bind_group(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_bind_group_layout(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_pipeline_layout(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_shader_module(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_command_encoder(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_render_pipeline(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_compute_pipeline(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def create_pipeline_cache(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


class BufferInterface(ABC):
    @abstractmethod
    def map_async(self, mode: Any, offset: int, size: int, callback: Any) -> None:
        pass

    @abstractmethod
    def unmap(self) -> None:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


class TextureInterface(ABC):
    @abstractmethod
    def create_view(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


class CommandEncoderInterface(ABC):
    @abstractmethod
    def begin_render_pass(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def begin_compute_pass(self, descriptor: Any) -> Any:
        pass

    @abstractmethod
    def finish(self, descriptor: Any = None) -> Any:
        pass


class QueueInterface(ABC):
    @abstractmethod
    def submit(self, command_buffers: List[Any]) -> None:
        pass

    @abstractmethod
    def write_buffer(self, buffer: Any, offset: int, data: bytes) -> None:
        pass

    @abstractmethod
    def write_texture(self, texture: Any, data: bytes, layout: Any, size: Any) -> None:
        pass
