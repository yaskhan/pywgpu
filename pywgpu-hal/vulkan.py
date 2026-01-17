from __future__ import annotations
from typing import List, Optional
from .base import Api as HalApi, Adapter as HalAdapter, Device as HalDevice, Buffer as HalBuffer
from pywgpu_types import DeviceDescriptor

class VulkanApi(HalApi):
    def enumerate_adapters(self) -> List[HalAdapter]:
        return [VulkanAdapter()]

class VulkanAdapter(HalAdapter):
    def request_device(self, descriptor: DeviceDescriptor) -> HalDevice:
        return VulkanDevice()

class VulkanDevice(HalDevice):
    def create_buffer(self, size: int, usage: int) -> HalBuffer:
        return VulkanBuffer()

class VulkanBuffer(HalBuffer):
    def destroy(self) -> None:
        pass
