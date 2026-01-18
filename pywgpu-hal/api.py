"""
Backend API protocol definitions.

This module provides Protocol definitions for HAL backends. These are
re-exported from lib.py for backward compatibility.
"""

from typing import Protocol, TypeVar, Optional, List, Any, Sequence

# Import core protocols from lib
from .lib import (
    Api as ApiProtocol,
    Instance as InstanceProtocol,
    Surface as SurfaceProtocol,
    Adapter as AdapterProtocol,
    Device as DeviceProtocol,
    Queue as QueueProtocol,
    CommandEncoder as CommandEncoderProtocol,
    # Descriptors
    BufferDescriptor,
    TextureDescriptor,
    TextureViewDescriptor,
    SamplerDescriptor,
    BindGroupLayoutDescriptor,
    InstanceDescriptor,
    SurfaceConfiguration,
    SurfaceCapabilities,
)

T = TypeVar('T')

# Re-export protocols with original names for compatibility
Instance = InstanceProtocol
Adapter = AdapterProtocol
Device = DeviceProtocol
Surface = SurfaceProtocol
Api = ApiProtocol
Queue = QueueProtocol
CommandEncoder = CommandEncoderProtocol

__all__ = [
    'Instance',
    'Adapter',
    'Device',
    'Surface',
    'Api',
    'Queue',
    'CommandEncoder',
    'BufferDescriptor',
    'TextureDescriptor',
    'TextureViewDescriptor',
    'SamplerDescriptor',
    'BindGroupLayoutDescriptor',
    'InstanceDescriptor',
    'SurfaceConfiguration',
    'SurfaceCapabilities',
]
