"""
Pipeline cache management.

This module implements pipeline cache validation and management. A pipeline
cache stores compiled pipeline state that can be reused across application
runs to reduce compilation time.

The module provides:
- Pipeline cache header validation
- Cache data validation
- Cache data creation with headers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import errors


@dataclass
class PipelineCacheValidationError(Exception):
    """
    Error validating pipeline cache data.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message

    def was_avoidable(self) -> bool:
        """
        Check if the error could have been avoided.
        
        Returns:
            True if the error was avoidable, False otherwise.
        """
        return self.message in ["DeviceMismatch"]


@dataclass
class PipelineCacheHeader:
    """
    Header for pipeline cache data.
    
    This header contains metadata about the cache data, including:
    - Magic number to identify the cache format
    - Version information
    - Backend and adapter information
    - Validation key
    - Data size
    - Hash space for future use
    
    Attributes:
        magic: Magic number to identify the cache format.
        header_version: Version of the header.
        cache_abi: Cache ABI version.
        backend: Backend identifier.
        adapter_key: Adapter identifier.
        validation_key: Validation key for cache compatibility.
        data_size: Size of the cache data.
        hash_space: Space reserved for hash.
    """

    magic: bytes
    header_version: int
    cache_abi: int
    backend: int
    adapter_key: bytes
    validation_key: bytes
    data_size: int
    hash_space: int

    @classmethod
    def read(cls, data: bytes) -> Optional[tuple[PipelineCacheHeader, bytes]]:
        """
        Read a pipeline cache header from data.
        
        Args:
            data: The cache data to read from.
        
        Returns:
            A tuple of (header, remaining_data), or None if data is truncated.
        """
        if len(data) < 64:  # Header size
            return None
        
        # Read header fields
        magic = data[0:8]
        header_version = int.from_bytes(data[8:12], 'big')
        cache_abi = int.from_bytes(data[12:16], 'big')
        backend = data[16]
        adapter_key = data[17:32]
        validation_key = data[32:48]
        data_size = int.from_bytes(data[48:56], 'big')
        hash_space = int.from_bytes(data[56:64], 'big')
        
        header = PipelineCacheHeader(
            magic=magic,
            header_version=header_version,
            cache_abi=cache_abi,
            backend=backend,
            adapter_key=adapter_key,
            validation_key=validation_key,
            data_size=data_size,
            hash_space=hash_space,
        )
        
        return (header, data[64:])

    def write(self, into: bytearray) -> Optional[None]:
        """
        Write the pipeline cache header to data.
        
        Args:
            into: The buffer to write to (must be at least 64 bytes).
        
        Returns:
            None if successful, error otherwise.
        """
        if len(into) < 64:
            return "Buffer too small for header"
        
        # Write header fields
        into[0:8] = self.magic
        into[8:12] = self.header_version.to_bytes(4, 'big')
        into[12:16] = self.cache_abi.to_bytes(4, 'big')
        into[16] = self.backend
        into[17:32] = self.adapter_key
        into[32:48] = self.validation_key
        into[48:56] = self.data_size.to_bytes(8, 'big')
        into[56:64] = self.hash_space.to_bytes(8, 'big')
        
        return None


def validate_pipeline_cache(
    cache_data: bytes,
    adapter: Any,
    validation_key: bytes,
) -> tuple[bytes, Optional[PipelineCacheValidationError]]:
    """
    Validate the data in a pipeline cache.
    
    Args:
        cache_data: The cache data to validate.
        adapter: The adapter info.
        validation_key: The validation key.
    
    Returns:
        A tuple of (validated_data, error).
    """
    header_result = PipelineCacheHeader.read(cache_data)
    if header_result is None:
        return (b"", PipelineCacheValidationError("The pipeline cache data was truncated"))
    
    header, remaining_data = header_result
    
    if header.magic != b"WGPUPLCH":
        return (b"", PipelineCacheValidationError("The pipeline cache data was corrupted"))
    
    if header.header_version != 1:
        return (b"", PipelineCacheValidationError("The pipeline cache data was out of date"))
    
    if header.cache_abi != 8:  # 64-bit
        return (b"", PipelineCacheValidationError("The pipeline cache data was out of date"))
    
    # More validation would go here
    return (remaining_data, None)


def add_cache_header(
    in_region: bytearray,
    data: bytes,
    adapter: Any,
    validation_key: bytes,
) -> None:
    """
    Add a cache header to the data.
    
    This function creates a pipeline cache header and writes it to the
    beginning of the buffer, followed by the cache data.
    
    Args:
        in_region: The buffer to write the header and data to.
        data: The cache data.
        adapter: The adapter info.
        validation_key: The validation key.
    """
    # Get adapter information
    backend = getattr(adapter, 'backend', 0)
    adapter_key = getattr(adapter, 'key', b'\x00' * 15)
    
    # Ensure adapter_key is exactly 15 bytes
    if isinstance(adapter_key, bytes):
        if len(adapter_key) > 15:
            adapter_key = adapter_key[:15]
        elif len(adapter_key) < 15:
            adapter_key = adapter_key + b'\x00' * (15 - len(adapter_key))
    else:
        adapter_key = b'\x00' * 15
    
    # Ensure validation_key is exactly 16 bytes
    if isinstance(validation_key, bytes):
        if len(validation_key) > 16:
            validation_key = validation_key[:16]
        elif len(validation_key) < 16:
            validation_key = validation_key + b'\x00' * (16 - len(validation_key))
    else:
        validation_key = b'\x00' * 16
    
    # Create header
    header = PipelineCacheHeader(
        magic=b"WGPUPLCH",
        header_version=1,
        cache_abi=8,  # 64-bit
        backend=backend,
        adapter_key=adapter_key,
        validation_key=validation_key,
        data_size=len(data),
        hash_space=0,  # Reserved for future use
    )
    
    # Write header to buffer
    header.write(in_region)
    
    # Write data after header
    if len(in_region) >= 64 + len(data):
        in_region[64:64+len(data)] = data
