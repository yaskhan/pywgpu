"""
GLSL type handling and validation.

This module provides type handling, validation, and conversion functionality
for GLSL shaders.
"""

from typing import Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass


class ImageDimension(Enum):
    """Image dimension types."""
    D1 = "1D"
    D2 = "2D"
    D3 = "3D"
    CUBE = "Cube"


class ImageClass(Enum):
    """Image class types."""
    STORAGE = "storage"
    DEPTH = "depth"
    SAMPLER = "sampler"


class StorageFormat(Enum):
    """Storage format types."""
    R8_UINT = "r8uint"
    R8_SINT = "r8sint"
    R8_UNORM = "r8unorm"
    R16_UINT = "r16uint"
    R16_SINT = "r16sint"
    R16_FLOAT = "r16float"
    R16_UNORM = "r16unorm"
    R16_SNORM = "r16snorm"
    R32_UINT = "r32uint"
    R32_SINT = "r32sint"
    R32_FLOAT = "r32float"
    RG8_UINT = "rg8uint"
    RG8_SINT = "rg8sint"
    RG8_UNORM = "rg8unorm"
    RG16_UINT = "rg16uint"
    RG16_SINT = "rg16sint"
    RG16_FLOAT = "r16float"
    RG16_UNORM = "rg16unorm"
    RG16_SNORM = "rg16snorm"
    RG32_UINT = "rg32uint"
    RG32_SINT = "rg32sint"
    RG32_FLOAT = "rg32float"
    RGBA8_UINT = "rgba8uint"
    RGBA8_SINT = "rgba8sint"
    RGBA8_UNORM = "rgba8unorm"
    RGBA8_SNORM = "rgba8snorm"
    RGBA16_UINT = "rgba16uint"
    RGBA16_SINT = "rgba16sint"
    RGBA16_FLOAT = "rgba16float"
    RGBA16_UNORM = "rgba16unorm"
    RGBA16_SNORM = "rgba16snorm"
    RGBA32_UINT = "rgba32uint"
    RGBA32_SINT = "rgba32sint"
    RGBA32_FLOAT = "rgba32float"


class StorageAccess(Enum):
    """Storage access flags."""
    LOAD = "load"
    STORE = "store"
    READ = "read"
    WRITE = "write"


@dataclass
class ImageType:
    """Image type information."""
    dim: ImageDimension
    arrayed: bool
    class_type: ImageClass
    format: Optional[StorageFormat] = None
    access: Optional[StorageAccess] = None


class TextureKind(Enum):
    """Texture kind types."""
    UNORM = "unorm"
    SNORM = "snorm"
    UINT = "uint"
    SINT = "sint"
    FLOAT = "float"
    SHADOW = "shadow"


class TypeParser:
    """Parser for GLSL type declarations."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.type_definitions: Dict[str, Any] = {}
    
    def parse_image_type(self, tokens: List[str]) -> Optional[ImageType]:
        """
        Parse image type from token list.
        
        Args:
            tokens: List of image type tokens
            
        Returns:
            ImageType or None if parsing fails
        """
        # Expected format: image[Size][Format][Type]
        # Example: image2D, image2DArray, uimage2D, etc.
        
        if not tokens:
            return None
        
        # TODO: Check that the texture format and the kind match (строка 159)
        # This should validate that the format specifier matches the kind specifier
        # For example, "uimage2D" should have both format (u = unsigned) and kind (image2D)
        
        # Remove "image" prefix
        base = tokens[0].replace("image", "")
        if not base:
            return None
        
        # Parse size dimension
        size_tokens = self._extract_size_tokens(base)
        dim = self._parse_image_dimension(size_tokens)
        arrayed = self._is_arrayed(size_tokens)
        
        # Parse format and kind
        format_kind = self._extract_format_kind(base)
        format = self._parse_storage_format(format_kind)
        kind = self._parse_texture_kind(format_kind)
        
        # TODO: Check that the texture format and the kind match
        # This should ensure:
        # 1. If format is specified, kind should be compatible
        # 2. Format suffixes (u, i, f) should match kind types
        # 3. Validate format-kind combinations
        
        if not self._validate_format_kind_match(format, kind):
            self.errors.append(f"Texture format and kind don't match: format={format}, kind={kind}")
            return None
        
        class_type = self._determine_image_class(format, kind)
        
        return ImageType(
            dim=dim,
            arrayed=arrayed,
            class_type=class_type,
            format=format,
            access=StorageAccess.LOAD | StorageAccess.STORE
        )
    
    def _extract_size_tokens(self, base: str) -> List[str]:
        """Extract size tokens from base string."""
        # TODO: Implement size token extraction
        # Should extract "1D", "2D", "3D", "Cube", "Rect" etc.
        return []
    
    def _parse_image_dimension(self, size_tokens: List[str]) -> ImageDimension:
        """Parse image dimension from size tokens."""
        # TODO: Implement image dimension parsing
        return ImageDimension.D2
    
    def _is_arrayed(self, size_tokens: List[str]) -> bool:
        """Check if image is arrayed."""
        # TODO: Implement arrayed check
        return "Array" in str(size_tokens)
    
    def _extract_format_kind(self, base: str) -> str:
        """Extract format and kind information."""
        # TODO: Implement format-kind extraction
        # Should extract format prefix (u, i, f, nothing) and kind suffix (1D, 2D, etc.)
        return base
    
    def _parse_storage_format(self, format_kind: str) -> Optional[StorageFormat]:
        """Parse storage format."""
        # TODO: Implement storage format parsing
        # Should map format strings to StorageFormat enum
        format_mappings = {
            "u": StorageFormat.R32_UINT,
            "i": StorageFormat.R32_SINT,
            "f": StorageFormat.R32_FLOAT,
        }
        
        for prefix, format_type in format_mappings.items():
            if format_kind.startswith(prefix):
                return format_type
        
        return StorageFormat.R32_FLOAT  # Default
    
    def _parse_texture_kind(self, format_kind: str) -> TextureKind:
        """Parse texture kind."""
        # TODO: Implement texture kind parsing
        # Should map format strings to TextureKind enum
        kind_mappings = {
            "u": TextureKind.UINT,
            "i": TextureKind.SINT,
            "f": TextureKind.FLOAT,
            "shadow": TextureKind.SHADOW,
        }
        
        for prefix, kind in kind_mappings.items():
            if format_kind.startswith(prefix):
                return kind
        
        return TextureKind.FLOAT  # Default
    
    def _validate_format_kind_match(self, format: Optional[StorageFormat], kind: TextureKind) -> bool:
        """
        Validate that texture format and kind match.
        
        Args:
            format: Storage format
            kind: Texture kind
            
        Returns:
            True if format and kind are compatible
        """
        # TODO: Check that the texture format and the kind match
        # This should validate:
        # 1. Unsigned formats should match UINT kind
        # 2. Signed formats should match SINT kind
        # 3. Float formats should match FLOAT kind
        # 4. Shadow formats should be compatible with depth textures
        
        if format is None:
            return True
        
        format_kind_map = {
            StorageFormat.R8_UINT: TextureKind.UINT,
            StorageFormat.R8_SINT: TextureKind.SINT,
            StorageFormat.R8_UNORM: TextureKind.UNORM,
            StorageFormat.R32_UINT: TextureKind.UINT,
            StorageFormat.R32_SINT: TextureKind.SINT,
            StorageFormat.R32_FLOAT: TextureKind.FLOAT,
            # Add more mappings...
        }
        
        expected_kind = format_kind_map.get(format)
        if expected_kind is None:
            return True  # Unknown format, assume valid
        
        return expected_kind == kind
    
    def _determine_image_class(self, format: Optional[StorageFormat], kind: TextureKind) -> ImageClass:
        """Determine image class based on format and kind."""
        # TODO: Implement image class determination
        if kind == TextureKind.SHADOW:
            return ImageClass.DEPTH
        elif format is not None:
            return ImageClass.STORAGE
        else:
            return ImageClass.SAMPLER
    
    def parse_storage_image_type(self, tokens: List[str]) -> Optional[ImageType]:
        """
        Parse storage image type.
        
        Args:
            tokens: Image type tokens
            
        Returns:
            ImageType or None if parsing fails
        """
        # TODO: glsl support multisampled storage images, naga doesn't (строка 167)
        # This is about adding support for multisampled storage images in GLSL.
        # Naga currently doesn't support this feature.
        
        # Multisampled storage images would be:
        # - image2DMS, image2DMSArray
        # - Different sample counts (2, 4, 8, etc.)
        # - Special handling for sample operations
        
        # Check if this is a multisampled storage image
        if any("MS" in str(token) for token in tokens):
            # TODO: Add multisampled storage image support
            # This should:
            # 1. Parse MS suffix and sample count
            # 2. Create appropriate ImageType with multisampling
            # 3. Handle sample operations
            # 4. Validate multisampling support
            
            self.errors.append("Multisampled storage images not supported in naga")
            return None
        
        # Regular storage image parsing
        return self.parse_image_type(tokens)
    
    def validate_texture_type(self, image_type: ImageType) -> bool:
        """
        Validate a texture type.
        
        Args:
            image_type: Image type to validate
            
        Returns:
            True if valid
        """
        # TODO: Implement texture type validation
        # Should check:
        # 1. Valid dimension combinations
        # 2. Valid format-kind combinations
        # 3. Arrayed image constraints
        # 4. Multisampling support
        
        return True
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors.copy()
    
    def clear_errors(self) -> None:
        """Clear validation errors."""
        self.errors.clear()