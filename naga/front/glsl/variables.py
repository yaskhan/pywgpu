"""
GLSL variable handling and declaration processing.

This module handles variable declarations, global variables, constants,
and their storage qualifiers in GLSL shaders.
"""

from typing import Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass


class StorageQualifier(Enum):
    """Storage qualifier types for variables."""
    ADDRESS_SPACE = "address_space"
    INPUT = "input"
    OUTPUT = "output"
    CONST = "const"


class AddressSpace(Enum):
    """Address space types."""
    FUNCTION = "function"
    PRIVATE = "private"
    WORKGROUP = "workgroup"
    UNIFORM = "uniform"
    STORAGE = "storage"
    HANDLE = "handle"
    PUBLIC = "public"


class ScalarKind(Enum):
    """Scalar kind types."""
    BOOL = "bool"
    FLOAT = "float"
    DOUBLE = "double"
    INT = "int"
    UINT = "uint"


class Interpolation(Enum):
    """Interpolation qualifier types."""
    PERSPECTIVE = "perspective"
    LINEAR = "linear"
    FLAT = "flat"


class Sampling(Enum):
    """Sampling qualifier types."""
    CENTER = "center"
    CENTROID = "centroid"
    SAMPLE = "sample"


@dataclass
class VariableDeclaration:
    """Variable declaration information."""
    name: Optional[str]
    ty: Any
    init: Optional[Any]
    meta: Any


@dataclass
class GlobalVariableInfo:
    """Global variable information."""
    name: Optional[str]
    space: AddressSpace
    binding: Optional[Any]
    ty: Any
    init: Optional[Any]


@dataclass
class EntryArg:
    """Entry point argument information."""
    name: Optional[str]
    binding: Any
    handle: Any
    storage: StorageQualifier


class VariableHandler:
    """Handler for GLSL variable declarations."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.entry_args: List[EntryArg] = []
        self.global_variables: List[GlobalVariableInfo] = []
    
    def parse_variable_declaration(self, qualifiers: Any, name: str, ty: Any, init: Optional[Any], meta: Any) -> Any:
        """
        Parse a variable declaration.
        
        Args:
            qualifiers: Variable qualifiers
            name: Variable name
            ty: Variable type
            init: Optional initializer
            meta: Metadata
            
        Returns:
            Parsed variable declaration
        """
        # TODO: Implement variable declaration parsing
        # Should handle different storage qualifiers and address spaces
        
        return VariableDeclaration(
            name=name,
            ty=ty,
            init=init,
            meta=meta
        )
    
    def add_global_variable(self, ctx: Any, declaration: VariableDeclaration) -> Any:
        """
        Add a global variable to the module.
        
        Args:
            ctx: Parsing context
            declaration: Variable declaration
            
        Returns:
            Global variable handle or lookup information
        """
        # Extract storage information from qualifiers
        storage = declaration.meta.get('storage', StorageQualifier.ADDRESS_SPACE)
        
        if storage in [StorageQualifier.INPUT, StorageQualifier.OUTPUT]:
            return self._add_input_output_variable(declaration, storage, ctx)
        else:
            return self._add_regular_variable(declaration, storage, ctx)
    
    def _add_input_output_variable(self, declaration: VariableDeclaration, storage: StorageQualifier, ctx: Any) -> Any:
        """
        Add input/output variable (vertex shader input, fragment shader output, etc.).
        
        Args:
            declaration: Variable declaration
            storage: Storage qualifier (INPUT or OUTPUT)
            ctx: Parsing context
            
        Returns:
            Variable handle or lookup
        """
        input_var = (storage == StorageQualifier.INPUT)
        
        # TODO: glslang seems to use a counter for variables without (строка 430)
        # explicit location (even if that causes collisions)
        # The original comment mentions that glslang uses a counter for variables
        # without explicit location layout qualifiers, even if this causes location
        # collisions. We need to implement similar behavior.
        
        location = declaration.meta.get('location', 0)
        if location is None:
            # Use auto-generated location counter
            # TODO: Implement location counter for variables without explicit location
            location = self._get_next_location_counter()
        
        interpolation = self._get_default_interpolation(declaration.ty)
        sampling = declaration.meta.get('sampling')
        
        # Create global variable
        global_var = GlobalVariableInfo(
            name=declaration.name,
            space=AddressSpace.PRIVATE,
            binding=None,
            ty=declaration.ty,
            init=declaration.init
        )
        
        # Add to global variables
        handle = len(self.global_variables)
        self.global_variables.append(global_var)
        
        # Create entry argument
        entry_arg = EntryArg(
            name=declaration.name,
            binding=self._create_location_binding(location, interpolation, sampling),
            handle=handle,
            storage=storage
        )
        
        self.entry_args.append(entry_arg)
        
        return handle
    
    def _add_regular_variable(self, declaration: VariableDeclaration, storage: StorageQualifier, ctx: Any) -> Any:
        """
        Add regular global variable (uniform, buffer, etc.).
        
        Args:
            declaration: Variable declaration
            storage: Storage qualifier
            ctx: Parsing context
            
        Returns:
            Variable handle or lookup
        """
        # TODO: Implement regular variable addition
        # Should handle:
        # - Uniform variables
        # - Buffer variables
        # - Private variables
        # - Workgroup variables
        
        address_space = self._determine_address_space(storage, declaration.meta)
        
        global_var = GlobalVariableInfo(
            name=declaration.name,
            space=address_space,
            binding=declaration.meta.get('binding'),
            ty=declaration.ty,
            init=declaration.init
        )
        
        handle = len(self.global_variables)
        self.global_variables.append(global_var)
        
        return handle
    
    def _determine_address_space(self, storage: StorageQualifier, meta: Any) -> AddressSpace:
        """
        Determine address space from storage qualifier.
        
        Args:
            storage: Storage qualifier
            meta: Variable metadata
            
        Returns:
            Address space
        """
        # TODO: Implement address space determination
        # Should map storage qualifiers to address spaces:
        # - const -> PRIVATE
        # - uniform -> UNIFORM
        # - buffer -> STORAGE
        # - workgroup -> WORKGROUP
        # - handle -> HANDLE
        
        if storage == StorageQualifier.CONST:
            return AddressSpace.PRIVATE
        elif 'uniform' in str(meta).lower():
            return AddressSpace.UNIFORM
        elif 'buffer' in str(meta).lower():
            return AddressSpace.STORAGE
        elif 'workgroup' in str(meta).lower():
            return AddressSpace.WORKGROUP
        else:
            return AddressSpace.PRIVATE
    
    def _get_default_interpolation(self, ty: Any) -> Interpolation:
        """
        Get default interpolation for a type.
        
        Args:
            ty: Variable type
            
        Returns:
            Default interpolation
        """
        # TODO: Implement default interpolation determination
        # Should return:
        # - Interpolation.PERSPECTIVE for float types
        # - Interpolation.FLAT for integer types
        
        scalar_kind = self._get_scalar_kind(ty)
        if scalar_kind == ScalarKind.FLOAT:
            return Interpolation.PERSPECTIVE
        else:
            return Interpolation.FLAT
    
    def _get_scalar_kind(self, ty: Any) -> ScalarKind:
        """
        Get scalar kind from type.
        
        Args:
            ty: Type to analyze
            
        Returns:
            Scalar kind
        """
        # TODO: Implement scalar kind extraction
        # Should determine the scalar kind from type information
        return ScalarKind.FLOAT
    
    def _create_location_binding(self, location: int, interpolation: Optional[Interpolation], sampling: Optional[Sampling]) -> Any:
        """
        Create location binding for input/output variables.
        
        Args:
            location: Location index
            interpolation: Interpolation qualifier
            sampling: Sampling qualifier
            
        Returns:
            Location binding
        """
        # TODO: Implement location binding creation
        # Should create binding information with location, interpolation, sampling
        return {
            'location': location,
            'interpolation': interpolation,
            'sampling': sampling,
            'blend_src': None,
            'per_primitive': False
        }
    
    def _get_next_location_counter(self) -> int:
        """
        Get next location counter for variables without explicit location.
        
        Returns:
            Next location index
        """
        # TODO: glslang seems to use a counter for variables without
        # explicit location (even if that causes collisions)
        # This should implement the same behavior as glslang
        if not hasattr(self, '_location_counter'):
            self._location_counter = 0
        else:
            self._location_counter += 1
        return self._location_counter
    
    def parse_image_variable(self, qualifiers: Any, name: str, dim: Any, arrayed: bool, ty: Any, meta: Any) -> Any:
        """
        Parse image variable declaration.
        
        Args:
            qualifiers: Variable qualifiers
            name: Variable name
            dim: Image dimension
            arrayed: Whether image is arrayed
            ty: Image type
            meta: Metadata
            
        Returns:
            Parsed image variable
        """
        # TODO: glsl supports images without format qualifier (строка 575)
        # if they are `writeonly`
        # The original comment mentions that GLSL supports images without format
        # qualifiers if they are writeonly. Naga currently requires format
        # qualifiers for all image types.
        
        format_qualifier = self._extract_format_qualifier(qualifiers)
        
        if format_qualifier is None:
            # Check if this is a writeonly image (format not required)
            if self._is_writeonly_image(qualifiers):
                # Use default format for writeonly images
                format_qualifier = self._get_default_writeonly_format()
            else:
                # Require format qualifier for non-writeonly images
                self.errors.append("image types require a format layout qualifier")
                return None
        
        # TODO: Add support for images without format qualifier for writeonly images
        # This should:
        # 1. Allow writeonly images to omit format qualifiers
        # 2. Use appropriate default formats for writeonly images
        # 3. Report errors for non-writeonly images without format qualifiers
        
        return self._create_image_variable(name, dim, arrayed, ty, format_qualifier, meta)
    
    def _extract_format_qualifier(self, qualifiers: Any) -> Optional[Any]:
        """Extract format qualifier from variable qualifiers."""
        # TODO: Implement format qualifier extraction
        # Should extract format from layout qualifiers
        return None
    
    def _is_writeonly_image(self, qualifiers: Any) -> bool:
        """Check if image is writeonly."""
        # TODO: Implement writeonly check
        # Should check memory qualifiers for writeonly/readonly
        return False
    
    def _get_default_writeonly_format(self) -> Any:
        """Get default format for writeonly images."""
        # TODO: Implement default writeonly format
        # Should return appropriate format for writeonly images
        return None
    
    def _create_image_variable(self, name: str, dim: Any, arrayed: bool, ty: Any, format_qualifier: Any, meta: Any) -> Any:
        """Create image variable with format qualifier."""
        # TODO: Implement image variable creation
        # Should create proper image type with format qualifier
        
        return {
            'name': name,
            'dim': dim,
            'arrayed': arrayed,
            'ty': ty,
            'format': format_qualifier,
            'meta': meta
        }
    
    def get_entry_args(self) -> List[EntryArg]:
        """Get entry point arguments."""
        return self.entry_args.copy()
    
    def get_global_variables(self) -> List[GlobalVariableInfo]:
        """Get global variables."""
        return self.global_variables.copy()
    
    def get_errors(self) -> List[str]:
        """Get variable parsing errors."""
        return self.errors.copy()
    
    def clear_errors(self) -> None:
        """Clear variable parsing errors."""
        self.errors.clear()