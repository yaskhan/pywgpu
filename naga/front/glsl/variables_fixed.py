from ...ir import (
    AddressSpace,
    ScalarKind,
    Interpolation,
    Sampling,
    StorageAccess,
    Binding,
    GlobalVariable,
)
from ...span import Span

from typing import Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass


# Replace local enums with IR imports (already done above)


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
        """
        # Determine address space from qualifiers
        space = AddressSpace.PRIVATE # Default
        if qualifiers:
             space = self._determine_address_space(qualifiers)
             
        return VariableDeclaration(
            name=name,
            ty=ty,
            init=init,
            meta={
                'qualifiers': qualifiers,
                'space': space,
                **(meta if isinstance(meta, dict) else {})
            }
        )
    
    def add_global_variable(self, ctx: Any, declaration: VariableDeclaration) -> Any:
        # Extract storage information from meta
        meta = declaration.meta or {}
        qualifiers = meta.get('qualifiers') or []
        
        from .token import TokenValue
        qualifier_values = [t.value for t in qualifiers if hasattr(t, 'value')]
        
        if TokenValue.IN in qualifier_values:
            return self._add_input_output_variable(declaration, "input", ctx)
        elif TokenValue.OUT in qualifier_values:
            return self._add_input_output_variable(declaration, "output", ctx)
        else:
            return self._add_regular_variable(declaration, qualifiers, ctx)
    
    def _add_input_output_variable(self, declaration: VariableDeclaration, storage: str, ctx: Any) -> Any:
        """Add input/output variable."""
        # Extract location from layout qualifiers
        meta = declaration.meta or {}
        layout = meta.get('layout') or {}
        location = layout.get('location')
        
        if location is None:
            location = self._get_next_location_counter()
        
        # Get interpolation and sampling from qualifiers
        qualifiers = meta.get('qualifiers') or []
        interpolation = None
        sampling = None
        
        from ...ir import Interpolation as IRInterpolation, Sampling as IRSampling, GlobalVariable
        
        for q in qualifiers:
            # Handle both token objects and potential raw values
            data = getattr(q, 'data', None)
            if isinstance(data, IRInterpolation):
                interpolation = data
            elif isinstance(data, IRSampling):
                sampling = data
        
        if interpolation is None:
            interpolation = self._get_default_interpolation(declaration.ty)

        # Create global variable
        space = AddressSpace.IN if storage == "input" else AddressSpace.OUT
        global_var = GlobalVariable(
            name=declaration.name,
            space=space, # Inputs/outputs are private in NAGA, mapped to EntryPoint args
            binding=None,
            ty=declaration.ty,
            init=declaration.init
        )
        
        # Create entry argument info
        binding = self._create_location_binding(location, interpolation, sampling)
        
        # We'll store both in the handler for now
        handle = len(self.global_variables)
        self.global_variables.append(global_var)
        
        # Update location counter for next variable
        # For now simple increment, should ideally account for type size
        # self._location_counter += self._get_type_location_size(declaration.ty)
        
        # storage to identify if input or output
        self.entry_args.append({
            'name': declaration.name,
            'binding': binding,
            'handle': handle,
            'storage': storage
        })
        
        return global_var
    
    def _add_regular_variable(self, declaration: VariableDeclaration, qualifiers: List[Any], ctx: Any) -> Any:
        """Add uniform, buffer, shared or const variable."""
        space = self._determine_address_space(qualifiers)
        
        # Extract binding/group from layout
        meta = declaration.meta or {}
        layout = meta.get('layout') or {}
        binding_idx = layout.get('binding')
        set_idx = layout.get('set', 0)
        
        from ...ir import GlobalVariable
        
        binding = None
        if binding_idx is not None:
            binding = {"group": set_idx, "binding": binding_idx}
            
        global_var = GlobalVariable(
            name=declaration.name,
            space=space,
            binding=binding,
            ty=declaration.ty,
            init=declaration.init
        )
        
        handle = len(self.global_variables)
        self.global_variables.append(global_var)
        
        return global_var
    
    def _determine_address_space(self, qualifiers: List[Any]) -> AddressSpace:
        """Determine address space from storage qualifiers."""
        from .token import TokenValue
        
        kinds = [t.value for t in qualifiers if hasattr(t, 'value')]
        
        if TokenValue.UNIFORM in kinds:
            return AddressSpace.UNIFORM
        elif TokenValue.BUFFER in kinds:
            return AddressSpace.STORAGE
        elif TokenValue.SHARED in kinds:
            return AddressSpace.WORKGROUP
        elif TokenValue.CONST in kinds:
            return AddressSpace.PRIVATE
        else:
            return AddressSpace.PRIVATE
    
    def _get_default_interpolation(self, ty: Any) -> Interpolation:
        """Get default interpolation for a type."""
        from ...ir import ScalarKind as IRScalarKind, Interpolation as IRInterpolation
        scalar_kind = self._get_scalar_kind(ty)
        if scalar_kind == IRScalarKind.FLOAT:
            return IRInterpolation.PERSPECTIVE
        else:
            return IRInterpolation.FLAT
    
    def _get_scalar_kind(self, ty_handle: Any) -> ScalarKind:
        """Get scalar kind from type handle."""
        # This requires access to the module but we only have handles
        # For now, we'll return a default or we need to pass frontend
        from ...ir import ScalarKind as IRScalarKind
        return IRScalarKind.FLOAT
    
    def _create_location_binding(self, location: int, interpolation: Optional[Interpolation], sampling: Optional[Sampling]) -> Any:
        """Create location binding."""
        return Binding.new_location(
            location=location,
            interpolation=interpolation,
            sampling=sampling
        )
    
    def _get_next_location_counter(self) -> int:
        if not hasattr(self, '_location_counter'):
            self._location_counter = 0
        
        current = self._location_counter
        # Increment by 1 slot (simple assumption for now)
        self._location_counter += 1
        return current
    
    def parse_image_variable(self, qualifiers: Any, name: str, dim: Any, arrayed: bool, ty: Any, meta: Any) -> Any:
        """
        Parse image variable declaration.
        """
        format_qualifier = self._extract_format_qualifier(qualifiers)
        
        if format_qualifier is None:
            if self._is_writeonly_image(qualifiers):
                from ...ir import StorageFormat
                format_qualifier = StorageFormat.RGBA8UNORM
            else:
                self.errors.append("image types require a format layout qualifier")
                return None
        
        # Create global variable for the image
        from ...ir import GlobalVariable, AddressSpace
        
        image_var = GlobalVariable(
            name=name,
            space=AddressSpace.HANDLE,
            binding=None, # To be filled by layout parser
            ty=ty,
            init=None
        )
        
        handle = len(self.global_variables)
        self.global_variables.append(image_var)
        
        # Store metadata for format and other image properties
        return {
            'name': name,
            'handle': handle,
            'format': format_qualifier,
            'meta': meta
        }
    
    def _extract_format_qualifier(self, qualifiers: Any) -> Optional[Any]:
        """Extract format qualifier from layout qualifiers."""
        # Qualifiers can be a list of tokens or a meta dict with 'layout'
        if isinstance(qualifiers, dict):
            layout = qualifiers.get('layout', {})
        else:
            # Try to get layout from meta if passed a declaration
            meta = getattr(qualifiers, 'meta', {})
            layout = meta.get('layout', {})
            
        # Common image formats in GLSL
        from ...ir import StorageFormat
        formats = {
            "rgba32f": StorageFormat.RGBA32FLOAT,
            "rgba16f": StorageFormat.RGBA16FLOAT,
            "rg32f": StorageFormat.RG32FLOAT,
            "rg16f": StorageFormat.RG16FLOAT,
            "r32f": StorageFormat.R32FLOAT,
            "r16f": StorageFormat.R16FLOAT,
            "rgba32ui": StorageFormat.RGBA32UINT,
            "rgba16ui": StorageFormat.RGBA16UINT,
            "rgba8ui": StorageFormat.RGBA8UINT,
            "rg32ui": StorageFormat.RG32UINT,
            "rg16ui": StorageFormat.RG16UINT,
            "rg8ui": StorageFormat.RG8UINT,
            "r32ui": StorageFormat.R32UINT,
            "r16ui": StorageFormat.R16UINT,
            "r8ui": StorageFormat.R8UINT,
            "rgba32i": StorageFormat.RGBA32SINT,
            "rgba16i": StorageFormat.RGBA16SINT,
            "rgba8i": StorageFormat.RGBA8SINT,
            "rg32i": StorageFormat.RG32SINT,
            "rg16i": StorageFormat.RG16SINT,
            "rg8i": StorageFormat.RG8SINT,
            "r32i": StorageFormat.R32SINT,
            "r16i": StorageFormat.R16SINT,
            "r8i": StorageFormat.R8SINT,
            "rgba8": StorageFormat.RGBA8UNORM,
            "rgba8_snorm": StorageFormat.RGBA8SNORM,
            "bgra8": StorageFormat.BGRA8UNORM,
        }
        
        for key in layout:
            if key in formats:
                return formats[key]
        return None
    
    def _is_writeonly_image(self, qualifiers: Any) -> bool:
        """Check if image is writeonly."""
        from .token import TokenValue
        qualifier_values = []
        if isinstance(qualifiers, list):
            qualifier_values = [t.value for t in qualifiers if hasattr(t, 'value')]
        
        return TokenValue.MEMORY_QUALIFIER in qualifier_values # Simplified
    
    def _create_image_variable(self, name: str, dim: Any, arrayed: bool, ty: Any, format_qualifier: Any, meta: Any) -> Any:
        """Create image variable with format qualifier."""
        from ...ir import GlobalVariable, AddressSpace, Type, TypeInner, Image
        
        # Image in NAGA is a TypeInner
        image_inner = TypeInner.new_image(
            dim=dim,
            arrayed=arrayed,
            class_info=None # Storage image
        )
        # Actually it should be a Storage image class
        
        return {
            'name': name,
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
    
