from enum import Enum, auto

@dataclass
class ResourceErrorIdent:
    """Information about a wgpu-core resource for error messages."""
    type: str
    label: str = ""

    def __str__(self) -> str:
        if self.label:
            return f"{self.type} with '{self.label}' label"
        return self.type


class BindingTypeName(Enum):
    Buffer = auto()
    Texture = auto()
    Sampler = auto()
    AccelerationStructure = auto()
    ExternalTexture = auto()


@dataclass
class ResourceType:
    """Base class for resource types used in validation."""
    pass

@dataclass
class BufferResourceType(ResourceType):
    size: int # wgt::BufferSize

@dataclass
class TextureResourceType(ResourceType):
    dim: Any # naga::ImageDimension
    arrayed: bool
    class_: Any # naga::ImageClass

@dataclass
class SamplerResourceType(ResourceType):
    comparison: bool

@dataclass
class AccelerationStructureResourceType(ResourceType):
    vertex_return: bool


class BindingError(errors.WebGpuError):
    """Error related to binding validation."""
    def __init__(self, message: str):
        super().__init__(message)
    
    @property
    def webgpu_error_type(self) -> errors.ErrorType:
        return errors.ErrorType.Validation

class BindingError_Missing(BindingError):
    def __init__(self): super().__init__("Binding is missing from the pipeline layout")

class BindingError_Invisible(BindingError):
    def __init__(self): super().__init__("Visibility flags don't include the shader stage")

class BindingError_WrongType(BindingError):
    def __init__(self, binding: BindingTypeName, shader: BindingTypeName):
        super().__init__(f"Type on the shader side ({shader.name}) does not match the pipeline binding ({binding.name})")

class BindingError_WrongAddressSpace(BindingError):
    def __init__(self, binding: Any, shader: Any):
        super().__init__(f"Storage class {binding} doesn't match the shader {shader}")

class BindingError_WrongBufferSize(BindingError):
    def __init__(self, buffer_size: int, min_binding_size: int):
        super().__init__(f"Buffer structure size {buffer_size} ended up greater than the given `min_binding_size` {min_binding_size}")

class BindingError_WrongTextureViewDimension(BindingError):
    def __init__(self, dim: Any, is_array: bool, binding: Any):
        super().__init__(f"View dimension {dim} (is array: {is_array}) doesn't match the binding {binding}")

class BindingError_WrongTextureClass(BindingError):
    def __init__(self, binding: Any, shader: Any):
        super().__init__(f"Texture class {binding} doesn't match the shader {shader}")

class BindingError_WrongSamplerComparison(BindingError):
    def __init__(self): super().__init__("Comparison flag doesn't match the shader")

class BindingError_BadStorageFormat(BindingError):
    def __init__(self, format: Any): super().__init__(f"Texture format {format} is not supported for storage use")


class FilteringError(errors.WebGpuError):
    """Error related to texture filtering."""
    @property
    def webgpu_error_type(self) -> errors.ErrorType:
        return errors.ErrorType.Validation

class FilteringError_Integer(FilteringError):
    def __init__(self): super().__init__("Integer textures can't be sampled with a filtering sampler")

class FilteringError_Float(FilteringError):
    def __init__(self): super().__init__("Non-filterable float textures can't be sampled with a filtering sampler")


class InputError(errors.WebGpuError):
    """Error related to shader input validation."""
    @property
    def webgpu_error_type(self) -> errors.ErrorType:
        return errors.ErrorType.Validation

class InputError_Missing(InputError):
    def __init__(self): super().__init__("Input is not provided by the earlier stage in the pipeline")

class InputError_WrongType(InputError):
    def __init__(self, ty: Any): super().__init__(f"Input type is not compatible with the provided {ty}")

class InputError_InterpolationMismatch(InputError):
    def __init__(self, interpolation: Any): super().__init__(f"Input interpolation doesn't match provided {interpolation}")

class InputError_SamplingMismatch(InputError):
    def __init__(self, sampling: Any): super().__init__(f"Input sampling doesn't match provided {sampling}")

class InputError_WrongPerPrimitive(InputError):
    def __init__(self, pipeline_input: bool, shader: bool):
        super().__init__(f"Pipeline input has per_primitive={pipeline_input}, but shader expects per_primitive={shader}")


class StageError(errors.WebGpuError):
    """Error related to shader stage validation."""
    @property
    def webgpu_error_type(self) -> errors.ErrorType:
        return errors.ErrorType.Validation

class StageError_MissingEntryPoint(StageError):
    def __init__(self, name: str): super().__init__(f"Unable to find entry point '{name}'")

class StageError_Binding(StageError):
    def __init__(self, binding: Any, error: BindingError):
        super().__init__(f"Shader global {binding} is not available in the pipeline layout: {error}")

class StageError_Filtering(StageError):
    def __init__(self, texture: Any, sampler: Any, error: FilteringError):
        super().__init__(f"Unable to filter the texture ({texture}) by the sampler ({sampler}): {error}")

class StageError_Input(StageError):
    def __init__(self, location: int, var: Any, error: InputError):
        super().__init__(f"Location[{location}] {var} is not provided by the previous stage outputs: {error}")


@dataclass(frozen=True)
class NumericDimension:
    """Numeric dimension for type validation."""
    # Matches Rust's NumericDimension enum
    type: str = "scalar" # scalar, vector, matrix
    size: int = 1 # for vector size or matrix columns
    rows: int = 1 # for matrix rows

    def __str__(self) -> str:
        if self.type == "scalar": return ""
        if self.type == "vector": return f"x{self.size}"
        if self.type == "matrix": return f"x{self.size}{self.rows}"
        return ""

@dataclass(frozen=True)
class NumericType:
    """Numeric type for validation."""
    dim: NumericDimension
    kind: str # float, sint, uint, etc.
    width: int # 1, 2, 4, 8 bytes

    def __str__(self) -> str:
        return f"{self.kind}{self.width * 8}{self.dim}"

    @classmethod
    def from_vertex_format(cls, format: Any) -> NumericType:
        """Create a NumericType from a vertex format."""
        # Mapping based on wgpu::VertexFormat
        # Logic from validation.rs lines 814-865
        import pywgpu_types as wgt
        vf = wgt.VertexFormat
        
        # This is a simplified mapping for now
        mapping = {
            vf.Uint8: ("scalar", "uint", 1),
            vf.Uint32: ("scalar", "uint", 4),
            vf.Float32: ("scalar", "float", 4),
            vf.Float32x4: ("vector", "float", 4), # size=4
            # ... and so on. TODO: Complete full mapping
        }
        # Default fallback
        res = mapping.get(format, ("scalar", "float", 4))
        return cls(dim=NumericDimension(type=res[0], size=res[2] if res[0] != "scalar" else 1), kind=res[1], width=res[2] if res[0] == "scalar" else 4)

    @classmethod
    def from_texture_format(cls, format: Any) -> NumericType:
        """Create a NumericType from a texture format."""
        # Logic from validation.rs lines 867-949
        return cls(dim=NumericDimension(), kind="float", width=4) # Placeholder

    def is_subtype_of(self, other: NumericType) -> bool:
        """Check if this type is a subtype of another."""
        # Logic from validation.rs lines 951-967
        if self.kind != other.kind or self.width > other.width:
            return False
        # Simplified dimension check
        return True # TODO: Implement full dimension subtype check


@dataclass
class InterfaceVar:
    """Interface variable for validation."""
    ty: NumericType
    interpolation: Optional[Any] = None
    sampling: Optional[Any] = None
    per_primitive: bool = False

    def __str__(self) -> str:
        return f"{self.ty} interpolated as {self.interpolation} with sampling {self.sampling}"


@dataclass
class Varying:
    """Varying for validation."""
    # Matches Rust's Varying enum
    location: Optional[int] = None
    iv: Optional[InterfaceVar] = None
    builtin: Optional[Any] = None


@dataclass
class EntryPointMeshInfo:
    max_vertices: int
    max_primitives: int


@dataclass
class EntryPoint:
    """Entry point for validation."""
    inputs: List[Varying]
    outputs: List[Varying]
    resources: List[Any] # Handles to Resource
    sampling_pairs: List[tuple[Any, Any]] = None # (Texture, Sampler)
    workgroup_size: tuple[int, int, int] = (0, 0, 0)
    dual_source_blending: bool = False
    task_payload_size: Optional[int] = None
    mesh_info: Optional[EntryPointMeshInfo] = None


@dataclass
class Resource:
    """Resource definition for validation."""
    name: Optional[str]
    bind: Any # naga::ResourceBinding
    ty: ResourceType
    class_: Any # naga::AddressSpace

    def check_binding_use(self, entry: Any) -> None:
        """Check if binding use is valid."""
        # Logic from validation.rs lines 550-724
        if isinstance(self.ty, BufferResourceType):
            # Check buffer binding
            # Simplified version of Rust logic
            pass
        elif isinstance(self.ty, SamplerResourceType):
            # Check sampler binding
            pass
        elif isinstance(self.ty, TextureResourceType):
            # Check texture/storage texture binding
            pass
        # TODO: Implement full exhaustive check for all resource types

    def derive_binding_type(self, is_reffed_by_sampler: bool) -> Any:
        """Derive binding type from resource."""
        # Logic from validation.rs lines 726-811
        # TODO: Implement full binding type derivation logic
        return None


@dataclass
class Interface:
    """
    Shader interface for validation.
    """
    limits: Any
    resources: List[Resource] # Simplified arena
    entry_points: dict[tuple[Any, str], EntryPoint] # (stage, name) -> EntryPoint

    def check_stage(
        self,
        layouts: Any,
        shader_binding_sizes: dict,
        entry_point_name: str,
        shader_stage: Any,
        inputs: Any
    ) -> Any:
        """Validate a shader stage."""
        # Logic from validation.rs lines 1218-1643
        pair = (shader_stage, entry_point_name)
        entry_point = self.entry_points.get(pair)
        if not entry_point:
            raise StageError_MissingEntryPoint(entry_point_name)

        # 1. Check resources visibility and compatibility
        for handle in entry_point.resources:
            # Handle is index or object in Python
            res = self.resources[handle] if isinstance(handle, int) else handle
            
            # Simple visibility and layout check
            # TODO: Implement layouts.get(...) and visibility bit checks
            try:
                res.check_binding_use(None) # Placeholder entry
            except BindingError as e:
                raise StageError_Binding(res.bind, e)

        # 2. Check texture/sampler compatibility (filtering)
        # TODO: Implement sampling_pairs check

        # 3. Check inputs compatibility
        # TODO: Implement inputs compatibility check loop

        return None # Success


class ShaderModule:
    """
    A compiled shader module for validation.
    """

    def __init__(self, interface: Optional[Interface] = None, label: str = "") -> None:
        """Initialize the shader module."""
        self.interface = interface
        self.label = label

    def finalize_entry_point_name(
        self,
        stage: Any,
        entry_point: Optional[str],
    ) -> str:
        """
        Finalize the entry point name.
        """
        if self.interface is None:
            if entry_point is None:
                raise StageError_MissingEntryPoint("None")
            return entry_point

        # Logic from validation.rs lines 1195-1214
        if entry_point is not None:
            return entry_point
        
        # Look for unique entry point for stage
        eps = [name for (stg, name) in self.interface.entry_points.keys() if stg == stage]
        if not eps:
            raise StageError_MissingEntryPoint("No entry points found for stage")
        if len(eps) > 1:
            raise StageError("Multiple entry points found but none specified")
        
        return eps[0]
def map_storage_format_to_naga(format: Any) -> Optional[Any]:
    """
    Map a wgpu texture format to a naga storage format.
    Logic from validation.rs lines 437-492.
    """
    import pywgpu_types as wgt
    tf = wgt.TextureFormat
    # Naga formats would be strings or enums in a Python wrapper
    # For now we use the descriptive names as used in Naga
    mapping = {
        tf.R8Unorm: "R8Unorm",
        tf.R8Snorm: "R8Snorm",
        tf.R8Uint: "R8Uint",
        tf.R8Sint: "R8Sint",
        tf.R16Uint: "R16Uint",
        tf.R16Sint: "R16Sint",
        tf.R16Float: "R16Float",
        tf.Rg8Unorm: "Rg8Unorm",
        tf.Rg8Snorm: "Rg8Snorm",
        tf.Rg8Uint: "Rg8Uint",
        tf.Rg8Sint: "Rg8Sint",
        tf.R32Uint: "R32Uint",
        tf.R32Sint: "R32Sint",
        tf.R32Float: "R32Float",
        tf.Rg16Uint: "Rg16Uint",
        tf.Rg16Sint: "Rg16Sint",
        tf.Rg16Float: "Rg16Float",
        tf.Rgba8Unorm: "Rgba8Unorm",
        tf.Rgba8Snorm: "Rgba8Snorm",
        tf.Rgba8Uint: "Rgba8Uint",
        tf.Rgba8Sint: "Rgba8Sint",
        tf.Bgra8Unorm: "Bgra8Unorm",
        tf.Rgb10a2Uint: "Rgb10a2Uint",
        tf.Rgb10a2Unorm: "Rgb10a2Unorm",
        tf.Rg11b10Ufloat: "Rg11b10Ufloat",
        tf.R64Uint: "R64Uint",
        tf.Rg32Uint: "Rg32Uint",
        tf.Rg32Sint: "Rg32Sint",
        tf.Rg32Float: "Rg32Float",
        tf.Rgba16Uint: "Rgba16Uint",
        tf.Rgba16Sint: "Rgba16Sint",
        tf.Rgba16Float: "Rgba16Float",
        tf.Rgba32Uint: "Rgba32Uint",
        tf.Rgba32Sint: "Rgba32Sint",
        tf.Rgba32Float: "Rgba32Float",
    }
    return mapping.get(format)


def map_storage_format_from_naga(format: Any) -> Optional[Any]:
    """
    Map a naga storage format to a wgpu texture format.
    Logic from validation.rs lines 494-547.
    """
    import pywgpu_types as wgt
    tf = wgt.TextureFormat
    mapping = {
        "R8Unorm": tf.R8Unorm,
        "R8Snorm": tf.R8Snorm,
        "R8Uint": tf.R8Uint,
        "R8Sint": tf.R8Sint,
        "R16Uint": tf.R16Uint,
        "R16Sint": tf.R16Sint,
        "R16Float": tf.R16Float,
        "Rg8Unorm": tf.Rg8Unorm,
        "Rg8Snorm": tf.Rg8Snorm,
        "Rg8Uint": tf.Rg8Uint,
        "Rg8Sint": tf.Rg8Sint,
        "R32Uint": tf.R32Uint,
        "R32Sint": tf.R32Sint,
        "R32Float": tf.R32Float,
        "Rg16Uint": tf.Rg16Uint,
        "Rg16Sint": tf.Rg16Sint,
        "Rg16Float": tf.Rg16Float,
        "Rgba8Unorm": tf.Rgba8Unorm,
        "Rgba8Snorm": tf.Rgba8Snorm,
        "Rgba8Uint": tf.Rgba8Uint,
        "Rgba8Sint": tf.Rgba8Sint,
        "Bgra8Unorm": tf.Bgra8Unorm,
        "Rgb10a2Uint": tf.Rgb10a2Uint,
        "Rgb10a2Unorm": tf.Rgb10a2Unorm,
        "Rg11b10Ufloat": tf.Rg11b10Ufloat,
        "R64Uint": tf.R64Uint,
        "Rg32Uint": tf.Rg32Uint,
        "Rg32Sint": tf.Rg32Sint,
        "Rg32Float": tf.Rg32Float,
        "Rgba16Uint": tf.Rgba16Uint,
        "Rgba16Sint": tf.Rgba16Sint,
        "Rgba16Float": tf.Rgba16Float,
        "Rgba32Uint": tf.Rgba32Uint,
        "Rgba32Sint": tf.Rgba32Sint,
        "Rgba32Float": tf.Rgba32Float,
    }
    return mapping.get(format)


def check_texture_format(format: Any, output: NumericType) -> Optional[NumericType]:
    """
    Return None if the texture format is covered by the provided output.
    Return the texture's numeric type otherwise.
    Logic from validation.rs lines 971-981.
    """
    nt = NumericType.from_texture_format(format)
    if nt.is_subtype_of(output):
        return None
    return nt
