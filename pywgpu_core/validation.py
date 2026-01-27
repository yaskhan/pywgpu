from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import pywgpu_types as wgt
from . import errors


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

    @classmethod
    def from_type(cls, ty: Any) -> "BindingTypeName":
        if hasattr(ty, 'Buffer'):
            return cls.Buffer
        if hasattr(ty, 'Texture'):
            return cls.Texture
        if hasattr(ty, 'Sampler'):
            return cls.Sampler
        if hasattr(ty, 'AccelerationStructure'):
            return cls.AccelerationStructure
        if hasattr(ty, 'ExternalTexture'):
            return cls.ExternalTexture
        raise ValueError(f"Unknown binding type: {ty}")


class ResourceType:
    """Base class for resource types used in validation."""
    pass


@dataclass
class BufferResourceType(ResourceType):
    size: int  # wgt::BufferSize


@dataclass
class TextureResourceType(ResourceType):
    dim: Any  # naga::ImageDimension
    arrayed: bool
    class_: Any  # naga::ImageClass


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

    @staticmethod
    def Scalar() -> "NumericDimension":
        return NumericDimension(type="scalar", size=1, rows=1)

    @staticmethod
    def Vector(size: Any) -> "NumericDimension":
        # size is naga.VectorSize
        return NumericDimension(type="vector", size=int(size), rows=1)

    @staticmethod
    def Matrix(columns: Any, rows: Any) -> "NumericDimension":
        # columns, rows are naga.VectorSize
        return NumericDimension(type="matrix", size=int(columns), rows=int(rows))

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
        """Create a NumericType from a vertex format.

        Mapping based on wgpu::VertexFormat
        Logic from validation.rs lines 814-865
        """
        import pywgpu_types as wgt
        from naga import Scalar, VectorSize

        vf = wgt.VertexFormat
        # Mapping based on wgpu::VertexFormat
        # Logic from validation.rs lines 814-865
        mapping = {
            # Unsigned integer formats
            vf.Uint8: (NumericDimension.Scalar(), Scalar.U32),
            vf.Uint8x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.U32),
            vf.Uint8x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),
            vf.Uint16: (NumericDimension.Scalar(), Scalar.U32),
            vf.Uint16x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.U32),
            vf.Uint16x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),
            vf.Uint32: (NumericDimension.Scalar(), Scalar.U32),
            vf.Uint32x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.U32),
            vf.Uint32x3: (NumericDimension.Vector(VectorSize.Tri), Scalar.U32),
            vf.Uint32x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),

            # Signed integer formats
            vf.Sint8: (NumericDimension.Scalar(), Scalar.I32),
            vf.Sint8x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.I32),
            vf.Sint8x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.I32),
            vf.Sint16: (NumericDimension.Scalar(), Scalar.I32),
            vf.Sint16x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.I32),
            vf.Sint16x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.I32),
            vf.Sint32: (NumericDimension.Scalar(), Scalar.I32),
            vf.Sint32x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.I32),
            vf.Sint32x3: (NumericDimension.Vector(VectorSize.Tri), Scalar.I32),
            vf.Sint32x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.I32),

            # Floating point formats
            vf.Unorm8: (NumericDimension.Scalar(), Scalar.F32),
            vf.Unorm8x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            vf.Unorm8x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Snorm8: (NumericDimension.Scalar(), Scalar.F32),
            vf.Snorm8x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            vf.Snorm8x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Unorm16: (NumericDimension.Scalar(), Scalar.F32),
            vf.Unorm16x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            vf.Unorm16x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Snorm16: (NumericDimension.Scalar(), Scalar.F32),
            vf.Snorm16x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            vf.Snorm16x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Float16: (NumericDimension.Scalar(), Scalar.F32),
            vf.Float16x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            vf.Float16x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Float32: (NumericDimension.Scalar(), Scalar.F32),
            vf.Float32x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            vf.Float32x3: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),
            vf.Float32x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Float64: (NumericDimension.Scalar(), Scalar.F64),
            vf.Float64x2: (NumericDimension.Vector(VectorSize.Bi), Scalar.F64),
            vf.Float64x3: (NumericDimension.Vector(VectorSize.Tri), Scalar.F64),
            vf.Float64x4: (NumericDimension.Vector(VectorSize.Quad), Scalar.F64),

            # Packed formats
            vf.Unorm10_10_10_2: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            vf.Unorm8x4Bgra: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),

            # RGB format
            vf.Rg11b10Ufloat: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),
        }

        # Default fallback
        dim, scalar = mapping.get(format, (NumericDimension.Scalar(), Scalar.F32))
        return cls(dim=dim, kind=scalar.kind, width=scalar.width)

    @classmethod
    def from_texture_format(cls, format: Any) -> NumericType:
        """Create a NumericType from a texture format.

        Logic from validation.rs lines 867-949
        """
        import pywgpu_types as wgt
        from naga import Scalar, VectorSize

        tf = wgt.TextureFormat
        # Mapping based on wgpu::TextureFormat
        # Logic from validation.rs lines 867-949
        mapping = {
            # Single channel formats
            tf.R8Unorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.R8Snorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.R16Float: (NumericDimension.Scalar(), Scalar.F32),
            tf.R32Float: (NumericDimension.Scalar(), Scalar.F32),
            tf.R8Uint: (NumericDimension.Scalar(), Scalar.U32),
            tf.R16Uint: (NumericDimension.Scalar(), Scalar.U32),
            tf.R32Uint: (NumericDimension.Scalar(), Scalar.U32),
            tf.R8Sint: (NumericDimension.Scalar(), Scalar.I32),
            tf.R16Sint: (NumericDimension.Scalar(), Scalar.I32),
            tf.R32Sint: (NumericDimension.Scalar(), Scalar.I32),
            tf.Rg11b10Ufloat: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),

            # Dual channel formats
            tf.Rg8Unorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Rg8Snorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Rg16Float: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Rg32Float: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Rg8Uint: (NumericDimension.Vector(VectorSize.Bi), Scalar.U32),
            tf.Rg16Uint: (NumericDimension.Vector(VectorSize.Bi), Scalar.U32),
            tf.Rg32Uint: (NumericDimension.Vector(VectorSize.Bi), Scalar.U32),
            tf.Rg8Sint: (NumericDimension.Vector(VectorSize.Bi), Scalar.I32),
            tf.Rg16Sint: (NumericDimension.Vector(VectorSize.Bi), Scalar.I32),
            tf.Rg32Sint: (NumericDimension.Vector(VectorSize.Bi), Scalar.I32),
            tf.R16Unorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.R16Snorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.Rg16Unorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Rg16Snorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),

            # Triple channel formats
            tf.Rgb9e5Ufloat: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),

            # Quad channel formats
            tf.Rgba8Unorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgba8UnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgba8Snorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bgra8Unorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bgra8UnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgb10a2Unorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgba16Float: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgba32Float: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgba8Uint: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),
            tf.Rgba16Uint: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),
            tf.Rgba32Uint: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),
            tf.Rgb10a2Uint: (NumericDimension.Vector(VectorSize.Quad), Scalar.U32),
            tf.Rgba8Sint: (NumericDimension.Vector(VectorSize.Quad), Scalar.I32),
            tf.Rgba16Sint: (NumericDimension.Vector(VectorSize.Quad), Scalar.I32),
            tf.Rgba32Sint: (NumericDimension.Vector(VectorSize.Quad), Scalar.I32),
            tf.Rgba16Unorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Rgba16Snorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.R64Uint: (NumericDimension.Scalar(), Scalar.U64),

            # Compressed formats
            tf.Bc1RgbaUnorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc1RgbaUnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc2RgbaUnorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc2RgbaUnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc3RgbaUnorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc3RgbaUnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc4RUnorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.Bc4RSnorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.Bc5RgUnorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Bc5RgSnorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.Bc6hRgbUfloat: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),
            tf.Bc6hRgbFloat: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),
            tf.Bc7RgbaUnorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Bc7RgbaUnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Etc2Rgb8Unorm: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),
            tf.Etc2Rgb8UnormSrgb: (NumericDimension.Vector(VectorSize.Tri), Scalar.F32),
            tf.Etc2Rgb8A1Unorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Etc2Rgb8A1UnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Etc2Rgba8Unorm: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.Etc2Rgba8UnormSrgb: (NumericDimension.Vector(VectorSize.Quad), Scalar.F32),
            tf.EacR11Unorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.EacR11Snorm: (NumericDimension.Scalar(), Scalar.F32),
            tf.EacRg11Unorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
            tf.EacRg11Snorm: (NumericDimension.Vector(VectorSize.Bi), Scalar.F32),
        }

        # Default fallback
        dim, scalar = mapping.get(format, (NumericDimension.Scalar(), Scalar.F32))
        return cls(dim=dim, kind=scalar.kind, width=scalar.width)

    def is_subtype_of(self, other: NumericType) -> bool:
        """Check if this type is a subtype of another.

        Logic from validation.rs lines 951-967
        """
        # Logic from validation.rs lines 951-967
        if self.width > other.width:
            return False
        if self.kind != other.kind:
            return False

        # Check dimension compatibility
        if isinstance(self.dim, NumericDimension.Scalar) and isinstance(other.dim, NumericDimension.Scalar):
            return True
        elif isinstance(self.dim, NumericDimension.Scalar) and isinstance(other.dim, NumericDimension.Vector):
            return True
        elif isinstance(self.dim, NumericDimension.Vector) and isinstance(other.dim, NumericDimension.Vector):
            return self.dim.size <= other.dim.size
        elif isinstance(self.dim, NumericDimension.Matrix) and isinstance(other.dim, NumericDimension.Matrix):
            return self.dim.columns == other.dim.columns and self.dim.rows == other.dim.rows
        else:
            return False


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
    bind: Any  # naga::ResourceBinding
    ty: ResourceType
    class_: Any  # naga::AddressSpace

    def check_binding_use(self, entry: Any) -> None:
        """Check if binding use is valid.

        Logic from validation.rs lines 550-724
        """
        # Logic from validation.rs lines 550-724
        if isinstance(self.ty, BufferResourceType):
            # Check buffer binding
            min_size = None
            if hasattr(entry, 'ty'):
                binding_type = entry.ty
                if hasattr(binding_type, 'Buffer'):
                    # BindingType::Buffer
                    ty_data = binding_type.Buffer
                    buffer_ty = ty_data.ty
                    has_dynamic_offset = ty_data.has_dynamic_offset
                    min_binding_size = ty_data.min_binding_size

                    # Determine address space
                    from pywgpu_types import BufferBindingType
                    if buffer_ty == BufferBindingType.Uniform:
                        class_ = wgt.AddressSpace.Uniform
                    elif buffer_ty == BufferBindingType.Storage:
                        read_only = hasattr(buffer_ty, 'read_only') and buffer_ty.read_only
                        import naga
                        naga_access = naga.StorageAccess.LOAD
                        if not read_only:
                            naga_access |= naga.StorageAccess.STORE
                        class_ = wgt.AddressSpace.Storage(naga_access)
                    else:
                        class_ = None

                    if self.class_ != class_:
                        raise BindingError_WrongAddressSpace(
                            binding=class_,
                            shader=self.class_
                        )

                    min_size = min_binding_size

            if min_size is None:
                raise BindingError_WrongType(
                    binding=BindingTypeName.Buffer,
                    shader=BindingTypeName.Buffer
                )

            # Check buffer size
            if min_size is not None and min_size < self.ty.size:
                raise BindingError_WrongBufferSize(
                    buffer_size=self.ty.size,
                    min_binding_size=min_size
                )

        elif isinstance(self.ty, SamplerResourceType):
            # Check sampler binding
            if hasattr(entry, 'ty') and hasattr(entry.ty, 'Sampler'):
                ty = entry.ty.Sampler
                from pywgpu_types import SamplerBindingType
                is_comparison = (ty == SamplerBindingType.Comparison)
                if is_comparison != self.ty.comparison:
                    raise BindingError_WrongSamplerComparison()
            else:
                raise BindingError_WrongType(
                    binding=BindingTypeName.Sampler,
                    shader=BindingTypeName.Sampler
                )

        elif isinstance(self.ty, TextureResourceType):
            # Check texture/storage texture binding
            dim = self.ty.dim
            arrayed = self.ty.arrayed
            class_ = self.ty.class_

            # Get view_dimension from entry
            view_dimension = None
            if hasattr(entry, 'ty'):
                binding_type = entry.ty
                if hasattr(binding_type, 'Texture'):
                    view_dimension = binding_type.Texture.view_dimension
                elif hasattr(binding_type, 'StorageTexture'):
                    view_dimension = binding_type.StorageTexture.view_dimension
                elif hasattr(binding_type, 'ExternalTexture'):
                    view_dimension = wgt.TextureViewDimension.D2
                else:
                    raise BindingError_WrongTextureViewDimension(
                        dim=dim,
                        is_array=arrayed,
                        binding=binding_type
                    )
            else:
                raise BindingError_WrongTextureViewDimension(
                    dim=dim,
                    is_array=arrayed,
                    binding=entry
                )

            # Check dimension compatibility
            from pywgpu_types import TextureViewDimension
            if arrayed:
                if not ((dim == wgt.ImageDimension.D2 and view_dimension == TextureViewDimension.D2Array) or
                        (dim == wgt.ImageDimension.Cube and view_dimension == TextureViewDimension.CubeArray)):
                    raise BindingError_WrongTextureViewDimension(
                        dim=dim,
                        is_array=True,
                        binding=entry
                    )
            else:
                if not ((dim == wgt.ImageDimension.D1 and view_dimension == TextureViewDimension.D1) or
                        (dim == wgt.ImageDimension.D2 and view_dimension == TextureViewDimension.D2) or
                        (dim == wgt.ImageDimension.D3 and view_dimension == TextureViewDimension.D3) or
                        (dim == wgt.ImageDimension.Cube and view_dimension == TextureViewDimension.Cube)):
                    raise BindingError_WrongTextureViewDimension(
                        dim=dim,
                        is_array=False,
                        binding=entry
                    )

            # Determine expected class and check
            expected_class = None
            if hasattr(entry, 'ty'):
                binding_type = entry.ty
                if hasattr(binding_type, 'Texture'):
                    ty_data = binding_type.Texture
                    sample_type = ty_data.sample_type
                    multi = ty_data.multisampled

                    if hasattr(sample_type, 'Float'):
                        expected_class = wgt.ImageClass.Sampled(
                            kind=wgt.ScalarKind.Float,
                            multi=multi
                        )
                    elif hasattr(sample_type, 'Sint'):
                        expected_class = wgt.ImageClass.Sampled(
                            kind=wgt.ScalarKind.Sint,
                            multi=multi
                        )
                    elif hasattr(sample_type, 'Uint'):
                        expected_class = wgt.ImageClass.Sampled(
                            kind=wgt.ScalarKind.Uint,
                            multi=multi
                        )
                    elif hasattr(sample_type, 'Depth'):
                        expected_class = wgt.ImageClass.Depth(multi=multi)
                elif hasattr(binding_type, 'StorageTexture'):
                    ty_data = binding_type.StorageTexture
                    access = ty_data.access
                    format = ty_data.format
                    view_dimension = ty_data.view_dimension

                    naga_format = map_storage_format_to_naga(format)
                    if naga_format is None:
                        raise BindingError_BadStorageFormat(format)

                    import naga
                    naga_access = naga.StorageAccess.LOAD
                    from pywgpu_types import StorageTextureAccess
                    if access == StorageTextureAccess.ReadOnly:
                        naga_access = naga.StorageAccess.LOAD
                    elif access == StorageTextureAccess.WriteOnly:
                        naga_access = naga.StorageAccess.STORE
                    elif access == StorageTextureAccess.ReadWrite:
                        naga_access = naga.StorageAccess.LOAD | naga.StorageAccess.STORE
                    elif access == StorageTextureAccess.Atomic:
                        naga_access = (naga.StorageAccess.ATOMIC |
                                       naga.StorageAccess.LOAD |
                                       naga.StorageAccess.STORE)

                    expected_class = wgt.ImageClass.Storage(
                        format=naga_format,
                        access=naga_access
                    )
                elif hasattr(binding_type, 'ExternalTexture'):
                    expected_class = wgt.ImageClass.External

            if expected_class is not None and class_ != expected_class:
                raise BindingError_WrongTextureClass(
                    binding=expected_class,
                    shader=class_
                )

        elif isinstance(self.ty, AccelerationStructureResourceType):
            # Check acceleration structure binding
            if hasattr(entry, 'ty') and hasattr(entry.ty, 'AccelerationStructure'):
                ty_data = entry.ty.AccelerationStructure
                entry_vertex_return = ty_data.vertex_return
                if self.ty.vertex_return != entry_vertex_return:
                    raise BindingError_WrongType(
                        binding=BindingTypeName.AccelerationStructure,
                        shader=BindingTypeName.AccelerationStructure
                    )
            else:
                raise BindingError_WrongType(
                    binding=BindingTypeName.AccelerationStructure,
                    shader=BindingTypeName.AccelerationStructure
                )

    def derive_binding_type(self, is_reffed_by_sampler: bool) -> Any:
        """Derive binding type from resource.

        Logic from validation.rs lines 726-811
        """
        # Logic from validation.rs lines 726-811
        if isinstance(self.ty, BufferResourceType):
            size = self.ty.size
            from pywgpu_types import BufferBindingType, BindingType

            # Determine binding type based on address space
            ty = None
            if self.class_ == wgt.AddressSpace.Uniform:
                ty = BufferBindingType.Uniform
            elif isinstance(self.class_, wgt.AddressSpace.Storage):
                read_only = self.class_.access == wgt.StorageAccess.LOAD
                ty = BufferBindingType.Storage(read_only=read_only)
            else:
                raise BindingError_WrongBufferAddressSpace(space=self.class_)

            return wgt.BindingType.Buffer(
                ty=ty,
                has_dynamic_offset=False,
                min_binding_size=size
            )

        elif isinstance(self.ty, SamplerResourceType):
            from pywgpu_types import SamplerBindingType, BindingType
            return wgt.BindingType.Sampler(
                SamplerBindingType.Comparison if self.ty.comparison else SamplerBindingType.Filtering
            )

        elif isinstance(self.ty, TextureResourceType):
            dim = self.ty.dim
            arrayed = self.ty.arrayed
            class_ = self.ty.class_
            from pywgpu_types import TextureViewDimension, BindingType, TextureSampleType

            # Determine view dimension
            view_dimension = None
            if dim == wgt.ImageDimension.D1:
                view_dimension = TextureViewDimension.D1
            elif dim == wgt.ImageDimension.D2:
                if arrayed:
                    view_dimension = TextureViewDimension.D2Array
                else:
                    view_dimension = TextureViewDimension.D2
            elif dim == wgt.ImageDimension.D3:
                view_dimension = TextureViewDimension.D3
            elif dim == wgt.ImageDimension.Cube:
                if arrayed:
                    view_dimension = TextureViewDimension.CubeArray
                else:
                    view_dimension = TextureViewDimension.Cube

            # Determine binding type based on image class
            if isinstance(class_, wgt.ImageClass.Sampled):
                multi = class_.multi
                kind = class_.kind

                if kind == wgt.ScalarKind.Float:
                    sample_type = TextureSampleType.Float(filterable=is_reffed_by_sampler)
                elif kind == wgt.ScalarKind.Sint:
                    sample_type = TextureSampleType.Sint
                elif kind == wgt.ScalarKind.Uint:
                    sample_type = TextureSampleType.Uint
                else:
                    raise ValueError(f"Unexpected scalar kind: {kind}")

                return wgt.BindingType.Texture(
                    sample_type=sample_type,
                    view_dimension=view_dimension,
                    multisampled=multi
                )
            elif isinstance(class_, wgt.ImageClass.Depth):
                multi = class_.multi
                return wgt.BindingType.Texture(
                    sample_type=TextureSampleType.Depth,
                    view_dimension=view_dimension,
                    multisampled=multi
                )
            elif isinstance(class_, wgt.ImageClass.Storage):
                format = class_.format
                access = class_.access

                from pywgpu_types import StorageTextureAccess
                naga_access = wgt.StorageAccess.LOAD | wgt.StorageAccess.STORE

                if access == wgt.StorageAccess.LOAD:
                    wgpu_access = StorageTextureAccess.ReadOnly
                elif access == wgt.StorageAccess.STORE:
                    wgpu_access = StorageTextureAccess.WriteOnly
                elif access == naga_access:
                    wgpu_access = StorageTextureAccess.ReadWrite
                elif access & wgt.StorageAccess.ATOMIC:
                    wgpu_access = StorageTextureAccess.Atomic
                else:
                    raise ValueError(f"Unexpected storage access: {access}")

                # Convert format
                wgpu_format = map_storage_format_from_naga(format)

                return wgt.BindingType.StorageTexture(
                    access=wgpu_access,
                    view_dimension=view_dimension,
                    format=wgpu_format
                )
            elif isinstance(class_, wgt.ImageClass.External):
                return wgt.BindingType.ExternalTexture

        elif isinstance(self.ty, AccelerationStructureResourceType):
            from pywgpu_types import BindingType
            return wgt.BindingType.AccelerationStructure(
                vertex_return=self.ty.vertex_return
            )

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
        """Validate a shader stage.

        Logic from validation.rs lines 1218-1643
        """
        pair = (shader_stage, entry_point_name)
        entry_point = self.entry_points.get(pair)
        if not entry_point:
            raise StageError_MissingEntryPoint(entry_point_name)

        # Check resources visibility and compatibility
        for handle in entry_point.resources:
            # Handle is index or object in Python
            res = self.resources[handle] if isinstance(handle, int) else handle

            result = None
            # Logic from validation.rs lines 1237-1307
            if hasattr(layouts, 'Provided'):
                # Provided layout - validate against it
                # Update binding size for buffers
                if isinstance(res.ty, BufferResourceType):
                    size = res.ty.size
                    if res.bind in shader_binding_sizes:
                        shader_binding_sizes[res.bind] = max(shader_binding_sizes[res.bind], size)
                    else:
                        shader_binding_sizes[res.bind] = size

                # Get layout entry
                try:
                    map_obj = layouts.get(res.bind.group)
                except (IndexError, AttributeError):
                    result = BindingError_Missing()

                if result is None:
                    entry = map_obj.get(res.bind.binding) if hasattr(map_obj, 'get') else None
                    if entry is None:
                        result = BindingError_Missing()
                    else:
                        # Check visibility
                        from pywgpu_types import ShaderStage
                        stage_bit = shader_stage.to_wgt_bit()
                        if not entry.visibility & stage_bit:
                            result = BindingError_Invisible()

                        # Check binding use
                        if result is None:
                            try:
                                res.check_binding_use(entry)
                            except BindingError as e:
                                result = e

            elif hasattr(layouts, 'Derived'):
                # Derived layout - build it from shader
                try:
                    map_obj = layouts.get(res.bind.group)
                except (IndexError, AttributeError):
                    result = BindingError_Missing()

                if result is None:
                    # Derive binding type
                    is_reffed = any(
                        texture_handle == handle
                        for (texture_handle, _) in entry_point.sampling_pairs
                    )
                    try:
                        ty = res.derive_binding_type(is_reffed)
                    except BindingError as e:
                        result = e

                    if result is None:
                        # Insert into layout map
                        from pywgpu_types import BindGroupLayoutEntry, ShaderStage
                        stage_bit = shader_stage.to_wgt_bit()

                        if hasattr(map_obj, '__contains__') and res.bind.binding in map_obj:
                            # Check consistency
                            if map_obj[res.bind.binding].ty != ty:
                                result = BindingError_WrongType(
                                    binding=BindingTypeName.from_type(ty),
                                    shader=BindingTypeName.from_type(map_obj[res.bind.binding].ty)
                                )
                            else:
                                # Update visibility
                                map_obj[res.bind.binding].visibility |= stage_bit
                        else:
                            # Add new entry
                            map_obj[res.bind.binding] = wgt.BindGroupLayoutEntry(
                                binding=res.bind.binding,
                                ty=ty,
                                visibility=stage_bit,
                                count=None
                            )

            if result is not None:
                raise StageError_Binding(res.bind, result)

        # Check texture/sampler compatibility (filtering)
        # Logic from validation.rs lines 1309-1355
        if hasattr(layouts, 'Provided'):
            for texture_handle, sampler_handle in entry_point.sampling_pairs:
                texture_bind = self.resources[texture_handle if isinstance(texture_handle, int) else 0].bind
                sampler_bind = self.resources[sampler_handle if isinstance(sampler_handle, int) else 0].bind

                texture_layout = layouts.get(texture_bind.group).get(texture_bind.binding)
                sampler_layout = layouts.get(sampler_bind.group).get(sampler_bind.binding)

                from pywgpu_types import SamplerBindingType, TextureSampleType
                sampler_filtering = (
                    hasattr(sampler_layout.ty, 'Sampler') and
                    sampler_layout.ty.Sampler == SamplerBindingType.Filtering
                )

                texture_sample_type = None
                if hasattr(texture_layout.ty, 'Texture'):
                    texture_sample_type = texture_layout.ty.Texture.sample_type
                elif hasattr(texture_layout.ty, 'ExternalTexture'):
                    texture_sample_type = TextureSampleType.Float(filterable=True)

                # Check filtering compatibility
                error = None
                if sampler_filtering:
                    if hasattr(texture_sample_type, 'Float') and not texture_sample_type.Float.filterable:
                        error = FilteringError_Float()
                    elif hasattr(texture_sample_type, 'Sint'):
                        error = FilteringError_Integer()
                    elif hasattr(texture_sample_type, 'Uint'):
                        error = FilteringError_Integer()

                if error:
                    raise StageError_Filtering(
                        texture=texture_bind,
                        sampler=sampler_bind,
                        error=error
                    )

        # Check inputs compatibility
        # Logic from validation.rs lines 1419-1485
        for input_var in entry_point.inputs:
            if hasattr(input_var, 'Local'):
                location = input_var.Local.location
                iv = input_var.Local.iv

                # Get provided input
                provided = inputs.get(location) if hasattr(inputs, 'get') else None

                if provided is None:
                    raise StageError_Input(
                        location=location,
                        var=iv,
                        error=InputError_Missing()
                    )

                # Check compatibility based on shader stage
                from naga import ShaderStage
                compatible = False
                per_primitive_correct = False

                if shader_stage == ShaderStage.Vertex:
                    # For vertex attributes, defaults are filled by driver
                    compatible = iv.ty.scalar.kind == provided.ty.scalar.kind
                    per_primitive_correct = not iv.per_primitive
                elif shader_stage == ShaderStage.Fragment:
                    # Fragment shader needs exact match
                    if iv.interpolation != provided.interpolation:
                        raise StageError_Input(
                            location=location,
                            var=iv,
                            error=InputError_InterpolationMismatch(provided.interpolation)
                        )

                    if iv.sampling != provided.sampling:
                        raise StageError_Input(
                            location=location,
                            var=iv,
                            error=InputError_SamplingMismatch(provided.sampling)
                        )

                    compatible = iv.ty.is_subtype_of(provided.ty)
                    per_primitive_correct = iv.per_primitive == provided.per_primitive
                else:
                    # Compute, Task, Mesh can't have varying inputs
                    compatible = False
                    per_primitive_correct = False

                if not compatible:
                    raise StageError_Input(
                        location=location,
                        var=iv,
                        error=InputError_WrongType(provided.ty)
                    )

                if not per_primitive_correct:
                    raise StageError_Input(
                        location=location,
                        var=iv,
                        error=InputError_WrongPerPrimitive(
                            pipeline_input=provided.per_primitive,
                            shader=iv.per_primitive
                        )
                    )

        return None  # Success


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
