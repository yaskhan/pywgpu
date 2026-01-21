"""
Backend for SPIR-V (Standard Portable Intermediate Representation).

This module contains the writer implementation for converting Naga IR
to SPIR-V binary format. This is a simplified implementation focusing
on the structure and basic functionality.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntEnum, IntFlag
import io

from ...error import ShaderError


class Capability(IntFlag):
    """SPIR-V capability flags."""

    Matrix = 1
    Shader = 256
    Geometry = 512
    Tessellation = 1024
    Address = 2048
    Float16 = 4096
    Float64 = 8192
    Int64 = 16384
    Int64Atomics = 32768
    ImageFloat32Controls = 65536
    DenormPreserve = 131072
    DenormFtz = 262144
    SignedZeroInfNanPreserve = 524288
    RoundingModeRTE = 1048576
    RoundingModeRTZ = 2097152
    RayTracing = 2097152
    AtomicStorageOps = 524288
    SampleMaskPostDepthCoverage = 1048576
    Image1D = 2097152
    ImageBufferArray = 4194304
    ImageRect = 8388608
    SampledRect = 16777216
    GenericPointer = 33554432
    Int8 = 67108864
    InputAttachment = 134217728
    SparseResidency = 268435456
    MinLod = 536870912
    Sampled1D = 1073741824
    ImageCubeArray = 2147483648
    Int16 = 4294967296
    TessellationPointSize = 8589934592
    GeometryPointSize = 17179869184
    ImageGatherExtended = 34359738368
    StorageImageMultisample = 68719476736
    UniformBufferArrayDynamicIndexing = 137438953472
    SampledBufferArrayDynamicIndexing = 274877906944
    StorageBufferArrayDynamicIndexing = 549755813888
    StorageImageArrayDynamicIndexing = 1099511627776
    ClipDistance = 2199023255552
    CullDistance = 4398046511104
    SampleRateShading = 17592186044416


class AddressingModel(IntEnum):
    """SPIR-V addressing model."""

    Logical = 0
    Physical32 = 1
    Physical64 = 2
    PhysicalStorageBuffer64 = 5348


class MemoryModel(IntEnum):
    """SPIR-V memory model."""

    Simple = 0
    Glsl450 = 1
    Vulkan = 2
    OpenCL = 3


class ExecutionModel(IntEnum):
    """SPIR-V execution model."""

    Vertex = 0
    TessellationControl = 1
    TessellationEvaluation = 2
    Geometry = 3
    Fragment = 4
    GLCompute = 5
    Kernel = 6
    Task = 5267
    Mesh = 5268
    RayGeneration = 5313
    Intersection = 5314
    AnyHit = 5315
    ClosestHit = 5316
    Miss = 5317
    Callable = 5318


class SpvStorageClass(IntEnum):
    """SPIR-V storage class."""

    UniformConstant = 0
    Input = 1
    Uniform = 2
    Output = 3
    Workgroup = 4
    CrossWorkgroup = 5
    Private = 6
    Function = 7
    Generic = 8
    PushConstant = 9
    AtomicCounter = 10
    Image = 11
    StorageBuffer = 12
    TileImage = 13
    RayPayload = 5338
    HitAttribute = 5339
    CallableData = 5344
    ScatterData = 5345


class SpvOp(IntEnum):
    """Basic SPIR-V opcodes."""

    OpNop = 0
    OpUndef = 1
    OpSourceContinued = 2
    OpSource = 3
    OpSourceExtension = 4
    OpExtension = 5
    OpExtInstImport = 6
    OpExtInst = 7
    OpMemoryModel = 8
    OpEntryPoint = 9
    OpExecutionMode = 10
    OpCapability = 11
    OpTypeVoid = 19
    OpTypeBool = 20
    OpTypeInt = 21
    OpTypeFloat = 22
    OpTypeVector = 23
    OpTypeMatrix = 24
    OpTypeImage = 25
    OpTypeSampledImage = 26
    OpTypeSampler = 27
    OpTypeAccelerationStructure = 46
    OpTypeStruct = 28
    OpTypePointer = 32
    OpTypeFunction = 33
    OpConstant = 43
    OpConstantComposite = 44
    OpVariable = 59
    OpLoad = 61
    OpStore = 62
    OpAccessChain = 65
    OpInBoundsAccessChain = 66
    OpPtrAccessChain = 67
    OpInBoundsPtrAccessChain = 68
    OpCompositeExtract = 79
    OpCompositeInsert = 80
    OpVectorExtractDynamic = 81
    OpVectorInsertDynamic = 82
    OpCompositeConstruct = 83
    OpConvertPtrToU = 187
    OpReinterpret = 189
    OpSelect = 190
    OpIAdd = 136
    OpFAdd = 137
    OpISub = 138
    OpFSub = 139
    OpIMul = 140
    OpFMul = 141
    OpUDiv = 142
    OpSDiv = 143
    OpFDiv = 144
    OpUMod = 144
    OpSMod = 144
    OpFMod = 144
    OpVectorTimesScalar = 162
    OpMatrixTimesScalar = 163
    OpVectorTimesMatrix = 164
    OpMatrixTimesVector = 165
    OpMatrixTimesMatrix = 166
    OpOuterProduct = 167
    OpDot = 168
    OpIAddCarry = 169
    OpISubBorrow = 170
    OpUMulExtended = 171
    OpSMulExtended = 172
    OpAny = 158
    OpAll = 159
    OpISign = 160
    OpFSign = 161
    OpFOrdEqual = 174
    OpFUnordEqual = 175
    OpFOrdNotEqual = 176
    OpFUnordNotEqual = 177
    OpFOrdLessThan = 178
    OpFUnordLessThan = 179
    OpFOrdGreaterThan = 180
    OpFUnordGreaterThan = 181
    OpFOrdLessThanEqual = 182
    OpFUnordLessThanEqual = 183
    OpFOrdGreaterThanEqual = 184
    OpFUnordGreaterThanEqual = 185
    OpFOrdTrue = 186
    OpFOrdFalse = 187
    OpFUnordTrue = 188
    OpFUnordFalse = 189
    OpIEqual = 42
    OpINotEqual = 40
    OpULessThan = 36
    OpSLessThan = 37
    OpUGreaterThan = 38
    OpSGreaterThan = 39
    OpULessThanEqual = 41
    OpSLessThanEqual = 42
    OpUGreaterThanEqual = 43
    OpSGreaterThanEqual = 44
    OpLogicalEqual = 49
    OpLogicalNotEqual = 50
    OpLogicalOr = 51
    OpLogicalAnd = 52
    OpLogicalNot = 53


class Options:
    """SPIR-V writer options."""

    def __init__(
        self,
        addressing_model: AddressingModel = AddressingModel.Logical,
        memory_model: MemoryModel = MemoryModel.Vulkan,
    ):
        """
        Initialize SPIR-V writer options.

        Args:
            addressing_model: SPIR-V addressing model
            memory_model: SPIR-V memory model
        """
        self.addressing_model = addressing_model
        self.memory_model = memory_model
        self.capabilities: Set[Capability] = set()
        self.operations: List[int] = []


class Instruction:
    """Represents a SPIR-V instruction."""

    def __init__(self, opcode: SpvOp, operands: List[int] = None):
        self.opcode = opcode
        self.operands = operands or []

    def to_words(self) -> List[int]:
        """Convert instruction to SPIR-V words."""
        word_count = len(self.operands) + 1
        result = [word_count | 0xE0000000]  # Add instruction format bits
        result.append(int(self.opcode))
        result.extend(self.operands)
        return result


class SpvType:
    """Represents a SPIR-V type."""

    def __init__(self, kind: str, name: str, parameters: List[Any] = None):
        self.kind = kind
        self.name = name
        self.parameters = parameters or []
        self.id = None

    def __str__(self):
        return f"Type({self.kind})"


class SpvVariable:
    """Represents a SPIR-V variable."""

    def __init__(self, name: str, type_id: int, storage_class: SpvStorageClass):
        self.name = name
        self.type_id = type_id
        self.storage_class = storage_class
        self.id = None


class Writer:
    """
    Writer for converting Naga IR modules to SPIR-V binary format.

    Maintains internal state to output a Module into SPIR-V format.
    """

    def __init__(self, module: Any, info: Any, options: Options):
        """
        Initialize the SPIR-V writer.

        Args:
            module: The Naga IR module
            info: Module validation information
            options: SPIR-V writer options
        """
        self.module = module
        self.info = info
        self.options = options

        # Internal state
        self.types: Dict[str, SpvType] = {}
        self.variables: Dict[str, SpvVariable] = {}
        self.instructions: List[Instruction] = []
        self.next_id = 1

        # Type mapping cache
        self.type_cache: Dict[str, int] = {}

    def write(self) -> bytes:
        """
        Write the complete module to SPIR-V binary.

        Returns:
            SPIR-V binary data

        Raises:
            ShaderError: If writing fails
        """
        try:
            self._initialize_basic_types()
            self._analyze_module()
            self._generate_types()
            self._generate_variables()
            self._generate_instructions()

            return self._create_spirv_binary()

        except Exception as e:
            raise ShaderError(f"SPIR-V writing failed: {e}") from e

    def _initialize_basic_types(self) -> None:
        """Initialize basic SPIR-V types."""
        # Void type
        self.types["void"] = SpvType("void", "void")

        # Boolean type
        self.types["bool"] = SpvType("bool", "bool")

        # Integer types
        self.types["i32"] = SpvType("int", "i32", [32])
        self.types["u32"] = SpvType("int", "u32", [32])
        self.types["i16"] = SpvType("int", "i16", [16])
        self.types["u16"] = SpvType("int", "u16", [16])
        self.types["i8"] = SpvType("int", "i8", [8])
        self.types["u8"] = SpvType("int", "u8", [8])
        self.types["i64"] = SpvType("int", "i64", [64])
        self.types["u64"] = SpvType("int", "u64", [64])

        # Floating point types
        self.types["f32"] = SpvType("float", "f32", [32])
        self.types["f64"] = SpvType("float", "f64", [64])
        self.types["f16"] = SpvType("float", "f16", [16])

    def _analyze_module(self) -> None:
        """Analyze module to determine required types and capabilities."""
        # Placeholder implementation
        # Would analyze module types, functions, etc.
        pass

    def _generate_types(self) -> None:
        """Generate SPIR-V type definitions."""
        for type_name, spv_type in self.types.items():
            self._emit_type_definition(spv_type)

    def _emit_type_definition(self, spv_type: SpvType) -> None:
        """Emit a type definition instruction."""
        operands = []

        if spv_type.kind == "int":
            operands.append(0)  # Signedness (0 = unsigned, 1 = signed)
            operands.extend(spv_type.parameters)
        elif spv_type.kind == "float":
            operands.extend(spv_type.parameters)
        elif spv_type.kind == "vector":
            operands.append(spv_type.parameters[0])  # Component type ID
            operands.append(spv_type.parameters[1])  # Component count
        elif spv_type.kind == "matrix":
            operands.append(spv_type.parameters[0])  # Column type ID
            operands.append(spv_type.parameters[1])  # Column count
        elif spv_type.kind == "struct":
            operands.extend(spv_type.parameters)  # Member type IDs
        elif spv_type.kind == "pointer":
            operands.append(spv_type.parameters[0])  # Storage class
            operands.append(spv_type.parameters[1])  # Pointed-to type ID
        elif spv_type.kind == "function":
            operands.append(spv_type.parameters[0])  # Return type ID
            operands.extend(spv_type.parameters[1:])  # Parameter type IDs

        opcode_map = {
            "void": SpvOp.OpTypeVoid,
            "bool": SpvOp.OpTypeBool,
            "int": SpvOp.OpTypeInt,
            "float": SpvOp.OpTypeFloat,
            "vector": SpvOp.OpTypeVector,
            "matrix": SpvOp.OpTypeMatrix,
            "struct": SpvOp.OpTypeStruct,
            "pointer": SpvOp.OpTypePointer,
            "function": SpvOp.OpTypeFunction,
        }

        opcode = opcode_map.get(spv_type.kind, SpvOp.OpTypeVoid)
        instruction = Instruction(opcode, operands)
        instruction.id = self._allocate_id()
        spv_type.id = instruction.id

        self.instructions.append(instruction)

    def _generate_variables(self) -> None:
        """Generate SPIR-V variable declarations."""
        for var_name, spv_var in self.variables.items():
            self._emit_variable_declaration(spv_var)

    def _emit_variable_declaration(self, spv_var: SpvVariable) -> None:
        """Emit a variable declaration instruction."""
        operands = [spv_var.storage_class.value]

        instruction = Instruction(SpvOp.OpVariable, operands)
        instruction.id = self._allocate_id()
        spv_var.id = instruction.id

        self.instructions.append(instruction)

    def _generate_instructions(self) -> None:
        """Generate function and instruction definitions."""
        # Placeholder implementation
        # Would generate actual function bodies, expressions, etc.
        pass

    def _create_spirv_binary(self) -> bytes:
        """Create the final SPIR-V binary."""
        # Header: Magic number, Version, Generator, Bound, Schema
        header = [
            0x07230203,  # SPIR-V magic number
            0x00010000,  # Version 1.0
            0x00000000,  # Generator (Naga)
            self.next_id,  # Maximum ID bound
            0,  # Schema
        ]

        # Convert instructions to words
        words = header.copy()
        for instruction in self.instructions:
            words.extend(instruction.to_words())

        # Convert to bytes
        return b"".join(word.to_bytes(4, "little") for word in words)

    def _allocate_id(self) -> int:
        """Allocate a new SPIR-V ID."""
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def _get_type_id(self, type_name: str) -> int:
        """Get or create a type ID for the given type name."""
        if type_name in self.type_cache:
            return self.type_cache[type_name]

        if type_name not in self.types:
            # Create missing types as needed
            if type_name.startswith("vec"):
                # Vector type
                comp_count = int(type_name[3:])
                comp_type = "f32" if "f" in type_name else "i32"
                self.types[type_name] = SpvType(
                    "vector", type_name, [self._get_type_id(comp_type), comp_count]
                )
            elif type_name.startswith("mat"):
                # Matrix type
                col_count = int(type_name[3])
                comp_type = "f32" if "f" in type_name else "i32"
                self.types[type_name] = SpvType(
                    "matrix", type_name, [self._get_type_id(comp_type), col_count]
                )
            else:
                # Fallback to generic type
                self.types[type_name] = SpvType("struct", type_name, [])

        spv_type = self.types[type_name]
        if spv_type.id is None:
            self._emit_type_definition(spv_type)

        self.type_cache[type_name] = spv_type.id
        return spv_type.id


def write_binary(module: Any, info: Any, options: Options) -> bytes:
    """
    Write a module to SPIR-V binary.

    Args:
        module: The Naga IR module
        info: Module validation info
        options: SPIR-V writer options

    Returns:
        Generated SPIR-V binary data
    """
    writer = Writer(module, info, options)
    return writer.write()
