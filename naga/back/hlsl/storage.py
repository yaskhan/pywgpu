"""
Storage buffer access support for HLSL backend.

Generating accesses to ByteAddressBuffer contents.

Naga IR globals in Storage address space are rendered as ByteAddressBuffers or
RWByteAddressBuffers in HLSL. These buffers don't have HLSL types (structs,
arrays, etc.); instead, they are just raw blocks of bytes, with methods to
load and store values of specific types at particular byte offsets.

To generate code for a Storage access:
- Call fill_access_chain on expression referring to value. This populates
  temp_access_chain with appropriate byte offset calculations.
- Call write_storage_address to emit an HLSL expression for a given slice
  of SubAccess values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import io

if TYPE_CHECKING:
    from ...ir.module import Module
    from ...ir.expression import Expression
    from ...ir.type import Type, TypeInner
    from ...ir import Handle


STORE_TEMP_NAME = "_value"


class SubAccessType(Enum):
    """Types of sub-access operations."""

    BUFFER_OFFSET = 0
    OFFSET = 1
    INDEX = 2


@dataclass
class SubAccess:
    """One step in accessing a Storage global's component or element.

    Describes how to compute byte offset of a particular element or member
    of some global variable in Storage address space.
    """

    type: SubAccessType
    group: Optional[int] = None
    offset: Optional[int] = None
    value: Optional[Handle[Expression]] = None
    stride: Optional[int] = None

    @staticmethod
    def buffer_offset(group: int, offset: int) -> "SubAccess":
        """Create a buffer offset access."""
        return SubAccess(
            type=SubAccessType.BUFFER_OFFSET,
            group=group,
            offset=offset
        )

    @staticmethod
    def offset(offset: int) -> "SubAccess":
        """Create a constant offset access."""
        return SubAccess(
            type=SubAccessType.OFFSET,
            offset=offset
        )

    @staticmethod
    def index(value: Handle[Expression], stride: int) -> "SubAccess":
        """Create a dynamic index access."""
        return SubAccess(
            type=SubAccessType.INDEX,
            value=value,
            stride=stride
        )


class StoreValueType(Enum):
    """Types of values that can be stored."""

    EXPRESSION = 0
    ZERO = 1


@dataclass
class StoreValue:
    """A value to store in a storage buffer."""

    type: StoreValueType
    expression: Optional[Handle[Expression]] = None

    @staticmethod
    def from_expression(expr: Handle[Expression]) -> "StoreValue":
        """Create a store value from an expression."""
        return StoreValue(
            type=StoreValueType.EXPRESSION,
            expression=expr
        )

    @staticmethod
    def zero() -> "StoreValue":
        """Create a zero store value."""
        return StoreValue(type=StoreValueType.ZERO)


class StorageAddress:
    """Computed address for storage buffer access."""

    def __init__(self, base: str, offset: str):
        """
        Initialize storage address.

        Args:
            base: Base buffer expression
            offset: Computed byte offset expression
        """
        self.base = base
        self.offset = offset

    def to_string(self) -> str:
        """Convert to HLSL string representation."""
        return f"{self.base}, {self.offset}"


class StorageWriter:
    """Writer for storage buffer access code in HLSL."""

    def __init__(self, out: io.StringIO, module: Module, names: Dict[str, str]):
        """
        Initialize storage writer.

        Args:
            out: Output stream
            module: The module being written
            names: Name mapping for identifiers
        """
        self.out = out
        self.module = module
        self.names = names
        self.temp_access_chain: List[SubAccess] = []

    def fill_access_chain(
        self,
        expression: Handle[Expression],
        ctx: Any,
        chain: Optional[List[SubAccess]] = None
    ) -> None:
        """
        Fill access chain for a storage access expression.

        Args:
            expression: The expression to analyze
            ctx: Function context
            chain: Optional existing chain to extend
        """
        if chain is not None:
            self.temp_access_chain = chain.copy()
        else:
            self.temp_access_chain = []

        # Get the expression
        expr = self.module.expressions[expression]

        from ...ir.expression import ExpressionType
        match expr.type:
            case ExpressionType.GLOBAL_VARIABLE:
                # This is the base - get the buffer info
                self._add_buffer_access(expr.global_variable)
            case ExpressionType.ACCESS:
                # Access operation - recurse into base, then add index
                self.fill_access_chain(expr.base, ctx)
                self._add_access_index(expr.index, ctx)
            case ExpressionType.ACCESS_INDEX:
                # AccessIndex operation - recurse into base, then add index
                self.fill_access_chain(expr.base, ctx)
                self._add_access_index(expr.index, ctx)
            case _:
                # Non-storage access - clear chain
                self.temp_access_chain = []

    def write_storage_address(self, chain: Optional[List[SubAccess]] = None) -> StorageAddress:
        """
        Write storage buffer address expression.

        Args:
            chain: Access chain to use (defaults to temp_access_chain)

        Returns:
            Storage address object
        """
        if chain is None:
            chain = self.temp_access_chain

        if not chain:
            raise ValueError("Empty access chain")

        # Start with base buffer
        base_expr, base_offset = self._process_buffer_offset(chain[0])

        # Process remaining accesses
        offset_exprs = []
        if base_offset:
            offset_exprs.append(str(base_offset))

        for access in chain[1:]:
            offset_exprs.append(self._sub_access_to_string(access))

        # Combine offsets
        if not offset_exprs:
            final_offset = "0"
        elif len(offset_exprs) == 1:
            final_offset = offset_exprs[0]
        else:
            final_offset = " + ".join(offset_exprs)

        return StorageAddress(base_expr, final_offset)

    def write_storage_load(
        self,
        access_chain: List[SubAccess],
        ty_handle: Handle[Type],
        ctx: Any
    ) -> str:
        """
        Write a load from storage buffer.

        Args:
            access_chain: Access chain for the load
            ty_handle: Type to load
            ctx: Function context

        Returns:
            HLSL expression for the loaded value
        """
        from ...ir.type import TypeInnerType, ScalarKind

        self.temp_access_chain = access_chain
        address = self.write_storage_address()

        ty = self.module.types[ty_handle]
        inner = ty.inner

        match inner.type:
            case TypeInnerType.SCALAR:
                return self._write_scalar_load(address, inner.scalar.kind)
            case TypeInnerType.VECTOR:
                return self._write_vector_load(address, inner.vector.size, inner.vector.scalar.kind)
            case TypeInnerType.MATRIX:
                return self._write_matrix_load(address, inner.matrix)
            case TypeInnerType.ARRAY:
                return self._write_array_load(address, inner.array.base)
            case TypeInnerType.STRUCT:
                return self._write_struct_load(address, ty_handle)
            case _:
                raise ValueError(f"Unsupported storage load type: {inner.type}")

    def write_storage_store(
        self,
        access_chain: List[SubAccess],
        value: StoreValue,
        ty_handle: Handle[Type],
        ctx: Any
    ) -> str:
        """
        Write a store to storage buffer.

        Args:
            access_chain: Access chain for the store
            value: Value to store
            ty_handle: Type of value
            ctx: Function context

        Returns:
            HLSL statement for the store
        """
        from ...ir.type import TypeInnerType

        self.temp_access_chain = access_chain
        address = self.write_storage_address()

        ty = self.module.types[ty_handle]
        inner = ty.inner

        # Get value expression
        if value.type == StoreValueType.EXPRESSION:
            value_str = self._get_expression_string(value.expression, ctx)
        else:
            value_str = "0"

        match inner.type:
            case TypeInnerType.SCALAR:
                return self._write_scalar_store(address, value_str, inner.scalar.kind)
            case TypeInnerType.VECTOR:
                return self._write_vector_store(address, value_str, inner.vector)
            case TypeInnerType.MATRIX:
                return self._write_matrix_store(address, value_str, inner.matrix)
            case TypeInnerType.ARRAY:
                return self._write_array_store(address, value_str, inner.array)
            case TypeInnerType.STRUCT:
                return self._write_struct_store(address, value_str, ty_handle)
            case _:
                raise ValueError(f"Unsupported storage store type: {inner.type}")

    def _add_buffer_access(self, var_handle: Handle[Expression]) -> None:
        """Add buffer base access to chain.

        Args:
            var_handle: Handle to global variable
        """
        from ...ir.expression import ExpressionType

        expr = self.module.expressions[var_handle]
        if expr.type == ExpressionType.GLOBAL_VARIABLE:
            # Get binding group and offset from global variable
            var = self.module.global_variables[expr.global_variable]
            binding = var.binding

            if binding:
                # Extract group and binding
                group = binding.group if hasattr(binding, 'group') else 0
                binding_num = binding.binding if hasattr(binding, 'binding') else 0

                # Add buffer offset access
                self.temp_access_chain.insert(0, SubAccess.buffer_offset(group, binding_num))

    def _add_access_index(self, index: Union[int, Handle[Expression]], ctx: Any) -> None:
        """Add index access to chain.

        Args:
            index: Index (constant or expression)
            ctx: Function context
        """
        # Get the current type being accessed
        # This is simplified - full implementation would track types through the chain
        if isinstance(index, int):
            # Constant index
            self.temp_access_chain.append(SubAccess.offset(index))
        else:
            # Dynamic index - need stride
            # For now, assume 4-byte stride (simplified)
            self.temp_access_chain.append(SubAccess.index(index, 4))

    def _process_buffer_offset(self, access: SubAccess) -> tuple[str, Optional[int]]:
        """Process buffer offset access.

        Args:
            access: The buffer offset access

        Returns:
            Tuple of (base expression, static offset)
        """
        if access.type == SubAccessType.BUFFER_OFFSET:
            # Get buffer name
            group = access.group if access.group is not None else 0
            offset = access.offset if access.offset is not None else 0

            # Construct buffer variable name
            buffer_name = f"_group{group}_binding{offset}"
            return buffer_name, None

        return "", None

    def _sub_access_to_string(self, access: SubAccess) -> str:
        """Convert sub-access to string.

        Args:
            access: The sub-access

        Returns:
            String representation
        """
        if access.type == SubAccessType.OFFSET:
            return str(access.offset) if access.offset else "0"
        elif access.type == SubAccessType.INDEX:
            value_str = self._get_expression_string(access.value, None)
            stride = access.stride if access.stride else 4
            return f"({value_str} * {stride})"
        else:
            return "0"

    def _write_scalar_load(self, address: StorageAddress, kind: Any) -> str:
        """Write a scalar load.

        Args:
            address: Storage address
            kind: Scalar kind

        Returns:
            HLSL load expression
        """
        from ...ir.type import ScalarKind

        method = self._get_load_method(kind, 1)
        return f"{address.base}.{method}({address.offset})"

    def _write_vector_load(self, address: StorageAddress, size: Any, kind: Any) -> str:
        """Write a vector load.

        Args:
            address: Storage address
            size: Vector size
            kind: Scalar kind

        Returns:
            HLSL load expression
        """
        from ...ir.type import VectorSize

        components = size.value if hasattr(size, 'value') else size
        method = self._get_load_method(kind, components)
        loaded = f"{address.base}.{method}({address.offset})"

        # Bitcast if necessary
        if kind != ScalarKind.UINT:
            target_type = self._get_type_string(kind, components)
            return f"as{target_type}({loaded})"

        return loaded

    def _write_matrix_load(self, address: StorageAddress, matrix: Any) -> str:
        """Write a matrix load.

        Args:
            address: Storage address
            matrix: Matrix type info

        Returns:
            HLSL load expression
        """
        # Simplified - load column by column
        cols = matrix.columns.value if hasattr(matrix.columns, 'value') else matrix.columns
        rows = matrix.rows.value if hasattr(matrix.rows, 'value') else matrix.rows
        kind = matrix.scalar.kind

        # This is simplified - full implementation would construct matrix properly
        return f"{address.base}.Load({address.offset})"

    def _write_array_load(self, address: StorageAddress, base: Handle[Type]) -> str:
        """Write an array load.

        Args:
            address: Storage address
            base: Base array type

        Returns:
            HLSL load expression
        """
        return f"{address.base}.Load({address.offset})"

    def _write_struct_load(self, address: StorageAddress, ty_handle: Handle[Type]) -> str:
        """Write a struct load.

        Args:
            address: Storage address
            ty_handle: Struct type handle

        Returns:
            HLSL load expression
        """
        # Simplified - full implementation would load member by member
        ty = self.module.types[ty_handle]
        struct_name = ty.name if ty.name else f"Struct{ty_handle.index}"

        # Return placeholder - actual implementation would construct struct
        return f"({struct_name})0"

    def _write_scalar_store(self, address: StorageAddress, value: str, kind: Any) -> str:
        """Write a scalar store.

        Args:
            address: Storage address
            value: Value expression
            kind: Scalar kind

        Returns:
            HLSL store statement
        """
        method = self._get_store_method(kind, 1)
        return f"{address.base}.{method}({address.offset}, {value});\n"

    def _write_vector_store(self, address: StorageAddress, value: str, vector: Any) -> str:
        """Write a vector store.

        Args:
            address: Storage address
            value: Value expression
            vector: Vector type info

        Returns:
            HLSL store statement
        """
        from ...ir.type import VectorSize

        components = vector.size.value if hasattr(vector.size, 'value') else vector.size
        kind = vector.scalar.kind

        # Bitcast to uint if necessary
        if kind != ScalarKind.UINT:
            value = f"asuint({value})"

        method = self._get_store_method(kind, components)
        return f"{address.base}.{method}({address.offset}, {value});\n"

    def _write_matrix_store(self, address: StorageAddress, value: str, matrix: Any) -> str:
        """Write a matrix store.

        Args:
            address: Storage address
            value: Value expression
            matrix: Matrix type info

        Returns:
            HLSL store statement
        """
        # Simplified - store column by column
        return f"{address.base}.Store({address.offset}, asuint({value}));\n"

    def _write_array_store(self, address: StorageAddress, value: str, array: Any) -> str:
        """Write an array store.

        Args:
            address: Storage address
            value: Value expression
            array: Array type info

        Returns:
            HLSL store statement
        """
        return f"{address.base}.Store({address.offset}, asuint({value}));\n"

    def _write_struct_store(self, address: StorageAddress, value: str, ty_handle: Handle[Type]) -> str:
        """Write a struct store.

        Args:
            address: Storage address
            value: Value expression
            ty_handle: Struct type handle

        Returns:
            HLSL store statement
        """
        # Simplified - full implementation would store member by member
        return f"{address.base}.Store({address.offset}, asuint({value}));\n"

    def _get_load_method(self, kind: Any, components: int) -> str:
        """Get the Load method name for a type.

        Args:
            kind: Scalar kind
            components: Number of components

        Returns:
            Method name
        """
        from ...ir.type import ScalarKind

        match kind:
            case ScalarKind.F16 | ScalarKind.F32 | ScalarKind.F64:
                # Floating point types use generic Load if available
                return "Load" if components == 1 else f"Load{components}"
            case ScalarKind.I32 | ScalarKind.UINT | ScalarKind.BOOL:
                # Integer types
                return "Load" if components == 1 else f"Load{components}"
            case _:
                return "Load"

    def _get_store_method(self, kind: Any, components: int) -> str:
        """Get the Store method name for a type.

        Args:
            kind: Scalar kind
            components: Number of components

        Returns:
            Method name
        """
        from ...ir.type import ScalarKind

        match kind:
            case ScalarKind.F16 | ScalarKind.F32 | ScalarKind.F64:
                return "Store" if components == 1 else f"Store{components}"
            case ScalarKind.I32 | ScalarKind.UINT | ScalarKind.BOOL:
                return "Store" if components == 1 else f"Store{components}"
            case _:
                return "Store"

    def _get_type_string(self, kind: Any, components: int) -> str:
        """Get HLSL type string.

        Args:
            kind: Scalar kind
            components: Number of components

        Returns:
            Type string
        """
        from ...ir.type import ScalarKind

        type_map = {
            ScalarKind.F16: "half",
            ScalarKind.F32: "float",
            ScalarKind.F64: "double",
            ScalarKind.I32: "int",
            ScalarKind.UINT: "uint",
            ScalarKind.BOOL: "bool",
        }

        base = type_map.get(kind, "float")
        if components == 1:
            return base
        else:
            return f"{base}{components}"

    def _get_expression_string(self, expr_handle: Handle[Expression], ctx: Any) -> str:
        """Get string representation of expression.

        Args:
            expr_handle: Expression handle
            ctx: Function context

        Returns:
            Expression string
        """
        # Simplified - full implementation would use expression writer
        return f"_e{expr_handle.index}"


__all__ = [
    # Constants
    "STORE_TEMP_NAME",
    # Classes
    "SubAccessType",
    "SubAccess",
    "StoreValueType",
    "StoreValue",
    "StorageAddress",
    "StorageWriter",
]
