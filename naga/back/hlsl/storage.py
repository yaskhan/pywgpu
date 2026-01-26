"""naga.back.hlsl.storage

A port of `wgpu-trunk/naga/src/back/hlsl/storage.rs`.

This module implements ByteAddressBuffer / RWByteAddressBuffer addressing,
loading and storing for Naga's Storage address space.

Upstream Naga implements this as methods on the HLSL writer. In this Python
port we expose the same logic via :class:`~StorageWriter`, which is intended to
be used as a mixin by a full HLSL writer implementation.

The implementation here mirrors the upstream control flow and emitted HLSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

from .. import INDENT, Level
from ...proc import NameKey, TypeResolutionHandle, TypeResolutionValue

if TYPE_CHECKING:
    from ...arena import Handle
    from ...ir import Expression, GlobalVariable, Module, Type, TypeInner, VectorSize
    from .. import FunctionCtx


STORE_TEMP_NAME = "_value"


class SubAccessType(Enum):
    """Kept for backward compatibility.

    Upstream uses an enum with variants; here the actual lowering is represented
    by :class:`SubAccess` instances.
    """

    BUFFER_OFFSET = "buffer_offset"
    OFFSET = "offset"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class SubAccess:
    """One step in accessing a Storage global's component or element."""

    type: SubAccessType
    group: int | None = None
    offset: int | None = None
    value: "Handle[Expression] | None" = None
    stride: int | None = None

    @staticmethod
    def buffer_offset(group: int, offset: int) -> "SubAccess":
        return SubAccess(SubAccessType.BUFFER_OFFSET, group=group, offset=offset)

    @staticmethod
    def offset_const(offset: int) -> "SubAccess":
        return SubAccess(SubAccessType.OFFSET, offset=offset)

    @staticmethod
    def index(value: "Handle[Expression]", stride: int) -> "SubAccess":
        return SubAccess(SubAccessType.INDEX, value=value, stride=stride)


class StoreValueType(Enum):
    """Kept for backward compatibility."""

    EXPRESSION = "expression"
    TEMP_INDEX = "temp_index"
    TEMP_ACCESS = "temp_access"
    TEMP_COLUMN_ACCESS = "temp_column_access"


@dataclass(frozen=True, slots=True)
class StoreValue:
    """A value to store.

    Mirrors `StoreValue` in upstream `storage.rs`.
    """

    type: StoreValueType

    expression: "Handle[Expression] | None" = None

    depth: int | None = None
    index: int | None = None
    ty: object | None = None

    base: "Handle[Type] | None" = None
    member_index: int | None = None

    column: int | None = None

    @staticmethod
    def expression(expr: "Handle[Expression]") -> "StoreValue":
        return StoreValue(StoreValueType.EXPRESSION, expression=expr)

    @staticmethod
    def temp_index(depth: int, index: int, ty: object) -> "StoreValue":
        return StoreValue(StoreValueType.TEMP_INDEX, depth=depth, index=index, ty=ty)

    @staticmethod
    def temp_access(depth: int, base: "Handle[Type]", member_index: int) -> "StoreValue":
        return StoreValue(StoreValueType.TEMP_ACCESS, depth=depth, base=base, member_index=member_index)

    @staticmethod
    def temp_column_access(
        depth: int,
        base: "Handle[Type]",
        member_index: int,
        column: int,
    ) -> "StoreValue":
        return StoreValue(
            StoreValueType.TEMP_COLUMN_ACCESS,
            depth=depth,
            base=base,
            member_index=member_index,
            column=column,
        )


class StorageWriter:
    """Implements HLSL storage buffer access helpers.

    This class is written in the same style as upstream `impl Writer` methods.

    The full writer is expected to provide the following hooks:

    - ``out``: text writer
    - ``names``: mapping from :class:`~naga.proc.NameKey` to identifiers
    - ``options``: writer options that may support dynamic buffer offsets
    - ``temp_access_chain``: list used for temporary access lowering
    - ``write_expr(module, expr, func_ctx)``
    - ``write_type(module, ty_handle)``
    - ``write_value_type(module, ty_inner)``
    - ``write_wrapped_constructor_function_name(module, constructor)``
    - ``write_array_size(module, base, size)``

    When used as a mixin, these hooks are provided by the main writer.
    """

    def __init__(self) -> None:
        self.temp_access_chain: list[SubAccess] = []

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _alignment_from_vector_size(rows: "VectorSize") -> int:
        # Matches `proc::Alignment::from(VectorSize)` in upstream.
        # vec2 is 8-byte aligned; vec3/vec4 are 16-byte aligned.
        v = getattr(rows, "value", rows)
        return 8 if int(v) == 2 else 16

    @staticmethod
    def _type_resolution_inner(module: "Module", ty: object) -> "TypeInner":
        # Accept either TypeResolutionHandle / TypeResolutionValue (Python proc)
        # or direct TypeInner.
        if isinstance(ty, TypeResolutionHandle):
            return module.types[ty.handle].inner
        if isinstance(ty, TypeResolutionValue):
            return ty.inner
        # Fallback
        return ty  # type: ignore[return-value]

    @staticmethod
    def _type_resolution_handle(ty: object) -> "Handle[Type] | None":
        if isinstance(ty, TypeResolutionHandle):
            # This module uses `Handle[Type]` for type handles.
            from ...arena import Handle

            return Handle.from_usize_unchecked(ty.handle)
        return None

    # ---------------------------------------------------------------------
    # Core: address
    # ---------------------------------------------------------------------

    def write_storage_address(
        self,
        module: "Module",
        chain: Sequence[SubAccess],
        func_ctx: "FunctionCtx",
    ) -> None:
        if not chain:
            self.out.write("0")  # type: ignore[attr-defined]
            return

        for i, access in enumerate(chain):
            if i != 0:
                self.out.write("+")  # type: ignore[attr-defined]
            if access.type == SubAccessType.BUFFER_OFFSET:
                group = int(access.group or 0)
                offset = int(access.offset or 0)
                self.out.write(f"__dynamic_buffer_offsets{group}._{offset}")  # type: ignore[attr-defined]
            elif access.type == SubAccessType.OFFSET:
                self.out.write(str(int(access.offset or 0)))  # type: ignore[attr-defined]
            elif access.type == SubAccessType.INDEX:
                if access.value is None or access.stride is None:
                    raise ValueError("Malformed SubAccess::Index")
                self.write_expr(module, access.value, func_ctx)
                self.out.write(f"*{int(access.stride)}")  # type: ignore[attr-defined]
            else:
                raise ValueError(f"Unknown SubAccess: {access.type}")

    def _write_storage_load_sequence(
        self,
        module: "Module",
        var_handle: "Handle[GlobalVariable]",
        sequence: Iterable[tuple[object, int]],
        func_ctx: "FunctionCtx",
    ) -> None:
        for i, (ty_resolution, offset) in enumerate(sequence):
            self.temp_access_chain.append(SubAccess.offset_const(offset))
            if i != 0:
                self.out.write(", ")  # type: ignore[attr-defined]
            self.write_storage_load(module, var_handle, ty_resolution, func_ctx)
            self.temp_access_chain.pop()

    # ---------------------------------------------------------------------
    # Loads
    # ---------------------------------------------------------------------

    def write_storage_load(
        self,
        module: "Module",
        var_handle: "Handle[GlobalVariable]",
        result_ty: object,
        func_ctx: "FunctionCtx",
    ) -> None:
        from .conv import hlsl_cast, hlsl_scalar
        from ...ir.type import ArraySizeType, TypeInnerType

        inner = self._type_resolution_inner(module, result_ty)

        if inner.type == TypeInnerType.SCALAR:
            scalar = inner.scalar
            if scalar is None:
                raise ValueError("Scalar TypeInner missing scalar")

            chain = self.temp_access_chain
            self.temp_access_chain = []
            var_name = self.names[NameKey.global_variable(var_handle)]  # type: ignore[attr-defined]

            if scalar.width == 4:
                cast = hlsl_cast(scalar.kind)
                self.out.write(f"{cast}({var_name}.Load(")  # type: ignore[attr-defined]
            else:
                ty = hlsl_scalar(scalar.kind, scalar.width)
                self.out.write(f"{var_name}.Load<{ty}(")  # type: ignore[attr-defined]

            self.write_storage_address(module, chain, func_ctx)
            self.out.write(")")  # type: ignore[attr-defined]
            if scalar.width == 4:
                self.out.write(")")  # type: ignore[attr-defined]

            self.temp_access_chain = chain
            return

        if inner.type == TypeInnerType.VECTOR:
            vector = inner.vector
            if vector is None:
                raise ValueError("Vector TypeInner missing vector")

            chain = self.temp_access_chain
            self.temp_access_chain = []
            var_name = self.names[NameKey.global_variable(var_handle)]  # type: ignore[attr-defined]
            size = int(vector.size.value)

            if vector.scalar.width == 4:
                cast = hlsl_cast(vector.scalar.kind)
                self.out.write(f"{cast}({var_name}.Load{size}(")  # type: ignore[attr-defined]
            else:
                ty = hlsl_scalar(vector.scalar.kind, vector.scalar.width)
                self.out.write(f"{var_name}.Load<{ty}{size}(")  # type: ignore[attr-defined]

            self.write_storage_address(module, chain, func_ctx)
            self.out.write(")")  # type: ignore[attr-defined]
            if vector.scalar.width == 4:
                self.out.write(")")  # type: ignore[attr-defined]

            self.temp_access_chain = chain
            return

        if inner.type == TypeInnerType.MATRIX:
            matrix = inner.matrix
            if matrix is None:
                raise ValueError("Matrix TypeInner missing matrix")

            scalar_ty = hlsl_scalar(matrix.scalar.kind, matrix.scalar.width)
            cols = int(matrix.columns.value)
            rows = int(matrix.rows.value)
            self.out.write(f"{scalar_ty}{cols}x{rows}(")  # type: ignore[attr-defined]

            row_stride = self._alignment_from_vector_size(matrix.rows) * matrix.scalar.width

            def _iter() -> Iterator[tuple[object, int]]:
                from ...ir.type import TypeInner

                for i in range(cols):
                    ty_inner = TypeInner.new_vector(matrix.rows, matrix.scalar)
                    yield TypeResolutionValue(ty_inner), i * row_stride

            self._write_storage_load_sequence(module, var_handle, _iter(), func_ctx)
            self.out.write(")")  # type: ignore[attr-defined]
            return

        if inner.type == TypeInnerType.ARRAY:
            array = inner.array
            if array is None:
                raise ValueError("Array TypeInner missing array")

            if array.size.type != ArraySizeType.CONSTANT or array.size.constant is None:
                raise ValueError("Only constant arrays are supported in storage loads")

            constructor = {"ty": self._type_resolution_handle(result_ty)}
            self.write_wrapped_constructor_function_name(module, constructor)  # type: ignore[arg-type]
            self.out.write("(")  # type: ignore[attr-defined]
            size = int(array.size.constant.value)
            stride = int(array.stride)
            base = array.base

            def _iter() -> Iterator[tuple[object, int]]:
                for i in range(size):
                    yield TypeResolutionHandle(base), stride * i

            self._write_storage_load_sequence(module, var_handle, _iter(), func_ctx)
            self.out.write(")")  # type: ignore[attr-defined]
            return

        if inner.type == TypeInnerType.STRUCT:
            struct = inner.struct
            if struct is None:
                raise ValueError("Struct TypeInner missing struct")

            constructor = {"ty": self._type_resolution_handle(result_ty)}
            self.write_wrapped_constructor_function_name(module, constructor)  # type: ignore[arg-type]
            self.out.write("(")  # type: ignore[attr-defined]

            def _iter() -> Iterator[tuple[object, int]]:
                for member in struct.members:
                    yield TypeResolutionHandle(member.ty), int(member.offset)

            self._write_storage_load_sequence(module, var_handle, _iter(), func_ctx)
            self.out.write(")")  # type: ignore[attr-defined]
            return

        raise ValueError(f"Unsupported storage load type: {inner.type}")

    # ---------------------------------------------------------------------
    # Stores
    # ---------------------------------------------------------------------

    def _write_store_value(self, module: "Module", value: StoreValue, func_ctx: "FunctionCtx") -> None:
        if value.type == StoreValueType.EXPRESSION:
            if value.expression is None:
                raise ValueError("StoreValue::Expression missing expression")
            self.write_expr(module, value.expression, func_ctx)
            return

        if value.type == StoreValueType.TEMP_INDEX:
            if value.depth is None or value.index is None:
                raise ValueError("StoreValue::TempIndex missing fields")
            self.out.write(f"{STORE_TEMP_NAME}{value.depth}[{value.index}]")  # type: ignore[attr-defined]
            return

        if value.type == StoreValueType.TEMP_ACCESS:
            if value.depth is None or value.base is None or value.member_index is None:
                raise ValueError("StoreValue::TempAccess missing fields")
            name = self.names[NameKey.struct_member(value.base, value.member_index)]  # type: ignore[attr-defined]
            self.out.write(f"{STORE_TEMP_NAME}{value.depth}.{name}")  # type: ignore[attr-defined]
            return

        if value.type == StoreValueType.TEMP_COLUMN_ACCESS:
            if (
                value.depth is None
                or value.base is None
                or value.member_index is None
                or value.column is None
            ):
                raise ValueError("StoreValue::TempColumnAccess missing fields")
            name = self.names[NameKey.struct_member(value.base, value.member_index)]  # type: ignore[attr-defined]
            self.out.write(f"{STORE_TEMP_NAME}{value.depth}.{name}_{value.column}")  # type: ignore[attr-defined]
            return

        raise ValueError(f"Unknown StoreValue type: {value.type}")

    def write_storage_store(
        self,
        module: "Module",
        var_handle: "Handle[GlobalVariable]",
        value: StoreValue,
        func_ctx: "FunctionCtx",
        level: Level,
        within_struct: "Handle[Type] | None" = None,
    ) -> None:
        from .conv import hlsl_scalar
        from ...ir.type import ArraySizeType, TypeInner, TypeInnerType, VectorSize

        # Determine the type resolution for the store value.
        if value.type == StoreValueType.EXPRESSION:
            if value.expression is None:
                raise ValueError("StoreValue::Expression missing expression")
            ty_resolution: object = func_ctx.info[value.expression].ty  # type: ignore[attr-defined]
        elif value.type == StoreValueType.TEMP_INDEX:
            if value.ty is None:
                raise ValueError("StoreValue::TempIndex missing ty")
            ty_resolution = value.ty
        elif value.type == StoreValueType.TEMP_ACCESS:
            if value.base is None or value.member_index is None:
                raise ValueError("StoreValue::TempAccess missing fields")
            struct_ty = module.types[value.base].inner
            if struct_ty.type != TypeInnerType.STRUCT or struct_ty.struct is None:
                raise ValueError("TempAccess base is not a struct")
            member_ty = struct_ty.struct.members[int(value.member_index)].ty
            ty_resolution = TypeResolutionHandle(member_ty)
        else:
            raise ValueError("TempColumnAccess should not be passed to write_storage_store")

        inner = self._type_resolution_inner(module, ty_resolution)

        var_name = self.names[NameKey.global_variable(var_handle)]  # type: ignore[attr-defined]

        if inner.type == TypeInnerType.SCALAR:
            scalar = inner.scalar
            if scalar is None:
                raise ValueError("Scalar TypeInner missing scalar")

            chain = self.temp_access_chain
            self.temp_access_chain = []

            if scalar.width == 4:
                self.out.write(f"{level}{var_name}.Store(")  # type: ignore[attr-defined]
                self.write_storage_address(module, chain, func_ctx)
                self.out.write(", asuint(")  # type: ignore[attr-defined]
                self._write_store_value(module, value, func_ctx)
                self.out.write("));\n")  # type: ignore[attr-defined]
            else:
                self.out.write(f"{level}{var_name}.Store(")  # type: ignore[attr-defined]
                self.write_storage_address(module, chain, func_ctx)
                self.out.write(", ")  # type: ignore[attr-defined]
                self._write_store_value(module, value, func_ctx)
                self.out.write(");\n")  # type: ignore[attr-defined]

            self.temp_access_chain = chain
            return

        if inner.type == TypeInnerType.VECTOR:
            vector = inner.vector
            if vector is None:
                raise ValueError("Vector TypeInner missing vector")

            chain = self.temp_access_chain
            self.temp_access_chain = []

            size = int(vector.size.value)
            if vector.scalar.width == 4:
                self.out.write(f"{level}{var_name}.Store{size}(")  # type: ignore[attr-defined]
                self.write_storage_address(module, chain, func_ctx)
                self.out.write(", asuint(")  # type: ignore[attr-defined]
                self._write_store_value(module, value, func_ctx)
                self.out.write("));\n")  # type: ignore[attr-defined]
            else:
                self.out.write(f"{level}{var_name}.Store(")  # type: ignore[attr-defined]
                self.write_storage_address(module, chain, func_ctx)
                self.out.write(", ")  # type: ignore[attr-defined]
                self._write_store_value(module, value, func_ctx)
                self.out.write(");\n")  # type: ignore[attr-defined]

            self.temp_access_chain = chain
            return

        if inner.type == TypeInnerType.MATRIX:
            matrix = inner.matrix
            if matrix is None:
                raise ValueError("Matrix TypeInner missing matrix")

            row_stride = self._alignment_from_vector_size(matrix.rows) * matrix.scalar.width
            cols = int(matrix.columns.value)

            self.out.write(f"{level}{{\n")  # type: ignore[attr-defined]

            if within_struct is not None and matrix.rows == VectorSize.BI:
                chain = list(self.temp_access_chain)
                for i in range(cols):
                    chain.append(SubAccess.offset_const(i * row_stride))
                    if value.type != StoreValueType.TEMP_ACCESS:
                        raise ValueError("within_struct requires TempAccess store value")
                    assert value.member_index is not None
                    column_value = StoreValue.temp_column_access(
                        depth=level.level,
                        base=within_struct,
                        member_index=int(value.member_index),
                        column=i,
                    )
                    if matrix.scalar.width == 4:
                        self.out.write(f"{level.next()}{var_name}.Store{int(matrix.rows.value)}(")  # type: ignore[attr-defined]
                        self.write_storage_address(module, chain, func_ctx)
                        self.out.write(", asuint(")  # type: ignore[attr-defined]
                        self._write_store_value(module, column_value, func_ctx)
                        self.out.write("));\n")  # type: ignore[attr-defined]
                    else:
                        self.out.write(f"{level.next()}{var_name}.Store(")  # type: ignore[attr-defined]
                        self.write_storage_address(module, chain, func_ctx)
                        self.out.write(", ")  # type: ignore[attr-defined]
                        self._write_store_value(module, column_value, func_ctx)
                        self.out.write(");\n")  # type: ignore[attr-defined]
                    chain.pop()
            else:
                depth = level.level + 1
                scalar_ty = hlsl_scalar(matrix.scalar.kind, matrix.scalar.width)
                self.out.write(
                    f"{level.next()}{scalar_ty}{int(matrix.columns.value)}x{int(matrix.rows.value)} {STORE_TEMP_NAME}{depth} = "
                )  # type: ignore[attr-defined]
                self._write_store_value(module, value, func_ctx)
                self.out.write(";\n")  # type: ignore[attr-defined]

                for i in range(cols):
                    self.temp_access_chain.append(SubAccess.offset_const(i * row_stride))
                    ty_inner = TypeInner.new_vector(matrix.rows, matrix.scalar)
                    sv = StoreValue.temp_index(depth=depth, index=i, ty=TypeResolutionValue(ty_inner))
                    self.write_storage_store(module, var_handle, sv, func_ctx, level.next(), None)
                    self.temp_access_chain.pop()

            self.out.write(f"{level}}}\n")  # type: ignore[attr-defined]
            return

        if inner.type == TypeInnerType.ARRAY:
            array = inner.array
            if array is None:
                raise ValueError("Array TypeInner missing array")
            if array.size.type != ArraySizeType.CONSTANT or array.size.constant is None:
                raise ValueError("Only constant arrays are supported in storage stores")

            self.out.write(f"{level}{{\n")  # type: ignore[attr-defined]
            self.out.write(f"{level.next()}")  # type: ignore[attr-defined]
            self.write_type(module, array.base)
            depth = level.next().level
            self.out.write(f" {STORE_TEMP_NAME}{depth}")  # type: ignore[attr-defined]
            self.write_array_size(module, array.base, array.size)
            self.out.write(" = ")  # type: ignore[attr-defined]
            self._write_store_value(module, value, func_ctx)
            self.out.write(";\n")  # type: ignore[attr-defined]

            size = int(array.size.constant.value)
            for i in range(size):
                self.temp_access_chain.append(SubAccess.offset_const(i * int(array.stride)))
                sv = StoreValue.temp_index(depth=depth, index=i, ty=TypeResolutionHandle(array.base))
                self.write_storage_store(module, var_handle, sv, func_ctx, level.next(), None)
                self.temp_access_chain.pop()

            self.out.write(f"{level}}}\n")  # type: ignore[attr-defined]
            return

        if inner.type == TypeInnerType.STRUCT:
            struct = inner.struct
            if struct is None:
                raise ValueError("Struct TypeInner missing struct")

            self.out.write(f"{level}{{\n")  # type: ignore[attr-defined]
            depth = level.next().level
            struct_ty_handle = self._type_resolution_handle(ty_resolution)
            if struct_ty_handle is None:
                raise ValueError("Struct store requires a type handle")
            struct_name = self.names[NameKey.type_(struct_ty_handle)]  # type: ignore[attr-defined]
            self.out.write(f"{level.next()}{struct_name} {STORE_TEMP_NAME}{depth} = ")  # type: ignore[attr-defined]
            self._write_store_value(module, value, func_ctx)
            self.out.write(";\n")  # type: ignore[attr-defined]

            for i, member in enumerate(struct.members):
                self.temp_access_chain.append(SubAccess.offset_const(int(member.offset)))
                sv = StoreValue.temp_access(depth=depth, base=struct_ty_handle, member_index=i)
                self.write_storage_store(
                    module,
                    var_handle,
                    sv,
                    func_ctx,
                    level.next(),
                    within_struct=struct_ty_handle,
                )
                self.temp_access_chain.pop()

            self.out.write(f"{level}}}\n")  # type: ignore[attr-defined]
            return

        raise ValueError(f"Unsupported storage store type: {inner.type}")

    # ---------------------------------------------------------------------
    # Access chain lowering
    # ---------------------------------------------------------------------

    def fill_access_chain(
        self,
        module: "Module",
        cur_expr: "Handle[Expression]",
        func_ctx: "FunctionCtx",
    ) -> "Handle[GlobalVariable]":
        """Set ``temp_access_chain`` to compute the byte offset of ``cur_expr``.

        Mirrors upstream `fill_access_chain`.
        """

        from ...ir import ExpressionType
        from ...ir.type import TypeInnerType

        self.temp_access_chain.clear()

        while True:
            expr = func_ctx.expressions[cur_expr]  # type: ignore[attr-defined]
            if expr.type == ExpressionType.GLOBAL_VARIABLE:
                var_handle = expr.global_variable
                if var_handle is None:
                    raise ValueError("Malformed GlobalVariable expression")

                # Dynamic storage buffer offsets, if supported by options.
                gv = module.global_variables[var_handle]
                binding = getattr(gv, "binding", None)
                if binding is not None and hasattr(self, "options"):
                    options = getattr(self, "options")
                    resolve = getattr(options, "resolve_resource_binding", None)
                    if resolve is not None:
                        bt = resolve(binding)
                        dyn_index = getattr(bt, "dynamic_storage_buffer_offsets_index", None)
                        if dyn_index is not None:
                            group = int(getattr(binding, "group", 0))
                            self.temp_access_chain.append(SubAccess.buffer_offset(group, int(dyn_index)))

                return var_handle

            if expr.type == ExpressionType.ACCESS:
                if expr.access_base is None or expr.access_index is None:
                    raise ValueError("Malformed Access expression")
                next_expr = expr.access_base
                access_index_expr = expr.access_index
                access_index: tuple[str, object] = ("expr", access_index_expr)

            elif expr.type == ExpressionType.ACCESS_INDEX:
                if expr.access_base is None or expr.access_index_value is None:
                    raise ValueError("Malformed AccessIndex expression")
                next_expr = expr.access_base
                access_index = ("const", int(expr.access_index_value))

            else:
                raise ValueError(f"Pointer access of {expr.type} is not supported")

            # Determine parent type.
            parent_ptr = func_ctx.resolve_type(next_expr, module)  # type: ignore[arg-type]
            if parent_ptr.type == TypeInnerType.POINTER:
                base = parent_ptr.pointer.base if parent_ptr.pointer is not None else None
                if base is None:
                    raise ValueError("Pointer missing base")
                parent = module.types[base].inner
            elif parent_ptr.type == TypeInnerType.VALUE_POINTER:
                scalar = parent_ptr.value_pointer.scalar if parent_ptr.value_pointer is not None else None
                if scalar is None:
                    raise ValueError("ValuePointer missing scalar")
                parent = None
                stride = int(scalar.width)
                parent_kind: tuple[str, object] = ("array", stride)
            else:
                raise ValueError("Unexpected parent pointer kind")

            if parent is not None:
                if parent.type == TypeInnerType.STRUCT:
                    parent_kind = ("struct", parent.struct.members if parent.struct is not None else [])
                elif parent.type == TypeInnerType.ARRAY:
                    stride = int(parent.array.stride) if parent.array is not None else 0
                    parent_kind = ("array", stride)
                elif parent.type == TypeInnerType.VECTOR:
                    scalar = parent.vector.scalar if parent.vector is not None else None
                    parent_kind = ("array", int(scalar.width if scalar is not None else 0))
                elif parent.type == TypeInnerType.MATRIX:
                    mat = parent.matrix
                    if mat is None:
                        raise ValueError("Matrix missing")
                    stride = self._alignment_from_vector_size(mat.rows) * int(mat.scalar.width)
                    parent_kind = ("array", stride)
                else:
                    raise ValueError("Unexpected pointer base type")

            # Produce SubAccess.
            if parent_kind[0] == "array":
                stride = int(parent_kind[1])
                if access_index[0] == "expr":
                    self.temp_access_chain.append(SubAccess.index(access_index[1], stride))  # type: ignore[arg-type]
                else:
                    self.temp_access_chain.append(SubAccess.offset_const(stride * int(access_index[1])))
            else:
                members = parent_kind[1]
                if access_index[0] != "const":
                    raise ValueError("Dynamic indexing into struct is unreachable")
                index = int(access_index[1])
                offset = int(members[index].offset)
                self.temp_access_chain.append(SubAccess.offset_const(offset))

            cur_expr = next_expr

    # --- hooks expected from the main writer ---

    def write_expr(self, module: "Module", expr: "Handle[Expression]", func_ctx: "FunctionCtx") -> None:  # pragma: no cover
        raise NotImplementedError

    def write_type(self, module: "Module", ty: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_value_type(self, module: "Module", inner: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_wrapped_constructor_function_name(self, module: "Module", constructor: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_array_size(self, module: "Module", base: object, size: object) -> None:  # pragma: no cover
        raise NotImplementedError


__all__ = [
    "STORE_TEMP_NAME",
    "SubAccessType",
    "SubAccess",
    "StoreValueType",
    "StoreValue",
    "StorageWriter",
]
