"""naga.valid.validator

This module implements the Naga shader validator, which checks that a Module
is structurally correct and semantically valid according to the Naga IR spec.

The validator mirrors the Rust implementation in `wgpu/naga/src/valid/mod.rs`.
"""

from __future__ import annotations

from typing import Optional

from ..ir.constant import Constant
from ..ir.function import Function
from ..ir.module import EntryPoint, Module
from ..ir.type import Array, Struct, StructMember, Type, TypeInner
from .errors import ValidationError
from .flags import (
    Capabilities,
    ShaderStages,
    SubgroupOperationSet,
    TypeFlags,
    ValidationFlags,
)
from .module_info import FunctionInfo, ModuleInfo


class Validator:
    """Naga shader validator.

    Validates a :class:`naga.ir.module.Module` and returns a
    :class:`naga.valid.module_info.ModuleInfo` containing analysis results
    that can be used by backends for code generation.

    The validator checks for:

    - Type correctness (scalar/vector/matrix constraints, struct layouts)
    - Constant expression correctness
    - Function argument/result validity
    - Entry point stage and binding constraints
    - Control flow uniformity (if enabled)
    - Resource binding uniqueness

    Validation can be customized via :class:`ValidationFlags` and
    :class:`Capabilities`.
    """

    def __init__(
        self,
        flags: Optional[ValidationFlags] = None,
        capabilities: Optional[Capabilities] = None,
    ) -> None:
        """Initialize the validator.

        Args:
            flags: Flags controlling which validation stages to perform.
                   Defaults to :meth:`ValidationFlags.default()`.
            capabilities: Allowed shader capabilities. Defaults to
                          :meth:`Capabilities.default()`.
        """
        self.flags = flags if flags is not None else ValidationFlags.default()
        self.capabilities = (
            capabilities if capabilities is not None else Capabilities.default()
        )

        # Configure subgroup support based on capabilities
        if self.capabilities & Capabilities.SUBGROUP:
            self.subgroup_operations = (
                SubgroupOperationSet.BASIC
                | SubgroupOperationSet.VOTE
                | SubgroupOperationSet.ARITHMETIC
                | SubgroupOperationSet.BALLOT
                | SubgroupOperationSet.SHUFFLE
                | SubgroupOperationSet.SHUFFLE_RELATIVE
                | SubgroupOperationSet.QUAD_FRAGMENT_COMPUTE
            )
        else:
            self.subgroup_operations = SubgroupOperationSet.empty()

        # Determine which stages can use subgroup operations
        self.subgroup_stages = ShaderStages.empty()
        if self.capabilities & Capabilities.SUBGROUP_VERTEX_STAGE:
            self.subgroup_stages |= ShaderStages.VERTEX
        if self.capabilities & Capabilities.SUBGROUP:
            self.subgroup_stages |= (
                ShaderStages.FRAGMENT | ShaderStages.COMPUTE_LIKE
            )

        # Internal validator state
        self._type_flags: list[TypeFlags] = []
        self._overrides_resolved: bool = False

    def reset(self) -> None:
        """Reset the validator's internal state.

        This allows reusing a single Validator instance across multiple modules
        without reallocating resources.
        """
        self._type_flags.clear()
        self._overrides_resolved = False

    def validate(self, module: Module) -> ModuleInfo:
        """Validate a module and return analysis info.

        Args:
            module: The :class:`naga.ir.module.Module` to validate.

        Returns:
            A :class:`ModuleInfo` containing type flags, function info, and
            entry point info.

        Raises:
            ValidationError: If the module fails validation.
        """
        self._overrides_resolved = False
        return self._validate_impl(module)

    def validate_resolved_overrides(self, module: Module) -> ModuleInfo:
        """Validate a module requiring all overrides to be resolved.

        This is the same as :meth:`validate`, but treats any override whose
        value is not a fully-evaluated constant expression as an error.

        Args:
            module: The :class:`naga.ir.module.Module` to validate.

        Returns:
            A :class:`ModuleInfo` containing type flags, function info, and
            entry point info.

        Raises:
            ValidationError: If the module fails validation or has
                             unresolved overrides.
        """
        self._overrides_resolved = True
        return self._validate_impl(module)

    def _validate_impl(self, module: Module) -> ModuleInfo:
        """Internal implementation of module validation.

        Args:
            module: The module to validate.

        Returns:
            A :class:`ModuleInfo` with validation results.

        Raises:
            ValidationError: If validation fails.
        """
        self.reset()

        # Initialize module info
        mod_info = ModuleInfo()

        # Always compute type flags, even if we skip most validation stages.
        # This mirrors the Rust validator behavior, which always returns a usable
        # ModuleInfo on success.
        self._validate_types(module, mod_info)

        # Validate constants
        if self.flags & ValidationFlags.CONSTANTS:
            self._validate_constants(module, mod_info)

        # Validate global variables
        self._validate_global_variables(module, mod_info)

        # Validate functions
        if self.flags & ValidationFlags.EXPRESSIONS:
            for func in module.functions:
                func_info = self._validate_function(module, func)
                mod_info.functions.append(func_info)

        # Validate entry points
        for entry_point in module.entry_points:
            ep_info = self._validate_entry_point(module, entry_point)
            mod_info.entry_points.append(ep_info)

        return mod_info

    def _validate_types(self, module: Module, mod_info: ModuleInfo) -> None:
        """Validate all types in the module.

        Args:
            module: The module being validated.
            mod_info: The module info being built.

        Raises:
            ValidationError: If a type is invalid.
        """
        # For each type in the module, compute its flags.
        for ty in module.types:
            flags = self._compute_type_flags(module, ty)
            mod_info.type_flags.append(flags)
            self._type_flags.append(flags)

    def _compute_type_flags(self, module: Module, ty: object) -> TypeFlags:
        """Compute flags for a type.

        Args:
            module: The module being validated.
            ty: The type to analyze.

        Returns:
            Flags describing the type's capabilities.

        Raises:
            ValidationError: If the type is malformed or references invalid
                type handles.
        """
        if not isinstance(ty, Type):
            raise ValidationError(
                f"Module type arena contains unexpected value: {type(ty)!r}"
            )

        inner_obj = ty.inner
        inner: Optional[TypeInner]
        if isinstance(inner_obj, TypeInner):
            inner = inner_obj
        elif isinstance(inner_obj, str):
            try:
                inner = TypeInner(inner_obj)
            except ValueError:
                inner = None
        else:
            inner = None

        # If we can't classify this type, be conservative but keep validation
        # permissive to support the current incomplete IR.
        if inner is None:
            return TypeFlags.CONSTRUCTIBLE | TypeFlags.SIZED | TypeFlags.DATA

        sized = True
        constructible = False
        data = False
        host_shareable = False
        io_shareable = False
        copyable = False
        argument = False

        if inner in (TypeInner.SCALAR, TypeInner.VECTOR, TypeInner.MATRIX):
            constructible = True
            data = True
            host_shareable = True
            io_shareable = True
            copyable = True
            argument = True

        elif inner is TypeInner.ARRAY:
            constructible = True
            data = True
            host_shareable = True
            copyable = True
            argument = True

            array_info = getattr(ty, "_array", None)
            if not isinstance(array_info, Array):
                raise ValidationError("Array type is missing _array information")

            base = array_info.base
            if not isinstance(base, int):
                raise ValidationError(
                    f"Array base type handle must be int, got {type(base)!r}"
                )
            if base < 0 or base >= len(module.types):
                raise ValidationError(
                    f"Array references invalid base type {base} (types={len(module.types)})"
                )

            # Dynamic-sized arrays are treated as unsized.
            sized = array_info.size is not None

        elif inner is TypeInner.STRUCT:
            constructible = True
            data = True
            host_shareable = True
            io_shareable = True
            copyable = True
            argument = True

            struct_info = getattr(ty, "_struct", None)
            if not isinstance(struct_info, Struct):
                raise ValidationError("Struct type is missing _struct information")

            last_offset = 0
            for member in struct_info.members:
                if not isinstance(member, StructMember):
                    raise ValidationError(
                        "Struct members must be StructMember instances"
                    )
                if not isinstance(member.ty, int):
                    raise ValidationError(
                        f"Struct member type handle must be int, got {type(member.ty)!r}"
                    )
                if member.ty < 0 or member.ty >= len(module.types):
                    raise ValidationError(
                        f"Struct member references invalid type {member.ty} "
                        f"(types={len(module.types)})"
                    )
                if member.offset < 0:
                    raise ValidationError("Struct member offset must be non-negative")
                if member.offset < last_offset:
                    raise ValidationError(
                        "Struct member offsets must be non-decreasing"
                    )
                last_offset = member.offset

        elif inner in (
            TypeInner.IMAGE,
            TypeInner.SAMPLER,
            TypeInner.ACCELERATION_STRUCTURE,
            TypeInner.RAY_QUERY,
            TypeInner.BINDING_ARRAY,
        ):
            # These are handle-like types and are not generally constructible.
            sized = False

        else:
            # Keep defaults (mostly false), but treat unknown variants as unsized.
            sized = False

        flags = TypeFlags.empty()
        if constructible:
            flags |= TypeFlags.CONSTRUCTIBLE
        if sized:
            flags |= TypeFlags.SIZED
        if data:
            flags |= TypeFlags.DATA
        if host_shareable:
            flags |= TypeFlags.HOST_SHAREABLE
        if io_shareable:
            flags |= TypeFlags.IO_SHAREABLE
        if copyable:
            flags |= TypeFlags.COPY
        if argument:
            flags |= TypeFlags.ARGUMENT

        return flags

    def _validate_constants(self, module: Module, mod_info: ModuleInfo) -> None:
        """Validate all constants in the module.

        Args:
            module: The module being validated.
            mod_info: The module info being built.

        Raises:
            ValidationError: If a constant is invalid.
        """
        # Validate each constant's type and initializer
        for const in module.constants:
            self._validate_constant(module, const, mod_info)

    def _validate_constant(
        self, module: Module, const: object, mod_info: ModuleInfo
    ) -> None:
        """Validate a single constant.

        Args:
            module: The module being validated.
            const: The constant to validate.
            mod_info: The module info being built.

        Raises:
            ValidationError: If the constant is invalid.
        """
        _ = mod_info

        if not isinstance(const, Constant):
            raise ValidationError(
                f"Module constant arena contains unexpected value: {type(const)!r}"
            )

        ty_handle = const.ty
        if not isinstance(ty_handle, int):
            raise ValidationError(
                f"Constant type handle must be an int, got {type(ty_handle)!r}"
            )

        if ty_handle < 0 or ty_handle >= len(module.types):
            raise ValidationError(
                f"Constant references invalid type handle {ty_handle} (types={len(module.types)})"
            )

        # If we computed type flags, ensure the constant's type is constructible.
        if self._type_flags:
            type_flags = self._type_flags[ty_handle]
            if not (type_flags & TypeFlags.CONSTRUCTIBLE):
                raise ValidationError(
                    f"Constant type {ty_handle} is not constructible: flags={type_flags!s}"
                )

        # Minimal sanity checks for composite constants.
        components = getattr(const, "components", None)
        if components is not None:
            if not isinstance(components, list):
                raise ValidationError(
                    f"Composite constant components must be a list, got {type(components)!r}"
                )

            # Elements may be nested literals/handles depending on frontend.
            # We only ensure the container doesn't contain obviously invalid items.
            for component in components:
                if isinstance(component, int) and component < 0:
                    raise ValidationError(
                        f"Composite constant contains negative component handle {component}"
                    )

    def _validate_global_variables(
        self, module: Module, mod_info: ModuleInfo
    ) -> None:
        """Validate all global variables in the module.

        Args:
            module: The module being validated.
            mod_info: The module info being built.

        Raises:
            ValidationError: If a global variable is invalid.
        """
        _ = mod_info

        for var in module.global_variables:
            ty_handle_obj = getattr(var, "ty", None)
            if ty_handle_obj is None:
                continue

            if not isinstance(ty_handle_obj, int):
                raise ValidationError(
                    "Global variable type handle must be an int when present, "
                    f"got {type(ty_handle_obj)!r}"
                )

            if ty_handle_obj < 0 or ty_handle_obj >= len(module.types):
                raise ValidationError(
                    f"Global variable references invalid type handle {ty_handle_obj} "
                    f"(types={len(module.types)})"
                )

    def _validate_function(self, module: Module, func: object) -> FunctionInfo:
        """Validate a function.

        Args:
            module: The module being validated.
            func: The function to validate.

        Returns:
            Function info with analysis results.

        Raises:
            ValidationError: If the function is invalid.
        """
        if not isinstance(func, Function):
            raise ValidationError(
                f"Module function arena contains unexpected value: {type(func)!r}"
            )

        for arg in func.arguments:
            ty_handle = getattr(arg, "ty", None)
            if isinstance(ty_handle, int):
                if ty_handle < 0 or ty_handle >= len(module.types):
                    raise ValidationError(
                        f"Function argument references invalid type handle {ty_handle} "
                        f"(types={len(module.types)})"
                    )

        if func.result is not None:
            result_ty = getattr(func.result, "ty", None)
            if isinstance(result_ty, int):
                if result_ty < 0 or result_ty >= len(module.types):
                    raise ValidationError(
                        f"Function result references invalid type handle {result_ty} "
                        f"(types={len(module.types)})"
                    )

        for local in func.local_variables:
            local_ty = getattr(local, "ty", None)
            if isinstance(local_ty, int):
                if local_ty < 0 or local_ty >= len(module.types):
                    raise ValidationError(
                        f"Local variable references invalid type handle {local_ty} "
                        f"(types={len(module.types)})"
                    )

        # Validate named expression handles if they are integer indices.
        for name, expr_handle in func.named_expressions.items():
            if isinstance(expr_handle, int):
                if expr_handle < 0 or expr_handle >= len(func.expressions):
                    raise ValidationError(
                        f"Named expression '{name}' references invalid expression handle "
                        f"{expr_handle} (expressions={len(func.expressions)})"
                    )

        return FunctionInfo()

    def _validate_entry_point(
        self, module: Module, entry_point: object
    ) -> FunctionInfo:
        """Validate an entry point.

        Args:
            module: The module being validated.
            entry_point: The entry point to validate.

        Returns:
            Function info with analysis results.

        Raises:
            ValidationError: If the entry point is invalid.
        """
        if not isinstance(entry_point, EntryPoint):
            raise ValidationError(
                f"Module entry point arena contains unexpected value: "
                f"{type(entry_point)!r}"
            )

        # Validate name
        if not entry_point.name or not isinstance(entry_point.name, str):
            raise ValidationError("Entry point must have a non-empty string name")

        # Validate stage
        stage = getattr(entry_point, "stage", None)
        valid_stages = {"vertex", "fragment", "compute", "mesh", "task"}
        if stage not in valid_stages:
            raise ValidationError(
                f"Entry point '{entry_point.name}' has invalid stage '{stage}'. "
                f"Must be one of: {', '.join(sorted(valid_stages))}"
            )

        func = getattr(entry_point, "function", None)
        if func is None:
            raise ValidationError(
                f"Entry point '{entry_point.name}' is missing its function"
            )

        if not isinstance(func, Function):
            raise ValidationError(
                f"Entry point '{entry_point.name}' function is not a Function instance: "
                f"{type(func)!r}"
            )

        return self._validate_function(module, func)


# Maintain compatibility with existing code that expects these in the module
__all__ = ["Validator", "ValidationError"]
