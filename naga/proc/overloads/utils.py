from typing import Iterator, List, Tuple, TypeVar, Iterable, Optional
from naga import ir, UniqueArena
from naga.proc import TypeResolution, TypeResolutionValue
from .rule import Rule, ConclusionValue
from .list import ListOverloadSet

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

def float_scalars() -> List[ir.Scalar]:
    from naga.ir.type import Scalar, ScalarKind
    return [
        Scalar(ScalarKind.ABSTRACT_FLOAT, 0),
        Scalar(ScalarKind.FLOAT, 4),
        Scalar(ScalarKind.FLOAT, 2),
        Scalar(ScalarKind.FLOAT, 8),
    ]

def float_scalars_unimplemented_abstract() -> List[ir.Scalar]:
    from naga.ir.type import Scalar, ScalarKind
    return [
        Scalar(ScalarKind.FLOAT, 4),
        Scalar(ScalarKind.FLOAT, 2),
        Scalar(ScalarKind.FLOAT, 8),
    ]

def scalar_or_vecn(scalar: ir.Scalar) -> List[ir.TypeInner]:
    from naga.ir.type import VectorSize
    return [
        ir.TypeInner.scalar(scalar),
        ir.TypeInner.vector(VectorSize.BI, scalar),
        ir.TypeInner.vector(VectorSize.TRI, scalar),
        ir.TypeInner.vector(VectorSize.QUAD, scalar),
    ]

def rule(args: Iterable[ir.TypeInner], ret: ir.TypeInner) -> Rule:
    return Rule(
        arguments=[TypeResolutionValue(a) for a in args],
        conclusion=ConclusionValue(ret)
    )

def list_set(rules: Iterable[Rule]) -> ListOverloadSet:
    return ListOverloadSet.from_rules(list(rules))

def pairs(left: Iterable[T], right: Iterable[U]) -> Iterator[Tuple[T, U]]:
    right_list = list(right)
    for l in left:
        for r in right_list:
            yield (l, r)

def triples(left: Iterable[T], middle: Iterable[U], right: Iterable[V]) -> Iterator[Tuple[T, U, V]]:
    middle_list = list(middle)
    right_list = list(right)
    for l in left:
        for m in middle_list:
            for r in right_list:
                yield (l, m, r)

# Helper from type_methods.rs
def concrete_int_scalars() -> List[ir.Scalar]:
    from naga.ir.type import Scalar, ScalarKind
    return [
        Scalar(ScalarKind.SINT, 4),
        Scalar(ScalarKind.UINT, 4),
        Scalar(ScalarKind.SINT, 8),
        Scalar(ScalarKind.UINT, 8),
    ]

# Helper from type_methods.rs
def vector_sizes() -> List[ir.VectorSize]:
    from naga.ir.type import VectorSize
    return [VectorSize.BI, VectorSize.TRI, VectorSize.QUAD]

def non_struct_equivalent(a: ir.TypeInner, b: ir.TypeInner, types: ir.UniqueArena[ir.Type]) -> bool:
    # Basic implementation, should ideally match Naga's non_struct_equivalent
    # For now, rely on __eq__ which is available on dataclasses
    return a == b

def automatically_converts_to(a: ir.TypeInner, b: ir.TypeInner, types: ir.UniqueArena[ir.Type]) -> Optional[Tuple[ir.Scalar, ir.Scalar]]:
    # Basic implementation of WGSL feasible automatic conversion
    from naga.ir.type import TypeInnerType
    if a.type != b.type:
        return None
    
    if a.type == TypeInnerType.SCALAR:
        return automatically_converts_to_scalar(a.scalar, b.scalar)
    elif a.type == TypeInnerType.VECTOR:
        if a.vector.size == b.vector.size:
            return automatically_converts_to_scalar(a.vector.scalar, b.vector.scalar)
    elif a.type == TypeInnerType.MATRIX:
        if a.matrix.columns == b.matrix.columns and a.matrix.rows == b.matrix.rows:
            return automatically_converts_to_scalar(a.matrix.scalar, b.matrix.scalar)
    
    return None

def automatically_converts_to_scalar(a: ir.Scalar, b: ir.Scalar) -> Optional[Tuple[ir.Scalar, ir.Scalar]]:
    from naga.ir.type import ScalarKind
    if a == b:
        return (a, b)
    if a.kind == ScalarKind.ABSTRACT_INT:
        if b.kind in (ScalarKind.SINT, ScalarKind.UINT, ScalarKind.FLOAT, ScalarKind.ABSTRACT_FLOAT):
            return (a, b)
    if a.kind == ScalarKind.ABSTRACT_FLOAT:
        if b.kind == ScalarKind.FLOAT:
            return (a, b)
    return None

def is_abstract(ty: ir.TypeInner, types: ir.UniqueArena[ir.Type]) -> bool:
    from naga.ir.type import TypeInnerType, ScalarKind
    if ty.type == TypeInnerType.SCALAR:
        return ty.scalar.kind in (ScalarKind.ABSTRACT_INT, ScalarKind.ABSTRACT_FLOAT)
    elif ty.type == TypeInnerType.VECTOR:
        return ty.vector.scalar.kind in (ScalarKind.ABSTRACT_INT, ScalarKind.ABSTRACT_FLOAT)
    elif ty.type == TypeInnerType.MATRIX:
        return ty.matrix.scalar.kind in (ScalarKind.ABSTRACT_INT, ScalarKind.ABSTRACT_FLOAT)
    return False
