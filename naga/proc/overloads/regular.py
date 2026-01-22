from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Iterator, Tuple, Optional, Any
from naga import ir, UniqueArena
from .overload_set import OverloadSet
from .rule import Rule, Conclusion, ConclusionValue, conclusion_for_frexp_modf
from .constructor_set import ConstructorSet, ConstructorSize
from .scalar_set import ScalarSet
from naga.proc import TypeResolution, TypeResolutionValue

class ConclusionRule(Enum):
    ArgumentType = auto()
    Scalar = auto()
    Frexp = auto()
    Modf = auto()
    U32 = auto()
    I32 = auto()
    Vec2F = auto()
    Vec4F = auto()
    Vec4I = auto()
    Vec4U = auto()

    def conclude(self, size: ConstructorSize, scalar: ir.Scalar) -> Conclusion:
        from naga.ir.type import VectorSize, ScalarKind
        from naga.ir.type import Scalar as IrScalar
        if self == ConclusionRule.ArgumentType:
            return ConclusionValue(size.to_inner(scalar))
        elif self == ConclusionRule.Scalar:
            return ConclusionValue(ir.TypeInner.scalar(scalar))
        elif self == ConclusionRule.Frexp:
            from naga.ir.operators import MathFunction
            return conclusion_for_frexp_modf(MathFunction.FREXP, size, scalar)
        elif self == ConclusionRule.Modf:
            from naga.ir.operators import MathFunction
            return conclusion_for_frexp_modf(MathFunction.MODF, size, scalar)
        elif self == ConclusionRule.U32:
            return ConclusionValue(ir.TypeInner.scalar(IrScalar(ScalarKind.UINT, 4)))
        elif self == ConclusionRule.I32:
            return ConclusionValue(ir.TypeInner.scalar(IrScalar(ScalarKind.SINT, 4)))
        elif self == ConclusionRule.Vec2F:
            return ConclusionValue(ir.TypeInner.vector(VectorSize.BI, IrScalar(ScalarKind.FLOAT, 4)))
        elif self == ConclusionRule.Vec4F:
            return ConclusionValue(ir.TypeInner.vector(VectorSize.QUAD, IrScalar(ScalarKind.FLOAT, 4)))
        elif self == ConclusionRule.Vec4I:
            return ConclusionValue(ir.TypeInner.vector(VectorSize.QUAD, IrScalar(ScalarKind.SINT, 4)))
        elif self == ConclusionRule.Vec4U:
            return ConclusionValue(ir.TypeInner.vector(VectorSize.QUAD, IrScalar(ScalarKind.UINT, 4)))
        raise ValueError(f"Unknown ConclusionRule: {self}")

@dataclass(frozen=True)
class Regular:
    arity: int
    constructors: ConstructorSet
    scalars: ScalarSet
    conclude: ConclusionRule

    @classmethod
    def empty_set(cls) -> 'Regular':
        return cls(0, ConstructorSet.EMPTY, ScalarSet(0), ConclusionRule.ArgumentType)

    def is_empty(self) -> bool:
        return self.constructors == ConstructorSet.EMPTY or self.scalars == ScalarSet(0)

    def min_arguments(self) -> int:
        if self.is_empty(): raise ValueError("OverloadSet is empty")
        return self.arity

    def max_arguments(self) -> int:
        if self.is_empty(): raise ValueError("OverloadSet is empty")
        return self.arity

    def arg(self, i: int, ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> 'Regular':
        if i >= self.arity:
            return self.empty_set()

        constructor = ConstructorSet.singleton(ty)
        ty_scalar = get_scalar_for_conversions(ty, types)
        
        scalars = ScalarSet.convertible_from(ty_scalar) if ty_scalar else ScalarSet(0)

        return Regular(
            arity=self.arity,
            constructors=self.constructors & constructor,
            scalars=self.scalars & scalars,
            conclude=self.conclude,
        )

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'Regular':
        return Regular(
            arity=self.arity,
            constructors=self.constructors,
            scalars=self.scalars & ScalarSet.CONCRETE,
            conclude=self.conclude,
        )

    def most_preferred(self) -> Rule:
        if self.is_empty(): raise ValueError("OverloadSet is empty")
        if not self.constructors.is_singleton():
             raise ValueError("Constructors not a singleton")
        
        size = self.constructors.size()
        scalar = self.scalars.most_general_scalar()
        return make_rule(self.arity, size, scalar, self.conclude)

    def members(self) -> Iterator[Tuple[ConstructorSize, ir.Scalar]]:
        for constructor in self.constructors.members():
            size = constructor.size()
            for singleton in self.scalars.members():
                yield (size, singleton.most_general_scalar())

    def rules(self) -> Iterator[Rule]:
        for size, scalar in self.members():
            yield make_rule(self.arity, size, scalar, self.conclude)

    def overload_list(self, gctx: Optional[Any] = None) -> List[Rule]:
        return list(self.rules())

    def allowed_args(self, i: int, gctx: Optional[Any] = None) -> List[TypeResolution]:
        if i >= self.arity:
            return []
        return [TypeResolutionValue(size.to_inner(scalar)) for size, scalar in self.members()]

    def for_debug(self, types: UniqueArena[ir.Type]) -> Any:
        return f"Regular(arity={self.arity}, constructors={self.constructors}, scalars={self.scalars}, conclude={self.conclude})"

def make_rule(arity: int, size: ConstructorSize, scalar: ir.Scalar, conclusion_rule: ConclusionRule) -> Rule:
    inner = size.to_inner(scalar)
    arg = TypeResolutionValue(inner)
    return Rule(
        arguments=[arg] * arity,
        conclusion=conclusion_rule.conclude(size, scalar)
    )

def get_scalar_for_conversions(ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> Optional[ir.Scalar]:
    from naga.ir.type import TypeInnerType
    if ty.type == TypeInnerType.SCALAR:
        return ty.scalar
    elif ty.type == TypeInnerType.VECTOR:
        return ty.vector.scalar
    elif ty.type == TypeInnerType.MATRIX:
        return ty.matrix.scalar
    elif ty.type == TypeInnerType.ATOMIC:
        return ty.atomic.scalar
    elif ty.type == TypeInnerType.VALUE_POINTER:
        return ty.value_pointer.scalar
    elif ty.type == TypeInnerType.POINTER:
        return get_scalar_for_conversions(types[ty.pointer.base].inner, types)
    return None
