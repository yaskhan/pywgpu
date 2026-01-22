from dataclasses import dataclass
from typing import List, Optional, Union
from naga import ir, UniqueArena
from naga.proc import TypeResolution
from .constructor_set import ConstructorSize, ConstructorSizeScalar, ConstructorSizeVector, ConstructorSizeMatrix

@dataclass
class Rule:
    """A single type rule."""
    arguments: List[TypeResolution]
    conclusion: 'Conclusion'

@dataclass(frozen=True)
class ConclusionValue:
    inner: ir.TypeInner

@dataclass(frozen=True)
class ConclusionPredeclared:
    predeclared: ir.PredeclaredType

Conclusion = Union[ConclusionValue, ConclusionPredeclared]

def conclusion_for_frexp_modf(
    function: ir.MathFunction,
    size: ConstructorSize,
    scalar: ir.Scalar,
) -> Conclusion:
    from naga.ir.operators import MathFunction
    from naga.common.predeclared import PredeclaredType
    
    if isinstance(size, ConstructorSizeScalar):
        vec_size = None
    elif isinstance(size, ConstructorSizeVector):
        vec_size = size.size
    else:
        # FrexpModf only supports scalars and vectors
        raise ValueError("FrexpModf only supports scalars and vectors")

    # Adapt to simplified PredeclaredType if necessary
    # In Rust it has fields, but here it's an Enum
    if function == MathFunction.FREXP:
        return ConclusionPredeclared(PredeclaredType.FREXP_RESULT)
    elif function == MathFunction.MODF:
        return ConclusionPredeclared(PredeclaredType.MODF_RESULT)
    else:
        raise ValueError("FrexpModf only supports Frexp and Modf")

def conclusion_into_resolution(
    conclusion: Conclusion,
    special_types: Optional[ir.SpecialTypes] = None,
) -> TypeResolution:
    from naga.proc import TypeResolutionValue, TypeResolutionHandle
    if isinstance(conclusion, ConclusionValue):
        return TypeResolutionValue(conclusion.inner)
    elif isinstance(conclusion, ConclusionPredeclared):
        if special_types is None:
            raise MissingSpecialType()
        handle = special_types.predeclared_types.get(conclusion.predeclared)
        if handle is None:
            raise MissingSpecialType()
        return TypeResolutionHandle(handle)
    raise TypeError(f"Unknown Conclusion type: {type(conclusion)}")

class MissingSpecialType(Exception):
    """Special type is not registered within the module."""
    def __init__(self):
        super().__init__("Special type is not registered within the module")
