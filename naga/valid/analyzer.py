"""
Module analyzer.

Figures out the following properties:
- control flow uniformity
- texture/sampler pairs
- expression reference counts
"""

from __future__ import annotations

from typing import Optional, Set
from enum import IntFlag
from dataclasses import dataclass

from ..arena import Handle, Arena
from ..ir import Expression, Statement, Function
from .flags import ValidationFlags, ShaderStages
from .errors import ExpressionError, FunctionError
from .module_info import ModuleInfo


NonUniformResult = Optional[Handle[Expression]]

DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE = True


class UniformityRequirements(IntFlag):
    """
    Kinds of expressions that require uniform control flow.
    
    Attributes:
        WORK_GROUP_BARRIER: Work group barrier operations
        DERIVATIVE: Derivative operations (disabled for fragment stage if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE is True)
        IMPLICIT_LEVEL: Implicit level sampling (disabled for fragment stage if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE is True)
        COOP_OPS: Cooperative operations
    """
    WORK_GROUP_BARRIER = 0x1
    DERIVATIVE = 0 if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE else 0x2
    IMPLICIT_LEVEL = 0 if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE else 0x4
    COOP_OPS = 0x8


@dataclass
class Uniformity:
    """
    Uniform control flow characteristics.
    
    Attributes:
        non_uniform_result: A child expression with non-uniform result.
            This means, when the relevant invocations are scheduled on a compute unit,
            they have to use vector registers to store an individual value
            per invocation.
            
            Whenever the control flow is conditioned on such value,
            the hardware needs to keep track of the mask of invocations,
            and process all branches of the control flow.
            
            Any operations that depend on non-uniform results also produce non-uniform.
        requirements: If this expression requires uniform control flow, store the reason here.
    """
    non_uniform_result: NonUniformResult
    requirements: UniformityRequirements
    
    @classmethod
    def new(cls) -> Uniformity:
        """Create a new Uniformity with default values."""
        return cls(
            non_uniform_result=None,
            requirements=UniformityRequirements(0),
        )


class ExitFlags(IntFlag):
    """
    Exit flags for control flow analysis.
    
    Attributes:
        MAY_RETURN: Control flow may return from the function, which makes all the
            subsequent statements within the current function (only!)
            to be executed in a non-uniform control flow.
        MAY_KILL: Control flow may be killed. Anything after Statement.Kill is
            considered inside non-uniform context.
    """
    MAY_RETURN = 0x1
    MAY_KILL = 0x2


@dataclass
class FunctionUniformity:
    """
    Uniformity characteristics of a function.
    
    Attributes:
        result: The uniformity of the function result
        exit: Exit flags for the function
    """
    result: Uniformity
    exit: ExitFlags
    
    def __or__(self, other: FunctionUniformity) -> FunctionUniformity:
        """Combine two FunctionUniformity instances."""
        # Combine non_uniform_result
        non_uniform = self.result.non_uniform_result or other.result.non_uniform_result
        
        return FunctionUniformity(
            result=Uniformity(
                non_uniform_result=non_uniform,
                requirements=self.result.requirements | other.result.requirements,
            ),
            exit=self.exit | other.exit,
        )


@dataclass
class GlobalUse:
    """
    Information about how a global variable is used within a function.
    
    Attributes:
        query: Set of query operations performed on the global
        sampling: Sampling operations performed on the global
    """
    query: Set[Handle[Expression]]
    sampling: Set[Handle[Expression]]
    
    @classmethod
    def new(cls) -> GlobalUse:
        """Create a new GlobalUse with empty sets."""
        return cls(query=set(), sampling=set())


class ExpressionInfo:
    """
    Information about an expression gathered during analysis.
    
    Attributes:
        uniformity: Uniformity characteristics
        ref_count: Number of references to this expression
        assignable_global: Handle to assignable global variable (if any)
        ty: Type resolution for this expression
    """
    
    def __init__(self) -> None:
        """Initialize a new ExpressionInfo."""
        self.uniformity = Uniformity.new()
        self.ref_count = 0
        self.assignable_global: Optional[Handle] = None
        self.ty: Optional[Any] = None


class FunctionInfo:
    """
    Information about a function gathered during analysis.
    
    Attributes:
        uniformity: Uniformity characteristics of the function
        may_kill: Whether the function may kill invocations
        sampling_set: Set of texture/sampler pairs used
        global_uses: Information about global variable usage
        expressions: Information about expressions in the function
    """
    
    def __init__(self) -> None:
        """Initialize a new FunctionInfo."""
        self.uniformity = Uniformity.new()
        self.may_kill = False
        self.sampling_set: Set[tuple] = set()
        self.global_uses: dict[Handle, GlobalUse] = {}
        self.expressions: list[ExpressionInfo] = []


class Analyzer:
    """
    Module analyzer for computing uniformity and other analysis information.
    
    The analyzer processes a module and its functions to determine:
    - Control flow uniformity
    - Expression reference counts
    - Global variable usage
    - Texture/sampler pairing
    """
    
    def __init__(self, module_info: ModuleInfo, flags: ValidationFlags) -> None:
        """
        Initialize a new Analyzer.
        
        Args:
            module_info: Module information to populate
            flags: Validation flags controlling analysis behavior
        """
        self.module_info = module_info
        self.flags = flags
    
    def analyze_function(
        self,
        function: Function,
        module: Any,
    ) -> FunctionInfo:
        """
        Analyze a function to gather uniformity and usage information.
        
        Args:
            function: The function to analyze
            module: The module containing the function
            
        Returns:
            FunctionInfo containing analysis results
            
        Raises:
            FunctionError: If the function is invalid
        """
        # Placeholder implementation
        info = FunctionInfo()
        
        # Analyze expressions
        for expr_handle, expr in function.expressions.items():
            expr_info = ExpressionInfo()
            info.expressions.append(expr_info)
        
        # Analyze statements
        # This would recursively analyze the function body
        
        return info
    
    def analyze_entry_point(
        self,
        entry_point: Any,
        module: Any,
    ) -> FunctionInfo:
        """
        Analyze an entry point function.
        
        Args:
            entry_point: The entry point to analyze
            module: The module containing the entry point
            
        Returns:
            FunctionInfo containing analysis results
            
        Raises:
            FunctionError: If the entry point is invalid
        """
        return self.analyze_function(entry_point.function, module)


__all__ = [
    "UniformityRequirements",
    "Uniformity",
    "ExitFlags",
    "FunctionUniformity",
    "GlobalUse",
    "ExpressionInfo",
    "FunctionInfo",
    "Analyzer",
    "NonUniformResult",
]
