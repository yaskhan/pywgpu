from typing import Any, List
from .handle_set_map import Handle, HandleSet, HandleMap
from .functions import FunctionTracer


class StatementTracer:
    """
    Traces statements to determine which expressions are used.
    """

    def __init__(self, function_tracer: FunctionTracer) -> None:
        self.function_tracer = function_tracer

    def trace_block(self, block: List[Any]) -> None:
        """Trace a block of statements."""
        worklist: List[List[Any]] = [block]

        while worklist:
            last = worklist.pop()
            for stmt in last:
                stmt_type = type(stmt).__name__

                if stmt_type == "Emit":
                    pass
                elif stmt_type == "Block":
                    worklist.append(stmt.block)
                elif stmt_type == "If":
                    self.function_tracer.expressions_used.insert(stmt.condition)
                    worklist.append(stmt.accept)
                    worklist.append(stmt.reject)
                elif stmt_type == "Switch":
                    self.function_tracer.expressions_used.insert(stmt.selector)
                    for case in stmt.cases:
                        worklist.append(case.body)
                elif stmt_type == "Loop":
                    if stmt.break_if:
                        self.function_tracer.expressions_used.insert(stmt.break_if)
                    worklist.append(stmt.body)
                    worklist.append(stmt.continuing)
                elif stmt_type == "Return":
                    if stmt.value:
                        self.function_tracer.expressions_used.insert(stmt.value)
                elif stmt_type == "Store":
                    self.function_tracer.expressions_used.insert(stmt.pointer)
                    self.function_tracer.expressions_used.insert(stmt.value)
                elif stmt_type == "ImageStore":
                    self.function_tracer.expressions_used.insert(stmt.image)
                    self.function_tracer.expressions_used.insert(stmt.coordinate)
                    if stmt.array_index:
                        self.function_tracer.expressions_used.insert(stmt.array_index)
                    self.function_tracer.expressions_used.insert(stmt.value)
                elif stmt_type == "Atomic":
                    self.function_tracer.expressions_used.insert(stmt.pointer)
                    self.trace_atomic_function(stmt.fun)
                    self.function_tracer.expressions_used.insert(stmt.value)
                    if stmt.result:
                        self.function_tracer.expressions_used.insert(stmt.result)
                elif stmt_type == "ImageAtomic":
                    self.function_tracer.expressions_used.insert(stmt.image)
                    self.function_tracer.expressions_used.insert(stmt.coordinate)
                    if stmt.array_index:
                        self.function_tracer.expressions_used.insert(stmt.array_index)
                    self.function_tracer.expressions_used.insert(stmt.value)
                elif stmt_type == "WorkGroupUniformLoad":
                    self.function_tracer.expressions_used.insert(stmt.pointer)
                    self.function_tracer.expressions_used.insert(stmt.result)
                elif stmt_type == "Call":
                    self.function_tracer.trace_call(stmt.function)
                    for expr in stmt.arguments:
                        self.function_tracer.expressions_used.insert(expr)
                    if stmt.result:
                        self.function_tracer.expressions_used.insert(stmt.result)
                elif stmt_type == "RayQuery":
                    self.function_tracer.expressions_used.insert(stmt.query)
                    self.trace_ray_query_function(stmt.fun)
                elif stmt_type == "SubgroupBallot":
                    if stmt.predicate:
                        self.function_tracer.expressions_used.insert(stmt.predicate)
                    self.function_tracer.expressions_used.insert(stmt.result)
                elif stmt_type == "SubgroupCollectiveOperation":
                    self.function_tracer.expressions_used.insert(stmt.argument)
                    self.function_tracer.expressions_used.insert(stmt.result)
                elif stmt_type == "SubgroupGather":
                    if hasattr(stmt.mode, "index") and stmt.mode.index:
                        self.function_tracer.expressions_used.insert(stmt.mode.index)
                    self.function_tracer.expressions_used.insert(stmt.argument)
                    self.function_tracer.expressions_used.insert(stmt.result)
                elif stmt_type == "CooperativeStore":
                    self.function_tracer.expressions_used.insert(stmt.target)
                    self.function_tracer.expressions_used.insert(stmt.data.pointer)
                    self.function_tracer.expressions_used.insert(stmt.data.stride)
                # Trivial statements: Break, Continue, Kill, ControlBarrier, MemoryBarrier, Return with no value

    def trace_atomic_function(self, fun: Any) -> None:
        """Trace an atomic function."""
        fun_type = type(fun).__name__

        if fun_type == "Exchange":
            if fun.compare:
                self.function_tracer.expressions_used.insert(fun.compare)

    def trace_ray_query_function(self, fun: Any) -> None:
        """Trace a ray query function."""
        fun_type = type(fun).__name__

        if fun_type == "Initialize":
            self.function_tracer.expressions_used.insert(fun.acceleration_structure)
            self.function_tracer.expressions_used.insert(fun.descriptor)
        elif fun_type == "Proceed":
            if fun.result:
                self.function_tracer.expressions_used.insert(fun.result)
        elif fun_type == "GenerateIntersection":
            self.function_tracer.expressions_used.insert(fun.hit_t)


class StatementCompactor:
    """
    Compacts statements by adjusting handles.
    """

    def __init__(self, expressions: HandleMap, functions: HandleMap) -> None:
        self.expressions = expressions
        self.functions = functions

    def adjust_body(self, block: List[Any]) -> None:
        """
        Adjust statements in a block.
        
        Adjusts expressions and function calls in all statements.
        """
        worklist: List[List[Any]] = [block]
        
        while worklist:
            current_block = worklist.pop()
            for stmt in current_block:
                stmt_type = type(stmt).__name__
                
                if stmt_type == "Emit":
                    # Adjust the range of emitted expressions
                    if hasattr(stmt, 'range'):
                        self.expressions.adjust_range(stmt.range, None)
                
                elif stmt_type == "Block":
                    worklist.append(stmt.block)
                
                elif stmt_type == "If":
                    self.expressions.adjust(stmt.condition)
                    worklist.append(stmt.accept)
                    worklist.append(stmt.reject)
                
                elif stmt_type == "Switch":
                    self.expressions.adjust(stmt.selector)
                    for case in stmt.cases:
                        worklist.append(case.body)
                
                elif stmt_type == "Loop":
                    if stmt.break_if is not None:
                        self.expressions.adjust(stmt.break_if)
                    worklist.append(stmt.body)
                    worklist.append(stmt.continuing)
                
                elif stmt_type == "Return":
                    if stmt.value is not None:
                        self.expressions.adjust(stmt.value)
                
                elif stmt_type == "Store":
                    self.expressions.adjust(stmt.pointer)
                    self.expressions.adjust(stmt.value)
                
                elif stmt_type == "ImageStore":
                    self.expressions.adjust(stmt.image)
                    self.expressions.adjust(stmt.coordinate)
                    if stmt.array_index is not None:
                        self.expressions.adjust(stmt.array_index)
                    self.expressions.adjust(stmt.value)
                
                elif stmt_type == "Atomic":
                    self.expressions.adjust(stmt.pointer)
                    self.adjust_atomic_function(stmt.fun)
                    self.expressions.adjust(stmt.value)
                    if stmt.result is not None:
                        self.expressions.adjust(stmt.result)
                
                elif stmt_type == "ImageAtomic":
                    self.expressions.adjust(stmt.image)
                    self.expressions.adjust(stmt.coordinate)
                    if stmt.array_index is not None:
                        self.expressions.adjust(stmt.array_index)
                    self.expressions.adjust(stmt.value)
                
                elif stmt_type == "WorkGroupUniformLoad":
                    self.expressions.adjust(stmt.pointer)
                    self.expressions.adjust(stmt.result)
                
                elif stmt_type == "Call":
                    self.functions.adjust(stmt.function)
                    for expr in stmt.arguments:
                        self.expressions.adjust(expr)
                    if stmt.result is not None:
                        self.expressions.adjust(stmt.result)
                
                elif stmt_type == "RayQuery":
                    self.expressions.adjust(stmt.query)
                    self.adjust_ray_query_function(stmt.fun)
                
                elif stmt_type == "SubgroupBallot":
                    if stmt.predicate is not None:
                        self.expressions.adjust(stmt.predicate)
                    self.expressions.adjust(stmt.result)
                
                elif stmt_type == "SubgroupCollectiveOperation":
                    self.expressions.adjust(stmt.argument)
                    self.expressions.adjust(stmt.result)
                
                elif stmt_type == "SubgroupGather":
                    # Adjust mode if it contains an index
                    if hasattr(stmt.mode, 'index') and stmt.mode.index is not None:
                        self.expressions.adjust(stmt.mode.index)
                    self.expressions.adjust(stmt.argument)
                    self.expressions.adjust(stmt.result)
                
                elif stmt_type == "CooperativeStore":
                    self.expressions.adjust(stmt.target)
                    self.expressions.adjust(stmt.data.pointer)
                    self.expressions.adjust(stmt.data.stride)
                
                # Trivial statements: Break, Continue, Kill, ControlBarrier, MemoryBarrier

    def adjust_atomic_function(self, fun: Any) -> None:
        """Adjust handles in an atomic function."""
        if hasattr(fun, 'compare') and fun.compare is not None:
            self.expressions.adjust(fun.compare)

    def adjust_ray_query_function(self, fun: Any) -> None:
        """Adjust handles in a ray query function."""
        fun_type = type(fun).__name__
        
        if fun_type == "Initialize":
            self.expressions.adjust(fun.acceleration_structure)
            self.expressions.adjust(fun.descriptor)
        elif fun_type == "Proceed":
            self.expressions.adjust(fun.result)
        elif fun_type == "GenerateIntersection":
            self.expressions.adjust(fun.hit_t)
