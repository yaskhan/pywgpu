#!/usr/bin/env python3
"""
Code generator for component-wise operations.

This script mimics the Rust macro gen_component_wise_extractor! from
naga/src/proc/constant_evaluator.rs to generate Python code for
component-wise operations.

Run this script to generate the component-wise function implementations.
"""

from typing import NamedTuple


class LiteralMapping(NamedTuple):
    """Mapping from Literal variant to Python type."""
    literal_name: str  # e.g., "AbstractFloat"
    mapping_name: str  # e.g., "AbstractFloat"
    py_type: str       # e.g., "float"


class ComponentWiseSpec(NamedTuple):
    """Specification for a component-wise extractor."""
    function_name: str  # e.g., "component_wise_scalar"
    target_enum: str    # e.g., "Scalar"
    literals: list[LiteralMapping]
    scalar_kinds: list[str]  # e.g., ["Float", "AbstractFloat", "Sint", "Uint", "AbstractInt"]


# Define the specifications matching the Rust macros
SPECS = [
    ComponentWiseSpec(
        function_name="component_wise_scalar",
        target_enum="Scalar",
        literals=[
            LiteralMapping("AbstractFloat", "AbstractFloat", "float"),
            LiteralMapping("F32", "F32", "float"),
            LiteralMapping("F16", "F16", "float"),
            LiteralMapping("AbstractInt", "AbstractInt", "int"),
            LiteralMapping("U32", "U32", "int"),
            LiteralMapping("I32", "I32", "int"),
            LiteralMapping("U64", "U64", "int"),
            LiteralMapping("I64", "I64", "int"),
        ],
        scalar_kinds=["Float", "AbstractFloat", "Sint", "Uint", "AbstractInt"],
    ),
    ComponentWiseSpec(
        function_name="component_wise_float",
        target_enum="Float",
        literals=[
            LiteralMapping("AbstractFloat", "Abstract", "float"),
            LiteralMapping("F32", "F32", "float"),
            LiteralMapping("F16", "F16", "float"),
        ],
        scalar_kinds=["Float", "AbstractFloat"],
    ),
    ComponentWiseSpec(
        function_name="component_wise_concrete_int",
        target_enum="ConcreteInt",
        literals=[
            LiteralMapping("U32", "U32", "int"),
            LiteralMapping("I32", "I32", "int"),
        ],
        scalar_kinds=["Sint", "Uint"],
    ),
    ComponentWiseSpec(
        function_name="component_wise_signed",
        target_enum="Signed",
        literals=[
            LiteralMapping("AbstractFloat", "AbstractFloat", "float"),
            LiteralMapping("AbstractInt", "AbstractInt", "int"),
            LiteralMapping("F32", "F32", "float"),
            LiteralMapping("F16", "F16", "float"),
            LiteralMapping("I32", "I32", "int"),
        ],
        scalar_kinds=["Sint", "AbstractInt", "Float", "AbstractFloat"],
    ),
]


def generate_enum_class(spec: ComponentWiseSpec) -> str:
    """Generate the enum-like class for extracted values."""
    lines = [
        f"class {spec.target_enum}:",
        f'    """Extracted {spec.target_enum.lower()} values for component-wise operations."""',
        "",
    ]
    
    # Generate variant classes
    for lit in spec.literals:
        lines.append(f"    @dataclass")
        lines.append(f"    class {lit.mapping_name}:")
        lines.append(f'        """Maps to Literal.{lit.literal_name}."""')
        lines.append(f"        values: list[{lit.py_type}]")
        lines.append("")
    
    return "\n".join(lines)


def generate_function(spec: ComponentWiseSpec) -> str:
    """Generate the component-wise function."""
    
    # Determine the handler type based on literals
    has_float = any(lit.py_type == "float" for lit in spec.literals)
    has_int = any(lit.py_type == "int" for lit in spec.literals)
    
    if has_float and has_int:
        handler_type = "float | int"
    elif has_float:
        handler_type = "float"
    else:
        handler_type = "int"
    
    lines = [
        f"def {spec.function_name}(",
        f"    eval: 'ConstantEvaluator',",
        f"    span: Span,",
        f"    exprs: list[Handle[Expression]],",
        f"    handler: Callable[[{spec.target_enum}], {spec.target_enum}],",
        f") -> Handle[Expression]:",
        f'    """Perform component-wise {spec.target_enum.lower()} operation on expressions.',
        f"",
        f"    If expressions are vectors of the same length, handler is called",
        f"    for each corresponding component of each vector.",
        f"",
        f"    Args:",
        f"        eval: Constant evaluator instance",
        f"        span: Span for error reporting",
        f"        exprs: List of expressions to process",
        f"        handler: Function to call on component values",
        f"",
        f"    Returns:",
        f"        Handle to the resulting expression",
        f'    """',
        f"    from naga.proc.constant_evaluator import ConstantEvaluatorError",
        f"    from naga import VectorSize",
        f"",
        f"    if not exprs:",
        f'        raise ConstantEvaluatorError("No expressions provided")',
        f"",
        f"    # Helper to evaluate zero values and splats",
        f"    def sanitize(expr: Handle[Expression]) -> Expression:",
        f"        return eval.expressions[eval.eval_zero_value_and_splat(expr, span)]",
        f"",
        f"    # Process first expression to determine type",
        f"    first_expr = sanitize(exprs[0])",
        f"",
        f"    # Match on expression type",
        f"    match first_expr:",
    ]
    
    # Generate literal cases
    for lit in spec.literals:
        lines.extend([
            f"        case Expression(type=ExpressionType.LITERAL, literal=Literal.{lit.literal_name}(value=x)):",
            f"            # Collect all values as {lit.mapping_name}",
            f"            values = [x]",
            f"            for expr_handle in exprs[1:]:",
            f"                expr = sanitize(expr_handle)",
            f"                match expr:",
            f"                    case Expression(type=ExpressionType.LITERAL, literal=Literal.{lit.literal_name}(value=v)):",
            f"                        values.append(v)",
            f"                    case _:",
            f'                        raise ConstantEvaluatorError("Invalid math argument")',
            f"",
            f"            # Call handler",
            f"            input_data = {spec.target_enum}.{lit.mapping_name}(values)",
            f"            result = handler(input_data)",
            f"",
            f"            # Convert result back to expression",
            f"            if len(result.values) == 1:",
            f"                new_expr = Expression(",
            f"                    type=ExpressionType.LITERAL,",
            f"                    literal=Literal.{lit.literal_name}(result.values[0]),",
            f"                )",
            f"            else:",
            f'                raise ConstantEvaluatorError("Unexpected result length")',
            f"",
            f"            return eval.register_evaluated_expr(new_expr, span)",
            f"",
        ])
    
    # Generate Compose case for vectors
    # Build scalar kind check string
    scalar_kinds_str = ", ".join(f'ScalarKind.{kind.upper()}' for kind in spec.scalar_kinds)
    
    lines.extend([
        f"        case Expression(type=ExpressionType.COMPOSE, compose_ty=ty, compose_components=components):",
        f"            # Handle vector composition",
        f"            type_inner = eval.types[ty].inner",
        f"            match type_inner:",
        f"                case TypeInner(type=TypeInnerType.VECTOR, vector_size=size, vector_scalar=scalar):",
        f"                    # Check if scalar kind matches",
        f"                    if scalar.kind not in [{scalar_kinds_str}]:",
        f'                        raise ConstantEvaluatorError("Invalid math argument")',
        f"",
        f"                    # Flatten first vector",
        f"                    from naga.proc.type_methods import flatten_compose",
        f"                    first_components = list(flatten_compose(ty, components, eval.expressions, eval.types))",
        f"",
        f"                    # Collect component groups from all expressions",
        f"                    component_groups = [first_components]",
        f"                    for expr_handle in exprs[1:]:",
        f"                        expr = sanitize(expr_handle)",
        f"                        match expr:",
        f"                            case Expression(type=ExpressionType.COMPOSE, compose_ty=expr_ty, compose_components=expr_comps):",
        f"                                if eval.types[expr_ty].inner != eval.types[ty].inner:",
        f'                                    raise ConstantEvaluatorError("Vector type mismatch")',
        f"                                component_groups.append(list(flatten_compose(expr_ty, expr_comps, eval.expressions, eval.types)))",
        f"                            case _:",
        f'                                raise ConstantEvaluatorError("Invalid math argument")',
        f"",
        f"                    # Process each component index",
        f"                    new_components = []",
        f"                    for idx in range(int(size)):",
        f"                        # Gather components at this index from all vectors",
        f"                        group = [comp_list[idx] for comp_list in component_groups]",
        f"                        # Recursively call this function for scalar components",
        f"                        result_handle = {spec.function_name}(eval, span, group, handler)",
        f"                        new_components.append(result_handle)",
        f"",
        f"                    # Create new Compose expression",
        f"                    new_expr = Expression(",
        f"                        type=ExpressionType.COMPOSE,",
        f"                        compose_ty=ty,",
        f"                        compose_components=new_components,",
        f"                    )",
        f"                    return eval.register_evaluated_expr(new_expr, span)",
        f"",
        f"                case _:",
        f'                    raise ConstantEvaluatorError("Invalid math argument")',
        f"",
        f"        case _:",
        f'            raise ConstantEvaluatorError("Invalid math argument")',
        f"",
    ])
    
    return "\n".join(lines)


def generate_all() -> str:
    """Generate all component-wise functions."""
    lines = [
        '"""',
        'Generated component-wise operations.',
        '',
        'This file is AUTO-GENERATED by generate_component_wise.py',
        'DO NOT EDIT MANUALLY!',
        '"""',
        '',
        'from __future__ import annotations',
        '',
        'from dataclasses import dataclass',
        'from typing import TYPE_CHECKING, Callable',
        '',
        'from naga import (',
        '    Expression, ExpressionType, Handle, Literal, Span,',
        '    ScalarKind, TypeInner, TypeInnerType,',
        ')',
        '',
        'if TYPE_CHECKING:',
        '    from naga.proc.constant_evaluator import ConstantEvaluator',
        '',
        '',
        '# ============================================================================',
        '# Extracted value types',
        '# ============================================================================',
        '',
    ]
    
    # Generate enum classes
    for spec in SPECS:
        lines.append(generate_enum_class(spec))
        lines.append("")
    
    lines.extend([
        '',
        '# ============================================================================',
        '# Component-wise functions',
        '# ============================================================================',
        '',
    ])
    
    # Generate functions
    for spec in SPECS:
        lines.append(generate_function(spec))
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating component-wise operations...")
    code = generate_all()
    
    output_file = "component_wise_generated.py"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"Generated {output_file}")
    print(f"Total lines: {len(code.splitlines())}")
