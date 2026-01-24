from typing import Any, Optional
from io import StringIO


class Options:
    """Configuration options for the dot backend."""
    
    def __init__(self, cfg_only: bool = False):
        """
        Initialize DOT writer options.
        
        Args:
            cfg_only: Only emit function bodies (control flow graph only)
        """
        self.cfg_only = cfg_only


def _name(name: Optional[str]) -> str:
    """Get name or empty string."""
    return name if name else ""


def write(module: Any, mod_info: Optional[Any] = None, options: Optional[Options] = None) -> str:
    """
    Write shader module to a DOT graph string.
    
    Args:
        module: The Naga IR module
        mod_info: Optional module validation information
        options: DOT writer options
        
    Returns:
        DOT graph representation as string
    """
    if options is None:
        options = Options()
    
    output = StringIO()
    output.write("digraph Module {\n")
    
    if not options.cfg_only:
        output.write("\tsubgraph cluster_globals {\n")
        output.write('\t\tlabel="Globals"\n')
        for handle in module.global_variables:
            var = module.global_variables[handle]
            output.write(
                f'\t\t_global{handle.index} [ shape=hexagon label="{handle.index} '
                f'{var.space}/\'{_name(var.name)}\'" ]\n'
            )
        output.write("\t}\n")
    
    for handle in module.functions:
        fun = module.functions[handle]
        prefix = f"_function{handle.index}"
        output.write(f"\tsubgraph cluster_{prefix} {{\n")
        output.write(f'\t\tlabel="Function{handle.index}/\'{_name(fun.name)}\'"\n')
        info = mod_info[handle] if mod_info else None
        _write_fun(output, prefix, fun, info, options)
        output.write("\t}\n")
    
    for ep_index, ep in enumerate(module.entry_points):
        prefix = f"ep{ep_index}"
        output.write(f"\tsubgraph cluster_{prefix} {{\n")
        output.write(f'\t\tlabel="{ep.stage}/\'{ep.name}\'"\n')
        info = mod_info.get_entry_point(ep_index) if mod_info else None
        _write_fun(output, prefix, ep.function, info, options)
        output.write("\t}\n")
    
    output.write("}\n")
    return output.getvalue()


def _write_fun(
    output: StringIO,
    prefix: str,
    fun: Any,
    info: Optional[Any],
    options: Options,
) -> None:
    """Write a function to the DOT output."""
    if not options.cfg_only:
        output.write("\t\tsubgraph cluster_" + prefix + "_expressions {\n")
        output.write('\t\t\tlabel="Expressions"\n')
        for handle in fun.expressions:
            expr = fun.expressions[handle]
            output.write(
                f'\t\t\t{prefix}_expr{handle.index} [ label="{handle.index}: '
                f'{type(expr).__name__}" ]\n'
            )
        output.write("\t\t}\n")
    
    output.write("\t\tsubgraph cluster_" + prefix + "_block {\n")
    output.write('\t\t\tlabel="Body"\n')
    _write_block(output, prefix, fun.body, 0)
    output.write("\t\t}\n")


def _write_block(output: StringIO, prefix: str, block: list, indent: int) -> None:
    """Write a block of statements to the DOT output."""
    indent_str = "\t" * (indent + 3)
    for i, stmt in enumerate(block):
        node_id = f"{prefix}_stmt{i}"
        stmt_type = type(stmt).__name__
        output.write(f'{indent_str}{node_id} [ label="{stmt_type}" ]\n')


__all__ = ["Options", "write"]
