from .. import Writer
from ... import Module, Function # Import Module and Function from the naga package
from io import StringIO

class DotWriter(Writer):
    """
    DOT Writer (Graphviz).
    """
    def __init__(self):
        self._output = StringIO()

    def write(self, module: Module, info: Any) -> str:
        self._output.write("digraph G {\n")
        self._output.write("  rankdir=\"LR\";\n") # Left to Right for better readability

        for i, func in enumerate(module.functions):
            self._output.write(f"  node{id(func)} [label=\"Function {i}\"];\n")

        # This is a very basic example. A full implementation would
        # traverse the IR and add nodes and edges for all elements.

        self._output.write("}\n")
        return self.finish()

    def finish(self) -> str:
        return self._output.getvalue()
