class Writer:
    """
    Base class for shader writers (SPV, MSL, HLSL, GLSL).
    """
    def write(self, module: Any) -> Any:
        pass
