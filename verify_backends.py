
import sys
import os

# Add the project root to sys.path
sys.path.append('c:\\Users\\Professional\\Documents\\GitHub\\pywgpu')

try:
    from naga.back.msl import Writer as MSLWriter, Options as MSLOptions, ShaderStage as MSLStage
    from naga.back.hlsl import Writer as HLSLWriter, Options as HLSLOptions, ShaderStage as HLSLStage
    from naga.ir.module import Module
    from naga.valid import ModuleInfo
    
    print("Successfully imported MSL and HLSL backends.")
    
    # Mock objects for initialization test
    module = Module()
    info = ModuleInfo()
    msl_options = MSLOptions()
    hlsl_options = HLSLOptions()
    
    msl_writer = MSLWriter("", module, info, msl_options, "main", MSLStage.Vertex)
    hlsl_writer = HLSLWriter("", module, info, hlsl_options)
    
    print("Successfully initialized MSL and HLSL writers.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
