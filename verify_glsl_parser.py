import sys
import os
from naga.front.glsl import GlslParser, Options, ShaderStage

def verify_parser():
    shader_source = """
    #version 450
    
    layout(location = 0) in vec3 a_Pos;
    layout(location = 1) in vec2 a_TexCoord;
    
    layout(set = 0, binding = 0) uniform Scene {
        mat4 u_ViewProj;
    };
    
    layout(location = 0) out vec2 v_TexCoord;
    
    void main() {
        v_TexCoord = a_TexCoord;
        gl_Position = u_ViewProj * vec4(a_Pos, 1.0);
    }
    """
    
    parser = GlslParser()
    options = Options(stage=ShaderStage.VERTEX, defines={})
    
    # Debug tokens
    from naga.front.glsl.lexer import Lexer
    lexer = Lexer(shader_source, {})
    tokens = lexer.tokenize()
    print(f"Tokens: {[t.value for t in tokens]}")
    
    try:
        module = parser.parse(shader_source, options)
        print("Successfully parsed GLSL shader!")
        
        print(f"Num Types: {len(module.types)}")
        print(f"Num Global Variables: {len(module.global_variables)}")
        
        for i, var in enumerate(module.global_variables):
            print(f"Global Var {i}: {var.name} (Space: {var.space})")
            
    except Exception as e:
        print(f"Failed to parse GLSL shader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_parser()
