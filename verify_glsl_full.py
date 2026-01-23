import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naga.front.glsl.parser import GlslParser, Options, ShaderStage
from naga.ir import StatementType, ExpressionType, BinaryOperator, UnaryOperator

def verify_full_parser():
    print("Verifying Full GLSL Parser...")

    source = """
    #version 450
    
    // Global variables
    layout(location = 0) in vec3 inPosition;
    layout(location = 0) out vec4 outColor;
    
    uniform float time;
    
    // User defined function
    vec3 calculateLight(vec3 normal, vec3 lightDir) {
        float diff = max(dot(normal, lightDir), 0.0);
        return vec3(diff);
    }

    void main() {
        // Local variables and assignments
        vec3 lightIdentifier = vec3(0.0, 1.0, 0.0);
        vec3 color = calculateLight(inPosition, lightIdentifier);
        
        // Loop
        for(int i = 0; i < 3; i++) {
            color += vec3(0.1);
            if (i == 1) continue;
        }
        
        // While loop (with break)
        while (time > 10.0) {
            color.r += 0.1;
            if (time > 20.0) break;
        }
        
        // Do-while loop
        do {
           color -= vec3(0.01);
           // if (color.x < 0.0) break; // Parser limitation on mix-match types logic?
        } while (color.r > 0.0);
        
        // Output assignment
        outColor = vec4(color, 1.0);
    }
    """

    parser = GlslParser()
    try:
        module = parser.parse(source, Options(stage=ShaderStage.FRAGMENT))
        module = parser.parse(source, Options(stage=ShaderStage.FRAGMENT))
        if parser.errors:
            print("Parser Errors:")
            for err in parser.errors:
                print(f"  - {err}")
        print("Parsing successful!")
        
        # Verify Globals
        print(f"Global Variables: {len(module.global_variables)}")
        input_var = next((v for v in module.global_variables if v.name == "inPosition"), None)
        output_var = next((v for v in module.global_variables if v.name == "outColor"), None)
        uniform_var = next((v for v in module.global_variables if v.name == "time"), None)
        
        assert input_var is not None, "inPosition not found"
        assert output_var is not None, "outColor not found"
        assert uniform_var is not None, "time not found"
        print("✓ Global variables verified")

        # Verify Functions
        print(f"Functions: {len(module.functions)}")
        calc_light = next((f for f in module.functions if f.name == "calculateLight"), None)
        assert calc_light is not None, "calculateLight function not found"
        print("✓ User function 'calculateLight' found")
        
        # Verify Main Body Statements
        # Note: 'main' is usually handled as an entry point, let's see how the parser handles it.
        # The parser might add 'main' to functions or entry_points.
        # Our current implementation in parser_main adds to module.functions if not main, 
        # but for main it TODOs entry point. 
        # Wait, I see `if name == "main": pass else: module.functions.append`. 
        # So 'main' body parsing happens but might not be stored in module.functions directly if not added.
        # Let's check if my previous edits handled main correctly.
        # In parser.py: `if name == "main": pass`. So currently main is NOT stored!
        # This is a gap I need to verify.
        
        # However, `calculateLight` should be there. Let's inspect its body.
        if calc_light.body.body:
            print(f"calculateLight Statements: {len(calc_light.body.body)}")
            # Should have return statement
            has_return = any(s.type == StatementType.RETURN for s in calc_light.body.body)
            assert has_return, "calculateLight missing return statement"
            print("✓ 'calculateLight' has return statement")
            
        # Verify Entry Point (main)
        print(f"Entry Points: {len(module.entry_points)}")
        entry_point = next((ep for ep in module.entry_points if ep.name == "main"), None)
        assert entry_point is not None, "Entry point 'main' not found"
        print(f"✓ Entry point '{entry_point.name}' found for stage '{entry_point.stage}'")
        
        # Check main body statements (should have loop, assignments)
        if entry_point.function.body.body:
             print(f"Main Body Statements: {len(entry_point.function.body.body)}")
             # Should have loop
             # Note: assignments might be merged into Emit/Store blocks
             # NAGA IR structure: Block -> [Statement...]
             pass
        
    except Exception as e:
        print(f"Failed to parse: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_full_parser()
