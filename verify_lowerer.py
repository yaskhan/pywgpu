import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from naga.front.wgsl import parse_str

def test_pipeline():
    source = """
    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
    }
    
    @vertex
    fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
        var out: VertexOutput;
        let x = f32(i32(in_vertex_index) - 1);
        let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
        out.position = vec4<f32>(x, y, 0.0, 1.0);
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        return out;
    }
    """
    
    print("Testing WGSL Parsing and Lowering...")
    try:
        module = parse_str(source)
        print("SUCCESS: Module created")
        print(f"Entry points: {[ep.name for ep in module.entry_points]}")
        print(f"Global variables: {len(module.global_variables)}")
        print(f"Types: {len(module.types)}")
        
        # Check for expected entry point
        assert len(module.entry_points) == 1
        assert module.entry_points[0].name == "vs_main"
        
        print("Pipeline verification passed!")
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
