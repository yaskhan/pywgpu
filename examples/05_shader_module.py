"""
Shader Module Example

Demonstrates shader module creation, compilation, and basic shader concepts using pywgpu.
This example shows how to create vertex and fragment shaders and use them in rendering.
"""

import asyncio

import pywgpu


async def main():
    print("Shader Module example using pywgpu")
    print("This example demonstrates shader creation and compilation")
    
    # Initialize the wgpu instance
    instance = pywgpu.Instance()
    
    # Request an adapter (physical GPU)
    adapter = await instance.request_adapter()
    
    if not adapter:
        print("No suitable adapter found")
        return
    
    print(f"Running on Adapter: {adapter.get_info()}")
    
    # Create a device and queue from the adapter
    device, queue = await adapter.request_device(pywgpu.DeviceDescriptor(
        label=None,
        required_features=[],
        required_limits=pywgpu.Limits.default(),
        experimental_features=pywgpu.ExperimentalFeatures.disabled(),
        memory_hints=pywgpu.MemoryHints.memory_usage,
        trace=pywgpu.Trace.off(),
    ))

    
    # 1. Create a simple vertex and fragment shader pair
    print("\\n1. Creating basic vertex and fragment shaders...")
    
    basic_shader_code = """
        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
        };
        
        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4<f32>(input.position, 1.0);
            output.color = input.color;
            return output;
        }
        
        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            return vec4<f32>(input.color, 1.0);
        }
    """
    
    basic_shader_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(
        label="Basic Shader",
        wgsl_code=basic_shader_code,
    ))
    print("  - Basic vertex/fragment shader pair created successfully")
    
    # 2. Create a compute shader
    print("\\n2. Creating compute shader...")
    
    compute_shader_code = """
        struct ComputeInput {
            @group(0) @binding(0) data: array<f32>,
            @group(0) @binding(1) result: array<f32>,
        };
        
        @compute @workgroup_size(64)
        fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index < arrayLength(&data)) {
                result[index] = data[index] * 2.0;
            }
        }
    """
    
    compute_shader_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(
        label="Compute Shader",
        wgsl_code=compute_shader_code,
    ))
    print("  - Compute shader created successfully")
    
    # 3. Create a more complex shader with uniforms
    print("\\n3. Creating shader with uniform data...")
    
    uniform_shader_code = """
        struct UniformData {
            transform: mat4x4<f32>,
            color_multiplier: vec4<f32>,
        };
        
        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
        };
        
        @group(0) @binding(0) var<uniform> uniforms: UniformData;
        
        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = uniforms.transform * vec4<f32>(input.position, 1.0);
            output.color = input.color * uniforms.color_multiplier.rgb;
            return output;
        }
        
        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            return vec4<f32>(input.color, 1.0);
        }
    """
    
    uniform_shader_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(
        label="Uniform Shader",
        wgsl_code=uniform_shader_code,
    ))
    print("  - Shader with uniform data created successfully")
    
    # 4. Demonstrate shader compilation info
    print("\\n4. Shader compilation information:")
    print(f"  - Basic shader: {basic_shader_module}")
    print(f"  - Compute shader: {compute_shader_module}")
    print(f"  - Uniform shader: {uniform_shader_module}")
    
    # 5. Show shader features and capabilities
    print("\\n5. Device shader capabilities:")
    features = device.features()
    limits = device.limits()
    print(f"  - Device features: {type(features).__name__}")
    print(f"  - Device limits: {type(limits).__name__}")
    
    print("\\nShader module operations completed successfully!")
    print("\\nThis example demonstrated:")
    print("- Basic vertex and fragment shader creation")
    print("- Compute shader module creation")
    print("- Shader modules with uniform data")
    print("- WGSL shader language syntax")
    print("- Shader binding and layout concepts")
    print("- Different shader stages (vertex, fragment, compute)")


if __name__ == "__main__":
    asyncio.run(main())