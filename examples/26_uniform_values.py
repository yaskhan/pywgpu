import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class UniformValuesExample(Example):
    TITLE = "Uniform Values Example (Mandelbrot Explorer)"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            struct AppState {
                pos_x: f32,
                pos_y: f32,
                zoom: f32,
                max_iterations: u32,
            }

            @group(0) @binding(0) var<uniform> app_state: AppState;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) coord: vec2<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                var vertices = array<vec2<f32>, 3>(
                    vec2<f32>(-1., 1.),
                    vec2<f32>(3.0, 1.),
                    vec2<f32>(-1., -3.0),
                );
                var out: VertexOutput;
                out.coord = vertices[vi];
                out.position = vec4<f32>(out.coord, 0.0, 1.0);
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let max_iter = app_state.max_iterations;
                let c = vec2<f32>(
                    in.coord.x * 3.0 / app_state.zoom + app_state.pos_x,
                    in.coord.y * 3.0 / app_state.zoom + app_state.pos_y
                );
                var z = c;
                var final_iter = max_iter;
                for (var i = 0u; i < max_iter; i++) {
                    let x2 = z.x * z.x;
                    let y2 = z.y * z.y;
                    if (x2 + y2 > 4.0) {
                        final_iter = i;
                        break;
                    }
                    z = vec2<f32>(x2 - y2 + c.x, 2.0 * z.x * z.y + c.y);
                }
                let val = f32(final_iter) / f32(max_iter);
                return vec4<f32>(val, val, val, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. State
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.zoom = 1.0
        self.max_iterations = 50
        
        # 3. Uniform Buffer
        self.uniform_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Uniform Buffer",
            size=16, # 4f
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        ))
        
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BufferBindingType.uniform())
        ]))
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[pywgpu.BindGroupEntry(binding=0, resource=self.uniform_buf.as_entire_binding())]
        ))

        # 4. Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState()
        ))

    def update(self, event):
        # We handle winit events here (mocked for now, but framework passes them)
        # Assuming event is a winit event object
        import winit
        if isinstance(event, winit.event.WindowEvent):
            if isinstance(event.event, winit.event.KeyboardInput):
                ki = event.event
                if ki.state == winit.event.ElementState.pressed:
                    step = 0.1 / self.zoom
                    if ki.logical_key == winit.keyboard.Key.Named(winit.keyboard.NamedKey.arrow_up):
                        self.pos_y += step
                    elif ki.logical_key == winit.keyboard.Key.Named(winit.keyboard.NamedKey.arrow_down):
                        self.pos_y -= step
                    elif ki.logical_key == winit.keyboard.Key.Named(winit.keyboard.NamedKey.arrow_left):
                        self.pos_x -= step
                    elif ki.logical_key == winit.keyboard.Key.Named(winit.keyboard.NamedKey.arrow_right):
                        self.pos_x += step
                    elif ki.text == "u":
                        self.max_iterations += 3
                    elif ki.text == "d":
                        self.max_iterations = max(1, self.max_iterations - 3)
            elif isinstance(event.event, winit.event.MouseWheel):
                delta = event.event.delta
                amount = 0.0
                if isinstance(delta, winit.event.MouseScrollDelta.LineDelta):
                    amount = delta.y
                elif isinstance(delta, winit.event.MouseScrollDelta.PixelDelta):
                    amount = delta.y / 20.0
                
                zoom_factor = 1.1 ** amount
                self.zoom *= zoom_factor

    def render(self, view, device, queue):
        # Update uniform buffer
        data = struct.pack("3fI", self.pos_x, self.pos_y, self.zoom, self.max_iterations)
        queue.write_buffer(self.uniform_buf, 0, data)
        
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0.5, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, self.bind_group)
        pass_enc.draw(vertices=range(3), instances=range(1))
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(UniformValuesExample))
 village
