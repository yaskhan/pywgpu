import asyncio
import struct
import random
import math
from typing import List

import pywgpu
from framework import Example, run_example

MAX_BUNNIES = 1 << 16 # Reduce from 1<<20 for Python performance
BUNNY_SIZE = 40.0
GRAVITY = -9.8 * 100.0
MAX_VELOCITY = 750.0

def mat4_ortho(left, right, bottom, top, near, far):
    return [
        2.0 / (right - left), 0, 0, 0,
        0, 2.0 / (top - bottom), 0, 0,
        0, 0, 1.0 / (near - far), 0,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), near / (near - far), 1
    ]

class BunnymarkExample(Example):
    TITLE = "Bunnymark Example"

    async def init(self, config, adapter, device, queue):
        self.width, self.height = config.width, config.height
        self.bunnies = []
        
        # Shader
        shader_code = """
            struct Globals {
                mvp: mat4x4<f32>,
                size: vec2<f32>,
                _pad0: u32,
                _pad1: u32,
            };

            struct Locals {
                position: vec2<f32>,
                velocity: vec2<f32>,
                color: u32,
                _pad0: u32,
                _pad1: u32,
                _pad2: u32,
            };

            @group(0) @binding(0) var<uniform> globals: Globals;
            @group(1) @binding(0) var<uniform> locals: Locals;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
                @location(1) color: vec4<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                let tc = vec2<f32>(f32(vi & 1u), 0.5 * f32(vi & 2u));
                let offset = vec2<f32>(tc.x * globals.size.x - globals.size.x * 0.5, tc.y * globals.size.y - globals.size.y * 0.5);
                let pos = globals.mvp * vec4<f32>(locals.position + offset, 0.0, 1.0);
                
                let r = f32(locals.color & 0xFFu) / 255.0;
                let g = f32((locals.color >> 8u) & 0xFFu) / 255.0;
                let b = f32((locals.color >> 16u) & 0xFFu) / 255.0;
                let a = f32((locals.color >> 24u) & 0xFFu) / 255.0;
                let color = vec4<f32>(r, g, b, a);
                
                return VertexOutput(pos, tc, color);
            }

            @group(0) @binding(1) var tex: texture_2d<f32>;
            @group(0) @binding(2) var sam: sampler;

            @fragment
            fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
                let tex_color = textureSampleLevel(tex, sam, vertex.tex_coords, 0.0);
                return vertex.color * tex_color;
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Procedural Texture (Circular bunny)
        size = 64
        tex_data = bytearray()
        for y in range(size):
            for x in range(size):
                dx, dy = x - size//2, y - size//2
                dist = math.sqrt(dx*dx + dy*dy)
                alpha = 255 if dist < size//2 else 0
                tex_data.extend([255, 255, 255, alpha])
        
        self.texture = device.create_texture(pywgpu.TextureDescriptor(
            size=(size, size, 1),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=pywgpu.TextureFormat.rgba8unorm,
            usage=[pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.COPY_DST]
        ))
        queue.write_texture(
            pywgpu.TexelCopyTextureInfo(texture=self.texture),
            tex_data,
            pywgpu.TexelCopyBufferLayout(bytes_per_row=4*size),
            (size, size, 1)
        )

        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor(
            mag_filter=pywgpu.FilterMode.linear,
            min_filter=pywgpu.FilterMode.linear
        ))

        # Buffers
        self.global_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            size=16*4 + 16,
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        ))
        
        self.min_alignment = device.limits.min_uniform_buffer_offset_alignment
        # Bunny struct size in WGSL is 16 + 8 + 4 + 4(pad) = 32? 
        # Locals in shader: pos(8), vel(8), col(4), pads(12) = 32 bytes exactly?
        # Actually in the shader: position: vec2<f32>(8), velocity: vec2<f32>(8), color: u32(4), pads: 12 bytes = 32 bytes.
        # But we need to honor min_alignment.
        self.bunny_size_padded = max(32, self.min_alignment)
        self.local_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            size=MAX_BUNNIES * self.bunny_size_padded,
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        ))

        # Bind Groups
        self.global_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.vertex, ty=pywgpu.BufferBindingType.uniform),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.TextureBindingType.float()),
            pywgpu.BindGroupLayoutEntry(binding=2, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.SamplerBindingType.filtering)
        ]))
        self.local_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.vertex, ty=pywgpu.BufferBindingType.uniform(has_dynamic_offset=True, min_binding_size=32))
        ]))

        self.global_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.global_bgl,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.global_buffer.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.texture.create_view()),
                pywgpu.BindGroupEntry(binding=2, resource=self.sampler)
            ]
        ))
        self.local_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.local_bgl,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=pywgpu.BufferBinding(buffer=self.local_buffer, offset=0, size=32))
            ]
        ))

        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.global_bgl, self.local_bgl]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format, blend=pywgpu.BlendState.alpha_blending)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))

        self.spawn_bunnies(100)
        self.update_globals(queue)

    def spawn_bunnies(self, count):
        for _ in range(count):
            if len(self.bunnies) < MAX_BUNNIES:
                self.bunnies.append({
                    "pos": [0.0, self.height / 2.0],
                    "vel": [(random.random() - 0.5) * MAX_VELOCITY, 0.0],
                    "color": random.randint(0, 0xFFFFFFFF)
                })

    def update_globals(self, queue):
        proj = mat4_ortho(0, self.width, 0, self.height, -1, 1)
        data = struct.pack("16f", *proj) + struct.pack("2f 2I", BUNNY_SIZE, BUNNY_SIZE, 0, 0)
        queue.write_buffer(self.global_buffer, 0, data)

    def render(self, view, device, queue):
        # Update bunnies
        dt = 0.01
        packed_data = bytearray()
        for b in self.bunnies:
            b["pos"][0] += b["vel"][0] * dt
            b["pos"][1] += b["vel"][1] * dt
            b["vel"][1] += GRAVITY * dt
            
            if (b["vel"][0] > 0 and b["pos"][0] + BUNNY_SIZE/2 > self.width) or \
               (b["vel"][0] < 0 and b["pos"][0] - BUNNY_SIZE/2 < 0):
                b["vel"][0] *= -1.0
            if b["vel"][1] < 0 and b["pos"][1] - BUNNY_SIZE/2 < 0:
                b["vel"][1] *= -0.8 # Bounce
                b["pos"][1] = BUNNY_SIZE/2
                if random.random() < 0.2: b["vel"][1] = 500.0 # Hop
            
            # Pack Local struct: pos(8), vel(8), col(4), pad(12) = 32
            # Offset must be multiple of min_alignment
            data = struct.pack("2f 2f I 3I", *b["pos"], *b["vel"], b["color"], 0, 0, 0)
            packed_data.extend(data)
            if self.bunny_size_padded > 32:
                packed_data.extend(b'\x00' * (self.bunny_size_padded - 32))

        queue.write_buffer(self.local_buffer, 0, packed_data)

        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, self.global_bg)
        
        for i in range(len(self.bunnies)):
            offset = i * self.bunny_size_padded
            pass_enc.set_bind_group(1, self.local_bg, [offset])
            pass_enc.draw(vertices=range(4), instances=range(1))
            
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(BunnymarkExample))
