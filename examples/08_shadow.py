import asyncio
import struct
import math
from typing import List, Tuple

import pywgpu
from framework import Example, run_example

# Constants
MAX_LIGHTS = 10
SHADOW_FORMAT = pywgpu.TextureFormat.depth32float
SHADOW_SIZE = (512, 512, MAX_LIGHTS)
DEPTH_FORMAT = pywgpu.TextureFormat.depth32float

def create_cube() -> Tuple[bytes, bytes, int]:
    def vertex(pos, nor):
        # pos: i8x3, nor: i8x3
        # In WGSL they are read as i32/f32, so we pack as signed bytes
        return struct.pack("4b4b", pos[0], pos[1], pos[2], 1, nor[0], nor[1], nor[2], 0)

    vertices = [
        # top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0, 1]), vertex([1, -1, 1], [0, 0, 1]), vertex([1, 1, 1], [0, 0, 1]), vertex([-1, 1, 1], [0, 0, 1]),
        # bottom (0, 0, -1)
        vertex([-1, 1, -1], [0, 0, -1]), vertex([1, 1, -1], [0, 0, -1]), vertex([1, -1, -1], [0, 0, -1]), vertex([-1, -1, -1], [0, 0, -1]),
        # right (1, 0, 0)
        vertex([1, -1, -1], [1, 0, 0]), vertex([1, 1, -1], [1, 0, 0]), vertex([1, 1, 1], [1, 0, 0]), vertex([1, -1, 1], [1, 0, 0]),
        # left (-1, 0, 0)
        vertex([-1, -1, 1], [-1, 0, 0]), vertex([-1, 1, 1], [-1, 0, 0]), vertex([-1, 1, -1], [-1, 0, 0]), vertex([-1, -1, -1], [-1, 0, 0]),
        # front (0, 1, 0)
        vertex([1, 1, -1], [0, 1, 0]), vertex([-1, 1, -1], [0, 1, 0]), vertex([-1, 1, 1], [0, 1, 0]), vertex([1, 1, 1], [0, 1, 0]),
        # back (0, -1, 0)
        vertex([1, -1, 1], [0, -1, 0]), vertex([-1, -1, 1], [0, -1, 0]), vertex([-1, -1, -1], [0, -1, 0]), vertex([1, -1, -1], [0, -1, 0]),
    ]
    
    indices = [
        0, 1, 2, 2, 3, 0, # top
        4, 5, 6, 6, 7, 4, # bottom
        8, 9, 10, 10, 11, 8, # right
        12, 13, 14, 14, 15, 12, # left
        16, 17, 18, 18, 19, 16, # front
        20, 21, 22, 22, 23, 20, # back
    ]
    
    return b"".join(vertices), struct.pack(f"{len(indices)}H", *indices), len(indices)

def create_plane(size: int) -> Tuple[bytes, bytes, int]:
    def vertex(pos, nor):
        return struct.pack("4b4b", pos[0], pos[1], pos[2], 1, nor[0], nor[1], nor[2], 0)
    
    vertices = [
        vertex([size, -size, 0], [0, 0, 1]),
        vertex([size, size, 0], [0, 0, 1]),
        vertex([-size, -size, 0], [0, 0, 1]),
        vertex([-size, size, 0], [0, 0, 1]),
    ]
    indices = [0, 1, 2, 2, 1, 3]
    return b"".join(vertices), struct.pack(f"{len(indices)}H", *indices), len(indices)

def mat4_identity():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

def mat4_perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2)
    nf = 1.0 / (near - far)
    return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, (2 * far * near) * nf, 0
    ]

def mat4_lookat(eye, target, up):
    def normalize(v):
        l = math.sqrt(sum(x*x for x in v))
        return [x/l for x in v]
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    z = normalize([eye[i] - target[i] for i in range(3)])
    x = normalize(cross(up, z))
    y = cross(z, x)
    
    return [
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
    ]

def mat4_mul(a, b):
    res = [0]*16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                res[i*4 + j] += a[i*4 + k] * b[k*4 + j]
    return res

class ShadowExample(Example):
    TITLE = "Shadow Example"

    async def init(self, config, adapter, device, queue):
        self.entities = []
        
        # Geometry
        cube_v, cube_i, cube_ic = create_cube()
        self.cube_v_buf = device.create_buffer_init(label="Cube VB", contents=cube_v, usage=pywgpu.BufferUsages.VERTEX)
        self.cube_i_buf = device.create_buffer_init(label="Cube IB", contents=cube_i, usage=pywgpu.BufferUsages.INDEX)
        
        plane_v, plane_i, plane_ic = create_plane(7)
        self.plane_v_buf = device.create_buffer_init(label="Plane VB", contents=plane_v, usage=pywgpu.BufferUsages.VERTEX)
        self.plane_i_buf = device.create_buffer_init(label="Plane IB", contents=plane_i, usage=pywgpu.BufferUsages.INDEX)

        # Uniform alignment
        entity_uniform_size = 64 + 16 # mat4 + vec4
        min_alignment = device.limits().min_uniform_buffer_offset_alignment
        self.uniform_alignment = (entity_uniform_size + min_alignment - 1) & ~(min_alignment - 1)
        
        num_entities = 5
        self.entity_uniform_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Entity Uniforms",
            size=num_entities * self.uniform_alignment,
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        ))

        # Entities
        # Plane
        self.entities.append({
            "world": mat4_identity(), "color": [1, 1, 1, 1], "vb": self.plane_v_buf, "ib": self.plane_i_buf, "ic": plane_ic, "offset": 0
        })
        # Cubes
        cube_descs = [
            ([-2.0, -2.0, 2.0], 10.0, 0.7),
            ([ 2.0, -2.0, 2.0], 50.0, 1.3),
            ([-2.0,  2.0, 2.0], 140.0, 1.1),
            ([ 2.0,  2.0, 2.0], 210.0, 0.9),
        ]
        for i, (pos, angle, scale) in enumerate(cube_descs):
            # Simplified world matrix (just translation/scale for now)
            world = mat4_identity()
            world[0] = scale; world[5] = scale; world[10] = scale
            world[12] = pos[0]; world[13] = pos[1]; world[14] = pos[2]
            
            self.entities.append({
                "world": world, "color": [0, 1, 0, 1], "vb": self.cube_v_buf, "ib": self.cube_i_buf, "ic": cube_ic, "offset": (i+1) * self.uniform_alignment
            })

        # Texture/Sampler for shadows
        self.shadow_sampler = device.create_sampler(pywgpu.SamplerDescriptor(
            label="Shadow Sampler",
            address_mode_u=pywgpu.AddressMode.clamp_to_edge,
            address_mode_v=pywgpu.AddressMode.clamp_to_edge,
            address_mode_w=pywgpu.AddressMode.clamp_to_edge,
            mag_filter=pywgpu.FilterMode.linear,
            min_filter=pywgpu.FilterMode.linear,
            compare=pywgpu.CompareFunction.less_equal
        ))
        
        self.shadow_texture = device.create_texture(pywgpu.TextureDescriptor(
            label="Shadow Texture",
            size=SHADOW_SIZE,
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=SHADOW_FORMAT,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.TEXTURE_BINDING]
        ))
        self.shadow_view = self.shadow_texture.create_view()
        
        # Lights
        self.lights = [
            {"pos": [7.0, -5.0, 10.0], "color": [0.5, 1.0, 0.5, 1.0], "fov": 60, "near": 1, "far": 20},
            {"pos": [-5.0, 7.0, 10.0], "color": [1.0, 0.5, 0.5, 1.0], "fov": 45, "near": 1, "far": 20},
        ]
        
        self.light_views = [
            self.shadow_texture.create_view(
                label=f"Light View {i}",
                format=SHADOW_FORMAT,
                dimension=pywgpu.TextureViewDimension.d2,
                base_array_layer=i,
                array_layer_count=1
            ) for i in range(len(self.lights))
        ]
        
        # Light Storage (or Uniform fallback)
        light_data_size = MAX_LIGHTS * (64 + 16 + 16) # proj + pos + color
        self.light_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Light Buffer",
            size=light_data_size,
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST, pywgpu.BufferUsages.COPY_SRC]
        ))

        # Shaders
        shader_code = """
            struct Globals {
                view_proj: mat4x4<f32>,
                num_lights: vec4<u32>,
            };
            @group(0) @binding(0) var<uniform> u_globals: Globals;

            struct Entity {
                world: mat4x4<f32>,
                color: vec4<f32>,
            };
            @group(1) @binding(0) var<uniform> u_entity: Entity;

            @vertex
            fn vs_bake(@location(0) position: vec4<i32>) -> @builtin(position) vec4<f32> {
                return u_globals.view_proj * u_entity.world * vec4<f32>(position);
            }

            struct VertexOutput {
                @builtin(position) proj_position: vec4<f32>,
                @location(0) world_normal: vec3<f32>,
                @location(1) world_position: vec4<f32>
            };

            @vertex
            fn vs_main(@location(0) position: vec4<i32>, @location(1) normal: vec4<i32>) -> VertexOutput {
                let w = u_entity.world;
                let world_pos = u_entity.world * vec4<f32>(position);
                var result: VertexOutput;
                result.world_normal = mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz);
                result.world_position = world_pos;
                result.proj_position = u_globals.view_proj * world_pos;
                return result;
            }

            struct Light {
                proj: mat4x4<f32>,
                pos: vec4<f32>,
                color: vec4<f32>,
            };

            @group(0) @binding(1) var<storage, read> s_lights: array<Light>;
            @group(0) @binding(2) var t_shadow: texture_depth_2d_array;
            @group(0) @binding(3) var sampler_shadow: sampler_comparison;

            fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
                if (homogeneous_coords.w <= 0.0) { return 1.0; }
                let flip_correction = vec2<f32>(0.5, -0.5);
                let proj_correction = 1.0 / homogeneous_coords.w;
                let light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
                return textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
            }

            @fragment
            fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
                let normal = normalize(vertex.world_normal);
                var color: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
                for(var i = 0u; i < u_globals.num_lights.x; i += 1u) {
                    let light = s_lights[i];
                    let shadow = fetch_shadow(i, light.proj * vertex.world_position);
                    let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
                    let diffuse = max(0.0, dot(normal, light_dir));
                    color += shadow * diffuse * light.color.xyz;
                }
                return vec4<f32>(color, 1.0) * u_entity.color;
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))
        
        # Pipelines
        entity_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
            entries=[pywgpu.BindGroupLayoutEntry(
                binding=0, visibility=[pywgpu.ShaderStages.VERTEX, pywgpu.ShaderStages.FRAGMENT],
                ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.uniform(), has_dynamic_offset=True, min_binding_size=entity_uniform_size)
            )]
        ))
        self.entity_bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=entity_bgl,
            entries=[pywgpu.BindGroupEntry(binding=0, resource=pywgpu.BufferBinding(buffer=self.entity_uniform_buf, offset=0, size=entity_uniform_size))]
        ))

        # Shadow Pass
        shadow_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
            entries=[pywgpu.BindGroupLayoutEntry(
                binding=0, visibility=pywgpu.ShaderStages.VERTEX,
                ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.uniform(), min_binding_size=80)
            )]
        ))
        self.shadow_uniform_buf = device.create_buffer(pywgpu.BufferDescriptor(label="Shadow UBO", size=80, usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        self.shadow_bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=shadow_bgl, entries=[pywgpu.BindGroupEntry(binding=0, resource=self.shadow_uniform_buf.as_entire_binding())]
        ))
        
        shadow_pl = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(bind_group_layouts=[shadow_bgl, entity_bgl]))
        self.shadow_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=shadow_pl,
            vertex=pywgpu.VertexState(
                module=self.shader, entry_point="vs_bake",
                buffers=[pywgpu.VertexBufferLayout(array_stride=8, step_mode=pywgpu.VertexStepMode.vertex,
                                                    attributes=[pywgpu.VertexAttribute(format=pywgpu.VertexFormat.sint8x4, offset=0, shader_location=0)])]
            ),
            fragment=None,
            primitive=pywgpu.PrimitiveState(cull_mode=pywgpu.CullMode.back),
            depth_stencil=pywgpu.DepthStencilState(format=SHADOW_FORMAT, depth_write_enabled=True, depth_compare=pywgpu.CompareFunction.less_equal)
        ))

        # Forward Pass
        forward_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
            entries=[
                pywgpu.BindGroupLayoutEntry(binding=0, visibility=[pywgpu.ShaderStages.VERTEX, pywgpu.ShaderStages.FRAGMENT], ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.uniform(), min_binding_size=80)),
                pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.FRAGMENT, ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.storage(read_only=True), min_binding_size=light_data_size)),
                pywgpu.BindGroupLayoutEntry(binding=2, visibility=pywgpu.ShaderStages.FRAGMENT, ty=pywgpu.BindingType.texture(sample_type=pywgpu.TextureSampleType.depth(), view_dimension=pywgpu.TextureViewDimension.d2_array)),
                pywgpu.BindGroupLayoutEntry(binding=3, visibility=pywgpu.ShaderStages.FRAGMENT, ty=pywgpu.BindingType.sampler(ty=pywgpu.SamplerBindingType.comparison()))
            ]
        ))
        self.forward_uniform_buf = device.create_buffer(pywgpu.BufferDescriptor(label="Forward UBO", size=80, usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        self.forward_bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=forward_bgl,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.forward_uniform_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.light_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=2, resource=self.shadow_view),
                pywgpu.BindGroupEntry(binding=3, resource=self.shadow_sampler)
            ]
        ))
        
        forward_pl = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(bind_group_layouts=[forward_bgl, entity_bgl]))
        self.forward_depth_texture = device.create_texture(pywgpu.TextureDescriptor(
            size=(config.width, config.height, 1), format=DEPTH_FORMAT, usage=pywgpu.TextureUsages.RENDER_ATTACHMENT
        ))
        self.forward_depth_view = self.forward_depth_texture.create_view()
        
        self.forward_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=forward_pl,
            vertex=pywgpu.VertexState(
                module=self.shader, entry_point="vs_main",
                buffers=[pywgpu.VertexBufferLayout(array_stride=8, step_mode=pywgpu.VertexStepMode.vertex,
                                                    attributes=[pywgpu.VertexAttribute(format=pywgpu.VertexFormat.sint8x4, offset=0, shader_location=0),
                                                                pywgpu.VertexAttribute(format=pywgpu.VertexFormat.sint8x4, offset=4, shader_location=1)])]
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(cull_mode=pywgpu.CullMode.back),
            depth_stencil=pywgpu.DepthStencilState(format=DEPTH_FORMAT, depth_write_enabled=True, depth_compare=pywgpu.CompareFunction.less)
        ))

    def render(self, view, device, queue):
        # Update uniforms
        # Lights
        light_raw = b""
        for l in self.lights:
            view_mat = mat4_lookat(l["pos"], [0, 0, 0], [0, 0, 1])
            proj_mat = mat4_perspective(l["fov"]*math.pi/180, 1.0, l["near"], l["far"])
            vp = mat4_mul(view_mat, proj_mat)
            light_raw += struct.pack("16f", *vp)
            light_raw += struct.pack("4f", *l["pos"], 1.0)
            light_raw += struct.pack("4f", *l["color"])
        queue.write_buffer(self.light_buf, 0, light_raw)
        
        # Globals (for forward pass)
        vp_mat = mat4_mul(mat4_lookat([3, -10, 6], [0, 0, 0], [0, 0, 1]), mat4_perspective(math.pi/4, 800/600, 1, 20))
        queue.write_buffer(self.forward_uniform_buf, 0, struct.pack("16f", *vp_mat))
        queue.write_buffer(self.forward_uniform_buf, 64, struct.pack("4I", len(self.lights), 0, 0, 0))

        # Entity Uniforms
        for e in self.entities:
            data = struct.pack("16f", *e["world"]) + struct.pack("4f", *e["color"])
            queue.write_buffer(self.entity_uniform_buf, e["offset"], data)

        encoder = device.create_command_encoder()
        
        # Shadow Passes
        for i, lv in enumerate(self.light_views):
            # Copy light VP to shadow globals
            encoder.copy_buffer_to_buffer(self.light_buf, i * 96, self.shadow_uniform_buf, 0, 64)
            
            shadow_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
                depth_stencil_attachment=pywgpu.RenderPassDepthStencilAttachment(
                    view=lv, depth_ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(1.0), store=pywgpu.StoreOp.store)
                )
            ))
            shadow_pass.set_pipeline(self.shadow_pipeline)
            shadow_pass.set_bind_group(0, self.shadow_bind_group)
            for e in self.entities:
                shadow_pass.set_bind_group(1, self.entity_bind_group, [e["offset"]])
                shadow_pass.set_vertex_buffer(0, e["vb"])
                shadow_pass.set_index_buffer(e["ib"], pywgpu.IndexFormat.uint16)
                shadow_pass.draw_indexed(indices=range(e["ic"]), instances=range(1))
            shadow_pass.end()

        # Forward Pass
        forward_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view, ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1.0)), store=pywgpu.StoreOp.store)
            )],
            depth_stencil_attachment=pywgpu.RenderPassDepthStencilAttachment(
                view=self.forward_depth_view, depth_ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(1.0), store=pywgpu.StoreOp.store)
            )
        ))
        forward_pass.set_pipeline(self.forward_pipeline)
        forward_pass.set_bind_group(0, self.forward_bind_group)
        for e in self.entities:
            forward_pass.set_bind_group(1, self.entity_bind_group, [e["offset"]])
            forward_pass.set_vertex_buffer(0, e["vb"])
            forward_pass.set_index_buffer(e["ib"], pywgpu.IndexFormat.uint16)
            forward_pass.draw_indexed(indices=range(e["ic"]), instances=range(1))
        forward_pass.end()

        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(ShadowExample))
