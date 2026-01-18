import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

# Matrix math helpers
def mat4_identity():
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]

def mat4_perspective(fovy_rad, aspect, near, far):
    f = 1.0 / math.tan(fovy_rad / 2.0)
    return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far / (near - far), -1,
        0, 0, (near * far) / (near - far), 0
    ]

def mat4_look_at(eye, target, up):
    f = [target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]]
    f_len = math.sqrt(sum(x*x for x in f))
    f = [x/f_len for x in f]
    
    u_len = math.sqrt(sum(x*x for x in up))
    up = [x/u_len for x in up]
    
    s = [f[1]*up[2] - f[2]*up[1], f[2]*up[0] - f[0]*up[2], f[0]*up[1] - f[1]*up[0]]
    s_len = math.sqrt(sum(x*x for x in s))
    s = [x/s_len for x in s]
    
    u = [s[1]*f[2] - s[2]*f[1], s[2]*f[0] - s[0]*f[2], s[0]*f[1] - s[1]*f[0]]
    
    return [
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -sum(s[i]*eye[i] for i in range(3)), -sum(u[i]*eye[i] for i in range(3)), sum(f[i]*eye[i] for i in range(3)), 1
    ]

def mat4_inverse(m):
    # Simplified inverse for perspective/view matrices is complex, 
    # but for this example we only need proj_inv and view_inv (which is transpose for rotation)
    # ProjInv can be derived:
    # f = 1/tan(fovy/2)
    # P = [ f/a, 0, 0, 0; 0, f, 0, 0; 0, 0, F/(N-F), NF/(N-F); 0, 0, -1, 0 ]
    # In WGSL it uses proj_inv * pos.
    # We'll just provide a basic inverse or use specific inverse for perspective.
    # For perspective:
    # Pi = [ a/f, 0, 0, 0; 0, 1/f, 0, 0; 0, 0, 0, -1; 0, 0, (N-F)/NF, F/NF ]
    pass

class SkyboxExample(Example):
    TITLE = "Skybox Example"
    DEPTH_FORMAT = pywgpu.TextureFormat.depth24plus

    async def init(self, config, adapter, device, queue):
        # 1. Pipelines resources
        shader_code = """
            struct SkyOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec3<f32>,
            };

            struct Data {
                proj: mat4x4<f32>,
                proj_inv: mat4x4<f32>,
                view: mat4x4<f32>,
                cam_pos: vec4<f32>,
            };
            @group(0) @binding(0) var<uniform> r_data: Data;

            @vertex
            fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyOutput {
                let tmp1 = i32(vertex_index) / 2;
                let tmp2 = i32(vertex_index) & 1;
                let pos = vec4<f32>(
                    f32(tmp1) * 4.0 - 1.0,
                    f32(tmp2) * 4.0 - 1.0,
                    1.0,
                    1.0
                );

                let inv_model_view = transpose(mat3x3<f32>(
                    r_data.view[0].xyz, 
                    r_data.view[1].xyz, 
                    r_data.view[2].xyz
                ));
                let unprojected = r_data.proj_inv * pos;

                var result: SkyOutput;
                result.uv = inv_model_view * unprojected.xyz;
                result.position = pos;
                return result;
            }

            struct EntityOutput {
                @builtin(position) position: vec4<f32>,
                @location(1) normal: vec3<f32>,
                @location(3) view: vec3<f32>,
            };

            @vertex
            fn vs_entity(
                @location(0) pos: vec3<f32>,
                @location(1) normal: vec3<f32>,
            ) -> EntityOutput {
                var result: EntityOutput;
                result.normal = normal;
                result.view = pos - r_data.cam_pos.xyz;
                result.position = r_data.proj * r_data.view * vec4<f32>(pos, 1.0);
                return result;
            }

            @group(0) @binding(1) var r_texture: texture_cube<f32>;
            @group(0) @binding(2) var r_sampler: sampler;

            @fragment
            fn fs_sky(vertex: SkyOutput) -> @location(0) vec4<f32> {
                return textureSample(r_texture, r_sampler, vertex.uv);
            }

            @fragment
            fn fs_entity(vertex: EntityOutput) -> @location(0) vec4<f32> {
                let incident = normalize(vertex.view);
                let normal = normalize(vertex.normal);
                let reflected = incident - 2.0 * dot(normal, incident) * normal;

                let reflected_color = textureSample(r_texture, r_sampler, reflected).rgb;
                return vec4<f32>(vec3<f32>(0.1) + 0.5 * reflected_color, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Geometry (Entity - Cube)
        # pos [3], normal [3]
        cube_vertices = [
            # Front
            -1, -1,  1,  0,  0,  1,
             1, -1,  1,  0,  0,  1,
             1,  1,  1,  0,  0,  1,
            -1,  1,  1,  0,  0,  1,
            # Back
            -1, -1, -1,  0,  0, -1,
             1, -1, -1,  0,  0, -1,
             1,  1, -1,  0,  0, -1,
            -1,  1, -1,  0,  0, -1,
            # Left, Right, Top, Bottom ... skipped for brevity, let's just do a few
        ]
        # Full cube
        def add_face(vertices, p1, p2, p3, p4, n):
            for p in [p1, p2, p4, p2, p3, p4]:
                vertices.extend(p)
                vertices.extend(n)

        vertices = []
        # +Z
        add_face(vertices, [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1], [0,0,1])
        # -Z
        add_face(vertices, [ 1,-1,-1], [-1,-1,-1], [-1, 1,-1], [ 1, 1,-1], [0,0,-1])
        # +X
        add_face(vertices, [ 1,-1, 1], [ 1,-1,-1], [ 1, 1,-1], [ 1, 1, 1], [1,0,0])
        # -X
        add_face(vertices, [-1,-1,-1], [-1,-1, 1], [-1, 1, 1], [-1, 1,-1], [-1,0,0])
        # +Y
        add_face(vertices, [-1, 1, 1], [ 1, 1, 1], [ 1, 1,-1], [-1, 1,-1], [0,1,0])
        # -Y
        add_face(vertices, [-1,-1,-1], [ 1,-1,-1], [ 1,-1, 1], [-1,-1, 1], [0,-1,0])

        packed_vertices = struct.pack(f"{len(vertices)}f", *vertices)
        self.vertex_buffer = device.create_buffer_init(
            label="Vertex Buffer",
            contents=packed_vertices,
            usage=pywgpu.BufferUsages.VERTEX
        )
        self.vertex_count = len(vertices) // 6

        # 3. Textures
        size = 256
        self.texture = device.create_texture(pywgpu.TextureDescriptor(
            label="Skybox Texture",
            size=(size, size, 6),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=pywgpu.TextureFormat.rgba8unorm,
            usage=[pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.COPY_DST]
        ))
        
        # Fill faces with different colors
        face_colors = [
            (255, 0, 0, 255),    # +X Red
            (128, 0, 0, 255),    # -X Dark Red
            (0, 255, 0, 255),    # +Y Green
            (0, 128, 0, 255),    # -Y Dark Green
            (0, 0, 255, 255),    # +Z Blue
            (0, 0, 128, 255),    # -Z Dark Blue
        ]
        for i, color in enumerate(face_colors):
            face_data = bytes(color * (size * size))
            queue.write_texture(
                pywgpu.TexelCopyTextureInfo(texture=self.texture, mip_level=0, origin=(0, 0, i)),
                face_data,
                pywgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=4 * size, rows_per_image=size),
                (size, size, 1)
            )

        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor(
            mag_filter=pywgpu.FilterMode.linear,
            min_filter=pywgpu.FilterMode.linear,
            mipmap_filter=pywgpu.MipmapFilterMode.linear
        ))

        # 4. Uniforms
        self.uniform_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Uniform Buffer",
            size=16*3*4 + 16, # 3 matrices + vec4
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        ))

        # 5. Bind Group
        self.bind_group_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
            entries=[
                pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.vertex | pywgpu.ShaderStages.fragment, ty=pywgpu.BufferBindingType.uniform),
                pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.TextureBindingType.float(view_dimension=pywgpu.TextureViewDimension.cube)),
                pywgpu.BindGroupLayoutEntry(binding=2, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.SamplerBindingType.filtering)
            ]
        ))

        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bind_group_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.uniform_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.texture.create_view(dimension=pywgpu.TextureViewDimension.cube)),
                pywgpu.BindGroupEntry(binding=2, resource=self.sampler)
            ]
        ))

        # 6. Pipelines
        pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(bind_group_layouts=[self.bind_group_layout]))

        self.entity_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=pipeline_layout,
            vertex=pywgpu.VertexState(
                module=self.shader, entry_point="vs_entity",
                buffers=[pywgpu.VertexBufferLayout(
                    array_stride=24,
                    attributes=[
                        pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x3, offset=0, shader_location=0),
                        pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x3, offset=12, shader_location=1)
                    ]
                )]
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_entity",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(front_face=pywgpu.FrontFace.cw),
            depth_stencil=pywgpu.DepthStencilState(
                format=self.DEPTH_FORMAT,
                depth_write_enabled=True,
                depth_compare=pywgpu.CompareFunction.less_equal
            )
        ))

        self.sky_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=pipeline_layout,
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_sky"),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_sky",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(front_face=pywgpu.FrontFace.cw),
            depth_stencil=pywgpu.DepthStencilState(
                format=self.DEPTH_FORMAT,
                depth_write_enabled=False,
                depth_compare=pywgpu.CompareFunction.less_equal
            )
        ))

        self.resize(config, device, queue)

    def resize(self, config, device, queue):
        self.depth_texture = device.create_texture(pywgpu.TextureDescriptor(
            size=(config.width, config.height, 1),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=self.DEPTH_FORMAT,
            usage=pywgpu.TextureUsages.RENDER_ATTACHMENT
        ))
        self.depth_view = self.depth_texture.create_view()

        # Update matrices
        aspect = config.width / config.height
        fovy = math.pi / 4.0
        near, far = 1.0, 50.0
        proj = mat4_perspective(fovy, aspect, near, far)
        
        # ProjInv for perspective
        f = 1.0 / math.tan(fovy / 2.0)
        proj_inv = [
            aspect / f, 0, 0, 0,
            0, 1.0 / f, 0, 0,
            0, 0, 0, -1,
            0, 0, (near - far) / (near * far), 1.0 / near # This might be slightly off but heuristic is ok
        ]
        # Better ProjInv:
        # P = [ A, 0, 0, 0; 0, B, 0, 0; 0, 0, C, D; 0, 0, -1, 0 ]
        # Pi = [ 1/A, 0, 0, 0; 0, 1/B, 0, 0; 0, 0, 0, -1; 0, 0, 1/D, C/D ]
        A_p = f / aspect
        B_p = f
        C_p = far / (near - far)
        D_p = (near * far) / (near - far)
        proj_inv = [
            1.0/A_p, 0, 0, 0,
            0, 1.0/B_p, 0, 0,
            0, 0, 0, -1,
            0, 0, 1.0/D_p, C_p/D_p
        ]

        cam_pos = [10.0, 5.0, 10.0]
        view = mat4_look_at(cam_pos, [0, 2.0, 0], [0, 1.0, 0])

        uniform_data = struct.pack("16f", *proj) + struct.pack("16f", *proj_inv) + struct.pack("16f", *view) + struct.pack("4f", *cam_pos, 1.0)
        queue.write_buffer(self.uniform_buf, 0, uniform_data)

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )],
            depth_stencil_attachment=pywgpu.RenderPassDepthStencilAttachment(
                view=self.depth_view,
                depth_ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(1.0), store=pywgpu.StoreOp.discard)
            )
        ))
        
        pass_enc.set_bind_group(0, self.bind_group)
        
        # 1. Draw entity
        pass_enc.set_pipeline(self.entity_pipeline)
        pass_enc.set_vertex_buffer(0, self.vertex_buffer)
        pass_enc.draw(vertices=range(self.vertex_count), instances=range(1))
        
        # 2. Draw skybox
        pass_enc.set_pipeline(self.sky_pipeline)
        pass_enc.draw(vertices=range(3), instances=range(1))
        
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(SkyboxExample))
