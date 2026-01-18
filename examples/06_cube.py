import asyncio
import struct
import math
from typing import List, Tuple

import pywgpu
from framework import Example, run_example

class Vertex:
    def __init__(self, pos: List[float], tex_coord: List[float]):
        self.pos = pos + [1.0]  # x, y, z, w
        self.tex_coord = tex_coord

def create_vertex(pos: List[int], tc: List[int]) -> Vertex:
    return Vertex([float(p) for p in pos], [float(t) for t in tc])

def create_vertices() -> Tuple[bytes, bytes, int]:
    vertex_data = [
        # top (0, 0, 1)
        create_vertex([-1, -1, 1], [0, 0]),
        create_vertex([1, -1, 1], [1, 0]),
        create_vertex([1, 1, 1], [1, 1]),
        create_vertex([-1, 1, 1], [0, 1]),
        # bottom (0, 0, -1)
        create_vertex([-1, 1, -1], [1, 0]),
        create_vertex([1, 1, -1], [0, 0]),
        create_vertex([1, -1, -1], [0, 1]),
        create_vertex([-1, -1, -1], [1, 1]),
        # right (1, 0, 0)
        create_vertex([1, -1, -1], [0, 0]),
        create_vertex([1, 1, -1], [1, 0]),
        create_vertex([1, 1, 1], [1, 1]),
        create_vertex([1, -1, 1], [0, 1]),
        # left (-1, 0, 0)
        create_vertex([-1, -1, 1], [1, 0]),
        create_vertex([-1, 1, 1], [0, 0]),
        create_vertex([-1, 1, -1], [0, 1]),
        create_vertex([-1, -1, -1], [1, 1]),
        # front (0, 1, 0)
        create_vertex([1, 1, -1], [1, 0]),
        create_vertex([-1, 1, -1], [0, 0]),
        create_vertex([-1, 1, 1], [0, 1]),
        create_vertex([1, 1, 1], [1, 1]),
        # back (0, -1, 0)
        create_vertex([1, -1, 1], [0, 0]),
        create_vertex([-1, -1, 1], [1, 0]),
        create_vertex([-1, -1, -1], [1, 1]),
        create_vertex([1, -1, -1], [0, 1]),
    ]

    index_data = [
        0, 1, 2, 2, 3, 0, # top
        4, 5, 6, 6, 7, 4, # bottom
        8, 9, 10, 10, 11, 8, # right
        12, 13, 14, 14, 15, 12, # left
        16, 17, 18, 18, 19, 16, # front
        20, 21, 22, 22, 23, 20, # back
    ]

    packed_vertices = b""
    for v in vertex_data:
        packed_vertices += struct.pack("4f2f", *v.pos, *v.tex_coord)
    
    packed_indices = struct.pack(f"{len(index_data)}H", *index_data)
    
    return packed_vertices, packed_indices, len(index_data)

def create_texels(size: int) -> bytes:
    texels = bytearray()
    for y in range(size):
        for x in range(size):
            cx = 3.0 * x / (size - 1) - 2.0
            cy = 2.0 * y / (size - 1) - 1.0
            zx, zy = cx, cy
            count = 0
            while count < 255 and zx * zx + zy * zy < 4.0:
                old_zx = zx
                zx = zx * zx - zy * zy + cx
                zy = 2.0 * old_zx * zy + cy
                count += 1
            texels.append(count)
    return bytes(texels)

def generate_matrix(aspect_ratio: float) -> List[float]:
    # Simplified perspective * view matrix
    fov = math.pi / 4.0
    near = 1.0
    far = 10.0
    f = 1.0 / math.tan(fov / 2.0)
    
    # Projection matrix (RH)
    projection = [
        f / aspect_ratio, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, far / (near - far), -1.0,
        0.0, 0.0, (near * far) / (near - far), 0.0,
    ]
    
    # View matrix (LookAt RH)
    eye = [1.5, -5.0, 3.0]
    target = [0.0, 0.0, 0.0]
    up = [0.0, 0.0, 1.0]
    
    def normalize(v):
        l = math.sqrt(sum(x*x for x in v))
        return [x/l for x in v]
    
    def cross(a, b):
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]
    
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    z_axis = normalize([eye[i] - target[i] for i in range(3)])
    x_axis = normalize(cross(up, z_axis))
    y_axis = cross(z_axis, x_axis)
    
    view = [
        x_axis[0], y_axis[0], z_axis[0], 0.0,
        x_axis[1], y_axis[1], z_axis[1], 0.0,
        x_axis[2], y_axis[2], z_axis[2], 0.0,
        -dot(x_axis, eye), -dot(y_axis, eye), -dot(z_axis, eye), 1.0,
    ]
    
    # Multiply projection * view
    res = [0.0] * 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                res[i*4 + j] += view[i*4 + k] * projection[k*4 + j]
    return res

class CubeExample(Example):
    TITLE = "Cube Example"

    async def init(self, config, adapter, device, queue):
        vertex_data, index_data, self.index_count = create_vertices()
        
        self.vertex_buf = device.create_buffer_init(
            label="Vertex Buffer",
            contents=vertex_data,
            usage=pywgpu.BufferUsages.VERTEX
        )
        
        self.index_buf = device.create_buffer_init(
            label="Index Buffer",
            contents=index_data,
            usage=pywgpu.BufferUsages.INDEX
        )
        
        bind_group_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
            entries=[
                pywgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=pywgpu.ShaderStages.VERTEX,
                    ty=pywgpu.BindingType.buffer(
                        ty=pywgpu.BufferBindingType.uniform(),
                        has_dynamic_offset=False,
                        min_binding_size=64
                    )
                ),
                pywgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=pywgpu.ShaderStages.FRAGMENT,
                    ty=pywgpu.BindingType.texture(
                        sample_type=pywgpu.TextureSampleType.uint(),
                        view_dimension=pywgpu.TextureViewDimension.d2,
                        multisampled=False
                    )
                )
            ]
        ))
        
        self.pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(
            bind_group_layouts=[bind_group_layout]
        ))
        
        size = 256
        texels = create_texels(size)
        texture_extent = pywgpu.Extent3d(width=size, height=size, depth_or_array_layers=1)
        texture = device.create_texture(pywgpu.TextureDescriptor(
            size=texture_extent,
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=pywgpu.TextureFormat.r8uint,
            usage=[pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.COPY_DST]
        ))
        texture_view = texture.create_view()
        
        queue.write_texture(
            pywgpu.TexelCopyTextureInfo(texture=texture),
            texels,
            pywgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=size, rows_per_image=None),
            texture_extent
        )
        
        matrix = generate_matrix(config.width / config.height)
        self.uniform_buf = device.create_buffer_init(
            label="Uniform Buffer",
            contents=struct.pack("16f", *matrix),
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        )
        
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=bind_group_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.uniform_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=texture_view)
            ]
        ))
        
        shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coord: vec2<f32>,
            };

            @group(0) @binding(0) var<uniform> transform: mat4x4<f32>;

            @vertex
            fn vs_main(
                @location(0) position: vec4<f32>,
                @location(1) tex_coord: vec2<f32>,
            ) -> VertexOutput {
                var out: VertexOutput;
                out.position = transform * position;
                out.tex_coord = tex_coord;
                return out;
            }

            @group(0) @binding(1) var r_color: texture_2d<u32>;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let texsize = textureDimensions(r_color);
                let uv = vec2<i32>(i32(in.tex_coord.x * f32(texsize.x)), i32(in.tex_coord.y * f32(texsize.y)));
                let value = textureLoad(r_color, uv, 0).r;
                let f_value = f32(value) / 255.0;
                return vec4<f32>(f_value, f_value, f_value, 1.0);
            }
        """
        
        shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))
        
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=self.pipeline_layout,
            vertex=pywgpu.VertexState(
                module=shader,
                entry_point="vs_main",
                buffers=[
                    pywgpu.VertexBufferLayout(
                        array_stride=24,
                        step_mode=pywgpu.VertexStepMode.vertex,
                        attributes=[
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x4, offset=0, shader_location=0),
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=16, shader_location=1)
                        ]
                    )
                ]
            ),
            fragment=pywgpu.FragmentState(
                module=shader,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(cull_mode=pywgpu.CullMode.back)
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[
                pywgpu.RenderPassColorAttachment(
                    view=view,
                    ops=pywgpu.Operations(
                        load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1.0)),
                        store=pywgpu.StoreOp.store
                    )
                )
            ]
        ))
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.set_index_buffer(self.index_buf, pywgpu.IndexFormat.uint16)
        render_pass.set_vertex_buffer(0, self.vertex_buf)
        render_pass.draw_indexed(indices=range(self.index_count), instances=range(1))
        render_pass.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(CubeExample))
