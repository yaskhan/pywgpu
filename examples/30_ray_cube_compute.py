import asyncio
import struct
import time
import math
import numpy as np
from typing import List, Optional, Any

import pywgpu
from framework import Example, run_example

# --- Matrix Helpers (Minimal replacement for glam) ---
class Mat4:
    def __init__(self, data: np.ndarray):
        self.data = data # 4x4

    @staticmethod
    def identity():
        return Mat4(np.identity(4, dtype=np.float32))

    @staticmethod
    def from_translation(v: np.ndarray):
        m = np.identity(4, dtype=np.float32)
        m[0:3, 3] = v
        return Mat4(m)

    @staticmethod
    def from_rotation_y(angle: float):
        c, s = math.cos(angle), math.sin(angle)
        m = np.identity(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return Mat4(m)

    @staticmethod
    def from_scale(v: np.ndarray):
        m = np.identity(4, dtype=np.float32)
        m[0, 0] = v[0]
        m[1, 1] = v[1]
        m[2, 2] = v[2]
        return Mat4(m)

    def transpose(self):
        return Mat4(self.data.T)

    def inverse(self):
        return Mat4(np.linalg.inv(self.data))

    def to_cols_array(self) -> List[float]:
        return self.data.T.flatten().tolist()

    @staticmethod
    def look_at_rh(eye: np.ndarray, center: np.ndarray, up: np.ndarray):
        f = (center - eye)
        f /= np.linalg.norm(f)
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        
        m = np.identity(4, dtype=np.float32)
        m[0, 0:3] = s
        m[1, 0:3] = u
        m[2, 0:3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return Mat4(m)

    @staticmethod
    def perspective_rh(fovy: float, aspect: float, near: float, far: float):
        f = 1.0 / math.tan(fovy / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = far / (near - far)
        m[2, 3] = (far * near) / (near - far)
        m[3, 2] = -1.0
        return Mat4(m)

class RayCubeComputeExample(Example):
    TITLE = "Ray Cube Compute (Software/Hardware RT Mix)"

    def required_features(self) -> List[pywgpu.Features]:
        return [pywgpu.Features.EXPERIMENTAL_RAY_QUERY]

    async def init(self, config, adapter, device, queue):
        # 1. Shaders
        compute_shader_code = """
            enable wgpu_ray_query;

            struct Uniforms {
                view_inv: mat4x4<f32>,
                proj_inv: mat4x4<f32>,
            };
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;
            @group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(2) var acc_struct: acceleration_structure;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let target_size = textureDimensions(output);
                if (global_id.x >= target_size.x || global_id.y >= target_size.y) { return; }
                
                let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5);
                let in_uv = pixel_center / vec2<f32>(target_size.xy);
                let d = in_uv * 2.0 - 1.0;

                let origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
                let temp = uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0);
                let direction = normalize((uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz);

                var rq: ray_query;
                rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, direction));
                rayQueryProceed(&rq);

                let intersection = rayQueryGetCommittedIntersection(&rq);
                var color = vec4<f32>(in_uv, 0.0, 1.0); // Background gradient
                if intersection.kind != 0u {
                    color = vec4<f32>(intersection.barycentrics, 1.0 - intersection.barycentrics.x - intersection.barycentrics.y, 1.0);
                }

                textureStore(output, global_id.xy, color);
            }
        """
        
        blit_shader_code = """
            struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
            @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> Out {
                let x = f32(i32(vi) / 2) * 2.0;
                let y = f32(i32(vi) & 1) * 2.0;
                return Out(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
            }
            @group(0) @binding(0) var t: texture_2d<f32>;
            @group(0) @binding(1) var s: sampler;
            @fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
                return textureSample(t, s, in.uv);
            }
        """

        self.cs_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=compute_shader_code))
        self.blit_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=blit_shader_code))

        # 2. Geometry (Cube)
        def vertex(p, t):
            return p + [1.0] + t
            
        vertex_data = []
        # top
        vertex_data += vertex([-1, -1, 1], [0, 0])
        vertex_data += vertex([1, -1, 1], [1, 0])
        vertex_data += vertex([1, 1, 1], [1, 1])
        vertex_data += vertex([-1, 1, 1], [0, 1])
        # bottom
        vertex_data += vertex([-1, 1, -1], [1, 0])
        vertex_data += vertex([1, 1, -1], [0, 0])
        vertex_data += vertex([1, -1, -1], [0, 1])
        vertex_data += vertex([-1, -1, -1], [1, 1])
        # right
        vertex_data += vertex([1, -1, -1], [0, 0])
        vertex_data += vertex([1, 1, -1], [1, 0])
        vertex_data += vertex([1, 1, 1], [1, 1])
        vertex_data += vertex([1, -1, 1], [0, 1])
        # left
        vertex_data += vertex([-1, -1, 1], [1, 0])
        vertex_data += vertex([-1, 1, 1], [0, 0])
        vertex_data += vertex([-1, 1, -1], [0, 1])
        vertex_data += vertex([-1, -1, -1], [1, 1])
        # front
        vertex_data += vertex([1, 1, -1], [1, 0])
        vertex_data += vertex([-1, 1, -1], [0, 0])
        vertex_data += vertex([-1, 1, 1], [0, 1])
        vertex_data += vertex([1, 1, 1], [1, 1])
        # back
        vertex_data += vertex([1, -1, 1], [0, 0])
        vertex_data += vertex([-1, -1, 1], [1, 0])
        vertex_data += vertex([-1, -1, -1], [1, 1])
        vertex_data += vertex([1, -1, -1], [0, 1])

        index_data = [
            0, 1, 2, 2, 3, 0, # top
            4, 5, 6, 6, 7, 4, # bottom
            8, 9, 10, 10, 11, 8, # right
            12, 13, 14, 14, 15, 12, # left
            16, 17, 18, 18, 19, 16, # front
            20, 21, 22, 22, 23, 20, # back
        ]

        self.vertices = np.array(vertex_data, dtype=np.float32)
        self.indices = np.array(index_data, dtype=np.uint16)
        
        self.v_buf = device.create_buffer(pywgpu.BufferDescriptor(
            size=self.vertices.nbytes,
            usage=[pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]
        ))
        queue.write_buffer(self.v_buf, 0, self.vertices.tobytes())
        
        self.i_buf = device.create_buffer(pywgpu.BufferDescriptor(
            size=self.indices.nbytes,
            usage=[pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]
        ))
        queue.write_buffer(self.i_buf, 0, self.indices.tobytes())

        # 3. BLAS & TLAS
        self.blas_size = pywgpu.BlasTriangleGeometrySizeDescriptor(
            vertex_format=pywgpu.VertexFormat.float32x3,
            vertex_count=24,
            index_format=pywgpu.IndexFormat.uint16,
            index_count=36,
            flags=pywgpu.AccelerationStructureGeometryFlags.OPAQUE
        )
        
        self.blas = device.create_blas(pywgpu.BlasDescriptor(), [self.blas_size])
        
        self.side_count = 6
        self.tlas = device.create_tlas(pywgpu.TlasDescriptor(
            max_instances=self.side_count * self.side_count,
            flags=pywgpu.AccelerationStructureFlags.PREFER_FAST_TRACE
        ))

        dist = 3.0
        for x in range(self.side_count):
            for y in range(self.side_count):
                idx = x + y * self.side_count
                tx = (x - self.side_count/2 + 0.5) * dist
                ty = (y - self.side_count/2 + 0.5) * dist
                self.tlas[idx] = pywgpu.TlasInstance(
                    blas=self.blas,
                    transform=Mat4.from_translation(np.array([tx, ty, -15.0], dtype=np.float32)).to_cols_array()[:12],
                    mask=0xFF
                )

        # Build AS
        encoder = device.create_command_encoder()
        encoder.build_acceleration_structures(
            blas=[pywgpu.BlasBuildEntry(
                blas=self.blas,
                geometry=[pywgpu.BlasTriangleGeometry(
                    size=self.blas_size,
                    vertex_buffer=self.v_buf,
                    vertex_stride=24, # pos (4f) + tex (2f) = 6 * 4 = 24
                    index_buffer=self.i_buf,
                    first_index=0
                )]
            )],
            tlas=self.tlas
        )
        queue.submit([encoder.finish()])

        # 4. Bindings
        self.storage_tex = device.create_texture(pywgpu.TextureDescriptor(
            size=(config.width, config.height, 1),
            format=pywgpu.TextureFormat.rgba8unorm,
            usage=[pywgpu.TextureUsages.STORAGE_BINDING, pywgpu.TextureUsages.texture_binding]
        ))
        self.storage_view = self.storage_tex.create_view()
        
        view_mat = Mat4.look_at_rh(np.array([0.0, 0.0, 10.0], dtype=np.float32), np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        proj_mat = Mat4.perspective_rh(math.radians(59.0), config.width / config.height, 0.001, 1000.0)
        
        uniform_data = struct.pack("16f16f", *view_mat.inverse().to_cols_array(), *proj_mat.inverse().to_cols_array())
        self.u_buf = device.create_buffer(pywgpu.BufferDescriptor(size=len(uniform_data), usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.u_buf, 0, uniform_data)

        self.cs_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BindingType.storage_texture(format=pywgpu.TextureFormat.rgba8unorm, access=pywgpu.StorageTextureAccess.WRITE_ONLY)),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.uniform()),
            pywgpu.BindGroupLayoutEntry(binding=2, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BindingType.acceleration_structure())
        ]))
        self.cs_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.cs_bgl,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.storage_view),
                pywgpu.BindGroupEntry(binding=1, resource=self.u_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=2, resource=self.tlas.as_entire_binding())
            ]
        ))
        self.cs_pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.cs_bgl]),
            module=self.cs_module,
            entry_point="main"
        ))

        # Blit
        self.sampler = device.create_sampler()
        self.blit_bgl = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.texture()),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.sampler())
        ]))
        self.blit_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.blit_bgl,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.storage_view),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))
        self.blit_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.blit_bgl]),
            vertex=pywgpu.VertexState(module=self.blit_module, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.blit_module, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState()
        ))

        self.start_time = time.time()

    def render(self, view, device, queue):
        # Update TLAS (bounce cubes?)
        t = time.time() - self.start_time
        dist = 3.0
        for x in range(self.side_count):
            for y in range(self.side_count):
                idx = x + y * self.side_count
                tx = (x - self.side_count/2 + 0.5) * dist
                ty = (y - self.side_count/2 + 0.5) * dist
                tz = -15.0 + math.sin(t + x + y) * 2.0
                self.tlas[idx].transform = Mat4.from_translation(np.array([tx, ty, tz], dtype=np.float32)).to_cols_array()[:12]
        
        encoder = device.create_command_encoder()
        encoder.build_acceleration_structures(tlas=self.tlas)
        
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.cs_pipeline)
        cpass.set_bind_group(0, self.cs_bg)
        cpass.dispatch_workgroups(self.storage_tex.width // 8, self.storage_tex.height // 8, 1)
        cpass.end()
        
        rpass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color.black), store=pywgpu.StoreOp.store)
            )]
        ))
        rpass.set_pipeline(self.blit_pipeline)
        rpass.set_bind_group(0, self.blit_bg)
        rpass.draw(vertices=range(3), instances=range(1))
        rpass.end()
        
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(RayCubeComputeExample))
