import asyncio
import struct
import time
import math
import numpy as np
from typing import List, Optional, Any

import pywgpu
from framework import Example, run_example

# --- Matrix Helpers ---
class Mat4:
    def __init__(self, data: np.ndarray):
        self.data = data # 4x4
    @staticmethod
    def identity(): return Mat4(np.identity(4, dtype=np.float32))
    def inverse(self): return Mat4(np.linalg.inv(self.data))
    def transpose(self): return Mat4(self.data.T)
    def to_cols_array(self) -> List[float]: return self.data.T.flatten().tolist()
    @staticmethod
    def look_at_rh(eye: np.ndarray, center: np.ndarray, up: np.ndarray):
        f = (center - eye); f /= np.linalg.norm(f)
        s = np.cross(f, up); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.identity(4, dtype=np.float32); m[0, 0:3] = s; m[1, 0:3] = u; m[2, 0:3] = -f
        m[0, 3] = -np.dot(s, eye); m[1, 3] = -np.dot(u, eye); m[2, 3] = np.dot(f, eye); return Mat4(m)
    @staticmethod
    def perspective_rh(fovy: float, aspect: float, near: float, far: float):
        f = 1.0 / math.tan(fovy / 2.0); m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect; m[1, 1] = f; m[2, 2] = far / (near - far); m[2, 3] = (far * near) / (near - far); m[3, 2] = -1.0; return Mat4(m)

class RayShadowsExample(Example):
    TITLE = "Ray Traced Shadows"

    def required_features(self) -> List[pywgpu.Features]:
        return [pywgpu.Features.EXPERIMENTAL_RAY_QUERY]

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            enable wgpu_ray_query;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) world_pos: vec3<f32>,
            };

            struct Uniforms { view_inv: mat4x4<f32>, proj_inv: mat4x4<f32>, mvp: mat4x4<f32> };
            struct LightData { light_pos: vec3<f32>, _pad: f32 };

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var acc_struct: acceleration_structure;
            @group(0) @binding(2) var<uniform> light: LightData;

            @vertex fn vs_main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @builtin(vertex_index) vi: u32) -> VertexOutput {
                var out: VertexOutput;
                out.position = uniforms.mvp * vec4<f32>(pos, 1.0);
                out.world_pos = pos;
                out.normal = normal;
                // mock uv
                out.uv = vec2<f32>(f32(vi)/8.0, 0.5);
                return out;
            }

            @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let camera_pos = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
                let light_vec = light.light_pos - in.world_pos;
                let light_dir = normalize(light_vec);
                let light_dist = length(light_vec);

                var normal = normalize(in.normal);
                let view_dir = normalize(camera_pos - in.world_pos);
                if (dot(view_dir, normal) < 0.0) { normal = -normal; }

                // Trace shadow ray
                var rq: ray_query;
                rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, light_dist, in.world_pos, light_dir));
                rayQueryProceed(&rq);

                let intersection = rayQueryGetCommittedIntersection(&rq);
                let shadow = f32(intersection.kind == 0u); // 1.0 if no hit (in light)
                
                let brightness = 0.5;
                let diff = max(dot(light_dir, normal), 0.1) * brightness;
                let color = vec3<f32>(diff * (0.2 + 0.8 * shadow));
                
                return vec4<f32>(color, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Geometry
        # Plane + Vertical quad
        vertices = np.array([
            # Floor
            -5.0, 0.0, -5.0,  0.0, 1.0, 0.0,
            -5.0, 0.0,  5.0,  0.0, 1.0, 0.0,
             5.0, 0.0, -5.0,  0.0, 1.0, 0.0,
             5.0, 0.0,  5.0,  0.0, 1.0, 0.0,
            # Wall (caster)
            -1.0, 0.0,  1.0,  0.0, 0.0, 1.0,
            -1.0, 2.0,  1.0,  0.0, 0.0, 1.0,
             1.0, 0.0,  1.0,  0.0, 0.0, 1.0,
             1.0, 2.0,  1.0,  0.0, 0.0, 1.0
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 2, 2, 1, 3, # floor
            4, 5, 6, 6, 5, 7  # wall
        ], dtype=np.uint16)

        self.v_buf = device.create_buffer(pywgpu.BufferDescriptor(size=vertices.nbytes, usage=[pywgpu.BufferUsages.VERTEX, pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.v_buf, 0, vertices.tobytes())
        self.i_buf = device.create_buffer(pywgpu.BufferDescriptor(size=indices.nbytes, usage=[pywgpu.BufferUsages.INDEX, pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.i_buf, 0, indices.tobytes())

        # 3. AS
        blas_size = pywgpu.BlasTriangleGeometrySizeDescriptor(vertex_format=pywgpu.VertexFormat.float32x3, vertex_count=8, index_format=pywgpu.IndexFormat.uint16, index_count=12)
        self.blas = device.create_blas(pywgpu.BlasDescriptor(), [blas_size])
        self.tlas = device.create_tlas(pywgpu.TlasDescriptor(max_instances=1))
        self.tlas[0] = pywgpu.TlasInstance(blas=self.blas, transform=Mat4.identity().to_cols_array()[:12])

        encoder = device.create_command_encoder()
        encoder.build_acceleration_structures(
            blas=[pywgpu.BlasBuildEntry(blas=self.blas, geometry=[pywgpu.BlasTriangleGeometry(size=blas_size, vertex_buffer=self.v_buf, vertex_stride=24, index_buffer=self.i_buf, first_index=0)])],
            tlas=self.tlas
        )
        queue.submit([encoder.finish()])

        # 4. Bindings
        self.u_buf = device.create_buffer(pywgpu.BufferDescriptor(size=192, usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        self.light_buf = device.create_buffer(pywgpu.BufferDescriptor(size=16, usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        
        view = Mat4.look_at_rh(np.array([0.0, 3.0, 8.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        proj = Mat4.perspective_rh(math.radians(59.0), config.width / config.height, 0.1, 100.0)
        mvp = Mat4(proj.data @ view.data)
        
        queue.write_buffer(self.u_buf, 0, struct.pack("16f16f16f", *view.inverse().to_cols_array(), *proj.inverse().to_cols_array(), *mvp.to_cols_array()))

        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=None,
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main", buffers=[
                pywgpu.VertexBufferLayout(array_stride=24, attributes=[
                    pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x3, offset=0, shader_location=0),
                    pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x3, offset=12, shader_location=1)
                ])
            ]),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState()
        ))
        
        self.bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.u_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.tlas.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=2, resource=self.light_buf.as_entire_binding())
            ]
        ))
        
        self.start_time = time.time()

    def render(self, view, device, queue):
        t = time.time() - self.start_time
        light_pos = [math.cos(t)*5.0, 4.0, math.sin(t)*5.0]
        queue.write_buffer(self.light_buf, 0, struct.pack("3f4x", *light_pos))

        encoder = device.create_command_encoder()
        rpass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(view=view, ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(0.1, 0.1, 0.1, 1.0)), store=pywgpu.StoreOp.store))]
        ))
        rpass.set_pipeline(self.pipeline)
        rpass.set_bind_group(0, self.bg)
        rpass.set_vertex_buffer(0, self.v_buf)
        rpass.set_index_buffer(self.i_buf, pywgpu.IndexFormat.uint16)
        rpass.draw_indexed(range(12), 0, range(1))
        rpass.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(RayShadowsExample))
