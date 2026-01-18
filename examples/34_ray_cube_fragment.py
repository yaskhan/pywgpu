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
    @staticmethod
    def from_rotation_translation(rotation: np.ndarray, translation: np.ndarray):
        m = np.identity(4, dtype=np.float32)
        m[0:3, 0:3] = rotation
        m[0:3, 3] = translation
        return Mat4(m)

def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)

class RayCubeFragmentExample(Example):
    TITLE = "Ray Cube Fragment (Real-time Ray Queries)"

    def required_features(self) -> List[pywgpu.Features]:
        return [pywgpu.Features.EXPERIMENTAL_RAY_QUERY]

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            enable wgpu_ray_query;

            struct VertexOutput { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
            struct Uniforms { view_inv: mat4x4<f32>, proj_inv: mat4x4<f32> };

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var acc_struct: acceleration_structure;

            @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                let x = f32(i32(vi) / 2) * 2.0; let y = f32(i32(vi) & 1) * 2.0;
                return VertexOutput(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
            }

            @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let d = in.uv * 2.0 - 1.0;
                let origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
                let temp = uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0);
                let direction = normalize((uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz);

                var rq: ray_query;
                rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, direction));
                rayQueryProceed(&rq);

                let intersection = rayQueryGetCommittedIntersection(&rq);
                if (intersection.kind != 0u) {
                    return vec4<f32>(intersection.barycentrics, 1.0, 1.0);
                }
                return vec4<f32>(0.0, 0.5, 0.0, 1.0); // GREEN background
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Geometry (Cube)
        vertices = np.array([
            # top (0, 0, 1)
            -1, -1, 1, 0, 0, 1, -1, 1, 1, 0, 1, 1, 1, 1, 1, -1, 1, 0,
            # bottom (0, 0, -1)
            -1, 1, -1, 1, 0, 1, 1, -1, 0, 0, 1, -1, -1, 0, 1, -1, -1, -1, 1, 1,
            # right (1, 0, 0)
            1, -1, -1, 0, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 1, -1, 1, 0, 1,
            # left (-1, 0, 0)
            -1, -1, 1, 1, 0, -1, 1, 1, 0, 0, -1, 1, -1, 0, 1, -1, -1, -1, 1, 1,
            # front (0, 1, 0)
            1, 1, -1, 1, 0, -1, 1, -1, 0, 0, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            # back (0, -1, 0)
            1, -1, 1, 0, 0, -1, -1, 1, 1, 0, -1, -1, -1, 1, 1, 1, -1, -1, 0, 1
        ], dtype=np.float32) # simplified
        
        # Better: 24 vertices for cube
        def get_cube():
            pts = [
                [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1],
                [-1,-1,-1], [-1, 1,-1], [ 1, 1,-1], [ 1,-1,-1]
            ]
            indices = [
                0, 1, 2, 2, 3, 0, # front
                4, 5, 6, 6, 7, 4, # back
                7, 6, 2, 2, 1, 7, # right
                4, 0, 3, 3, 5, 4, # left
                3, 2, 6, 6, 5, 3, # top
                4, 7, 1, 1, 0, 4  # bottom
            ]
            v = []
            for p in pts: v += p + [0, 0] # pos + uv
            return np.array(v, dtype=np.float32), np.array(indices, dtype=np.uint16)

        cube_v, cube_i = get_cube()
        self.v_buf = device.create_buffer(pywgpu.BufferDescriptor(size=cube_v.nbytes, usage=[pywgpu.BufferUsages.VERTEX, pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.v_buf, 0, cube_v.tobytes())
        self.i_buf = device.create_buffer(pywgpu.BufferDescriptor(size=cube_i.nbytes, usage=[pywgpu.BufferUsages.INDEX, pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.i_buf, 0, cube_i.tobytes())

        # 3. AS
        blas_size = pywgpu.BlasTriangleGeometrySizeDescriptor(vertex_format=pywgpu.VertexFormat.float32x3, vertex_count=8, index_format=pywgpu.IndexFormat.uint16, index_count=36)
        self.blas = device.create_blas(pywgpu.BlasDescriptor(), [blas_size])
        self.tlas = device.create_tlas(pywgpu.TlasDescriptor(max_instances=64))

        encoder = device.create_command_encoder()
        encoder.build_acceleration_structures(
            blas=[pywgpu.BlasBuildEntry(blas=self.blas, geometry=[pywgpu.BlasTriangleGeometry(size=blas_size, vertex_buffer=self.v_buf, vertex_stride=20, index_buffer=self.i_buf, first_index=0)])],
            tlas=self.tlas
        )
        queue.submit([encoder.finish()])

        # 4. Uniforms
        self.view_mat = Mat4.look_at_rh(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        proj = Mat4.perspective_rh(math.radians(59.0), config.width / config.height, 0.001, 1000.0)
        self.u_buf = device.create_buffer(pywgpu.BufferDescriptor(size=128, usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.u_buf, 0, struct.pack("16f16f", *self.view_mat.inverse().to_cols_array(), *proj.inverse().to_cols_array()))

        # 5. Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=None,
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState()
        ))
        self.bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.u_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.tlas.as_entire_binding())
            ]
        ))
        self.start_time = time.time()

    def render(self, view, device, queue):
        t = time.time() - self.start_time
        dist = 12.0
        side = 8
        for x in range(side):
            for y in range(side):
                xf = (x / (side-1) * 2.0 - 1.0) * dist
                yf = (y / (side-1) * 2.0 - 1.0) * dist
                # rotation
                angle = t * 0.5 * (1.0 + (x+y*side)*0.01)
                rot = quat_to_mat3(np.array([math.sin(angle), math.sin(angle*0.7), math.sin(angle*0.5), math.cos(angle)]))
                rot /= np.linalg.norm(rot, axis=0) # normalize
                transform = Mat4.from_rotation_translation(rot, np.array([xf, yf, -24.0]))
                self.tlas[x + y*side] = pywgpu.TlasInstance(blas=self.blas, transform=transform.to_cols_array()[:12], custom_index=0)

        encoder = device.create_command_encoder()
        encoder.build_acceleration_structures(tlas=self.tlas)
        
        rpass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(view=view, ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color.black), store=pywgpu.StoreOp.store))]
        ))
        rpass.set_pipeline(self.pipeline)
        rpass.set_bind_group(0, self.bg)
        rpass.draw(3, 1)
        rpass.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(RayCubeFragmentExample))
