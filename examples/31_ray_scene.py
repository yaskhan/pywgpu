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
    @staticmethod
    def from_translation(v: np.ndarray):
        m = np.identity(4, dtype=np.float32); m[0:3, 3] = v; return Mat4(m)
    @staticmethod
    def from_rotation_y(angle: float):
        c, s = math.cos(angle), math.sin(angle)
        m = np.identity(4, dtype=np.float32); m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c; return Mat4(m)
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

class RaySceneExample(Example):
    TITLE = "Ray Scene (Multiple Geometries & Materials)"

    def required_features(self) -> List[pywgpu.Features]:
        return [pywgpu.Features.EXPERIMENTAL_RAY_QUERY]

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            enable wgpu_ray_query;

            struct VertexOutput { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
            @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                let x = f32(i32(vi) / 2) * 2.0; let y = f32(i32(vi) & 1) * 2.0;
                return VertexOutput(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
            }

            struct Uniforms { view_inv: mat4x4<f32>, proj_inv: mat4x4<f32> };
            struct Vertex { pos: vec3<f32>, normal: vec3<f32>, uv: vec2<f32> };
            struct InstanceData { first_vertex: u32, first_geometry: u32, last_geometry: u32, _pad: u32 };
            struct Material { albedo: vec3<f32>, roughness: f32 };
            struct Geometry { first_index: u32, material: Material };

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
            @group(0) @binding(2) var<storage, read> indices: array<u32>;
            @group(0) @binding(3) var<storage, read> geometries: array<Geometry>;
            @group(0) @binding(4) var<storage, read> instances_data: array<InstanceData>;
            @group(0) @binding(5) var acc_struct: acceleration_structure;

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
                    let inst = instances_data[intersection.instance_custom_data];
                    let geom = geometries[intersection.geometry_index + inst.first_geometry];
                    let mat = geom.material;
                    
                    // Simple lighting
                    let first_idx = intersection.primitive_index * 3u + geom.first_index;
                    let v0 = vertices[inst.first_vertex + indices[first_idx + 0u]];
                    let v1 = vertices[inst.first_vertex + indices[first_idx + 1u]];
                    let v2 = vertices[inst.first_vertex + indices[first_idx + 2u]];
                    
                    let b = vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y, intersection.barycentrics);
                    let normal = normalize(v0.normal * b.x + v1.normal * b.y + v2.normal * b.z);
                    let light_dir = normalize(vec3<f32>(1.0, 2.0, 3.0));
                    let diff = max(dot(normal, light_dir), 0.2);
                    
                    return vec4<f32>(mat.albedo * diff, 1.0);
                }
                return vec4<f32>(0.1, 0.1, 0.1, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Scene Data
        # Cube
        cube_v = []
        def add_cube_face(p1, p2, p3, p4, n):
            for p in [p1, p2, p3, p1, p3, p4]:
                cube_v.append({'pos': p, 'normal': n, 'uv': [0, 0]})
        
        # simplified 24 vertices for cube
        add_cube_face([-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1], [ 0, 0, 1])
        add_cube_face([-1,-1,-1], [-1, 1,-1], [ 1, 1,-1], [ 1,-1,-1], [ 0, 0,-1])
        # ... and so on, but for demo let's just use indices
        
        # Let's use more structured vertex data
        scene_vertices = []
        scene_indices = []
        
        # Geometry 1: Cube
        v_start = len(scene_vertices)
        i_start = len(scene_indices)
        # Cube vertices (8)
        cube_pts = [
            [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
            [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]
        ]
        for p in cube_pts:
            scene_vertices.append({'pos': p, 'normal': [0,0,0], 'uv': [0,0]}) # Normals will be approximate or we use 24 vertices
        
        # Better: 24 vertices for cube to get sharp normals
        scene_vertices = []
        def add_quad(pts, normal):
            start = len(scene_vertices)
            for p in pts:
                scene_vertices.append({'pos': p, 'normal': normal, 'uv': [0,0]})
            scene_indices.extend([start, start+1, start+2, start, start+2, start+3])

        add_quad([[-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]], [0,0,1]) # Front
        add_quad([[-1,-1,-1], [-1, 1,-1], [ 1, 1,-1], [ 1,-1,-1]], [0,0,-1]) # Back
        add_quad([[ 1,-1,-1], [ 1, 1,-1], [ 1, 1, 1], [ 1,-1, 1]], [1,0,0]) # Right
        add_quad([[-1,-1,-1], [-1,-1, 1], [-1, 1, 1], [-1, 1,-1]], [-1,0,0]) # Left
        add_quad([[-1, 1,-1], [-1, 1, 1], [ 1, 1, 1], [ 1, 1,-1]], [0,1,0]) # Top
        add_quad([[-1,-1,-1], [ 1,-1,-1], [ 1,-1, 1], [-1,-1, 1]], [0,-1,0]) # Bottom
        cube_v_count = 24
        cube_i_count = 36
        
        # Geometry 2: Pyramid
        pyram_v_start = len(scene_vertices)
        pyram_i_start = len(scene_indices)
        pts = [[0, 1, 0], [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]]
        # 4 sides
        def add_tri(p1, p2, p3):
            n = np.cross(np.array(p2)-p1, np.array(p3)-p1).tolist()
            start = len(scene_vertices)
            for p in [p1, p2, p3]:
                scene_vertices.append({'pos': p, 'normal': n, 'uv': [0,0]})
            scene_indices.extend([start, start+1, start+2])
        
        add_tri(pts[0], pts[1], pts[2])
        add_tri(pts[0], pts[2], pts[3])
        add_tri(pts[0], pts[3], pts[4])
        add_tri(pts[0], pts[4], pts[1])
        # base
        add_quad([pts[1], pts[4], pts[3], pts[2]], [0,-1,0])
        pyram_v_count = len(scene_vertices) - pyram_v_start
        pyram_i_count = len(scene_indices) - pyram_i_start

        # Upload Scene Buffers
        raw_v = []
        for v in scene_vertices:
            raw_v += v['pos'] + [0] # padding to 16 bytes for storage alignment if needed, or structured
            raw_v += v['normal'] + [0]
            raw_v += v['uv']
            raw_v += [0, 0] # pad
        
        self.v_buf = device.create_buffer(pywgpu.BufferDescriptor(size=len(raw_v)*4, usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.v_buf, 0, np.array(raw_v, dtype=np.float32).tobytes())
        
        self.i_buf = device.create_buffer(pywgpu.BufferDescriptor(size=len(scene_indices)*4, usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.BLAS_INPUT, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.i_buf, 0, np.array(scene_indices, dtype=np.uint32).tobytes())

        # Descriptors
        cube_size = pywgpu.BlasTriangleGeometrySizeDescriptor(vertex_format=pywgpu.VertexFormat.float32x3, vertex_count=cube_v_count, index_format=pywgpu.IndexFormat.uint32, index_count=cube_i_count)
        pyram_size = pywgpu.BlasTriangleGeometrySizeDescriptor(vertex_format=pywgpu.VertexFormat.float32x3, vertex_count=pyram_v_count, index_format=pywgpu.IndexFormat.uint32, index_count=pyram_i_count)
        
        self.blas_cube = device.create_blas(pywgpu.BlasDescriptor(), [cube_size])
        self.blas_pyram = device.create_blas(pywgpu.BlasDescriptor(), [pyram_size])
        
        self.tlas = device.create_tlas(pywgpu.TlasDescriptor(max_instances=2))
        self.tlas[0] = pywgpu.TlasInstance(blas=self.blas_cube, transform=Mat4.from_translation(np.array([-2, 0, -5])).to_cols_array()[:12], custom_index=0)
        self.tlas[1] = pywgpu.TlasInstance(blas=self.blas_pyram, transform=Mat4.from_translation(np.array([2, 0, -5])).to_cols_array()[:12], custom_index=1)

        # Build AS
        encoder = device.create_command_encoder()
        encoder.build_acceleration_structures(
            blas=[
                pywgpu.BlasBuildEntry(blas=self.blas_cube, geometry=[pywgpu.BlasTriangleGeometry(size=cube_size, vertex_buffer=self.v_buf, first_vertex=0, vertex_stride=40, index_buffer=self.i_buf, first_index=0)]),
                pywgpu.BlasBuildEntry(blas=self.blas_pyram, geometry=[pywgpu.BlasTriangleGeometry(size=pyram_size, vertex_buffer=self.v_buf, first_vertex=24, vertex_stride=40, index_buffer=self.i_buf, first_index=36)])
            ],
            tlas=self.tlas
        )
        queue.submit([encoder.finish()])

        # Materials & Instance Data Storage
        # Material: albedo (3f), roughness (1f)
        # Geometry: first_index (1u), material (4f)
        # InstanceData: first_vertex (1u), first_geometry (1u), last_geometry (1u), pad (1u)
        
        mats = [0.8, 0.2, 0.2, 0.5,  1.0, 0.8, 0.2, 0.1] # Red cube, Yellow pyramid
        geoms = [0, 0.8, 0.2, 0.2, 0.5,  36, 1.0, 0.8, 0.2, 0.1] # geom0: cube, geom1: pyram
        insts = [0, 0, 1, 0,  24, 1, 2, 0] # inst0: cube, inst1: pyram
        
        self.mat_buf = device.create_buffer(pywgpu.BufferDescriptor(size=len(geoms)*4, usage=pywgpu.BufferUsages.STORAGE))
        queue.write_buffer(self.mat_buf, 0, np.array(geoms, dtype=np.float32).tobytes())
        
        self.inst_buf = device.create_buffer(pywgpu.BufferDescriptor(size=len(insts)*4, usage=pywgpu.BufferUsages.STORAGE))
        queue.write_buffer(self.inst_buf, 0, np.array(insts, dtype=np.uint32).tobytes())

        # Uniforms
        view_mat = Mat4.look_at_rh(np.array([0.0, 2.0, 10.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        proj_mat = Mat4.perspective_rh(math.radians(59.0), config.width / config.height, 0.1, 100.0)
        self.u_buf = device.create_buffer(pywgpu.BufferDescriptor(size=128, usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]))
        queue.write_buffer(self.u_buf, 0, struct.pack("16f16f", *view_mat.inverse().to_cols_array(), *proj_mat.inverse().to_cols_array()))

        # Pipeline
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
                pywgpu.BindGroupEntry(binding=1, resource=self.v_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=2, resource=self.i_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=3, resource=self.mat_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=4, resource=self.inst_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=5, resource=self.tlas.as_entire_binding())
            ]
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        rpass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(view=view, ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color.black), store=pywgpu.StoreOp.store))]
        ))
        rpass.set_pipeline(self.pipeline)
        rpass.set_bind_group(0, self.bg)
        rpass.draw(3, 1)
        rpass.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(RaySceneExample))
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
village
 village
 village
 village
