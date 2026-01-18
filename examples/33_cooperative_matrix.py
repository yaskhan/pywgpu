import asyncio
import struct
import time
import math
import numpy as np
from typing import List, Optional, Any

import pywgpu
from framework import Example, run_example

class CooperativeMatrixExample(Example):
    TITLE = "Cooperative Matrix Multiply"

    def required_features(self) -> List[pywgpu.Features]:
        # Note: These are experimental/native-only in WGPU
        return [
            pywgpu.Features.EXPERIMENTAL_COOPERATIVE_MATRIX,
            pywgpu.Features.SHADER_F16
        ]

    async def init(self, config, adapter, device, queue):
        # 1. Check for support
        props = adapter.get_cooperative_matrix_properties()
        if not props:
            # For the example to "run" even on unsupported hardware (in mock mode),
            # we'll use a default 8x8 configuration if none found.
            print("Cooperative matrix not supported by hardware, using mock 8x8 f32 config.")
            selected_config = {
                'm_size': 8, 'n_size': 8, 'k_size': 8,
                'ab_type': 'f32', 'cr_type': 'f32'
            }
        else:
            # Find f32 8x8 or similar
            selected_config = None
            for p in props:
                if p.m_size == 8 and p.ab_type == 'f32':
                    selected_config = p
                    break
            if not selected_config:
                selected_config = props[0]
            print(f"Hardware supported config: {selected_config}")

        self.config = selected_config
        self.TILE_SIZE = selected_config['m_size']
        self.use_f16 = selected_config['ab_type'] == 'f16'

        # 2. Shader
        precision = "f16" if self.use_f16 else "f32"
        shader_code = f"""
            enable wgpu_cooperative_matrix;
            { "enable f16;" if self.use_f16 else "" }

            const TILE_SIZE: u32 = {self.TILE_SIZE}u;

            @group(0) @binding(0) var<storage, read> matrix_a: array<{precision}>;
            @group(0) @binding(1) var<storage, read> matrix_b: array<{precision}>;
            @group(0) @binding(2) var<storage, read_write> matrix_c: array<{precision}>;
            @group(0) @binding(3) var<uniform> dims: vec4<u32>; // M, N, K, stride

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(workgroup_id) wid: vec3<u32>) {{
                let M = dims.x; let N = dims.y; let K = dims.z; let stride = dims.w;
                let t_row = wid.x * TILE_SIZE;
                let t_col = wid.y * TILE_SIZE;

                let c_off = t_row * stride + t_col;
                var c_tile = coopLoad<coop_mat{self.TILE_SIZE}x{self.TILE_SIZE}<{precision}, C>>(&matrix_c[c_off], stride);

                for (var k: u32 = 0u; k < K; k += TILE_SIZE) {{
                    let a_off = t_row * K + k;
                    let a_tile = coopLoad<coop_mat{self.TILE_SIZE}x{self.TILE_SIZE}<{precision}, A>>(&matrix_a[a_off], K);
                    
                    let b_off = k * stride + t_col;
                    let b_tile = coopLoad<coop_mat{self.TILE_SIZE}x{self.TILE_SIZE}<{precision}, B>>(&matrix_b[b_off], stride);

                    c_tile = coopMultiplyAdd(a_tile, b_tile, c_tile);
                }}

                coopStore(c_tile, &matrix_c[c_off], stride);
            }}
        """
        self.module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 3. Data
        M, N, K = 64, 64, 64
        dtype = np.float16 if self.use_f16 else np.float32
        
        a_data = (np.arange(M * K) % 7 * 0.1).astype(dtype)
        b_data = (np.arange(K * N) % 11 * 0.1).astype(dtype)
        c_data = np.zeros(M * N, dtype=dtype)

        self.a_buf = device.create_buffer(pywgpu.BufferDescriptor(size=a_data.nbytes, usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_DST]))
        self.b_buf = device.create_buffer(pywgpu.BufferDescriptor(size=b_data.nbytes, usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_DST]))
        self.c_buf = device.create_buffer(pywgpu.BufferDescriptor(size=c_data.nbytes, usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC, pywgpu.BufferUsages.COPY_DST]))
        self.u_buf = device.create_buffer(pywgpu.BufferDescriptor(size=16, usage=pywgpu.BufferUsages.UNIFORM))

        queue.write_buffer(self.a_buf, 0, a_data.tobytes())
        queue.write_buffer(self.b_buf, 0, b_data.tobytes())
        queue.write_buffer(self.c_buf, 0, c_data.tobytes())
        queue.write_buffer(self.u_buf, 0, struct.pack("4I", M, N, K, N))

        # 4. Pipeline
        self.pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=None,
            module=self.module,
            entry_point="main"
        ))
        self.bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.a_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.b_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=2, resource=self.c_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=3, resource=self.u_buf.as_entire_binding())
            ]
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, self.bg)
        cpass.dispatch_workgroups(64 // self.TILE_SIZE, 64 // self.TILE_SIZE, 1)
        cpass.end()
        queue.submit([encoder.finish()])
        print("Cooperative Matrix MatMul dispatched.")

if __name__ == "__main__":
    asyncio.run(run_example(CooperativeMatrixExample))
