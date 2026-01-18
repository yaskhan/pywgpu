import asyncio
import struct
import random
from typing import List

import pywgpu
from framework import Example, run_example

NUM_PARTICLES = 1500
PARTICLES_PER_GROUP = 64

class BoidsExample(Example):
    TITLE = "Boids Example"

    async def init(self, config, adapter, device, queue):
        self.frame_num = 0
        
        # Shaders
        compute_shader_code = """
            struct Particle {
                pos : vec2<f32>,
                vel : vec2<f32>,
            };

            struct SimParams {
                deltaT : f32,
                rule1Distance : f32,
                rule2Distance : f32,
                rule3Distance : f32,
                rule1Scale : f32,
                rule2Scale : f32,
                rule3Scale : f32,
            };

            @group(0) @binding(0) var<uniform> params : SimParams;
            @group(0) @binding(1) var<storage, read> particlesSrc : array<Particle>;
            @group(0) @binding(2) var<storage, read_write> particlesDst : array<Particle>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
                let total = arrayLength(&particlesSrc);
                let index = global_invocation_id.x;
                if (index >= total) {
                    return;
                }

                var vPos : vec2<f32> = particlesSrc[index].pos;
                var vVel : vec2<f32> = particlesSrc[index].vel;

                var cMass : vec2<f32> = vec2<f32>(0.0, 0.0);
                var cVel : vec2<f32> = vec2<f32>(0.0, 0.0);
                var colVel : vec2<f32> = vec2<f32>(0.0, 0.0);
                var cMassCount : i32 = 0;
                var cVelCount : i32 = 0;

                for (var i: u32 = 0u; i < total; i = i + 1u) {
                    if (i == index) {
                        continue;
                    }

                    let pos = particlesSrc[i].pos;
                    let vel = particlesSrc[i].vel;

                    if (distance(pos, vPos) < params.rule1Distance) {
                        cMass += pos;
                        cMassCount += 1;
                    }
                    if (distance(pos, vPos) < params.rule2Distance) {
                        colVel -= pos - vPos;
                    }
                    if (distance(pos, vPos) < params.rule3Distance) {
                        cVel += vel;
                        cVelCount += 1;
                    }
                }
                
                if (cMassCount > 0) {
                    cMass = cMass * (1.0 / f32(cMassCount)) - vPos;
                }
                if (cVelCount > 0) {
                    cVel *= 1.0 / f32(cVelCount);
                }

                vVel = vVel + (cMass * params.rule1Scale) +
                    (colVel * params.rule2Scale) +
                    (cVel * params.rule3Scale);

                vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
                vPos += vVel * params.deltaT;

                if (vPos.x < -1.0) { vPos.x = 1.0; }
                if (vPos.x > 1.0) { vPos.x = -1.0; }
                if (vPos.y < -1.0) { vPos.y = 1.0; }
                if (vPos.y > 1.0) { vPos.y = -1.0; }

                particlesDst[index] = Particle(vPos, vVel);
            }
        """
        
        draw_shader_code = """
            @vertex
            fn main_vs(
                @location(0) particle_pos: vec2<f32>,
                @location(1) particle_vel: vec2<f32>,
                @location(2) position: vec2<f32>,
            ) -> @builtin(position) vec4<f32> {
                let angle = -atan2(particle_vel.x, particle_vel.y);
                let pos = vec2<f32>(
                    position.x * cos(angle) - position.y * sin(angle),
                    position.x * sin(angle) + position.y * cos(angle)
                );
                return vec4<f32>(pos + particle_pos, 0.0, 1.0);
            }

            @fragment
            fn main_fs() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }
        """

        compute_shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=compute_shader_code))
        draw_shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=draw_shader_code))

        # Simulation parameters
        sim_param_data = struct.pack("7f", 0.04, 0.1, 0.025, 0.025, 0.02, 0.05, 0.005)
        sim_param_buffer = device.create_buffer_init(
            label="Simulation Parameter Buffer",
            contents=sim_param_data,
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        )

        # Compute bind group layout
        compute_bind_group_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
            entries=[
                pywgpu.BindGroupLayoutEntry(
                    binding=0, visibility=pywgpu.ShaderStages.COMPUTE,
                    ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.uniform(), min_binding_size=28)
                ),
                pywgpu.BindGroupLayoutEntry(
                    binding=1, visibility=pywgpu.ShaderStages.COMPUTE,
                    ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.storage(read_only=True), min_binding_size=NUM_PARTICLES * 16)
                ),
                pywgpu.BindGroupLayoutEntry(
                    binding=2, visibility=pywgpu.ShaderStages.COMPUTE,
                    ty=pywgpu.BindingType.buffer(ty=pywgpu.BufferBindingType.storage(read_only=False), min_binding_size=NUM_PARTICLES * 16)
                )
            ]
        ))
        
        compute_pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(
            bind_group_layouts=[compute_bind_group_layout]
        ))

        self.compute_pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=compute_pipeline_layout,
            module=compute_shader,
            entry_point="main"
        ))

        # Render pipeline
        render_pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(
            bind_group_layouts=[]
        ))

        self.render_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=render_pipeline_layout,
            vertex=pywgpu.VertexState(
                module=draw_shader,
                entry_point="main_vs",
                buffers=[
                    pywgpu.VertexBufferLayout(
                        array_stride=16,
                        step_mode=pywgpu.VertexStepMode.instance,
                        attributes=[
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=0, shader_location=0),
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=8, shader_location=1)
                        ]
                    ),
                    pywgpu.VertexBufferLayout(
                        array_stride=8,
                        step_mode=pywgpu.VertexStepMode.vertex,
                        attributes=[
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=0, shader_location=2)
                        ]
                    )
                ]
            ),
            fragment=pywgpu.FragmentState(
                module=draw_shader,
                entry_point="main_fs",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState()
        ))

        # Vertex buffer (triangle shape)
        vertex_buffer_data = struct.pack("6f", -0.01, -0.02, 0.01, -0.02, 0.00, 0.02)
        self.vertices_buffer = device.create_buffer_init(
            label="Vertex Buffer",
            contents=vertex_buffer_data,
            usage=pywgpu.BufferUsages.VERTEX
        )

        # Initial particle data
        initial_particle_data = b""
        for _ in range(NUM_PARTICLES):
            pos_x = random.uniform(-1, 1)
            pos_y = random.uniform(-1, 1)
            vel_x = random.uniform(-0.1, 0.1)
            vel_y = random.uniform(-0.1, 0.1)
            initial_particle_data += struct.pack("4f", pos_x, pos_y, vel_x, vel_y)

        # Buffers for ping-pong
        self.particle_buffers = []
        for i in range(2):
            self.particle_buffers.append(device.create_buffer_init(
                label=f"Particle Buffer {i}",
                contents=initial_particle_data,
                usage=[pywgpu.BufferUsages.VERTEX, pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_DST]
            ))

        # Bind groups for compute
        self.particle_bind_groups = []
        for i in range(2):
            self.particle_bind_groups.append(device.create_bind_group(pywgpu.BindGroupDescriptor(
                layout=compute_bind_group_layout,
                entries=[
                    pywgpu.BindGroupEntry(binding=0, resource=sim_param_buffer.as_entire_binding()),
                    pywgpu.BindGroupEntry(binding=1, resource=self.particle_buffers[i].as_entire_binding()),
                    pywgpu.BindGroupEntry(binding=2, resource=self.particle_buffers[(i + 1) % 2].as_entire_binding())
                ]
            )))

        self.work_group_count = (NUM_PARTICLES + PARTICLES_PER_GROUP - 1) // PARTICLES_PER_GROUP

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        # Compute pass
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.compute_pipeline)
        compute_pass.set_bind_group(0, self.particle_bind_groups[self.frame_num % 2])
        compute_pass.dispatch_workgroups(self.work_group_count, 1, 1)
        compute_pass.end()

        # Render pass
        render_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[
                pywgpu.RenderPassColorAttachment(
                    view=view,
                    ops=pywgpu.Operations(
                        load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)),
                        store=pywgpu.StoreOp.store
                    )
                )
            ]
        ))
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2])
        render_pass.set_vertex_buffer(1, self.vertices_buffer)
        render_pass.draw(vertices=range(3), instances=range(NUM_PARTICLES))
        render_pass.end()

        self.frame_num += 1
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(BoidsExample))
