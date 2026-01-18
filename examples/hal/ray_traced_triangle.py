"""
Ray Traced Triangle Example - Simplified Python Translation

This is a simplified ray tracing example demonstrating acceleration structure usage.

Original: wgpu-trunk/wgpu-hal/examples/ray-traced-triangle/main.rs (1200 lines)
"""

import sys
import os

# Add pywgpu-hal to path
hal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pywgpu-hal')
if hal_path not in sys.path:
    sys.path.insert(0, hal_path)

import lib as hal
import pywgpu_types as wgt


class RayTracedTriangleExample:
    """Simplified ray tracing example."""
    
    def __init__(self):
        """Initialize the example."""
        print("Initializing ray tracing example...")
        
        # Create instance with ray tracing support
        instance_desc = hal.InstanceDescriptor(
            name="ray_traced_triangle",
            flags=wgt.InstanceFlags.default(),
            memory_budget_thresholds=wgt.MemoryBudgetThresholds(),
            backend_options=wgt.BackendOptions(),
            telemetry=None,
            display=None
        )
        
        self.instance = hal.Instance.init(instance_desc)
        
        # Enumerate adapters
        print("Enumerating adapters...")
        adapters = self.instance.enumerate_adapters(None)
        
        if not adapters:
            raise RuntimeError("No adapters found")
        
        exposed = adapters[0]
        self.adapter = exposed.adapter
        self.features = exposed.features
        
        # Check for ray tracing support
        if not (self.features & wgt.Features.RAY_TRACING_ACCELERATION_STRUCTURE):
            raise RuntimeError("Ray tracing not supported on this adapter")
        
        print(f"Using adapter: {exposed.info.name}")
        print(f"Ray tracing supported: Yes")
        
        # Open device with ray tracing features
        print("Opening device...")
        open_device = self.adapter.open(
            features=self.features,
            limits=wgt.Limits.default(),
            memory_hints=wgt.MemoryHints.PERFORMANCE
        )
        
        self.device = open_device.device
        self.queue = open_device.queue
        
        print("Device opened!")
    
    def create_blas(self):
        """Create bottom-level acceleration structure.
        
        Returns:
            The created BLAS.
        """
        print("Creating BLAS...")
        
        # Triangle vertices
        vertices = [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0]
        
        # Create vertex buffer
        vertex_buffer_desc = hal.BufferDescriptor(
            label="vertices",
            size=len(vertices) * 4,  # 4 bytes per float
            usage=wgt.BufferUses.BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            memory_flags=hal.MemoryFlags.PREFER_COHERENT
        )
        
        vertex_buffer = self.device.create_buffer(vertex_buffer_desc)
        
        # Define BLAS geometry
        blas_triangles = hal.AccelerationStructureTriangles(
            vertex_buffer=vertex_buffer,
            first_vertex=0,
            vertex_format=wgt.VertexFormat.FLOAT32X3,
            vertex_count=3,
            vertex_stride=12,  # 3 floats * 4 bytes
            indices=None,
            transform=None,
            flags=hal.AccelerationStructureGeometryFlags.OPAQUE
        )
        
        blas_entries = hal.AccelerationStructureEntries.Triangles([blas_triangles])
        
        # Get build sizes
        build_sizes = self.device.get_acceleration_structure_build_sizes(
            hal.GetAccelerationStructureBuildSizesDescriptor(
                entries=blas_entries,
                flags=hal.AccelerationStructureBuildFlags.PREFER_FAST_TRACE
            )
        )
        
        # Create BLAS
        blas_desc = hal.AccelerationStructureDescriptor(
            label="blas",
            size=build_sizes.acceleration_structure_size,
            format=hal.AccelerationStructureFormat.BOTTOM_LEVEL,
            allow_compaction=False
        )
        
        blas = self.device.create_acceleration_structure(blas_desc)
        
        print(f"BLAS created (size: {build_sizes.acceleration_structure_size} bytes)")
        
        return blas, vertex_buffer, blas_entries, build_sizes
    
    def create_tlas(self, blas):
        """Create top-level acceleration structure.
        
        Args:
            blas: The BLAS to instance.
            
        Returns:
            The created TLAS.
        """
        print("Creating TLAS...")
        
        # Get BLAS device address
        blas_address = self.device.get_acceleration_structure_device_address(blas)
        
        # Create instance
        instance = hal.AccelerationStructureInstance(
            transform=[1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0],
            custom_index=0,
            mask=0xFF,
            shader_binding_table_record_offset=0,
            flags=0,
            acceleration_structure_reference=blas_address
        )
        
        # Create instances buffer
        instances_buffer_desc = hal.BufferDescriptor(
            label="instances",
            size=64,  # Size of AccelerationStructureInstance
            usage=wgt.BufferUses.TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            memory_flags=hal.MemoryFlags.PREFER_COHERENT
        )
        
        instances_buffer = self.device.create_buffer(instances_buffer_desc)
        
        # Define TLAS entries
        tlas_entries = hal.AccelerationStructureEntries.Instances(
            hal.AccelerationStructureInstances(
                buffer=instances_buffer,
                count=1,
                offset=0
            )
        )
        
        # Get build sizes
        build_sizes = self.device.get_acceleration_structure_build_sizes(
            hal.GetAccelerationStructureBuildSizesDescriptor(
                entries=tlas_entries,
                flags=hal.AccelerationStructureBuildFlags.PREFER_FAST_TRACE
            )
        )
        
        # Create TLAS
        tlas_desc = hal.AccelerationStructureDescriptor(
            label="tlas",
            size=build_sizes.acceleration_structure_size,
            format=hal.AccelerationStructureFormat.TOP_LEVEL,
            allow_compaction=False
        )
        
        tlas = self.device.create_acceleration_structure(tlas_desc)
        
        print(f"TLAS created (size: {build_sizes.acceleration_structure_size} bytes)")
        
        return tlas, instances_buffer, tlas_entries, build_sizes
    
    def build_acceleration_structures(self, blas, tlas, blas_entries, tlas_entries, 
                                     blas_build_sizes, tlas_build_sizes):
        """Build the acceleration structures.
        
        Args:
            blas: The BLAS to build.
            tlas: The TLAS to build.
            blas_entries: BLAS geometry entries.
            tlas_entries: TLAS instance entries.
            blas_build_sizes: BLAS build size info.
            tlas_build_sizes: TLAS build size info.
        """
        print("Building acceleration structures...")
        
        # Create scratch buffer
        scratch_size = max(blas_build_sizes.build_scratch_size, 
                          tlas_build_sizes.build_scratch_size)
        
        scratch_buffer_desc = hal.BufferDescriptor(
            label="scratch",
            size=scratch_size,
            usage=wgt.BufferUses.ACCELERATION_STRUCTURE_SCRATCH,
            memory_flags=hal.MemoryFlags.NONE
        )
        
        scratch_buffer = self.device.create_buffer(scratch_buffer_desc)
        
        # Create command encoder
        encoder_desc = hal.CommandEncoderDescriptor(
            label=None,
            queue=self.queue
        )
        
        encoder = self.device.create_command_encoder(encoder_desc)
        encoder.begin_encoding(None)
        
        # Build BLAS
        encoder.build_acceleration_structures(
            1,
            [hal.BuildAccelerationStructureDescriptor(
                mode=hal.AccelerationStructureBuildMode.BUILD,
                flags=hal.AccelerationStructureBuildFlags.PREFER_FAST_TRACE,
                destination_acceleration_structure=blas,
                scratch_buffer=scratch_buffer,
                entries=blas_entries,
                source_acceleration_structure=None,
                scratch_buffer_offset=0
            )]
        )
        
        # Barrier
        encoder.place_acceleration_structure_barrier(
            hal.AccelerationStructureBarrier(
                usage=hal.StateTransition(
                    from_=hal.AccelerationStructureUses.BUILD_OUTPUT,
                    to=hal.AccelerationStructureUses.BUILD_INPUT
                )
            )
        )
        
        # Build TLAS
        encoder.build_acceleration_structures(
            1,
            [hal.BuildAccelerationStructureDescriptor(
                mode=hal.AccelerationStructureBuildMode.BUILD,
                flags=hal.AccelerationStructureBuildFlags.PREFER_FAST_TRACE,
                destination_acceleration_structure=tlas,
                scratch_buffer=scratch_buffer,
                entries=tlas_entries,
                source_acceleration_structure=None,
                scratch_buffer_offset=0
            )]
        )
        
        # End encoding
        cmd_buf = encoder.end_encoding()
        
        # Submit
        fence = self.device.create_fence()
        self.queue.submit([cmd_buf], [], (fence, 1))
        
        # Wait
        self.device.wait(fence, 1, None)
        
        print("Acceleration structures built!")
    
    def run(self):
        """Run the example."""
        # Create BLAS
        blas, vertex_buffer, blas_entries, blas_build_sizes = self.create_blas()
        
        # Create TLAS
        tlas, instances_buffer, tlas_entries, tlas_build_sizes = self.create_tlas(blas)
        
        # Build acceleration structures
        self.build_acceleration_structures(
            blas, tlas, blas_entries, tlas_entries,
            blas_build_sizes, tlas_build_sizes
        )
        
        print("\nRay tracing setup complete!")
        print("In a full example, you would now:")
        print("  1. Create a compute shader with ray queries")
        print("  2. Bind the TLAS to the shader")
        print("  3. Dispatch the compute shader")
        print("  4. Copy results to screen")


def main():
    """Main function."""
    print("PyWGPU HAL Ray Traced Triangle Example")
    print("=" * 40)
    
    try:
        example = RayTracedTriangleExample()
        example.run()
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
