#pragma once

// Declarations for headers that hold dvv-gpu objects by pointer or reference. Includes nothing.

namespace dvv::gpu {
class GpuRegistry;                // the declared class until 1.0.0; 1.0.0: class GpuInstance;
using GpuInstance = GpuRegistry;  // the name to use; 1.0.0: this line goes
class ExecutionContext;
class GpuBuffer;
class GpuImage;
class MappedRead;
class MappedWrite;
}  // namespace dvv::gpu
