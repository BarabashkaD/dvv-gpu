#include <cstdio>

#include "consumer_shaders/add_one.h"
#include "dvv/base/error.h"
#include "dvv/gpu/bundled_drivers.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/embedded_shader.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_fwd.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_info_json.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/presenter_vulkan.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "dvv/gpu/testing/fault_injection.h"

// Every public header compiles against the installed package (one use each; gpu_fwd.h only
// declares).
static_assert(sizeof(dvv::gpu::DriverChoice) > 0);
static_assert(sizeof(dvv::gpu::BufferCopy) > 0);
static_assert(sizeof(dvv::gpu::ComputeBindings) > 0);
static_assert(sizeof(&dvv::gpu::vulkanFormat) > 0);
static_assert(sizeof(dvv::gpu::DeviceId) > 0);
static_assert(sizeof(dvv::gpu::EmbeddedShader) > 0);
static_assert(sizeof(dvv::gpu::ExecutionContext) > 0);
static_assert(sizeof(dvv::gpu::GpuInfo) > 0);
static_assert(sizeof(&dvv::gpu::toJson) > 0);
static_assert(sizeof(dvv::gpu::GpuImage) > 0);
static_assert(sizeof(dvv::gpu::PresenterOptions) > 0);
static_assert(sizeof(dvv::gpu::PresentFrame) > 0);
static_assert(sizeof(dvv::gpu::BufferDesc) > 0);
static_assert(sizeof(dvv::gpu::RecordFunction) > 0);
static_assert(sizeof(&dvv::gpu::testing::injectDeviceLost) > 0);

int main() {
    // A shader embedded by dvv_add_shaders().
    if (consumer_shaders::addOne().spirv.empty() ||
        consumer_shaders::addOne().spirv[0] != 0x07230203U) {
        std::puts("the embedded shader is not SPIR-V");
        return 1;
    }
    // Without a Vulkan loader (CI's Windows runners) the GPU instance reports Unsupported; anything
    // else is a failure. On macOS the third-party prefix always has the loader (found through the
    // build rpath), so any error is a failure there. Validation stays off: a system's older layer
    // may be installed.
    const auto gpu_instance =
        dvv::gpu::GpuInstance::create({.validation = dvv::gpu::ValidationMode::Off});
#if defined(__APPLE__)
    constexpr bool LOADER_REQUIRED = true;
#else
    constexpr bool LOADER_REQUIRED = false;
#endif
    if (!gpu_instance && (LOADER_REQUIRED || gpu_instance.error().code != dvv::Errc::Unsupported)) {
        std::printf("%s\n", gpu_instance.error().describe().c_str());
        return 1;
    }
    std::puts(gpu_instance ? "dvv_gpu package works" : "dvv_gpu package works (no Vulkan loader)");
    return 0;
}
