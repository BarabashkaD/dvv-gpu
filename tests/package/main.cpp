#include <cstdio>

#include "consumer_shaders/add_one.h"
#include "dvv/base/error.h"
#include "dvv/gpu/gpu_registry.h"

int main() {
    // A shader embedded by dvv_add_shaders().
    if (consumer_shaders::addOne().spirv.empty() ||
        consumer_shaders::addOne().spirv[0] != 0x07230203U) {
        std::puts("the embedded shader is not SPIR-V");
        return 1;
    }
    // Without a Vulkan loader (CI's Windows runners) the registry reports Unsupported; anything
    // else is a failure. On macOS the third-party prefix always has the loader (found through the
    // build rpath), so any error is a failure there. Validation stays off: a system's older layer
    // may be installed.
    const auto registry =
        dvv::gpu::GpuRegistry::create({.validation = dvv::gpu::ValidationMode::Off});
#if defined(__APPLE__)
    constexpr bool LOADER_REQUIRED = true;
#else
    constexpr bool LOADER_REQUIRED = false;
#endif
    if (!registry && (LOADER_REQUIRED || registry.error().code != dvv::Errc::Unsupported)) {
        std::printf("%s\n", registry.error().describe().c_str());
        return 1;
    }
    std::puts(registry ? "dvv_gpu package works" : "dvv_gpu package works (no Vulkan loader)");
    return 0;
}
