// The neutral dvv-gpu headers include no Vulkan header.
#include <memory>

#include "dvv/gpu/bundled_drivers.h"
#include "dvv/gpu/gpu_fwd.h"

#if defined(VULKAN_CORE_H_)
#error "a neutral dvv-gpu header included a Vulkan header"
#endif

#include <doctest/doctest.h>

TEST_CASE("the neutral headers declare the GPU types without Vulkan") {
    std::shared_ptr<dvv::gpu::GpuInstance> instance;
    dvv::gpu::ExecutionContext* context = nullptr;
    dvv::gpu::GpuBuffer* buffer = nullptr;
    dvv::gpu::GpuImage* image = nullptr;
    dvv::gpu::MappedRead* read_scope = nullptr;
    dvv::gpu::MappedWrite* write_scope = nullptr;
    CHECK(instance == nullptr);
    CHECK(context == nullptr);
    CHECK(buffer == nullptr);
    CHECK(image == nullptr);
    CHECK(read_scope == nullptr);
    CHECK(write_scope == nullptr);
}
