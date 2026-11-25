// The neutral dvv-gpu headers include no Vulkan header.
#include <memory>

#include "dvv/gpu/bundled_drivers.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/glyph_strip.h"
#include "dvv/gpu/gpu_fwd.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_info_json.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/testing/fault_injection.h"

#if defined(VULKAN_CORE_H_)
#error "a neutral dvv-gpu header included a Vulkan header"
#endif

#include <doctest/doctest.h>

#include <optional>

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

TEST_CASE("the app-facing headers declare their options without Vulkan") {
    CHECK(dvv::gpu::PresenterOptions{}.window.empty());
    CHECK_FALSE(dvv::gpu::ContextOptions{}.presentation);
    CHECK(dvv::gpu::GpuInstanceOptions{}.windowSystem == std::nullopt);
    CHECK(dvv::gpu::PresentFrameParams{
              .extent = {}, .frameNumber = 0, .previousFrameCompleted = false}
              .encoding == dvv::gpu::PresentEncoding::Linear);
    CHECK(dvv::gpu::FontSpec{}.family.empty());
}
