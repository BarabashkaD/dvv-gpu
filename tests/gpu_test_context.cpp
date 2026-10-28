#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/platform/native_window.h"
#include "gpu_fixture.h"

using dvv::gpu::DeviceId;
using dvv::gpu::ExecutionContext;
using dvv::gpu::detail::Access;
using dvv::gpu::test::GpuTest;

namespace {

// A buffer made directly through VMA, outside GpuBuffer, to exercise the retire queue alone.
vk::Buffer rawBuffer(dvv::gpu::detail::DeviceCore& core, VmaAllocation& allocation) {
    const vk::BufferCreateInfo info{.size = 256,
                                    .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                    .sharingMode = vk::SharingMode::eExclusive};
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    VkBuffer buffer = VK_NULL_HANDLE;
    REQUIRE(vmaCreateBuffer(core.allocator(), &static_cast<const VkBufferCreateInfo&>(info),
                            &allocation_info, &buffer, &allocation, nullptr) == VK_SUCCESS);
    return vk::Buffer(buffer);
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a context runs on the chosen device in its first generation") {
    ExecutionContext& ctx = context();
    CHECK(ctx.deviceId() == device());
    CHECK(ctx.generation() == dvv::gpu::FIRST_GENERATION);
    CHECK(ctx.deviceInfo().name == registry().devices()[device().index()].name);
    CHECK(ctx.registry().get() == &registry());
    CHECK(ctx.ownerId() == registry().objectId());
    CHECK(ctx.typeName() == "dvv::gpu::ExecutionContext");

    const auto stats = ctx.retireStats();
    CHECK(stats.retiring == 0);
    CHECK(stats.retired == 0);
    CHECK(stats.freed == 0);
    CHECK(ctx.drain().has_value());
}

TEST_CASE_FIXTURE(GpuTest, "createContext rejects a device that does not exist") {
    const auto missing =
        registry().createContext(DeviceId(static_cast<std::uint32_t>(registry().devices().size())));
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code == dvv::Errc::Internal);
}

TEST_CASE("a presentation context enables VK_KHR_swapchain") {
    const auto registry =
        dvv::gpu::GpuRegistry::create({.windowSystem = dvv::platform::defaultWindowSystem()});
    if (!registry) {
        FAIL(registry.error().describe());
    }
    const auto best_device = (*registry)->bestDevice();
    REQUIRE(best_device.has_value());
    const DeviceId best = best_device.value_or(DeviceId(0));
    {
        const auto presenting = (*registry)->createContext(best, {.presentation = true});
        if (!presenting) {
            FAIL(presenting.error().describe());
        }
        CHECK((*presenting)->presentation());
        CHECK(Access::core(**presenting).dispatch().vkCreateSwapchainKHR != nullptr);
    }
    {
        const auto plain = (*registry)->createContext(best);
        if (!plain) {
            FAIL(plain.error().describe());
        }
        CHECK_FALSE((*plain)->presentation());
        CHECK(Access::core(**plain).dispatch().vkCreateSwapchainKHR == nullptr);
    }
    CHECK((*registry)->validationMessageCount() == 0);
}

TEST_CASE_FIXTURE(GpuTest, "presentation needs a registry with a window system") {
    const auto context = registry().createContext(device(), {.presentation = true});
    REQUIRE_FALSE(context.has_value());
    CHECK(context.error().code == dvv::Errc::Internal);
}

TEST_CASE_FIXTURE(GpuTest, "a device that misses requirements gets no device") {
    // White-box: the usable device as if it lacked a required feature.
    dvv::gpu::GpuInfo info = registry().devices()[device().index()];
    info.missingRequirements = {"maintenance6"};
    const auto& instance = Access::instance(registry());
    const auto core =
        dvv::gpu::detail::DeviceCore::create(registry().shared_from_this(), instance,
                                             instance.physicalDevices()[device().index()], info);
    REQUIRE_FALSE(core.has_value());
    CHECK(core.error().code == dvv::Errc::Unsupported);
}

TEST_CASE_FIXTURE(GpuTest, "retired objects are freed once the timeline reaches their value") {
    auto& core = Access::core(context());
    VmaAllocation first_allocation = nullptr;
    const vk::Buffer first = rawBuffer(core, first_allocation);

    // Nothing submitted yet: tagged 0, which the timeline has already reached.
    core.retire(first, first_allocation);
    CHECK(context().retireStats().retiring == 1);
    CHECK(core.collect() == 1);
    CHECK(context().retireStats().retiring == 0);

    // As if a submission signalling 1 were being recorded: what is retired now waits for it.
    CHECK(core.reserveSubmitValue() == 1);
    CHECK(core.lastReservedValue() == 1);
    CHECK(core.lastSubmittedValue() == 0);
    VmaAllocation second_allocation = nullptr;
    const vk::Buffer second = rawBuffer(core, second_allocation);
    core.retire(second, second_allocation);
    CHECK(core.collect() == 0);
    CHECK(context().retireStats().retiring == 1);
    {
        const auto lock = core.lockQueue();
        core.setSubmitted(1, lock);
    }
    CHECK(core.lastSubmittedValue() == 1);

    // The "submission" completes.
    const std::uint64_t one = 1;
    const vk::SemaphoreSignalInfo signal{.semaphore = core.timeline(), .value = one};
    REQUIRE(core.device().signalSemaphore(signal, core.dispatch()) == vk::Result::eSuccess);
    CHECK(core.completedValue().value() == 1);
    CHECK(core.collect() == 1);

    const auto stats = context().retireStats();
    CHECK(stats.retiring == 0);
    CHECK(stats.retired == 2);
    CHECK(stats.freed == 2);
}

TEST_CASE_FIXTURE(GpuTest, "drain frees everything retired after waiting for the device") {
    auto& core = Access::core(context());
    {
        const std::uint64_t value = core.reserveSubmitValue();
        const auto lock = core.lockQueue();
        core.setSubmitted(value, lock);  // never signalled
    }
    VmaAllocation allocation = nullptr;
    const vk::Buffer buffer = rawBuffer(core, allocation);
    core.retire(buffer, allocation);
    CHECK(core.collect() == 0);
    // The device is idle, so nothing can still use the buffer.
    REQUIRE(context().drain().has_value());
    CHECK(context().retireStats().retiring == 0);
    CHECK(context().retireStats().freed == 1);
}

TEST_CASE("a context keeps its registry alive") {
    auto registry = dvv::gpu::GpuRegistry::create().value();
    const dvv::ObjectId registry_id = registry->objectId();
    std::shared_ptr<ExecutionContext> ctx =
        registry->createContext(registry->rankedDevices().front()).value();  // main() checked it
    const std::weak_ptr<dvv::gpu::GpuRegistry> watch = registry;
    registry.reset();
    CHECK_FALSE(watch.expired());
    CHECK(ctx->registry()->objectId() == registry_id);
    CHECK(ctx->drain().has_value());
    auto kept = watch.lock();
    REQUIRE(kept);
    ctx.reset();  // destroys the device; `kept` holds the registry
    // Counted after the device is gone, so what destroying it reported counts too.
    CHECK(kept->validationMessageCount() == 0);
    CHECK(kept.use_count() == 1);
    kept.reset();
    CHECK(watch.expired());
}
