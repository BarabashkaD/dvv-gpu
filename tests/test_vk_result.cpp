#include <doctest/doctest.h>

#include <string>

#include "dvv/base/error.h"
#include "vk_result.h"

using dvv::Errc;
using dvv::gpu::detail::errcOf;
using dvv::gpu::detail::vkCheck;
using dvv::gpu::detail::vkError;

TEST_CASE("Vulkan results map to dvv error codes") {
    CHECK(errcOf(vk::Result::eErrorDeviceLost) == Errc::DeviceLost);

    CHECK(errcOf(vk::Result::eErrorOutOfHostMemory) == Errc::OutOfMemory);
    CHECK(errcOf(vk::Result::eErrorOutOfDeviceMemory) == Errc::OutOfMemory);
    CHECK(errcOf(vk::Result::eErrorOutOfPoolMemory) == Errc::OutOfMemory);
    CHECK(errcOf(vk::Result::eErrorFragmentedPool) == Errc::OutOfMemory);
    CHECK(errcOf(vk::Result::eErrorFragmentation) == Errc::OutOfMemory);
    CHECK(errcOf(vk::Result::eErrorMemoryMapFailed) == Errc::OutOfMemory);
    CHECK(errcOf(vk::Result::eErrorTooManyObjects) == Errc::OutOfMemory);

    CHECK(errcOf(vk::Result::eErrorInitializationFailed) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorIncompatibleDriver) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorLayerNotPresent) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorExtensionNotPresent) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorFeatureNotPresent) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorFormatNotSupported) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorImageUsageNotSupportedKHR) == Errc::Unsupported);
    CHECK(errcOf(vk::Result::eErrorNotPermitted) == Errc::Unsupported);

    CHECK(errcOf(vk::Result::eErrorUnknown) == Errc::Internal);
    CHECK(errcOf(vk::Result::eErrorInvalidExternalHandle) == Errc::Internal);
    CHECK(errcOf(vk::Result::eTimeout) == Errc::Internal);
    CHECK(errcOf(vk::Result::eNotReady) == Errc::Internal);
}

TEST_CASE("a Vulkan error names the call and the result") {
    const auto error = vkError(vk::Result::eErrorDeviceLost, "vkQueueSubmit2").error();
    CHECK(error.code == Errc::DeviceLost);
    CHECK(error.message == "vkQueueSubmit2 failed: ErrorDeviceLost (VkResult -4)");
    CHECK(std::string(error.where.file_name()).find("test_vk_result.cpp") != std::string::npos);

    const auto c_error = vkError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vmaCreateBuffer").error();
    CHECK(c_error.code == Errc::OutOfMemory);
    CHECK(c_error.message == "vmaCreateBuffer failed: ErrorOutOfDeviceMemory (VkResult -2)");
}

TEST_CASE("vkCheck passes success and converts everything else") {
    CHECK(vkCheck(vk::Result::eSuccess, "vkDeviceWaitIdle").has_value());
    CHECK(vkCheck(VK_SUCCESS, "vmaFlushAllocation").has_value());

    const auto timeout = vkCheck(vk::Result::eTimeout, "vkWaitSemaphores");
    REQUIRE_FALSE(timeout.has_value());
    CHECK(timeout.error().code == Errc::Internal);
    CHECK(timeout.error().message == "vkWaitSemaphores failed: Timeout (VkResult 2)");

    const auto layer = vkCheck(VK_ERROR_LAYER_NOT_PRESENT, "vkCreateInstance");
    REQUIRE_FALSE(layer.has_value());
    CHECK(layer.error().code == Errc::Unsupported);
}
