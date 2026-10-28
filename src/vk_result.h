#pragma once

// Vulkan results become dvv::Error here and nowhere else (spec §11).

#include <expected>
#include <source_location>
#include <string_view>

#include "dvv/base/error.h"
#include "vk.h"

namespace dvv::gpu::detail {

// VK_ERROR_DEVICE_LOST -> DeviceLost; out of host/device/pool memory, fragmentation, a failed map
// and too many objects -> OutOfMemory; a missing driver, layer, extension, feature or format and a
// refused permission -> Unsupported; everything else (including success codes that a caller
// treats as a failure) -> Internal.
[[nodiscard]] Errc errcOf(vk::Result result) noexcept;

// Errc from errcOf(), message "<what> failed: <result name> (VkResult <number>)".
[[nodiscard]] std::unexpected<Error> vkError(
    vk::Result result, std::string_view what,
    std::source_location where = std::source_location::current());
// For the C results of VMA and vk-bootstrap.
[[nodiscard]] std::unexpected<Error> vkError(
    VkResult result, std::string_view what,
    std::source_location where = std::source_location::current());

// Success -> {}, anything else -> vkError().
[[nodiscard]] Result<void> vkCheck(vk::Result result, std::string_view what,
                                   std::source_location where = std::source_location::current());
[[nodiscard]] Result<void> vkCheck(VkResult result, std::string_view what,
                                   std::source_location where = std::source_location::current());

}  // namespace dvv::gpu::detail
