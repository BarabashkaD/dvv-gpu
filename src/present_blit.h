#pragma once

#include <string_view>

#include "vk.h"

namespace dvv::gpu::detail {

// The filter of CommandRecorder::presentBlit: NEAREST for a 1:1 copy, LINEAR for scaling when the
// source format supports linear filtering, NEAREST otherwise.
[[nodiscard]] constexpr VkFilter presentBlitFilter(VkExtent2D source, VkExtent2D destination,
                                                   VkFormatFeatureFlags2 source_features) noexcept {
    if (source.width == destination.width && source.height == destination.height) {
        return VK_FILTER_NEAREST;
    }
    return (source_features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0
               ? VK_FILTER_LINEAR
               : VK_FILTER_NEAREST;
}

// Whether an image of `format`, whose optimal-tiling features are `features`, can be blitted to a
// UNORM or SRGB swapchain image: BLIT_SRC, one plane, 1x1x1 texel blocks, colour (not depth or
// stencil) and not integer. The single rule shared by CommandRecorder::presentBlit and the
// presenter's image check (design M1b §5.2 step 2), so the two cannot drift apart.
[[nodiscard]] inline bool presentBlitSource(VkFormat format,
                                            VkFormatFeatureFlags2 features) noexcept {
    const auto vk_format = static_cast<vk::Format>(format);
    if ((features & VK_FORMAT_FEATURE_2_BLIT_SRC_BIT) == 0 || vk::componentCount(vk_format) == 0 ||
        vk::planeCount(vk_format) != 1) {
        return false;
    }
    const auto block = vk::blockExtent(vk_format);
    if (block[0] != 1 || block[1] != 1 || block[2] != 1) {
        return false;
    }
    const std::string_view name = vk::componentName(vk_format, 0);
    const std::string_view numeric = vk::componentNumericFormat(vk_format, 0);
    return name != "D" && name != "S" && numeric != "UINT" && numeric != "SINT";
}

// Whether a swapchain format, whose optimal-tiling features are `features`, can take presentBlit:
// BLIT_DST, one plane, colour, and UNORM or SRGB (design M1b D16).
[[nodiscard]] inline bool presentBlitDestination(VkFormat format,
                                                 VkFormatFeatureFlags2 features) noexcept {
    const auto vk_format = static_cast<vk::Format>(format);
    if ((features & VK_FORMAT_FEATURE_2_BLIT_DST_BIT) == 0 || vk::componentCount(vk_format) == 0 ||
        vk::planeCount(vk_format) != 1) {
        return false;
    }
    const std::string_view numeric = vk::componentNumericFormat(vk_format, 0);
    return numeric == "UNORM" || numeric == "SRGB";
}

}  // namespace dvv::gpu::detail
