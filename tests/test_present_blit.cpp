#include <doctest/doctest.h>

#include <vulkan/vulkan_core.h>

#include "present_blit.h"

using dvv::gpu::detail::presentBlitDestination;
using dvv::gpu::detail::presentBlitFilter;
using dvv::gpu::detail::presentBlitSource;

TEST_CASE("the blit filter is NEAREST for 1:1 and LINEAR only for scaling with linear filtering") {
    constexpr VkFormatFeatureFlags2 LINEAR = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    CHECK(presentBlitFilter({.width = 4, .height = 4}, {.width = 4, .height = 4}, LINEAR) ==
          VK_FILTER_NEAREST);
    CHECK(presentBlitFilter({.width = 2, .height = 1}, {.width = 4, .height = 1}, LINEAR) ==
          VK_FILTER_LINEAR);
    CHECK(presentBlitFilter({.width = 2, .height = 1}, {.width = 4, .height = 1}, 0) ==
          VK_FILTER_NEAREST);
    CHECK(presentBlitFilter({.width = 4, .height = 4}, {.width = 2, .height = 2},
                            VK_FORMAT_FEATURE_2_BLIT_SRC_BIT) == VK_FILTER_NEAREST);
}

TEST_CASE("presentBlitSource accepts blittable colour formats only") {
    constexpr VkFormatFeatureFlags2 B = VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;
    for (const VkFormat format :
         {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R16G16B16A16_SFLOAT,
          VK_FORMAT_A2B10G10R10_UNORM_PACK32}) {
        CAPTURE(static_cast<int>(format));
        CHECK(presentBlitSource(format, B));
    }
    CHECK_FALSE(presentBlitSource(VK_FORMAT_R8G8B8A8_UNORM, 0));
    for (const VkFormat format :
         {VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_D16_UNORM,
          VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_S8_UINT, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
          VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_UNDEFINED}) {
        CAPTURE(static_cast<int>(format));
        CHECK_FALSE(presentBlitSource(format, B));
    }
}

TEST_CASE("presentBlitDestination accepts UNORM and SRGB formats with BLIT_DST only") {
    constexpr VkFormatFeatureFlags2 D = VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
    for (const VkFormat format :
         {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_A2B10G10R10_UNORM_PACK32}) {
        CAPTURE(static_cast<int>(format));
        CHECK(presentBlitDestination(format, D));
    }
    for (const VkFormat format :
         {VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT}) {
        CAPTURE(static_cast<int>(format));
        CHECK_FALSE(presentBlitDestination(format, D));
    }
    CHECK_FALSE(presentBlitDestination(VK_FORMAT_B8G8R8A8_UNORM, 0));
}
