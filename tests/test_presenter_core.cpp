#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include <vulkan/vulkan_core.h>

#include "dvv/platform/native_window.h"
#include "presenter_core.h"

using dvv::gpu::detail::chooseSurfaceFormat;
using dvv::gpu::detail::compositeAlpha;
using dvv::gpu::detail::swapchainExtent;
using dvv::gpu::detail::swapchainImageCount;
using dvv::platform::PixelSize;

namespace {

// value_or() for a missing format; REQUIRE has checked that one is there.
constexpr VkSurfaceFormatKHR NO_FORMAT{VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_MAX_ENUM_KHR};

VkSurfaceCapabilitiesKHR imageCounts(std::uint32_t min, std::uint32_t max) {
    VkSurfaceCapabilitiesKHR caps{};
    caps.minImageCount = min;
    caps.maxImageCount = max;
    return caps;
}

}  // namespace

TEST_CASE("chooseSurfaceFormat prefers an 8-bit UNORM format with sRGB non-linear colour") {
    const std::array srgb_first{
        VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
        VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
    const auto chosen = chooseSurfaceFormat(srgb_first);
    REQUIRE(chosen.has_value());
    CHECK(chosen.value_or(NO_FORMAT).format == VK_FORMAT_B8G8R8A8_UNORM);

    const std::array rgba{
        VkSurfaceFormatKHR{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
    const auto chosen_rgba = chooseSurfaceFormat(rgba);
    REQUIRE(chosen_rgba.has_value());
    CHECK(chosen_rgba.value_or(NO_FORMAT).format == VK_FORMAT_R8G8B8A8_UNORM);

    const std::array none_fits{
        VkSurfaceFormatKHR{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
        VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT}};
    const auto chosen_first = chooseSurfaceFormat(none_fits);
    REQUIRE(chosen_first.has_value());
    CHECK(chosen_first.value_or(NO_FORMAT).format == VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    CHECK(chosen_first.value_or(NO_FORMAT).colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);

    CHECK_FALSE(chooseSurfaceFormat(std::span<const VkSurfaceFormatKHR>{}).has_value());
}

TEST_CASE(
    "swapchainImageCount asks for one more than the minimum, at least two, within the "
    "maximum") {
    CHECK(swapchainImageCount(imageCounts(2, 0)) == 3);
    CHECK(swapchainImageCount(imageCounts(1, 0)) == 2);
    CHECK(swapchainImageCount(imageCounts(2, 2)) == 2);
    CHECK(swapchainImageCount(imageCounts(3, 8)) == 4);
}

TEST_CASE(
    "swapchainExtent takes the surface's extent and clamps the hint only when the surface "
    "has none") {
    VkSurfaceCapabilitiesKHR caps{};
    caps.currentExtent = {.width = 800, .height = 600};
    caps.minImageExtent = {.width = 1, .height = 1};
    caps.maxImageExtent = {.width = 4096, .height = 4096};
    VkExtent2D extent = swapchainExtent(caps, PixelSize{.width = 1, .height = 1});
    CHECK(extent.width == 800);
    CHECK(extent.height == 600);

    caps.currentExtent = {.width = 0xFFFFFFFFU, .height = 0xFFFFFFFFU};
    extent = swapchainExtent(caps, PixelSize{.width = 5000, .height = 300});
    CHECK(extent.width == 4096);
    CHECK(extent.height == 300);

    caps.currentExtent = {.width = 0, .height = 0};
    caps.minImageExtent = {.width = 0, .height = 0};
    caps.maxImageExtent = {.width = 0, .height = 0};
    extent = swapchainExtent(caps, PixelSize{.width = 320, .height = 240});
    CHECK(extent.width == 0);
    CHECK(extent.height == 0);
}

TEST_CASE("compositeAlpha prefers opaque, then inherit") {
    CHECK(compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR | VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
    CHECK(compositeAlpha(VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR |
                         VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
    CHECK(compositeAlpha(VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);
}
