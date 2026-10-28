#include "vk_result.h"

#include <format>
#include <string>

namespace dvv::gpu::detail {

Errc errcOf(vk::Result result) noexcept {
    switch (result) {
        case vk::Result::eErrorDeviceLost: return Errc::DeviceLost;
        case vk::Result::eErrorOutOfHostMemory:
        case vk::Result::eErrorOutOfDeviceMemory:
        case vk::Result::eErrorOutOfPoolMemory:
        case vk::Result::eErrorFragmentedPool:
        case vk::Result::eErrorMemoryMapFailed:
        case vk::Result::eErrorTooManyObjects:
        case vk::Result::eErrorFragmentation: return Errc::OutOfMemory;
        case vk::Result::eErrorInitializationFailed:
        case vk::Result::eErrorIncompatibleDriver:
        case vk::Result::eErrorLayerNotPresent:
        case vk::Result::eErrorExtensionNotPresent:
        case vk::Result::eErrorFeatureNotPresent:
        case vk::Result::eErrorFormatNotSupported:
        case vk::Result::eErrorNotPermitted:
        case vk::Result::eErrorImageUsageNotSupportedKHR: return Errc::Unsupported;
        default: return Errc::Internal;
    }
}

std::unexpected<Error> vkError(vk::Result result, std::string_view what,
                               std::source_location where) {
    return makeError(errcOf(result),
                     std::format("{} failed: {} (VkResult {})", what, vk::to_string(result),
                                 static_cast<int>(result)),
                     where);
}

std::unexpected<Error> vkError(VkResult result, std::string_view what, std::source_location where) {
    return vkError(static_cast<vk::Result>(result), what, where);
}

Result<void> vkCheck(vk::Result result, std::string_view what, std::source_location where) {
    if (result == vk::Result::eSuccess) {
        return {};
    }
    return vkError(result, what, where);
}

Result<void> vkCheck(VkResult result, std::string_view what, std::source_location where) {
    return vkCheck(static_cast<vk::Result>(result), what, where);
}

}  // namespace dvv::gpu::detail
