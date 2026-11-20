#include "bundled_drivers.h"

#include <cassert>
#include <format>
#include <system_error>

#include "dvv/base/environment.h"
#include "dvv/base/utf8.h"

namespace dvv::gpu {

namespace detail {

namespace {

// The OS list separator: the one OS whose path separator is '\' (Windows) is the one whose list
// separator is ';'.
constexpr char LIST_SEPARATOR = std::filesystem::path::preferred_separator == '\\' ? ';' : ':';

// The loader treats an empty value as unset, and so does this.
bool isSet(const std::function<std::optional<std::string>(std::string_view)>& get_env,
           std::string_view name) {
    const std::optional<std::string> value = get_env(name);
    return value.has_value() && !value->empty();
}

}  // namespace

Result<DriverSetting> bundledDriverSetting(
    std::span<const std::filesystem::path> manifests,
    const std::function<std::optional<std::string>(std::string_view)>& get_env) {
    if (isSet(get_env, "VK_DRIVER_FILES") || isSet(get_env, "VK_ICD_FILENAMES")) {
        return DriverSetting{.choice = DriverChoice::UserChoice};
    }
    DriverSetting setting;
    for (const std::filesystem::path& manifest : manifests) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(manifest, error)) {
            continue;
        }
        const std::string text = pathText(manifest);
        // The loader splits the value at every separator, and a separator cannot be escaped.
        if (text.find(LIST_SEPARATOR) != std::string::npos) {
            return makeError(Errc::Unsupported,
                             std::format("the driver manifest '{}' contains the list separator "
                                         "'{}', so VK_DRIVER_FILES cannot name it",
                                         pathDisplayText(manifest), LIST_SEPARATOR));
        }
        if (setting.choice == DriverChoice::Bundled) {
            setting.value += LIST_SEPARATOR;
        }
        setting.value += text;
        setting.choice = DriverChoice::Bundled;
    }
    return setting;
}

}  // namespace detail

Result<DriverChoice> useBundledDrivers(std::span<const std::filesystem::path> manifests) {
    DVV_TRY(const detail::DriverSetting setting,
            detail::bundledDriverSetting(manifests, getEnvironment));
    if (setting.choice == DriverChoice::Bundled) {
        // setEnvironment would remove the variable for an empty value; a Bundled value names at
        // least one manifest, whose path text is not empty.
        assert(!setting.value.empty());
        if (!setEnvironment("VK_DRIVER_FILES", setting.value)) {
            return makeError(Errc::Internal, "cannot set VK_DRIVER_FILES");
        }
    }
    return setting.choice;
}

}  // namespace dvv::gpu
