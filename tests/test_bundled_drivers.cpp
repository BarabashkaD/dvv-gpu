#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "bundled_drivers.h"
#include "dvv/base/environment.h"
#include "dvv/base/error.h"
#include "dvv/base/utf8.h"
#include "dvv/gpu/bundled_drivers.h"

using dvv::gpu::DriverChoice;
using dvv::gpu::useBundledDrivers;
using dvv::gpu::detail::bundledDriverSetting;
using dvv::gpu::detail::DriverSetting;

namespace {

// The OS list separator: ';' where the path separator is '\' (Windows), ':' elsewhere.
constexpr char SEP = std::filesystem::path::preferred_separator == '\\' ? ';' : ':';

// macOS refuses no value (APFS names are UTF-8 and setenv has no length limit), so the refusal
// case is reported as skipped there; the Windows and Linux runs cover that branch.
#if defined(_WIN32) || defined(__linux__)
constexpr bool REFUSAL_SKIPPED = false;
#else
constexpr bool REFUSAL_SKIPPED = true;
#endif

// A new empty folder, removed with its contents at the end of the scope.
class TemporaryFolder {
public:
    TemporaryFolder() {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                std::format("dvv-gpu-drivers-{}-{}",
                            std::chrono::steady_clock::now().time_since_epoch().count(),
                            counter.fetch_add(1));
        std::filesystem::create_directories(path_);
    }
    TemporaryFolder(const TemporaryFolder&) = delete;
    TemporaryFolder& operator=(const TemporaryFolder&) = delete;
    TemporaryFolder(TemporaryFolder&&) = delete;
    TemporaryFolder& operator=(TemporaryFolder&&) = delete;
    ~TemporaryFolder() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Creates the file `path` holding an empty JSON object, and returns `path`.
std::filesystem::path writeManifest(const std::filesystem::path& path) {
    std::ofstream(path) << "{}";
    REQUIRE(std::filesystem::is_regular_file(path));
    return path;
}

// A get_env that reads `variables` only.
auto environmentOf(std::map<std::string, std::string> variables) {
    return [variables = std::move(variables)](std::string_view name) -> std::optional<std::string> {
        const auto found = variables.find(std::string(name));
        if (found == variables.end()) {
            return std::nullopt;
        }
        return found->second;
    };
}

// Saves one environment variable and restores it at the end of the scope. A variable that was set
// to "" comes back unset: dvv-base sets no empty value, and the loader treats both the same.
class RestoredVariable {
public:
    explicit RestoredVariable(std::string_view name)
        : name_(name), saved_(dvv::getEnvironment(name)) {}
    RestoredVariable(const RestoredVariable&) = delete;
    RestoredVariable& operator=(const RestoredVariable&) = delete;
    RestoredVariable(RestoredVariable&&) = delete;
    RestoredVariable& operator=(RestoredVariable&&) = delete;
    ~RestoredVariable() { static_cast<void>(dvv::setEnvironment(name_, saved_.value_or(""))); }

private:
    std::string_view name_;
    std::optional<std::string> saved_;
};

// Removes VK_DRIVER_FILES and VK_ICD_FILENAMES and restores them at the end of the scope, so that
// the end-to-end cases start from an environment with no user choice.
class ClearedDriverVariables {
public:
    ClearedDriverVariables() {
        REQUIRE(dvv::setEnvironment("VK_DRIVER_FILES", ""));
        REQUIRE(dvv::setEnvironment("VK_ICD_FILENAMES", ""));
    }
    ClearedDriverVariables(const ClearedDriverVariables&) = delete;
    ClearedDriverVariables& operator=(const ClearedDriverVariables&) = delete;
    ClearedDriverVariables(ClearedDriverVariables&&) = delete;
    ClearedDriverVariables& operator=(ClearedDriverVariables&&) = delete;
    ~ClearedDriverVariables() = default;

private:
    // Both are saved before the constructor's body clears them, so they are restored even when a
    // REQUIRE there fails.
    RestoredVariable driver_files_{"VK_DRIVER_FILES"};
    RestoredVariable icd_filenames_{"VK_ICD_FILENAMES"};
};

}  // namespace

TEST_CASE("the existing manifests are joined with the OS list separator") {
    const TemporaryFolder folder;
    const std::array manifests{writeManifest(folder.path() / "a.json"),
                               writeManifest(folder.path() / "b.json")};
    const DriverSetting setting = bundledDriverSetting(manifests, environmentOf({}));
    CHECK(setting.choice == DriverChoice::Bundled);
    CHECK(setting.value == dvv::pathText(manifests[0]) + SEP + dvv::pathText(manifests[1]));
}

TEST_CASE("VK_DRIVER_FILES and VK_ICD_FILENAMES win") {
    const TemporaryFolder folder;
    const std::array manifests{writeManifest(folder.path() / "a.json")};
    for (const char* name : {"VK_DRIVER_FILES", "VK_ICD_FILENAMES"}) {
        CAPTURE(name);
        const DriverSetting setting = bundledDriverSetting(manifests, environmentOf({{name, "x"}}));
        CHECK(setting.choice == DriverChoice::UserChoice);
        CHECK(setting.value.empty());
    }
}

TEST_CASE("empty variables count as unset") {
    const TemporaryFolder folder;
    const std::array manifests{writeManifest(folder.path() / "a.json")};
    const DriverSetting setting = bundledDriverSetting(
        manifests, environmentOf({{"VK_DRIVER_FILES", ""}, {"VK_ICD_FILENAMES", ""}}));
    CHECK(setting.choice == DriverChoice::Bundled);
    CHECK(setting.value == dvv::pathText(manifests[0]));
}

TEST_CASE("missing manifests are skipped; none left gives NoManifest") {
    const TemporaryFolder folder;
    const std::filesystem::path present = writeManifest(folder.path() / "a.json");
    const std::array some{folder.path() / "missing.json", present,
                          folder.path() / "also-missing.json"};
    const DriverSetting found = bundledDriverSetting(some, environmentOf({}));
    CHECK(found.choice == DriverChoice::Bundled);
    CHECK(found.value == dvv::pathText(present));

    const std::array none{folder.path() / "missing.json"};
    const DriverSetting missing = bundledDriverSetting(none, environmentOf({}));
    CHECK(missing.choice == DriverChoice::NoManifest);
    CHECK(missing.value.empty());
}

TEST_CASE("a manifest that is a folder is skipped; an empty list gives NoManifest") {
    const TemporaryFolder folder;
    const std::filesystem::path not_a_file = folder.path() / "c.json";
    std::filesystem::create_directories(not_a_file);
    const std::array manifests{not_a_file, writeManifest(folder.path() / "a.json")};
    const DriverSetting setting = bundledDriverSetting(manifests, environmentOf({}));
    CHECK(setting.choice == DriverChoice::Bundled);
    CHECK(setting.value == dvv::pathText(manifests[1]));

    const DriverSetting empty = bundledDriverSetting({}, environmentOf({}));
    CHECK(empty.choice == DriverChoice::NoManifest);
    CHECK(empty.value.empty());
}

TEST_CASE("a non-ASCII path keeps its path text") {
    const TemporaryFolder folder;
    const std::filesystem::path bundle = folder.path() / std::filesystem::path(u8"é 日本");
    std::filesystem::create_directories(bundle);
    const std::array manifests{writeManifest(bundle / "icd.json")};
    const DriverSetting setting = bundledDriverSetting(manifests, environmentOf({}));
    CHECK(setting.choice == DriverChoice::Bundled);
    CHECK(setting.value == dvv::pathText(manifests[0]));
}

TEST_CASE("useBundledDrivers sets VK_DRIVER_FILES") {
    const TemporaryFolder folder;
    const std::array manifests{writeManifest(folder.path() / "a.json")};
    const ClearedDriverVariables cleared;
    const dvv::Result<DriverChoice> choice = useBundledDrivers(manifests);
    CHECK(choice == DriverChoice::Bundled);
    CHECK(dvv::getEnvironment("VK_DRIVER_FILES") == dvv::pathText(manifests[0]));
}

TEST_CASE("a value the environment refuses gives Errc::Internal" * doctest::skip(REFUSAL_SKIPPED)) {
#if defined(_WIN32)
    // One manifest, repeated until the joined value is 32767 UTF-16 units or longer.
    const TemporaryFolder folder;
    const std::filesystem::path manifest = writeManifest(folder.path() / "a.json");
    const std::vector<std::filesystem::path> manifests((32767 / (manifest.native().size() + 1)) + 1,
                                                       manifest);
#elif defined(__linux__)
    // A folder name that is not valid UTF-8.
    const TemporaryFolder folder;
    const std::string not_utf8{
        "bad\xFF"
        "dir"};
    const std::filesystem::path bad_folder = folder.path() / not_utf8;
    std::filesystem::create_directories(bad_folder);
    const std::vector<std::filesystem::path> manifests{writeManifest(bad_folder / "a.json")};
#endif
#if defined(_WIN32) || defined(__linux__)
    const ClearedDriverVariables cleared;
    const dvv::Result<DriverChoice> choice = useBundledDrivers(manifests);
    REQUIRE_FALSE(choice.has_value());
    if (!choice) {
        CHECK(choice.error().code == dvv::Errc::Internal);
        CHECK(choice.error().message == "cannot set VK_DRIVER_FILES");
    }
    CHECK(dvv::getEnvironment("VK_DRIVER_FILES") == std::nullopt);
#endif
}
