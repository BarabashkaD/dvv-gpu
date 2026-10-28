#include <doctest/doctest.h>

#include <format>
#include <string>

#include "dvv/base/error.h"
#include "dvv/platform/dynamic_library.h"
#include "loader.h"

#if defined(_WIN32)
#include <filesystem>

#include <windows.h>
#endif

using dvv::gpu::detail::loaderLibrary;
using dvv::gpu::detail::loadVulkanLoader;
using dvv::platform::LibrarySearch;

TEST_CASE("a missing Vulkan loader is reported as unsupported") {
    const auto loader = loadVulkanLoader("dvv-test-no-such-vulkan-loader");
    REQUIRE_FALSE(loader.has_value());
    CHECK(loader.error().code == dvv::Errc::Unsupported);
    CHECK(loader.error().message == "Vulkan loader not found ('dvv-test-no-such-vulkan-loader')");
}

TEST_CASE("the default loader is the platform's library from the system directory") {
    const auto library = loaderLibrary("");
#if defined(_WIN32)
    CHECK(library.name == "vulkan-1.dll");
#else
    CHECK(library.name == "libvulkan.so.1");
#endif
    CHECK(library.search == LibrarySearch::SystemDirectory);
}

TEST_CASE("a named loader is searched the usual way") {
    const auto library = loaderLibrary("x/libvulkan-test.so");
    CHECK(library.name == "x/libvulkan-test.so");
    CHECK(library.search == LibrarySearch::Default);
}

TEST_CASE("a library without vkGetInstanceProcAddr is not a Vulkan loader") {
#if defined(_WIN32)
    const std::string name = "kernel32.dll";
#else
    const std::string name = "libc.so.6";
#endif
    const auto loader = loadVulkanLoader(name);
    REQUIRE_FALSE(loader.has_value());
    CHECK(loader.error().code == dvv::Errc::Unsupported);
    CHECK(loader.error().message ==
          std::format("Vulkan loader not found ('{}' has no vkGetInstanceProcAddr)", name));
}

#if defined(_WIN32)
TEST_CASE("the default loader comes from the system directory") {
    const auto loader = dvv::gpu::detail::vulkanLoader();
    if (!loader && loader.error().code == dvv::Errc::Unsupported) {
        MESSAGE("no Vulkan loader");
        return;
    }
    REQUIRE(loader.has_value());
    HMODULE module = nullptr;
    REQUIRE(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(*loader), &module) != 0);
    std::wstring module_path(32768, L'\0');
    const DWORD length =
        GetModuleFileNameW(module, module_path.data(), static_cast<DWORD>(module_path.size()));
    REQUIRE(length > 0);
    REQUIRE(length < module_path.size());
    module_path.resize(length);
    std::wstring system_directory(32768, L'\0');
    const UINT system_length =
        GetSystemDirectoryW(system_directory.data(), static_cast<UINT>(system_directory.size()));
    REQUIRE(system_length > 0);
    REQUIRE(system_length < system_directory.size());
    system_directory.resize(system_length);
    const std::filesystem::path folder = std::filesystem::path(module_path).parent_path();
    INFO(folder.string());
    CHECK(_wcsicmp(folder.native().c_str(),
                   std::filesystem::path(system_directory).native().c_str()) == 0);
}
#endif
