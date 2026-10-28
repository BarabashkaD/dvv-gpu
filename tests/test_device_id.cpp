#include <doctest/doctest.h>

#include <cstdint>
#include <format>
#include <functional>
#include <type_traits>
#include <unordered_set>

#include "dvv/gpu/device_id.h"

using dvv::gpu::DeviceId;

static_assert(!std::is_convertible_v<std::uint32_t, DeviceId>, "construction must be explicit");
static_assert(!std::is_convertible_v<DeviceId, std::uint32_t>);
static_assert(std::is_trivially_copyable_v<DeviceId>);
static_assert(DeviceId(2).index() == 2);
static_assert(DeviceId(1) == DeviceId(1));
static_assert(DeviceId(1) < DeviceId(2));
static_assert(dvv::gpu::FIRST_GENERATION == 1);

TEST_CASE("a DeviceId compares, hashes and formats as its index") {
    const DeviceId first(0);
    const DeviceId second(1);
    CHECK(first != second);
    CHECK(first < second);
    CHECK(std::format("{}", second) == "1");
    CHECK(std::format("[{:>3}]", second) == "[  1]");
    CHECK(std::hash<DeviceId>{}(second) == std::hash<std::uint32_t>{}(1));

    const std::unordered_set<DeviceId> ids{first, second, DeviceId(1)};
    CHECK(ids.size() == 2);
}
