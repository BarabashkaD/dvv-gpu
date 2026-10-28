#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>

namespace dvv::gpu {

// A physical device: its index in the GpuRegistry's enumeration (spec §5.2). A strong type, so it
// cannot be mixed up with other integers.
class DeviceId {
public:
    constexpr explicit DeviceId(std::uint32_t index) noexcept : index_(index) {}

    [[nodiscard]] constexpr std::uint32_t index() const noexcept { return index_; }

    friend constexpr bool operator==(DeviceId, DeviceId) noexcept = default;
    friend constexpr std::strong_ordering operator<=>(DeviceId, DeviceId) noexcept = default;

private:
    std::uint32_t index_;
};

// Device generation (spec §5.2): 1 for the first device of a context; a device-lost recovery
// recreates the device with the next generation. GPU resources record their generation.
using DeviceGeneration = std::uint32_t;
inline constexpr DeviceGeneration FIRST_GENERATION = 1;

}  // namespace dvv::gpu

template <>
struct std::hash<dvv::gpu::DeviceId> {
    std::size_t operator()(dvv::gpu::DeviceId id) const noexcept {
        return std::hash<std::uint32_t>{}(id.index());
    }
};

// Formats as the index, e.g. "0".
template <>
struct std::formatter<dvv::gpu::DeviceId> : std::formatter<std::uint32_t> {
    auto format(dvv::gpu::DeviceId id, std::format_context& context) const {
        return std::formatter<std::uint32_t>::format(id.index(), context);
    }
};
