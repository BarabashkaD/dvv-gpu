#pragma once

// A shader compiled at build time by dvv_add_shaders() (spec §5.4, §13.2).

#include <cstdint>
#include <span>
#include <string_view>

namespace dvv::gpu {

struct EmbeddedShader {
    std::string_view name{};                 // the .slang file name without the extension
    std::span<const std::uint32_t> spirv{};  // SPIR-V 1.6, one compute entry point named main
    std::string_view reflectionJson{};       // slangc -reflection-json output
};

}  // namespace dvv::gpu
