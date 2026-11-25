#include "dvv/gpu/glyph_strip_vulkan.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <utility>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/glyph_strip.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/resources.h"

namespace dvv::gpu {

namespace {

constexpr std::size_t TEXEL_BYTES = 4;  // RGBA8

}  // namespace

Result<GlyphStripUpload> uploadGlyphStrip(ExecutionContext& context, const GlyphStrip& strip) {
    const std::size_t texels = std::size_t{strip.width} * strip.height;
    if (texels == 0 || strip.coverage.size() != texels) {
        return makeError(Errc::Internal,
                         std::format("glyph strip upload: a {} x {} strip with {} coverage bytes",
                                     strip.width, strip.height, strip.coverage.size()));
    }
    auto image = createImage(
        context, ImageDesc{.width = strip.width,
                           .height = strip.height,
                           .format = VK_FORMAT_R8G8B8A8_UNORM,
                           .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                           .name = "glyph strip"});
    if (!image) {
        return std::unexpected(std::move(image.error()));
    }
    auto staging = createBuffer(context, BufferDesc{.size = texels * TEXEL_BYTES,
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    .memory = MemoryUsage::HostUpload,
                                                    .name = "glyph strip staging"});
    if (!staging) {
        return std::unexpected(std::move(staging.error()));
    }
    {
        MappedWrite scope = (*staging)->mapWrite();
        const std::span<std::byte> bytes = scope.bytes();
        for (std::size_t i = 0; i < texels; ++i) {
            const auto c = static_cast<std::byte>(strip.coverage[i]);
            for (std::size_t channel = 0; channel < TEXEL_BYTES; ++channel) {
                bytes[(i * TEXEL_BYTES) + channel] = c;
            }
        }
        if (auto flushed = scope.flush(); !flushed) {
            return std::unexpected(std::move(flushed.error()));
        }
    }
    GlyphStripUpload upload;
    upload.image = *image;
    upload.record = [staging = std::move(*staging),
                     image = std::move(*image)](CommandRecorder& recorder) {
        recorder.copyBufferToImage(*staging, *image);
        recorder.transition(*image, VK_IMAGE_LAYOUT_GENERAL);
    };
    return upload;
}

}  // namespace dvv::gpu
