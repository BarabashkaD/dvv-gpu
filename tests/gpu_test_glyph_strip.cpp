// Glyph strips uploaded into a storage image (dvv/gpu/glyph_strip_vulkan.h).
#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/glyph_strip.h"
#include "dvv/gpu/glyph_strip_vulkan.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "gpu_fixture.h"

using dvv::Errc;
using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::GlyphStrip;
using dvv::gpu::GlyphStripUpload;
using dvv::gpu::MappedRead;
using dvv::gpu::MemoryUsage;
using dvv::gpu::RecordFunction;
using dvv::gpu::test::GpuTest;

namespace {

constexpr std::size_t TEXEL_BYTES = 4;  // RGBA8

// Uploads `strip`, reads the image back in the same submission and returns the texels. The upload
// is dropped before the submission runs: its record function alone must keep the staging buffer.
std::vector<std::uint8_t> uploadAndRead(GpuTest& test, const GlyphStrip& strip) {
    auto upload = uploadGlyphStrip(test.context(), strip);
    REQUIRE_MESSAGE(upload.has_value(), (upload ? "" : upload.error().describe()));
    const std::shared_ptr<dvv::gpu::GpuImage> image = upload->image;
    REQUIRE(image != nullptr);
    const auto& desc = vulkanDesc(*image);
    CHECK(desc.width == strip.width);
    CHECK(desc.height == strip.height);
    CHECK(desc.format == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK((desc.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0);
    CHECK(desc.name == "glyph strip");

    const std::size_t size = std::size_t{strip.width} * strip.height * TEXEL_BYTES;
    const auto texels =
        createBuffer(test.context(), BufferDesc{.size = size,
                                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                .memory = MemoryUsage::HostReadback,
                                                .name = "glyph strip texels"})
            .value();
    RecordFunction record = std::move(upload->record);
    {
        const GlyphStripUpload dropped = std::move(*upload);
    }
    REQUIRE(test.submitAndWait(
                    [record = std::move(record), image, texels](CommandRecorder& recorder) mutable {
                        record(recorder);
                        recorder.copyImageToBuffer(*image, *texels);
                    })
                .has_value());
    CHECK(vulkanLayout(*image) == VK_IMAGE_LAYOUT_GENERAL);

    const MappedRead scope = texels->mapRead().value();
    std::vector<std::uint8_t> values(size);
    std::memcpy(values.data(), scope.bytes().data(), size);
    return values;
}

// The number of texels that are not {c, c, c, c} for the strip's coverage c.
std::size_t mismatches(const GlyphStrip& strip, const std::vector<std::uint8_t>& texels) {
    REQUIRE(texels.size() == strip.coverage.size() * TEXEL_BYTES);
    std::size_t count = 0;
    for (std::size_t i = 0; i < strip.coverage.size(); ++i) {
        const std::uint8_t c = strip.coverage[i];
        const std::array<std::uint8_t, TEXEL_BYTES> wanted{c, c, c, c};
        count += std::memcmp(&texels[i * TEXEL_BYTES], wanted.data(), TEXEL_BYTES) != 0 ? 1 : 0;
    }
    return count;
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "glyph strip upload: read back equal") {
    GlyphStrip strip;
    strip.width = 20;
    strip.height = 7;
    strip.advance = 2;
    strip.ascent = 5;
    strip.descent = 2;
    strip.coverage.resize(std::size_t{strip.width} * strip.height);
    for (std::uint32_t y = 0; y < strip.height; ++y) {
        for (std::uint32_t x = 0; x < strip.width; ++x) {
            strip.coverage[(std::size_t{y} * strip.width) + x] =
                static_cast<std::uint8_t>(((x * 13) + (y * 7)) & 0xffU);
        }
    }
    CHECK(mismatches(strip, uploadAndRead(*this, strip)) == 0);
}

TEST_CASE_FIXTURE(GpuTest, "glyph strip upload: empty strip") {
    const auto upload = uploadGlyphStrip(context(), GlyphStrip{});
    REQUIRE_FALSE(upload.has_value());
    CHECK(upload.error().code == Errc::Internal);
    // Sizes that do not match the coverage are refused the same way.
    GlyphStrip strip;
    strip.width = 4;
    strip.height = 2;
    strip.coverage.resize(7);
    const auto mismatched = uploadGlyphStrip(context(), strip);
    REQUIRE_FALSE(mismatched.has_value());
    CHECK(mismatched.error().code == Errc::Internal);
}

TEST_CASE_FIXTURE(GpuTest, "glyph strip upload: a real strip") {
    const auto strip =
        dvv::gpu::rasterGlyphStrip({.family = "", .sizePoints = 11}, 2, "0123456789");
    REQUIRE_MESSAGE(strip.has_value(), (strip ? "" : strip.error().describe()));
    CHECK(mismatches(*strip, uploadAndRead(*this, *strip)) == 0);
}
