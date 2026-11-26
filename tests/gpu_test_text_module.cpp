// The glyph text shader module (shaders/gpu_text.slang), imported by the test probe as
// dvv_gpu_text.gpu_text and compared with the strip's bytes.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/glyph_strip.h"
#include "dvv/gpu/glyph_strip_vulkan.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "gpu_fixture.h"
#include "test_shaders/text_probe.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::ComputeBindings;
using dvv::gpu::GlyphStrip;
using dvv::gpu::GpuImage;
using dvv::gpu::ImageDesc;
using dvv::gpu::MappedRead;
using dvv::gpu::MemoryUsage;
using dvv::gpu::rasterGlyphStrip;
using dvv::gpu::RecordFunction;
using dvv::gpu::test::GpuTest;

namespace {

constexpr std::uint32_t GROUP_SIZE = 8;  // [numthreads(8, 8, 1)] in text_probe.slang
constexpr std::uint32_t MODE_DECIMAL = 0;
constexpr std::uint32_t MODE_CELL = 1;
constexpr std::uint32_t MODE_DIGITS = 2;

// text_probe's push constants.
struct ProbePush {
    std::uint32_t mode = 0;
    std::uint32_t value = 0;
    std::uint32_t cellCount = 0;
    std::int32_t base = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
static_assert(sizeof(ProbePush) == 24);

// Runs the probe over a width x height output with `strip` bound and returns the output's values,
// row-major.
std::vector<std::uint32_t> runProbe(GpuTest& test, const GlyphStrip& strip, const ProbePush& push) {
    const auto pipeline = createComputePipeline(test.context(), test_shaders::textProbe(),
                                                sizeof(ProbePush), ComputeBindings{2})
                              .value();
    const auto output =
        createImage(test.context(),
                    ImageDesc{.width = push.width,
                              .height = push.height,
                              .format = VK_FORMAT_R32_UINT,
                              .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              .name = "text probe output"})
            .value();
    const std::size_t bytes = std::size_t{push.width} * push.height * sizeof(std::uint32_t);
    const auto texels =
        createBuffer(test.context(), BufferDesc{.size = bytes,
                                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                .memory = MemoryUsage::HostReadback,
                                                .name = "text probe texels"})
            .value();
    auto upload = uploadGlyphStrip(test.context(), strip);
    REQUIRE_MESSAGE(upload.has_value(), (upload ? "" : upload.error().describe()));
    const std::shared_ptr<GpuImage> glyphs = upload->image;
    RecordFunction record = std::move(upload->record);
    const std::array<GpuImage*, 2> images{output.get(), glyphs.get()};
    REQUIRE(test.submitAndWait([&](CommandRecorder& recorder) {
                    record(recorder);
                    recorder.dispatch(*pipeline, std::as_bytes(std::span(&push, 1)), images,
                                      {(push.width + GROUP_SIZE - 1) / GROUP_SIZE,
                                       (push.height + GROUP_SIZE - 1) / GROUP_SIZE});
                    recorder.copyImageToBuffer(*output, *texels);
                })
                .has_value());
    const MappedRead scope = texels->mapRead().value();
    std::vector<std::uint32_t> values(std::size_t{push.width} * push.height);
    std::memcpy(values.data(), scope.bytes().data(), bytes);
    return values;
}

std::uint32_t decimalDigitCount(std::uint32_t value) {
    return static_cast<std::uint32_t>(std::to_string(value).size());
}

// The strip's byte at (x, y).
std::uint32_t stripAt(const GlyphStrip& strip, std::size_t x, std::uint32_t y) {
    return strip.coverage[(std::size_t{y} * strip.width) + x];
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "text module: decimal coverage equals the strip's cells") {
    constexpr std::uint32_t CELLS = 10;
    constexpr std::int32_t BASE = 3;  // the text starts at output column BASE
    const auto strip = rasterGlyphStrip({"", 12}, 2.0, "0123456789");
    REQUIRE(strip.has_value());
    // An all-zero strip would make the shader and the reference agree on nothing.
    REQUIRE(std::ranges::any_of(strip->coverage, [](auto byte) { return byte > 0; }));
    const std::uint32_t advance = strip->width / CELLS;
    REQUIRE(advance > 0);
    for (const std::uint32_t value : {0U, 7U, 42U, 1023U, 65535U, 4294967295U}) {
        INFO("value " << value);
        const std::uint32_t length = decimalDigitCount(value) * advance;
        // Columns before the text and past it, rows past the glyph height.
        const ProbePush push{.mode = MODE_DECIMAL,
                             .value = value,
                             .cellCount = CELLS,
                             .base = BASE,
                             .width = BASE + length + 5,
                             .height = strip->height + 2};
        const std::vector<std::uint32_t> got = runProbe(*this, *strip, push);
        const std::string text = std::to_string(value);
        std::size_t mismatches = 0;
        for (std::uint32_t y = 0; y < push.height; ++y) {
            for (std::uint32_t x = 0; x < push.width; ++x) {
                const std::int64_t offset = std::int64_t{x} - BASE;
                std::uint32_t wanted = 0;
                if (offset >= 0 && offset < length && y < strip->height) {
                    const auto digit = static_cast<std::uint32_t>(
                        text[static_cast<std::size_t>(offset) / advance] - '0');
                    wanted =
                        stripAt(*strip, (std::size_t{digit} * advance) + (offset % advance), y);
                }
                mismatches += got[(std::size_t{y} * push.width) + x] != wanted ? 1 : 0;
            }
        }
        CHECK(mismatches == 0);
    }
}

TEST_CASE_FIXTURE(GpuTest, "text module: the cell count comes from the caller") {
    constexpr std::uint32_t CELLS = 17;
    constexpr std::uint32_t CELL = 10;  // the 'Y'
    constexpr std::int32_t BASE = 2;
    const auto strip = rasterGlyphStrip({"", 12}, 2.0, "0123456789YUVRGBA");
    REQUIRE(strip.has_value());
    // An all-zero strip would make the shader and the reference agree on nothing.
    REQUIRE(std::ranges::any_of(strip->coverage, [](auto byte) { return byte > 0; }));
    const std::uint32_t advance = strip->width / CELLS;
    REQUIRE(advance > 0);

    const ProbePush cell{.mode = MODE_CELL,
                         .value = CELL,
                         .cellCount = CELLS,
                         .base = BASE,
                         .width = advance + 4,
                         .height = strip->height + 1};
    const std::vector<std::uint32_t> got = runProbe(*this, *strip, cell);
    std::size_t mismatches = 0;
    for (std::uint32_t y = 0; y < cell.height; ++y) {
        for (std::uint32_t x = 0; x < cell.width; ++x) {
            const std::int64_t gx = std::int64_t{x} - BASE;
            const std::uint32_t wanted =
                gx >= 0 && gx < advance && y < strip->height
                    ? stripAt(*strip, (std::size_t{CELL} * advance) + static_cast<std::size_t>(gx),
                              y)
                    : 0;
            mismatches += got[(std::size_t{y} * cell.width) + x] != wanted ? 1 : 0;
        }
    }
    CHECK(mismatches == 0);

    // The grid, the digit count and the digits: without a cell count there is no grid.
    for (const std::uint32_t cells : {CELLS, 0U}) {
        const ProbePush digits{
            .mode = MODE_DIGITS, .value = 90210, .cellCount = cells, .width = 6, .height = 4};
        const std::vector<std::uint32_t> rows = runProbe(*this, *strip, digits);
        CHECK(rows[0] == 5);  // decimalDigits
        const std::array<std::uint32_t, 6> wanted_digits{9, 0, 2, 1, 0, 0};
        for (std::uint32_t x = 0; x < 6; ++x) {
            CHECK(rows[digits.width + x] == wanted_digits[x]);
        }
        CHECK(rows[std::size_t{2} * digits.width] ==
              (cells == 0 ? 0 : advance));  // glyphCells().advance
        CHECK(rows[std::size_t{3} * digits.width] == (cells == 0 ? 0 : strip->height));
    }
}
