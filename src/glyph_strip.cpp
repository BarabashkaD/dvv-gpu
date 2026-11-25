#include "dvv/gpu/glyph_strip.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <string_view>

#include "dvv/base/error.h"
#include "glyph_strip_os.h"

namespace dvv::gpu {

namespace detail {

CellLayout cellLayout(std::span<const double> advances) {
    CellLayout layout;
    for (const double glyph_advance : advances) {
        layout.advance = std::max(layout.advance, glyph_advance);
    }
    layout.advance = std::ceil(layout.advance);
    layout.glyphX.reserve(advances.size());
    for (std::size_t i = 0; i < advances.size(); ++i) {
        layout.glyphX.push_back((static_cast<double>(i) * layout.advance) +
                                ((layout.advance - advances[i]) / 2));
    }
    return layout;
}

Result<void> checkStripSize(double width, double height, double advance, std::string_view os_api) {
    if (advance == 0 || height == 0) {
        return makeError(
            Errc::Unsupported,
            std::format("glyph strip: {} gave an unusable strip ({} x {})", os_api, width, height));
    }
    if (width > MAX_GLYPH_STRIP_SIDE || height > MAX_GLYPH_STRIP_SIDE) {
        return makeError(
            Errc::Internal,
            std::format("glyph strip: the strip would be {} x {} pixels, a side over {}", width,
                        height, MAX_GLYPH_STRIP_SIDE));
    }
    return {};
}

}  // namespace detail

namespace {

constexpr std::size_t MAX_CHARS = 64;
// The largest font size in output pixels: keeps every strip dimension far inside 32 bits.
constexpr double MAX_PIXEL_SIZE = 4096;

bool positiveFinite(double value) {
    return std::isfinite(value) && value > 0;
}

// Printable ASCII: the space up to '~'.
bool printableAscii(char c) {
    return c >= ' ' && c <= '~';
}

}  // namespace

Result<GlyphStrip> rasterGlyphStrip(const FontSpec& font, double scale, std::string_view chars) {
    if (!positiveFinite(font.sizePoints)) {
        return makeError(
            Errc::Internal,
            std::format("glyph strip: the font size {} is not > 0 and finite", font.sizePoints));
    }
    if (!positiveFinite(scale)) {
        return makeError(Errc::Internal,
                         std::format("glyph strip: the scale {} is not > 0 and finite", scale));
    }
    if (chars.empty() || chars.size() > MAX_CHARS) {
        return makeError(Errc::Internal, std::format("glyph strip: {} characters, not 1 to {}",
                                                     chars.size(), MAX_CHARS));
    }
    if (!std::ranges::all_of(chars, printableAscii)) {
        return makeError(Errc::Internal, "glyph strip: the characters are not all printable ASCII");
    }
    const double pixel_size = font.sizePoints * scale;
    if (pixel_size > MAX_PIXEL_SIZE) {  // an infinite product too
        return makeError(Errc::Internal,
                         std::format("glyph strip: the size {} x the scale {} is over {} pixels",
                                     font.sizePoints, scale, MAX_PIXEL_SIZE));
    }
    return detail::rasterGlyphStripOs(font, scale, chars);
}

}  // namespace dvv::gpu
