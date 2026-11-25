#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/gpu/glyph_strip.h"

namespace dvv::gpu::detail {

// Where the glyphs go in a strip of equal cells.
struct CellLayout {
    double advance = 0;          // the cell width: the widest advance, rounded up
    std::vector<double> glyphX;  // each glyph's origin: its cell's start + the centring space
};

// The cells for glyphs with these advances (pixels): glyphX[i] = i x advance +
// (advance - advances[i]) / 2. No OS code; every OS raster places its glyphs with it.
[[nodiscard]] CellLayout cellLayout(std::span<const double> advances);

// Every OS raster checks its measured strip with this before any narrowing and before it draws:
// Errc::Unsupported when the OS gave a zero advance or height ("glyph strip: <os_api> gave an
// unusable strip"), Errc::Internal when a side is over MAX_GLYPH_STRIP_SIDE.
[[nodiscard]] Result<void> checkStripSize(double width, double height, double advance,
                                          std::string_view os_api);

// The OS raster behind rasterGlyphStrip, after its argument checks: the font is created at
// `font.sizePoints` (> 0, finite) and drawn `scale` times larger (> 0, finite; sizePoints x scale
// at most 4096 pixels), `chars` 1 to 64 printable ASCII characters.
[[nodiscard]] Result<GlyphStrip> rasterGlyphStripOs(const FontSpec& font, double scale,
                                                    std::string_view chars);

}  // namespace dvv::gpu::detail
