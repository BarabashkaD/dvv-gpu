#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "dvv/base/error.h"

namespace dvv::gpu {

// A font as the OS names it.
struct FontSpec {
    // UTF-8; "" = the OS's UI font, resolved per OS. On Windows a GDI face name, as font lists
    // show it, e.g. "Arial Narrow" or "Segoe UI Semibold".
    std::string family;
    double sizePoints = 0;  // the OS's font points, > 0

    bool operator==(const FontSpec&) const = default;
};

// A row of equal cells, one character each, rasterised with an OS font: white coverage on black.
// Every cell is `advance` pixels wide; each glyph's origin is centred in its cell (glyphs narrower
// than the widest one get equal space on both sides of their advance).
struct GlyphStrip {
    std::vector<std::uint8_t> coverage;  // width x height, row-major, 0..255
    std::uint32_t width = 0;             // the number of characters x advance
    std::uint32_t height = 0;            // ascent + descent
    std::uint32_t advance = 0;           // the cell width: the widest glyph's advance, rounded up
    std::uint32_t ascent = 0;            // whole pixels, rounded up: rows 0 to ascent - 1 lie above
                                         // the baseline, the top edge of row `ascent`
    std::uint32_t descent = 0;           // whole pixels, rounded up
    // The family the OS drew with. It can differ from the name asked for without a fallback: on
    // Windows "Segoe UI Semibold" draws with the family "Segoe UI".
    std::string family;

    bool operator==(const GlyphStrip&) const = default;
};

// The largest strip width and height in pixels, on every OS: a strip that would be wider or
// taller is refused before it is drawn. It fits every OS raster and the 2D images of current
// desktop GPUs.
inline constexpr std::uint32_t MAX_GLYPH_STRIP_SIDE = 16384;

// Rasterises `chars` with `font`, one cell per character, in order. `scale`: output pixels per
// font point (the caller's points-to-pixels rule): the font is the OS's font at `sizePoints` (its
// design for that size), drawn `scale` times larger. `chars`: 1 to 64 printable ASCII characters.
// Errc::Internal for a size or scale that is not > 0 and finite, a size x scale above 4096 pixels,
// bad chars, or a strip wider or taller than MAX_GLYPH_STRIP_SIDE (its width is the number of
// characters x the widest advance); Errc::Unsupported when the OS text API fails. A family the OS
// does not know falls back to an OS font. Safe to call from any thread, also at the same time:
// the calls share no state but the OS text library's, which is thread-safe.
[[nodiscard]] Result<GlyphStrip> rasterGlyphStrip(const FontSpec& font, double scale,
                                                  std::string_view chars);

}  // namespace dvv::gpu
