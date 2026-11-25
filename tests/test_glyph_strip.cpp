#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#include <filesystem>
#include <system_error>

#include <windows.h>
#endif

#include "dvv/base/error.h"
#include "dvv/gpu/glyph_strip.h"
#include "glyph_strip_os.h"

namespace {

using dvv::Errc;
using dvv::gpu::FontSpec;
using dvv::gpu::GlyphStrip;
using dvv::gpu::rasterGlyphStrip;

constexpr std::string_view DIGITS = "0123456789";
// The OS's UI font at 11 points.
FontSpec systemFont() {
    return {.family = "", .sizePoints = 11};
}

GlyphStrip strip(const FontSpec& font, double scale, std::string_view chars) {
    auto result = rasterGlyphStrip(font, scale, chars);
    REQUIRE_MESSAGE(result.has_value(), (result ? "" : result.error().describe()));
    return std::move(*result);
}

// The first and the last column in [x0, x1) with any coverage; `found` is false for no ink.
struct InkColumns {
    bool found = false;
    std::uint32_t first = 0;
    std::uint32_t last = 0;
};

InkColumns inkColumns(const GlyphStrip& glyphs, std::uint32_t x0, std::uint32_t x1) {
    InkColumns ink;
    for (std::uint32_t x = x0; x < x1; ++x) {
        for (std::uint32_t y = 0; y < glyphs.height; ++y) {
            if (glyphs.coverage[(static_cast<std::size_t>(y) * glyphs.width) + x] != 0) {
                if (!ink.found) {
                    ink = InkColumns{.found = true, .first = x, .last = x};
                }
                ink.last = x;
                break;
            }
        }
    }
    return ink;
}

std::uint8_t maxCoverage(const GlyphStrip& glyphs) {
    return glyphs.coverage.empty() ? 0 : *std::ranges::max_element(glyphs.coverage);
}

Errc failure(const FontSpec& font, double scale, std::string_view chars) {
    auto result = rasterGlyphStrip(font, scale, chars);
    REQUIRE_FALSE(result.has_value());
    return result.error().code;
}

// Each glyph's origin sits in its cell where the centring puts it. The cells place glyph origins,
// not ink: the side bearings stay as the font has them. A glyph's ink in its cell is where its
// one-character strip has it, moved by the difference of the two centrings, (advance -
// one-character advance) / 2.
void checkOrigins(const FontSpec& font, double scale, std::string_view chars) {
    const GlyphStrip row = strip(font, scale, chars);
    const double cell = row.advance;
    for (std::uint32_t i = 0; i < chars.size(); ++i) {
        CAPTURE(chars[i]);
        const GlyphStrip single = strip(font, scale, chars.substr(i, 1));
        const auto single_ink = inkColumns(single, 0, single.width);
        const auto ink = inkColumns(row, i * row.advance, (i + 1) * row.advance);
        REQUIRE(single_ink.found);
        REQUIRE(ink.found);
        const double expected = (i * cell) + single_ink.first + ((cell - single.advance) / 2);
        CHECK(std::abs(static_cast<double>(ink.first) - expected) <= 1.0);
    }
}

// No cell's ink reaches more than 1 px (antialiasing) into a neighbour cell. Each digit is drawn
// alone between two spaces, so its ink is not clipped by its own cell; its cell in the ten-digit
// strip moves it by (advance - padded advance) / 2, which may add half a pixel.
void checkNeighbourReach(const FontSpec& font, double scale) {
    const GlyphStrip ten = strip(font, scale, DIGITS);
    const double cell = ten.advance;
    for (const char digit : DIGITS) {
        CAPTURE(digit);
        const GlyphStrip padded = strip(font, scale, std::string{' ', digit, ' '});
        const auto ink = inkColumns(padded, 0, padded.width);
        REQUIRE(ink.found);
        REQUIRE(ink.first > 0);
        REQUIRE(ink.last + 1 < padded.width);
        const double move = (cell - padded.advance) / 2;
        const double left = static_cast<double>(ink.first) - padded.advance + move;
        const double right = static_cast<double>(ink.last) - padded.advance + move;
        CHECK(left >= -1.5);
        CHECK(right <= cell + 0.5);
    }
}

}  // namespace

TEST_CASE("glyph strip: cell layout centres narrower glyphs") {
    constexpr std::array<double, 3> ADVANCES{6.0, 7.5, 7.0};
    const auto layout = dvv::gpu::detail::cellLayout(ADVANCES);
    CHECK(layout.advance == 8.0);
    REQUIRE(layout.glyphX.size() == 3);
    CHECK(layout.glyphX[0] == 1.0);
    CHECK(layout.glyphX[1] == 8.25);
    CHECK(layout.glyphX[2] == 16.5);
}

TEST_CASE("glyph strip: bad arguments") {
    constexpr double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();
    constexpr double INFINITY_VALUE = std::numeric_limits<double>::infinity();
    for (const double size : {0.0, -1.0, NOT_A_NUMBER, INFINITY_VALUE}) {
        CAPTURE(size);
        CHECK(failure({.family = "", .sizePoints = size}, 1, DIGITS) == Errc::Internal);
    }
    for (const double scale : {0.0, -1.0, NOT_A_NUMBER, INFINITY_VALUE}) {
        CAPTURE(scale);
        CHECK(failure(systemFont(), scale, DIGITS) == Errc::Internal);
    }
    // Over 4096 pixels: refused before the OS raster sees it.
    for (const auto& [size, scale] : {std::pair{4096.5, 1.0}, std::pair{1e12, 1.0},
                                      std::pair{11.0, 1e6}, std::pair{1e300, 1e300}}) {
        CAPTURE(size);
        CAPTURE(scale);
        CHECK(failure({.family = "", .sizePoints = size}, scale, "0") == Errc::Internal);
    }
    CHECK(rasterGlyphStrip({.family = "", .sizePoints = 1024}, 4, "0").has_value());
    // A strip wider than MAX_GLYPH_STRIP_SIDE is refused the same way on every OS; a wide one
    // inside it is drawn.
    CHECK(failure({.family = "", .sizePoints = 1024}, 4, std::string(64, '0')) == Errc::Internal);
    const auto wide = rasterGlyphStrip({.family = "", .sizePoints = 128}, 1, std::string(64, '0'));
    REQUIRE(wide.has_value());
    CHECK(wide->width <= dvv::gpu::MAX_GLYPH_STRIP_SIDE);
    CHECK(wide->width > dvv::gpu::MAX_GLYPH_STRIP_SIDE / 4);
    const std::string too_many(65, '0');
    for (const std::string_view chars :
         {std::string_view{}, std::string_view{"\x01"}, std::string_view{"0\x7f"},
          std::string_view{"\xc3\xa9"}, std::string_view{too_many}}) {
        CAPTURE(chars);
        CHECK(failure(systemFont(), 1, chars) == Errc::Internal);
    }
    CHECK(rasterGlyphStrip(systemFont(), 1, std::string(64, '0')).has_value());
}

TEST_CASE("glyph strip: metrics") {
    const GlyphStrip one = strip(systemFont(), 1, DIGITS);
    const GlyphStrip two = strip(systemFont(), 2, DIGITS);
    for (const GlyphStrip* glyphs : {&one, &two}) {
        CHECK(glyphs->advance > 0);
        CHECK(glyphs->width == DIGITS.size() * glyphs->advance);
        CHECK(glyphs->height == glyphs->ascent + glyphs->descent);
        CHECK(glyphs->ascent > glyphs->descent);
        CHECK(glyphs->descent > 0);
        CHECK(glyphs->coverage.size() == static_cast<std::size_t>(glyphs->width) * glyphs->height);
        CHECK(maxCoverage(*glyphs) > 128);
    }
    CHECK(std::abs(static_cast<double>(two.advance) - (2.0 * one.advance)) <= 1.0);
}

TEST_CASE("glyph strip: glyph origins") {
    for (const double scale : {1.0, 2.0}) {
        CAPTURE(scale);
        checkOrigins(systemFont(), scale, DIGITS);
        // Letters of unequal advances: every glyph but the widest moves by whole pixels.
        checkOrigins(systemFont(), scale, "il0W");
        checkNeighbourReach(systemFont(), scale);
    }
}

// The first and the last row with any coverage; `found` is false for no ink.
struct InkRows {
    bool found = false;
    std::uint32_t top = 0;
    std::uint32_t bottom = 0;
};

InkRows inkRows(const GlyphStrip& glyphs) {
    InkRows ink;
    for (std::uint32_t y = 0; y < glyphs.height; ++y) {
        const auto row =
            std::span(glyphs.coverage).subspan(std::size_t{y} * glyphs.width, glyphs.width);
        if (std::ranges::any_of(row, [](std::uint8_t c) { return c != 0; })) {
            ink.top = ink.found ? ink.top : y;
            ink.bottom = y;
            ink.found = true;
        }
    }
    return ink;
}

TEST_CASE("glyph strip: the baseline is the top edge of row ascent") {
    // A '1' sits flat on the baseline: its lowest ink row is exactly ascent - 1, so a baseline one
    // row off either way fails. Round digits overshoot the baseline downwards only, so the lowest
    // ink row of all digits is ascent - 1 or ascent. Their tops reach into the upper half of the
    // ascent rows. A line drawn from its top, or off by the descent, fails both.
    for (const double scale : {1.0, 2.0}) {
        CAPTURE(scale);
        const GlyphStrip one = strip(systemFont(), scale, "1");
        const InkRows flat = inkRows(one);
        REQUIRE(flat.found);
        CAPTURE(one.ascent);
        CHECK(flat.bottom == one.ascent - 1);

        const GlyphStrip glyphs = strip(systemFont(), scale, DIGITS);
        const InkRows ink = inkRows(glyphs);
        REQUIRE(ink.found);
        CAPTURE(glyphs.ascent);
        CAPTURE(ink.top);
        CAPTURE(ink.bottom);
        CHECK(ink.bottom >= glyphs.ascent - 1);
        CHECK(ink.bottom <= glyphs.ascent);
        CHECK(ink.top < glyphs.ascent / 2);
    }
}

TEST_CASE("glyph strip: the scale multiplies the point size's metrics") {
    // A bound check only: the cell at scale k lies in the k-pixel window (k x (advance - 1),
    // k x advance] of the scale-1 cell. It does not tell which design was drawn; the design test
    // below does that where the UI font has optical sizes.
    const GlyphStrip one = strip(systemFont(), 1, DIGITS);
    for (const double scale : {2.0, 4.0, 8.0}) {
        CAPTURE(scale);
        const GlyphStrip scaled = strip(systemFont(), scale, DIGITS);
        CHECK(static_cast<double>(scaled.advance) > scale * (one.advance - 1.0));
        CHECK(static_cast<double>(scaled.advance) <= scale * one.advance);
    }
}

#if defined(__APPLE__) || defined(_WIN32)
TEST_CASE("glyph strip: proportional digits stay in their cells") {
    // Georgia's old-style digits have different advances; only the neighbour reach applies.
    const FontSpec georgia{.family = "Georgia", .sizePoints = 11};
    for (const double scale : {1.0, 2.0}) {
        CAPTURE(scale);
        checkOrigins(georgia, scale, DIGITS);
        checkNeighbourReach(georgia, scale);
    }
}

#if defined(__APPLE__)
// Only macOS: its UI font has optical sizes; Windows' message font (Segoe UI) is one design for
// every size, so the two strips may be equal there.
TEST_CASE("glyph strip: a scaled strip keeps the point size's design") {
    // 11 points drawn at scale 2 is the 11-point design, not the 22-point one the OS picks for 22
    // points at scale 1.
    for (const double scale : {2.0, 4.0}) {
        CAPTURE(scale);
        const GlyphStrip scaled = strip(systemFont(), scale, DIGITS);
        const GlyphStrip large = strip({.family = "", .sizePoints = 11 * scale}, 1, DIGITS);
        CHECK(scaled != large);
    }
}
#endif

TEST_CASE("glyph strip: tabular figures") {
    // The system font draws its tabular digits, all one width.
    const GlyphStrip ten = strip(systemFont(), 2, DIGITS);
    for (std::size_t i = 0; i < DIGITS.size(); ++i) {
        CAPTURE(DIGITS[i]);
        CHECK(strip(systemFont(), 2, DIGITS.substr(i, 1)).advance == ten.advance);
    }
}
#endif

#if defined(_WIN32)
namespace {

// Whether the Windows fonts folder holds this font file.
bool windowsFont(const wchar_t* file) {
    std::array<wchar_t, MAX_PATH> folder{};
    const UINT length = GetWindowsDirectoryW(folder.data(), static_cast<UINT>(folder.size()));
    if (length == 0 || length >= folder.size()) {
        return false;
    }
    std::error_code error;
    return std::filesystem::exists(std::filesystem::path{folder.data()} / L"Fonts" / file, error);
}

}  // namespace

TEST_CASE("glyph strip: a GDI face name") {
    // Font lists on Windows show GDI face names, which are not always DirectWrite family names.
    const GlyphStrip system = strip(systemFont(), 1, DIGITS);
    if (windowsFont(L"ARIALN.TTF")) {
        const GlyphStrip narrow = strip({.family = "Arial Narrow", .sizePoints = 11}, 1, DIGITS);
        CHECK(narrow.family != system.family);
    } else {
        MESSAGE("Arial Narrow is not installed: skipped");
    }
    // A weight in the face name: the same family as the UI font, other glyphs. Segoe UI Semibold
    // ships with Windows, so this part always runs: a missing file fails the test.
    REQUIRE_MESSAGE(windowsFont(L"seguisb.ttf"), "Segoe UI Semibold is not installed");
    const GlyphStrip semibold = strip({.family = "Segoe UI Semibold", .sizePoints = 11}, 1, DIGITS);
    // The literal, not the UI font's family: East Asian locales have another message font.
    CHECK(semibold.family == "Segoe UI");
    CHECK(semibold.coverage != system.coverage);
}

TEST_CASE("glyph strip: tabular figures of a font with proportional digits") {
    // Constantia's default digits have different advances (the UI font's are already tabular), so
    // equal cells here show that the tabular-figures feature reached the shaper. Constantia ships
    // with Windows, so this test always runs: a missing file fails it.
    REQUIRE_MESSAGE(windowsFont(L"constan.ttf"), "Constantia is not installed");
    const FontSpec constantia{.family = "Constantia", .sizePoints = 11};
    const GlyphStrip ten = strip(constantia, 2, DIGITS);
    CHECK(ten.family == "Constantia");
    for (std::size_t i = 0; i < DIGITS.size(); ++i) {
        CAPTURE(DIGITS[i]);
        CHECK(strip(constantia, 2, DIGITS.substr(i, 1)).advance == ten.advance);
    }
}
#endif

TEST_CASE("glyph strip: the system font is a real font") {
    const GlyphStrip glyphs = strip(systemFont(), 1, DIGITS);
    CHECK_FALSE(glyphs.family.empty());
    CHECK(glyphs.family.find("Times") == std::string::npos);
}

TEST_CASE("glyph strip: unknown family") {
    for (const std::string family :
         {"NoSuchFamily-dvv", "\xc3\x9cnknown \xe5\xad\x97\xe4\xbd\x93"}) {
        CAPTURE(family);
        const GlyphStrip glyphs = strip({.family = family, .sizePoints = 11}, 1, DIGITS);
        CHECK(maxCoverage(glyphs) > 128);
    }
}
