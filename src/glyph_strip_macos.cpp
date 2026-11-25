#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>

#include "dvv/base/error.h"
#include "dvv/gpu/glyph_strip.h"
#include "glyph_strip_os.h"

namespace dvv::gpu::detail {

namespace {

struct CfRelease {
    void operator()(CFTypeRef ref) const noexcept { CFRelease(ref); }
};

// Owns one Core Foundation reference (a Create or Copy result).
template <class Ref>
using CfPtr = std::unique_ptr<std::remove_pointer_t<Ref>, CfRelease>;

CfPtr<CFStringRef> cfString(std::string_view utf8) {
    return CfPtr<CFStringRef>{
        CFStringCreateWithBytes(kCFAllocatorDefault, reinterpret_cast<const UInt8*>(utf8.data()),
                                static_cast<CFIndex>(utf8.size()), kCFStringEncodingUTF8, false)};
}

std::string utf8String(CFStringRef text) {
    if (text == nullptr) {
        return {};
    }
    const CFIndex length = CFStringGetLength(text);
    CFIndex bytes = 0;
    CFStringGetBytes(text, CFRangeMake(0, length), kCFStringEncodingUTF8, 0, false, nullptr, 0,
                     &bytes);
    std::string result(static_cast<std::size_t>(bytes), '\0');
    CFStringGetBytes(text, CFRangeMake(0, length), kCFStringEncodingUTF8, 0, false,
                     reinterpret_cast<UInt8*>(result.data()), bytes, nullptr);
    return result;
}

CfPtr<CFDictionaryRef> dictionary(CFStringRef key, CFTypeRef value) {
    const void* keys[] = {key};
    const void* values[] = {value};
    return CfPtr<CFDictionaryRef>{CFDictionaryCreate(kCFAllocatorDefault, keys, values, 1,
                                                     &kCFTypeDictionaryKeyCallBacks,
                                                     &kCFTypeDictionaryValueCallBacks)};
}

// The font of a family, or nothing when no installed font has that family name. The descriptor
// matches family names (what font lists show), not PostScript names.
CfPtr<CTFontRef> familyFont(const std::string& family, double size_points) {
    const auto name = cfString(family);
    if (!name) {
        return {};
    }
    const auto attributes = dictionary(kCTFontFamilyNameAttribute, name.get());
    if (!attributes) {
        return {};
    }
    const CfPtr<CTFontDescriptorRef> wanted{CTFontDescriptorCreateWithAttributes(attributes.get())};
    if (!wanted) {
        return {};
    }
    const CfPtr<CTFontDescriptorRef> found{
        CTFontDescriptorCreateMatchingFontDescriptor(wanted.get(), nullptr)};
    if (!found) {
        return {};
    }
    return CfPtr<CTFontRef>{CTFontCreateWithFontDescriptor(found.get(), size_points, nullptr)};
}

// The font asked for (the UI font for "" or a family no installed font has), with tabular figures
// requested: digits of one width where the font has them.
Result<CfPtr<CTFontRef>> tabularFont(const FontSpec& font) {
    CfPtr<CTFontRef> base;
    if (!font.family.empty()) {
        base = familyFont(font.family, font.sizePoints);
    }
    if (!base) {
        // The UI font itself, never by its hidden name (a name lookup gives Times).
        base.reset(CTFontCreateUIFontForLanguage(kCTFontUIFontSystem, font.sizePoints, nullptr));
    }
    if (!base) {
        return makeError(Errc::Unsupported, "glyph strip: Core Text gave no system font");
    }
    const auto tag = cfString("tnum");
    const int on = 1;
    const CfPtr<CFNumberRef> value{CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &on)};
    if (!tag || !value) {
        return makeError(Errc::Unsupported, "glyph strip: Core Foundation failed");
    }
    const void* keys[] = {kCTFontOpenTypeFeatureTag, kCTFontOpenTypeFeatureValue};
    const void* values[] = {tag.get(), value.get()};
    const CfPtr<CFDictionaryRef> feature{CFDictionaryCreate(kCFAllocatorDefault, keys, values, 2,
                                                            &kCFTypeDictionaryKeyCallBacks,
                                                            &kCFTypeDictionaryValueCallBacks)};
    const void* features_items[] = {feature.get()};
    const CfPtr<CFArrayRef> features{
        feature ? CFArrayCreate(kCFAllocatorDefault, features_items, 1, &kCFTypeArrayCallBacks)
                : nullptr};
    const auto attributes =
        features ? dictionary(kCTFontFeatureSettingsAttribute, features.get()) : nullptr;
    const CfPtr<CTFontDescriptorRef> descriptor{
        attributes ? CTFontDescriptorCreateWithAttributes(attributes.get()) : nullptr};
    if (!descriptor) {
        return makeError(Errc::Unsupported, "glyph strip: Core Foundation failed");
    }
    CfPtr<CTFontRef> tabular{
        CTFontCreateCopyWithAttributes(base.get(), 0.0, nullptr, descriptor.get())};
    if (!tabular) {
        return makeError(Errc::Unsupported, "glyph strip: Core Text could not copy the font");
    }
    return tabular;
}

// One shaped glyph: drawn with its run's font (a fallback font for a character the font lacks).
struct ShapedGlyph {
    CTFontRef font = nullptr;  // owned by the line
    CGGlyph glyph = 0;
    double x = 0;  // from the character's origin
};

// One character shaped alone: feature settings act only in shaping, so its glyph comes from a
// line, not from the character map.
struct ShapedChar {
    CfPtr<CTLineRef> line;
    std::vector<ShapedGlyph> glyphs;
    double advance = 0;
};

Result<ShapedChar> shapeChar(CTFontRef font, char c) {
    const auto text = cfString(std::string_view{&c, 1});
    const auto attributes = dictionary(kCTFontAttributeName, font);
    const CfPtr<CFAttributedStringRef> attributed{
        text && attributes
            ? CFAttributedStringCreate(kCFAllocatorDefault, text.get(), attributes.get())
            : nullptr};
    if (!attributed) {
        return makeError(Errc::Unsupported, "glyph strip: Core Foundation failed");
    }
    ShapedChar shaped;
    shaped.line.reset(CTLineCreateWithAttributedString(attributed.get()));
    if (!shaped.line) {
        return makeError(Errc::Unsupported,
                         std::format("glyph strip: Core Text could not shape '{}'", c));
    }
    shaped.advance = CTLineGetTypographicBounds(shaped.line.get(), nullptr, nullptr, nullptr);
    CFArrayRef runs = CTLineGetGlyphRuns(shaped.line.get());
    for (CFIndex r = 0; r < CFArrayGetCount(runs); ++r) {
        auto* run = static_cast<CTRunRef>(const_cast<void*>(CFArrayGetValueAtIndex(runs, r)));
        auto* run_font = static_cast<CTFontRef>(
            CFDictionaryGetValue(CTRunGetAttributes(run), kCTFontAttributeName));
        const CFIndex count = CTRunGetGlyphCount(run);
        std::vector<CGGlyph> glyphs(static_cast<std::size_t>(count));
        std::vector<CGPoint> positions(static_cast<std::size_t>(count));
        CTRunGetGlyphs(run, CFRangeMake(0, count), glyphs.data());
        CTRunGetPositions(run, CFRangeMake(0, count), positions.data());
        for (std::size_t g = 0; g < glyphs.size(); ++g) {
            shaped.glyphs.push_back({.font = run_font != nullptr ? run_font : font,
                                     .glyph = glyphs[g],
                                     .x = positions[g].x});
        }
    }
    return shaped;
}

// A length in whole output pixels, rounded up; nothing for a negative one.
double wholePixels(double value) {
    return value > 0 ? std::ceil(value) : 0;
}

}  // namespace

Result<GlyphStrip> rasterGlyphStripOs(const FontSpec& font, double scale, std::string_view chars) {
    auto tabular = tabularFont(font);
    if (!tabular) {
        return std::unexpected(std::move(tabular.error()));
    }
    const CTFontRef ct_font = tabular->get();

    // Shaped in font points; the cells are laid out in output pixels.
    std::vector<ShapedChar> shaped;
    std::vector<double> advances;
    shaped.reserve(chars.size());
    advances.reserve(chars.size());
    for (const char c : chars) {
        auto one = shapeChar(ct_font, c);
        if (!one) {
            return std::unexpected(std::move(one.error()));
        }
        advances.push_back(one->advance * scale);
        shaped.push_back(std::move(*one));
    }
    const CellLayout layout = cellLayout(advances);

    // Checked before any narrowing; a MAX_GLYPH_STRIP_SIDE strip fits Core Graphics' int sizes.
    const double advance = wholePixels(layout.advance);
    const double ascent = wholePixels(CTFontGetAscent(ct_font) * scale);
    const double descent = wholePixels(CTFontGetDescent(ct_font) * scale);
    const double width = static_cast<double>(chars.size()) * advance;
    const double height = ascent + descent;
    if (auto size = checkStripSize(width, height, advance, "Core Text"); !size) {
        return std::unexpected(std::move(size.error()));
    }
    GlyphStrip strip;
    strip.advance = static_cast<std::uint32_t>(advance);
    strip.ascent = static_cast<std::uint32_t>(ascent);
    strip.descent = static_cast<std::uint32_t>(descent);
    strip.width = static_cast<std::uint32_t>(width);
    strip.height = static_cast<std::uint32_t>(height);
    const CfPtr<CFStringRef> family{CTFontCopyFamilyName(ct_font)};
    strip.family = utf8String(family.get());
    strip.coverage.assign(static_cast<std::size_t>(strip.width) * strip.height, 0);

    // White coverage on black in an 8-bit gray bitmap. Its first row is the top: the baseline,
    // the top edge of row `ascent`, is `descent` pixels above the bottom in Core Graphics' upward
    // y. The font draws at its point size in a context scaled to output pixels.
    const CfPtr<CGColorSpaceRef> gray{CGColorSpaceCreateDeviceGray()};
    const CfPtr<CGContextRef> context{
        CGBitmapContextCreate(strip.coverage.data(), strip.width, strip.height, 8, strip.width,
                              gray.get(), static_cast<std::uint32_t>(kCGImageAlphaNone))};
    if (!gray || !context) {
        return makeError(Errc::Unsupported, "glyph strip: Core Graphics gave no bitmap");
    }
    CGContextSetAllowsFontSmoothing(context.get(), false);
    CGContextSetShouldSmoothFonts(context.get(), false);
    CGContextSetShouldAntialias(context.get(), true);
    CGContextSetAllowsFontSubpixelPositioning(context.get(), true);
    CGContextSetShouldSubpixelPositionFonts(context.get(), true);
    CGContextSetAllowsFontSubpixelQuantization(context.get(), false);
    CGContextSetShouldSubpixelQuantizeFonts(context.get(), false);
    CGContextSetGrayFillColor(context.get(), 1.0, 1.0);
    CGContextScaleCTM(context.get(), scale, scale);
    CGContextSetTextMatrix(context.get(), CGAffineTransformIdentity);
    const double baseline = descent / scale;
    for (std::size_t i = 0; i < shaped.size(); ++i) {
        for (const ShapedGlyph& glyph : shaped[i].glyphs) {
            const CGPoint position{.x = (layout.glyphX[i] / scale) + glyph.x, .y = baseline};
            CTFontDrawGlyphs(glyph.font, &glyph.glyph, &position, 1, context.get());
        }
    }
    CGContextFlush(context.get());
    return strip;
}

}  // namespace dvv::gpu::detail
