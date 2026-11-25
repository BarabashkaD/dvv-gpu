#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <dwrite.h>
#include <dwrite_2.h>
#include <windows.h>
#include <wrl/client.h>

#include "dvv/base/error.h"
#include "dvv/gpu/glyph_strip.h"
#include "glyph_strip_os.h"

namespace dvv::gpu::detail {

namespace {

using Microsoft::WRL::ComPtr;

// DIPs (1/96 inch) per OS font point (1/72 inch).
constexpr double DIPS_PER_POINT = 96.0 / 72.0;
constexpr wchar_t LOCALE[] = L"en-us";
// The most glyphs one character may shape to.
constexpr UINT32 MAX_GLYPHS = 16;

std::wstring wideString(std::string_view utf8) {
    if (utf8.empty() || utf8.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    const int bytes = static_cast<int>(utf8.size());
    const int length =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), bytes, nullptr, 0);
    if (length <= 0) {
        return {};
    }
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), bytes, wide.data(), length);
    return wide;
}

std::string utf8String(std::wstring_view wide) {
    if (wide.empty() || wide.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    const int length = static_cast<int>(wide.size());
    const int bytes =
        WideCharToMultiByte(CP_UTF8, 0, wide.data(), length, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) {
        return {};
    }
    std::string utf8(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), length, utf8.data(), bytes, nullptr, nullptr);
    return utf8;
}

ComPtr<IDWriteFont> gdiFont(IDWriteGdiInterop* interop, const LOGFONTW& log_font) {
    ComPtr<IDWriteFont> font;
    if (FAILED(interop->CreateFontFromLOGFONT(&log_font, &font))) {
        return {};
    }
    return font;
}

// The regular font of a GDI face name (what Windows font lists show, e.g. "Arial Narrow" or
// "Segoe UI Semibold"); nothing when no installed font has that face name.
ComPtr<IDWriteFont> faceFont(IDWriteGdiInterop* interop, const std::string& family) {
    const std::wstring face = wideString(family);
    if (face.empty() || face.size() >= LF_FACESIZE) {
        return {};
    }
    LOGFONTW log_font{};
    log_font.lfWeight = FW_NORMAL;
    log_font.lfCharSet = DEFAULT_CHARSET;
    face.copy(log_font.lfFaceName, face.size());
    return gdiFont(interop, log_font);
}

// The UI font: the font of message boxes.
ComPtr<IDWriteFont> messageFont(IDWriteGdiInterop* interop) {
    NONCLIENTMETRICSW metrics{};
    metrics.cbSize = sizeof(metrics);
    if (SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof(metrics), &metrics, 0) == FALSE) {
        return {};
    }
    return gdiFont(interop, metrics.lfMessageFont);
}

// The font's family name in en-us, else its first one.
std::string familyName(IDWriteFont* font) {
    ComPtr<IDWriteFontFamily> family;
    ComPtr<IDWriteLocalizedStrings> names;
    if (FAILED(font->GetFontFamily(&family)) || FAILED(family->GetFamilyNames(&names)) ||
        names->GetCount() == 0) {
        return {};
    }
    UINT32 index = 0;
    BOOL exists = FALSE;
    if (FAILED(names->FindLocaleName(LOCALE, &index, &exists)) || exists == FALSE) {
        index = 0;
    }
    UINT32 length = 0;
    if (FAILED(names->GetStringLength(index, &length))) {
        return {};
    }
    std::wstring name(static_cast<std::size_t>(length) + 1, L'\0');
    if (FAILED(names->GetString(index, name.data(), length + 1))) {
        return {};
    }
    name.resize(length);
    return utf8String(name);
}

// The text of one script analysis: one run, left to right, in the en-us locale. Lives on the
// stack for one call: the reference count is not used.
class OneRunSource final : public IDWriteTextAnalysisSource {
public:
    explicit OneRunSource(std::wstring_view text) : text_(text) {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** object) override {
        if (id == __uuidof(IUnknown) || id == __uuidof(IDWriteTextAnalysisSource)) {
            *object = static_cast<IDWriteTextAnalysisSource*>(this);
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }

    HRESULT STDMETHODCALLTYPE GetTextAtPosition(UINT32 position, const WCHAR** text,
                                                UINT32* length) override {
        if (position >= text_.size()) {
            *text = nullptr;
            *length = 0;
        } else {
            *text = text_.data() + position;
            *length = static_cast<UINT32>(text_.size()) - position;
        }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetTextBeforePosition(UINT32 position, const WCHAR** text,
                                                    UINT32* length) override {
        if (position == 0 || position > text_.size()) {
            *text = nullptr;
            *length = 0;
        } else {
            *text = text_.data();
            *length = position;
        }
        return S_OK;
    }
    DWRITE_READING_DIRECTION STDMETHODCALLTYPE GetParagraphReadingDirection() override {
        return DWRITE_READING_DIRECTION_LEFT_TO_RIGHT;
    }
    HRESULT STDMETHODCALLTYPE GetLocaleName(UINT32 position, UINT32* length,
                                            const WCHAR** locale) override {
        *length = rest(position);
        *locale = LOCALE;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetNumberSubstitution(
        UINT32 position, UINT32* length, IDWriteNumberSubstitution** substitution) override {
        *length = rest(position);
        *substitution = nullptr;
        return S_OK;
    }

private:
    UINT32 rest(UINT32 position) const {
        return position < text_.size() ? static_cast<UINT32>(text_.size()) - position : 0;
    }

    std::wstring_view text_;
};

// Keeps the first script the analysis reports. Lives on the stack for one call.
class ScriptSink final : public IDWriteTextAnalysisSink {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** object) override {
        if (id == __uuidof(IUnknown) || id == __uuidof(IDWriteTextAnalysisSink)) {
            *object = static_cast<IDWriteTextAnalysisSink*>(this);
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }

    HRESULT STDMETHODCALLTYPE SetScriptAnalysis(UINT32 /*position*/, UINT32 /*length*/,
                                                const DWRITE_SCRIPT_ANALYSIS* script) override {
        if (!found_) {
            script_ = *script;
            found_ = true;
        }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE
    SetLineBreakpoints(UINT32 /*position*/, UINT32 /*length*/,
                       const DWRITE_LINE_BREAKPOINT* /*breakpoints*/) override {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetBidiLevel(UINT32 /*position*/, UINT32 /*length*/,
                                           UINT8 /*explicit_level*/,
                                           UINT8 /*resolved_level*/) override {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE
    SetNumberSubstitution(UINT32 /*position*/, UINT32 /*length*/,
                          IDWriteNumberSubstitution* /*substitution*/) override {
        return S_OK;
    }

    [[nodiscard]] bool found() const { return found_; }
    [[nodiscard]] const DWRITE_SCRIPT_ANALYSIS& script() const { return script_; }

private:
    DWRITE_SCRIPT_ANALYSIS script_{};
    bool found_ = false;
};

// One character shaped alone, with tabular figures: its glyphs and their placements in DIPs.
struct ShapedChar {
    std::vector<UINT16> glyphs;
    std::vector<FLOAT> advances;
    std::vector<DWRITE_GLYPH_OFFSET> offsets;
    double advance = 0;  // the sum of the glyph advances
};

Result<ShapedChar> shapeChar(IDWriteTextAnalyzer* analyzer, IDWriteFontFace* face, FLOAT em_size,
                             char c) {
    const WCHAR text = static_cast<WCHAR>(c);
    OneRunSource source{std::wstring_view{&text, 1}};
    ScriptSink sink;
    if (FAILED(analyzer->AnalyzeScript(&source, 0, 1, &sink)) || !sink.found()) {
        return makeError(Errc::Unsupported,
                         std::format("glyph strip: DirectWrite could not analyse '{}'", c));
    }
    DWRITE_FONT_FEATURE tabular{.nameTag = DWRITE_FONT_FEATURE_TAG_TABULAR_FIGURES, .parameter = 1};
    const DWRITE_TYPOGRAPHIC_FEATURES features{.features = &tabular, .featureCount = 1};
    const DWRITE_TYPOGRAPHIC_FEATURES* feature_ranges[] = {&features};
    const UINT32 feature_lengths[] = {1};

    UINT16 cluster_map[1]{};
    DWRITE_SHAPING_TEXT_PROPERTIES text_properties[1]{};
    std::vector<DWRITE_SHAPING_GLYPH_PROPERTIES> glyph_properties(MAX_GLYPHS);
    ShapedChar shaped;
    shaped.glyphs.resize(MAX_GLYPHS);
    UINT32 count = 0;
    if (FAILED(analyzer->GetGlyphs(&text, 1, face, FALSE, FALSE, &sink.script(), LOCALE, nullptr,
                                   feature_ranges, feature_lengths, 1, MAX_GLYPHS, cluster_map,
                                   text_properties, shaped.glyphs.data(), glyph_properties.data(),
                                   &count))) {
        return makeError(Errc::Unsupported,
                         std::format("glyph strip: DirectWrite could not shape '{}'", c));
    }
    shaped.glyphs.resize(count);
    shaped.advances.resize(count);
    shaped.offsets.resize(count);
    if (count > 0 &&
        FAILED(analyzer->GetGlyphPlacements(
            &text, cluster_map, text_properties, 1, shaped.glyphs.data(), glyph_properties.data(),
            count, face, em_size, FALSE, FALSE, &sink.script(), LOCALE, feature_ranges,
            feature_lengths, 1, shaped.advances.data(), shaped.offsets.data()))) {
        return makeError(Errc::Unsupported,
                         std::format("glyph strip: DirectWrite could not place '{}'", c));
    }
    for (const FLOAT glyph_advance : shaped.advances) {
        shaped.advance += glyph_advance;
    }
    return shaped;
}

// A length in whole output pixels, rounded up; nothing for a negative one.
double wholePixels(double value) {
    return value > 0 ? std::ceil(value) : 0;
}

// Draws one character's glyphs, its origin at (x, baseline) in output pixels, into the strip:
// grayscale coverage, no hinting, unrounded positions.
Result<void> drawChar(IDWriteFactory2* factory, IDWriteFontFace* face, FLOAT em_size,
                      double pixels_per_dip, const ShapedChar& shaped, double x, double baseline,
                      GlyphStrip& strip) {
    if (shaped.glyphs.empty()) {
        return {};
    }
    DWRITE_GLYPH_RUN run{};
    run.fontFace = face;
    run.fontEmSize = em_size;
    run.glyphCount = static_cast<UINT32>(shaped.glyphs.size());
    run.glyphIndices = shaped.glyphs.data();
    run.glyphAdvances = shaped.advances.data();
    run.glyphOffsets = shaped.offsets.data();
    // DIPs to output pixels, moved to the character's origin.
    const DWRITE_MATRIX transform{.m11 = static_cast<FLOAT>(pixels_per_dip),
                                  .m12 = 0,
                                  .m21 = 0,
                                  .m22 = static_cast<FLOAT>(pixels_per_dip),
                                  .dx = static_cast<FLOAT>(x),
                                  .dy = static_cast<FLOAT>(baseline)};
    ComPtr<IDWriteGlyphRunAnalysis> analysis;
    if (FAILED(factory->CreateGlyphRunAnalysis(
            &run, &transform, DWRITE_RENDERING_MODE_NATURAL_SYMMETRIC,
            DWRITE_MEASURING_MODE_NATURAL, DWRITE_GRID_FIT_MODE_DISABLED,
            DWRITE_TEXT_ANTIALIAS_MODE_GRAYSCALE, 0, 0, &analysis))) {
        return makeError(Errc::Unsupported, "glyph strip: DirectWrite could not analyse a glyph");
    }
    RECT bounds{};
    if (FAILED(analysis->GetAlphaTextureBounds(DWRITE_TEXTURE_ALIASED_1x1, &bounds))) {
        return makeError(Errc::Unsupported, "glyph strip: DirectWrite gave no glyph bounds");
    }
    if (bounds.right <= bounds.left || bounds.bottom <= bounds.top) {
        return {};  // no ink, e.g. a space
    }
    const auto texture_width = static_cast<std::size_t>(bounds.right - bounds.left);
    const auto texture_height = static_cast<std::size_t>(bounds.bottom - bounds.top);
    std::vector<BYTE> texture(texture_width * texture_height);
    if (FAILED(analysis->CreateAlphaTexture(DWRITE_TEXTURE_ALIASED_1x1, &bounds, texture.data(),
                                            static_cast<UINT32>(texture.size())))) {
        return makeError(Errc::Unsupported, "glyph strip: DirectWrite gave no glyph coverage");
    }
    // Neighbouring glyphs may overlap: the larger coverage wins. Ink outside the strip is cut.
    for (std::size_t row = 0; row < texture_height; ++row) {
        const LONG y = bounds.top + static_cast<LONG>(row);
        if (y < 0 || y >= static_cast<LONG>(strip.height)) {
            continue;
        }
        for (std::size_t column = 0; column < texture_width; ++column) {
            const LONG strip_x = bounds.left + static_cast<LONG>(column);
            if (strip_x < 0 || strip_x >= static_cast<LONG>(strip.width)) {
                continue;
            }
            std::uint8_t& pixel = strip.coverage[(static_cast<std::size_t>(y) * strip.width) +
                                                 static_cast<std::size_t>(strip_x)];
            pixel = std::max(pixel, texture[(row * texture_width) + column]);
        }
    }
    return {};
}

}  // namespace

// DirectWrite with the shared factory (thread-safe): the font is the OS's at the point size, its
// em in DIPs; a transform draws it `scale` pixels per font point.
Result<GlyphStrip> rasterGlyphStripOs(const FontSpec& font, double scale, std::string_view chars) {
    ComPtr<IDWriteFactory2> factory;
    ComPtr<IDWriteGdiInterop> interop;
    ComPtr<IDWriteTextAnalyzer> analyzer;
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory2),
                                   reinterpret_cast<IUnknown**>(factory.GetAddressOf()))) ||
        FAILED(factory->GetGdiInterop(&interop)) ||
        FAILED(factory->CreateTextAnalyzer(&analyzer))) {
        return makeError(Errc::Unsupported, "glyph strip: DirectWrite is not available");
    }
    // A family no installed face has falls back to the UI font.
    ComPtr<IDWriteFont> dwrite_font;
    if (!font.family.empty()) {
        dwrite_font = faceFont(interop.Get(), font.family);
    }
    if (!dwrite_font) {
        dwrite_font = messageFont(interop.Get());
    }
    ComPtr<IDWriteFontFace> face;
    if (!dwrite_font || FAILED(dwrite_font->CreateFontFace(&face))) {
        return makeError(Errc::Unsupported, "glyph strip: DirectWrite gave no system font");
    }
    const double em_dips = font.sizePoints * DIPS_PER_POINT;
    const auto em_size = static_cast<FLOAT>(em_dips);
    const double pixels_per_dip = scale / DIPS_PER_POINT;

    // Shaped in DIPs; the cells are laid out in output pixels.
    std::vector<ShapedChar> shaped;
    std::vector<double> advances;
    shaped.reserve(chars.size());
    advances.reserve(chars.size());
    for (const char c : chars) {
        auto one = shapeChar(analyzer.Get(), face.Get(), em_size, c);
        if (!one) {
            return std::unexpected(std::move(one.error()));
        }
        advances.push_back(one->advance * pixels_per_dip);
        shaped.push_back(std::move(*one));
    }
    const CellLayout layout = cellLayout(advances);

    DWRITE_FONT_METRICS metrics{};
    face->GetMetrics(&metrics);
    if (metrics.designUnitsPerEm == 0) {
        return makeError(Errc::Unsupported, "glyph strip: DirectWrite gave no font metrics");
    }
    const double design_to_pixels = em_dips * pixels_per_dip / metrics.designUnitsPerEm;
    // Checked before any narrowing; a MAX_GLYPH_STRIP_SIDE strip fits the int sizes.
    const double advance = wholePixels(layout.advance);
    const double ascent = wholePixels(metrics.ascent * design_to_pixels);
    const double descent = wholePixels(metrics.descent * design_to_pixels);
    const double width = static_cast<double>(chars.size()) * advance;
    const double height = ascent + descent;
    if (auto size = checkStripSize(width, height, advance, "DirectWrite"); !size) {
        return std::unexpected(std::move(size.error()));
    }
    GlyphStrip strip;
    strip.advance = static_cast<std::uint32_t>(advance);
    strip.ascent = static_cast<std::uint32_t>(ascent);
    strip.descent = static_cast<std::uint32_t>(descent);
    strip.width = static_cast<std::uint32_t>(width);
    strip.height = static_cast<std::uint32_t>(height);
    strip.family = familyName(dwrite_font.Get());
    strip.coverage.assign(static_cast<std::size_t>(strip.width) * strip.height, 0);

    // The baseline is the top edge of row `ascent`.
    for (std::size_t i = 0; i < shaped.size(); ++i) {
        auto drawn = drawChar(factory.Get(), face.Get(), em_size, pixels_per_dip, shaped[i],
                              layout.glyphX[i], ascent, strip);
        if (!drawn) {
            return std::unexpected(std::move(drawn.error()));
        }
    }
    return strip;
}

}  // namespace dvv::gpu::detail
