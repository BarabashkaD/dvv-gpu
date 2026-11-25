#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cairo.h>
#include <glib-object.h>
#include <pango/pango.h>
#include <pango/pangocairo.h>

#include "dvv/base/error.h"
#include "dvv/gpu/glyph_strip.h"
#include "glyph_strip_os.h"

namespace dvv::gpu::detail {

namespace {

struct GObjectUnref {
    void operator()(gpointer object) const noexcept { g_object_unref(object); }
};

struct FontDescriptionFree {
    void operator()(PangoFontDescription* description) const noexcept {
        pango_font_description_free(description);
    }
};

struct AttrListUnref {
    void operator()(PangoAttrList* list) const noexcept { pango_attr_list_unref(list); }
};

struct FontMetricsUnref {
    void operator()(PangoFontMetrics* metrics) const noexcept { pango_font_metrics_unref(metrics); }
};

struct FontOptionsDestroy {
    void operator()(cairo_font_options_t* options) const noexcept {
        cairo_font_options_destroy(options);
    }
};

struct SurfaceDestroy {
    void operator()(cairo_surface_t* surface) const noexcept { cairo_surface_destroy(surface); }
};

struct CairoDestroy {
    void operator()(cairo_t* cairo) const noexcept { cairo_destroy(cairo); }
};

// Owns one GObject reference (a new object, or one taken with g_object_ref).
template <class Object>
using GPtr = std::unique_ptr<Object, GObjectUnref>;

using FontDescriptionPtr = std::unique_ptr<PangoFontDescription, FontDescriptionFree>;
using AttrListPtr = std::unique_ptr<PangoAttrList, AttrListUnref>;
using FontMetricsPtr = std::unique_ptr<PangoFontMetrics, FontMetricsUnref>;
using FontOptionsPtr = std::unique_ptr<cairo_font_options_t, FontOptionsDestroy>;
using SurfacePtr = std::unique_ptr<cairo_surface_t, SurfaceDestroy>;
using CairoPtr = std::unique_ptr<cairo_t, CairoDestroy>;

// A length in whole output pixels, rounded up; nothing for a negative one.
double wholePixels(double value) {
    return value > 0 ? std::ceil(value) : 0;
}

// Pango units (fixed point) to font points.
double points(int pango_units) {
    return static_cast<double>(pango_units) / PANGO_SCALE;
}

// The font asked for: the family ("" = "Sans", which fontconfig maps to the desktop's sans font),
// its em `font.sizePoints` font points, which is that many pixels at scale 1.
FontDescriptionPtr fontDescription(const FontSpec& font) {
    FontDescriptionPtr description{pango_font_description_new()};
    if (description) {
        pango_font_description_set_family(description.get(),
                                          font.family.empty() ? "Sans" : font.family.c_str());
        pango_font_description_set_absolute_size(description.get(), font.sizePoints * PANGO_SCALE);
    }
    return description;
}

// The font a one-line layout draws its first character with (a fallback font when the asked font
// lacks it); nothing when no font was found.
PangoFont* firstRunFont(PangoLayout* layout) {
    PangoLayoutLine* line = pango_layout_get_line_readonly(layout, 0);
    if (line == nullptr || line->runs == nullptr) {
        return nullptr;
    }
    const auto* run = static_cast<const PangoGlyphItem*>(line->runs->data);
    return run != nullptr && run->item != nullptr ? run->item->analysis.font : nullptr;
}

}  // namespace

// Pango on a private cairo font map per call (no shared state but fontconfig's, which is
// thread-safe), drawn on an 8-bit alpha surface: the coverage is the alpha. Gray antialiasing,
// no hinting and unrounded glyph positions, so the metrics scale linearly.
Result<GlyphStrip> rasterGlyphStripOs(const FontSpec& font, double scale, std::string_view chars) {
    // Declared first, released last: the layouts go before their context, the context before
    // its font map, the font map before the surface.
    SurfacePtr surface;
    CairoPtr cairo;
    const GPtr<PangoFontMap> font_map{pango_cairo_font_map_new()};
    const GPtr<PangoContext> context{font_map ? pango_font_map_create_context(font_map.get())
                                              : nullptr};
    const FontOptionsPtr options{cairo_font_options_create()};
    const FontDescriptionPtr description = fontDescription(font);
    const AttrListPtr attributes{pango_attr_list_new()};
    if (!context || !options || !description || !attributes) {
        return makeError(Errc::Unsupported, "glyph strip: Pango could not start");
    }
    cairo_font_options_set_antialias(options.get(), CAIRO_ANTIALIAS_GRAY);
    cairo_font_options_set_hint_style(options.get(), CAIRO_HINT_STYLE_NONE);
    cairo_font_options_set_hint_metrics(options.get(), CAIRO_HINT_METRICS_OFF);
    pango_cairo_context_set_font_options(context.get(), options.get());
    pango_context_set_round_glyph_positions(context.get(), FALSE);
    // The font is the one for its point size, drawn `scale` times larger.
    PangoMatrix matrix = PANGO_MATRIX_INIT;
    pango_matrix_scale(&matrix, scale, scale);
    pango_context_set_matrix(context.get(), &matrix);
    pango_context_set_font_description(context.get(), description.get());
    // Tabular figures: digits of one width where the font has them.
    pango_attr_list_insert(attributes.get(), pango_attr_font_features_new("tnum 1"));

    // One layout per character: its own advance, in font points; the cells in output pixels.
    std::vector<GPtr<PangoLayout>> layouts;
    std::vector<double> advances;
    layouts.reserve(chars.size());
    advances.reserve(chars.size());
    for (const char c : chars) {
        GPtr<PangoLayout> layout{pango_layout_new(context.get())};
        if (!layout) {
            return makeError(Errc::Unsupported, "glyph strip: Pango gave no layout");
        }
        pango_layout_set_attributes(layout.get(), attributes.get());
        pango_layout_set_text(layout.get(), &c, 1);
        PangoRectangle logical{};
        pango_layout_get_extents(layout.get(), nullptr, &logical);
        advances.push_back(points(logical.width) * scale);
        layouts.push_back(std::move(layout));
    }
    const CellLayout layout = cellLayout(advances);

    PangoFont* drawn = firstRunFont(layouts.front().get());
    if (drawn == nullptr) {
        return makeError(Errc::Unsupported,
                         "glyph strip: fontconfig found no font (none installed?)");
    }
    const FontMetricsPtr metrics{pango_font_get_metrics(drawn, nullptr)};
    const FontDescriptionPtr resolved{pango_font_describe(drawn)};
    if (!metrics || !resolved) {
        return makeError(Errc::Unsupported, "glyph strip: Pango gave no font metrics");
    }

    // Checked before any narrowing; a MAX_GLYPH_STRIP_SIDE strip fits a cairo image surface.
    const double advance = wholePixels(layout.advance);
    const double ascent = wholePixels(points(pango_font_metrics_get_ascent(metrics.get())) * scale);
    const double descent =
        wholePixels(points(pango_font_metrics_get_descent(metrics.get())) * scale);
    const double width = static_cast<double>(chars.size()) * advance;
    const double height = ascent + descent;
    if (auto size = checkStripSize(width, height, advance, "Pango"); !size) {
        return std::unexpected(std::move(size.error()));
    }
    GlyphStrip strip;
    strip.advance = static_cast<std::uint32_t>(advance);
    strip.ascent = static_cast<std::uint32_t>(ascent);
    strip.descent = static_cast<std::uint32_t>(descent);
    strip.width = static_cast<std::uint32_t>(width);
    strip.height = static_cast<std::uint32_t>(height);
    const char* family = pango_font_description_get_family(resolved.get());
    strip.family = family != nullptr ? family : "";

    // The baseline is the top edge of row `ascent`. The font draws at its point size in a
    // context scaled to output pixels.
    surface.reset(cairo_image_surface_create(CAIRO_FORMAT_A8, static_cast<int>(strip.width),
                                             static_cast<int>(strip.height)));
    if (cairo_surface_status(surface.get()) != CAIRO_STATUS_SUCCESS) {
        return makeError(Errc::Unsupported, "glyph strip: cairo gave no image surface");
    }
    cairo.reset(cairo_create(surface.get()));
    if (cairo_status(cairo.get()) != CAIRO_STATUS_SUCCESS) {
        return makeError(Errc::Unsupported, "glyph strip: cairo gave no drawing context");
    }
    cairo_scale(cairo.get(), scale, scale);
    pango_cairo_update_context(cairo.get(), context.get());
    const double baseline = ascent / scale;
    for (std::size_t i = 0; i < layouts.size(); ++i) {
        pango_layout_context_changed(layouts[i].get());
        PangoLayoutLine* line = pango_layout_get_line_readonly(layouts[i].get(), 0);
        if (line == nullptr) {
            continue;
        }
        cairo_move_to(cairo.get(), layout.glyphX[i] / scale, baseline);
        pango_cairo_show_layout_line(cairo.get(), line);
    }
    cairo_surface_flush(surface.get());
    if (cairo_status(cairo.get()) != CAIRO_STATUS_SUCCESS) {
        return makeError(Errc::Unsupported, "glyph strip: cairo could not draw the glyphs");
    }

    // Rows of the surface are `stride` bytes apart; the strip's are `width`.
    const unsigned char* pixels = cairo_image_surface_get_data(surface.get());
    const auto stride = static_cast<std::size_t>(cairo_image_surface_get_stride(surface.get()));
    strip.coverage.resize(static_cast<std::size_t>(strip.width) * strip.height);
    for (std::size_t y = 0; y < strip.height; ++y) {
        std::memcpy(strip.coverage.data() + (y * strip.width), pixels + (y * stride), strip.width);
    }
    return strip;
}

}  // namespace dvv::gpu::detail
