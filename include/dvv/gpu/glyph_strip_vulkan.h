#pragma once
// A glyph strip (glyph_strip.h) in a storage image on a context's device; part of dvv::gpu_text.

#include <memory>

#include "dvv/base/error.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/glyph_strip.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/submission.h"

namespace dvv::gpu {

struct GlyphStripUpload {
    // strip.width x strip.height, VK_FORMAT_R8G8B8A8_UNORM, STORAGE | TRANSFER_DST | TRANSFER_SRC
    // usage (TRANSFER_SRC lets the image be read back), named "glyph strip".
    std::shared_ptr<GpuImage> image;
    // Copies the strip into `image` (the coverage in R, G, B and A), then moves the image to
    // VK_IMAGE_LAYOUT_GENERAL. It holds the staging buffer. Move it into the caller's record
    // function and call it there once, before any dispatch that binds `image`; the submission then
    // keeps the staging buffer until the GPU has finished. A record function that refers to it
    // instead must keep it alive until the submission has been awaited.
    RecordFunction record;
};

// Creates the image and fills a host-visible staging buffer now (on the caller's thread); nothing
// is recorded until `record` runs. Errc::Internal for an empty strip or one whose coverage is not
// width x height bytes; the errors of createImage and createBuffer (context_vulkan.h) pass through,
// e.g. Errc::Unsupported for a strip larger than the device's images.
[[nodiscard]] Result<GlyphStripUpload> uploadGlyphStrip(ExecutionContext& context,
                                                        const GlyphStrip& strip);

}  // namespace dvv::gpu
