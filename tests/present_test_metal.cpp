// macOS only: the layer geometry rule (the GUI thread owns the CAMetalLayer's frame and
// contentsScale, nothing in dvv writes drawableSize, the driver's swapchain sets it from the
// swapchain extent, and the presenter's extent is the caller's physical PixelSize), for layer
// bounds of a fractional number of points (an odd pixel size on a 2x display).

// The build adds this file only on macOS; the guard keeps it an empty translation unit where
// clang-tidy reads every tests/*.cpp (Linux lint).
#if defined(__APPLE__)

#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

#include <objc/message.h>
#include <objc/runtime.h>

#include "dvv/gpu/presenter.h"
#include "dvv/platform/native_window.h"
#include "dvv/platform/testing/test_window.h"
#include "present_fixture.h"

namespace dvv::gpu::test {

namespace {

// CGSize and CGRect, without CoreGraphics' headers (the same layout).
struct LayerSize {
    double width;
    double height;
};
struct LayerRect {
    double x;
    double y;
    double width;
    double height;
};

struct LayerGeometry {
    LayerRect bounds;      // points
    double contentsScale;  // pixels per point
    LayerSize drawableSize;
};

LayerGeometry layerGeometry(const platform::testing::TestWindow& window) {
    const platform::NativeWindow native = window.native();
    REQUIRE(std::holds_alternative<platform::MetalWindow>(native));
    const auto layer = static_cast<id>(std::get<platform::MetalWindow>(native).layer);
    REQUIRE(layer != nil);
    // The window and its layer belong to this (the main) thread.
    return {.bounds = reinterpret_cast<LayerRect (*)(id, SEL)>(objc_msgSend)(
                layer, sel_registerName("bounds")),
            .contentsScale = reinterpret_cast<double (*)(id, SEL)>(objc_msgSend)(
                layer, sel_registerName("contentsScale")),
            .drawableSize = reinterpret_cast<LayerSize (*)(id, SEL)>(objc_msgSend)(
                layer, sel_registerName("drawableSize"))};
}

bool sameSize(LayerSize drawable, platform::PixelSize size) {
    return drawable.width == static_cast<double>(size.width) &&
           drawable.height == static_cast<double>(size.height);
}

}  // namespace

TEST_CASE_FIXTURE(PresentTest, "the swapchain sizes the layer's drawable to the caller's extent") {
    // 333x177 and 97x61 pixels are fractional point sizes on a 2x display.
    window().resize({.width = 333, .height = 177});
    const LayerGeometry before = layerGeometry(window());
    CAPTURE(before.bounds.width);
    CAPTURE(before.contentsScale);
    CAPTURE(before.drawableSize.width);
    CHECK(before.bounds.width * before.contentsScale == 333.0);
    CHECK(before.bounds.height * before.contentsScale == 177.0);

    PresenterOptions presenter_options = options();
    presenter_options.extent = window().clientSize();
    auto made = context().createPresenter(std::move(presenter_options));
    REQUIRE(made);
    const std::unique_ptr<Presenter> p = std::move(*made);

    const platform::PixelSize first{.width = 333, .height = 177};
    REQUIRE(pumpUntil(window(), [&] {
        p->requestFrame();
        return p->stats().presentedExtent == first;
    }));
    const LayerGeometry created = layerGeometry(window());
    CAPTURE(created.drawableSize.width);
    CAPTURE(created.drawableSize.height);
    CHECK(sameSize(created.drawableSize, first));

    const platform::PixelSize second{.width = 97, .height = 61};
    const std::uint64_t swapchains = p->stats().swapchainsCreated;
    window().resize(second);
    REQUIRE(window().clientSize() == second);
    p->resize(window().clientSize());
    REQUIRE(pumpUntil(window(), [&] {
        p->requestFrame();
        const PresenterStats s = p->stats();
        return s.presentedExtent == second && s.swapchainsCreated > swapchains;
    }));
    const LayerGeometry resized = layerGeometry(window());
    CAPTURE(resized.drawableSize.width);
    CAPTURE(resized.drawableSize.height);
    CHECK(resized.bounds.width * resized.contentsScale == 97.0);
    CHECK(resized.bounds.height * resized.contentsScale == 61.0);
    CHECK(sameSize(resized.drawableSize, second));
    CHECK_FALSE(p->failure());
    CHECK(errors().size() == 0);
}

}  // namespace dvv::gpu::test

#endif  // __APPLE__
