# dvv-gpu

Vulkan 1.4 GPU core of the dvv stack, headless unless a window system is asked for. C++23;
depends on [dvv-base](https://github.com/BarabashkaD/dvv-base),
[dvv-platform](https://github.com/BarabashkaD/dvv-platform) and on the pinned Vulkan-Headers,
VulkanMemoryAllocator and vk-bootstrap from the dvv-build third-party prefix. Public headers use
the Vulkan C types only. The Vulkan loader is loaded at run time, never linked; `vulkan-1.dll` is
searched for only in the Windows system directory.

Headers. The app-facing headers are gpu_instance.h, execution_context.h, presenter.h, image.h,
gpu_info.h, gpu_info_json.h, bundled_drivers.h, glyph_strip.h and testing/fault_injection.h; they
include no Vulkan header. The Vulkan entry points are in context_vulkan.h, presenter_vulkan.h and
glyph_strip_vulkan.h (`dvv::gpu_text`), which include resources.h, command_recorder.h,
compute_pipeline.h and submission.h. Headers that only hold dvv-gpu objects by pointer or
reference include gpu_fwd.h.

## Build

Builds and tests on Windows (Visual Studio 2026), Linux (Clang 19 with libstdc++ 14) and macOS
(Apple Silicon) with
[dvv-build](https://github.com/BarabashkaD/dvv-build), which provides the presets, the shared CMake
modules, the third-party dependencies (including `slangc`) and the CI workflow; its README lists
the prerequisites. dvv-build is cloned next to dvv-gpu at the version of the `dvv-build` line in
`dvv-deps.txt`; `fetch-deps.cmake` clones dvv-base and dvv-platform next to them at their minimum
versions from the same file:

```powershell
git clone https://github.com/BarabashkaD/dvv-gpu
git clone --branch v0.6.0 https://github.com/BarabashkaD/dvv-build
python -m pip install -r dvv-build/tools/requirements.txt
cmake -DREPO=dvv-gpu -P dvv-build/tools/fetch-deps.cmake
./dvv-build/bootstrap.ps1
cd dvv-gpu
cmake --workflow --preset vs2026
```

On Linux: `./dvv-build/bootstrap.sh`, then `cmake --workflow --preset ninja-clang`. On macOS:
`./dvv-build/bootstrap.sh`, then `cmake --workflow --preset ninja-macos`.
Applications use dvv-gpu as a git submodule in their `libs/` folder, next to dvv-build.

## Quick start

A GPU instance, a context on the best device, two buffers and one awaited submission:

```cpp
#include <cstddef>
#include <cstring>
#include <optional>

#include "dvv/base/executors.h"
#include "dvv/base/task.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/gpu_instance.h"

// Fills a buffer on the host, copies it on the GPU and reads the copy back.
dvv::Task<dvv::Result<void>> copyOnGpu(dvv::gpu::ExecutionContext& context,
                                       dvv::IExecutor& executor) {
    auto upload = co_try(createBuffer(context, {.size = 256,
                                                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                .memory = dvv::gpu::MemoryUsage::HostUpload}));
    auto readback = co_try(createBuffer(context, {.size = 256,
                                                  .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                  .memory = dvv::gpu::MemoryUsage::HostReadback}));
    dvv::gpu::MappedWrite write = upload->mapWrite();
    std::memset(write.bytes().data(), 7, write.bytes().size());
    co_try(write.flush());
    co_try(co_await submit(context, executor,
                           [upload, readback](dvv::gpu::CommandRecorder& recorder) {
                               recorder.copyBuffer(*upload, *readback,
                                                   dvv::gpu::BufferCopy{.size = 256});
                           }));
    const dvv::gpu::MappedRead read = co_try(readback->mapRead());
    if (read.bytes()[255] != std::byte{7}) {
        co_return dvv::makeError(dvv::Errc::Internal, "the copy differs");
    }
    co_return dvv::Result<void>{};
}

int main() {
    auto instance = dvv::gpu::GpuInstance::create({.applicationName = "demo"});
    if (!instance) {
        return 1;  // instance.error().message says why, e.g. "Vulkan loader not found"
    }
    const std::optional<dvv::gpu::DeviceId> device = (*instance)->bestDevice();
    if (!device) {
        return 1;  // no device meets the requirements (see dvv-gpu-info)
    }
    auto context = (*instance)->createContext(*device);
    auto pool = dvv::ExecutorFactory::makeThreadPool(1);
    if (!context || !pool) {
        return 1;
    }
    return dvv::syncWait(copyOnGpu(**context, **pool)) ? 0 : 1;
}
```

`createBuffer` and `submit` are found through their `ExecutionContext` argument. The record function
runs on the submit thread; capturing the buffers keeps them alive until the GPU has finished the
copy. Call `flush()` on the write scope before the submission, and `mapRead()` after it completed.
`syncWait` blocks, so it belongs in `main` or a test; in a coroutine, `co_await` the task.

## Shaders

Compute shaders are Slang files compiled at build time by the pinned `slangc` of the third-party
prefix and embedded in the program:

```cmake
find_package(dvv_gpu 0.5 REQUIRED)   # or dvv_gpu in-tree
dvv_add_shaders(my_target NAMESPACE my_shaders SHADERS shaders/add_one.slang)
```

`#include "my_shaders/add_one.h"` then gives `const dvv::gpu::EmbeddedShader& my_shaders::addOne()`:
the SPIR-V words and slangc's reflection JSON. The namespace defaults to `<target>_shaders`. Each
shader has one compute entry point `main`. Configuring fails with a clear message when `slangc` is
missing, cannot run or is not the pinned version, and when two shaders of a target give the same
function name. The target must use the keyword form of `target_link_libraries()`; the generated
headers are not installed.

The Slang version is pinned in `cmake/DvvShaders.cmake` (`DVV_SLANGC_VERSION`) and in
dvv-build's `deps/versions.cmake`; update both together.

### Shader imports

A library shares Slang sources with the shaders of other targets:

```cmake
dvv_export_shader_sources(my_lib SOURCES shaders/line_math.slang)          # namespace my_lib
dvv_add_shaders(my_app SHADERS shaders/draw.slang IMPORTS my::lib)         # an alias is fine
```

`draw.slang` then says `import my_lib.line_math;`. The namespace defaults to the target name as a C
identifier (`NAMESPACE` sets another). The sources are copied into the build tree and installed to
`<datadir>/<target>/shaders/<namespace>` (`<datadir>` is `CMAKE_INSTALL_DATADIR` at the target's
first export call); a changed source rebuilds the shaders that import it. Several export calls for
one target add up, also from other directories; make them all before the `dvv_add_shaders` call
that imports the target. Configuring fails when an import is not a target, exports nothing, or
when two imports export the same `<namespace>/<file>`.

dvv-gpu owns the install folder: the exported target carries it in `DVV_SHADER_INSTALL_DIR`. An
installed package's config file passes its prefix, after its targets file:

```cmake
find_dependency(dvv_gpu 0.5)   # gives dvv_import_shader_sources()
include("${CMAKE_CURRENT_LIST_DIR}/my_libTargets.cmake")
dvv_import_shader_sources(my::lib PREFIX "${PACKAGE_PREFIX_DIR}")
```

## Compute pipelines and the pipeline cache

`createComputePipeline(*ctx, my_shaders::addOne(), sizeof(AddOne))` makes a pipeline with one
push-constant range at offset 0 (none for size 0). Shaders reach buffers through their device
addresses, passed in the push constants. The C++ push-constant struct (`AddOne` here) is written by
hand; its size must be a multiple of 4 and cover the shader's push-constant block.
`recorder.dispatch(pipeline, std::as_bytes(std::span(&push, 1)), groups)` takes exactly that many
bytes.

The recorder also fills buffers (`fillBuffer`), copies between buffers (`copyBuffer`), between
buffers and images (`copyBufferToImage`, `copyImageToBuffer`) and moves images between layouts
(`transition`).

The command recorder puts a global memory barrier before every work command it records and before
`raw()`. Commands recorded through `raw()` must order themselves among each other and bring their
own barriers for stages other than compute and transfer.

A host-visible buffer is written through `mapWrite()` and read through `mapRead()`. Call `flush()`
on the write scope after writing and before the GPU reads; `mapRead()` makes the GPU's writes
visible, so call it after the work completed. `mapWrite()` needs a non-const buffer.

`ContextOptions::pipelineCacheFile` names the pipeline-cache file (empty: in memory only);
`defaultPipelineCacheFile(app, device)` gives one file per device in the user's cache directory. A
damaged or foreign file is ignored with a log line. The cache is saved atomically by
`savePipelineCache()` and, if it changed, when the context closes.

## Images in shaders

Compute shaders use storage images bound through push descriptors, declared in set 0, one binding
each, without a format:

```slang
[[vk::binding(0, 0)]] [format("unknown")] RWTexture2D<uint> image;
```

One shader then serves R8_UINT, R16_UINT and other formats; check `formatFeatures(*ctx, format)` for
`VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT` before relying on it for a format.

- **Binding:** `createComputePipeline(*ctx, shader, sizeof(Push), {.storageImages = n})` makes a
  pipeline whose shader must declare exactly set 0, bindings `0..n-1` (at most the device's
  `maxPushDescriptors`). `recorder.dispatch(pipeline, push, images, {groups_x, groups_y})` binds
  exactly `n` images, binding `i` = `images[i]`; each has `VK_IMAGE_USAGE_STORAGE_BIT` and appears
  once.
- **Layout:** a bound image is used in `GENERAL`; the dispatch's barrier moves it there (an image
  in `UNDEFINED` loses its contents).
- **Errors:** too many images or other reflection bindings make `createComputePipeline` fail with
  `Errc::Internal` (checked when the shader has reflection JSON; unreadable reflection JSON is
  `Errc::Format`). A wrong image count, an image of another device, one without `STORAGE` usage or
  one given twice fails the recorder (`Errc::Internal`).
- **Images and views:** `createImage` accepts `VK_IMAGE_USAGE_STORAGE_BIT` when the format supports
  `STORAGE_IMAGE` (`Errc::Unsupported` otherwise). An image with `STORAGE` or `SAMPLED` usage has a
  2D view of the whole image (`vulkanView(*image)`; `VK_NULL_HANDLE` otherwise).
- **Required features:** `pushDescriptor`, `shaderStorageImageExtendedFormats`,
  `shaderStorageImageWriteWithoutFormat`, `shaderStorageImageReadWithoutFormat` and the six
  descriptor-indexing features; `dvv-gpu-info` lists a device without them as not usable.

## Glyph strips

`dvv::gpu_text` (a separate target) draws short text, such as ruler numbers, for shaders to
compose. It takes two steps:

1. **Raster** (glyph_strip.h, no GPU): `rasterGlyphStrip(font, scale, chars)` draws 1 to 64
   printable ASCII characters with an OS font into a `GlyphStrip`: one row of equal cells (the
   widest glyph's advance), each glyph centred in its cell, white coverage (0..255) on black.
   `ascent`, `descent` and `advance` are whole pixels; the baseline is the top edge of row
   `ascent`.
2. **Upload** (glyph_strip_vulkan.h): `uploadGlyphStrip(*ctx, strip)` creates the strip's image
   (`VK_FORMAT_R8G8B8A8_UNORM`, the coverage in every channel, `STORAGE`, `TRANSFER_DST` and
   `TRANSFER_SRC` usage) and fills a staging buffer at once. Its `record` function copies the strip
   in and leaves the image in `GENERAL`. Move it into your own record function and call it there
   once, before the dispatch that binds the image; the submission then keeps the staging buffer
   until the GPU has finished.

Shaders read the strip through the exported module `dvv_gpu_text.gpu_text` (gpu_text.slang, namespace
`dvv::gpu`, dvv-gpu 0.7 or later), which `dvv_add_shaders(... IMPORTS dvv::gpu_text)` takes:

```slang
import dvv_gpu_text.gpu_text;
using namespace dvv::gpu;  // or dvv::gpu::decimalCoverage(...)
```

`glyphCells(strip, cellCount)` gives the cell grid (`GlyphCells`: `advance` and `height`; both 0
without a usable grid), `cellCoverage` the coverage of one cell at a pixel, `decimalDigits` and
`decimalDigitAt` the digits of an unsigned value, and `decimalCoverage` the coverage of a value
drawn in decimal from the strip's digit cells. The strip does not hold its cell count: the caller
passes it.

Units: `FontSpec::sizePoints` is in the OS's font points and `scale` is output pixels per point.
The font keeps the design of its point size (optical sizes) and is drawn `scale` times larger, so
a strip at scale 2 shows the scale-1 design at twice the resolution. Size x scale is at most 4096
pixels, and a strip at most `MAX_GLYPH_STRIP_SIDE` (16384) pixels wide and tall; larger requests
are `Errc::Internal` on every OS.

The family `""` is the OS's UI font: the system font on macOS (Core Text), the message font of
the Windows settings (DirectWrite), and `Sans` on Linux, which fontconfig maps to the desktop's
sans font (Pango). A family the OS does not know falls back to an OS font; `GlyphStrip::family`
names the one it drew with. On Windows `FontSpec::family` is a GDI face name, as font lists show
it, and the family drawn with can differ from it without a fallback: "Segoe UI Semibold" draws
with "Segoe UI".

Only `dvv::gpu_text` links the OS text libraries: Core Text, Core Graphics and Core Foundation on
macOS, DirectWrite and user32 on Windows, pangocairo 1.44 or later (pkg-config) on Linux. Taken
in-tree (FetchContent or add_subdirectory) on Linux, dvv-gpu defines `dvv::gpu_text` only where
pkg-config finds pangocairo, so the users of `dvv::gpu` alone need no Pango; dvv-gpu's own build
and its tests require it. An installed package loads it as a component (dvv-gpu 0.6 or later):

```cmake
find_package(dvv_gpu 0.6 REQUIRED COMPONENTS text)
target_link_libraries(my_target PRIVATE dvv::gpu_text)
```

`rasterGlyphStrip` may be called from any thread, also at the same time.

On Linux, LeakSanitizer and ThreadSanitizer report the system font libraries that Pango uses, not
dvv code: fontconfig's configuration as a leak (a false positive, fixed upstream but in no release
yet), and races on Pango's worker thread, whose GLib locks the sanitizer cannot see. A consumer's
sanitizer builds that draw glyph strips get the same reports and need the same suppressions.
dvv-gpu's sanitizer presets suppress them for its text tests only; the entries and their reasons
are in tests/text_lsan.supp and tests/text_tsan.supp (not installed). Reports with no frame in
these libraries still fail the tests.

## Presentation

`GpuInstance::create({.windowSystem = dvv::platform::defaultWindowSystem()})` also enables the
surface extensions on the instance (`Errc::Unsupported` when the loader lacks them); without
`windowSystem` the instance stays headless. A context created with
`ContextOptions{.presentation = true}` enables `VK_KHR_swapchain` on its device: `Errc::Internal` on
a headless instance, `Errc::Unsupported` for a device without the extension
(`GpuInfo::optional.swapchain`).

`createPresenter(*context, options, frames)` (presenter_vulkan.h) returns a
`std::unique_ptr<Presenter>` for one native window. The presenter keeps its context alive and runs
its own thread: it calls the frame provider, submits the frame's commands, blits the frame's image
into the swapchain image and presents (FIFO). The provider renders into its own image
(`TRANSFER_SRC` usage, a format with `BLIT_SRC`, e.g. `R8G8B8A8_UNORM`); the blit scales it when
the sizes differ.

```cpp
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/presenter_vulkan.h"
#include "dvv/platform/window_handle_win32.h"
// createPresenter and createImage are found through their ExecutionContext argument.
auto presenter = createPresenter(*context,
    {.window = dvv::platform::windowHandleOf(dvv::platform::Win32Window{
         .hinstance = instance, .hwnd = hwnd}),
     .extent = {.width = 1280, .height = 720},  // client size in physical pixels (a hint)
     .onError = [](const dvv::Error& /*error*/) noexcept { /* forward to the GUI thread */ }},
    [context](const dvv::gpu::PresentFrameParams& params)
        -> dvv::Result<dvv::gpu::PresentFrame> {
        DVV_TRY(auto image, createImage(*context, {.width = params.extent.width,
                                                   .height = params.extent.height,
                                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                                   .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT}));
        return dvv::gpu::PresentFrame{
            .record = [image](dvv::gpu::CommandRecorder& /*recorder*/) { /* render */ },
            .image = image};
    });
```

`params.encoding` says whether the present step applies the sRGB transfer function:
`PresentEncoding::Srgb` means the image holds linear values, `Linear` means it holds
display-encoded values. The values are those the blit reads through the image's format: a `*_SRGB`
image is decoded to linear first, so with `Linear` the window shows linear values. To control the
encoding, render into a `UNORM` (or float) image.

A real provider keeps its image while the extent stays the same
(`PresentFrameParams::previousFrameCompleted` is false after a dropped frame).

- **Threads.** `resize()`, `show()`, `hide()`, `requestFrame()` and `stats()` may be called from
  any thread and never block; `resize()` is a hint. The provider and `onError` run on the presenter
  thread; they must never wait for the window's thread and never destroy the presenter. Create and
  destroy the presenter on the window's thread, and do not call `ExecutionContext::drain()` there
  while a presenter lives.
- **Completion callback.** `PresentFrame::completed` (optional) is called once on the presenter
  thread with the submission's result (the timeline value, or the error) after the frame's
  submission has finished and before the next provider call; a frame dropped before its submission
  is destroyed without a call. The provider's rules apply: it must not block long, wait for the
  window's thread or destroy the presenter.
- **Resize.** A swapchain recreate does not wait for queue idle: the old swapchain is destroyed
  once the first frame on the new one has completed, so presenters on one context do not stall each
  other.
- **One frame in flight.** The provider is called again only after the previous frame's submission
  has completed, so it can reuse one image. While hidden (`hide()`, a minimised window or a 0×0
  surface) nothing is rendered.
- **Destroy it before the window.** `~Presenter` waits for the frame in flight and for queue idle.
  A lost surface or device stops the presenter: `stats().failed`, `failure()` and one `onError`
  call; replacing the presenter is the recovery.

## Device loss

A context enters a permanent lost state (`ExecutionContext::lost()`) when one of its Vulkan calls
returns `VK_ERROR_DEVICE_LOST` or when `dvv::gpu::testing::injectDeviceLost(context)` is called.
The first cause is logged once (Warn).

- Submissions that complete after the loss, and all later ones, complete with `Errc::DeviceLost`
  ("the device was lost", or "the device was lost (injected)"). `createBuffer`, `createImage`
  and `createComputePipeline` (context_vulkan.h) and `createPresenter` (presenter_vulkan.h)
  return `DeviceLost`. A presenter fails with `DeviceLost` at its next frame
  (`failure()`, one `onError`).
- Waits end once the device is lost, so a silent driver cannot hang them. A lost context does not
  save the pipeline cache.
- A lost context stays lost; the application destroys it and creates a new one.
- `dvv/gpu/testing/fault_injection.h` (part of `dvv_gpu`, for tests and an application's
  self-test of device loss) is thread-safe and idempotent; the real device stays healthy.

## Bundled drivers

An application that ships its own Vulkan driver calls `useBundledDrivers(manifests)`
(bundled_drivers.h) as the first statement of `main`, before any thread starts and before
`GpuInstance::create`. It sets `VK_DRIVER_FILES` to the manifests that exist, unless
`VK_DRIVER_FILES` or `VK_ICD_FILENAMES` is already set and not empty, and returns what it did:
`DriverChoice::Bundled`, `UserChoice` (the user's variable is left as it is) or `NoManifest`
(nothing set).

- **Errors.** `Errc::Unsupported`, and nothing set, when an existing manifest's path contains the list
  separator (`;` on Windows, `:` elsewhere): the loader would split the value there. On macOS a `/`
  typed in a Finder folder name is a `:` in the path. `Errc::Internal` when the variable cannot be
  set.
- **Process-wide.** The variable is part of the process environment: child processes inherit it
  and use only the bundled driver. A second call in the same process returns `UserChoice`, because
  the first call's value is now set.
- **Paths.** Pass absolute paths. The loader reads the variable when the instance is created and
  resolves a relative path against the current directory at that time.
- The Vulkan loader ignores the variable in an elevated process (a Windows administrator, a setuid
  program on Linux).

## Tests and dvv-gpu-info

CTest tests with the label `gpu`: `dvv_gpu.gpu` (the device tests; the compute demo is
`tests/gpu_test_compute.cpp`), `dvv_gpu.gpu_info` (checks `dvv-gpu-info`) and `dvv_gpu.present`
(also `display`; uses dvv-platform's `TestWindow`). Without a usable Vulkan 1.4 device or display
they exit with 77, which CTest reports as skipped. CTest uses the validation layer of the
third-party prefix, and every test fails on a validation message. Environment:

- `DVV_REQUIRE_GPU=1`: a missing device or display fails the tests instead of skipping them.
- `DVV_VK_VALIDATION=off|on|sync`: validation mode; default `on` in Debug when the layer is present.
- `DVV_VK_VALIDATION_REQUIRED=1`: a missing validation layer is an error.

```powershell
$env:DVV_REQUIRE_GPU='1'; $env:DVV_VK_VALIDATION='sync'; $env:DVV_VK_VALIDATION_REQUIRED='1'
ctest --preset vs2026 -L gpu
```

`dvv-gpu-info [--json]` prints the loader version, the validation mode and every device: name, type,
Vulkan and driver versions, device-local memory, missing requirements and optional features. The
device that would be selected is marked. Exit codes: 0 a usable device exists, 1 bad arguments, 4
GPU error, e.g. no loader or no usable device. To use the prefix's validation layer outside CTest,
set `VK_ADD_LAYER_PATH` to `<prefix>/bin`.

Licence: MIT.
