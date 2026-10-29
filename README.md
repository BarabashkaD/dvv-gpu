# dvv-gpu

Vulkan 1.4 GPU core of the dvv stack used by rawreel, headless unless a window system is asked
for. C++23; depends on [dvv-base](https://github.com/BarabashkaD/dvv-base),
[dvv-platform](https://github.com/BarabashkaD/dvv-platform) and on the pinned Vulkan-Headers,
VulkanMemoryAllocator and vk-bootstrap from the dvv-build third-party prefix. Public headers use
the Vulkan C types only. The Vulkan loader is loaded at run time, never linked; `vulkan-1.dll` is
searched for only in the Windows system directory.

## Build

Builds and tests on Windows (Visual Studio 2026) and Linux (Clang 19 with libstdc++ 14) with
[dvv-build](https://github.com/BarabashkaD/dvv-build), which provides the presets, the shared CMake
modules, the third-party dependencies (including `slangc`) and the CI workflow; its README lists
the prerequisites. dvv-build is cloned next to dvv-gpu at the version of the `dvv-build` line in
`dvv-deps.txt`; `fetch-deps.cmake` clones dvv-base and dvv-platform next to them at their minimum
versions from the same file:

```powershell
git clone https://github.com/BarabashkaD/dvv-gpu
git clone --branch v0.1.1 https://github.com/BarabashkaD/dvv-build
python -m pip install -r dvv-build/tools/requirements.txt
cmake -DREPO=dvv-gpu -P dvv-build/tools/fetch-deps.cmake
./dvv-build/bootstrap.ps1
cd dvv-gpu
cmake --workflow --preset vs2026
```

On Linux: `./dvv-build/bootstrap.sh`, then `cmake --workflow --preset ninja-clang`.
Applications use dvv-gpu as a git submodule in their `libs/` folder, next to dvv-build.

## Shaders

Compute shaders are Slang files compiled at build time by the pinned `slangc` of the third-party
prefix and embedded in the program:

```cmake
find_package(dvv_gpu 0.1 REQUIRED)   # or dvv_gpu in-tree
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

## Compute pipelines and the pipeline cache

`ctx->createComputePipeline(my_shaders::addOne(), sizeof(AddOne))` makes a pipeline with one
push-constant range at offset 0 (none for size 0). Shaders reach buffers through their device
addresses, passed in the push constants. The C++ push-constant struct (`AddOne` here) is written by
hand; its size must be a multiple of 4 and cover the shader's push-constant block.
`recorder.dispatch(pipeline, std::as_bytes(std::span(&push, 1)), groups)` takes exactly that many
bytes.

The command recorder puts a global memory barrier before every work command it records and before
`raw()`. Commands recorded through `raw()` must order themselves among each other and bring their
own barriers for stages other than compute and transfer.

Host-visible buffers are not flushed or invalidated automatically: call `flush()` after writing a
mapped buffer and before the GPU reads it, and `invalidate()` after the GPU work completed and
before reading it.

`ResourcePool::acquireBuffer()` returns a lease; a released buffer is handed out again only after
the GPU has finished every submission made before its return. Keep the lease alive until the
submission that uses the buffer has completed.

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

One shader then serves R8_UINT, R16_UINT and other formats; check `ctx->formatFeatures(format)` for
`VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT` before relying on it for a format.

- **Binding:** `ctx->createComputePipeline(shader, sizeof(Push), {.storageImages = n})` makes a
  pipeline whose shader must declare exactly set 0, bindings `0..n-1` (at most the device's
  `maxPushDescriptors`). `recorder.dispatch(pipeline, push, images, groups_x, groups_y)` binds
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
  2D view of the whole image (`GpuImage::view()`; `VK_NULL_HANDLE` otherwise).
- **Required features:** `pushDescriptor`, `shaderStorageImageExtendedFormats`,
  `shaderStorageImageWriteWithoutFormat`, `shaderStorageImageReadWithoutFormat` and the six
  descriptor-indexing features; `dvv-gpu-info` lists a device without them as not usable.

## Presentation

`GpuRegistry::create({.windowSystem = dvv::platform::defaultWindowSystem()})` also enables the
surface extensions on the instance (`Errc::Unsupported` when the loader lacks them); without
`windowSystem` the registry stays headless. A context created with
`ContextOptions{.presentation = true}` enables `VK_KHR_swapchain` on its device: `Errc::Internal` on
a headless registry, `Errc::Unsupported` for a device without the extension
(`GpuInfo::optional.swapchain`).

`ExecutionContext::createPresenter(PresenterOptions)` returns a `std::unique_ptr<Presenter>` for
one native window. The presenter keeps its context alive and runs its own thread: it calls the frame
provider, submits the frame's commands, blits the frame's image into the swapchain image and
presents (FIFO). The provider renders into its own image (`TRANSFER_SRC` usage, a format with
`BLIT_SRC`, e.g. `R8G8B8A8_UNORM`); the blit scales it when the sizes differ.

```cpp
using namespace dvv::gpu;
auto presenter = context->createPresenter({
    .window = dvv::platform::Win32Window{.hinstance = instance, .hwnd = hwnd},
    .extent = {.width = 1280, .height = 720},  // client size in physical pixels (a hint)
    .frames = [context](const PresentFrameInfo& info) -> dvv::Result<PresentFrame> {
        DVV_TRY(auto image, context->createImage({info.extent.width, info.extent.height,
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT}));
        return PresentFrame{.record = [image](CommandRecorder& r) { /* render into *image */ },
                            .image = image};
    },
    .onError = [](const dvv::Error& error) noexcept { /* forward to the GUI thread */ }});
```

A real provider keeps its image while the extent stays the same
(`PresentFrameInfo::previousFrameCompleted` is false after a dropped frame).

- **Threads.** `resize()`, `setVisible()`, `requestFrame()` and `stats()` may be called from any
  thread and never block; `resize()` is a hint. The provider and `onError` run on the presenter
  thread; they must never wait for the window's thread and never destroy the presenter. Create and
  destroy the presenter on the window's thread, and do not call `ExecutionContext::drain()` there
  while a presenter lives.
- **One frame in flight.** The provider is called again only after the previous frame's submission
  has completed, so it can reuse one image. While hidden (`setVisible(false)`, a minimised window or
  a 0×0 surface) nothing is rendered.
- **Destroy it before the window.** `~Presenter` waits for the frame in flight and for queue idle.
  A lost surface or device stops the presenter: `stats().failed`, `failure()` and one `onError`
  call; replacing the presenter is the recovery.

## Device loss

A context enters a permanent lost state (`ExecutionContext::lost()`) when one of its Vulkan calls
returns `VK_ERROR_DEVICE_LOST` or when `dvv::gpu::testing::injectDeviceLost(context)` is called.
The first cause is logged once (Warn).

- Submissions that complete after the loss, and all later ones, complete with `Errc::DeviceLost`
  ("the device was lost", or "the device was lost (injected)"). `createBuffer`, `createImage`,
  `createResourcePool`, `createComputePipeline` and `createPresenter` return `DeviceLost`. A
  presenter fails with `DeviceLost` at its next frame (`failure()`, one `onError`).
- Waits end once the device is lost, so a silent driver cannot hang them. A lost context does not
  save the pipeline cache.
- A lost context stays lost; the application destroys it and creates a new one.
- `dvv/gpu/testing/fault_injection.h` (part of `dvv_gpu`, for tests and `rawreel-gui --selftest
  lost`) is thread-safe and idempotent; the real device stays healthy.

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
