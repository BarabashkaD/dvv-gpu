# Changelog

All notable changes to dvv-gpu. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning with
the dvv rules: compatible = the same major and at least the minimum version, also for 0.x; breaking
changes only in a new major, after a deprecation. This file is the release notes (there are no
GitHub Releases).

## [0.3.0] - 2026-11-20

### Added
- `dvv/gpu/context_vulkan.h`: free `createBuffer`, `createImage`, `createComputePipeline`,
  `formatFeatures` and `submit` on an `ExecutionContext`, and `vulkanDesc`, `vulkanFormat`,
  `vulkanHandle`, `vulkanView`, `vulkanLayout` for a `GpuImage`.
- `GpuImage::name()`.
- `dvv/gpu/presenter_vulkan.h`: free `createPresenter(context, options, frames)`.
- `PresentEncoding` and `PresentFrameParams::encoding`: whether the present step applies the sRGB
  transfer function. `PresentFrameParams` names the frame provider's input. The encoding is `Srgb`
  when any component of the swapchain format is sRGB (also `A8B8G8R8_SRGB_PACK32`, whose first
  component is alpha), else `Linear`.
- `GpuBuffer::mapWrite()` and `GpuBuffer::mapRead()` with the scopes `MappedWrite` (write, then
  `flush()`) and `MappedRead` (after the invalidate).
- `GpuInstance`, `GpuInstanceOptions`, `dvv/gpu/gpu_instance.h` and
  `ExecutionContext::gpuInstance()`: the names of the process's Vulkan instance.
- `dvv/gpu/gpu_fwd.h`: declares `GpuInstance`, `ExecutionContext`, `GpuBuffer`, `GpuImage`,
  `MappedRead` and `MappedWrite` for headers that hold them by pointer. Declare `GpuInstance`
  ahead through this header, not with your own `class GpuInstance;`.
- `dvv/gpu/bundled_drivers.h`: `useBundledDrivers(manifests)` points the Vulkan loader at bundled
  driver manifests unless the user chose drivers.
- `BufferCopy` and `GroupCount` forms of `CommandRecorder::copyBuffer` and `dispatch`;
  `Presenter::show()` and `hide()`.
- dvv-gpu creates its Vulkan surfaces itself.

### Changed
- Requires dvv-platform 0.3.0 and dvv-base 0.5.0.
- The pipeline cache is written through dvv-platform's atomic replace. An existing file keeps its
  permission bits (POSIX) and its attributes (Windows); the temporary file's name has no process id.
  On Windows a busy file is still tried 5 times; on macOS and Linux the replace is tried once (it
  was tried up to 5 times). A cache path that is a symbolic link is no longer replaced: the save
  fails with an I/O error.
- Paths in pipeline-cache messages are display text: a name that has no UTF-8 form shows U+FFFD
  instead of failing.
- The Linux build of dvv-gpu needs the X11 headers (the Xlib surface code moved from dvv-platform,
  which needed them for the same reason).
- Per-OS code lives in per-OS files; the os-blocks baseline is gone.

### Deprecated (removed in 1.0.0)
- `PresentFrameInfo` (use `PresentFrameParams`) and its `format` field (use `encoding`).
- `ExecutionContext::createBuffer`, `createImage`, both `createComputePipeline`, `formatFeatures`
  and `submit` (use context_vulkan.h); `GpuImage::desc()`, `format()`, `handle()`, `view()`,
  `layout()` (use context_vulkan.h) and `GpuImage`'s public `enable_shared_from_this` base.
- `ExecutionContext::createPresenter` and `PresenterOptions::frames` (use the free
  `createPresenter`); reaching `PresentFrame` and `FrameProvider` through presenter.h (include
  presenter_vulkan.h).
- The type of `PresenterOptions::window`: it becomes `dvv::platform::WindowHandle`.
- `ResourcePool`, `ResourcePoolStats`, `ExecutionContext::createResourcePool`,
  `GpuBuffer::mapped()`, `GpuBuffer::flush()` and `GpuBuffer::invalidate()` (use the map scopes;
  frame-memory reuse belongs to the frame store).
- The five-value `copyBuffer`, the `dispatch` overloads with trailing group counts and
  `Presenter::setVisible(bool)` (use the `BufferCopy` and `GroupCount` forms, `show()` and
  `hide()`).
- `GpuRegistry`, `GpuRegistryOptions`, `dvv/gpu/gpu_registry.h` and `ExecutionContext::registry()`
  (use `GpuInstance`, `GpuInstanceOptions`, `dvv/gpu/gpu_instance.h` and `gpuInstance()`).

### Fixed
- A frame provider exception that does not derive from `std::exception` skips the frame (reported
  through onError) instead of terminating the process.
- Releasing a GPU resource never allocates, so it cannot terminate on out-of-memory.

## [0.2.1] - 2026-11-08

### Fixed
- The pipeline cache tests pass on a driver that returns a truncated cache header. Mesa's lavapipe
  26.2 (macOS CI, from Homebrew) writes the 32-byte header but reports 20 bytes
  (`lvp_GetPipelineCacheData`), so its caches never load again; dvv already rejected such data.
  The tests now ask the driver directly and, on such a driver, check only that dvv rejects the
  data instead of a round trip. No library code changed.

## [0.2.0] - 2026-11-07

### Added
- macOS (Apple Silicon) support: the Vulkan loader is opened as `@rpath/libvulkan.1.dylib`
  (resolved through the executable's rpath), presentation goes to a CAMetalLayer through
  VK_EXT_metal_surface (KosmicKrisp, Vulkan 1.4).
- macOS tests, including the layer-geometry check of the presentation surface.

### Changed
- The record, frame and error callbacks take `dvv::MoveOnlyFunction` (`std::move_only_function`
  where available).
- The swapchain image count respects surfaces whose minimum equals their maximum.
- The CMake package requires dvv-base 0.2.0 and dvv-platform 0.2.0 or later in the 0.x series.

## [0.1.0] - 2026-10-28

First versioned release.

### Added
- Vulkan 1.4 GPU core: devices, execution contexts, buffers, images and resource pools (VMA),
  headless unless a window system is asked for; the Vulkan loader is loaded at run time, never
  linked.
- Slang compute shaders compiled at build time by the pinned `slangc` and embedded in the program
  (`dvv_add_shaders()`), compute pipelines and the pipeline cache.
- Presentation to native windows (`Presenter`) and device-loss handling with fault injection for
  tests.
- The `dvv-gpu-info` tool.
- The CMake package `dvv_gpu` (`find_package(dvv_gpu 0.1)`, `SameMajorVersion`, `dvv_add_shaders()`
  included); it requires dvv-base 0.1.0 and dvv-platform 0.1.0 or later in the 0.x series.
