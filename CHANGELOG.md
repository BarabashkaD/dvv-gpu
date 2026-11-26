# Changelog

All notable changes to dvv-gpu. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning with
the dvv rules: compatible = the same 0.minor and at least the minimum version; a replaced or
unused API is removed in the next 0.x minor, with no deprecation period. This file is the release
notes (there are no GitHub Releases).

## [0.7.0] - 2026-11-26

### Added
- `shaders/gpu_text.slang`, the exported shader module of `dvv::gpu_text` (the `text` component):
  `import dvv_gpu_text.gpu_text;`, in namespace `dvv::gpu`. It reads a glyph strip's storage image
  (as `uploadGlyphStrip` makes it): `GlyphCells` and `glyphCells(strip, cellCount)`,
  `cellCoverage`, `decimalDigits`, `decimalDigitAt` and `decimalCoverage`, which draws a
  non-negative integer in decimal from the strip's digit cells.
- `PresentFrame::completed` (may be empty): called once on the presenter thread with the
  submission's result (the timeline value, or the error) after the frame's submission finished and
  before the next provider call. A frame dropped before its submission destroys it without a call.

### Fixed
- A swapchain recreate no longer waits for the queue to be idle: the old swapchain is retired and
  destroyed once the first frame on the new one has completed (or in the presenter's teardown), so
  resizing one presenter does not stall other presenters or submitters on the same context.

### Changed
- The dvv-build minimum is 0.6.1.

## [0.6.0] - 2026-11-25

### Added
- `dvv::gpu_text`, a separate library (the installed package's `text` component:
  `find_package(dvv_gpu COMPONENTS text)`): glyph strips drawn with the OS's fonts, Core Text on
  macOS, DirectWrite on Windows and Pango (pangocairo 1.44 or later) on Linux. Only this target
  links the OS text libraries; taken in-tree on Linux, it is defined only where pkg-config finds
  pangocairo. An install without the component has no text headers.
- `dvv/gpu/glyph_strip.h`: `FontSpec`, `GlyphStrip`, `MAX_GLYPH_STRIP_SIDE` and
  `rasterGlyphStrip(font, scale, chars)`: up to 64 printable ASCII characters in equal cells,
  white coverage on black, in output pixels (`scale` pixels per OS font point), from any thread.
- `dvv/gpu/glyph_strip_vulkan.h`: `uploadGlyphStrip(context, strip)` and `GlyphStripUpload`: an
  RGBA8 storage image of the strip and the record function that copies it in, holding its staging
  buffer until the submission completes.
- Linux sanitizer suppressions for the text tests `dvv_gpu.text_unit` and `dvv_gpu.text_gpu`
  only: under LeakSanitizer and ThreadSanitizer, `rasterGlyphStrip` gets reports from the system
  font libraries, not from dvv code (a false positive leak of fontconfig's configuration, and
  races on Pango's worker thread, whose GLib locks the sanitizer cannot see). A consumer's
  sanitizer builds get them too and need the same suppressions; the entries and their reasons are
  in tests/text_lsan.supp and tests/text_tsan.supp (not installed). Reports with no frame in these
  libraries still fail the tests.

### Changed
- dvv-gpu's own Linux build and its tests now require pangocairo 1.44 or later (pkg-config).

## [0.5.0] - 2026-11-24

### Added
- `dvv_export_shader_sources(<target> [NAMESPACE <namespace>] SOURCES <file.slang>...)`: exports
  Slang sources for the shaders of other targets (`import <namespace>.<stem>;`), in the build tree
  and installed; the target property `DVV_SHADER_IMPORT_DIR` names their folder, and the exported
  target property `DVV_SHADER_INSTALL_DIR` their install folder.
- `dvv_import_shader_sources(<target> PREFIX <package prefix>)`: in an installed package's config
  file, points the imported target at its installed sources.
- `dvv_add_shaders(... IMPORTS <target>...)`: compiles the shaders with the sources the imports
  export; a changed imported source rebuilds them.

## [0.4.0] - 2026-11-22

### Removed
- `GpuRegistry`, `GpuRegistryOptions`, `dvv/gpu/gpu_registry.h` and `ExecutionContext::registry()`.
  `GpuInstance`, `GpuInstanceOptions`, `ValidationMode` and `validationModeName` are declared in
  `dvv/gpu/gpu_instance.h`; `dvv/gpu/gpu_fwd.h` declares `class GpuInstance;`.
- The `ExecutionContext` members `createBuffer`, `createImage`, both `createComputePipeline`,
  `formatFeatures`, `submit`, `createPresenter` and `createResourcePool`. The free functions of
  context_vulkan.h and presenter_vulkan.h are the only creators.
- `ResourcePool`, `ResourcePoolStats` and `dvv/gpu/resource_pool.h`.
- `ExecutionContext::retireStats()` and `RetireStats`.
- `PresentFrameInfo` and its `format` field; `PresenterOptions::frames`; `Presenter::setVisible`.
- `GpuBuffer::mapped()`, `flush()` and `invalidate()`: use `mapWrite()` with `MappedWrite::flush()`,
  and `mapRead()`.
- `GpuImage::desc()`, `format()`, `handle()`, `view()` and `layout()` (use `vulkanDesc`,
  `vulkanFormat`, `vulkanHandle`, `vulkanView` and `vulkanLayout` in context_vulkan.h) and its
  public `enable_shared_from_this` base.
- The five-value `CommandRecorder::copyBuffer` and the `dispatch` overloads with trailing group
  counts: use the `BufferCopy` and `GroupCount` forms.

### Changed
- Requires dvv-build 0.6.0, dvv-base 0.6.0 and dvv-platform 0.4.0.
- `PresenterOptions::window` is a `dvv::platform::WindowHandle`, built with `windowHandleOf()`
  of the window system's header, e.g. `windowHandleOf(Win32Window{...})`.
- `GpuImage` is declared in `dvv/gpu/image.h`. `PresentFrame` and `FrameProvider` are declared in
  presenter_vulkan.h.
- gpu_instance.h, execution_context.h, presenter.h and image.h include no Vulkan header, and no
  longer include resources.h, submission.h, compute_pipeline.h, command_recorder.h or
  embedded_shader.h. execution_context.h, and gpu_instance.h through it, also no longer include
  presenter.h, image.h, dvv/base/executor.h, dvv/base/move_only_function.h,
  dvv/platform/pixel_size.h, `<atomic>`, `<coroutine>`, `<cstddef>` or `<utility>`;
  execution_context.h also drops dvv/platform/window_handle.h and `<optional>`. Code includes the
  header that declares each name it uses: code that names `GpuBuffer`, `BufferDesc`, `ImageDesc`,
  `GpuImage`, `RecordFunction`, `SubmitOperation`, `ComputePipeline`, `CommandRecorder`,
  `EmbeddedShader`, `Presenter`, `PresenterOptions`, `PresenterStats`, `PresentFrameParams`,
  `PresentEncoding`, `IExecutor` or a Vulkan type includes the header that declares it, or
  context_vulkan.h or presenter_vulkan.h.
- `PresentFrameParams` has four fields (`extent`, `frameNumber`, `previousFrameCompleted`,
  `encoding`), so a structured binding of it needs four names.
- Leak reports and live-object lists name the GPU instance `dvv::gpu::GpuInstance` (was
  `dvv::gpu::GpuRegistry`).
- The CMake package accepts a request for the same 0.minor only (`SameMinorVersion`):
  `find_package(dvv_gpu 0.4)`.

## [0.3.0] - 2026-11-21

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
  driver manifests unless the user chose drivers, and returns a `DriverChoice`. A manifest whose
  path contains the list separator (`;` on Windows, `:` elsewhere) gives `Errc::Unsupported`: the
  loader would split the value there.
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
- `PresentFrameInfo` (`PresentFrameParams`) has a fifth field, `encoding`, so a structured binding
  of it needs five names.

### Deprecated (removed in 1.0.0)
- `PresentFrameInfo` (use `PresentFrameParams`) and its `format` field (use `encoding`).
- `ExecutionContext::createBuffer`, `createImage`, both `createComputePipeline`, `formatFeatures`
  and `submit` (use the free functions of the same names in context_vulkan.h, which take the
  context first).
- `GpuImage::desc()`, `format()`, `handle()`, `view()` and `layout()` (use `vulkanDesc`,
  `vulkanFormat`, `vulkanHandle`, `vulkanView` and `vulkanLayout` in context_vulkan.h) and
  `GpuImage`'s public `enable_shared_from_this` base (hold the image's `shared_ptr`).
- `ExecutionContext::createPresenter` and `PresenterOptions::frames` (use the free
  `createPresenter(context, options, frames)` in presenter_vulkan.h); reaching `PresentFrame` and
  `FrameProvider` through presenter.h (include presenter_vulkan.h).
- The type of `PresenterOptions::window`: nothing changes in 0.3.x; 1.0.0 makes it a
  `dvv::platform::WindowHandle`.
- `GpuBuffer::mapped()` with `flush()` (use `mapWrite()` and `MappedWrite::flush()`), and
  `invalidate()` with `mapped()` (use `mapRead()`).
- `ResourcePool`, `ResourcePoolStats`, `ExecutionContext::createResourcePool` and the header
  `dvv/gpu/resource_pool.h` (no replacement; frame-memory reuse belongs to the frame store).
- The five-value `copyBuffer`, the `dispatch` overloads with trailing group counts and
  `Presenter::setVisible(bool)` (use the `BufferCopy` and `GroupCount` forms, `show()` and
  `hide()`).
- `GpuRegistry`, `GpuRegistryOptions`, `dvv/gpu/gpu_registry.h` and `ExecutionContext::registry()`
  (use `GpuInstance`, `GpuInstanceOptions`, `dvv/gpu/gpu_instance.h` and `gpuInstance()`).
- Reaching names through execution_context.h, gpu_instance.h or presenter.h that 1.0.0 removes
  from them: these headers stop including the Vulkan headers, resources.h and submission.h, and with
  them compute_pipeline.h, command_recorder.h and resource_pool.h; `GpuImage` moves to image.h.
  Code that uses `GpuBuffer`, `BufferDesc`, `ImageDesc`, `RecordFunction`, `SubmitOperation`,
  `ComputePipeline`, `CommandRecorder` or a Vulkan type includes the header that declares it, or
  context_vulkan.h or presenter_vulkan.h.

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
