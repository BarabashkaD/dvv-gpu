# Changelog

All notable changes to dvv-gpu. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning with
the dvv rules: compatible = the same major and at least the minimum version, also for 0.x; breaking
changes only in a new major, after a deprecation. This file is the release notes (there are no
GitHub Releases).

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
