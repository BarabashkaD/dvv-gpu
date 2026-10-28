# Changelog

All notable changes to dvv-gpu. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning with
the dvv rules: compatible = the same major and at least the minimum version, also for 0.x; breaking
changes only in a new major, after a deprecation. This file is the release notes (there are no
GitHub Releases).

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
