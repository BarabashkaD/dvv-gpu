#pragma once

// The one place that configures Vulkan-Hpp and VMA. Every dvv-gpu source includes Vulkan through
// this header, so the configuration is identical in every translation unit:
// - VK_NO_PROTOTYPES (set by the build for every user of dvv_gpu): nothing links the Vulkan
//   loader; it is loaded at run time.
// - Dynamic dispatch with explicit dispatchers: every instance and device has its own
//   vk::detail::DispatchLoaderDynamic, passed to every call. No default dispatcher storage is
//   defined, so a call that forgets its dispatcher fails to link.
// - No exceptions: calls return results, which are converted to dvv::Error (vk_result.h).
//   VULKAN_HPP_ASSERT_ON_RESULT is a no-op; by default Debug builds would abort on recoverable
//   errors such as VK_ERROR_OUT_OF_DEVICE_MEMORY.
// - No struct constructors: structures are filled with designated initialisers.
// - VMA fetches every Vulkan function through vkGetInstanceProcAddr / vkGetDeviceProcAddr.
// - The format traits (texel block sizes) come with it; they need the same configuration.

#if defined(VULKAN_HPP)
#error "vulkan.hpp was included before vk.h; include Vulkan only through vk.h"
#endif
#if !defined(VK_NO_PROTOTYPES)
#error "dvv_gpu is built with VK_NO_PROTOTYPES"
#endif

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_ASSERT_ON_RESULT(condition) static_cast<void>(0)
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_format_traits.hpp>  // vk::blockSize() and other format properties
