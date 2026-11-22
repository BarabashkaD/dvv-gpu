#pragma once

#include <atomic>
#include <memory>

#include "device_core.h"  // DeviceCore, vk.h, vk_mem_alloc.h (VmaAllocation_T)
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/resources.h"  // ImageDesc

namespace dvv::gpu::detail {
// Everything of a GpuImage behind image.h: the Vulkan objects, the tracked layout and the weak
// self that the command recorder locks to keep an image alive until its submission completes.
struct ImageState {
    std::shared_ptr<DeviceCore> core;
    ImageDesc desc;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation_T* allocation = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    std::atomic<VkImageLayout> layout{VK_IMAGE_LAYOUT_UNDEFINED};
    DeviceGeneration generation = 0;  // set by createImage (the device's generation)
    std::weak_ptr<GpuImage> self;     // set by createImage before the image is returned
};
}  // namespace dvv::gpu::detail
