#include "dvv/gpu/image.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "device_core.h"
#include "image_state.h"
#include "vk.h"

namespace dvv::gpu {

static_assert(ManagedType<GpuImage>);

GpuImage::GpuImage(CreationKey<ExecutionContext> key,
                   std::unique_ptr<detail::ImageState> state) noexcept
    : Managed(std::move(key)), state_(std::move(state)) {}

GpuImage::~GpuImage() {
    // The retire queue frees in order, so the view is never destroyed after its image.
    if (state_->view != VK_NULL_HANDLE) {
        state_->core->retire(vk::ImageView(state_->view), nullptr);
    }
    state_->core->retire(vk::Image(state_->image), state_->allocation);
    state_->core->collect();
}

std::string_view GpuImage::name() const noexcept {
    return state_->desc.name;
}

std::uint32_t GpuImage::width() const noexcept {
    return state_->desc.width;
}

std::uint32_t GpuImage::height() const noexcept {
    return state_->desc.height;
}

DeviceId GpuImage::deviceId() const noexcept {
    return state_->core->id();
}

DeviceGeneration GpuImage::generation() const noexcept {
    return state_->generation;
}

}  // namespace dvv::gpu
