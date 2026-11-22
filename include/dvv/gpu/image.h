#pragma once

// A GPU image without its Vulkan side, which context_vulkan.h reads (vulkanHandle(), vulkanView(),
// vulkanFormat(), vulkanLayout(), vulkanDesc()).

#include <cstdint>
#include <memory>
#include <string_view>

#include "dvv/base/lifetime.h"
#include "dvv/gpu/device_id.h"

namespace dvv::gpu {

class ExecutionContext;

namespace detail {
struct ImageState;
struct Access;
}  // namespace detail

// A 2D image with one mip level and one layer in device-local memory, optimal tiling, made by
// createImage (context_vulkan.h). The destructor hands the view, the image and its memory to the
// context's retire queue, which frees them once the GPU has finished every submission made so far.
class GpuImage final : public Managed<GpuImage> {
public:
    GpuImage(CreationKey<ExecutionContext> key, std::unique_ptr<detail::ImageState> state) noexcept;
    ~GpuImage();

    [[nodiscard]] std::string_view name() const noexcept;  // the debug name given at creation
    [[nodiscard]] std::uint32_t width() const noexcept;
    [[nodiscard]] std::uint32_t height() const noexcept;
    [[nodiscard]] DeviceId deviceId() const noexcept;
    [[nodiscard]] DeviceGeneration generation() const noexcept;

private:
    friend struct detail::Access;
    std::unique_ptr<detail::ImageState> state_;
};

}  // namespace dvv::gpu
