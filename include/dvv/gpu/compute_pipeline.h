#pragma once

// Compute pipelines (spec §5.4).

#include <cstdint>
#include <memory>
#include <string>

#include <vulkan/vulkan_core.h>

#include "dvv/base/lifetime.h"
#include "dvv/gpu/device_id.h"

namespace dvv::gpu {

class ExecutionContext;

namespace detail {
class DeviceCore;
struct Access;
}  // namespace detail

// The descriptors a compute pipeline binds (spec §5.4 as changed by M2 D6): storage images in
// set 0, bindings 0..storageImages-1, pushed per dispatch
// (VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT). Pipelines without bindings have no
// descriptor set at all.
struct ComputeBindings {
    std::uint32_t storageImages = 0;
    bool operator==(const ComputeBindings&) const = default;
};

// A compute pipeline made from an EmbeddedShader (entry point main), a push-constant range of a
// given size and its ComputeBindings (shaders receive buffers by device address, storage images
// through push descriptors). Created by ExecutionContext::createComputePipeline() with the
// context's pipeline cache. Like a resource, the destructor hands the pipeline and its layouts to
// the retire queue, so a pipeline may be released while the GPU still runs it.
class ComputePipeline final : public Managed<ComputePipeline> {
public:
    ComputePipeline(CreationKey<ExecutionContext> key, std::shared_ptr<detail::DeviceCore> core,
                    std::string name, VkPipelineLayout layout, VkPipeline pipeline,
                    std::uint32_t push_constant_size, VkDescriptorSetLayout set_layout,
                    ComputeBindings bindings) noexcept;
    ~ComputePipeline();

    // The shader's name.
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] std::uint32_t pushConstantSize() const noexcept { return push_constant_size_; }
    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }
    [[nodiscard]] const ComputeBindings& bindings() const noexcept { return bindings_; }
    // The push-descriptor set layout (set 0); VK_NULL_HANDLE without bindings.
    [[nodiscard]] VkDescriptorSetLayout setLayout() const noexcept { return set_layout_; }
    [[nodiscard]] DeviceId deviceId() const noexcept;
    [[nodiscard]] DeviceGeneration generation() const noexcept { return generation_; }

private:
    friend struct detail::Access;  // the command recorder checks the device

    std::shared_ptr<detail::DeviceCore> core_;
    std::string name_;
    VkPipelineLayout layout_;
    VkPipeline pipeline_;
    std::uint32_t push_constant_size_;
    DeviceGeneration generation_;
    VkDescriptorSetLayout set_layout_;
    ComputeBindings bindings_;
};

}  // namespace dvv::gpu
