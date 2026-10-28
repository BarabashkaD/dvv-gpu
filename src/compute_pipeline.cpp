#include "dvv/gpu/compute_pipeline.h"

#include <utility>

#include "device_core.h"

namespace dvv::gpu {

static_assert(ManagedType<ComputePipeline>);

ComputePipeline::ComputePipeline(CreationKey<ExecutionContext> key,
                                 std::shared_ptr<detail::DeviceCore> core, std::string name,
                                 VkPipelineLayout layout, VkPipeline pipeline,
                                 std::uint32_t push_constant_size, VkDescriptorSetLayout set_layout,
                                 ComputeBindings bindings) noexcept
    : Managed(std::move(key)),
      core_(std::move(core)),
      name_(std::move(name)),
      layout_(layout),
      pipeline_(pipeline),
      push_constant_size_(push_constant_size),
      generation_(core_->generation()),
      set_layout_(set_layout),
      bindings_(bindings) {}

ComputePipeline::~ComputePipeline() {
    core_->retire(vk::Pipeline(pipeline_), nullptr);
    core_->retire(vk::PipelineLayout(layout_), nullptr);
    if (set_layout_ != VK_NULL_HANDLE) {
        core_->retire(vk::DescriptorSetLayout(set_layout_), nullptr);
    }
    core_->collect();
}

DeviceId ComputePipeline::deviceId() const noexcept {
    return core_->id();
}

}  // namespace dvv::gpu
