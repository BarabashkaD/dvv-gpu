// Compile-fail cases for dvv-gpu's managed types (spec §9.4). DVV_COMPILE_FAIL_CASE selects one:
//   0  control: a registry created through its factory compiles
//   1  constructing a CreationKey<GpuRegistry> outside the registry
//   2  constructing a GpuRegistry without a key
//   3  constructing an ExecutionContext without a key
//   4  creating a GpuBuffer without its context
//   5  creating a GpuImage without its context
//   6  creating a ResourcePool without its context
//   7  creating a ComputePipeline without its context
//   8  creating a Presenter without its context
#include <memory>
#include <string>

#include "dvv/base/lifetime.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/resource_pool.h"
#include "dvv/gpu/resources.h"

int main() {
    const auto registry = dvv::gpu::GpuRegistry::create();
#if DVV_COMPILE_FAIL_CASE == 1
    dvv::CreationKey<dvv::gpu::GpuRegistry> forged(
        dvv::LifetimeRegistry::instance().issue<dvv::gpu::GpuRegistry>());
#elif DVV_COMPILE_FAIL_CASE == 2
    dvv::gpu::GpuRegistry direct(std::unique_ptr<dvv::gpu::detail::InstanceCore>{});
#elif DVV_COMPILE_FAIL_CASE == 3
    dvv::gpu::ExecutionContext direct(*registry, std::shared_ptr<dvv::gpu::detail::DeviceCore>{});
#elif DVV_COMPILE_FAIL_CASE == 4
    auto buffer = std::make_shared<dvv::gpu::GpuBuffer>(dvv::gpu::BufferDesc{.size = 4});
#elif DVV_COMPILE_FAIL_CASE == 5
    auto image = std::make_shared<dvv::gpu::GpuImage>(dvv::gpu::ImageDesc{.width = 4});
#elif DVV_COMPILE_FAIL_CASE == 6
    // Every argument but the key.
    auto pool = std::make_shared<dvv::gpu::ResourcePool>(
        std::shared_ptr<dvv::gpu::ExecutionContext>{},
        std::shared_ptr<dvv::gpu::detail::ResourcePoolState>{});
#elif DVV_COMPILE_FAIL_CASE == 7
    // Every argument but the key.
    auto pipeline = std::make_shared<dvv::gpu::ComputePipeline>(
        std::shared_ptr<dvv::gpu::detail::DeviceCore>{}, std::string("add_one"), VkPipelineLayout{},
        VkPipeline{}, 16U, VkDescriptorSetLayout{}, dvv::gpu::ComputeBindings{});
#elif DVV_COMPILE_FAIL_CASE == 8
    // Every argument but the key.
    auto presenter = std::make_unique<dvv::gpu::Presenter>(
        std::shared_ptr<dvv::gpu::ExecutionContext>{}, nullptr);
#endif
    return registry.has_value() ? 0 : 1;
}
