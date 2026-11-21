// Compile-fail cases for dvv-gpu's managed types (spec §9.4). DVV_COMPILE_FAIL_CASE selects one:
//   0  control: a GPU instance created through its factory compiles
//   1  constructing a CreationKey<GpuInstance> outside the GPU instance
//   2  constructing a GpuInstance without a key
//   3  constructing an ExecutionContext without a key
//   4  creating a GpuBuffer without its context
//   5  creating a GpuImage without its context
//   6  creating a ResourcePool without its context
//   7  creating a ComputePipeline without its context
//   8  creating a Presenter without its context
//   9  writing a buffer through a const holder (mapWrite() is not const)
//   10 making a MappedWrite outside its buffer
#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include "dvv/base/lifetime.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/resource_pool.h"
#include "dvv/gpu/resources.h"

int main() {
    const auto gpu_instance = dvv::gpu::GpuInstance::create();
#if DVV_COMPILE_FAIL_CASE == 1
    dvv::CreationKey<dvv::gpu::GpuInstance> forged(
        dvv::LifetimeRegistry::instance().issue<dvv::gpu::GpuInstance>());
#elif DVV_COMPILE_FAIL_CASE == 2
    dvv::gpu::GpuInstance direct(std::unique_ptr<dvv::gpu::detail::InstanceCore>{});
#elif DVV_COMPILE_FAIL_CASE == 3
    dvv::gpu::ExecutionContext direct(*gpu_instance,
                                      std::shared_ptr<dvv::gpu::detail::DeviceCore>{});
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
#elif DVV_COMPILE_FAIL_CASE == 9
    const auto write = [](const dvv::gpu::GpuBuffer& buffer) { return buffer.mapWrite(); };
#elif DVV_COMPILE_FAIL_CASE == 10
    const auto forge = [](dvv::gpu::GpuBuffer& buffer) {
        dvv::gpu::MappedWrite scope(buffer, std::span<std::byte>{});
    };
#endif
    return gpu_instance.has_value() ? 0 : 1;
}
