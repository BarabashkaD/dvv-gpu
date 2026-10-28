#include "dvv/gpu/execution_context.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "device_core.h"
#include "dvv/base/json.h"
#include "dvv/base/log.h"
#include "dvv/base/paths.h"
#include "dvv/gpu/gpu_registry.h"
#include "pipeline_cache.h"
#include "resource_pool_state.h"
#include "submitter.h"
#include "vk_result.h"

namespace dvv::gpu {

static_assert(ManagedType<ExecutionContext>);

namespace {

// The bytes of push constants the shader's reflection (slangc -reflection-json) declares: the end
// of its largest global push-constant block, 0 without one. Errc::Format when it cannot be read.
Result<std::uint64_t> reflectedPushConstantSize(std::string_view reflection_json) {
    DVV_TRY(const json::Value reflection, json::parse(reflection_json));
    try {
        const json::Value::json_pointer kind("/binding/kind");
        const json::Value::json_pointer block("/type/elementVarLayout/binding");
        std::uint64_t size = 0;
        const auto parameters = reflection.find("parameters");
        if (parameters != reflection.end()) {
            for (const json::Value& parameter : *parameters) {
                if (parameter.contains(kind) && parameter.at(kind) == "pushConstantBuffer") {
                    const json::Value& layout = parameter.at(block);
                    size = std::max(size, layout.at("offset").get<std::uint64_t>() +
                                              layout.at("size").get<std::uint64_t>());
                }
            }
        }
        return size;
    } catch (const json::Value::exception& error) {
        return makeError(Errc::Format, error.what());
    }
}

// {space, index} of every descriptor binding the shader's reflection declares (parameters whose
// binding kind is descriptorTableSlot). Errc::Format when it cannot be read.
Result<std::vector<std::pair<std::uint32_t, std::uint32_t>>> reflectedDescriptorBindings(
    std::string_view reflection_json) {
    DVV_TRY(const json::Value reflection, json::parse(reflection_json));
    try {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> bindings;
        const auto parameters = reflection.find("parameters");
        if (parameters != reflection.end()) {
            for (const json::Value& parameter : *parameters) {
                const auto binding = parameter.find("binding");
                if (binding != parameter.end() &&
                    binding->value("kind", std::string()) == "descriptorTableSlot") {
                    bindings.emplace_back(binding->value("space", 0U),
                                          binding->at("index").get<std::uint32_t>());
                }
            }
        }
        std::ranges::sort(bindings);
        return bindings;
    } catch (const json::Value::exception& error) {
        return makeError(Errc::Format, error.what());
    }
}

}  // namespace

ExecutionContext::ExecutionContext(CreationKey<GpuRegistry> key,
                                   std::shared_ptr<GpuRegistry> registry,
                                   std::shared_ptr<detail::DeviceCore> core,
                                   std::unique_ptr<detail::PipelineCache> pipeline_cache,
                                   std::unique_ptr<detail::Submitter> submitter) noexcept
    : Managed(std::move(key)),
      registry_(std::move(registry)),
      core_(std::move(core)),
      pipeline_cache_(std::move(pipeline_cache)),
      submitter_(std::move(submitter)) {}

ExecutionContext::~ExecutionContext() {
    submitter_->stop();  // its waits are bounded, so this returns on a lost device too
    // The idle wait comes before the cache decision: it is the last call that can see a loss no
    // other call has seen (e.g. a driver reset while the context was idle).
    const auto drained = core_->drain();
    if (core_->lost()) {
        // The driver's cache data cannot be trusted after a loss.
        DVV_LOG_INFO("the pipeline cache of device {} is not saved: the device was lost",
                     core_->id());
    } else {
        try {
            if (const auto saved = pipeline_cache_->saveIfChanged(); !saved) {
                DVV_LOG_ERROR("saving the pipeline cache of device {} failed: {}", core_->id(),
                              saved.error().describe());
            }
        } catch (const std::exception& error) {
            DVV_LOG_ERROR("saving the pipeline cache of device {} failed: {}", core_->id(),
                          error.what());
        }
    }
    if (!drained && drained.error().code != Errc::DeviceLost) {
        DVV_LOG_ERROR("closing the context of device {} failed: {}", core_->id(),
                      drained.error().describe());
    }
}

DeviceId ExecutionContext::deviceId() const noexcept {
    return core_->id();
}

DeviceGeneration ExecutionContext::generation() const noexcept {
    return core_->generation();
}

const GpuInfo& ExecutionContext::deviceInfo() const noexcept {
    return core_->info();
}

bool ExecutionContext::presentation() const noexcept {
    return core_->presentation();
}

bool ExecutionContext::lost() const noexcept {
    return core_->lost();
}

Result<void> ExecutionContext::drain() {
    return core_->drain();
}

Result<std::shared_ptr<ResourcePool>> ExecutionContext::createResourcePool() {
    if (core_->lost()) {
        return std::unexpected(core_->lostError());
    }
    auto state = std::make_shared<detail::ResourcePoolState>(core_);
    return std::make_shared<ResourcePool>(
        CreationKey<ExecutionContext>(LifetimeRegistry::instance().issue<ResourcePool>(objectId())),
        shared_from_this(), std::move(state));
}

Result<std::shared_ptr<ComputePipeline>> ExecutionContext::createComputePipeline(
    const EmbeddedShader& shader, std::uint32_t push_constant_size) {
    if (core_->lost()) {
        return std::unexpected(core_->lostError());
    }
    return createComputePipeline(shader, push_constant_size, ComputeBindings{});
}

Result<std::shared_ptr<ComputePipeline>> ExecutionContext::createComputePipeline(
    const EmbeddedShader& shader, std::uint32_t push_constant_size, ComputeBindings bindings) {
    if (core_->lost()) {
        return std::unexpected(core_->lostError());
    }
    constexpr std::uint32_t SPIRV_MAGIC = 0x07230203;
    constexpr std::size_t SPIRV_HEADER_WORDS = 5;
    if (shader.spirv.size() < SPIRV_HEADER_WORDS || shader.spirv.front() != SPIRV_MAGIC) {
        return makeError(Errc::Internal, std::format("shader '{}' is not SPIR-V", shader.name));
    }
    if (push_constant_size % 4 != 0) {
        return makeError(Errc::Internal,
                         std::format("shader '{}': {} bytes of push constants are not a multiple "
                                     "of 4",
                                     shader.name, push_constant_size));
    }
    if (!shader.reflectionJson.empty()) {
        const auto needed = reflectedPushConstantSize(shader.reflectionJson);
        if (!needed) {
            return makeError(Errc::Format,
                             std::format("shader '{}': its reflection cannot be read: {}",
                                         shader.name, needed.error().message));
        }
        if (push_constant_size < *needed) {
            return makeError(Errc::Internal,
                             std::format("shader '{}' needs {} bytes of push constants, not {}",
                                         shader.name, *needed, push_constant_size));
        }
    }
    const std::uint32_t max_push = core_->limits().maxPushConstantsSize;
    if (push_constant_size > max_push) {
        return makeError(Errc::Unsupported,
                         std::format("shader '{}': {} bytes of push constants exceed the {} bytes "
                                     "of device {}",
                                     shader.name, push_constant_size, max_push, core_->id()));
    }
    const std::uint32_t max_images = core_->maxPushDescriptors();
    if (bindings.storageImages > max_images) {
        return makeError(Errc::Internal,
                         std::format("shader '{}': {} storage images exceed the device's "
                                     "maxPushDescriptors ({})",
                                     shader.name, bindings.storageImages, max_images));
    }
    if (!shader.reflectionJson.empty()) {
        const auto declared = reflectedDescriptorBindings(shader.reflectionJson);
        if (!declared) {
            return makeError(Errc::Format,
                             std::format("shader '{}': its reflection cannot be read: {}",
                                         shader.name, declared.error().message));
        }
        std::vector<std::pair<std::uint32_t, std::uint32_t>> wanted;
        wanted.reserve(bindings.storageImages);
        for (std::uint32_t index = 0; index < bindings.storageImages; ++index) {
            wanted.emplace_back(0U, index);
        }
        if (*declared != wanted) {
            std::string list;
            for (const auto& [space, index] : *declared) {
                list +=
                    std::format("{}(set {}, binding {})", list.empty() ? "" : ", ", space, index);
            }
            return makeError(Errc::Internal,
                             std::format("shader '{}' declares descriptor bindings [{}]; the "
                                         "pipeline binds {} storage images (set 0, bindings "
                                         "0..n-1)",
                                         shader.name, list, bindings.storageImages));
        }
    }
    core_->collect();

    std::string name(shader.name);  // allocated before any Vulkan object exists
    const vk::Device device = core_->device();
    const auto& dispatch = core_->dispatch();
    vk::DescriptorSetLayout set_layout{};
    if (bindings.storageImages > 0) {
        std::vector<vk::DescriptorSetLayoutBinding> slots(bindings.storageImages);
        for (std::uint32_t index = 0; index < bindings.storageImages; ++index) {
            slots[index] = {.binding = index,
                            .descriptorType = vk::DescriptorType::eStorageImage,
                            .descriptorCount = 1,
                            .stageFlags = vk::ShaderStageFlagBits::eCompute};
        }
        const auto created = device.createDescriptorSetLayout(
            {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
             .bindingCount = bindings.storageImages,
             .pBindings = slots.data()},
            nullptr, dispatch);
        DVV_TRY_VOID(detail::vkCheck(created.result, "vkCreateDescriptorSetLayout"));
        set_layout = created.value;
    }
    // Destroys the set layout when a later step fails.
    const auto destroy_set_layout = [&]() noexcept {
        if (set_layout) {
            device.destroyDescriptorSetLayout(set_layout, nullptr, dispatch);
        }
    };
    const vk::PushConstantRange range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = push_constant_size};
    const auto layout =
        device.createPipelineLayout({.setLayoutCount = set_layout ? 1U : 0U,
                                     .pSetLayouts = &set_layout,
                                     .pushConstantRangeCount = push_constant_size > 0 ? 1U : 0U,
                                     .pPushConstantRanges = &range},
                                    nullptr, dispatch);
    if (layout.result != vk::Result::eSuccess) {
        destroy_set_layout();
        return detail::vkError(layout.result, "vkCreatePipelineLayout");
    }
    // maintenance5: the SPIR-V is chained to the stage; there is no VkShaderModule.
    const vk::ShaderModuleCreateInfo code{.codeSize = shader.spirv.size_bytes(),
                                          .pCode = shader.spirv.data()};
    const vk::ComputePipelineCreateInfo pipeline_info{
        .stage = {.pNext = &code, .stage = vk::ShaderStageFlagBits::eCompute, .pName = "main"},
        .layout = layout.value};
    const auto pipeline =
        device.createComputePipeline(pipeline_cache_->handle(), pipeline_info, nullptr, dispatch);
    if (pipeline.result != vk::Result::eSuccess) {
        device.destroyPipelineLayout(layout.value, nullptr, dispatch);
        destroy_set_layout();
        return detail::vkError(pipeline.result, std::format("vkCreateComputePipelines ({})", name));
    }
    core_->setDebugName(pipeline.value, name);
    core_->setDebugName(layout.value, name);
    if (set_layout) {
        core_->setDebugName(set_layout, name);
    }
    try {
        return std::make_shared<ComputePipeline>(
            CreationKey<ExecutionContext>(
                LifetimeRegistry::instance().issue<ComputePipeline>(objectId())),
            core_, std::move(name), static_cast<VkPipelineLayout>(layout.value),
            static_cast<VkPipeline>(pipeline.value), push_constant_size,
            static_cast<VkDescriptorSetLayout>(set_layout), bindings);
    } catch (...) {
        device.destroyPipeline(pipeline.value, nullptr, dispatch);
        device.destroyPipelineLayout(layout.value, nullptr, dispatch);
        destroy_set_layout();
        throw;
    }
}

VkFormatFeatureFlags2 ExecutionContext::formatFeatures(VkFormat format) const {
    return core_->formatFeatures(format);
}

Result<void> ExecutionContext::savePipelineCache() {
    return pipeline_cache_->save();
}

Result<std::filesystem::path> defaultPipelineCacheFile(std::string_view app,
                                                       const GpuInfo& device) {
    DVV_TRY(std::filesystem::path directory, paths::userCacheDir(app));
    return directory / std::format("vulkan-pipelines-{}.bin", formatUuid(device.deviceUuid));
}

SubmitOperation ExecutionContext::submit(IExecutor& resume_on, RecordFunction record) {
    return {shared_from_this(), resume_on, std::move(record)};
}

SubmitOperation::SubmitOperation(std::shared_ptr<ExecutionContext> context, IExecutor& resume_on,
                                 RecordFunction record) noexcept
    : context_(std::move(context)), resume_on_(&resume_on), record_(std::move(record)) {}

void SubmitOperation::await_suspend(std::coroutine_handle<> continuation) {
    auto submission = std::make_unique<detail::Submission>();
    submission->record = std::move(record_);
    submission->resumeOn = resume_on_;
    submission->continuation = continuation;
    submission->result = &result_;
    // The last use of this object: the coroutine may resume (on another thread) at once.
    context_->submitter_->enqueue(std::move(submission));
}

RetireStats ExecutionContext::retireStats() const noexcept {
    return core_->retireStats();
}

Result<std::shared_ptr<GpuBuffer>> ExecutionContext::createBuffer(BufferDesc desc) {
    if (core_->lost()) {
        return std::unexpected(core_->lostError());
    }
    if (desc.size == 0) {
        return makeError(Errc::Internal, std::format("buffer '{}': the size is zero", desc.name));
    }
    core_->collect();  // memory the GPU has finished with is reused first

    const vk::BufferCreateInfo buffer_info{
        .size = desc.size,
        .usage = vk::BufferUsageFlags(desc.usage) | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive};
    VmaAllocationCreateInfo allocation_info{};
    switch (desc.memory) {
        case MemoryUsage::DeviceOnly:
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
        case MemoryUsage::HostUpload:
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MemoryUsage::HostReadback:
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
            allocation_info.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocated{};
    DVV_TRY_VOID(detail::vkCheck(
        vmaCreateBuffer(core_->allocator(), &static_cast<const VkBufferCreateInfo&>(buffer_info),
                        &allocation_info, &buffer, &allocation, &allocated),
        "vmaCreateBuffer"));
    const VkDeviceAddress address =
        core_->device().getBufferAddress({.buffer = vk::Buffer(buffer)}, core_->dispatch());
    std::span<std::byte> mapped;
    if (desc.memory != MemoryUsage::DeviceOnly) {
        mapped = {static_cast<std::byte*>(allocated.pMappedData),
                  static_cast<std::size_t>(desc.size)};
    }
    if (!desc.name.empty()) {
        core_->setDebugName(vk::Buffer(buffer), desc.name);
        vmaSetAllocationName(core_->allocator(), allocation, desc.name.c_str());
    }
    try {
        return std::make_shared<GpuBuffer>(
            CreationKey<ExecutionContext>(
                LifetimeRegistry::instance().issue<GpuBuffer>(objectId())),
            core_, std::move(desc), buffer, allocation, address, mapped);
    } catch (...) {
        vmaDestroyBuffer(core_->allocator(), buffer, allocation);
        throw;
    }
}

Result<std::shared_ptr<GpuImage>> ExecutionContext::createImage(ImageDesc desc) {
    if (core_->lost()) {
        return std::unexpected(core_->lostError());
    }
    if (desc.width == 0 || desc.height == 0 || desc.format == VK_FORMAT_UNDEFINED ||
        desc.usage == 0) {
        return makeError(
            Errc::Internal,
            std::format("image '{}': width, height, format and usage must be set", desc.name));
    }
    const auto format = static_cast<vk::Format>(desc.format);
    const auto usage = vk::ImageUsageFlags(desc.usage);
    if ((desc.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
        (core_->formatFeatures(desc.format) & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) == 0) {
        return makeError(Errc::Unsupported,
                         std::format("image '{}': device {} has no storage images of {}", desc.name,
                                     core_->id(), vk::to_string(format)));
    }
    const auto supported =
        core_->physicalDevice().getImageFormatProperties2({.format = format,
                                                           .type = vk::ImageType::e2D,
                                                           .tiling = vk::ImageTiling::eOptimal,
                                                           .usage = usage},
                                                          core_->dispatch());
    if (supported.result == vk::Result::eErrorFormatNotSupported) {
        return makeError(
            Errc::Unsupported,
            std::format("image '{}': device {} does not support {} images with usage {}", desc.name,
                        core_->id(), vk::to_string(format), vk::to_string(usage)));
    }
    DVV_TRY_VOID(detail::vkCheck(supported.result, "vkGetPhysicalDeviceImageFormatProperties2"));
    const vk::Extent3D& max = supported.value.imageFormatProperties.maxExtent;
    if (desc.width > max.width || desc.height > max.height) {
        return makeError(Errc::Unsupported,
                         std::format("image '{}': {}x{} exceeds the largest {} image of device {} "
                                     "({}x{})",
                                     desc.name, desc.width, desc.height, vk::to_string(format),
                                     core_->id(), max.width, max.height));
    }
    core_->collect();

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {.width = desc.width, .height = desc.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined};
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    DVV_TRY_VOID(detail::vkCheck(
        vmaCreateImage(core_->allocator(), &static_cast<const VkImageCreateInfo&>(image_info),
                       &allocation_info, &image, &allocation, nullptr),
        "vmaCreateImage"));
    if (!desc.name.empty()) {
        core_->setDebugName(vk::Image(image), desc.name);
        vmaSetAllocationName(core_->allocator(), allocation, desc.name.c_str());
    }
    VkImageView view = VK_NULL_HANDLE;
    if ((desc.usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) != 0) {
        const auto created = core_->device().createImageView(
            {.image = vk::Image(image),
             .viewType = vk::ImageViewType::e2D,
             .format = format,
             .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                  .baseMipLevel = 0,
                                  .levelCount = 1,
                                  .baseArrayLayer = 0,
                                  .layerCount = 1}},
            nullptr, core_->dispatch());
        if (created.result != vk::Result::eSuccess) {
            vmaDestroyImage(core_->allocator(), image, allocation);
            return detail::vkError(created.result, "vkCreateImageView");
        }
        view = static_cast<VkImageView>(created.value);
        if (!desc.name.empty()) {
            core_->setDebugName(created.value, desc.name);
        }
    }
    try {
        return std::make_shared<GpuImage>(
            CreationKey<ExecutionContext>(LifetimeRegistry::instance().issue<GpuImage>(objectId())),
            core_, std::move(desc), image, allocation, view);
    } catch (...) {
        if (view != VK_NULL_HANDLE) {
            core_->device().destroyImageView(vk::ImageView(view), nullptr, core_->dispatch());
        }
        vmaDestroyImage(core_->allocator(), image, allocation);
        throw;
    }
}

}  // namespace dvv::gpu
