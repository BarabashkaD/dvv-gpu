#include "dvv/gpu/gpu_registry.h"

#include <format>
#include <utility>

#include "device_core.h"
#include "devices.h"
#include "dvv/base/log.h"
#include "instance.h"
#include "pipeline_cache.h"
#include "submitter.h"

namespace dvv::gpu {

static_assert(ManagedType<GpuRegistry>);

std::string_view validationModeName(ValidationMode mode) noexcept {
    switch (mode) {
        case ValidationMode::Off: return "off";
        case ValidationMode::On: return "on";
        case ValidationMode::Sync: return "sync";
    }
    return "?";
}

Result<std::shared_ptr<GpuRegistry>> GpuRegistry::create(const GpuRegistryOptions& options) {
    DVV_TRY(auto instance, detail::InstanceCore::create(options));
    DVV_TRY(auto enumeration, detail::enumerateDevices(instance->instance(), instance->dispatch()));
    instance->setPhysicalDevices(std::move(enumeration.devices));
    std::vector<DeviceId> ranked = rankDevices(enumeration.infos);
    for (const GpuInfo& info : enumeration.infos) {
        DVV_LOG_INFO("Vulkan device {}: {} ({}, Vulkan {}){}", info.id, info.name,
                     deviceTypeName(info.type), formatVulkanVersion(info.apiVersion),
                     info.usable() ? "" : ", not usable");
    }
    return std::make_shared<GpuRegistry>(
        CreationKey<GpuRegistry>(LifetimeRegistry::instance().issue<GpuRegistry>()),
        std::move(instance), std::move(enumeration.infos), std::move(ranked));
}

GpuRegistry::GpuRegistry(CreationKey<GpuRegistry> key,
                         std::unique_ptr<detail::InstanceCore> instance,
                         std::vector<GpuInfo> devices, std::vector<DeviceId> ranked) noexcept
    : Managed(std::move(key)),
      instance_(std::move(instance)),
      devices_(std::move(devices)),
      ranked_(std::move(ranked)) {}

GpuRegistry::~GpuRegistry() = default;

ValidationMode GpuRegistry::validation() const noexcept {
    return instance_->validation();
}

std::uint64_t GpuRegistry::validationMessageCount() const noexcept {
    return instance_->validationMessageCount();
}

std::uint32_t GpuRegistry::loaderVersion() const noexcept {
    return instance_->loaderVersion();
}

std::optional<platform::WindowSystem> GpuRegistry::windowSystem() const noexcept {
    return instance_->windowSystem();
}

std::optional<DeviceId> GpuRegistry::bestDevice() const noexcept {
    if (ranked_.empty() || !devices_[ranked_.front().index()].usable()) {
        return std::nullopt;
    }
    return ranked_.front();
}

Result<std::shared_ptr<ExecutionContext>> GpuRegistry::createContext(
    DeviceId id, const ContextOptions& options) {
    if (id.index() >= devices_.size()) {
        return makeError(Errc::Internal, std::format("there is no Vulkan device {}", id));
    }
    if (options.presentation && !instance_->windowSystem()) {
        return makeError(Errc::Internal,
                         "ContextOptions::presentation needs a GpuRegistry created with a window "
                         "system (GpuRegistryOptions::windowSystem)");
    }
    DVV_TRY(auto core, detail::DeviceCore::create(shared_from_this(), *instance_,
                                                  instance_->physicalDevices()[id.index()],
                                                  devices_[id.index()], options.presentation));
    DVV_TRY(auto pipeline_cache, detail::PipelineCache::create(core, options.pipelineCacheFile));
    DVV_TRY(auto submitter, detail::Submitter::create(core));
    return std::make_shared<ExecutionContext>(
        CreationKey<GpuRegistry>(LifetimeRegistry::instance().issue<ExecutionContext>(objectId())),
        shared_from_this(), std::move(core), std::move(pipeline_cache), std::move(submitter));
}

}  // namespace dvv::gpu
