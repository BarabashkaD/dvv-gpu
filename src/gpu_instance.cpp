#include "dvv/gpu/gpu_instance.h"

#include <format>
#include <utility>

#include "device_core.h"
#include "devices.h"
#include "dvv/base/log.h"
#include "instance.h"
#include "pipeline_cache.h"
#include "submitter.h"

namespace dvv::gpu {

static_assert(ManagedType<GpuInstance>);

std::string_view validationModeName(ValidationMode mode) noexcept {
    switch (mode) {
        case ValidationMode::Off: return "off";
        case ValidationMode::On: return "on";
        case ValidationMode::Sync: return "sync";
    }
    return "?";
}

Result<std::shared_ptr<GpuInstance>> GpuInstance::create(const GpuInstanceOptions& options) {
    DVV_TRY(auto instance, detail::InstanceCore::create(options));
    DVV_TRY(auto enumeration, detail::enumerateDevices(instance->instance(), instance->dispatch()));
    instance->setPhysicalDevices(std::move(enumeration.devices));
    std::vector<DeviceId> ranked = rankDevices(enumeration.infos);
    for (const GpuInfo& info : enumeration.infos) {
        DVV_LOG_INFO("Vulkan device {}: {} ({}, Vulkan {}){}", info.id, info.name,
                     deviceTypeName(info.type), formatVulkanVersion(info.apiVersion),
                     info.usable() ? "" : ", not usable");
    }
    return std::make_shared<GpuInstance>(
        CreationKey<GpuInstance>(LifetimeRegistry::instance().issue<GpuInstance>()),
        std::move(instance), std::move(enumeration.infos), std::move(ranked));
}

GpuInstance::GpuInstance(CreationKey<GpuInstance> key,
                         std::unique_ptr<detail::InstanceCore> instance,
                         std::vector<GpuInfo> devices, std::vector<DeviceId> ranked) noexcept
    : Managed(std::move(key)),
      instance_(std::move(instance)),
      devices_(std::move(devices)),
      ranked_(std::move(ranked)) {}

GpuInstance::~GpuInstance() = default;

ValidationMode GpuInstance::validation() const noexcept {
    return instance_->validation();
}

std::uint64_t GpuInstance::validationMessageCount() const noexcept {
    return instance_->validationMessageCount();
}

std::uint32_t GpuInstance::loaderVersion() const noexcept {
    return instance_->loaderVersion();
}

std::optional<platform::WindowSystem> GpuInstance::windowSystem() const noexcept {
    return instance_->windowSystem();
}

std::optional<DeviceId> GpuInstance::bestDevice() const noexcept {
    if (ranked_.empty() || !devices_[ranked_.front().index()].usable()) {
        return std::nullopt;
    }
    return ranked_.front();
}

Result<std::shared_ptr<ExecutionContext>> GpuInstance::createContext(
    DeviceId id, const ContextOptions& options) {
    if (id.index() >= devices_.size()) {
        return makeError(Errc::Internal, std::format("there is no Vulkan device {}", id));
    }
    if (options.presentation && !instance_->windowSystem()) {
        return makeError(Errc::Internal,
                         "ContextOptions::presentation needs a GPU instance created with a window "
                         "system (GpuInstanceOptions::windowSystem)");
    }
    DVV_TRY(auto core, detail::DeviceCore::create(shared_from_this(), *instance_,
                                                  instance_->physicalDevices()[id.index()],
                                                  devices_[id.index()], options));
    DVV_TRY(auto pipeline_cache, detail::PipelineCache::create(core, options.pipelineCacheFile));
    DVV_TRY(auto submitter, detail::Submitter::create(core));
    return std::make_shared<ExecutionContext>(
        CreationKey<GpuInstance>(LifetimeRegistry::instance().issue<ExecutionContext>(objectId())),
        shared_from_this(), std::move(core), std::move(pipeline_cache), std::move(submitter));
}

}  // namespace dvv::gpu
