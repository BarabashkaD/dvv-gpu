#include "dvv/gpu/presenter.h"

#include <format>
#include <new>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include "access.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "presenter_core.h"

namespace dvv::gpu {

static_assert(ManagedType<Presenter>);

Presenter::Presenter(CreationKey<ExecutionContext> key, std::shared_ptr<ExecutionContext> context,
                     std::unique_ptr<detail::PresenterCore> core) noexcept
    : Managed(std::move(key)), context_(std::move(context)), core_(std::move(core)) {}

Presenter::~Presenter() {
    // Joins the thread and destroys swapchain, surface and fence while the context (device and
    // instance) is alive; the member order guarantees the same.
    core_.reset();
}

void Presenter::resize(platform::PixelSize extent) noexcept {
    core_->resize(extent);
}

void Presenter::setVisible(bool visible) noexcept {
    core_->setVisible(visible);
}

void Presenter::requestFrame() noexcept {
    core_->requestFrame();
}

PresenterStats Presenter::stats() const {
    return core_->stats();
}

std::optional<Error> Presenter::failure() const {
    return core_->failure();
}

Result<std::unique_ptr<Presenter>> ExecutionContext::createPresenter(PresenterOptions options) {
    if (core_->lost()) {
        return std::unexpected(core_->lostError());
    }
    if (!presentation()) {
        return makeError(Errc::Internal,
                         std::format("presenter '{}': the context of device {} was created "
                                     "without ContextOptions::presentation",
                                     options.name, deviceId()));
    }
    if (!options.frames) {
        return makeError(Errc::Internal,
                         std::format("presenter '{}' has no frame provider", options.name));
    }
    const std::optional<platform::WindowSystem> system = platform::windowSystemOf(options.window);
    const std::optional<platform::WindowSystem> wanted = registry_->windowSystem();
    if (!system || system != wanted) {
        const std::string_view have = system ? platform::windowSystemName(*system) : "no window";
        const std::string_view need =
            wanted ? platform::windowSystemName(*wanted) : "no window system";
        return makeError(Errc::Internal,
                         std::format("presenter '{}': the window is {}, the registry presents "
                                     "to {}",
                                     options.name, have, need));
    }
    auto core = std::make_unique<detail::PresenterCore>(
        *this, core_, detail::Access::instance(*registry_), std::move(options));
    DVV_TRY_VOID(core->initialise());
    try {
        core->start();
    } catch (const std::system_error& error) {
        return makeError(Errc::Internal,
                         std::format("the presenter thread could not start: {}", error.what()));
    } catch (const std::bad_alloc&) {
        return makeError(Errc::OutOfMemory, "the presenter thread could not start: out of memory");
    }
    return std::make_unique<Presenter>(
        CreationKey<ExecutionContext>(LifetimeRegistry::instance().issue<Presenter>(objectId())),
        shared_from_this(), std::move(core));
}

}  // namespace dvv::gpu
