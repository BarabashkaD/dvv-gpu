#pragma once

// The thread behind a Presenter (design M1b §5): surface, swapchain, acquire fence and the loop
// that executes PresenterLogic's actions. An internal part of Presenter (spec §9.4: exempt, owned
// only by its Presenter, created only by ExecutionContext::createPresenter()).
//
// Threads: the presenter thread owns every Vulkan object here and the logic; the GUI thread only
// sets the request block (resize, setVisible, requestFrame) and reads stats and failure; the
// context's waiter thread posts frame completions into the mailbox. The request block, mailbox,
// stats and failure are guarded by one mutex. No Vulkan call is made under it.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/presenter.h"
#include "presenter_logic.h"
#include "vk.h"

namespace dvv::gpu {
class ExecutionContext;
}  // namespace dvv::gpu

namespace dvv::gpu::detail {

class DeviceCore;
class InstanceCore;
class PresenterCore;

// D16: the first B8G8R8A8_UNORM or R8G8B8A8_UNORM format with SRGB_NONLINEAR, else the first;
// nullopt for an empty list.
[[nodiscard]] std::optional<VkSurfaceFormatKHR> chooseSurfaceFormat(
    std::span<const VkSurfaceFormatKHR> formats) noexcept;
// max(minImageCount + 1, 2), capped at maxImageCount when that is not 0.
[[nodiscard]] std::uint32_t swapchainImageCount(const VkSurfaceCapabilitiesKHR& caps) noexcept;
// currentExtent; when it is 0xFFFFFFFF, the hint clamped to [minImageExtent, maxImageExtent] (D8).
[[nodiscard]] VkExtent2D swapchainExtent(const VkSurfaceCapabilitiesKHR& caps,
                                         platform::PixelSize hint) noexcept;
// OPAQUE, else INHERIT, else PRE_MULTIPLIED, else POST_MULTIPLIED.
[[nodiscard]] VkCompositeAlphaFlagBitsKHR compositeAlpha(
    VkCompositeAlphaFlagsKHR supported) noexcept;

// The presenter's executor (design §5.1): the waiter thread posts frame completions here, and the
// presenter thread runs them. post() pushes under the presenter mutex and notifies; it never runs
// work inline.
class PresenterMailbox final : public IExecutor {
public:
    explicit PresenterMailbox(PresenterCore& core) noexcept : core_(&core) {}
    void post(std::move_only_function<void()> work) override;

private:
    PresenterCore* core_;
};

class PresenterCore {
public:
    // Allocates only; no Vulkan object exists yet. `context` and `instance` outlive the core: the
    // Presenter holds the context, which holds the registry and its instance.
    PresenterCore(ExecutionContext& context, std::shared_ptr<DeviceCore> device,
                  const InstanceCore& instance, PresenterOptions options);
    PresenterCore(const PresenterCore&) = delete;
    PresenterCore& operator=(const PresenterCore&) = delete;
    PresenterCore(PresenterCore&&) = delete;
    PresenterCore& operator=(PresenterCore&&) = delete;
    // stop(), then destroys what still exists (only a core whose thread never ran has anything).
    ~PresenterCore();

    // Surface, present support, swapchain usage and format, acquire fence (errors: see
    // ExecutionContext::createPresenter).
    [[nodiscard]] Result<void> initialise();
    // Starts the presenter thread; throws std::system_error or std::bad_alloc.
    void start();
    // Sets stop and waits until the thread has ended (design §5.5), dispatching sent messages on
    // Windows. Idempotent; never called on the presenter thread.
    void stop() noexcept;

    void resize(platform::PixelSize extent) noexcept;
    void setVisible(bool visible) noexcept;
    void requestFrame() noexcept;
    [[nodiscard]] PresenterStats stats() const;
    [[nodiscard]] std::optional<Error> failure() const;

    // The mailbox: queues work for the presenter thread; throws std::bad_alloc when it cannot.
    void post(std::move_only_function<void()> work);
    // Called by the frame's coroutine on the presenter thread when its submission has finished.
    void frameFinished(Result<std::uint64_t> result) noexcept;

private:
    using Action = PresenterLogic::Action;

    // The request block (design §5.1). Every resize, setVisible and requestFrame increments
    // requestSeq.
    struct Requests {
        platform::PixelSize extent{};
        bool visible = true;
        std::uint64_t requestSeq = 0;
        bool stop = false;
    };

    void run() noexcept;
    void waitForWork(std::unique_lock<std::mutex>& lock);
    void step(Action action);
    void renderFrame();
    [[nodiscard]] Result<PresentFrame> callProvider(const PresentFrameInfo& info);
    [[nodiscard]] Result<void> checkImage(const std::shared_ptr<GpuImage>& image);
    [[nodiscard]] VkOutcome acquire(std::uint32_t& index);
    void startFrame(PresentFrame frame, std::uint32_t index) noexcept;
    void processCompletion() noexcept;
    void present();
    void recreateSwapchain();
    void failSwapchain(VkResult result, std::string_view what);
    void teardown() noexcept;
    void releaseVulkanObjects() noexcept;
    void fail(Error error) noexcept;
    void reportFrameError(const Error& error);
    void publishState();

    // Fixed after construction.
    ExecutionContext* context_;
    std::shared_ptr<DeviceCore> device_;
    VkInstance instance_;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr_;
    platform::NativeWindow window_;
    FrameProvider frames_;
    PresenterErrorHandler on_error_;
    std::string name_;
    PresenterMailbox mailbox_executor_;

    // Presenter thread only (and initialise()/the destructor while no thread runs).
    PresenterLogic logic_;
    Requests snapshot_{};
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR format_{};
    VkFence fence_ = VK_NULL_HANDLE;
    bool fence_leaked_ = false;  // the 5 s cap of design M1b §5.2 step 4 passed: never destroyed
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    VkExtent2D extent_{};
    std::uint32_t acquired_index_ = 0;
    bool in_flight_ = false;
    std::optional<Result<std::uint64_t>> completion_;
    std::uint64_t frame_number_ = 0;

    // Shared, under mutex_.
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    Requests requests_{};
    std::vector<std::move_only_function<void()>> mailbox_;
    PresenterStats stats_{};
    std::optional<Error> failure_;

    std::jthread thread_;  // started by start(), joined by stop()
};

}  // namespace dvv::gpu::detail
