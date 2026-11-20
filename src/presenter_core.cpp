#include "presenter_core.h"

#include <algorithm>
#include <chrono>
#include <coroutine>
#include <exception>
#include <format>
#include <new>
#include <string_view>
#include <utility>

#include "access.h"
#include "device_core.h"
#include "dvv/base/log.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/execution_context.h"
#include "instance.h"
#include "present_blit.h"
#include "surface_vulkan.h"
#include "thread_wait.h"
#include "vk_result.h"

namespace dvv::gpu::detail {

namespace {

using namespace std::chrono_literals;

constexpr std::uint64_t ACQUIRE_TIMEOUT_NS = 100'000'000;  // design M1b §5.2 step 3
constexpr std::uint64_t FENCE_SLICE_NS = 100'000'000;      // design M1b §5.2 step 4
constexpr auto FENCE_CAP = 5s;                             // design M1b §5.2 step 4
constexpr auto RETRY_DELAY = 10ms;                         // design M1b §5.2 WaitAndRetry
constexpr std::uint32_t UNDEFINED_EXTENT = 0xFFFFFFFFU;

Error surfaceLost() {
    return makeError(Errc::Internal,
                     "the window's Vulkan surface was lost (was the window destroyed before its "
                     "presenter?)")
        .error();
}

// A VK_ERROR_DEVICE_LOST that the presenter sees puts its context into the lost state (design M3
// §4.1), so creators refuse and destruction does not save the pipeline cache. Called without the
// queue lock (markLost logs) and before fail(), so that onError already sees a lost context.
void markIfLost(DeviceCore& device, vk::Result result, std::string_view where) noexcept {
    if (result == vk::Result::eErrorDeviceLost) {
        device.markLost(LostCause::Driver, where);
    }
}

// detail::FrameJob (design M1b §5.1): a fire-and-forget coroutine that awaits one frame's
// submission. It runs at once (suspend_never), suspends in co_await submit(), is resumed on the
// presenter thread by the mailbox once the GPU has finished, and frees its frame when it returns.
// The body catches everything, so unhandled_exception() is unreachable.
struct FrameJob {
    struct promise_type {
        FrameJob get_return_object() noexcept { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::terminate(); }
    };
};

// Pointers, not references: the coroutine frame keeps its parameters. The core outlives the job
// (its thread never exits with a frame in flight); the context outlives the core.
FrameJob runFrame(PresenterCore* core, ExecutionContext* context, IExecutor* mailbox,
                  RecordFunction record) {
    Result<std::uint64_t> result{};
    try {
        result = co_await submit(*context, *mailbox, std::move(record));
    } catch (const std::bad_alloc&) {
        // SubmitOperation::await_suspend could not queue the submission (design M1b §5.1).
        result = makeError(Errc::OutOfMemory, "queuing a presenter frame: out of memory");
    } catch (const std::exception& error) {
        result = makeError(Errc::Internal,
                           std::format("queuing a presenter frame failed: {}", error.what()));
    } catch (...) {
        result = makeError(Errc::Internal, "queuing a presenter frame failed");
    }
    core->frameFinished(std::move(result));
}

}  // namespace

std::optional<VkSurfaceFormatKHR> chooseSurfaceFormat(
    std::span<const VkSurfaceFormatKHR> formats) noexcept {
    for (const VkSurfaceFormatKHR& format : formats) {
        if ((format.format == VK_FORMAT_B8G8R8A8_UNORM ||
             format.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    if (formats.empty()) {
        return std::nullopt;
    }
    return formats.front();
}

std::uint32_t swapchainImageCount(const VkSurfaceCapabilitiesKHR& caps) noexcept {
    std::uint32_t count = std::max(caps.minImageCount + 1, 2U);
    if (caps.maxImageCount != 0) {
        count = std::min(count, caps.maxImageCount);
    }
    return count;
}

VkExtent2D swapchainExtent(const VkSurfaceCapabilitiesKHR& caps,
                           platform::PixelSize hint) noexcept {
    if (caps.currentExtent.width != UNDEFINED_EXTENT) {
        return caps.currentExtent;
    }
    return {
        .width = std::clamp(hint.width, caps.minImageExtent.width, caps.maxImageExtent.width),
        .height = std::clamp(hint.height, caps.minImageExtent.height, caps.maxImageExtent.height)};
}

VkCompositeAlphaFlagBitsKHR compositeAlpha(VkCompositeAlphaFlagsKHR supported) noexcept {
    for (const VkCompositeAlphaFlagBitsKHR bit :
         {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
          VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
        if ((supported & bit) != 0) {
            return bit;
        }
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;  // the specification requires at least one bit
}

void PresenterMailbox::post(MoveOnlyFunction<void()> work) {
    core_->post(std::move(work));
}

PresenterCore::PresenterCore(ExecutionContext& context, std::shared_ptr<DeviceCore> device,
                             const InstanceCore& instance, PresenterOptions options,
                             FrameProvider frames)
    : context_(&context),
      device_(std::move(device)),
      instance_(static_cast<VkInstance>(instance.instance())),
      get_instance_proc_addr_(instance.dispatch().vkGetInstanceProcAddr),
      window_(platform::windowHandleOf(options.window)),
      frames_(std::move(frames)),
      on_error_(std::move(options.onError)),
      name_(std::move(options.name)),
      mailbox_executor_(*this) {
    // The presenter starts with one pending request (design M1b §5.1): requestSeq 1, snapshot 0.
    requests_ = {.extent = options.extent, .visible = options.visible, .requestSeq = 1};
    stats_.hidden = !options.visible || options.extent.empty();
}

PresenterCore::~PresenterCore() {
    stop();
    releaseVulkanObjects();
}

Result<void> PresenterCore::initialise() {
    // Stays C: surface creators are reached through vkGetInstanceProcAddr (no vk-bootstrap surface
    // helpers).
    const auto surface = detail::createSurface(instance_, get_instance_proc_addr_, window_);
    if (!surface) {
        return vkError(surface.error(),
                       std::format("creating the window surface of presenter '{}'", name_));
    }
    surface_ = *surface;
    const auto& api = device_->dispatch();
    const vk::PhysicalDevice physical = device_->physicalDevice();
    const vk::SurfaceKHR surface_handle(surface_);

    // The out-parameter form: the value form copies an unset Bool32 when the call fails.
    vk::Bool32 supported = vk::False;
    DVV_TRY_VOID(vkCheck(
        physical.getSurfaceSupportKHR(device_->queueFamily(), surface_handle, &supported, api),
        "vkGetPhysicalDeviceSurfaceSupportKHR"));
    if (supported == vk::False) {
        return makeError(Errc::Unsupported,
                         std::format("presenter '{}': the queue of device {} cannot present to "
                                     "the window",
                                     name_, device_->id()));
    }
    const auto caps = physical.getSurfaceCapabilitiesKHR(surface_handle, api);
    DVV_TRY_VOID(vkCheck(caps.result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
    if (!(caps.value.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)) {
        return makeError(Errc::Unsupported,
                         std::format("presenter '{}': swapchain images of device {} cannot be "
                                     "transfer destinations",
                                     name_, device_->id()));
    }
    // The count-and-fill form, not the vector one: that one reads its count unset when the first
    // call fails.
    std::vector<vk::SurfaceFormatKHR> listed_formats;
    vk::Result listed = vk::Result::eIncomplete;
    while (listed == vk::Result::eIncomplete) {
        std::uint32_t count = 0;
        DVV_TRY_VOID(vkCheck(physical.getSurfaceFormatsKHR(surface_handle, &count, nullptr, api),
                             "vkGetPhysicalDeviceSurfaceFormatsKHR"));
        listed_formats.resize(count);
        listed = physical.getSurfaceFormatsKHR(surface_handle, &count, listed_formats.data(), api);
        listed_formats.resize(count);
    }
    DVV_TRY_VOID(vkCheck(listed, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
    const std::vector<VkSurfaceFormatKHR> formats(listed_formats.begin(), listed_formats.end());
    const std::optional<VkSurfaceFormatKHR> format = chooseSurfaceFormat(formats);
    if (!format) {
        return makeError(Errc::Unsupported,
                         std::format("presenter '{}': the window has no surface format", name_));
    }
    if (!presentBlitDestination(format->format, device_->formatFeatures(format->format))) {
        return makeError(
            Errc::Unsupported,
            std::format("presenter '{}': the swapchain format {} cannot take the present blit "
                        "(needs UNORM or SRGB with BLIT_DST)",
                        name_, vk::to_string(static_cast<vk::Format>(format->format))));
    }
    format_ = *format;
    encoding_ = presentEncoding(format_.format);
    const auto fence = device_->device().createFence(vk::FenceCreateInfo{}, nullptr, api);
    DVV_TRY_VOID(vkCheck(fence.result, "vkCreateFence"));
    fence_ = fence.value;
    device_->setDebugName(fence_, name_ + " acquire fence");
    return {};
}

void PresenterCore::start() {
    thread_ = std::jthread([this] { run(); });
}

void PresenterCore::stop() noexcept {
    if (!thread_.joinable()) {
        return;
    }
    if (thread_.get_id() == std::this_thread::get_id()) {
        DVV_LOG_ERROR("presenter '{}' destroyed on its own thread", name_);
        std::terminate();
    }
    {
        const std::lock_guard lock(mutex_);
        requests_.stop = true;
    }
    wake_.notify_all();
    // R15: the presenter thread may be inside vkAcquireNextImageKHR or vkQueuePresentKHR, waiting
    // for a message that the WSI sends to this thread (the window's thread).
    joinDispatchingSentMessages(thread_);
}

void PresenterCore::resize(platform::PixelSize extent) noexcept {
    {
        const std::lock_guard lock(mutex_);
        requests_.extent = extent;
        ++requests_.requestSeq;
    }
    wake_.notify_one();
}

void PresenterCore::setVisible(bool visible) noexcept {
    {
        const std::lock_guard lock(mutex_);
        requests_.visible = visible;
        ++requests_.requestSeq;  // setVisible(true) requests a frame (design M1b §5.1)
    }
    wake_.notify_one();
}

void PresenterCore::requestFrame() noexcept {
    {
        const std::lock_guard lock(mutex_);
        ++requests_.requestSeq;
    }
    wake_.notify_one();
}

PresenterStats PresenterCore::stats() const {
    const std::lock_guard lock(mutex_);
    return stats_;
}

std::optional<Error> PresenterCore::failure() const {
    const std::lock_guard lock(mutex_);
    return failure_;
}

void PresenterCore::post(MoveOnlyFunction<void()> work) {
    // Notified under the mutex: the waiter thread posts the last completion, after which the
    // presenter thread may exit and the GUI thread destroy the core. Notifying after the unlock
    // could touch wake_ while it is being destroyed.
    const std::lock_guard lock(mutex_);
    mailbox_.push_back(std::move(work));
    wake_.notify_one();
}

void PresenterCore::frameFinished(Result<std::uint64_t> result) noexcept {
    completion_ = std::move(result);
}

// design M1b §5.2: one iteration per wake-up. The thread never exits while a frame is in flight, so
// a suspended FrameJob is never destroyed (SubmitOperation contract).
void PresenterCore::run() noexcept {
    std::vector<MoveOnlyFunction<void()>> work;
    while (true) {
        {
            std::unique_lock lock(mutex_);
            waitForWork(lock);
            snapshot_ = requests_;
            work.swap(mailbox_);  // mailbox_ gets the empty vector back: no allocation
        }
        logic_.setRequests(snapshot_.extent, snapshot_.visible, snapshot_.requestSeq,
                           snapshot_.stop);
        for (auto& item : work) {
            item();  // a frame completion: resumes its FrameJob, which calls frameFinished()
        }
        work.clear();
        processCompletion();
        const Action action = logic_.next();
        if (action == Action::Exit) {
            teardown();
            publishState();
            return;
        }
        // An exception here is an allocation failure outside the guarded spots (swapchain image
        // list, debug names, error texts). No frame is in flight then: startFrame() is noexcept.
        try {
            step(action);
        } catch (const std::bad_alloc&) {
            fail(makeError(Errc::OutOfMemory, "presenter: out of memory").error());
            logic_.acquired(VkOutcome::OtherError);
        } catch (const std::exception& error) {
            fail(makeError(Errc::Internal, std::format("presenter '{}': {}", name_, error.what()))
                     .error());
            logic_.acquired(VkOutcome::OtherError);
        } catch (...) {
            fail(makeError(Errc::Internal, std::format("presenter '{}': unknown exception", name_))
                     .error());
            logic_.acquired(VkOutcome::OtherError);
        }
        publishState();
    }
}

void PresenterCore::waitForWork(std::unique_lock<std::mutex>& lock) {
    const Action due = logic_.next();  // with the previous snapshot
    if (due != Action::Wait && due != Action::WaitAndRetry) {
        return;  // RecreateSwapchain, RenderFrame, Present or Exit: no wait
    }
    // While stopping with a frame in flight, only its completion matters (design M1b §5.2).
    const bool completion_only = snapshot_.stop && in_flight_;
    const auto woken = [this, completion_only] {
        if (!mailbox_.empty()) {
            return true;
        }
        if (completion_only) {
            return false;
        }
        return requests_.requestSeq != snapshot_.requestSeq || requests_.stop != snapshot_.stop;
    };
    if (due == Action::WaitAndRetry) {
        // design M1b §5.2: the 10 ms after an acquire timeout. Only stop ends it early; requests
        // are served after it (no frame is in flight, so the mailbox stays empty).
        wake_.wait_for(lock, RETRY_DELAY, [this] { return requests_.stop; });
        logic_.retryElapsed();
        return;
    }
    wake_.wait(lock, woken);
}

void PresenterCore::step(Action action) {
    switch (action) {
        case Action::Wait:
        case Action::WaitAndRetry: return;  // waitForWork() waited and called retryElapsed()
        case Action::RecreateSwapchain: recreateSwapchain(); return;
        case Action::RenderFrame: renderFrame(); return;
        case Action::Present: present(); return;
        case Action::Exit: return;  // run() handles it
    }
}

// design M1b §5.2 RenderFrame, steps 1-5.
void PresenterCore::renderFrame() {
    const PresentFrameParams params{.extent = {.width = extent_.width, .height = extent_.height},
                                    .format = format_.format,
                                    .frameNumber = frame_number_++,
                                    .previousFrameCompleted = logic_.previousFrameCompleted(),
                                    .encoding = encoding_};
    Result<PresentFrame> provided = callProvider(params);  // step 1
    if (provided) {
        if (auto usable = checkImage(provided->image); !usable) {   // step 2
            provided = std::unexpected(std::move(usable).error());  // the frame is released unrun
        }
    }
    if (!provided && (provided.error().code == Errc::DeviceLost || device_->lost())) {
        // A lost device is terminal, not a skipped frame: fail once and request no more frames
        // (the same path as failSwapchain).
        fail(provided.error().code == Errc::DeviceLost ? std::move(provided).error()
                                                       : device_->lostError());
        logic_.acquired(VkOutcome::DeviceLost);
        return;
    }
    if (!provided) {
        reportFrameError(provided.error());
        logic_.frameSkipped();  // the request counts as served; nothing was acquired
        return;
    }
    std::uint32_t index = 0;
    const VkOutcome acquired = acquire(index);  // steps 3 and 4
    logic_.acquired(acquired);
    if (acquired != VkOutcome::Success && acquired != VkOutcome::Suboptimal) {
        return;  // dropped: the record function is destroyed here without running
    }
    logic_.frameStarted();  // step 5
    startFrame(std::move(*provided), index);
}

Result<PresentFrame> PresenterCore::callProvider(const PresentFrameParams& params) {
    try {
        return frames_(params);
    } catch (const std::bad_alloc&) {
        return makeError(
            Errc::OutOfMemory,
            std::format("the frame provider of presenter '{}' ran out of memory", name_));
    } catch (const std::exception& error) {
        return makeError(Errc::Internal, std::format("the frame provider of presenter '{}' "
                                                     "threw: {}",
                                                     name_, error.what()));
    } catch (...) {
        return makeError(Errc::Internal,
                         std::format("the frame provider of presenter '{}' threw an unknown "
                                     "exception",
                                     name_));
    }
}

// design M1b §5.2 step 2: refuses exactly what CommandRecorder::presentBlit refuses (the same
// device and generation, TRANSFER_SRC usage, presentBlitSource()), so an unusable image skips the
// frame instead of failing the submission and with it the presenter.
Result<void> PresenterCore::checkImage(const std::shared_ptr<GpuImage>& image) {
    if (!image) {
        return makeError(Errc::Internal,
                         std::format("presenter '{}': the frame has no image", name_));
    }
    if (&Access::core(*image) != device_.get() || image->generation() != device_->generation()) {
        return makeError(Errc::Internal,
                         std::format("presenter '{}': image '{}' belongs to another context", name_,
                                     image->name()));
    }
    if ((vulkanDesc(*image).usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
        return makeError(Errc::Internal, std::format("presenter '{}': image '{}' lacks "
                                                     "VK_IMAGE_USAGE_TRANSFER_SRC_BIT",
                                                     name_, image->name()));
    }
    const VkFormat format = vulkanFormat(*image);
    if (!presentBlitSource(format, device_->formatFeatures(format))) {
        return makeError(
            Errc::Internal,
            std::format("presenter '{}': the format {} of image '{}' cannot be "
                        "blitted to the swapchain",
                        name_, vk::to_string(static_cast<vk::Format>(format)), image->name()));
    }
    return {};
}

// design M1b §5.2 steps 3 and 4: acquire with the fence, then wait for it on the host.
VkOutcome PresenterCore::acquire(std::uint32_t& index) {
    const auto& api = device_->dispatch();
    const vk::Device device = device_->device();
    // The out-parameter form: it returns every result unchecked (eErrorOutOfDateKHR too), and the
    // value form copies an unset index when the call fails.
    const vk::Result result =
        device.acquireNextImageKHR(swapchain_, ACQUIRE_TIMEOUT_NS, {}, fence_, &index, api);
    switch (result) {
        case vk::Result::eSuccess:
        case vk::Result::eSuboptimalKHR: break;
        case vk::Result::eTimeout:
        case vk::Result::eNotReady: return VkOutcome::Timeout;
        case vk::Result::eErrorOutOfDateKHR: return VkOutcome::OutOfDate;
        case vk::Result::eErrorSurfaceLostKHR: fail(surfaceLost()); return VkOutcome::SurfaceLost;
        case vk::Result::eErrorDeviceLost:
            markIfLost(*device_, result, "vkAcquireNextImageKHR");
            fail(vkError(result, "vkAcquireNextImageKHR").error());
            return VkOutcome::DeviceLost;
        default:
            fail(vkError(result, "vkAcquireNextImageKHR").error());
            return VkOutcome::OtherError;
    }
    // Binding (design M1b §5.2 step 4): once an image is acquired, wait for the fence until it
    // signals or the device is lost, also when stop is set; stop is acted on only afterwards.
    const auto deadline = std::chrono::steady_clock::now() + FENCE_CAP;
    while (true) {
        const vk::Result waited = device.waitForFences(fence_, vk::True, FENCE_SLICE_NS, api);
        if (waited == vk::Result::eSuccess) {
            break;
        }
        if (waited != vk::Result::eTimeout) {
            markIfLost(*device_, waited, "vkWaitForFences (acquire fence)");
            fail(vkError(waited, "vkWaitForFences (acquire fence)").error());
            return waited == vk::Result::eErrorDeviceLost ? VkOutcome::DeviceLost
                                                          : VkOutcome::OtherError;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            fence_leaked_ = true;  // destroying a fence the driver may still signal is invalid
            fail(makeError(Errc::Internal, "acquire fence did not signal").error());
            return VkOutcome::OtherError;
        }
    }
    if (const vk::Result reset = device.resetFences(fence_, api); reset != vk::Result::eSuccess) {
        markIfLost(*device_, reset, "vkResetFences");
        fail(vkError(reset, "vkResetFences").error());
        return reset == vk::Result::eErrorDeviceLost ? VkOutcome::DeviceLost
                                                     : VkOutcome::OtherError;
    }
    return result == vk::Result::eSuccess ? VkOutcome::Success : VkOutcome::Suboptimal;
}

// design M1b §5.2 step 5: hand the frame to the submit thread through a FrameJob.
void PresenterCore::startFrame(PresentFrame frame, std::uint32_t index) noexcept {
    in_flight_ = true;
    acquired_index_ = index;
    try {
        // D6: blit into the swapchain image, at the swapchain's real extent, and move it to
        // PRESENT_SRC_KHR (Task 8).
        RecordFunction record = [client = std::move(frame.record), image = std::move(frame.image),
                                 target = static_cast<VkImage>(images_[index]),
                                 extent = extent_](CommandRecorder& recorder) mutable {
            if (client) {
                client(recorder);
            }
            Access::presentBlit(recorder, *image, target, extent);
        };
        runFrame(this, context_, &mailbox_executor_, std::move(record));
    } catch (const std::bad_alloc&) {
        completion_ = makeError(Errc::OutOfMemory, "starting a presenter frame: out of memory");
    } catch (const std::exception& error) {
        completion_ = makeError(Errc::Internal,
                                std::format("starting a presenter frame failed: {}", error.what()));
    } catch (...) {
        completion_ =
            makeError(Errc::Internal, "starting a presenter frame failed: unknown exception");
    }
    processCompletion();  // a frame that could not be queued has completed already
}

// design M1b §5.2 step 6, first half: the logic learns the frame's outcome.
void PresenterCore::processCompletion() noexcept {
    if (!completion_) {
        return;
    }
    Result<std::uint64_t> result = std::move(*completion_);
    completion_.reset();
    in_flight_ = false;
    if (result) {
        logic_.frameCompleted(std::nullopt);
        return;
    }
    const Errc code = result.error().code;
    fail(std::move(result).error());  // DeviceLost, Internal or OutOfMemory: Failed (design M1b §7)
    logic_.frameCompleted(code);
}

// design M1b §5.2 step 6, second half: present without wait semaphores (D4), under the queue lock.
void PresenterCore::present() {
    const vk::PresentInfoKHR info{.waitSemaphoreCount = 0,
                                  .pWaitSemaphores = nullptr,
                                  .swapchainCount = 1,
                                  .pSwapchains = &swapchain_,
                                  .pImageIndices = &acquired_index_,
                                  .pResults = nullptr};
    vk::Result result = vk::Result::eSuccess;
    {
        const auto queue_lock = device_->lockQueue();
        // A result, not an assert, for eErrorOutOfDateKHR too: VULKAN_HPP_ASSERT_ON_RESULT is a
        // no-op (vk.h).
        result = device_->queue().presentKHR(info, device_->dispatch());
    }
    VkOutcome outcome = VkOutcome::Success;
    switch (result) {
        case vk::Result::eSuccess: break;
        case vk::Result::eSuboptimalKHR: outcome = VkOutcome::Suboptimal; break;
        case vk::Result::eErrorOutOfDateKHR: outcome = VkOutcome::OutOfDate; break;
        case vk::Result::eErrorSurfaceLostKHR:
            fail(surfaceLost());
            outcome = VkOutcome::SurfaceLost;
            break;
        case vk::Result::eErrorDeviceLost:
            markIfLost(*device_, result, "vkQueuePresentKHR");  // after the queue-lock scope
            fail(vkError(result, "vkQueuePresentKHR").error());
            outcome = VkOutcome::DeviceLost;
            break;
        default:
            fail(vkError(result, "vkQueuePresentKHR").error());
            outcome = VkOutcome::OtherError;
            break;
    }
    if (outcome == VkOutcome::Success || outcome == VkOutcome::Suboptimal) {
        const std::lock_guard lock(mutex_);
        ++stats_.framesPresented;
        stats_.presentedExtent = {.width = extent_.width, .height = extent_.height};
    }
    logic_.presented(outcome);
}

// design M1b §5.3: only in phase Idle (the logic asks for it only then).
void PresenterCore::recreateSwapchain() {
    const auto& api = device_->dispatch();
    const vk::Device device = device_->device();
    const auto caps =
        device_->physicalDevice().getSurfaceCapabilitiesKHR(vk::SurfaceKHR(surface_), api);
    if (caps.result != vk::Result::eSuccess) {
        failSwapchain(caps.result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        return;
    }
    const VkExtent2D extent = swapchainExtent(caps.value, snapshot_.extent);
    if (extent.width == 0 || extent.height == 0) {
        logic_.surfaceHidden();  // step 2: keep the old swapchain and the request
        return;
    }
    const vk::SwapchainCreateInfoKHR info{
        .surface = vk::SurfaceKHR(surface_),
        .minImageCount = swapchainImageCount(caps.value),
        .imageFormat = static_cast<vk::Format>(format_.format),
        .imageColorSpace = static_cast<vk::ColorSpaceKHR>(format_.colorSpace),
        .imageExtent = {.width = extent.width, .height = extent.height},
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = caps.value.currentTransform,
        .compositeAlpha = static_cast<vk::CompositeAlphaFlagBitsKHR>(compositeAlpha(
            static_cast<VkCompositeAlphaFlagsKHR>(caps.value.supportedCompositeAlpha))),
        .presentMode = vk::PresentModeKHR::eFifo,
        .clipped = vk::True,
        .oldSwapchain = swapchain_};
    const auto created = device.createSwapchainKHR(info, nullptr, api);
    if (created.result != vk::Result::eSuccess) {
        // The old swapchain is retired all the same; teardown() destroys it.
        failSwapchain(created.result, "vkCreateSwapchainKHR");
        return;
    }
    const vk::SwapchainKHR old = std::exchange(swapchain_, created.value);
    images_.clear();
    extent_ = extent;
    if (old) {
        // D7: presentation is done with the old images once the queue is idle (accepted practice
        // without VK_KHR_swapchain_maintenance1; exact retirement is F3).
        vk::Result idle = vk::Result::eSuccess;
        {
            const auto queue_lock = device_->lockQueue();
            idle = device_->queue().waitIdle(api);
        }
        device.destroySwapchainKHR(old, nullptr, api);
        if (idle != vk::Result::eSuccess) {
            failSwapchain(idle, "vkQueueWaitIdle");
            return;
        }
    }
    // The count-and-fill form, not the vector one: that one reads its count unset when the first
    // call fails, and it would retry an eIncomplete that is an error here.
    std::uint32_t count = 0;
    vk::Result listed = device.getSwapchainImagesKHR(swapchain_, &count, nullptr, api);
    if (listed == vk::Result::eSuccess) {
        images_.resize(count);
        listed = device.getSwapchainImagesKHR(swapchain_, &count, images_.data(), api);
    }
    if (listed != vk::Result::eSuccess) {
        failSwapchain(listed, "vkGetSwapchainImagesKHR");
        return;
    }
    device_->setDebugName(swapchain_, name_ + " swapchain");
    for (std::size_t i = 0; i < images_.size(); ++i) {
        device_->setDebugName(images_[i], std::format("{} swapchain image {}", name_, i));
    }
    {
        const std::lock_guard lock(mutex_);
        ++stats_.swapchainsCreated;
    }
    DVV_LOG_DEBUG("presenter '{}': swapchain {}x{}, {} images", name_, extent.width, extent.height,
                  images_.size());
    logic_.swapchainCreated(snapshot_.extent);  // stores the hint used; internal re-request
}

void PresenterCore::failSwapchain(vk::Result result, std::string_view what) {
    if (result == vk::Result::eErrorSurfaceLostKHR) {
        fail(surfaceLost());
        logic_.acquired(VkOutcome::SurfaceLost);  // plan choice: see the logic contract, rule 6
        return;
    }
    markIfLost(*device_, result, what);
    fail(vkError(result, std::format("{} (presenter '{}')", what, name_)).error());
    logic_.acquired(result == vk::Result::eErrorDeviceLost ? VkOutcome::DeviceLost
                                                           : VkOutcome::OtherError);
}

// design M1b §5.5: Exit. No frame is in flight here.
void PresenterCore::teardown() noexcept {
    vk::Result idle = vk::Result::eSuccess;
    {
        const auto queue_lock = device_->lockQueue();
        idle = device_->queue().waitIdle(device_->dispatch());
    }
    markIfLost(*device_, idle, "vkQueueWaitIdle");
    if (idle != vk::Result::eSuccess) {
        DVV_LOG_ERROR("presenter '{}': vkQueueWaitIdle failed while closing: {}", name_,
                      vk::to_string(idle));
    }
    releaseVulkanObjects();
}

void PresenterCore::releaseVulkanObjects() noexcept {
    const auto& api = device_->dispatch();
    const vk::Device device = device_->device();
    if (swapchain_) {
        device.destroySwapchainKHR(swapchain_, nullptr, api);
        swapchain_ = nullptr;
    }
    images_.clear();
    if (fence_ && !fence_leaked_) {
        device.destroyFence(fence_, nullptr, api);
    }
    fence_ = nullptr;
    if (surface_ != VK_NULL_HANDLE) {
        vk::Instance(instance_).destroySurfaceKHR(vk::SurfaceKHR(surface_), nullptr, api);
        surface_ = VK_NULL_HANDLE;
    }
}

// Failed is terminal (design M1b §7): the first error is kept and reported once.
void PresenterCore::fail(Error error) noexcept {
    {
        const std::lock_guard lock(mutex_);
        if (failure_) {
            return;
        }
        failure_ = std::move(error);
        stats_.failed = true;
    }
    // failure_ is written once, so reading it unlocked from here on is safe.
    DVV_LOG_ERROR("presenter '{}' stopped presenting: {}", name_, failure_->describe());
    if (on_error_) {
        on_error_(*failure_);
    }
}

void PresenterCore::reportFrameError(const Error& error) {
    {
        const std::lock_guard lock(mutex_);
        ++stats_.frameErrors;
    }
    DVV_LOG_WARN("presenter '{}' skipped a frame: {}", name_, error.describe());
    if (on_error_) {
        on_error_(error);
    }
}

void PresenterCore::publishState() {
    const bool hidden = !snapshot_.visible || snapshot_.extent.empty() || logic_.hidden();
    const bool failed = logic_.failed();
    const std::lock_guard lock(mutex_);
    stats_.hidden = hidden;
    stats_.failed = stats_.failed || failed;
}

}  // namespace dvv::gpu::detail
