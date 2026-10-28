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
#include "dvv/gpu/execution_context.h"
#include "dvv/platform/surface.h"
#include "instance.h"
#include "present_blit.h"
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
void markIfLost(DeviceCore& device, VkResult result, std::string_view where) noexcept {
    if (result == VK_ERROR_DEVICE_LOST) {
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
        result = co_await context->submit(*mailbox, std::move(record));
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

void PresenterMailbox::post(std::move_only_function<void()> work) {
    core_->post(std::move(work));
}

PresenterCore::PresenterCore(ExecutionContext& context, std::shared_ptr<DeviceCore> device,
                             const InstanceCore& instance, PresenterOptions options)
    : context_(&context),
      device_(std::move(device)),
      instance_(static_cast<VkInstance>(instance.instance())),
      get_instance_proc_addr_(instance.dispatch().vkGetInstanceProcAddr),
      window_(options.window),
      frames_(std::move(options.frames)),
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
    const auto surface = platform::createSurface(instance_, get_instance_proc_addr_, window_);
    if (!surface) {
        return vkError(surface.error(),
                       std::format("creating the window surface of presenter '{}'", name_));
    }
    surface_ = *surface;
    const auto& api = device_->dispatch();
    const auto physical = static_cast<VkPhysicalDevice>(device_->physicalDevice());

    VkBool32 supported = VK_FALSE;
    DVV_TRY_VOID(vkCheck(api.vkGetPhysicalDeviceSurfaceSupportKHR(physical, device_->queueFamily(),
                                                                  surface_, &supported),
                         "vkGetPhysicalDeviceSurfaceSupportKHR"));
    if (supported == VK_FALSE) {
        return makeError(Errc::Unsupported,
                         std::format("presenter '{}': the queue of device {} cannot present to "
                                     "the window",
                                     name_, device_->id()));
    }
    VkSurfaceCapabilitiesKHR caps{};
    DVV_TRY_VOID(vkCheck(api.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps),
                         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
    if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
        return makeError(Errc::Unsupported,
                         std::format("presenter '{}': swapchain images of device {} cannot be "
                                     "transfer destinations",
                                     name_, device_->id()));
    }
    std::vector<VkSurfaceFormatKHR> formats;
    VkResult listed = VK_INCOMPLETE;
    while (listed == VK_INCOMPLETE) {
        std::uint32_t count = 0;
        DVV_TRY_VOID(
            vkCheck(api.vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, nullptr),
                    "vkGetPhysicalDeviceSurfaceFormatsKHR"));
        formats.resize(count);
        listed =
            api.vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, formats.data());
        formats.resize(count);
    }
    DVV_TRY_VOID(vkCheck(listed, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
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
    const VkFenceCreateInfo fence_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0};
    DVV_TRY_VOID(vkCheck(
        api.vkCreateFence(static_cast<VkDevice>(device_->device()), &fence_info, nullptr, &fence_),
        "vkCreateFence"));
    device_->setDebugName(vk::Fence(fence_), name_ + " acquire fence");
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
#if defined(_WIN32)
    // R15: the presenter thread may be inside vkAcquireNextImageKHR or vkQueuePresentKHR, waiting
    // for a message that the WSI sends to this thread (the window's thread).
    waitForThreadDispatchingSentMessages(thread_.native_handle());
#endif
    thread_.join();
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

void PresenterCore::post(std::move_only_function<void()> work) {
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
    std::vector<std::move_only_function<void()>> work;
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
    const PresentFrameInfo info{.extent = {.width = extent_.width, .height = extent_.height},
                                .format = format_.format,
                                .frameNumber = frame_number_++,
                                .previousFrameCompleted = logic_.previousFrameCompleted()};
    Result<PresentFrame> provided = callProvider(info);  // step 1
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

Result<PresentFrame> PresenterCore::callProvider(const PresentFrameInfo& info) {
    try {
        return frames_(info);
    } catch (const std::bad_alloc&) {
        return makeError(
            Errc::OutOfMemory,
            std::format("the frame provider of presenter '{}' ran out of memory", name_));
    } catch (const std::exception& error) {
        return makeError(Errc::Internal, std::format("the frame provider of presenter '{}' "
                                                     "threw: {}",
                                                     name_, error.what()));
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
                                     image->desc().name));
    }
    if ((image->desc().usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
        return makeError(Errc::Internal, std::format("presenter '{}': image '{}' lacks "
                                                     "VK_IMAGE_USAGE_TRANSFER_SRC_BIT",
                                                     name_, image->desc().name));
    }
    const VkFormat format = image->format();
    if (!presentBlitSource(format, device_->formatFeatures(format))) {
        return makeError(
            Errc::Internal,
            std::format("presenter '{}': the format {} of image '{}' cannot be "
                        "blitted to the swapchain",
                        name_, vk::to_string(static_cast<vk::Format>(format)), image->desc().name));
    }
    return {};
}

// design M1b §5.2 steps 3 and 4: acquire with the fence, then wait for it on the host.
VkOutcome PresenterCore::acquire(std::uint32_t& index) {
    const auto& api = device_->dispatch();
    const auto device = static_cast<VkDevice>(device_->device());
    const VkResult result = api.vkAcquireNextImageKHR(device, swapchain_, ACQUIRE_TIMEOUT_NS,
                                                      VK_NULL_HANDLE, fence_, &index);
    switch (result) {
        case VK_SUCCESS:
        case VK_SUBOPTIMAL_KHR: break;
        case VK_TIMEOUT:
        case VK_NOT_READY: return VkOutcome::Timeout;
        case VK_ERROR_OUT_OF_DATE_KHR: return VkOutcome::OutOfDate;
        case VK_ERROR_SURFACE_LOST_KHR: fail(surfaceLost()); return VkOutcome::SurfaceLost;
        case VK_ERROR_DEVICE_LOST:
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
        const VkResult waited = api.vkWaitForFences(device, 1, &fence_, VK_TRUE, FENCE_SLICE_NS);
        if (waited == VK_SUCCESS) {
            break;
        }
        if (waited != VK_TIMEOUT) {
            markIfLost(*device_, waited, "vkWaitForFences (acquire fence)");
            fail(vkError(waited, "vkWaitForFences (acquire fence)").error());
            return waited == VK_ERROR_DEVICE_LOST ? VkOutcome::DeviceLost : VkOutcome::OtherError;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            fence_leaked_ = true;  // destroying a fence the driver may still signal is invalid
            fail(makeError(Errc::Internal, "acquire fence did not signal").error());
            return VkOutcome::OtherError;
        }
    }
    if (const VkResult reset = api.vkResetFences(device, 1, &fence_); reset != VK_SUCCESS) {
        markIfLost(*device_, reset, "vkResetFences");
        fail(vkError(reset, "vkResetFences").error());
        return reset == VK_ERROR_DEVICE_LOST ? VkOutcome::DeviceLost : VkOutcome::OtherError;
    }
    return result == VK_SUCCESS ? VkOutcome::Success : VkOutcome::Suboptimal;
}

// design M1b §5.2 step 5: hand the frame to the submit thread through a FrameJob.
void PresenterCore::startFrame(PresentFrame frame, std::uint32_t index) noexcept {
    in_flight_ = true;
    acquired_index_ = index;
    try {
        // D6: blit into the swapchain image, at the swapchain's real extent, and move it to
        // PRESENT_SRC_KHR (Task 8).
        RecordFunction record = [client = std::move(frame.record), image = std::move(frame.image),
                                 target = images_[index],
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
    const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .pNext = nullptr,
                                .waitSemaphoreCount = 0,
                                .pWaitSemaphores = nullptr,
                                .swapchainCount = 1,
                                .pSwapchains = &swapchain_,
                                .pImageIndices = &acquired_index_,
                                .pResults = nullptr};
    VkResult result = VK_SUCCESS;
    {
        const auto queue_lock = device_->lockQueue();
        result =
            device_->dispatch().vkQueuePresentKHR(static_cast<VkQueue>(device_->queue()), &info);
    }
    VkOutcome outcome = VkOutcome::Success;
    switch (result) {
        case VK_SUCCESS: break;
        case VK_SUBOPTIMAL_KHR: outcome = VkOutcome::Suboptimal; break;
        case VK_ERROR_OUT_OF_DATE_KHR: outcome = VkOutcome::OutOfDate; break;
        case VK_ERROR_SURFACE_LOST_KHR:
            fail(surfaceLost());
            outcome = VkOutcome::SurfaceLost;
            break;
        case VK_ERROR_DEVICE_LOST:
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
    const auto device = static_cast<VkDevice>(device_->device());
    VkSurfaceCapabilitiesKHR caps{};
    if (const VkResult result = api.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            static_cast<VkPhysicalDevice>(device_->physicalDevice()), surface_, &caps);
        result != VK_SUCCESS) {
        failSwapchain(result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        return;
    }
    const VkExtent2D extent = swapchainExtent(caps, snapshot_.extent);
    if (extent.width == 0 || extent.height == 0) {
        logic_.surfaceHidden();  // step 2: keep the old swapchain and the request
        return;
    }
    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface_,
        .minImageCount = swapchainImageCount(caps),
        .imageFormat = format_.format,
        .imageColorSpace = format_.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = caps.currentTransform,
        .compositeAlpha = compositeAlpha(caps.supportedCompositeAlpha),
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = swapchain_};
    VkSwapchainKHR created = VK_NULL_HANDLE;
    if (const VkResult result = api.vkCreateSwapchainKHR(device, &info, nullptr, &created);
        result != VK_SUCCESS) {
        // The old swapchain is retired all the same; teardown() destroys it.
        failSwapchain(result, "vkCreateSwapchainKHR");
        return;
    }
    const VkSwapchainKHR old = std::exchange(swapchain_, created);
    images_.clear();
    extent_ = extent;
    if (old != VK_NULL_HANDLE) {
        // D7: presentation is done with the old images once the queue is idle (accepted practice
        // without VK_KHR_swapchain_maintenance1; exact retirement is F3).
        VkResult idle = VK_SUCCESS;
        {
            const auto queue_lock = device_->lockQueue();
            idle = api.vkQueueWaitIdle(static_cast<VkQueue>(device_->queue()));
        }
        api.vkDestroySwapchainKHR(device, old, nullptr);
        if (idle != VK_SUCCESS) {
            failSwapchain(idle, "vkQueueWaitIdle");
            return;
        }
    }
    std::uint32_t count = 0;
    VkResult listed = api.vkGetSwapchainImagesKHR(device, swapchain_, &count, nullptr);
    if (listed == VK_SUCCESS) {
        images_.resize(count);
        listed = api.vkGetSwapchainImagesKHR(device, swapchain_, &count, images_.data());
    }
    if (listed != VK_SUCCESS) {
        failSwapchain(listed, "vkGetSwapchainImagesKHR");
        return;
    }
    device_->setDebugName(vk::SwapchainKHR(swapchain_), name_ + " swapchain");
    for (std::size_t i = 0; i < images_.size(); ++i) {
        device_->setDebugName(vk::Image(images_[i]),
                              std::format("{} swapchain image {}", name_, i));
    }
    {
        const std::lock_guard lock(mutex_);
        ++stats_.swapchainsCreated;
    }
    DVV_LOG_DEBUG("presenter '{}': swapchain {}x{}, {} images", name_, extent.width, extent.height,
                  images_.size());
    logic_.swapchainCreated(snapshot_.extent);  // stores the hint used; internal re-request
}

void PresenterCore::failSwapchain(VkResult result, std::string_view what) {
    if (result == VK_ERROR_SURFACE_LOST_KHR) {
        fail(surfaceLost());
        logic_.acquired(VkOutcome::SurfaceLost);  // plan choice: see the logic contract, rule 6
        return;
    }
    markIfLost(*device_, result, what);
    fail(vkError(result, std::format("{} (presenter '{}')", what, name_)).error());
    logic_.acquired(result == VK_ERROR_DEVICE_LOST ? VkOutcome::DeviceLost : VkOutcome::OtherError);
}

// design M1b §5.5: Exit. No frame is in flight here.
void PresenterCore::teardown() noexcept {
    VkResult idle = VK_SUCCESS;
    {
        const auto queue_lock = device_->lockQueue();
        idle = device_->dispatch().vkQueueWaitIdle(static_cast<VkQueue>(device_->queue()));
    }
    markIfLost(*device_, idle, "vkQueueWaitIdle");
    if (idle != VK_SUCCESS) {
        DVV_LOG_ERROR("presenter '{}': vkQueueWaitIdle failed while closing: {}", name_,
                      vk::to_string(static_cast<vk::Result>(idle)));
    }
    releaseVulkanObjects();
}

void PresenterCore::releaseVulkanObjects() noexcept {
    const auto& api = device_->dispatch();
    const auto device = static_cast<VkDevice>(device_->device());
    if (swapchain_ != VK_NULL_HANDLE) {
        api.vkDestroySwapchainKHR(device, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    images_.clear();
    if (fence_ != VK_NULL_HANDLE && !fence_leaked_) {
        api.vkDestroyFence(device, fence_, nullptr);
    }
    fence_ = VK_NULL_HANDLE;
    if (surface_ != VK_NULL_HANDLE) {
        api.vkDestroySurfaceKHR(instance_, surface_, nullptr);
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
