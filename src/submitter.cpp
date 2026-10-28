#include "submitter.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <string_view>
#include <system_error>
#include <utility>

#include "device_core.h"
#include "dvv/base/log.h"
#include "dvv/gpu/command_recorder.h"
#include "vk_result.h"

namespace dvv::gpu::detail {

namespace {

constexpr std::uint64_t WAIT_SLICE_NS = 1'000'000'000;  // re-check the lost flag every second

// A DeviceLost error without allocating a message (the out-of-memory fallback).
Error bareDeviceLost() noexcept {
    Error error;
    error.code = Errc::DeviceLost;
    return error;
}

// core.lostError(), or bareDeviceLost() when its message cannot be allocated.
Error lostErrorOf(const DeviceCore& core) noexcept {
    try {
        return core.lostError();
    } catch (...) {
        return bareDeviceLost();
    }
}

// Waits until the timeline reaches `value`, in slices of WAIT_SLICE_NS: after a slice that timed
// out, a lost device ends the wait (eErrorDeviceLost); otherwise it goes on. VK_ERROR_DEVICE_LOST
// from the wait marks the device lost. Returns eSuccess when the value is reached.
vk::Result waitForTimeline(DeviceCore& core, std::uint64_t value, const char* where) noexcept {
    const vk::Semaphore timeline = core.timeline();
    const vk::SemaphoreWaitInfo wait{
        .semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &value};
    while (true) {
        const vk::Result waited =
            core.device().waitSemaphores(wait, WAIT_SLICE_NS, core.dispatch());
        if (waited == vk::Result::eTimeout) {
            if (core.lost()) {
                return vk::Result::eErrorDeviceLost;
            }
            continue;  // a long submission; the device is fine
        }
        if (waited == vk::Result::eErrorDeviceLost) {
            core.markLost(LostCause::Driver, where);
        }
        return waited;
    }
}

}  // namespace

void SubmissionQueue::push(std::unique_ptr<Submission> submission) noexcept {
    Submission* const last = submission.get();
    if (tail_ == nullptr) {
        head_ = std::move(submission);
    } else {
        tail_->next = std::move(submission);
    }
    tail_ = last;
}

std::unique_ptr<Submission> SubmissionQueue::pop() noexcept {
    std::unique_ptr<Submission> first = std::move(head_);
    if (first) {
        head_ = std::move(first->next);
        if (!head_) {
            tail_ = nullptr;
        }
    }
    return first;
}

Result<std::unique_ptr<Submitter>> Submitter::create(const std::shared_ptr<DeviceCore>& core) {
    const vk::CommandPoolCreateInfo pool_info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = core->queueFamily()};
    const auto pool = core->device().createCommandPool(pool_info, nullptr, core->dispatch());
    DVV_TRY_VOID(vkCheck(pool.result, "vkCreateCommandPool"));
    std::unique_ptr<Submitter> submitter;
    try {
        submitter = std::make_unique<Submitter>(core, pool.value);
    } catch (...) {
        core->device().destroyCommandPool(pool.value, nullptr, core->dispatch());
        throw;
    }
    try {
        submitter->submit_thread_ =
            std::jthread([raw = submitter.get()] { raw->runSubmitThread(); });
        submitter->waiter_thread_ =
            std::jthread([raw = submitter.get()] { raw->runWaiterThread(); });
    } catch (const std::system_error& error) {
        return makeError(Errc::Internal,
                         std::format("starting the submit threads of device {} failed: {}",
                                     core->id(), error.what()));
    }
    return submitter;
}

Submitter::Submitter(std::shared_ptr<DeviceCore> core, vk::CommandPool pool) noexcept
    : core_(std::move(core)), pool_(pool) {}

Submitter::~Submitter() {
    stop();
    // Every submission has finished (stop() waited for it), so no command buffer is in use.
    core_->device().destroyCommandPool(pool_, nullptr, core_->dispatch());
}

void Submitter::enqueue(std::unique_ptr<Submission> submission) noexcept {
    // Notified under the lock: once it is released, the submission may complete and its
    // coroutine may release the context, and with it this object.
    const std::scoped_lock lock(submit_mutex_);
    queued_.push(std::move(submission));
    submit_ready_.notify_one();
}

void Submitter::stop() noexcept {
    const std::scoped_lock stop_lock(stop_mutex_);
    {
        const std::scoped_lock lock(submit_mutex_);
        stopping_ = true;
        submit_ready_.notify_all();
    }
    if (submit_thread_.joinable()) {
        submit_thread_.join();  // after it has submitted everything queued
    }
    {
        const std::scoped_lock lock(wait_mutex_);
        submit_thread_done_ = true;
        wait_ready_.notify_all();
    }
    if (waiter_thread_.joinable()) {
        waiter_thread_.join();  // after it has completed everything submitted
    }
}

void Submitter::runSubmitThread() {
    while (true) {
        std::unique_ptr<Submission> submission;
        {
            std::unique_lock lock(submit_mutex_);
            submit_ready_.wait(lock, [this] { return stopping_ || !queued_.empty(); });
            if (queued_.empty()) {
                return;  // stopping, and everything queued is submitted
            }
            submission = queued_.pop();
        }
        submitOne(*submission);
        const std::scoped_lock lock(wait_mutex_);
        in_flight_.push(std::move(submission));
        wait_ready_.notify_one();
    }
}

void Submitter::submitOne(Submission& submission) noexcept {
    // A lost device runs nothing more: no timeline value is reserved and nothing is recorded; the
    // waiter completes the submission without waiting (design M3 §4.1). A loss set right after
    // this check is harmless: an injection keeps a healthy device, a real loss fails the submit.
    if (core_->lost()) {
        submission.error = lostErrorOf(*core_);
        return;
    }
    submission.value = core_->reserveSubmitValue();
    try {
        recordAndSubmit(submission);
    } catch (...) {
        // Out of memory outside the record function (which has its own handler). The error has no
        // message, which would need memory too.
        if (!submission.error) {
            Error out_of_memory;
            out_of_memory.code = Errc::OutOfMemory;
            submission.error = std::move(out_of_memory);
        }
        if (!submission.signalled) {
            signalOnly(submission);
        }
    }
}

void Submitter::recordAndSubmit(Submission& submission) {
    auto acquired = acquireCommandBuffer();
    if (!acquired) {
        submission.error = std::move(acquired).error();
        submitBatch(submission, {});
        return;
    }
    const vk::CommandBuffer commands = *acquired;
    const vk::Result begun = commands.begin(
        {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}, core_->dispatch());
    if (begun != vk::Result::eSuccess) {
        submission.error = vkError(begun, "vkBeginCommandBuffer").error();
        free_buffers_.push_back(commands);  // never reallocates (capacity reserved)
        submitBatch(submission, {});
        return;
    }

    CommandRecorder recorder(*core_, static_cast<VkCommandBuffer>(commands));
    if (submission.record) {
        try {
            submission.record(recorder);
        } catch (const std::exception& error) {
            submission.error =
                makeError(Errc::Internal, std::format("recording submission {} "
                                                      "failed: {}",
                                                      submission.value, error.what()))
                    .error();
        } catch (...) {
            submission.error =
                makeError(Errc::Internal, std::format("recording submission {} failed with an "
                                                      "unknown exception",
                                                      submission.value))
                    .error();
        }
    }
    if (const std::optional<Error>& failed = recorder.error(); failed && !submission.error) {
        submission.error = *failed;
    }
    if (!submission.error) {
        recorder.finish();
    }
    const vk::Result ended = commands.end(core_->dispatch());
    if (ended != vk::Result::eSuccess && !submission.error) {
        submission.error = vkError(ended, "vkEndCommandBuffer").error();
    }

    // A failed recording submits only its signal: none of its commands run.
    if (submitBatch(submission, submission.error ? vk::CommandBuffer{} : commands)) {
        recorder.applyLayouts();
        busy_buffers_.push_back({.value = submission.value, .buffer = commands});
    } else {
        free_buffers_.push_back(commands);
    }
    // The images the recorder kept are released with the submission, on its executor.
    submission.images = std::move(recorder.layouts_);
}

bool Submitter::submitBatch(Submission& submission, vk::CommandBuffer commands) {
    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = commands};
    const vk::SemaphoreSubmitInfo signal{.semaphore = core_->timeline(),
                                         .value = submission.value,
                                         .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
    vk::SubmitInfo2 batch{.commandBufferInfoCount = commands ? 1U : 0U,
                          .pCommandBufferInfos = &command_info,
                          .signalSemaphoreInfoCount = 1,
                          .pSignalSemaphoreInfos = &signal};
    vk::Result result = vk::Result::eSuccess;
    bool device_lost = false;
    {
        const auto lock = core_->lockQueue();
        result = core_->queue().submit2(batch, nullptr, core_->dispatch());
        if (result == vk::Result::eSuccess) {
            markSignalled(submission, lock);
            return static_cast<bool>(commands);
        }
        device_lost = result == vk::Result::eErrorDeviceLost;
        if (commands && !device_lost) {
            // The signal alone keeps the timeline going for the submissions after this one.
            batch.commandBufferInfoCount = 0;
            const vk::Result signalled = core_->queue().submit2(batch, nullptr, core_->dispatch());
            if (signalled == vk::Result::eSuccess) {
                markSignalled(submission, lock);
            }
            device_lost = signalled == vk::Result::eErrorDeviceLost;
        }
    }
    // After the queue lock: markLost logs.
    if (device_lost) {
        core_->markLost(LostCause::Driver, "vkQueueSubmit2");
    }
    // Nothing can signal on a lost device; its waits end instead of hanging.
    if (!submission.signalled && !core_->lost()) {
        hostSignal(submission);
    }
    if (!submission.error) {
        submission.error =
            core_->lost() ? lostErrorOf(*core_) : vkError(result, "vkQueueSubmit2").error();
    }
    if (!submission.signalled && !core_->lost()) {
        DVV_LOG_ERROR("device {}: timeline value {} cannot be signalled: {}", core_->id(),
                      submission.value, vk::to_string(result));
    }
    return false;
}

void Submitter::signalOnly(Submission& submission) noexcept {
    const vk::SemaphoreSubmitInfo signal{.semaphore = core_->timeline(),
                                         .value = submission.value,
                                         .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
    const vk::SubmitInfo2 batch{.signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal};
    vk::Result result = vk::Result::eSuccess;
    {
        const auto lock = core_->lockQueue();
        result = core_->queue().submit2(batch, nullptr, core_->dispatch());
        if (result == vk::Result::eSuccess) {
            markSignalled(submission, lock);
            return;
        }
    }
    if (result == vk::Result::eErrorDeviceLost) {
        core_->markLost(LostCause::Driver, "vkQueueSubmit2");
    }
    if (core_->lost()) {
        if (!submission.error) {
            submission.error = lostErrorOf(*core_);
        }
        return;  // nothing can signal on a lost device
    }
    hostSignal(submission);
}

void Submitter::hostSignal(Submission& submission) noexcept {
    // A host signal must not overtake a pending signal: first wait until the latest value whose
    // signal was submitted is reached. On a lost device the wait ends (DEVICE_LOST, or the lost
    // flag after a timed-out slice), and nothing is signalled.
    const vk::Semaphore timeline = core_->timeline();
    if (waitForTimeline(*core_, last_signalled_, "vkWaitSemaphores") != vk::Result::eSuccess) {
        return;
    }
    const vk::SemaphoreSignalInfo signal{.semaphore = timeline, .value = submission.value};
    if (core_->device().signalSemaphore(signal, core_->dispatch()) == vk::Result::eSuccess) {
        const auto lock = core_->lockQueue();
        markSignalled(submission, lock);
    }
}

void Submitter::markSignalled(Submission& submission,
                              const std::unique_lock<std::mutex>& queue_lock) noexcept {
    submission.signalled = true;
    last_signalled_ = submission.value;
    core_->setSubmitted(submission.value, queue_lock);
}

Result<vk::CommandBuffer> Submitter::acquireCommandBuffer() {
    if (free_buffers_.empty() && !busy_buffers_.empty()) {
        DVV_TRY(const std::uint64_t completed, core_->completedValue());
        // Busy buffers are in submission order.
        const auto running = std::ranges::find_if(
            busy_buffers_, [completed](const BusyBuffer& busy) { return busy.value > completed; });
        for (auto busy = busy_buffers_.begin(); busy != running; ++busy) {
            free_buffers_.push_back(busy->buffer);
        }
        busy_buffers_.erase(busy_buffers_.begin(), running);
    }
    if (!free_buffers_.empty()) {
        const vk::CommandBuffer buffer = free_buffers_.back();
        free_buffers_.pop_back();
        return buffer;
    }
    // Both lists can hold every buffer, so moving buffers between them never reallocates.
    const std::size_t total = busy_buffers_.size() + 1;
    busy_buffers_.reserve(total);
    free_buffers_.reserve(total);
    const vk::CommandBufferAllocateInfo allocate_info{
        .commandPool = pool_, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    vk::CommandBuffer buffer;
    DVV_TRY_VOID(
        vkCheck(core_->device().allocateCommandBuffers(&allocate_info, &buffer, core_->dispatch()),
                "vkAllocateCommandBuffers"));
    return buffer;
}

void Submitter::runWaiterThread() {
    while (true) {
        std::unique_ptr<Submission> submission;
        {
            std::unique_lock lock(wait_mutex_);
            wait_ready_.wait(lock, [this] { return submit_thread_done_ || !in_flight_.empty(); });
            if (in_flight_.empty()) {
                return;  // the submit thread has stopped, and everything submitted is complete
            }
            submission = in_flight_.pop();
        }
        // Bounded: a lost device ends the wait within one slice (design M3 §4.1).
        if (submission->signalled) {
            const vk::Result waited =
                waitForTimeline(*core_, submission->value, "vkWaitSemaphores");
            if (waited != vk::Result::eSuccess && !submission->error && !core_->lost()) {
                submission->error = vkError(waited, "vkWaitSemaphores").error();
            }
        }
        core_->collect();
        // Every completion after the loss reports DeviceLost instead of its value; an error the
        // submission already had (e.g. a throwing record function) stays.
        if (core_->lost() && !submission->error) {
            submission->error = lostErrorOf(*core_);
        }
        complete(std::move(submission));
    }
}

void Submitter::complete(std::unique_ptr<Submission> submission) {
    IExecutor& executor = *submission->resumeOn;
    // The posted work owns the submission; until a post succeeds, this function does.
    Submission* const posted = submission.release();
    // The executor could not queue the work (out of memory, or whatever its post() throws).
    // Resuming the coroutine here would run it on the waiter thread, so the post is retried until
    // it succeeds.
    bool reported = false;
    const auto retry_later = [this, posted, &reported](std::string_view reason) {
        if (!reported) {
            DVV_LOG_ERROR("device {}: completing submission {} failed, retrying: {}", core_->id(),
                          posted->value, reason);
            reported = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    while (true) {
        try {
            executor.post([posted] {
                std::unique_ptr<Submission> owned(posted);
                const std::coroutine_handle<> continuation = owned->continuation;
                if (owned->error) {
                    *owned->result = std::unexpected(std::move(*owned->error));
                } else {
                    *owned->result = owned->value;
                }
                owned.reset();  // releases the record function and everything it captured
                continuation.resume();
            });
            return;
        } catch (const std::exception& error) {
            retry_later(error.what());
        } catch (...) {
            retry_later("an unknown exception");
        }
    }
}

}  // namespace dvv::gpu::detail
