#pragma once

// The submit thread and the waiter thread of an ExecutionContext (spec §4). An internal part of
// the context (exempt from spec §9.4 registration: owned only by its context, created only by
// GpuRegistry::createContext()).
//
// Submit thread: owns the command pool; takes the queued submissions in order, reserves each one's
// timeline value, records it through a CommandRecorder and submits it (vkQueueSubmit2) with a
// signal of that value. A submission that fails to record or to submit still signals its value
// (see the contract below), so the timeline never stalls; it completes with the error. The record
// functions run here one at a time: they must not block (spec §6.5); the host signal's wait below
// is the one documented exception, on a path that only queue failures reach.
// Waiter thread: waits until the timeline reaches each submitted value in order, frees what the
// retire queue can free, and posts the completion to the submission's executor, which releases the
// record function and resumes the awaiting coroutine.
//
// Contract with DeviceCore:
// - Every reserved timeline value is signalled: by its batch, else by a batch with the signal
//   alone, else by the host once every earlier signal is reached (a host signal must not overtake
//   a pending one). Only a lost device leaves a value unsignalled; its waits then fail instead of
//   hanging.
// - Device loss (design M3 §4.1): both threads wait on the timeline in slices of one second and
//   end a wait when the device is lost (DeviceCore::lost(), set by VK_ERROR_DEVICE_LOST or an
//   injection). On a lost device a submission reserves no value and records nothing, and every
//   completion reports Errc::DeviceLost unless the submission already had an error.
// - No batch waits on a semaphore: nothing submitted can block the queue, so vkDeviceWaitIdle in
//   drain() always returns and every submitted value is reached by then.
// - Nothing is released while the queue lock is held, and neither thread releases the last
//   reference to the device core or, apart from the out-of-memory path, to a resource: the record
//   function, what it captured and the images whose layout the submission changed are released on
//   the submission's executor, and this object, which keeps the core, is destroyed by the context
//   after both threads have been joined (~DeviceCore takes the queue lock and joins nothing).

#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "vk.h"

namespace dvv::gpu::detail {

class DeviceCore;
struct Access;

// One ctx->submit() on its way through both threads. Allocated when it is queued, so neither
// thread allocates to hand it on.
struct Submission {
    RecordFunction record{};
    IExecutor* resumeOn = nullptr;
    std::coroutine_handle<> continuation{};
    Result<std::uint64_t>* result = nullptr;  // in the awaiting coroutine's frame

    // Set by the submit thread.
    std::uint64_t value = 0;     // the reserved timeline value
    bool signalled = false;      // value is signalled (its signal is on the queue, or the host's)
    std::optional<Error> error;  // why the submission failed
    // The images whose layout the submission changed (from its recorder), released with it.
    std::vector<std::pair<std::shared_ptr<GpuImage>, VkImageLayout>> images;

    std::unique_ptr<Submission> next;  // intrusive queue link
};

// First-in, first-out list of submissions; pushing and popping never allocate.
class SubmissionQueue {
public:
    SubmissionQueue() = default;
    SubmissionQueue(const SubmissionQueue&) = delete;
    SubmissionQueue& operator=(const SubmissionQueue&) = delete;
    SubmissionQueue(SubmissionQueue&&) = delete;
    SubmissionQueue& operator=(SubmissionQueue&&) = delete;
    // One at a time, so a long list cannot exhaust the stack through the links.
    ~SubmissionQueue() {
        while (pop() != nullptr) {
        }
    }

    [[nodiscard]] bool empty() const noexcept { return head_ == nullptr; }
    void push(std::unique_ptr<Submission> submission) noexcept;
    [[nodiscard]] std::unique_ptr<Submission> pop() noexcept;

private:
    std::unique_ptr<Submission> head_;
    Submission* tail_ = nullptr;
};

class Submitter {
public:
    // Creates the command pool and starts both threads.
    [[nodiscard]] static Result<std::unique_ptr<Submitter>> create(
        const std::shared_ptr<DeviceCore>& core);

    Submitter(const Submitter&) = delete;
    Submitter& operator=(const Submitter&) = delete;
    Submitter(Submitter&&) = delete;
    Submitter& operator=(Submitter&&) = delete;
    // stop(), then destroys the command pool.
    ~Submitter();

    // Queues a submission for the submit thread.
    void enqueue(std::unique_ptr<Submission> submission) noexcept;
    // Finishes every queued and in-flight submission, then joins both threads. Idempotent; must
    // not be called from either thread (the context is never released there: the record
    // functions and coroutines are released on their executors).
    void stop() noexcept;

    // Use create().
    Submitter(std::shared_ptr<DeviceCore> core, vk::CommandPool pool) noexcept;

private:
    friend struct Access;  // white-box tests reach hostSignal()

    void runSubmitThread();
    void runWaiterThread();
    // Reserves the timeline value, records and submits; the value is signalled whatever fails
    // (submit thread).
    void submitOne(Submission& submission) noexcept;
    void recordAndSubmit(Submission& submission);
    // Submits the recorded commands with the signal, or only the signal when there are no
    // commands or they cannot be submitted. True when the commands were submitted.
    bool submitBatch(Submission& submission, vk::CommandBuffer commands);
    // Submits only the signal, without allocating (the out-of-memory path).
    void signalOnly(Submission& submission) noexcept;
    // The last resort when the queue refuses even the signal alone (see the contract above).
    void hostSignal(Submission& submission) noexcept;
    void markSignalled(Submission& submission,
                       const std::unique_lock<std::mutex>& queue_lock) noexcept;
    // A command buffer whose earlier submission has finished, or a new one (submit thread).
    [[nodiscard]] Result<vk::CommandBuffer> acquireCommandBuffer();
    // Posts the completion to the submission's executor (waiter thread).
    void complete(std::unique_ptr<Submission> submission);

    std::shared_ptr<DeviceCore> core_;
    vk::CommandPool pool_;

    // Submit thread only: command buffers in use (with their timeline value) and free ones.
    struct BusyBuffer {
        std::uint64_t value = 0;
        vk::CommandBuffer buffer;
    };
    std::vector<BusyBuffer> busy_buffers_;
    std::vector<vk::CommandBuffer> free_buffers_;
    std::uint64_t last_signalled_ = 0;  // the latest value whose signal is submitted or done

    std::mutex submit_mutex_;
    std::condition_variable submit_ready_;
    SubmissionQueue queued_;
    bool stopping_ = false;

    std::mutex wait_mutex_;
    std::condition_variable wait_ready_;
    SubmissionQueue in_flight_;
    bool submit_thread_done_ = false;

    std::mutex stop_mutex_;
    std::jthread submit_thread_;
    std::jthread waiter_thread_;
};

}  // namespace dvv::gpu::detail
