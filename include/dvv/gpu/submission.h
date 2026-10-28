#pragma once

// The awaitable of ExecutionContext::submit() (spec §4).

#include <coroutine>
#include <cstdint>
#include <functional>
#include <memory>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/command_recorder.h"

namespace dvv::gpu {

class ExecutionContext;

// The record function of a submission; runs on the context's submit thread.
using RecordFunction = std::move_only_function<void(CommandRecorder&)>;

// co_await ctx->submit(executor, record) hands the record function to the submit thread, which
// records and submits it; the waiter thread waits until the GPU has finished, releases the record
// function on the executor and resumes the awaiting coroutine there (never inline). The result is
// the submission's timeline value, or the error that stopped it (recording, submission or the
// device). Nothing happens until the operation is awaited; it keeps the context alive meanwhile.
// There is no cancellation: the awaiting coroutine must not be destroyed while it is suspended
// here (the waiter thread writes the result into its frame and resumes it).
class [[nodiscard]] SubmitOperation {
public:
    SubmitOperation(const SubmitOperation&) = delete;
    SubmitOperation& operator=(const SubmitOperation&) = delete;
    SubmitOperation(SubmitOperation&&) = delete;
    SubmitOperation& operator=(SubmitOperation&&) = delete;
    ~SubmitOperation() = default;

    [[nodiscard]] bool await_ready() const noexcept { return false; }
    // Queues the submission; throws std::bad_alloc when it cannot.
    void await_suspend(std::coroutine_handle<> continuation);
    [[nodiscard]] Result<std::uint64_t> await_resume() noexcept { return std::move(result_); }

private:
    friend class ExecutionContext;

    SubmitOperation(std::shared_ptr<ExecutionContext> context, IExecutor& resume_on,
                    RecordFunction record) noexcept;

    std::shared_ptr<ExecutionContext> context_;
    IExecutor* resume_on_;
    RecordFunction record_;
    Result<std::uint64_t> result_{};
};

}  // namespace dvv::gpu
