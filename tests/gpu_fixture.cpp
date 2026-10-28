#include "gpu_fixture.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <string>
#include <system_error>
#include <utility>

#include "dvv/base/task.h"

namespace dvv::gpu::test {

namespace {

Task<Result<std::uint64_t>> submitTask(ExecutionContext& context, IExecutor& executor,
                                       RecordFunction record) {
    co_return co_await context.submit(executor, std::move(record));
}

}  // namespace

TemporaryDirectory::TemporaryDirectory() {
    static std::atomic<int> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            std::format("dvv-gpu-test-{}-{}",
                        std::chrono::steady_clock::now().time_since_epoch().count(),
                        counter.fetch_add(1));
    std::filesystem::create_directories(path_);
}

TemporaryDirectory::~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
}

CapturedLog::CapturedLog(LogLevel level) {
    Logger::instance().clearSinks();
    Logger::instance().addSink(sink_);
    Logger::instance().setLevel(level);
}

CapturedLog::~CapturedLog() {
    Logger::instance().setLevel(LogLevel::Warn);
    Logger::instance().clearSinks();
    Logger::instance().addSink(std::make_shared<StderrSink>());
}

bool CapturedLog::contains(const std::string& text) const {
    const auto lines = sink_->lines();
    return std::ranges::any_of(
        lines, [&text](const std::string& line) { return line.find(text) != std::string::npos; });
}

Result<std::uint64_t> submitAndWait(ExecutionContext& context, IExecutor& executor,
                                    RecordFunction record) {
    return syncWait(submitTask(context, executor, std::move(record)));
}

GpuTest::GpuTest() {
    auto registry = GpuRegistry::create();
    if (!registry) {
        FAIL(registry.error().describe());
    }
    registry_ = std::move(*registry);
}

GpuTest::~GpuTest() {
    context_.reset();
    executor_.reset();
    CHECK(registry_->validationMessageCount() == 0);
}

DeviceId GpuTest::device() const {
    if (const auto best = registry_->bestDevice()) {
        return *best;
    }
    FAIL("no usable Vulkan device");
    return DeviceId(0);  // not reached: FAIL throws
}

ExecutionContext& GpuTest::context() {
    if (!context_) {
        auto context = registry_->createContext(device());
        if (!context) {
            FAIL(context.error().describe());
        }
        context_ = std::move(*context);
    }
    return *context_;
}

ThreadPool& GpuTest::executor() {
    if (!executor_) {
        executor_ = ExecutorFactory::makeThreadPool(1).value();
    }
    return *executor_;
}

Result<std::uint64_t> GpuTest::submitAndWait(RecordFunction record) {
    return test::submitAndWait(context(), executor(), std::move(record));
}

}  // namespace dvv::gpu::test
