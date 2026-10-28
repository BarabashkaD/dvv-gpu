#include "dvv/gpu/testing/fault_injection.h"

#include "access.h"
#include "device_core.h"
#include "dvv/gpu/execution_context.h"

namespace dvv::gpu::testing {

void injectDeviceLost(ExecutionContext& context) noexcept {
    detail::Access::core(context).markLost(detail::LostCause::Injected, "injected");
}

}  // namespace dvv::gpu::testing
