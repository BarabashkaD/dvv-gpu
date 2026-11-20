// As a consumer that declared the class ahead against 0.2.1: the forward declaration comes first,
// then gpu_fwd.h, gpu_instance.h and resources.h.
namespace dvv::gpu {
class GpuRegistry;
}  // namespace dvv::gpu

#include "dvv/gpu/gpu_fwd.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/resources.h"

// The order is what this test checks, so clang-format must not move the dvv headers down.
// clang-format off
#include <type_traits>

#include <doctest/doctest.h>
// clang-format on

TEST_CASE("a forward declaration of GpuRegistry and gpu_fwd.h coexist with the definitions") {
    CHECK(std::is_same_v<dvv::gpu::GpuRegistry, dvv::gpu::GpuInstance>);
}
