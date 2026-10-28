#pragma once

// The JSON form of a GpuInfo (dvv-gpu-info --json). A header of its own: the JSON library is
// compiled only where it is used.

#include "dvv/base/json.h"
#include "dvv/gpu/gpu_info.h"

namespace dvv::gpu {

// Every field; versions as "major.minor.patch" text, the UUID as formatUuid() text.
[[nodiscard]] json::Value toJson(const GpuInfo& info);

}  // namespace dvv::gpu
