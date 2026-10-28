// The one translation unit that compiles VMA's implementation. VMA is a SYSTEM include, so its
// warnings stay out of ours; vk.h sets its configuration.
#include "vk.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
