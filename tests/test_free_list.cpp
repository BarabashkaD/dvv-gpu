#include <doctest/doctest.h>

#include <optional>

#include "free_list.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::MemoryUsage;
using dvv::gpu::detail::BufferKey;
using dvv::gpu::detail::bufferKey;
using dvv::gpu::detail::FreeList;

TEST_CASE("a buffer key ignores the name and always has the device-address usage") {
    const BufferKey key = bufferKey({.size = 64,
                                     .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     .memory = MemoryUsage::HostUpload,
                                     .name = "a"});
    CHECK(key == bufferKey({.size = 64,
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            .memory = MemoryUsage::HostUpload,
                            .name = "b"}));
    CHECK(key != bufferKey({.size = 128,
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            .memory = MemoryUsage::HostUpload}));
    CHECK(key != bufferKey({.size = 64,
                            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            .memory = MemoryUsage::HostUpload}));
    CHECK(key != bufferKey({.size = 64,
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            .memory = MemoryUsage::DeviceOnly}));
}

TEST_CASE("the free list hands out matching items once the timeline has reached them") {
    const BufferKey small{.size = 64};
    const BufferKey large{.size = 128};
    FreeList<int> list;
    list.add(small, 1, 5);
    list.add(small, 2, 3);
    list.add(large, 3, 0);
    CHECK(list.size() == 3);
    CHECK(list.take(BufferKey{.size = 32}, 100) == std::nullopt);

    CHECK(list.take(small, 2) == std::nullopt);  // neither 5 nor 3 reached
    CHECK(list.take(small, 4) == 2);             // 3 reached, 5 not
    CHECK(list.take(small, 4) == std::nullopt);
    CHECK(list.take(small, 9) == 1);
    CHECK(list.take(small, 100) == std::nullopt);
    CHECK(list.take(large, 0) == 3);
    CHECK(list.size() == 0);
}

TEST_CASE("the free list hands out the oldest ready item first") {
    const BufferKey key{.size = 64};
    FreeList<int> list;
    list.add(key, 1, 1);
    list.add(key, 2, 2);
    CHECK(list.take(key, 10) == 1);
    CHECK(list.take(key, 10) == 2);
}

TEST_CASE("takeAll empties the free list") {
    FreeList<int> list;
    list.add(BufferKey{.size = 64}, 1, 0);
    list.add(BufferKey{.size = 128}, 2, 7);
    FreeList<int> all = list.takeAll();
    CHECK(all.size() == 2);
    CHECK(list.size() == 0);
    CHECK(list.take(BufferKey{.size = 64}, 100) == std::nullopt);
    CHECK(all.take(BufferKey{.size = 128}, 7) == 2);
}
