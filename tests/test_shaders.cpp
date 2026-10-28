#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "dvv/base/json.h"
#include "dvv/gpu/embedded_shader.h"
#include "test_shaders/add_one.h"
#include "test_shaders/write_image.h"

TEST_CASE("an embedded shader holds SPIR-V 1.6 for its entry point") {
    const dvv::gpu::EmbeddedShader& shader = test_shaders::addOne();
    CHECK(shader.name == "add_one");
    REQUIRE(shader.spirv.size() > 5);
    CHECK(shader.spirv[0] == 0x07230203U);  // the SPIR-V magic number
    CHECK(shader.spirv[1] == 0x00010600U);  // version 1.6
    CHECK(shader.spirv[3] > 1);             // the id bound
    CHECK(shader.spirv[4] == 0);            // the reserved schema word
}

TEST_CASE("an embedded shader's reflection describes its entry point and push constants") {
    const dvv::gpu::EmbeddedShader& shader = test_shaders::addOne();
    const auto reflection = dvv::json::parse(shader.reflectionJson);
    REQUIRE(reflection.has_value());

    const auto& entry_points = reflection->at("entryPoints");
    REQUIRE(entry_points.size() == 1);
    CHECK(dvv::json::get<std::string>(entry_points[0], "name") == "main");
    CHECK(dvv::json::get<std::string>(entry_points[0], "stage") == "compute");
    CHECK(entry_points[0].at("threadGroupSize") == dvv::json::Value::parse("[64, 1, 1]"));

    const auto& parameters = reflection->at("parameters");
    REQUIRE(parameters.size() == 1);
    CHECK(dvv::json::get<std::string>(parameters[0], "name") == "pc");
    CHECK(dvv::json::get<std::string>(parameters[0].at("binding"), "kind") == "pushConstantBuffer");
    // The push constants are 16 bytes: an 8-byte address, a 4-byte count and padding.
    const auto& layout = parameters[0].at("type").at("elementVarLayout").at("binding");
    CHECK(dvv::json::get<std::uint32_t>(layout, "size") == 16);
}

TEST_CASE("write_image declares one storage image of unknown format") {
    const dvv::gpu::EmbeddedShader& shader = test_shaders::writeImage();
    const std::span<const std::uint32_t> words = shader.spirv;
    REQUIRE(words.size() > 5);

    constexpr std::uint32_t OP_CAPABILITY = 17;
    constexpr std::uint32_t OP_TYPE_IMAGE = 25;
    constexpr std::uint32_t STORAGE_IMAGE_WRITE_WITHOUT_FORMAT = 56;
    constexpr std::uint32_t SAMPLED_STORAGE = 2;  // read and written without a sampler
    constexpr std::uint32_t FORMAT_UNKNOWN = 0;
    int storage_images = 0;
    int images = 0;
    bool write_without_format = false;
    std::size_t at = 5;  // after the header
    while (at < words.size()) {
        const std::uint32_t length = words[at] >> 16U;
        const std::uint32_t opcode = words[at] & 0xFFFFU;
        REQUIRE(length > 0);
        REQUIRE(at + length <= words.size());
        if (opcode == OP_TYPE_IMAGE) {
            REQUIRE(length > 8);
            ++images;
            if (words[at + 7] == SAMPLED_STORAGE && words[at + 8] == FORMAT_UNKNOWN) {
                ++storage_images;
            }
        } else if (opcode == OP_CAPABILITY) {
            REQUIRE(length > 1);
            write_without_format =
                write_without_format || words[at + 1] == STORAGE_IMAGE_WRITE_WITHOUT_FORMAT;
        }
        at += length;
    }
    CHECK(images == 1);
    CHECK(storage_images == 1);
    CHECK(write_without_format);

    const auto reflection = dvv::json::parse(shader.reflectionJson);
    REQUIRE(reflection.has_value());
    int descriptors = 0;
    for (const auto& parameter : reflection->at("parameters")) {
        const auto& binding = parameter.at("binding");
        if (dvv::json::get<std::string>(binding, "kind") == "descriptorTableSlot") {
            ++descriptors;
            CHECK(dvv::json::get<std::uint32_t>(binding, "index") == 0);
        }
    }
    CHECK(descriptors == 1);
}
