#include "DebugSDFRenderer.hpp"
#include "../ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>
#include <iostream>

DebugSDFRenderer::DebugSDFRenderer() {}

DebugSDFRenderer::~DebugSDFRenderer() { cleanup(); }

void DebugSDFRenderer::init(VulkanApp* app) {
    createCubeBuffers(app);
    createDescriptorSet(app);

    vertModule = app->createShaderModule(FileReader::readFile("shaders/debug_sdf.vert.spv"));
    fragModule = app->createShaderModule(FileReader::readFile("shaders/debug_sdf.frag.spv"));

    ShaderStage vertStage(vertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragStage(fragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayout> setLayouts = {
        app->getDescriptorSetLayout(),
        descriptorSetLayout
    };

    auto [pipelineHandle, layoutHandle] = app->createGraphicsPipeline(
        { vertStage.info, fragStage.info },
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{0, sizeof(CubeVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        },
        {
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(CubeVertex, position)},
            VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32_UINT, offsetof(CubeVertex, cornerIndex)}
        },
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true,
        true,
        VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );

    pipeline = pipelineHandle;
    pipelineLayout = layoutHandle;
}

void DebugSDFRenderer::createCubeBuffers(VulkanApp* app) {
    const std::vector<CubeVertex> vertices = {
        {{0.0f, 0.0f, 0.0f}, 0}, {{0.0f, 0.0f, 1.0f}, 1}, {{0.0f, 1.0f, 1.0f}, 3}, {{0.0f, 1.0f, 0.0f}, 2},
        {{1.0f, 0.0f, 0.0f}, 4}, {{1.0f, 1.0f, 0.0f}, 6}, {{1.0f, 1.0f, 1.0f}, 7}, {{1.0f, 0.0f, 1.0f}, 5},
        {{0.0f, 0.0f, 0.0f}, 0}, {{1.0f, 0.0f, 0.0f}, 4}, {{1.0f, 0.0f, 1.0f}, 5}, {{0.0f, 0.0f, 1.0f}, 1},
        {{0.0f, 1.0f, 0.0f}, 2}, {{0.0f, 1.0f, 1.0f}, 3}, {{1.0f, 1.0f, 1.0f}, 7}, {{1.0f, 1.0f, 0.0f}, 6},
        {{0.0f, 0.0f, 0.0f}, 0}, {{0.0f, 1.0f, 0.0f}, 2}, {{1.0f, 1.0f, 0.0f}, 6}, {{1.0f, 0.0f, 0.0f}, 4},
        {{0.0f, 0.0f, 1.0f}, 1}, {{1.0f, 0.0f, 1.0f}, 5}, {{1.0f, 1.0f, 1.0f}, 7}, {{0.0f, 1.0f, 1.0f}, 3}
    };

    std::vector<uint32_t> indices;
    indices.reserve(36);
    for (uint32_t face = 0; face < 6; ++face) {
        const uint32_t base = face * 4;
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }

    vertexBuffer = app->createDeviceLocalBuffer(vertices.data(),
        vertices.size() * sizeof(CubeVertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    indexBuffer = app->createDeviceLocalBuffer(indices.data(),
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    indexCount = static_cast<uint32_t>(indices.size());
}

void DebugSDFRenderer::createDescriptorSet(VulkanApp* app) {
    VkDescriptorSetLayoutBinding instanceBinding{};
    instanceBinding.binding = 0;
    instanceBinding.descriptorCount = 1;
    instanceBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    instanceBinding.pImmutableSamplers = nullptr;
    instanceBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &instanceBinding;

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("DebugSDFRenderer: failed to create descriptor set layout");
    }
    app->resources.addDescriptorSetLayout(descriptorSetLayout, "DebugSDFRenderer: descriptorSetLayout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("DebugSDFRenderer: failed to create descriptor pool");
    }
    app->resources.addDescriptorPool(descriptorPool, "DebugSDFRenderer: descriptorPool");

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("DebugSDFRenderer: failed to allocate descriptor set");
    }
    app->resources.addDescriptorSet(descriptorSet, "DebugSDFRenderer: descriptorSet");

    instanceBufferCapacity = 128;
    instanceBuffer = app->createBuffer(
        instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4) * 3),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = instanceBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);
}

void DebugSDFRenderer::updateInstanceBuffer(VulkanApp* app) {
    if (activeCubes.empty()) return;

    struct InstanceData {
        glm::mat4 model;
        glm::vec4 sdf0;
        glm::vec4 sdf1;
        glm::vec4 meta; // meta.x = brushIndex
    };

    if (activeCubes.size() > instanceBufferCapacity) {
        instanceBuffer = {};
        instanceBufferCapacity = static_cast<uint32_t>(activeCubes.size() * 2);
        instanceBuffer = app->createBuffer(
            instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4) * 3),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = instanceBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);
    }

    std::vector<InstanceData> instanceData;
    instanceData.reserve(activeCubes.size());
    for (const CubeSDF& cube : activeCubes) {
        InstanceData inst{};
        inst.model = glm::translate(glm::mat4(1.0f), cube.cube.getMin())
                   * glm::scale(glm::mat4(1.0f), cube.cube.getLength());
        inst.sdf0 = glm::vec4(cube.sdf[0], cube.sdf[1], cube.sdf[2], cube.sdf[3]);
        inst.sdf1 = glm::vec4(cube.sdf[4], cube.sdf[5], cube.sdf[6], cube.sdf[7]);
        inst.meta = glm::vec4(static_cast<float>(cube.brushIndex), 0.0f, 0.0f, 0.0f);
        instanceData.push_back(inst);
    }

    void* data = nullptr;
    vkMapMemory(app->getDevice(), instanceBuffer.memory, 0,
        instanceData.size() * sizeof(InstanceData), 0, &data);
    std::memcpy(data, instanceData.data(), instanceData.size() * sizeof(InstanceData));
    vkUnmapMemory(app->getDevice(), instanceBuffer.memory);
}

void DebugSDFRenderer::setCubes(const std::vector<CubeSDF>& cubes) {
    activeCubes = cubes;
}

void DebugSDFRenderer::render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet mainDescriptorSet) {
    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE ||
        activeCubes.empty() || vertexBuffer.buffer == VK_NULL_HANDLE ||
        indexBuffer.buffer == VK_NULL_HANDLE || indexCount == 0) {
        return;
    }

    updateInstanceBuffer(app);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(app->getWidth());
    viewport.height = static_cast<float>(app->getHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(app->getWidth()), static_cast<uint32_t>(app->getHeight())};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDescriptorSet descriptorSets[] = {mainDescriptorSet, descriptorSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
        0, 2, descriptorSets, 0, nullptr);

    const VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, static_cast<uint32_t>(activeCubes.size()), 0, 0, 0);
}

void DebugSDFRenderer::cleanup() {
    pipeline = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
    vertexBuffer = {};
    indexBuffer = {};
    indexCount = 0;
    descriptorSetLayout = VK_NULL_HANDLE;
    descriptorPool = VK_NULL_HANDLE;
    descriptorSet = VK_NULL_HANDLE;
    instanceBuffer = {};
    instanceBufferCapacity = 0;
    activeCubes.clear();
}
