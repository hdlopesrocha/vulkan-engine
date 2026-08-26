#include "DebugSDFRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "../ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>
#include <iostream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vulkan/vulkan_core.h>
#include "../includes/locations.hpp"

// A storage buffer binding may not exceed VkPhysicalDeviceLimits::maxStorageBufferRange
// (typically 128 MiB). Cap the SDF-cube instance count so the buffer we expose to the
// shader stays within that hard limit, otherwise vkUpdateDescriptorSets aborts.
static uint32_t maxSDFInstanceCount(VulkanApp* app) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(app->getPhysicalDevice(), &props);
    const VkDeviceSize stride = sizeof(glm::mat4) + sizeof(glm::vec4) * 3;
    return static_cast<uint32_t>(props.limits.maxStorageBufferRange / stride);
}

DebugSDFRenderer::DebugSDFRenderer() {}

DebugSDFRenderer::~DebugSDFRenderer() { cleanup(nullptr); }

void DebugSDFRenderer::init(VulkanApp* app) {
    createCubeBuffers(app);
    createDescriptorSet(app);

    vertModule = app->getOrCreateShaderModule("shaders/debug_sdf.vert.spv");
    fragModule = app->getOrCreateShaderModule("shaders/debug_sdf.frag.spv");

    ShaderStage vertStage(vertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragStage(fragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayout> setLayouts = {
        app->getDescriptorSetLayout(),
        descriptorSetLayout
    };

    GraphicsPipelineConfig cfg{};
    cfg.cullMode = VK_CULL_MODE_NONE;
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    auto [pipelineHandle, layoutHandle] = app->createGraphicsPipeline(
        { vertStage.info, fragStage.info },
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{0, sizeof(CubeVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        },
        {
            VkVertexInputAttributeDescription{ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(CubeVertex, position)},
            VkVertexInputAttributeDescription{ATTR_COLOR, 0, VK_FORMAT_R32_UINT, offsetof(CubeVertex, cornerIndex)}
        },
        setLayouts,
        nullptr,
        cfg
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

    // Upload via the async transfer path (graphics-family transfer queue when
    // available, else main graphics queue). The completion semaphore is
    // registered so drawFrame waits for the copy before the buffers are first
    // consumed. A dedicated transfer-family queue is deliberately not used
    // (documented RADV/RENOIR GPUVM-instability safety net in VulkanApp.cpp).
    vertexBuffer = app->createDeviceLocalBufferAsync(vertices.data(),
        vertices.size() * sizeof(CubeVertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, nullptr);
    indexBuffer = app->createDeviceLocalBufferAsync(indices.data(),
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, nullptr);
    indexCount = static_cast<uint32_t>(indices.size());
}

void DebugSDFRenderer::createDescriptorSet(VulkanApp* app) {
    DescriptorAllocator descAlloc{app->getDevice(), app};

    VkDescriptorSetLayoutBinding instanceBinding{};
    instanceBinding.binding = 0;
    instanceBinding.descriptorCount = 1;
    instanceBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    instanceBinding.pImmutableSamplers = nullptr;
    instanceBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    descriptorSetLayout = descAlloc.createLayout(
        &instanceBinding, 1,
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        nullptr,
        "DebugSDFRenderer: descriptorSetLayout");

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    descriptorPool = descAlloc.createPool(
        &poolSize, 1, 1,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "DebugSDFRenderer: descriptorPool");

    descriptorSet = descAlloc.allocateSet(descriptorPool, descriptorSetLayout, "DebugSDFRenderer: descriptorSet");

    const uint32_t maxInstances = maxSDFInstanceCount(app);
    instanceBufferCapacity = std::min<uint32_t>(128, maxInstances);
    instanceBuffer = app->createBuffer(
        instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4) * 3),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    DescriptorWriter(app->getDevice())
        .writeBuffer(descriptorSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     instanceBuffer.buffer, 0, VK_WHOLE_SIZE)
        .flush();
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
        const uint32_t maxInstances = maxSDFInstanceCount(app);
        instanceBufferCapacity = static_cast<uint32_t>(activeCubes.size() * 2);
        instanceBufferCapacity = std::min(instanceBufferCapacity, maxInstances);
        instanceBuffer = app->createBuffer(
            instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4) * 3),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        DescriptorWriter(app->getDevice())
            .writeBuffer(descriptorSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         instanceBuffer.buffer, 0, VK_WHOLE_SIZE)
            .flush();
    }

    const uint32_t drawCount = std::min<uint32_t>(static_cast<uint32_t>(activeCubes.size()),
                                                  instanceBufferCapacity);
    std::vector<InstanceData> instanceData;
    instanceData.reserve(drawCount);
    for (uint32_t i = 0; i < drawCount; ++i) {
        const CubeSDF& cube = activeCubes[i];
        InstanceData inst{};
        inst.model = glm::translate(glm::mat4(1.0f), cube.cube.getMin())
                   * glm::scale(glm::mat4(1.0f), cube.cube.getLength());
        inst.sdf0 = glm::vec4(cube.sdf[0], cube.sdf[1], cube.sdf[2], cube.sdf[3]);
        inst.sdf1 = glm::vec4(cube.sdf[4], cube.sdf[5], cube.sdf[6], cube.sdf[7]);
        inst.meta = glm::vec4(static_cast<float>(cube.brushIndex), 0.0f, 0.0f, 0.0f);
        instanceData.push_back(inst);
    }

    std::memcpy(instanceBuffer.mappedData, instanceData.data(), instanceData.size() * sizeof(InstanceData));
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

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDescriptorSet descriptorSets[] = {mainDescriptorSet, descriptorSet};
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 2, descriptorSets, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
        0, 2, descriptorSets, 0, nullptr);

    const VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    const uint32_t drawCount = std::min<uint32_t>(static_cast<uint32_t>(activeCubes.size()),
                                                  instanceBufferCapacity);
    vkCmdDrawIndexed(cmd, indexCount, drawCount, 0, 0, 0);
}

void DebugSDFRenderer::cleanup(VulkanApp* app) {
    (void)app;
    vertexBuffer = {};
    indexBuffer = {};
    instanceBuffer = {};
    indexCount = 0;
    instanceBufferCapacity = 0;
    activeCubes.clear();
}

namespace {

constexpr float kDebugSDFClip = 10.0f;

const uint32_t kDebugSDFFaces[6][4] = {
    {0, 1, 3, 2},
    {4, 6, 7, 5},
    {0, 4, 5, 1},
    {2, 3, 7, 6},
    {0, 2, 6, 4},
    {1, 5, 7, 3}
};

bool isDrawableSDF(float v) {
    return std::isfinite(v) && std::abs(v) <= kDebugSDFClip;
}

bool hasDrawableSDFFace(const std::array<float, 8>& sdf) {
    for (const auto& face : kDebugSDFFaces) {
        for (uint32_t corner : face) {
            if (isDrawableSDF(sdf[corner])) {
                return true;
            }
        }

        for (uint32_t edge = 0; edge < 4; ++edge) {
            const float a = sdf[face[edge]];
            const float b = sdf[face[(edge + 1) % 4]];
            if (std::isfinite(a) && std::isfinite(b) && ((a <= 0.0f && b >= 0.0f) || (a >= 0.0f && b <= 0.0f))) {
                return true;
            }
        }
    }
    return false;
}

void collectLeafSDFCubes(OctreeNode* node, const BoundingCube& cube, OctreeAllocator& allocator,
                         std::vector<DebugSDFRenderer::CubeSDF>& out) {
    if (!node) return;

    if (node->getLod() == 1u) {
        DebugSDFRenderer::CubeSDF debugCube{};
        debugCube.cube = cube;
        for (size_t i = 0; i < debugCube.sdf.size(); ++i) {
            debugCube.sdf[i] = node->sdf[i];
        }
        debugCube.brushIndex = node->vertex.brushIndex;
        if (hasDrawableSDFFace(debugCube.sdf)) {
            out.push_back(debugCube);
            return;  // Parent covers this subtree — children are redundant
        }
        // Parent is simplified but has no drawable faces; traverse children
        // in case individual child leaves still have visible SDF faces.
    }

    ChildBlock* block = node->getBlock(allocator);
    if (!block) return;
    for (uint32_t i = 0; i < 8; ++i) {
        OctreeNode* child = block->get(i, allocator);
        if (child && child != node) {
            collectLeafSDFCubes(child, cube.getChild(i), allocator, out);
        }
    }
}

} // namespace

void DebugSDFRenderer::updateCubesForChunk(NodeID nid, const OctreeNodeData& nd, const Octree& tree) {
    if (!nd.node || !tree.allocator) return;

    std::vector<CubeSDF> cubes;
    collectLeafSDFCubes(nd.node, nd.cube, *tree.allocator, cubes);

    std::lock_guard<std::recursive_mutex> lock(cubesMutex);
    if (cubes.empty()) {
        nodeDebugSDFCubes.erase(nid);
    } else {
        nodeDebugSDFCubes[nid] = std::move(cubes);
    }
}

void DebugSDFRenderer::removeCubesForNode(NodeID id) {
    std::lock_guard<std::recursive_mutex> lock(cubesMutex);
    nodeDebugSDFCubes.erase(id);
}

void DebugSDFRenderer::clearCubes() {
    std::lock_guard<std::recursive_mutex> lock(cubesMutex);
    nodeDebugSDFCubes.clear();
}

std::vector<DebugSDFRenderer::CubeSDF> DebugSDFRenderer::getCubes() const {
    std::lock_guard<std::recursive_mutex> lock(cubesMutex);
    std::vector<CubeSDF> out;
    size_t total = 0;
    for (const auto& entry : nodeDebugSDFCubes) {
        total += entry.second.size();
    }
    out.reserve(total);
    for (const auto& entry : nodeDebugSDFCubes) {
        out.insert(out.end(), entry.second.begin(), entry.second.end());
    }
    return out;
}
