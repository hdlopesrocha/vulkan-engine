#include "DebugSDFRenderer.hpp"
#include "IndirectRenderer.hpp"
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
DebugSDFRenderer::DebugSDFRenderer() {}

DebugSDFRenderer::~DebugSDFRenderer() { cleanup(nullptr); }

void DebugSDFRenderer::init(VulkanApp* app) {
    createCubeBuffers(app);
    createDescriptorSet(app);

    vertModule = app->getOrCreateShaderModule("shaders/debug_sdf.vert.spv");
    fragModule = app->getOrCreateShaderModule("shaders/debug_sdf.frag.spv");

    createCullResources(app);

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
    // Binding 0 (per-cube instance payload) is written each frame in render()
    // from the current cull frame's instance buffer (cf.instance), so no static
    // buffer is needed here.
}

void DebugSDFRenderer::createCullResources(VulkanApp* app) {
    cullApp_ = app;

    cmdDrawIndexedIndirectCount =
        (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCountKHR");
    if (!cmdDrawIndexedIndirectCount)
        cmdDrawIndexedIndirectCount =
            (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCount");
    if (!cmdDrawIndexedIndirectCount)
        throw std::runtime_error("DebugSDFRenderer: vkCmdDrawIndexedIndirectCountKHR not available");
}

void DebugSDFRenderer::ensureCullCapacity(uint32_t frame, uint32_t cubeCount) {
    CullFrame& cf = cullFrames[frame];
    if (cubeCount <= cf.capacity) return;

    // Grow with headroom; destroy old buffers (deferred to frame fence by app).
    uint32_t newCap = cubeCount + cubeCount / 4 + 64;
    // The instance buffer is bound with VK_WHOLE_SIZE, which must not exceed
    // maxStorageBufferRange. Clamp so the descriptor write stays valid.
    const size_t maxFit = static_cast<size_t>(cullApp_->getMaxStorageBufferRange()) / sizeof(InstanceData);
    if (newCap > maxFit) newCap = static_cast<uint32_t>(maxFit);
    VulkanApp* app = nullptr; // not needed; use stored device via resources
    (void)app;

    auto makeBuf = [&](VkDeviceSize bytes, VkBufferUsageFlags usage) -> Buffer {
        // app pointer is captured lazily; ensureCullCapacity is only called from
        // prepareCull which has the command buffer but not the app. We stash the
        // app in createCullResources via a member.
        return cullApp_->createBuffer(bytes, usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    };

    if (cf.instance.buffer != VK_NULL_HANDLE) cullApp_->resources.removeBufferVma(cf.instance.buffer, cf.instance.allocation);

    cf.instance = makeBuf(newCap * sizeof(InstanceData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    cf.capacity = newCap;
}

void DebugSDFRenderer::writeCullFrameData(uint32_t frame, uint32_t cubeCount) {
    CullFrame& cf = cullFrames[frame];
    if (cf.capacity == 0 || cf.instance.mappedData == nullptr)
        return;

    auto* inst = static_cast<InstanceData*>(cf.instance.mappedData);

    const uint32_t n = std::min(cubeCount, cf.capacity);
    for (uint32_t i = 0; i < n; i++) {
        const CubeSDF& cube = activeCubes[i];

        glm::vec3 minp = cube.cube.getMin();
        glm::vec3 len = cube.cube.getLength();

        inst[i].model = glm::translate(glm::mat4(1.0f), minp) * glm::scale(glm::mat4(1.0f), len);
        inst[i].sdf0 = glm::vec4(cube.sdf[0], cube.sdf[1], cube.sdf[2], cube.sdf[3]);
        inst[i].sdf1 = glm::vec4(cube.sdf[4], cube.sdf[5], cube.sdf[6], cube.sdf[7]);
        inst[i].meta = glm::vec4(static_cast<float>(cube.brushIndex), 0.0f, 0.0f, 0.0f);
    }
}

void DebugSDFRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj,
                                    glm::vec3 camPos, float lodBias, int maxTargetLod) {
    // NOTE: the SDF cube frustum cull + compaction is performed by the SOLID
    // IndirectRenderer's prepareCull (indirect.comp) in the SAME dispatch as the
    // terrain. Here we only upload the per-cube instance payload (model + sdf
    // values) so the SDF vertex shader can read instances[localIdx]; the visible
    // DrawCmd stream + count live in the terrain IR's SDF output buffers.
    if (pipeline == VK_NULL_HANDLE) return;
    if (activeCubes.empty()) {
        hasCubes_ = false;
        return;
    }
    hasCubes_ = true;

    const uint32_t f = currentCullFrame % SDF_CULL_FRAMES;
    const uint32_t count = static_cast<uint32_t>(activeCubes.size());
    ensureCullCapacity(f, count);
    writeCullFrameData(f, count); // writes cf.instance (indexed by cube order == SDF-local index)

    CullFrame& cf = cullFrames[f];

    // HOST writes (instance payload) → VERTEX reads.
    VkBufferMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    b.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = cf.instance.buffer;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void DebugSDFRenderer::setCubes(const std::vector<CubeSDF>& cubes) {
    activeCubes = cubes;
}

void DebugSDFRenderer::render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet mainDescriptorSet) {
    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE ||
        !hasCubes_ || vertexBuffer.buffer == VK_NULL_HANDLE ||
        indexBuffer.buffer == VK_NULL_HANDLE || indexCount == 0) {
        return;
    }

    const uint32_t f = currentCullFrame % SDF_CULL_FRAMES;
    CullFrame& cf = cullFrames[f];
    // The visible SDF DrawCmd stream + count are produced by the solid
    // IndirectRenderer's merged indirect.comp dispatch (folded into the terrain cull).
    VkBuffer sdfCompact = VK_NULL_HANDLE;
    VkBuffer sdfCount = VK_NULL_HANDLE;
    if (terrainIR_) {
        sdfCompact = terrainIR_->getSdfCompactBuffer(f);
        sdfCount = terrainIR_->getSdfCountBuffer(f);
    }
    if (sdfCompact == VK_NULL_HANDLE || sdfCount == VK_NULL_HANDLE ||
        cf.instance.buffer == VK_NULL_HANDLE)
        return;

    // Point the instance descriptor set at this frame's instance buffer.
    DescriptorWriter(app->getDevice())
        .writeBuffer(descriptorSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     cf.instance.buffer, 0, VK_WHOLE_SIZE)
        .flush();

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

    // Indirect-count draw: only the cubes that survived the GPU frustum cull
    // (written into the terrain IR's SDF stream by indirect.comp, count in sdfCount) are drawn.
    // maxDrawCount must bound the SDF *command* buffer (sdfCompactBuf, capacity MAX_SDF_CUBES),
    // NOT the instance buffer capacity (cf.capacity, which holds every AABB and can be larger).
    if (!cmdDrawIndexedIndirectCount) return;
    // maxDrawCount bounds the SDF *command* buffer (capacity MAX_SDF_CUBES), not the
    // instance buffer (cf.capacity, which holds every AABB and may be larger).
    const uint32_t maxSdfDraws = terrainIR_ ? terrainIR_->getMaxSdfCommands() : 8192u;
    const uint32_t maxDrawCount = std::min(cf.capacity, maxSdfDraws);

    // The SDF command + count buffers were written by the terrain IndirectRenderer's
    // indirect.comp dispatch in the (separate) cull command buffer for THIS frame's
    // cull slot. Make those compute-shader writes visible to the indirect draw and to
    // the vertex shader (which reads the per-instance AABBs) before consuming them —
    // without this barrier the draw can race the dispatch and intermittently read
    // pre-dispatch (zero/garbage) state, causing the cubes to flicker.
    VkBufferMemoryBarrier2 sdfBarriers[2] = {};
    auto setupBarrier = [&](VkBufferMemoryBarrier2& b, VkBuffer buf) {
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        b.buffer = buf;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
    };
    setupBarrier(sdfBarriers[0], sdfCompact);
    setupBarrier(sdfBarriers[1], sdfCount);
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 2;
    dep.pBufferMemoryBarriers = sdfBarriers;
    vkCmdPipelineBarrier2(cmd, &dep);

    cmdDrawIndexedIndirectCount(cmd, sdfCompact, 0, sdfCount, 0,
                                maxDrawCount, sizeof(VkDrawIndexedIndirectCommand));
}

void DebugSDFRenderer::cleanup(VulkanApp* app) {
    (void)app;
    vertexBuffer = {};
    indexBuffer = {};
    indexCount = 0;
    activeCubes.clear();
    for (auto& cf : cullFrames) {
        cf.instance = {};
        cf.capacity = 0;
    }
    hasCubes_ = false;
}

namespace {

// SDF face drawability is now decided in LocalScene::requestSDFCubes (sdfCubeDrawable),
// which walks the octree the same way requestModel3D does and only emits lod==1 nodes
// that carry a drawable SDF face. DebugSDFRenderer just renders whatever cubes arrive.

} // namespace

void DebugSDFRenderer::updateCubesForChunk(NodeID nid, const std::vector<CubeSDF>& cubes) {
    std::lock_guard<std::recursive_mutex> lock(cubesMutex);
    if (cubes.empty()) {
        nodeDebugSDFCubes.erase(nid);
    } else {
        nodeDebugSDFCubes[nid] = cubes;
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
