#include "OctreeTopologyCompute.hpp"
#include "../vulkan/core/VulkanApp.hpp"
#include "../vulkan/resources/Buffer.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
// Mirrors the push constant block in OctreeTriangles.comp.
struct TopologyPC {
    uint32_t cellCount;
    int32_t targetLod;
    int32_t fromId;
    int32_t fromBrush;
    glm::vec4 rootMinSize; // xyz = world root min, w = world root size
    float worldSize;
    uint32_t maxVertices;
};

// Output sizing: the shader packs one vertex per slot (xyz position + brush
// bits, normal, hsv) into host-visible storage buffers. The arena is capped;
// when the shader reports dropped triangles the caller falls back to the CPU
// walker, so a cap can never silently lose geometry.
constexpr size_t kVerticesPerCellHint = 24; // 12 edges x 2 triangles x 3 verts is the walker's bound
constexpr size_t kMaxVerticesPerCall = 1u << 19; // 512k vertices -> 24 MB of arenas
}

OctreeTopologyCompute* OctreeTopologyCompute::globalInstance = nullptr;

OctreeTopologyCompute::OctreeTopologyCompute() {
    globalInstance = this;
}

void OctreeTopologyCompute::init(VulkanApp* app) {
    this->app = app;
    if (pipeline != VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    VkDescriptorSetLayoutBinding bindings[12] = {};
    for (int b = 0; b < 12; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 12;
    dli.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: desc layout");
    app->resources.addDescriptorSetLayout(descLayout, "OctreeTopologyCompute: descLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(TopologyPC);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &descLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: pipeline layout");
    app->resources.addPipelineLayout(layout, "OctreeTopologyCompute: layout");

    VkShaderModule sm = app->getOrCreateShaderModule("shaders/renderer/octree/OctreeTriangles.comp.spv");

    VkPipelineShaderStageCreateInfo ssi{};
    ssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssi.module = sm;
    ssi.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = ssi;
    cpi.layout = layout;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &cpi, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: pipeline");
    app->resources.addPipeline(pipeline, "OctreeTopologyCompute: pipeline");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 12;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    dpi.maxSets = 1;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device, &dpi, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: desc pool");
    app->resources.addDescriptorPool(descPool, "OctreeTopologyCompute: descPool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout;
    if (vkAllocateDescriptorSets(device, &ai, &descSet) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: alloc desc set");
    app->resources.addDescriptorSet(descSet, "OctreeTopologyCompute: descSet");
}

void OctreeTopologyCompute::destroy(VulkanApp* app) {
    if (app != nullptr) {
        app->destroyBuffer(bSdf0);
        app->destroyBuffer(bSdf1);
        app->destroyBuffer(bCube);
        app->destroyBuffer(bMeta);
        app->destroyBuffer(bHsv);
        app->destroyBuffer(bChild03);
        app->destroyBuffer(bChild47);
        app->destroyBuffer(bCells);
        app->destroyBuffer(bPos);
        app->destroyBuffer(bNrm);
        app->destroyBuffer(bOutHsv);
        app->destroyBuffer(bCounter);
    }
    nodeCap = cellCap = vertexCap = 0;
    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    descLayout = VK_NULL_HANDLE;
    descPool = VK_NULL_HANDLE;
}

void OctreeTopologyCompute::ensureCapacity(size_t nodeCount, size_t cellCount, size_t vertexCount) {
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (nodeCount > nodeCap) {
        app->destroyBuffer(bSdf0);
        app->destroyBuffer(bSdf1);
        app->destroyBuffer(bCube);
        app->destroyBuffer(bMeta);
        app->destroyBuffer(bHsv);
        app->destroyBuffer(bChild03);
        app->destroyBuffer(bChild47);
        bSdf0 = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bSdf1 = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bCube = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bMeta = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bHsv = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bChild03 = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bChild47 = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        nodeCap = nodeCount;
    }
    if (cellCount > cellCap) {
        app->destroyBuffer(bCells);
        bCells = app->createBuffer(sizeof(uint32_t) * cellCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        cellCap = cellCount;
    }
    if (vertexCount > vertexCap) {
        app->destroyBuffer(bPos);
        app->destroyBuffer(bNrm);
        app->destroyBuffer(bOutHsv);
        bPos = app->createBuffer(sizeof(float) * 4 * vertexCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bNrm = app->createBuffer(sizeof(float) * 4 * vertexCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bOutHsv = app->createBuffer(sizeof(float) * 4 * vertexCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        vertexCap = vertexCount;
    }
    if (bCounter.buffer == VK_NULL_HANDLE) {
        bCounter = app->createBuffer(sizeof(uint32_t) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
    }
}

OctreeTopologyResult OctreeTopologyCompute::compute(const OctreeTopologyInput& input) {
    std::lock_guard<std::mutex> lock(mutex);
    OctreeTopologyResult res;
    const size_t nodeCount = input.nodes.size();
    const size_t cellCount = input.cells.size();
    if (app == nullptr || nodeCount == 0 || cellCount == 0) return res;
    if (pipeline == VK_NULL_HANDLE) init(app);

    const auto t0 = std::chrono::steady_clock::now();

    VkDevice device = app->getDevice();

    const size_t want = cellCount * kVerticesPerCellHint;
    const size_t arena = std::min(want, kMaxVerticesPerCall);
    ensureCapacity(nodeCount, cellCount, arena);
    const size_t vertexArena = std::min(arena, vertexCap);

    // Upload the flattened neighborhood.
    float* s0 = (float*)bSdf0.mappedData;
    float* s1 = (float*)bSdf1.mappedData;
    float* cube = (float*)bCube.mappedData;
    int32_t* meta = (int32_t*)bMeta.mappedData;
    float* hsv = (float*)bHsv.mappedData;
    int32_t* child03 = (int32_t*)bChild03.mappedData;
    int32_t* child47 = (int32_t*)bChild47.mappedData;
    uint32_t* cells = (uint32_t*)bCells.mappedData;
    for (size_t i = 0; i < nodeCount; ++i) {
        const OctreeGpuNode& n = input.nodes[i];
        for (int k = 0; k < 4; ++k) {
            s0[i * 4 + k] = n.sdf[k];
            s1[i * 4 + k] = n.sdf[4 + k];
        }
        cube[i * 4 + 0] = n.cubeMin.x;
        cube[i * 4 + 1] = n.cubeMin.y;
        cube[i * 4 + 2] = n.cubeMin.z;
        cube[i * 4 + 3] = n.cubeSize;
        meta[i * 4 + 0] = n.lod;
        meta[i * 4 + 1] = n.type;
        meta[i * 4 + 2] = n.brushIndex;
        meta[i * 4 + 3] = n.flags;
        hsv[i * 4 + 0] = n.hsv.x;
        hsv[i * 4 + 1] = n.hsv.y;
        hsv[i * 4 + 2] = n.hsv.z;
        hsv[i * 4 + 3] = 0.0f;
        for (int k = 0; k < 4; ++k) child03[i * 4 + k] = n.children[k];
        for (int k = 0; k < 4; ++k) child47[i * 4 + k] = n.children[4 + k];
    }
    for (size_t i = 0; i < cellCount; ++i) cells[i] = input.cells[i];
    uint32_t* counters = (uint32_t*)bCounter.mappedData;
    counters[0] = 0; // emitted vertices
    counters[1] = 0; // dropped triangles (arena overflow)

    VkDescriptorBufferInfo infos[12];
    VkBuffer bufs[12] = {bSdf0.buffer, bSdf1.buffer, bCube.buffer, bMeta.buffer, bHsv.buffer,
                         bChild03.buffer, bChild47.buffer, bCells.buffer,
                         bPos.buffer, bNrm.buffer, bOutHsv.buffer, bCounter.buffer};
    for (int b = 0; b < 12; ++b) infos[b] = {bufs[b], 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[12] = {};
    for (int b = 0; b < 12; ++b) {
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = descSet;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    vkUpdateDescriptorSets(device, 12, writes, 0, nullptr);

    TopologyPC pc{};
    pc.cellCount = (uint32_t)cellCount;
    pc.targetLod = input.targetLod;
    pc.fromId = (int32_t)input.fromId;
    pc.fromBrush = input.fromBrush;
    pc.rootMinSize = glm::vec4(input.rootMin, input.rootSize);
    pc.worldSize = input.rootSize;
    pc.maxVertices = (uint32_t)vertexArena;

    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
        uint32_t groups = (uint32_t)((cellCount + 63) / 64);
        vkCmdDispatch(cmd, groups, 1, 1);
        // Make the shader's triangle writes visible to the host readback that
        // runs after this submission completes. Sync2: compute writes -> host reads.
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
    });

    const uint32_t rawCount = counters[0];
    const uint32_t dropped = counters[1];
    res.truncated = dropped > 0;
    res.vertexCount = std::min<uint32_t>(rawCount, (uint32_t)vertexArena);
    res.vertexCount -= res.vertexCount % 3u;
    if (res.truncated) {
        return res;
    }

    const float* pp = (const float*)bPos.mappedData;
    const float* pn = (const float*)bNrm.mappedData;
    const float* ph = (const float*)bOutHsv.mappedData;

    // Deterministic, CPU-faithful triangle order: the shader appends vertices
    // atomically (completion order) and tags each with its scan-cell index, so
    // a stable bucket sort by that index rebuilds the walker's cell-major
    // order (cells are handed to the shader in walkLadder order, and each
    // cell emits edges/segments/triangles in program order).
    const uint32_t n = res.vertexCount;
    std::vector<OctreeTopologyVertex> app(n);
    std::vector<uint32_t> appCell(n);
    for (uint32_t i = 0; i < n; ++i) {
        OctreeTopologyVertex v;
        v.position = glm::vec3(pp[i * 4 + 0], pp[i * 4 + 1], pp[i * 4 + 2]);
        uint32_t brushBits = 0;
        std::memcpy(&brushBits, &pp[i * 4 + 3], sizeof(uint32_t));
        v.brushIndex = (int32_t)brushBits;
        v.normal = glm::vec3(pn[i * 4 + 0], pn[i * 4 + 1], pn[i * 4 + 2]);
        v.hsv = glm::vec3(ph[i * 4 + 0], ph[i * 4 + 1], ph[i * 4 + 2]);
        app[i] = v;
        appCell[i] = (uint32_t)pn[i * 4 + 3];
    }

    std::vector<uint32_t> count(cellCount + 1, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        if (appCell[i] >= cellCount) {
            // Cannot happen for a completed dispatch (the shader tags each
            // vertex with its own invocation index). Treat it as garbage from a
            // lost device rather than emitting vertices with no owning cell.
            res.truncated = true;
            res.vertexCount = 0;
            res.triangleCount = 0;
            return res;
        }
        ++count[appCell[i]];
    }
    uint32_t running = 0;
    for (size_t c = 0; c < cellCount; ++c) {
        const uint32_t cells_ = count[c];
        count[c] = running;
        running += cells_;
    }
    count[cellCount] = running;

    res.vertices.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t ci = appCell[i];
        res.vertices[count[ci]++] = app[i];
    }

    res.triangleCount = res.vertexCount / 3u;

    // Rolling stats (log every 256 chunks) so the GPU path can be compared
    // against VULKAN_CPU_TOPOLOGY=1 without UI wiring.
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    statMsLast = us / 1000.0;
    ++statCalls;
    statMs += statMsLast;
    statCells += double(cellCount);
    statNodes += double(nodeCount);
    statTriangles += double(res.triangleCount);
    if ((statCalls % 256u) == 0u) {
        const double inv = 1.0 / double(statCalls);
        std::cerr << "[OctreeTopology] chunks=" << statCalls
                  << " avg ms=" << (statMs * inv)
                  << " avg cells=" << (statCells * inv)
                  << " avg nodes=" << (statNodes * inv)
                  << " avg triangles=" << (statTriangles * inv) << "\n";
    }
    return res;
}

OctreeTopologyStats OctreeTopologyCompute::stats() {
    std::lock_guard<std::mutex> lock(mutex);
    OctreeTopologyStats s;
    s.calls = statCalls;
    s.avgMsLast = statMsLast;
    s.avgMs = statCalls ? statMs / double(statCalls) : 0.0;
    s.avgCells = statCalls ? statCells / double(statCalls) : 0.0;
    s.avgNodes = statCalls ? statNodes / double(statCalls) : 0.0;
    s.avgTriangles = statCalls ? statTriangles / double(statCalls) : 0.0;
    return s;
}
