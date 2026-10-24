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
    uint32_t keyCapacity;  // rawKeys[] capacity: hard bound on raw vertices
    // pass 0 = walk (keys only), 1 = dedup + pool vertex write, 2 = index
    // stream write. Every pass writes device memory only; nothing is read back
    // except the integer counters.
    uint32_t pass;
    uint32_t vertexBase;
    uint32_t vertexCapacity;
    uint32_t indexBase;
    uint32_t indexCapacity;
};

// Measured raw-vertex density per scan cell, used to size the key space.
constexpr size_t kVerticesPerCellHint = 24;
// Hard ceiling on the walk's raw vertex count (3 per triangle): the chunk-local
// index stream has one entry per raw vertex, so this is also the index-span
// bound. Measured worst chunk: ~3.8M raw vertices.
constexpr size_t kMaxKeysPerCall = 1u << 23;     // 8M -> 32 MB of keys (device local)
}

OctreeTopologyCompute* OctreeTopologyCompute::globalInstance = nullptr;

OctreeTopologyCompute::OctreeTopologyCompute() {
    globalInstance = this;
}

void OctreeTopologyCompute::init(VulkanApp* vkApp) {
    this->app = vkApp;
    if (pipeline != VK_NULL_HANDLE) return;
    VkDevice device = vkApp->getDevice();

    // 8 input tables + counters/keys/table + the two packed pool spans.
    constexpr uint32_t kBindingCount = 13;
    VkDescriptorSetLayoutBinding bindings[kBindingCount] = {};
    for (uint32_t b = 0; b < kBindingCount; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = kBindingCount;
    dli.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: desc layout");
    vkApp->resources.addDescriptorSetLayout(descLayout, "OctreeTopologyCompute: descLayout");

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
    vkApp->resources.addPipelineLayout(layout, "OctreeTopologyCompute: layout");

    VkShaderModule sm = vkApp->getOrCreateShaderModule("shaders/computation/octree/OctreeTriangles.comp.spv");

    VkPipelineShaderStageCreateInfo ssi{};
    ssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssi.module = sm;
    ssi.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = ssi;
    cpi.layout = layout;
    if (vkCreateComputePipelines(device, vkApp->getPipelineCache(), 1, &cpi, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: pipeline");
    vkApp->resources.addPipeline(pipeline, "OctreeTopologyCompute: pipeline");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = kBindingCount;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    dpi.maxSets = 1;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device, &dpi, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: desc pool");
    vkApp->resources.addDescriptorPool(descPool, "OctreeTopologyCompute: descPool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout;
    if (vkAllocateDescriptorSets(device, &ai, &descSet) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: alloc desc set");
    vkApp->resources.addDescriptorSet(descSet, "OctreeTopologyCompute: descSet");
}

void OctreeTopologyCompute::destroy(VulkanApp* vkApp) {
    if (globalInstance == this) globalInstance = nullptr;
    if (vkApp != nullptr) {
        vkApp->destroyBuffer(bSdf0);
        vkApp->destroyBuffer(bSdf1);
        vkApp->destroyBuffer(bCube);
        vkApp->destroyBuffer(bMeta);
        vkApp->destroyBuffer(bHsv);
        vkApp->destroyBuffer(bChild03);
        vkApp->destroyBuffer(bChild47);
        vkApp->destroyBuffer(bCells);
        vkApp->destroyBuffer(bCounter);
        vkApp->destroyBuffer(bRawKeys);
        vkApp->destroyBuffer(bDedupTable);
    }
    nodeCap = cellCap = keyCap = tableCap = 0;
    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    descLayout = VK_NULL_HANDLE;
    descPool = VK_NULL_HANDLE;
}

void OctreeTopologyCompute::ensureCapacity(size_t nodeCount, size_t cellCount,
                                           size_t keyCount) {
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const size_t tableCount = nodeCount * kVertexKeysPerNode;
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
        // 8 floats per node: vec4(hsv.xyz, textureType), vec4(data.xy, 0, 0).
        bHsv = app->createBuffer(sizeof(float) * 8 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bChild03 = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bChild47 = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        nodeCap = nodeCount;
    }
    if (cellCount > cellCap) {
        app->destroyBuffer(bCells);
        bCells = app->createBuffer(sizeof(uint32_t) * cellCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        cellCap = cellCount;
    }
    if (keyCount > keyCap) {
        // Device-local: the keys are written by the walk and only ever read by
        // the two resident passes.
        app->destroyBuffer(bRawKeys);
        bRawKeys = app->createBuffer(sizeof(uint32_t) * keyCount,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        keyCap = keyCount;
    }
    if (tableCount > tableCap) {
        // Device-local: the dedup table is only ever touched by the compute
        // pass (claimed with atomicCompSwap, then read back by the index pass).
        app->destroyBuffer(bDedupTable);
        bDedupTable = app->createBuffer(sizeof(uint32_t) * tableCount,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        tableCap = tableCount;
    }
    if (bCounter.buffer == VK_NULL_HANDLE) {
        bCounter = app->createBuffer(sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
    }
}

namespace {
// Compute writes -> host reads (integer counters only). Every submission that
// hands counters back to the host ends with this; the vertex/index pools are
// never read by the host.
void hostReadBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}
} // namespace

// Uploads the flattened neighborhood into the host-visible input buffers and
// resets the four counters. Caller must hold `mutex` and have called
// ensureCapacity() for this input.
void OctreeTopologyCompute::uploadInput(const OctreeTopologyInput& input) {
    const size_t nodeCount = input.nodes.size();
    const size_t cellCount = input.cells.size();
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
        hsv[i * 8 + 0] = n.hsv.x;
        hsv[i * 8 + 1] = n.hsv.y;
        hsv[i * 8 + 2] = n.hsv.z;
        hsv[i * 8 + 3] = static_cast<float>(n.textureType);
        hsv[i * 8 + 4] = n.data.x;
        hsv[i * 8 + 5] = n.data.y;
        hsv[i * 8 + 6] = 0.0f;
        hsv[i * 8 + 7] = 0.0f;
        for (int k = 0; k < 4; ++k) child03[i * 4 + k] = n.children[k];
        for (int k = 0; k < 4; ++k) child47[i * 4 + k] = n.children[4 + k];
    }
    for (size_t i = 0; i < cellCount; ++i) cells[i] = input.cells[i];
    uint32_t* counters = (uint32_t*)bCounter.mappedData;
    counters[0] = 0; // emitted vertices (raw, 3 per triangle)
    counters[1] = 0; // reserved (historical arena overflow counter)
    counters[2] = 0; // deduped vertices written into the resident pool span
    counters[3] = 0; // drops: key capacity or pool span overflow
}

// Points the descriptor set at the resident target's pools. Bindings 9/10
// always point at the keys/table; 11/12 fall back to a small input buffer for
// the walk pass (which never touches them, but every declared binding still
// needs a valid buffer).
// A storage descriptor's range must not exceed
// VkPhysicalDeviceLimits::maxStorageBufferRange (128 MB on some drivers, while
// the packed pools are 768/192 MB) and its offset must be aligned. Bind the
// chunk's own span instead of the whole pool: the range stays tiny and valid
// on every device, and the shader's pool indices become span-relative
// (`relVertexBase`/`relIndexBase` absorb the alignment padding, which is zero
// whenever the alignment divides the 64/4-byte element stride).
struct PoolSpanBinding {
    VkDeviceSize vertexOffset = 0;
    VkDeviceSize vertexRange = 0;
    uint32_t relVertexBase = 0;
    VkDeviceSize indexOffset = 0;
    VkDeviceSize indexRange = 0;
    uint32_t relIndexBase = 0;
};

static PoolSpanBinding makePoolSpanBinding(const OctreeTopologyTarget& target,
                                           VkDeviceSize alignment) {
    PoolSpanBinding b;
    const VkDeviceSize align = std::max<VkDeviceSize>(1, alignment);
    const VkDeviceSize vBytes = VkDeviceSize(target.vertexCapacity) * sizeof(Vertex);
    const VkDeviceSize vAbsolute = VkDeviceSize(target.baseVertex) * sizeof(Vertex);
    b.vertexOffset = vAbsolute & ~(align - 1);
    b.relVertexBase = static_cast<uint32_t>((vAbsolute - b.vertexOffset) / sizeof(Vertex));
    // The descriptor range must cover the alignment padding as well as the span:
    // the shader addresses it through relVertexBase/relIndexBase, so the last
    // rel elements live just past `capacity * stride`. Without the padding the
    // tail of the span falls outside the bound range and is never written
    // (stale pool entries -> stretched triangles).
    b.vertexRange = VkDeviceSize(b.relVertexBase) * sizeof(Vertex) + vBytes;

    const VkDeviceSize iBytes = VkDeviceSize(target.indexCapacity) * sizeof(uint32_t);
    const VkDeviceSize iAbsolute = VkDeviceSize(target.firstIndex) * sizeof(uint32_t);
    b.indexOffset = iAbsolute & ~(align - 1);
    b.relIndexBase = static_cast<uint32_t>((iAbsolute - b.indexOffset) / sizeof(uint32_t));
    b.indexRange = VkDeviceSize(b.relIndexBase) * sizeof(uint32_t) + iBytes;
    return b;
}

void OctreeTopologyCompute::writeDescriptors(const OctreeTopologyTarget* target) {
    constexpr uint32_t kBindingCount = 13;
    const bool resident = target != nullptr &&
                          target->vertexBuffer != VK_NULL_HANDLE &&
                          target->indexBuffer != VK_NULL_HANDLE;
    VkBuffer bufs[kBindingCount] = {bSdf0.buffer, bSdf1.buffer, bCube.buffer, bMeta.buffer,
                                    bHsv.buffer, bChild03.buffer, bChild47.buffer, bCells.buffer,
                                    bCounter.buffer, bRawKeys.buffer, bDedupTable.buffer,
                                    resident ? target->vertexBuffer : bCells.buffer,
                                    resident ? target->indexBuffer : bCells.buffer};
    // Every declared binding must name a valid buffer even when the current
    // pass never touches it; fall back to a small always-present input buffer.
    for (uint32_t b = 0; b < kBindingCount; ++b) {
        if (bufs[b] == VK_NULL_HANDLE) bufs[b] = bCells.buffer;
    }
    VkDescriptorBufferInfo infos[kBindingCount];
    VkWriteDescriptorSet writes[kBindingCount] = {};
    for (uint32_t b = 0; b < kBindingCount; ++b) {
        infos[b] = {bufs[b], 0, app->getClampedWholeSizeRange(bufs[b])};
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = descSet;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    // Bindings 11/12 are the renderer's packed pools: bind this chunk's span
    // (offset+range) so the descriptor stays inside every device's
    // maxStorageBufferRange; the shader indexes it span-relative.
    if (resident) {
        const PoolSpanBinding span = makePoolSpanBinding(*target,
                                                         app->getMinStorageBufferOffsetAlignment());
        infos[11] = {bufs[11], span.vertexOffset, span.vertexRange};
        infos[12] = {bufs[12], span.indexOffset, span.indexRange};
    }
    vkUpdateDescriptorSets(app->getDevice(), kBindingCount, writes, 0, nullptr);
}

namespace {
void countStats(uint64_t& calls, double& totalMs, double& lastMs, double& cells,
                double& nodes, double& triangles, double ms, size_t cellCount,
                size_t nodeCount, double triangleCount) {
    lastMs = ms;
    ++calls;
    totalMs += ms;
    cells += double(cellCount);
    nodes += double(nodeCount);
    triangles += triangleCount;
    if ((calls % 256u) == 0u) {
        const double inv = 1.0 / double(calls);
        std::cerr << "[OctreeTopology] chunks=" << calls
                  << " avg ms=" << (totalMs * inv)
                  << " avg cells=" << (cells * inv)
                  << " avg nodes=" << (nodes * inv)
                  << " avg triangles=" << (triangles * inv) << "\n";
    }
}
} // namespace

// Resident path. Two submissions with a host round trip in between:
//   1) walk — writes the dedup keys and the raw count. The index stream has
//      exactly one entry per raw vertex, so its span can only be sized once
//      that count is known; `provider` reserves the span.
//   2) dedup + index stream — writes the renderer's packed pools directly
//      (device-local, never read back). Only integer counters return to the
//      host; no Cartesian vertex ever crosses back from the GPU.
// No vertex data is ever uploaded: the app only sends the node SDF tables.
OctreeTopologyResidentResult OctreeTopologyCompute::computeResident(
    const OctreeTopologyInput& input, const OctreeTopologyTargetProvider& provider) {
    std::lock_guard<std::mutex> lock(mutex);
    OctreeTopologyResidentResult res;
    const size_t nodeCount = input.nodes.size();
    const size_t cellCount = input.cells.size();
    if (app == nullptr || nodeCount == 0 || cellCount == 0 || !provider) return res;
    if (pipeline == VK_NULL_HANDLE) init(app);

    const auto t0 = std::chrono::steady_clock::now();

    // The walk's key capacity is an estimate (kVerticesPerCellHint per scan
    // cell). Real chunks can overshoot it slightly (dense multi-segment
    // edges), so the walk is retried with a doubled key space instead of
    // dropping the chunk; kMaxKeysPerCall caps the retries.
    size_t keyWant = std::min(cellCount * kVerticesPerCellHint, kMaxKeysPerCall);
    size_t keys = 0;
    size_t maxUnique = 0;
    uint32_t rawCount = 0;
    bool walkOk = false;

    TopologyPC pc{};
    pc.cellCount = (uint32_t)cellCount;
    pc.targetLod = input.targetLod;
    pc.fromId = (int32_t)input.fromId;
    pc.fromBrush = input.fromBrush;
    pc.rootMinSize = glm::vec4(input.rootMin, input.rootSize);
    pc.worldSize = input.rootSize;
    pc.pass = 0u;

    for (int attempt = 0; attempt < 6 && !walkOk; ++attempt) {
        ensureCapacity(nodeCount, cellCount, keyWant);
        keys = std::min(keyWant, keyCap);
        maxUnique = std::min(nodeCount * kVertexKeysPerNode, keys);
        uploadInput(input); // also resets the four counters
        writeDescriptors(nullptr);
        pc.keyCapacity = (uint32_t)keys;

        // Phase 1: the walk. Only the keys are written; the counters tell us
        // how big the chunk's index stream is.
        const uint32_t cellGroups = (uint32_t)((cellCount + 63) / 64);
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
            vkCmdDispatch(cmd, cellGroups, 1, 1);
            hostReadBarrier(cmd);
        });

        const uint32_t* counters = (const uint32_t*)bCounter.mappedData;
        const uint32_t emitted = counters[0];
        const uint32_t walkDropped = counters[3];
        if (walkDropped == 0 && emitted <= keys) {
            rawCount = std::min<uint32_t>(emitted, (uint32_t)keys);
            rawCount -= rawCount % 3u;
            walkOk = true;
            break;
        }
        if (keys >= kMaxKeysPerCall) break;
        const size_t nextWant = std::min<size_t>(keys * 2, kMaxKeysPerCall);
        if (nextWant <= keys) break;
        static std::atomic<int> retryLogged{0};
        if (retryLogged.fetch_add(1) < 8)
            std::cerr << "[OctreeTopology] walk key capacity too small (emitted=" << emitted
                      << " cap=" << keys << " cells=" << cellCount
                      << ") — retrying with " << nextWant << "\n";
        keyWant = nextWant;
    }

    if (!walkOk) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 8)
            std::cerr << "[OctreeTopology] chunk walk overflowed the key capacity (cap="
                      << keys << " cells=" << cellCount << "): skipped\n";
        return res; // ok == false
    }

    res.indexCount = rawCount;
    if (rawCount < 3u) {
        res.ok = true; // empty chunk: nothing to draw, not a failure
        return res;
    }

    // Reserve the packed span with the walk's exact sizes: the deduped vertex
    // count is bounded by the key space and the index stream is exactly
    // rawCount entries long.
    OctreeTopologyDemand demand;
    demand.rawVertexCount = rawCount;
    demand.maxUniqueVertices = (uint32_t)maxUnique;
    demand.suggestedVertexCapacity = std::min<uint32_t>((uint32_t)maxUnique, rawCount);
    OctreeTopologyTarget target = provider(demand);
    res.slotIndex = target.slotIndex;
    const bool targetValid = target.vertexBuffer != VK_NULL_HANDLE &&
                             target.indexBuffer != VK_NULL_HANDLE;
    if (!targetValid || target.vertexCapacity < demand.suggestedVertexCapacity ||
        target.indexCapacity < rawCount) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 8)
            std::cerr << "[OctreeTopology] span too small for chunk (raw=" << rawCount
                      << " needVerts=" << demand.suggestedVertexCapacity
                      << " gotVerts=" << target.vertexCapacity
                      << " gotIdx=" << target.indexCapacity
                      << (targetValid ? "" : " no-target") << "): skipped\n";
        return res; // ok == false
    }

    // Phase 2: dedup into the pool span + chunk-local index stream.
    writeDescriptors(&target);
    const PoolSpanBinding poolSpan = makePoolSpanBinding(
        target, app->getMinStorageBufferOffsetAlignment());
    pc.vertexBase = poolSpan.relVertexBase; // span-relative inside binding 11
    pc.vertexCapacity = target.vertexCapacity;
    pc.indexBase = poolSpan.relIndexBase; // span-relative inside binding 12
    pc.indexCapacity = target.indexCapacity;
    const uint32_t vertexGroups = (uint32_t)((rawCount + 63) / 64);

    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);

        auto computeBarrier = [&](VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
            VkMemoryBarrier2 mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = srcAccess;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = dstAccess;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &dep);
        };

        // The dedup table starts free (0) for every dispatch. Device-local
        // fill: a transfer op, so it needs its own barrier before pass 1 claims
        // entries with atomicCompSwap.
        vkCmdFillBuffer(cmd, bDedupTable.buffer, 0,
                        VkDeviceSize(nodeCount * kVertexKeysPerNode * sizeof(uint32_t)), 0);
        VkMemoryBarrier2 fill{};
        fill.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fill.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        fill.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        fill.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fill.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo fillDep{};
        fillDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        fillDep.memoryBarrierCount = 1;
        fillDep.pMemoryBarriers = &fill;
        vkCmdPipelineBarrier2(cmd, &fillDep);

        // Pass 1 (dedup into the pooled span) reads the keys the walk wrote;
        // pass 2 (index stream) reads the table pass 1 filled.
        computeBarrier(VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        pc.pass = 1u;
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
        vkCmdDispatch(cmd, vertexGroups, 1, 1);

        computeBarrier(VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        pc.pass = 2u;
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
        vkCmdDispatch(cmd, vertexGroups, 1, 1);

        // Pool writes -> the cull's shader reads, the indirect draws and the
        // ray-tracing build inputs that read the same buffers.
        VkBufferMemoryBarrier2 pool[2] = {};
        for (int b = 0; b < 2; ++b) {
            pool[b].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            pool[b].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            pool[b].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            pool[b].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT
                                 | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT
                                 | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                 | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            pool[b].dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT
                                  | VK_ACCESS_2_INDEX_READ_BIT
                                  | VK_ACCESS_2_SHADER_READ_BIT
                                  | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            pool[b].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pool[b].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pool[b].offset = 0;
            pool[b].size = VK_WHOLE_SIZE;
        }
        pool[0].buffer = target.vertexBuffer;
        pool[1].buffer = target.indexBuffer;
        VkDependencyInfo poolDep{};
        poolDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        poolDep.bufferMemoryBarrierCount = 2;
        poolDep.pBufferMemoryBarriers = pool;
        vkCmdPipelineBarrier2(cmd, &poolDep);

        hostReadBarrier(cmd);
    });

    const uint32_t* phase2Counters = (const uint32_t*)bCounter.mappedData;
    const uint32_t unique = phase2Counters[2];
    const uint32_t dropPhase2 = phase2Counters[3]; // includes the (already zero) phase-1 drops
    res.poolVertexCount = std::min(unique, target.vertexCapacity);
    const bool spanFits = unique <= target.vertexCapacity;
    res.ok = dropPhase2 == 0 && spanFits;
    if (!res.ok) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 8)
            std::cerr << "[OctreeTopology] resident write dropped output for chunk (raw="
                      << rawCount << " unique=" << unique
                      << " capacity=" << target.vertexCapacity
                      << " drops=" << dropPhase2 << "): skipped\n";
        return res;
    }

    countStats(statCalls, statMs, statMsLast, statCells, statNodes, statTriangles,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               cellCount, nodeCount, double(rawCount) / 3.0);
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
