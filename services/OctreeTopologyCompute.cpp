#include "OctreeTopologyCompute.hpp"
#include "../vulkan/core/VulkanApp.hpp"
#include "../vulkan/resources/Buffer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
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
    // pass 0 = walk (emits raw node keys), 1 = finalize (index stream + one
    // vertex per host-assigned surface-node slot). Every pass writes device
    // memory only; nothing is read back except the integer counters.
    uint32_t pass;
    uint32_t vertexBase;
    uint32_t vertexCapacity;
    uint32_t indexBase;
    uint32_t indexCapacity;
    uint32_t indexCount;   // finalize: raw vertices in the index stream
    uint32_t nodeCount;    // finalize: node-table entries (vertex tail bound)
    uint32_t diag;         // 1 = accumulate findCellAt/reuse counters
};

// Raw-vertex density per scan cell used to size the walk's key space. The
// emitted keys are unchanged by this hint; only the allocation is sized.
// Dense multi-segment chunks overshoot the old 24/cell estimate by 12-24
// vertices, and every overshoot cost a full re-upload + re-walk (L9), so the
// initial capacity carries headroom instead.
constexpr size_t kVerticesPerCellHint = 25;
constexpr size_t kKeyCapacityHeadroom = 64;
// Hard ceiling on the walk's raw vertex count (3 per triangle): the chunk-local
// index stream has one entry per raw vertex, so this is also the index-span
// bound. Measured worst chunk: ~3.8M raw vertices.
constexpr size_t kMaxKeysPerCall = 1u << 23;     // 8M -> 32 MB of keys (device local)
}

OctreeTopologyCompute* OctreeTopologyCompute::globalInstance = nullptr;

OctreeTopologyCompute::OctreeTopologyCompute() {
    globalInstance = this;
    // Cached once: every timing site below is guarded by this flag, so the
    // normal path does no clock reads, no string work and no output.
    setupDiag = std::getenv("VULKAN_SETUP_DIAG") != nullptr;
}

OctreeTopologyCompute::~OctreeTopologyCompute() {
    // destroy() is not called on the normal teardown path (the buffers are
    // owned by the app's resource manager), so the destructor guarantees the
    // final profiler table still reaches the log. Silent when diag is off.
    printProfileSummary();
    if (globalInstance == this) globalInstance = nullptr;
}

void OctreeTopologyCompute::init(VulkanApp* vkApp) {
    this->app = vkApp;
    if (pipeline != VK_NULL_HANDLE) return;
    VkDevice device = vkApp->getDevice();

    // 8 input tables + counters/raw keys + the two packed pool spans.
    constexpr uint32_t kBindingCount = 12;
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

void OctreeTopologyCompute::ensureCapacity(size_t nodeCount, size_t cellCount,
                                           size_t keyCount) {
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // Any recreated input buffer invalidates the descriptor set: bump the
    // generation so the next writeDescriptors() rewrites bindings 0-9. When
    // nothing is recreated (the common case after warm-up) the generation is
    // unchanged and those bindings keep their already-correct descriptors.
    bool recreated = false;
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
        recreated = true;
    }
    if (cellCount > cellCap) {
        app->destroyBuffer(bCells);
        bCells = app->createBuffer(sizeof(uint32_t) * cellCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        cellCap = cellCount;
        recreated = true;
    }
    if (keyCount > keyCap) {
        // Device-local: the keys are written by the walk and read by the
        // finalize index pass.
        app->destroyBuffer(bRawKeys);
        bRawKeys = app->createBuffer(sizeof(uint32_t) * keyCount,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        keyCap = keyCount;
        recreated = true;
    }
    if (bCounter.buffer == VK_NULL_HANDLE) {
        // 0 raw vertices, 1 fixed-array overflows, 2 unused (deduped count),
        // 3 dropped pool writes, 4 findCellAt descents, 5 quadrant reuse hits.
        bCounter = app->createBuffer(sizeof(uint32_t) * 6, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        recreated = true;
    }
    if (recreated) ++bufferGeneration;
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
    counters[1] = 0; // fixed-capacity recursion buffer overflow
    counters[2] = 0; // unused (historical deduped-vertex count)
    counters[3] = 0; // drops: key capacity or pool span overflow
    counters[4] = 0; // findCellAt descents (VULKAN_SETUP_DIAG only)
    counters[5] = 0; // quadrant lookups avoided by AABB reuse (diag only)
}

// Points the descriptor set at the resident target's pools. Binding 9 always
// points at the raw keys; 10/11 fall back to a small input buffer for
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
    constexpr uint32_t kBindingCount = 12;
    constexpr uint32_t kInputBindingCount = 10; // 0..9: node tables, cells, counters, raw keys
    const bool resident = target != nullptr &&
                          target->vertexBuffer != VK_NULL_HANDLE &&
                          target->indexBuffer != VK_NULL_HANDLE;
    VkDescriptorBufferInfo infos[kBindingCount]{};
    VkWriteDescriptorSet writes[kBindingCount] = {};

    // Bindings 0-9 name buffers that only change when ensureCapacity()
    // recreates one (bufferGeneration). Write them once per capacity change;
    // on the steady-state path they keep the descriptors from the first call.
    const bool rewriteInputs = bufferGeneration != lastWrittenGeneration;
    if (rewriteInputs) {
        VkBuffer bufs[kInputBindingCount] = {bSdf0.buffer, bSdf1.buffer, bCube.buffer, bMeta.buffer,
                                             bHsv.buffer, bChild03.buffer, bChild47.buffer, bCells.buffer,
                                             bCounter.buffer, bRawKeys.buffer};
        // Every declared binding must name a valid buffer even when the current
        // pass never touches it; fall back to a small always-present input buffer.
        for (uint32_t b = 0; b < kInputBindingCount; ++b) {
            if (bufs[b] == VK_NULL_HANDLE) bufs[b] = bCells.buffer;
        }
        for (uint32_t b = 0; b < kInputBindingCount; ++b) {
            infos[b] = {bufs[b], 0, app->getClampedWholeSizeRange(bufs[b])};
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = descSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        lastWrittenGeneration = bufferGeneration;
    }

    // Bindings 10/11 are the renderer's packed pools and change per chunk.
    // The walk pass passes nullptr and only needs a valid binding: use the
    // cells buffer, exactly as the old fallback did.
    VkBuffer poolVertex = resident ? target->vertexBuffer : bCells.buffer;
    VkBuffer poolIndex = resident ? target->indexBuffer : bCells.buffer;
    for (uint32_t b = kInputBindingCount; b < kBindingCount; ++b) {
        const VkBuffer buf = (b == 10) ? poolVertex : poolIndex;
        infos[b] = {buf, 0, app->getClampedWholeSizeRange(buf)};
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = descSet;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    // Bind this chunk's span (offset+range) so the descriptor stays inside
    // every device's maxStorageBufferRange; the shader indexes it span-relative.
    if (resident) {
        const PoolSpanBinding span = makePoolSpanBinding(*target,
                                                         app->getMinStorageBufferOffsetAlignment());
        infos[10] = {poolVertex, span.vertexOffset, span.vertexRange};
        infos[11] = {poolIndex, span.indexOffset, span.indexRange};
    }

    const uint32_t firstWrite = rewriteInputs ? 0u : kInputBindingCount;
    const uint32_t writeCount = kBindingCount - firstWrite;
    vkUpdateDescriptorSets(app->getDevice(), writeCount, writes + firstWrite, 0, nullptr);
    statDescriptorWrites += writeCount;
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

// VULKAN_SETUP_DIAG profiler. Called with `mutex` held (from computeResident)
// so the accumulators need no further synchronization. Prints one block every
// 256 chunks with cumulative averages and the last-window rates; the normal
// (diagnostics-off) log is untouched.
void OctreeTopologyCompute::recordProfile(const ProfileSample& s) {
    if (!setupDiag) return;
    auto add = [](ProfileSample& a, const ProfileSample& b) {
        a.totalMs += b.totalMs;
        a.uploadMs += b.uploadMs;
        a.descriptorMs += b.descriptorMs;
        a.walkMs += b.walkMs;
        a.providerMs += b.providerMs;
        a.phase2Ms += b.phase2Ms;
        a.nodes += b.nodes;
        a.cells += b.cells;
        a.rawVertices += b.rawVertices;
        a.activeVertices += b.activeVertices;
        a.uploadedBytes += b.uploadedBytes;
        a.descriptorWrites += b.descriptorWrites;
        a.retries += b.retries;
        a.keyCapacity += b.keyCapacity;
        a.findCellCalls += b.findCellCalls;
        a.reuseHits += b.reuseHits;
        ++a.chunks;
    };
    add(profTotal, s);
    add(profWindow, s);
    if ((profTotal.chunks % 256u) != 0u) return;

    const double inv = 1.0 / double(profTotal.chunks);
    const double win = 1.0 / double(profWindow.chunks);
    std::cerr << "[OctreeTopology][setup-diag] chunks=" << profTotal.chunks
              << " avg ms: total=" << (profTotal.totalMs * inv)
              << " upload=" << (profTotal.uploadMs * inv)
              << " desc=" << (profTotal.descriptorMs * inv)
              << " walk=" << (profTotal.walkMs * inv)
              << " provider=" << (profTotal.providerMs * inv)
              << " phase2=" << (profTotal.phase2Ms * inv)
              << " | last " << profWindow.chunks << " ms: total=" << (profWindow.totalMs * win)
              << " upload=" << (profWindow.uploadMs * win)
              << " desc=" << (profWindow.descriptorMs * win)
              << " walk=" << (profWindow.walkMs * win)
              << " provider=" << (profWindow.providerMs * win)
              << " phase2=" << (profWindow.phase2Ms * win) << "\n";
    std::cerr << "[OctreeTopology][setup-diag] chunks=" << profTotal.chunks
              << " avg/chunk: nodes=" << (double(profTotal.nodes) * inv)
              << " cells=" << (double(profTotal.cells) * inv)
              << " raw=" << (double(profTotal.rawVertices) * inv)
              << " activeVerts=" << (double(profTotal.activeVertices) * inv)
              << " bytes=" << (double(profTotal.uploadedBytes) * inv)
              << " descWrites=" << (double(profTotal.descriptorWrites) * inv)
              << " keyCap=" << (double(profTotal.keyCapacity) * inv)
              << " findCell=" << (double(profTotal.findCellCalls) * inv)
              << " reuse=" << (double(profTotal.reuseHits) * inv)
              << " | last " << profWindow.chunks << ": raw=" << (double(profWindow.rawVertices) * win)
              << " bytes=" << (double(profWindow.uploadedBytes) * win)
              << " descWrites=" << (double(profWindow.descriptorWrites) * win)
              << " keyCap=" << (double(profWindow.keyCapacity) * win)
              << " findCell=" << (double(profWindow.findCellCalls) * win)
              << " reuse=" << (double(profWindow.reuseHits) * win)
              << " retries=" << profWindow.retries << "\n";
    profWindow = ProfileSample{};
}

// Final table at teardown, from destroy() or the destructor (whichever comes
// first). Silent unless VULKAN_SETUP_DIAG was set and a chunk completed.
void OctreeTopologyCompute::printProfileSummary() {
    if (profileSummaryPrinted || !setupDiag || profTotal.chunks == 0) return;
    profileSummaryPrinted = true;
    const double inv = 1.0 / double(profTotal.chunks);
    std::cerr << "[OctreeTopology][setup-diag] final chunks=" << profTotal.chunks
              << " avg ms: total=" << (profTotal.totalMs * inv)
              << " upload=" << (profTotal.uploadMs * inv)
              << " desc=" << (profTotal.descriptorMs * inv)
              << " walk=" << (profTotal.walkMs * inv)
              << " provider=" << (profTotal.providerMs * inv)
              << " phase2=" << (profTotal.phase2Ms * inv)
              << " | totals: nodes=" << profTotal.nodes
              << " cells=" << profTotal.cells
              << " raw=" << profTotal.rawVertices
              << " activeVerts=" << profTotal.activeVertices
              << " uploadedMB=" << (double(profTotal.uploadedBytes) / (1024.0 * 1024.0))
              << " descWrites=" << profTotal.descriptorWrites
              << " retries=" << profTotal.retries
              << " keyCap=" << profTotal.keyCapacity
              << " findCell=" << profTotal.findCellCalls
              << " reuse=" << profTotal.reuseHits << "\n";
}

// Resident path. Two submissions with a host round trip in between:
//   1) walk — writes one raw node key per emitted triangle corner and the raw
//      count. The index stream has exactly one entry per raw vertex, so its
//      span can only be sized once that count is known; `provider` reserves
//      the span (and the host-assigned surface-node vertex slots).
//   2) finalize — one dispatch rewrites the index stream from the raw keys and
//      writes one vertex per host-assigned surface-node slot directly into the
//      renderer's packed pools (device-local, never read back). Only integer
//      counters return to the host; no Cartesian vertex ever crosses back from
//      the GPU.
// No vertex data is ever uploaded: the app only sends the node SDF tables.
OctreeTopologyResidentResult OctreeTopologyCompute::computeResident(
    const OctreeTopologyInput& input, const OctreeTopologyTargetProvider& provider) {
    // Profiler entry stamp sits before the lock so totalMs includes the wait
    // for a busy service (the convoy the report measures); the legacy statMs
    // below keeps its old post-lock meaning.
    const auto tEnter = setupDiag ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    std::lock_guard<std::mutex> lock(mutex);
    // Read the descriptor-write counter under the same lock that guards it.
    const uint64_t descWritesAtStart = statDescriptorWrites;
    OctreeTopologyResidentResult res;
    const size_t nodeCount = input.nodes.size();
    const size_t cellCount = input.cells.size();
    if (app == nullptr || nodeCount == 0) return res;
    if (cellCount == 0) {
        // Valid empty chunk: no scan cell at this rung straddles the surface.
        res.ok = true;
        return res;
    }
    if (!provider) return res;
    if (pipeline == VK_NULL_HANDLE) init(app);

    const auto t0 = std::chrono::steady_clock::now();
    const auto now = [] { return std::chrono::steady_clock::now(); };
    auto msSince = [](const std::chrono::steady_clock::time_point& a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    // Runs `fn` and adds its wall time to `acc` only when the profiler is on:
    // the normal path pays one predictable branch per phase and no clock read.
    auto timed = [&](double& acc, auto&& fn) {
        if (!setupDiag) {
            fn();
            return;
        }
        const auto t = now();
        fn();
        acc += msSince(t);
    };

    // Capacity headroom (L9): 25 keys per scan cell plus a small constant,
    // capped by kMaxKeysPerCall. The hint was calibrated per all targetLod
    // cells, so use the pre-compaction count when the caller provides it;
    // otherwise the compacted list is a conservative fallback. This only sizes
    // the key space -- the emitted keys are unchanged. The retry loop below
    // stays as the safety net for genuinely denser chunks.
    const size_t hintCells = input.scanCellCount != 0 ? input.scanCellCount : cellCount;
    size_t keyWant = std::min(hintCells * kVerticesPerCellHint + kKeyCapacityHeadroom,
                              kMaxKeysPerCall);
    size_t keys = 0;
    const size_t activeVerts = input.activeVertexCount;
    uint32_t rawCount = 0;
    bool walkOk = false;
    // Per-phase setup timings (VULKAN_SETUP_DIAG only; zero otherwise).
    double uploadMs = 0.0, descriptorMs = 0.0, walkMs = 0.0, providerMs = 0.0, phase2Ms = 0.0;
    uint64_t attempts = 0;

    TopologyPC pc{};
    pc.cellCount = (uint32_t)cellCount;
    pc.targetLod = input.targetLod;
    pc.fromId = (int32_t)input.fromId;
    pc.fromBrush = input.fromBrush;
    pc.rootMinSize = glm::vec4(input.rootMin, input.rootSize);
    pc.worldSize = input.rootSize;
    pc.pass = 0u;
    pc.indexCount = 0u;
    pc.nodeCount = (uint32_t)nodeCount;
    pc.diag = setupDiag ? 1u : 0u;

    for (int attempt = 0; attempt < 6 && !walkOk; ++attempt) {
        ++attempts;
        ensureCapacity(nodeCount, cellCount, keyWant);
        keys = std::min(keyWant, keyCap);
        timed(uploadMs, [&] { uploadInput(input); }); // also resets the counters
        timed(descriptorMs, [&] { writeDescriptors(nullptr); });
        pc.keyCapacity = (uint32_t)keys;

        // Phase 1: the walk. Only the keys are written; the counters tell us
        // how big the chunk's index stream is.
        const uint32_t cellGroups = (uint32_t)((cellCount + 63) / 64);
        timed(walkMs, [&] {
            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
                vkCmdDispatch(cmd, cellGroups, 1, 1);
                hostReadBarrier(cmd);
            });
        });

        const uint32_t* counters = (const uint32_t*)bCounter.mappedData;
        const uint32_t emitted = counters[0];
        const uint32_t walkOverflow = counters[1];
        const uint32_t walkDropped = counters[3];
        // A fixed-capacity recursion buffer overflow means the GPU walk could
        // differ from the CPU reference; more key space cannot fix it, so fail
        // the chunk instead of retrying or drawing truncated geometry.
        if (walkOverflow != 0) break;
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
        if (logged.fetch_add(1) < 8) {
            const uint32_t* finalCounters = (const uint32_t*)bCounter.mappedData;
            std::cerr << "[OctreeTopology] chunk walk failed (cap=" << keys
                      << " cells=" << cellCount << " overflow=" << finalCounters[1]
                      << "): skipped\n";
        }
        return res; // ok == false
    }

    res.indexCount = rawCount;
    if (rawCount < 3u) {
        res.ok = true; // empty chunk: nothing to draw, not a failure
        return res;
    }

    // Reserve the packed span with the walk's exact index size and the host's
    // dense surface-node slot count. The finalize pass writes every slot, so
    // the vertex span must cover all of them (indices reference a subset).
    OctreeTopologyDemand demand;
    demand.rawVertexCount = rawCount;
    demand.maxUniqueVertices = (uint32_t)activeVerts;
    demand.suggestedVertexCapacity = (uint32_t)activeVerts;
    OctreeTopologyTarget target;
    timed(providerMs, [&] { target = provider(demand); });
    res.slotIndex = target.slotIndex;
    res.baseVertex = target.baseVertex;
    res.firstIndex = target.firstIndex;
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

    // Finalize: one dispatch writes the chunk-local index stream (one entry per
    // raw vertex, pointing at its emitting node's host-assigned slot) and the
    // vertex payload for every surface-node slot. The two ranges are disjoint
    // and independent, so one dispatch covers both; the index range comes first
    // because it is the larger one.
    timed(descriptorMs, [&] { writeDescriptors(&target); });
    const PoolSpanBinding poolSpan = makePoolSpanBinding(
        target, app->getMinStorageBufferOffsetAlignment());
    pc.vertexBase = poolSpan.relVertexBase; // span-relative inside binding 10
    pc.vertexCapacity = target.vertexCapacity;
    pc.indexBase = poolSpan.relIndexBase; // span-relative inside binding 11
    pc.indexCapacity = target.indexCapacity;
    pc.indexCount = rawCount;
    pc.nodeCount = (uint32_t)nodeCount;
    pc.pass = 1u;
    const uint64_t finalizeWork = uint64_t(rawCount) + uint64_t(nodeCount);
    const uint32_t finalizeGroups = (uint32_t)((finalizeWork + 63) / 64);

    // Finalize timing covers the index/vertex dispatch and its fence wait --
    // the submission the report calls "phase 2".
    const auto tPhase2 = setupDiag ? now() : std::chrono::steady_clock::time_point{};
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
        vkCmdDispatch(cmd, finalizeGroups, 1, 1);

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
    if (setupDiag) phase2Ms += msSince(tPhase2);

    const uint32_t* finalCounters = (const uint32_t*)bCounter.mappedData;
    const uint32_t dropFinal = finalCounters[3];
    res.poolVertexCount = (uint32_t)activeVerts;
    const bool spanFits = activeVerts <= target.vertexCapacity;
    res.ok = dropFinal == 0 && spanFits;
    if (!res.ok) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 8)
            std::cerr << "[OctreeTopology] finalize dropped output for chunk (raw="
                      << rawCount << " activeVerts=" << activeVerts
                      << " capacity=" << target.vertexCapacity
                      << " drops=" << dropFinal << "): skipped\n";
        return res;
    }

    countStats(statCalls, statMs, statMsLast, statCells, statNodes, statTriangles,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               cellCount, nodeCount, double(rawCount) / 3.0);

    if (setupDiag) {
        ProfileSample s;
        s.totalMs = msSince(tEnter); // includes the wait for `mutex`
        s.uploadMs = uploadMs;
        s.descriptorMs = descriptorMs;
        s.walkMs = walkMs;
        s.providerMs = providerMs;
        s.phase2Ms = phase2Ms;
        s.nodes = nodeCount;
        s.cells = cellCount;
        s.rawVertices = rawCount;
        s.activeVertices = activeVerts;
        s.uploadedBytes = uint64_t(nodeCount) * 128u; // 7 input tables, 128 B/node
        s.descriptorWrites = statDescriptorWrites - descWritesAtStart;
        s.retries = attempts > 0 ? attempts - 1 : 0;
        s.keyCapacity = keys;
        s.findCellCalls = finalCounters[4];
        s.reuseHits = finalCounters[5];
        recordProfile(s);
    }
    return res;
}

