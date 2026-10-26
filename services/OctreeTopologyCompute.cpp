#include "OctreeTopologyCompute.hpp"
#include "OctreeTopologyCapacity.hpp"
#include "../vulkan/core/VulkanApp.hpp"
#include "../vulkan/resources/Buffer.hpp"
#include <algorithm>
#include <atomic>
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
    uint32_t vertexBase;     // span-relative base inside the bound vertex window
    uint32_t vertexCapacity; // pool elements reserved for this chunk
    uint32_t indexBase;      // span-relative base inside the bound index window
    uint32_t indexCapacity;  // index pool elements reserved for this chunk
    uint32_t nodeCount;      // node-table entries (nodeWritten[] bound)
    uint32_t diag;           // 1 = accumulate findCellAt/reuse counters
};

// A span overflow is resolved by releasing the chunk's reservation and asking
// the provider for the exact capacity the shader reported. Two retries are
// enough (index growth, then a defensive vertex growth); more would only spin
// on a provider that cannot fit the required span.
constexpr uint32_t kMaxSpanRetries = 3;
} // namespace

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

    // 7 node tables + cells + counters + node-written flags + the two pools.
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

void OctreeTopologyCompute::ensureCapacity(size_t nodeCount, size_t cellCount) {
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
        app->destroyBuffer(bNodeWritten);
        bSdf0 = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bSdf1 = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bCube = app->createBuffer(sizeof(float) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bMeta = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        // 8 floats per node: vec4(hsv.xyz, textureType), vec4(data.xy, 0, 0).
        bHsv = app->createBuffer(sizeof(float) * 8 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bChild03 = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bChild47 = app->createBuffer(sizeof(int32_t) * 4 * nodeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        // Device-local: written by vkCmdFillBuffer immediately before the
        // geometry dispatch (atomicExchange elects one vertex writer per node).
        bNodeWritten = app->createBuffer(sizeof(uint32_t) * nodeCount,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        nodeCap = nodeCount;
        recreated = true;
    }
    if (cellCount > cellCap) {
        app->destroyBuffer(bCells);
        bCells = app->createBuffer(sizeof(uint32_t) * cellCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        cellCap = cellCount;
        recreated = true;
    }
    if (bCounter.buffer == VK_NULL_HANDLE) {
        // 0 indices appended, 1 fixed-array overflows, 2 vertex extent,
        // 3 dropped triangles, 4 findCellAt descents, 5 quadrant reuse hits,
        // 6 index-span overflows, 7 vertex-span overflows, 8 invalid slots,
        // 9 distinct vertex payloads written. (12 uints keeps the buffer a
        // multiple of the 4-byte fill alignment and leaves headroom.)
        bCounter = app->createBuffer(sizeof(uint32_t) * 12, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
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
// resets the counters. Caller must hold `mutex` and have called
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
    for (int i = 0; i < 12; ++i) counters[i] = 0;
}

// Points the descriptor set at the resident target's pools and the per-chunk
// node-written flags.
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

void OctreeTopologyCompute::writeDescriptors(const OctreeTopologyTarget& target) {
    constexpr uint32_t kBindingCount = 12;
    constexpr uint32_t kInputBindingCount = 10; // 0..9: node tables, cells, counters, node flags
    VkDescriptorBufferInfo infos[kBindingCount]{};
    VkWriteDescriptorSet writes[kBindingCount] = {};

    // Bindings 0-9 name buffers that only change when ensureCapacity()
    // recreates one (bufferGeneration). Write them once per capacity change;
    // on the steady-state path they keep the descriptors from the first call.
    const bool rewriteInputs = bufferGeneration != lastWrittenGeneration;
    if (rewriteInputs) {
        VkBuffer bufs[kInputBindingCount] = {bSdf0.buffer, bSdf1.buffer, bCube.buffer, bMeta.buffer,
                                             bHsv.buffer, bChild03.buffer, bChild47.buffer, bCells.buffer,
                                             bCounter.buffer, bNodeWritten.buffer};
        // Every declared binding must name a valid buffer; fall back to the
        // small always-present cells buffer if a table is unexpectedly empty.
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
    for (uint32_t b = kInputBindingCount; b < kBindingCount; ++b) {
        const VkBuffer buf = (b == 10) ? target.vertexBuffer : target.indexBuffer;
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
    const PoolSpanBinding span = makePoolSpanBinding(
        target, app->getMinStorageBufferOffsetAlignment());
    infos[10] = {target.vertexBuffer, span.vertexOffset, span.vertexRange};
    infos[11] = {target.indexBuffer, span.indexOffset, span.indexRange};

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
        a.providerMs += b.providerMs;
        a.dispatchMs += b.dispatchMs;
        a.nodes += b.nodes;
        a.cells += b.cells;
        a.indices += b.indices;
        a.vertices += b.vertices;
        a.crossEdges += b.crossEdges;
        a.vertexCapacity += b.vertexCapacity;
        a.indexCapacity += b.indexCapacity;
        a.uploadedBytes += b.uploadedBytes;
        a.descriptorWrites += b.descriptorWrites;
        a.retries += b.retries;
        a.indexOverflow += b.indexOverflow;
        a.vertexOverflow += b.vertexOverflow;
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
              << " provider=" << (profTotal.providerMs * inv)
              << " dispatch=" << (profTotal.dispatchMs * inv)
              << " | last " << profWindow.chunks << " ms: total=" << (profWindow.totalMs * win)
              << " upload=" << (profWindow.uploadMs * win)
              << " desc=" << (profWindow.descriptorMs * win)
              << " provider=" << (profWindow.providerMs * win)
              << " dispatch=" << (profWindow.dispatchMs * win) << "\n";
    std::cerr << "[OctreeTopology][setup-diag] chunks=" << profTotal.chunks
              << " avg/chunk: nodes=" << (double(profTotal.nodes) * inv)
              << " cells=" << (double(profTotal.cells) * inv)
              << " crossEdges=" << (double(profTotal.crossEdges) * inv)
              << " indices=" << (double(profTotal.indices) * inv)
              << " writtenVerts=" << (double(profTotal.vertices) * inv)
              << " vCap=" << (double(profTotal.vertexCapacity) * inv)
              << " iCap=" << (double(profTotal.indexCapacity) * inv)
              << " bytes=" << (double(profTotal.uploadedBytes) * inv)
              << " descWrites=" << (double(profTotal.descriptorWrites) * inv)
              << " findCell=" << (double(profTotal.findCellCalls) * inv)
              << " reuse=" << (double(profTotal.reuseHits) * inv)
              << " | last " << profWindow.chunks << ": indices=" << (double(profWindow.indices) * win)
              << " crossEdges=" << (double(profWindow.crossEdges) * win)
              << " retries=" << profWindow.retries
              << " idxOverflow=" << profWindow.indexOverflow
              << " vtxOverflow=" << profWindow.vertexOverflow << "\n";
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
              << " provider=" << (profTotal.providerMs * inv)
              << " dispatch=" << (profTotal.dispatchMs * inv)
              << " | totals: nodes=" << profTotal.nodes
              << " cells=" << profTotal.cells
              << " crossEdges=" << profTotal.crossEdges
              << " indices=" << profTotal.indices
              << " writtenVerts=" << profTotal.vertices
              << " vCap=" << profTotal.vertexCapacity
              << " iCap=" << profTotal.indexCapacity
              << " uploadedMB=" << (double(profTotal.uploadedBytes) / (1024.0 * 1024.0))
              << " descWrites=" << profTotal.descriptorWrites
              << " retries=" << profTotal.retries
              << " idxOverflow=" << profTotal.indexOverflow
              << " vtxOverflow=" << profTotal.vertexOverflow
              << " findCell=" << profTotal.findCellCalls
              << " reuse=" << profTotal.reuseHits << "\n";
}

// Single geometry submission per chunk:
//   1) vkCmdFillBuffer clears the per-node written flags (device-local).
//   2) One dispatch scans the compacted cells, appends indices directly into
//      the reserved span and writes each referenced vertex payload exactly
//      once. Only integer counters return to the host; no Cartesian vertex
//      ever crosses back from the GPU.
// Spans are sized on the CPU from the cross-edge topology bound, so there is no
// counting walk. If a defensive overflow is reported, the shader has not
// written out of bounds; the exact required counts are read back and only this
// chunk's span is released and re-reserved.
OctreeTopologyResidentResult OctreeTopologyCompute::computeResident(
    const OctreeTopologyInput& input, const OctreeTopologyTargetProvider& provider,
    const OctreeTopologyReleaseCallback& release) {
    // Profiler entry stamp sits before the lock so totalMs includes the wait
    // for a busy service (the convoy the report measures); the legacy statMs
    // below keeps its old post-lock meaning.
    const auto tEnter = setupDiag ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    std::lock_guard<std::mutex> lock(mutex);
    // Read the descriptor-write counter under the same lock that guards it.
    const uint64_t descWritesAtStart = statDescriptorWrites;
    OctreeTopologyResidentResult res;
    res.crossEdgeCount = input.crossEdgeCount;
    res.maxSegmentsPerCrossEdge = input.maxSegmentsPerCrossEdge;
    res.attempts = 0; // incremented per submitted dispatch
    const size_t nodeCount = input.nodes.size();
    const size_t cellCount = input.cells.size();
    if (app == nullptr || nodeCount == 0) return res;
    if (cellCount == 0 || input.activeVertexCount == 0) {
        // Valid empty chunk: no scan cell at this rung straddles the surface, or
        // no surface node in the flattened neighborhood owns a slot, so the
        // shader cannot emit a triangle. Do not ask the pool for a zero-sized
        // span (it rejects those); settle the revision as empty instead.
        res.attempts = 0; // no dispatch was submitted
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

    ensureCapacity(nodeCount, cellCount);

    OctreeTopologyDemand demand;
    demand.crossEdgeCount = input.crossEdgeCount;
    demand.maxSegmentsPerCrossEdge = std::max<uint32_t>(input.maxSegmentsPerCrossEdge, 1u);
    demand.suggestedVertexCapacity = input.activeVertexCount;
    // The CPU hint is the per-cell dyadic segment sum. If a caller predates the
    // cross-edge accounting, fall back to the full 12-edges-per-cell bound
    // rather than an arbitrary multiplier.
    uint64_t fallbackSegments = uint64_t(input.crossEdgeCount) * demand.maxSegmentsPerCrossEdge;
    if (fallbackSegments == 0)
        fallbackSegments = uint64_t(cellCount) * 12u * demand.maxSegmentsPerCrossEdge;
    demand.suggestedIndexCapacity = input.indexCapacityHint != 0
        ? input.indexCapacityHint
        : octree_topology::indexCapacityFromSegments(fallbackSegments);

    // Per-phase setup timings (VULKAN_SETUP_DIAG only; zero otherwise).
    double uploadMs = 0.0, descriptorMs = 0.0, providerMs = 0.0, dispatchMs = 0.0;
    uint32_t attempts = 0;
    // Worst attempt across retries: a successful grown retry resets the shader
    // counters, but the profile must still report the overflow that caused the
    // growth. Taking the max keeps these counts comparable to one dispatch's
    // triangle count instead of double-counting the same triangles per attempt.
    uint32_t maxIndexOverflow = 0, maxVertexOverflow = 0, maxInvalidSlots = 0;
    OctreeTopologyTarget target;

    const uint32_t cellGroups = static_cast<uint32_t>((cellCount + 63) / 64);

    for (uint32_t attempt = 0; attempt < kMaxSpanRetries; ++attempt) {
        ++attempts;
        timed(providerMs, [&] { target = provider(demand); });
        res.slotIndex = target.slotIndex;
        res.baseVertex = target.baseVertex;
        res.firstIndex = target.firstIndex;
        res.attempts = attempts;

        const bool targetValid = target.vertexBuffer != VK_NULL_HANDLE &&
                                 target.indexBuffer != VK_NULL_HANDLE;
        if (!targetValid ||
            target.vertexCapacity < demand.suggestedVertexCapacity ||
            target.indexCapacity < demand.suggestedIndexCapacity) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 8)
                std::cerr << "[OctreeTopology] span too small for chunk (needVerts="
                          << demand.suggestedVertexCapacity
                          << " needIdx=" << demand.suggestedIndexCapacity
                          << " gotVerts=" << target.vertexCapacity
                          << " gotIdx=" << target.indexCapacity
                          << (targetValid ? "" : " no-target") << "): skipped\n";
            res.vertexCapacity = target.vertexCapacity;
            res.indexCapacity = target.indexCapacity;
            return res; // ok == false
        }
        res.vertexCapacity = target.vertexCapacity;
        res.indexCapacity = target.indexCapacity;

        timed(uploadMs, [&] { uploadInput(input); }); // also resets the counters
        timed(descriptorMs, [&] { writeDescriptors(target); });

        const PoolSpanBinding poolSpan = makePoolSpanBinding(
            target, app->getMinStorageBufferOffsetAlignment());
        TopologyPC pc{};
        pc.cellCount = static_cast<uint32_t>(cellCount);
        pc.targetLod = input.targetLod;
        pc.fromId = static_cast<int32_t>(input.fromId);
        pc.fromBrush = input.fromBrush;
        pc.rootMinSize = glm::vec4(input.rootMin, input.rootSize);
        pc.worldSize = input.rootSize;
        pc.vertexBase = poolSpan.relVertexBase; // span-relative inside binding 10
        pc.vertexCapacity = target.vertexCapacity;
        pc.indexBase = poolSpan.relIndexBase;   // span-relative inside binding 11
        pc.indexCapacity = target.indexCapacity;
        pc.nodeCount = static_cast<uint32_t>(nodeCount);
        pc.diag = setupDiag ? 1u : 0u;

        timed(dispatchMs, [&] {
            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                // Clear the per-node "payload already written" flags. Filling
                // the device-local buffer here keeps the reset on the GPU and
                // scoped to this chunk's dispatch.
                vkCmdFillBuffer(cmd, bNodeWritten.buffer, 0,
                                VkDeviceSize(nodeCount) * sizeof(uint32_t), 0u);
                // Fill writes -> compute atomics/reads of nodeWritten[].
                {
                    VkBufferMemoryBarrier2 fill{};
                    fill.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    fill.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                    fill.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    fill.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    fill.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                    fill.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    fill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    fill.buffer = bNodeWritten.buffer;
                    fill.offset = 0;
                    fill.size = VkDeviceSize(nodeCount) * sizeof(uint32_t);
                    VkDependencyInfo dep{};
                    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.bufferMemoryBarrierCount = 1;
                    dep.pBufferMemoryBarriers = &fill;
                    vkCmdPipelineBarrier2(cmd, &dep);
                }

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
                vkCmdDispatch(cmd, cellGroups, 1, 1);

                // Pool writes -> the cull's shader reads, the indirect draws and
                // the ray-tracing build inputs that read the same buffers.
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
        });

        const uint32_t* counters = (const uint32_t*)bCounter.mappedData;
        const uint32_t attemptedIndices = counters[0];
        const uint32_t fixedOverflow = counters[1];
        const uint32_t vertexExtent = counters[2];
        const uint32_t dropped = counters[3];
        const uint32_t indexOverflow = counters[6];
        const uint32_t vertexOverflow = counters[7];
        const uint32_t invalidSlots = counters[8];
        const uint32_t writtenVertices = counters[9];

        maxIndexOverflow = std::max(maxIndexOverflow, indexOverflow);
        maxVertexOverflow = std::max(maxVertexOverflow, vertexOverflow);
        maxInvalidSlots = std::max(maxInvalidSlots, invalidSlots);
        res.indexCount = attemptedIndices;
        res.writtenVertexCount = writtenVertices;
        res.vertexExtent = vertexExtent;
        res.indexOverflow = maxIndexOverflow;
        res.vertexOverflow = maxVertexOverflow;
        res.invalidSlots = maxInvalidSlots;
        res.attempts = attempts;

        if (fixedOverflow != 0 || invalidSlots != 0) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 8)
                std::cerr << "[OctreeTopology] geometry pass rejected chunk (fixedOverflow="
                          << fixedOverflow << " invalidSlots=" << invalidSlots
                          << " cells=" << cellCount << "): skipped\n";
            return res; // ok == false; caller releases the span
        }

        if (indexOverflow != 0 || vertexOverflow != 0 || dropped != 0) {
            // The shader appended/counted past the span but never wrote out of
            // bounds. Grow only this chunk to the exact reported requirement.
            const uint32_t needIndex = std::max(attemptedIndices, demand.suggestedIndexCapacity);
            const uint32_t needVertex = std::max(vertexExtent, demand.suggestedVertexCapacity);
            const bool noProgress = needIndex <= demand.suggestedIndexCapacity &&
                                    needVertex <= demand.suggestedVertexCapacity;
            if (!release || attempt + 1u >= kMaxSpanRetries || noProgress) {
                static std::atomic<int> logged{0};
                if (logged.fetch_add(1) < 8)
                    std::cerr << "[OctreeTopology] span overflow could not be resolved (idx="
                              << attemptedIndices << "/" << target.indexCapacity
                              << " vert=" << vertexExtent << "/" << target.vertexCapacity
                              << " retries=" << attempt << "): skipped\n";
                return res; // ok == false; caller releases the span
            }
            // Free the reservation before asking the provider for the grown
            // capacities: only this chunk's spans are released/re-reserved.
            release(target.slotIndex);
            res.slotIndex = UINT32_MAX; // already released (caller must not repeat)
            demand.suggestedIndexCapacity = needIndex;
            demand.suggestedVertexCapacity = needVertex;
            res.grew = true;
            continue;
        }

        res.poolVertexCount = vertexExtent;
        res.ok = true;
        break;
    }

    if (!res.ok) return res;

    countStats(statCalls, statMs, statMsLast, statCells, statNodes, statTriangles,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               cellCount, nodeCount, double(res.indexCount) / 3.0);

    if (setupDiag) {
        ProfileSample s;
        s.totalMs = msSince(tEnter); // includes the wait for `mutex`
        s.uploadMs = uploadMs;
        s.descriptorMs = descriptorMs;
        s.providerMs = providerMs;
        s.dispatchMs = dispatchMs;
        s.nodes = nodeCount;
        s.cells = cellCount;
        s.indices = res.indexCount;
        s.vertices = res.writtenVertexCount;
        s.crossEdges = input.crossEdgeCount;
        s.vertexCapacity = res.vertexCapacity;
        s.indexCapacity = res.indexCapacity;
        s.uploadedBytes = uint64_t(nodeCount) * 128u; // 7 input tables, 128 B/node
        s.descriptorWrites = statDescriptorWrites - descWritesAtStart;
        s.retries = attempts > 0 ? attempts - 1 : 0;
        s.indexOverflow = res.indexOverflow;
        s.vertexOverflow = res.vertexOverflow;
        s.findCellCalls = ((const uint32_t*)bCounter.mappedData)[4];
        s.reuseHits = ((const uint32_t*)bCounter.mappedData)[5];
        recordProfile(s);
    }
    return res;
}
