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
    uint32_t maxVertices;  // arena capacity (0 = resident-only: no arena)
    uint32_t keyCapacity;  // rawKeys[] capacity: hard bound on raw vertices
    // Resident pipeline (pc.resident != 0): pass 0 = walk, 1 = dedup + pool
    // vertex write, 2 = index stream write.
    uint32_t resident;
    uint32_t pass;
    uint32_t vertexBase;
    uint32_t vertexCapacity;
    uint32_t indexBase;
    uint32_t indexCapacity;
};

// Output sizing. The arena holds the walk's raw triangle stream (xyz position +
// brush bits, normal + cell tag, hsv) and is only used by the non-resident
// readback: the resident pipeline rebuilds every vertex from its dedup key, so
// its arena can can stay a heuristic and can never invalidate the pool output.
constexpr size_t kVerticesPerCellHint = 24;      // measured raw-vertex density per scan cell
constexpr size_t kMaxVerticesPerCall = 1u << 19; // 512k vertices -> 24 MB of arena buffers
// Hard ceiling on the walk's raw vertex count (3 per triangle) for the
// resident path. The chunk-local index stream has one entry per raw vertex, so
// this is also the index-span bound. Measured worst chunk: ~3.8M raw vertices.
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

    VkDescriptorSetLayoutBinding bindings[18] = {};
    for (int b = 0; b < 18; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 18;
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
    ps.descriptorCount = 18;
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
        vkApp->destroyBuffer(bPos);
        vkApp->destroyBuffer(bNrm);
        vkApp->destroyBuffer(bOutHsv);
        vkApp->destroyBuffer(bCounter);
        vkApp->destroyBuffer(bRawKeys);
        vkApp->destroyBuffer(bDedupTable);
        vkApp->destroyBuffer(bBridgeVerts);
        vkApp->destroyBuffer(bBridgeIdx);
    }
    nodeCap = cellCap = vertexCap = keyCap = tableCap = bridgeVertCap = bridgeIdxCap = 0;
    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    descLayout = VK_NULL_HANDLE;
    descPool = VK_NULL_HANDLE;
}

void OctreeTopologyCompute::ensureCapacity(size_t nodeCount, size_t cellCount,
                                           size_t vertexCount, size_t keyCount) {
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

void OctreeTopologyCompute::ensureBridgeCapacity(size_t uniqueCapacity, size_t indexCapacity) {
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (uniqueCapacity > bridgeVertCap) {
        // 4 x vec4 per deduped vertex = the renderer's 64-byte Vertex.
        app->destroyBuffer(bBridgeVerts);
        bBridgeVerts = app->createBuffer(sizeof(float) * 16 * uniqueCapacity,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bridgeVertCap = uniqueCapacity;
    }
    if (indexCapacity > bridgeIdxCap) {
        app->destroyBuffer(bBridgeIdx);
        bBridgeIdx = app->createBuffer(sizeof(uint32_t) * indexCapacity,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
        bridgeIdxCap = indexCapacity;
    }
}

namespace {
// Compute writes -> host reads. Every submission that hands counters or bridge
// data back to the host ends with this.
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
        hsv[i * 4 + 0] = n.hsv.x;
        hsv[i * 4 + 1] = n.hsv.y;
        hsv[i * 4 + 2] = n.hsv.z;
        hsv[i * 4 + 3] = 0.0f;
        for (int k = 0; k < 4; ++k) child03[i * 4 + k] = n.children[k];
        for (int k = 0; k < 4; ++k) child47[i * 4 + k] = n.children[4 + k];
    }
    for (size_t i = 0; i < cellCount; ++i) cells[i] = input.cells[i];
    uint32_t* counters = (uint32_t*)bCounter.mappedData;
    counters[0] = 0; // emitted vertices (raw, 3 per triangle)
    counters[1] = 0; // arena overflow (bridge readback truncated)
    counters[2] = 0; // deduped vertices written into the resident pool span
    counters[3] = 0; // drops: key capacity or pool span overflow
}

// Points the descriptor set at the arena (non-resident) or at the resident
// target's pools. Bindings 12/13 always point at the keys/table; 14/15 fall
// back to arena buffers when there is no target (the resident passes do not
// run in that case, but every declared binding still needs a valid buffer).
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
    // (stale pool entries -> stretched triangles), while the host-side bridge
    // mirror stays correct — invisible to a bridge-only self-check.
    b.vertexRange = VkDeviceSize(b.relVertexBase) * sizeof(Vertex) + vBytes;

    const VkDeviceSize iBytes = VkDeviceSize(target.indexCapacity) * sizeof(uint32_t);
    const VkDeviceSize iAbsolute = VkDeviceSize(target.firstIndex) * sizeof(uint32_t);
    b.indexOffset = iAbsolute & ~(align - 1);
    b.relIndexBase = static_cast<uint32_t>((iAbsolute - b.indexOffset) / sizeof(uint32_t));
    b.indexRange = VkDeviceSize(b.relIndexBase) * sizeof(uint32_t) + iBytes;
    return b;
}

void OctreeTopologyCompute::writeDescriptors(const OctreeTopologyTarget* target) {
    const bool resident = target != nullptr &&
                          target->vertexBuffer != VK_NULL_HANDLE &&
                          target->indexBuffer != VK_NULL_HANDLE;
    VkBuffer bufs[18] = {bSdf0.buffer, bSdf1.buffer, bCube.buffer, bMeta.buffer, bHsv.buffer,
                         bChild03.buffer, bChild47.buffer, bCells.buffer,
                         bPos.buffer, bNrm.buffer, bOutHsv.buffer, bCounter.buffer,
                         bRawKeys.buffer, bDedupTable.buffer,
                         resident ? target->vertexBuffer : bPos.buffer,
                         resident ? target->indexBuffer : bCells.buffer,
                         bBridgeVerts.buffer, bBridgeIdx.buffer};
    // Every declared binding must name a valid buffer even when the current
    // mode never touches it (the resident path allocates no arena, the
    // non-resident path no bridge). Fall back to a small always-present input
    // buffer: reads/writes of it would be a bug, but the layout stays valid.
    for (int b = 0; b < 18; ++b) {
        if (bufs[b] == VK_NULL_HANDLE) bufs[b] = bCells.buffer;
    }
    VkDescriptorBufferInfo infos[18];
    VkWriteDescriptorSet writes[18] = {};
    for (int b = 0; b < 18; ++b) {
        infos[b] = {bufs[b], 0, app->getClampedWholeSizeRange(bufs[b])};
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = descSet;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    // Bindings 14/15 are the renderer's packed pools: bind this chunk's span
    // (offset+range) so the descriptor stays inside every device's
    // maxStorageBufferRange; the shader indexes it span-relative.
    if (resident) {
        const PoolSpanBinding span = makePoolSpanBinding(*target,
                                                         app->getMinStorageBufferOffsetAlignment());
        infos[14] = {bufs[14], span.vertexOffset, span.vertexRange};
        infos[15] = {bufs[15], span.indexOffset, span.indexRange};
    }
    vkUpdateDescriptorSets(app->getDevice(), 18, writes, 0, nullptr);
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

// Non-resident path: the walk writes the raw triangle stream into the
// host-visible arena, which this assembles back into emission order (stable
// bucket sort by the shader's per-vertex scan-cell tag, so the result matches
// the CPU walker's cell-major order index for index).
OctreeTopologyResult OctreeTopologyCompute::compute(const OctreeTopologyInput& input) {
    std::lock_guard<std::mutex> lock(mutex);
    OctreeTopologyResult res;
    const size_t nodeCount = input.nodes.size();
    const size_t cellCount = input.cells.size();
    if (app == nullptr || nodeCount == 0 || cellCount == 0) return res;
    if (pipeline == VK_NULL_HANDLE) init(app);

    const auto t0 = std::chrono::steady_clock::now();

    const size_t want = cellCount * kVerticesPerCellHint;
    const size_t arena = std::min(want, kMaxVerticesPerCall);
    ensureCapacity(nodeCount, cellCount, arena, /*keyCount=*/arena);
    const size_t vertexArena = std::min(arena, vertexCap);

    uploadInput(input);
    writeDescriptors(nullptr);

    TopologyPC pc{};
    pc.cellCount = (uint32_t)cellCount;
    pc.targetLod = input.targetLod;
    pc.fromId = (int32_t)input.fromId;
    pc.fromBrush = input.fromBrush;
    pc.rootMinSize = glm::vec4(input.rootMin, input.rootSize);
    pc.worldSize = input.rootSize;
    pc.maxVertices = (uint32_t)vertexArena;
    // Non-resident: the arena IS the output, so the arena cap is also the hard
    // representable bound (an overflowing triangle is dropped and reported).
    pc.keyCapacity = (uint32_t)vertexArena;
    pc.resident = 0u;

    const uint32_t cellGroups = (uint32_t)((cellCount + 63) / 64);

    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
        vkCmdDispatch(cmd, cellGroups, 1, 1);
        hostReadBarrier(cmd);
    });

    const uint32_t* counters = (const uint32_t*)bCounter.mappedData;
    const uint32_t rawCount = counters[0];
    const uint32_t arenaOverflow = counters[1];
    const uint32_t dropped = counters[3];
    res.truncated = arenaOverflow > 0 || dropped > 0;
    res.vertexCount = std::min<uint32_t>(rawCount, (uint32_t)vertexArena);
    res.vertexCount -= res.vertexCount % 3u;
    if (res.truncated) {
        countStats(statCalls, statMs, statMsLast, statCells, statNodes, statTriangles,
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                   cellCount, nodeCount, 0.0);
        return res;
    }

    const float* pp = (const float*)bPos.mappedData;
    const float* pn = (const float*)bNrm.mappedData;
    const float* ph = (const float*)bOutHsv.mappedData;

    const uint32_t n = res.vertexCount;
    std::vector<OctreeTopologyVertex> appVerts(n);
    std::vector<uint32_t> appCell(n);
    for (uint32_t i = 0; i < n; ++i) {
        OctreeTopologyVertex v;
        v.position = glm::vec3(pp[i * 4 + 0], pp[i * 4 + 1], pp[i * 4 + 2]);
        uint32_t brushBits = 0;
        std::memcpy(&brushBits, &pp[i * 4 + 3], sizeof(uint32_t));
        v.brushIndex = (int32_t)brushBits;
        v.normal = glm::vec3(pn[i * 4 + 0], pn[i * 4 + 1], pn[i * 4 + 2]);
        v.hsv = glm::vec3(ph[i * 4 + 0], ph[i * 4 + 1], ph[i * 4 + 2]);
        appVerts[i] = v;
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
        const uint32_t run = count[c];
        count[c] = running;
        running += run;
    }
    count[cellCount] = running;

    res.vertices.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t ci = appCell[i];
        res.vertices[count[ci]++] = appVerts[i];
    }
    res.triangleCount = res.vertexCount / 3u;

    countStats(statCalls, statMs, statMsLast, statCells, statNodes, statTriangles,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               cellCount, nodeCount, double(res.triangleCount));
    return res;
}

// Resident path. Two submissions with a host round trip in between:
//   1) walk — writes the dedup keys and the raw count. The index stream has
//      exactly one entry per raw vertex, so its span can only be sized once
//      that count is known; `provider` reserves the span.
//   2) dedup + index stream — writes the renderer's packed pools directly
//      (device-local, never read back) and mirrors the result into the
//      host-visible bridge the CPU-side effect consumers read.
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
    pc.maxVertices = 0u;          // resident: the arena is not needed
    pc.resident = 1u;
    pc.pass = 0u;

    for (int attempt = 0; attempt < 6 && !walkOk; ++attempt) {
        ensureCapacity(nodeCount, cellCount, /*vertexCount=*/0, keyWant);
        ensureBridgeCapacity(nodeCount * kVertexKeysPerNode, keyWant);
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
    pc.vertexBase = poolSpan.relVertexBase; // span-relative inside binding 14
    pc.vertexCapacity = target.vertexCapacity;
    pc.indexBase = poolSpan.relIndexBase; // span-relative inside binding 15
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

        // Pass 1 (dedup into the pooled span + bridge mirror) reads the keys
        // the walk wrote; pass 2 (index stream) reads the table pass 1 filled.
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

    // Bridge: host-visible mirror of exactly what the pool span holds, in the
    // same slot order the index stream references.
    res.bridge.vertices.resize(unique);
    if (unique > 0)
        std::memcpy(res.bridge.vertices.data(), bBridgeVerts.mappedData,
                    size_t(unique) * sizeof(Vertex));
    res.bridge.indices.resize(rawCount);
    if (rawCount > 0)
        std::memcpy(res.bridge.indices.data(), bBridgeIdx.mappedData,
                    size_t(rawCount) * sizeof(uint32_t));

    // Optional verification (VULKAN_POOL_SELFCHECK=1): read the packed pool span
    // back and compare it with the bridge. The topology self-check compares the
    // bridge (what the shader intended) with the CPU walker; this check proves
    // the device pool the renderer actually draws contains the same bytes.
    {
        static const bool poolCheck = std::getenv("VULKAN_POOL_SELFCHECK") != nullptr;
        static std::atomic<int> poolChecks{0};
        if (poolCheck && unique > 0 && rawCount > 0 && poolChecks.fetch_add(1) < 16) {
            const VkDeviceSize vBytes = VkDeviceSize(unique) * sizeof(Vertex);
            const VkDeviceSize iBytes = VkDeviceSize(rawCount) * sizeof(uint32_t);
            Buffer stage = app->createBuffer(vBytes + iBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (stage.buffer != VK_NULL_HANDLE) {
                app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                    VkBufferCopy vc{};
                    vc.srcOffset = VkDeviceSize(target.baseVertex) * sizeof(Vertex);
                    vc.dstOffset = 0;
                    vc.size = vBytes;
                    vkCmdCopyBuffer(cmd, target.vertexBuffer, stage.buffer, 1, &vc);
                    VkBufferCopy ic{};
                    ic.srcOffset = VkDeviceSize(target.firstIndex) * sizeof(uint32_t);
                    ic.dstOffset = vBytes;
                    ic.size = iBytes;
                    vkCmdCopyBuffer(cmd, target.indexBuffer, stage.buffer, 1, &ic);
                });
                const uint8_t* host = static_cast<const uint8_t*>(stage.mappedData);
                size_t vBad = 0;
                size_t iBad = 0;
                for (size_t v = 0; v < unique; ++v) {
                    if (std::memcmp(host + v * sizeof(Vertex), &res.bridge.vertices[v], sizeof(Vertex)) != 0) ++vBad;
                }
                const uint32_t* hostIdx = reinterpret_cast<const uint32_t*>(host + vBytes);
                for (size_t i = 0; i < rawCount; ++i) {
                    if (hostIdx[i] != res.bridge.indices[i]) {
                        if (iBad < 4) {
                            std::cerr << "[pool-check]   idx[" << i << "] pool=" << hostIdx[i]
                                      << " bridge=" << res.bridge.indices[i] << "\n";
                        }
                        ++iBad;
                    }
                }
                if (vBad || iBad) {
                    std::cerr << "[pool-check] MISMATCH verts=" << unique << " bad=" << vBad
                              << " idx=" << rawCount << " bad=" << iBad
                              << " baseVertex=" << target.baseVertex
                              << " vertexBase=" << poolSpan.relVertexBase
                              << " firstIndex=" << target.firstIndex
                              << " indexBase=" << poolSpan.relIndexBase << "\n";
                } else {
                    std::cerr << "[pool-check] ok verts=" << unique << " idx=" << rawCount << "\n";
                }
                app->destroyBuffer(stage);
            }
        }
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
