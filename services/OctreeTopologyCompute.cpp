#include "OctreeTopologyCompute.hpp"
#include "OctreeTopologyCapacity.hpp"
#include "../vulkan/core/VulkanApp.hpp"
#include "../vulkan/resources/Buffer.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
    uint32_t refine;         // 1 = read OctreeVirtualRefine.comp positions
};

// Mirrors the push constant block in OctreeVirtualRefine.comp.
struct RefinePC {
    uint32_t nodeCount;
    uint32_t maxDepth;      // 1..3
    uint32_t budget;        // 1..8
    float minPriority;
    float featureSlope;     // [0, 1]; 1 disables the blend
    float tolerance;        // normalized by node size
    uint32_t flags;         // bit0 = final gradient projection
    uint32_t reserved;
};

// OctreeVirtualRefine.comp statistics layout.
enum RefineStatIndex : uint32_t {
    kRefNodes = 0,
    kRefRefined = 1,
    kRefVirtualCells = 2,
    kRefSdfEvals = 3,
    kRefIntersections = 4,
    kRefSelected = 5,
    kRefMaxDepth = 6,
    kRefBaseResidual = 7,
    kRefRefResidual = 8,
    kRefBaseDist = 9,
    kRefRefDist = 10,
    kRefAbsSdf = 11,
    kRefSkippedAccurate = 12,
    kRefSkippedUnsupported = 13,
    kRefOverflow = 14,
    kRefStatCount = 16,
};

uint32_t envUInt(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v) return fallback;
    return static_cast<uint32_t>(std::clamp<unsigned long>(parsed, lo, hi));
}

float envFloat(const char* name, float fallback, float lo, float hi) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) return fallback;
    return std::clamp(parsed, lo, hi);
}

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

    // Virtual adaptive refinement is on by default; VULKAN_VIRTUAL_REFINE=0
    // restores the exact pre-refinement Dual-Contouring vertex positions.
    // The remaining knobs are bounded here so the shader's local work-queue
    // capacities (depth <= 3, budget <= 8) can never overflow.
    if (const char* v = std::getenv("VULKAN_VIRTUAL_REFINE")) {
        refineEnabled.store(!(v[0] == '0' && v[1] == '\0'));
    }
    refineMaxDepth = envUInt("VULKAN_REFINE_DEPTH", 2, 0, 3);
    refineBudget = envUInt("VULKAN_REFINE_BUDGET", 8, 1, 8);
    refineMinPriority = envFloat("VULKAN_REFINE_MIN_PRIORITY", 0.35f, 0.0f, 1.0f);
    refineFeatureSlope = envFloat("VULKAN_REFINE_SLOPE", 0.90f, 0.0f, 1.0f);
    refineTolerance = envFloat("VULKAN_REFINE_TOL", 0.010f, 1e-5f, 0.5f);
    if (const char* v = std::getenv("VULKAN_REFINE_PROJECT")) {
        refineProject = !(v[0] == '0' && v[1] == '\0');
    }

    std::cerr << "[OctreeTopology] virtual refinement "
              << (refineEnabled.load() ? "on" : "off")
              << " depth=" << refineMaxDepth
              << " budget=" << refineBudget
              << " minPriority=" << refineMinPriority
              << " featureSlope=" << refineFeatureSlope
              << " tol=" << refineTolerance
              << " project=" << (refineProject ? 1 : 0) << "\n";
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

    // 7 node tables + cells + counters + node-written flags + the two pools
    // + refined positions + refinement diagnostics.
    constexpr uint32_t kBindingCount = 14;
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

    // Virtual-refinement pass: same descriptor set layout, different push
    // constants.  The descriptor set is compatible because it is allocated
    // from the same VkDescriptorSetLayout.
    VkPushConstantRange refineRange{};
    refineRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    refineRange.offset = 0;
    refineRange.size = sizeof(RefinePC);

    VkPipelineLayoutCreateInfo refinePli{};
    refinePli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    refinePli.setLayoutCount = 1;
    refinePli.pSetLayouts = &descLayout;
    refinePli.pushConstantRangeCount = 1;
    refinePli.pPushConstantRanges = &refineRange;
    if (vkCreatePipelineLayout(device, &refinePli, nullptr, &refineLayout) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: refine pipeline layout");
    vkApp->resources.addPipelineLayout(refineLayout, "OctreeTopologyCompute: refineLayout");
    initRefinePipeline(device, "shaders/computation/octree/OctreeVirtualRefine.comp.spv");

    // Diagnostic GPU timestamps (VULKAN_SETUP_DIAG only).  The query pool is
    // created per diagnostic chunk, so no lifetime tracking is needed and the
    // normal path never touches a query.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vkApp->getPhysicalDevice(), &props);
    timestampPeriodNs = props.limits.timestampPeriod;
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vkApp->getPhysicalDevice(), &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    if (familyCount > 0) {
        vkGetPhysicalDeviceQueueFamilyProperties(vkApp->getPhysicalDevice(), &familyCount, families.data());
    }
    bool timestampQueue = false;
    for (const VkQueueFamilyProperties& family : families) {
        if ((family.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && family.timestampValidBits > 0) {
            timestampQueue = true;
            break;
        }
    }
    refineTimestamps = timestampPeriodNs > 0.0f &&
                       props.limits.timestampComputeAndGraphics == VK_TRUE &&
                       timestampQueue;
    if (setupDiag && !refineTimestamps) {
        std::cerr << "[OctreeTopology] refine GPU timestamps unavailable on this device; "
                     "reporting host dispatch time only\n";
    }

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

void OctreeTopologyCompute::initRefinePipeline(VkDevice device, const char* modulePath) {
    VkShaderModule sm = app->getOrCreateShaderModule(modulePath);
    VkPipelineShaderStageCreateInfo ssi{};
    ssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssi.module = sm;
    ssi.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = ssi;
    cpi.layout = refineLayout;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &cpi, nullptr, &refinePipeline) != VK_SUCCESS)
        throw std::runtime_error("OctreeTopologyCompute: refine pipeline");
    app->resources.addPipeline(refinePipeline, "OctreeTopologyCompute: refinePipeline");
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
        // Device-local refined positions: written by OctreeVirtualRefine.comp,
        // read by OctreeTriangles.comp in the same submission.  Never read back.
        app->destroyBuffer(bRefinedPos);
        bRefinedPos = app->createBuffer(sizeof(float) * 4 * nodeCount,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
    if (bRefineStats.buffer == VK_NULL_HANDLE) {
        bRefineStats = app->createBuffer(sizeof(uint32_t) * kRefStatCount,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
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
    if (bRefineStats.mappedData != nullptr) {
        uint32_t* refineStats = (uint32_t*)bRefineStats.mappedData;
        for (uint32_t i = 0; i < kRefStatCount; ++i) refineStats[i] = 0;
    }
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
    constexpr uint32_t kBindingCount = 14;
    constexpr uint32_t kInputBindingCount = 10; // 0..9: node tables, cells, counters, node flags
    VkDescriptorBufferInfo infos[kBindingCount]{};
    VkWriteDescriptorSet writes[kBindingCount] = {};
    uint32_t writeCount = 0;

    auto addWrite = [&](uint32_t binding, VkBuffer buf, VkDeviceSize offset, VkDeviceSize range) {
        infos[binding] = {buf, offset, range};
        VkWriteDescriptorSet& w = writes[writeCount++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descSet;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &infos[binding];
    };

    // Bindings 0-9 and 12/13 name buffers that only change when
    // ensureCapacity() recreates one (bufferGeneration). Write them once per
    // capacity change; on the steady-state path they keep the descriptors
    // from the first call.
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
            addWrite(b, bufs[b], 0, app->getClampedWholeSizeRange(bufs[b]));
        }
        VkBuffer extra[2] = {bRefinedPos.buffer, bRefineStats.buffer};
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t b = 12u + i;
            if (extra[i] == VK_NULL_HANDLE) extra[i] = bCells.buffer;
            addWrite(b, extra[i], 0, app->getClampedWholeSizeRange(extra[i]));
        }
        lastWrittenGeneration = bufferGeneration;
    }

    // Bindings 10/11 are the renderer's packed pools and change per chunk.
    // Bind this chunk's span (offset+range) so the descriptor stays inside
    // every device's maxStorageBufferRange; the shader indexes it span-relative.
    const PoolSpanBinding span = makePoolSpanBinding(
        target, app->getMinStorageBufferOffsetAlignment());
    addWrite(10, target.vertexBuffer, span.vertexOffset, span.vertexRange);
    addWrite(11, target.indexBuffer, span.indexOffset, span.indexRange);

    vkUpdateDescriptorSets(app->getDevice(), writeCount, writes, 0, nullptr);
    statDescriptorWrites += writeCount;
}

namespace {
void countStats(uint64_t& calls, double& totalMs, double& cells, double& nodes,
                double& triangles, double ms, size_t cellCount, size_t nodeCount,
                double triangleCount) {
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
        a.refineGpuMs += b.refineGpuMs;
        a.refineNodes += b.refineNodes;
        a.refineRefined += b.refineRefined;
        a.refineCells += b.refineCells;
        a.refineEvals += b.refineEvals;
        a.refineIntersections += b.refineIntersections;
        a.refineSelected += b.refineSelected;
        a.refineMaxDepth = std::max(a.refineMaxDepth, b.refineMaxDepth);
        a.refineSkippedAccurate += b.refineSkippedAccurate;
        a.refineSkippedUnsupported += b.refineSkippedUnsupported;
        a.refineOverflow += b.refineOverflow;
        a.refineBaseResidual += b.refineBaseResidual;
        a.refineRefResidual += b.refineRefResidual;
        a.refineBaseDist += b.refineBaseDist;
        a.refineRefDist += b.refineRefDist;
        a.refineAbsSdf += b.refineAbsSdf;
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
    {
        const double measuredNodes = double(std::max<uint64_t>(
            profTotal.refineNodes > profTotal.refineSkippedUnsupported
                ? profTotal.refineNodes - profTotal.refineSkippedUnsupported : uint64_t(0),
            uint64_t(1)));
        const double invMeasured = 1.0 / measuredNodes;
        std::cerr << "[OctreeTopology][setup-diag] refine: gpu="
                  << (profTotal.refineGpuMs * inv * 1000.0) << "us/chunk"
                  << " nodes=" << (double(profTotal.refineNodes) * inv)
                  << " refined=" << (double(profTotal.refineRefined) * inv)
                  << " cells=" << (double(profTotal.refineCells) * inv)
                  << " evals=" << (double(profTotal.refineEvals) * inv)
                  << " inter=" << (double(profTotal.refineIntersections) * inv)
                  << " selected=" << (double(profTotal.refineSelected) * inv)
                  << " depth=" << profTotal.refineMaxDepth
                  << " skipAcc=" << (double(profTotal.refineSkippedAccurate) * inv)
                  << " skipUnsup=" << (double(profTotal.refineSkippedUnsupported) * inv)
                  << "\n";
        std::cerr << "[OctreeTopology][setup-diag] refine quality: resBase="
                  << (double(profTotal.refineBaseResidual) * invMeasured * 1e-3)
                  << " resRef=" << (double(profTotal.refineRefResidual) * invMeasured * 1e-3)
                  << " sdfBase=" << (double(profTotal.refineBaseDist) * invMeasured * 1e-3)
                  << " sdfRef=" << (double(profTotal.refineRefDist) * invMeasured * 1e-3)
                  << " absSdf=" << (double(profTotal.refineAbsSdf) * invMeasured * 1e-3)
                  << " overflow=" << profTotal.refineOverflow << "\n";
    }
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
    const uint64_t measured = profTotal.refineNodes > profTotal.refineSkippedUnsupported
        ? profTotal.refineNodes - profTotal.refineSkippedUnsupported : 0u;
    const double invMeasured = measured ? 1.0 / double(measured) : 0.0;
    std::cerr << "[OctreeTopology][setup-diag] refine totals: gpuMs="
              << profTotal.refineGpuMs
              << " nodes=" << profTotal.refineNodes
              << " refined=" << profTotal.refineRefined
              << " cells=" << profTotal.refineCells
              << " evals=" << profTotal.refineEvals
              << " inter=" << profTotal.refineIntersections
              << " selected=" << profTotal.refineSelected
              << " maxDepth=" << profTotal.refineMaxDepth
              << " skipAcc=" << profTotal.refineSkippedAccurate
              << " skipUnsup=" << profTotal.refineSkippedUnsupported
              << " resBase=" << (double(profTotal.refineBaseResidual) * invMeasured * 1e-3)
              << " resRef=" << (double(profTotal.refineRefResidual) * invMeasured * 1e-3)
              << " sdfBase=" << (double(profTotal.refineBaseDist) * invMeasured * 1e-3)
              << " sdfRef=" << (double(profTotal.refineRefDist) * invMeasured * 1e-3)
              << " absSdf=" << (double(profTotal.refineAbsSdf) * invMeasured * 1e-3)
              << " overflow=" << profTotal.refineOverflow << "\n";
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
    demand.suggestedVertexCapacity = input.activeVertexCount;
    // The CPU hint is the per-cell dyadic segment sum. If a caller predates the
    // cross-edge accounting, fall back to the full 12-edges-per-cell bound
    // rather than an arbitrary multiplier.
    const uint32_t maxSegmentsPerCrossEdge = std::max<uint32_t>(input.maxSegmentsPerCrossEdge, 1u);
    uint64_t fallbackSegments = uint64_t(input.crossEdgeCount) * maxSegmentsPerCrossEdge;
    if (fallbackSegments == 0)
        fallbackSegments = uint64_t(cellCount) * 12u * maxSegmentsPerCrossEdge;
    demand.suggestedIndexCapacity = input.indexCapacityHint != 0
        ? input.indexCapacityHint
        : octree_topology::indexCapacityFromSegments(fallbackSegments);

    // Per-phase setup timings (VULKAN_SETUP_DIAG only; zero otherwise).
    double uploadMs = 0.0, descriptorMs = 0.0, providerMs = 0.0, dispatchMs = 0.0;
    double refineGpuMs = 0.0; // timestamp query around the refine dispatch
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

        const bool doRefine = refineEnabled.load() && refineMaxDepth > 0 &&
                              refinePipeline != VK_NULL_HANDLE && bRefinedPos.buffer != VK_NULL_HANDLE;
        pc.refine = doRefine ? 1u : 0u;

        // Diagnostic GPU timestamp around only the refinement dispatch.  A
        // per-chunk query pool keeps the lifetime local; this path is gated by
        // VULKAN_SETUP_DIAG and is never used by the normal renderer.
        VkQueryPool timestampPool = VK_NULL_HANDLE;
        const bool wantTimestamp = doRefine && setupDiag && refineTimestamps;
        if (wantTimestamp) {
            VkQueryPoolCreateInfo qp{};
            qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qp.queryCount = 2;
            if (vkCreateQueryPool(app->getDevice(), &qp, nullptr, &timestampPool) != VK_SUCCESS) {
                timestampPool = VK_NULL_HANDLE;
            }
        }
        refineGpuMs = 0.0;

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

                if (doRefine) {
                    // Virtual refinement writes refinedPos[] for every node
                    // entry (usable or fallback).  It must complete before the
                    // geometry pass reads a position, hence the compute ->
                    // compute buffer barrier below.
                    if (timestampPool != VK_NULL_HANDLE) {
                        vkCmdResetQueryPool(cmd, timestampPool, 0, 2);
                        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             timestampPool, 0);
                    }

                    RefinePC rpc{};
                    rpc.nodeCount = static_cast<uint32_t>(nodeCount);
                    rpc.maxDepth = refineMaxDepth;
                    rpc.budget = refineBudget;
                    rpc.minPriority = refineMinPriority;
                    rpc.featureSlope = refineFeatureSlope;
                    rpc.tolerance = refineTolerance;
                    rpc.flags = refineProject ? 1u : 0u;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, refinePipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, refineLayout, 0, 1,
                                            &descSet, 0, nullptr);
                    vkCmdPushConstants(cmd, refineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(RefinePC), &rpc);
                    const uint32_t refineGroups = static_cast<uint32_t>((nodeCount + 63) / 64);
                    vkCmdDispatch(cmd, refineGroups, 1, 1);

                    if (timestampPool != VK_NULL_HANDLE) {
                        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             timestampPool, 1);
                    }

                    VkBufferMemoryBarrier2 refined{};
                    refined.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    refined.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    refined.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    refined.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    refined.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    refined.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    refined.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    refined.buffer = bRefinedPos.buffer;
                    refined.offset = 0;
                    refined.size = VK_WHOLE_SIZE;
                    VkDependencyInfo refineDep{};
                    refineDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    refineDep.bufferMemoryBarrierCount = 1;
                    refineDep.pBufferMemoryBarriers = &refined;
                    vkCmdPipelineBarrier2(cmd, &refineDep);
                }

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TopologyPC), &pc);
                vkCmdDispatch(cmd, cellGroups, 1, 1);

                // Pool writes -> the cull's shader reads, the indirect draws and
                // the ray-tracing build inputs that read the same buffers. The
                // acceleration-structure bits require VK_KHR_acceleration_structure
                // and must stay off the raster path (VUID-10751).
                const bool rtEnabled = app->rayTracingEnabled();
                VkBufferMemoryBarrier2 pool[2] = {};
                for (int b = 0; b < 2; ++b) {
                    pool[b].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    pool[b].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    pool[b].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    pool[b].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT
                                         | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT
                                         | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                         | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    pool[b].dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT
                                          | VK_ACCESS_2_INDEX_READ_BIT
                                          | VK_ACCESS_2_SHADER_READ_BIT;
                    if (rtEnabled) {
                        pool[b].dstStageMask |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                        pool[b].dstAccessMask |= VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    }
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

        if (timestampPool != VK_NULL_HANDLE) {
            uint64_t stamps[2] = {0u, 0u};
            const VkResult qr = vkGetQueryPoolResults(
                app->getDevice(), timestampPool, 0, 2, sizeof(stamps), stamps,
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (qr == VK_SUCCESS && stamps[1] >= stamps[0]) {
                refineGpuMs = double(stamps[1] - stamps[0]) * double(timestampPeriodNs) / 1e6;
            }
            vkDestroyQueryPool(app->getDevice(), timestampPool, nullptr);
            timestampPool = VK_NULL_HANDLE;
        }

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

        res.ok = true;
        break;
    }

    if (!res.ok) return res;

    countStats(statCalls, statMs, statCells, statNodes, statTriangles,
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
        if (bRefineStats.mappedData != nullptr) {
            const uint32_t* rs = (const uint32_t*)bRefineStats.mappedData;
            s.refineGpuMs = refineGpuMs;
            s.refineNodes = rs[kRefNodes];
            s.refineRefined = rs[kRefRefined];
            s.refineCells = rs[kRefVirtualCells];
            s.refineEvals = rs[kRefSdfEvals];
            s.refineIntersections = rs[kRefIntersections];
            s.refineSelected = rs[kRefSelected];
            s.refineMaxDepth = rs[kRefMaxDepth];
            s.refineSkippedAccurate = rs[kRefSkippedAccurate];
            s.refineSkippedUnsupported = rs[kRefSkippedUnsupported];
            s.refineOverflow = rs[kRefOverflow];
            s.refineBaseResidual = rs[kRefBaseResidual];
            s.refineRefResidual = rs[kRefRefResidual];
            s.refineBaseDist = rs[kRefBaseDist];
            s.refineRefDist = rs[kRefRefDist];
            s.refineAbsSdf = rs[kRefAbsSdf];
        }
        recordProfile(s);
    }
    return res;
}
