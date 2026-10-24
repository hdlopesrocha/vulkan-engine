#include "VegetationRenderer.hpp"
#include "../indirect/IndirectRenderer.hpp"
#include "../DescriptorAllocator.hpp"
#include "../DescriptorWriter.hpp"
#include "../RendererUtils.hpp"

#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include "../../../math/Common.hpp" // for NodeID
#include "VegetationRenderer.hpp"
#include "../../../utils/FileReader.hpp"
#include "../../../utils/GrassPlacement.hpp"
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include "../../includes/shader/locations.hpp"
#include "../../includes/shader/vertex_layouts.hpp"

// GPU injection APIs removed. Instances must be generated via compute shader.


VegetationRenderer::VegetationRenderer() {}
VegetationRenderer::~VegetationRenderer() { /* caller must call cleanup(app) */ }

void VegetationRenderer::init() {
    // Default funnel so the shared wind field visibly drives vegetation and
    // fire out of the box (terrain near origin sits ~20-90 m; height 700
    // spans slopes, flames and low sky). User-editable/deletable in the
    // wind widget; TornadoSettings defaults stay inert for slots 1-3.
    {
        TornadoSettings& t0 = windFieldSettings.tornadoes[0];
        t0.baseXZ = glm::vec2(0.0f, 0.0f);
        t0.groundY = 70.0f;
        t0.radius = 60.0f;
        t0.height = 700.0f;
        t0.strength = 35.0f;
        t0.direction = 1.0f;
        t0.deltaTime = 1.0f;
        t0.driftVelocity = glm::vec2(1.0f, 0.0f);
        t0.active = true;
    }
}


void VegetationRenderer::cleanup(VulkanApp* app) {
    (void)app;
    std::vector<NodeID> idsToDestroy;
    idsToDestroy.reserve(chunkBuffers.size());
    for (const auto& [id, _] : chunkBuffers) idsToDestroy.push_back(id);
    for (NodeID id : idsToDestroy) destroyInstanceBuffer(id);
    chunkBuffers.clear();
    chunkInstanceCounts.clear();
    // Stable descriptor set: owned by the app descriptor pool, never freed or
    // deferred-destroyed in the render loop. Just drop our handle; pool
    // teardown reclaims it. Re-init reallocates on demand via ensure.
    vegDescriptorSet = VK_NULL_HANDLE;
    if (vegetationTextureArrayManager && vegTextureListenerId != -1) {
        vegetationTextureArrayManager->removeAllocationListener(vegTextureListenerId);
        vegTextureListenerId = -1;
    }
    billboardVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    billboardVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    billboardVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    billboardVBO.indexBuffer.memory = VK_NULL_HANDLE;
    billboardVBO.indexCount = 0;
    impostorVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    impostorVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    impostorVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    impostorVBO.indexBuffer.memory = VK_NULL_HANDLE;
    impostorVBO.indexCount = 0;
    if (appPtr && windParamsBuffer.buffer != VK_NULL_HANDLE) {
        appPtr->destroyBuffer(windParamsBuffer);
        windParamsBuffer = {};
        windParamsMapped = nullptr;
    }
    if (appPtr && windFieldBuffer.buffer != VK_NULL_HANDLE) {
        appPtr->destroyBuffer(windFieldBuffer);
        windFieldBuffer = {};
        windFieldMapped = nullptr;
        windFieldCacheValid = false;
    }
    destroyCulling();
    appPtr = nullptr;
}

void VegetationRenderer::destroyCulling() {
    if (!appPtr) return;
    VkDevice device = appPtr->getDevice();
    if (concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE) {
        appPtr->destroyBuffer(concatenatedInstanceBuffer);
        concatenatedInstanceBuffer = {};
    }
    if (concatenatedInstanceBufferAlt.buffer != VK_NULL_HANDLE) {
        appPtr->destroyBuffer(concatenatedInstanceBufferAlt);
        concatenatedInstanceBufferAlt = {};
    }
    // Baked heights aux (perf report 22 C2/H4): same lifetime as the
    // concatenated instances they parallel (both pairs).
    if (vegBakedHeightsBuffer.buffer != VK_NULL_HANDLE) {
        appPtr->destroyBuffer(vegBakedHeightsBuffer);
        vegBakedHeightsBuffer = {};
    }
    if (vegBakedHeightsBufferAlt.buffer != VK_NULL_HANDLE) {
        appPtr->destroyBuffer(vegBakedHeightsBufferAlt);
        vegBakedHeightsBufferAlt = {};
    }
    concatActiveUsesAlt = false;
    concatSubmittedUsesAlt = false;
    // Staging pool (perf report 22 M10): destroy every pooled buffer. Any
    // late fence callback is guarded by an index/size check, so clearing the
    // vector here cannot cause an out-of-bounds release.
    for (auto& e : stagingPool) {
        if (e.buffer.buffer != VK_NULL_HANDLE) {
            appPtr->destroyBuffer(e.buffer);
            e.buffer = {};
        }
    }
    stagingPool.clear();
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f) {
        if (compactedCmdBuffers[f].buffer != VK_NULL_HANDLE) {
            appPtr->destroyBuffer(compactedCmdBuffers[f]);
            compactedCmdBuffers[f] = {};
        }
        if (visibleCountBuffers[f].buffer != VK_NULL_HANDLE) {
            appPtr->destroyBuffer(visibleCountBuffers[f]);
            visibleCountBuffers[f] = {};
        }
        if (impostorCompactBuffers[f].buffer != VK_NULL_HANDLE) {
            appPtr->destroyBuffer(impostorCompactBuffers[f]);
            impostorCompactBuffers[f] = {};
        }
        if (impostorCountBuffers[f].buffer != VK_NULL_HANDLE) {
            appPtr->destroyBuffer(impostorCountBuffers[f]);
            impostorCountBuffers[f] = {};
        }
    }
    if (consolidationFence != VK_NULL_HANDLE) {
        VulkanApp::waitFence(device, consolidationFence);
    }
    consolidationPending = false;
    vegNumChunks = 0;
    vegMainCompactCapacity = 0;
    vegChunkInfoCapacity = 0;
    vegCascadeCompactCapacity = 0;
    vegCascadeCullInited = false;
    vegPreallocatedChunks = 0;
    vegPreallocatedInstances = 0;
    vegPreallocated = false;
    vegChunkInfoMapped = nullptr;
    vegConsolidationDirty = true;
}

void VegetationRenderer::preallocate(VulkanApp* app, uint32_t maxChunks,
                                     uint32_t maxInstancesPerChunk) {
    if (!app) return;
    if (vegPreallocated) {
        // Idempotent: same reservation is a no-op; different sizes are a
        // sizing bug (reallocation would reintroduce runtime churn).
        const bool same = (vegPreallocatedChunks == maxChunks) &&
            (vegPreallocatedInstances == static_cast<VkDeviceSize>(maxChunks) * maxInstancesPerChunk);
        if (!same) {
            std::cerr << "[veg] preallocate: already reserved chunks=" << vegPreallocatedChunks
                      << " — reallocation refused; size once at startup\n";
            assert(false && "VegetationRenderer::preallocate called twice with different sizes");
        }
        return;
    }
    auto device = app->getDevice();

    // Concatenated instance buffer: worst-case instances, device-local.
    // Single large buffer + offset management (no sparse aliasing: VMA has
    // no sparse allocator and most iGPUs lack sparseResidencyBuffer).
    // Stride is kInstanceStride (pos vec4 + normal vec4 per instance).
    const VkDeviceSize concatSize =
        static_cast<VkDeviceSize>(maxChunks) * maxInstancesPerChunk * kInstanceStride;
    concatenatedInstanceBuffer = app->createBuffer(concatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Baked heights aux buffer (perf report 22 C2/H4): one float per
    // concatenated slot (worst case, like the instance buffer it parallels:
    // +32 MB at defaults). Written by the bake dispatch during
    // consolidateChunks before any draw can reference it.
    const VkDeviceSize bakedSize =
        static_cast<VkDeviceSize>(maxChunks) * maxInstancesPerChunk * sizeof(float);
    vegBakedHeightsBuffer = app->createBuffer(bakedSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Fail fast like the pipeline creations below: every vegetation draw
    // binds this buffer for the baked-height attribute, so a missing aux
    // buffer would trip vertex-input validation on first draw.
    if (vegBakedHeightsBuffer.buffer == VK_NULL_HANDLE)
        throw std::runtime_error("VegetationRenderer: failed to create baked heights buffer");

    // Second pair (flicker fix): consolidation copies into the inactive pair
    // while the draws read the active one; the pair swaps on the fence-
    // complete publish (pollConsolidation). Same sizes/usages as pair A.
    concatenatedInstanceBufferAlt = app->createBuffer(concatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vegBakedHeightsBufferAlt = app->createBuffer(bakedSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (concatenatedInstanceBufferAlt.buffer == VK_NULL_HANDLE || vegBakedHeightsBufferAlt.buffer == VK_NULL_HANDLE)
        throw std::runtime_error("VegetationRenderer: failed to create double-buffered instance pair");

    // Bake compute pipeline + descriptor sets, written once here (all four
    // buffer handles are fixed for the session, so the sets never need
    // refresh; the consolidation binds the set matching its copy targets).
    {
        VkShaderModule bakeModule = app->getOrCreateShaderModule("shaders/computation/vegetation/VegetationRendererBake.comp.spv");
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &vegBakeDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: failed to create bake descriptor set layout");
        app->resources.addDescriptorSetLayout(vegBakeDescSetLayout, "VegetationRenderer: bakeDescSetLayout");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(uint32_t);
        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &vegBakeDescSetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
        if (vkCreatePipelineLayout(device, &plInfo, nullptr, &vegBakePipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: failed to create bake pipeline layout");
        app->resources.addPipelineLayout(vegBakePipelineLayout, "VegetationRenderer: bakePipelineLayout");

        VkComputePipelineCreateInfo pipeInfo{};
        pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = bakeModule;
        pipeInfo.stage.pName = "main";
        pipeInfo.layout = vegBakePipelineLayout;
        if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &pipeInfo, nullptr, &vegBakePipeline) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: failed to create bake pipeline");
        app->resources.addPipeline(vegBakePipeline, "VegetationRenderer: bakePipeline");

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 2; // one per double-buffered pair
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &vegBakeDescPool) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: failed to create bake descriptor pool");
        app->resources.addDescriptorPool(vegBakeDescPool, "VegetationRenderer: bakeDescPool");
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = vegBakeDescPool;
        allocInfo.descriptorSetCount = 2;
        VkDescriptorSetLayout bakeLayouts[2] = { vegBakeDescSetLayout, vegBakeDescSetLayout };
        allocInfo.pSetLayouts = bakeLayouts;
        VkDescriptorSet bakeSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(device, &allocInfo, bakeSets) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: failed to allocate bake descriptor sets");
        app->resources.addDescriptorSet(bakeSets[0], "VegetationRenderer: bakeDescSet");
        app->resources.addDescriptorSet(bakeSets[1], "VegetationRenderer: bakeDescSetAlt");
        vegBakeDescSet = bakeSets[0];
        vegBakeDescSetAlt = bakeSets[1];
        // Set 0 -> pair A (concat A + heights A); set 1 -> pair B. The
        // consolidation dispatch binds the set matching its copy targets
        // (the inactive pair), so each generation bakes its own heights
        // alongside its own instances.
        DescriptorWriter(device)
            .writeBuffer(bakeSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         concatenatedInstanceBuffer.buffer, 0,
                         app->getClampedWholeSizeRange(concatenatedInstanceBuffer.buffer))
            .writeBuffer(bakeSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegBakedHeightsBuffer.buffer, 0,
                         app->getClampedWholeSizeRange(vegBakedHeightsBuffer.buffer))
            .writeBuffer(bakeSets[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         concatenatedInstanceBufferAlt.buffer, 0,
                         app->getClampedWholeSizeRange(concatenatedInstanceBufferAlt.buffer))
            .writeBuffer(bakeSets[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegBakedHeightsBufferAlt.buffer, 0,
                         app->getClampedWholeSizeRange(vegBakedHeightsBufferAlt.buffer))
            .flush();
        bakeModule = VK_NULL_HANDLE;
    }

    // Per-frame compact/count buffers (billboard + impostor) at maxChunks.
    // DEVICE_LOCAL cull outputs (GPU-only): zero-initialized by createBuffer,
    // reset each frame by the merged dispatch's vkCmdFillBuffer. Never mapped.
    const VkDeviceSize compactedSize =
        static_cast<VkDeviceSize>(std::max(256u, maxChunks)) * sizeof(VkDrawIndexedIndirectCommand);
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f) {
        compactedCmdBuffers[f] = app->createBuffer(compactedSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (visibleCountBuffers[f].buffer == VK_NULL_HANDLE) {
            visibleCountBuffers[f] = app->createBuffer(sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (impostorCompactBuffers[f].buffer == VK_NULL_HANDLE) {
            impostorCompactBuffers[f] = app->createBuffer(compactedSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (impostorCountBuffers[f].buffer == VK_NULL_HANDLE) {
            impostorCountBuffers[f] = app->createBuffer(sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
    }
    vegMainCompactCapacity = maxChunks;

    // Shared GPU chunk-info table (vec4 triple per chunk).
    vegChunkInfoCapacity = maxChunks;
    vegChunkInfoBuffer = app->createBuffer(sizeof(glm::vec4) * 3 * maxChunks,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vegChunkInfoMapped = vegChunkInfoBuffer.map(0);

    // Cascade buffers at maxChunks (per frame × 3 cascades, billboard + impostor).
    // DEVICE_LOCAL cull outputs (GPU-only): zero-initialized by createBuffer,
    // reset each frame by vkCmdFillBuffer in the merged dispatch. Never mapped.
    const VkDeviceSize cascadeCompactSize =
        static_cast<VkDeviceSize>(maxChunks) * sizeof(VkDrawIndexedIndirectCommand);
    if (!vegCascadeCullInited) {
        vegCascadeCullInited = true;
        for (uint32_t f = 0; f < VEG_CULL_FRAMES; f++) {
            for (uint32_t c = 0; c < 3; c++) {
                vegCascadeCullFrames[f].compactBuffers[c] = app->createBuffer(cascadeCompactSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                vegCascadeCullFrames[f].countBuffers[c] = app->createBuffer(sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                vegCascadeCullFrames[f].impostorCompactBuffers[c] = app->createBuffer(cascadeCompactSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                vegCascadeCullFrames[f].impostorCountBuffers[c] = app->createBuffer(sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            }
        }
    }
    vegCascadeCompactCapacity = maxChunks;

    vegPreallocatedChunks = maxChunks;
    vegPreallocatedInstances = static_cast<VkDeviceSize>(maxChunks) * maxInstancesPerChunk;
    vegPreallocated = true;
    (void)device;
    std::cerr << "[veg] preallocated: chunks=" << maxChunks
              << " instances=" << vegPreallocatedInstances
              << " (concat " << (concatSize >> 20) << " MB x2 pairs, zero runtime realloc)\n";
}

void VegetationRenderer::consolidateChunks(VulkanApp* app) {
    if (chunkBuffers.empty()) return;
    if (!app) return;

    // Generational submit (no-vegetation fix): at most one consolidation in
    // flight; a new generation submits only when the chunk set moved since
    // the last submit. Publish (vegNumChunks + completed snapshot) happens
    // exclusively in pollConsolidation() on fence completion, so every
    // generation terminates independently of later publishes — the old
    // version-match-to-publish rule livelocked (dirty never cleared while
    // streaming) and starved the cull forever. Same-queue FIFO orders this
    // submit ahead of the cull/draw CBs recorded later this frame, and the
    // in-CB barriers below cover the copy->bake->vertex chain as before.
    std::lock_guard<std::mutex> lk(consolidationMutex);
    if (consolidationPending) return; // one generation in flight; dirty re-triggers later
    // Capture the version BEFORE building the snapshot and record THAT exact
    // value as submitted below. Re-loading at the end could observe a publish
    // that landed after the snapshot was frozen, marking the un-snapshotted
    // change as "already submitted" — the gate would then skip it until some
    // unrelated later publish re-bumped the version (tail chunks of a stream
    // could stay unconsolidated = randomly missing chunks). The pre-snapshot
    // load is conservative: a racing mutation leaves version !=
    // submittedVersion_, forcing at worst one redundant next-frame generation
    // instead of losing the update.
    const uint64_t versionAtSnapshot = chunkBufferVersion.load(std::memory_order_acquire);
    if (versionAtSnapshot == submittedVersion_) return; // nothing new
    // Don't submit while the streaming batch copies are still in flight: the
    // snapshot below pins their device buffers, and the batch fence callback
    // (which fills chunkBuffers) may not have run yet, so the map would miss
    // them. Defer a frame; the dirty flag stays set. Ground-truth query, so
    // a stalled pump cannot wedge this forever.
    if (streamingUploadFence != VK_NULL_HANDLE && !app->isFenceDone(streamingUploadFence)) {
        return;
    }
    auto device = app->getDevice();

    // All culling buffers are pre-allocated ONCE via preallocate() at startup
    // (SceneRenderer::initSlottedMode). First use without preallocate() falls
    // back to reserving worst case now; afterwards capacity is FIXED — any
    // overflow is a sizing bug (assert), never a runtime realloc, so the
    // steady state issues zero vmaCreateBuffer calls.
    if (!vegPreallocated) {
        preallocate(app);
    }

    // Count chunks with valid geometry (GPU metadata buffer removed — the CPU
    // culling paths write draw commands directly). The submit snapshot below
    // freezes (chunk, buffer, count, dstOffset, AABB) rows: the copy loop and
    // writeVegChunkInfo both consume the snapshot, never the live map, so
    // publishes landing mid-flight cannot skew copy regions apart from the
    // info table (the unordered_map order instability that the old
    // version-match rule papered over by starving the cull).
    // vegNumChunks is NOT set here: it counts the last FENCE-COMPLETE
    // generation (published by pollConsolidation), which is what the cull
    // and the draws may reference.
    submittedSnapshot_.clear();
    size_t totalInstances = 0;
    {
        VkDeviceSize off = 0;
        for (const auto& [chunkId, buf] : chunkBuffers) {
            if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
            ConsolEntry e;
            e.chunkId = chunkId;
            e.buffer = buf.buffer;
            e.count = buf.count;
            e.dstOffset = off * kInstanceStride;
            e.aabbMin = buf.aabbMin;
            e.aabbMax = buf.aabbMax;
            submittedSnapshot_.push_back(e);
            off += buf.count;
        }
        totalInstances = static_cast<size_t>(off);
    }
    uint32_t chunkIdx = static_cast<uint32_t>(submittedSnapshot_.size());

    if (chunkIdx == 0) {
        // Map emptied between the dirty flag and this submit: nothing to
        // copy; record the (empty) submit so the version gate converges.
        submittedVersion_ = versionAtSnapshot;
        return;
    }
    {
        VkMemoryRequirements reqs{};
        vkGetBufferMemoryRequirements(device, concatenatedInstanceBuffer.buffer, &reqs);
        const VkDeviceSize concatSize = static_cast<VkDeviceSize>(totalInstances) * kInstanceStride;
        if (reqs.size < concatSize) {
            std::cerr << "[veg] consolidateChunks: instance capacity exceeded "
                      << "(need " << totalInstances << " have "
                      << (reqs.size / kInstanceStride) << ") — bump preallocate() estimates\n";
            assert(false && "VegetationRenderer instance capacity exceeded");
            submittedSnapshot_.clear();
            return;
        }
        if (chunkIdx > vegMainCompactCapacity) {
            std::cerr << "[veg] consolidateChunks: chunk capacity exceeded "
                      << "(need " << chunkIdx << " cap " << vegMainCompactCapacity << ")\n";
            assert(false && "VegetationRenderer chunk capacity exceeded");
            submittedSnapshot_.clear();
            return;
        }
    }

    // Copy per-chunk instance data into the concatenated buffer (GPU→GPU).
    // Regions come from the submit snapshot (frozen above), never the live
    // map, so the bytes always match the info table built from the same
    // snapshot in writeVegChunkInfo.
    if (inactiveConcatenatedBuffer().buffer != VK_NULL_HANDLE) {
        // Manual record+submit (instead of runSingleTimeCommandsAsync) so the
        // copy can carry a TIMELINE WAIT: it overwrites the INACTIVE pair,
        // whose last readers were the previous era's vegetation draws on the
        // vegetation queue (and transitively the shadow cascade draws). The
        // frame thread pushes the latest signaled vegetation timeline value
        // (setConsolidationWait); waiting it orders this copy strictly after
        // those readers, so no draw ever sees partially overwritten bytes.
        VkCommandBuffer consolCmd = app->allocatePrimaryCommandBuffer();
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (consolCmd == VK_NULL_HANDLE || vkBeginCommandBuffer(consolCmd, &cbi) != VK_SUCCESS) {
            // Rare allocation hiccup: skip this generation; the dirty flag
            // keeps it scheduled and the next frame retries.
            return;
        }
        auto recordConsolidation = [&](VkCommandBuffer cmd) {
            // WRITE-AFTER-READ/WRITE guard: the concatenated instance buffer is
            // read as vertex attributes by the previous frame's vegetation
            // draws, and its per-chunk sources were written by earlier transfer
            // submits. ALL_COMMANDS srcStage covers the full draw-pipeline span
            // sync validation attributes vertex-attribute reads to
            // (SYNC-HAZARD-WRITE-AFTER-READ reported on GPU-assisted runs).
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            // SHADER_WRITE: the per-chunk sources may have been produced by the
            // GPU placement dispatch (compute) instead of the legacy staging
            // copy (TRANSFER_WRITE); both are made available here. ALL_COMMANDS
            // srcStage already spans the compute stage.
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            mb.srcAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                               VK_ACCESS_2_TRANSFER_READ_BIT |
                               VK_ACCESS_2_TRANSFER_WRITE_BIT |
                               VK_ACCESS_2_SHADER_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cmd, &dep);

            uint32_t bakedTotal = 0;
            for (const auto& e : submittedSnapshot_) {
                if (e.buffer == VK_NULL_HANDLE || e.count == 0) continue;
                VkBufferCopy region{};
                region.srcOffset = 0;
                region.dstOffset = e.dstOffset;
                region.size = static_cast<VkDeviceSize>(e.count) * kInstanceStride;
                vkCmdCopyBuffer(cmd, e.buffer, inactiveConcatenatedBuffer().buffer, 1, &region);
                bakedTotal += e.count;
            }

            // Bake per-instance heights (perf report 22 C2/H4): the copies
            // above wrote the live prefix [0, off) (TRANSFER_WRITE); publish
            // to the bake dispatch, run it, then publish the aux buffer to
            // vertex-attribute reads. Same synchronous submit: aux is valid
            // before any draw can reference it, and stays in lockstep with
            // the concatenated instances (baked together, never modified
            // independently).
            if (vegBakePipeline != VK_NULL_HANDLE && vegBakeDescSet != VK_NULL_HANDLE
                && inactiveBakedHeightsBuffer().buffer != VK_NULL_HANDLE && bakedTotal > 0) {
                VkMemoryBarrier2 bakeBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                bakeBar.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                bakeBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                bakeBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                bakeBar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                VkDependencyInfo bakeDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                bakeDep.memoryBarrierCount = 1;
                bakeDep.pMemoryBarriers = &bakeBar;
                vkCmdPipelineBarrier2(cmd, &bakeDep);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegBakePipeline);
                {
                    // Bind the set whose pair matches this submit's copy targets (the
                    // INACTIVE pair): active A -> set B (Alt), active B -> set A.
                    VkDescriptorSet bakeSet = concatActiveUsesAlt ? vegBakeDescSet : vegBakeDescSetAlt;
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            vegBakePipelineLayout, 0, 1, &bakeSet, 0, nullptr);
                }
                const uint32_t bakeCount = bakedTotal;
                vkCmdPushConstants(cmd, vegBakePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(uint32_t), &bakeCount);
                vkCmdDispatch(cmd, (bakeCount + 63u) / 64u, 1, 1);

                VkMemoryBarrier2 bakeOutBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                bakeOutBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                bakeOutBar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                bakeOutBar.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
                bakeOutBar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                VkDependencyInfo bakeOutDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                bakeOutDep.memoryBarrierCount = 1;
                bakeOutDep.pMemoryBarriers = &bakeOutBar;
                vkCmdPipelineBarrier2(cmd, &bakeOutDep);
            }
        }; // end recordConsolidation
        recordConsolidation(consolCmd);

        // Submit on the graphics queue (submission order to vkEndCommandBuffer
        // is handled by the submit helper), carrying the vegetation-timeline
        // wait (see setConsolidationWait). registerSignal=false: no timeline
        // registration, the returned fence IS this generation's completion.
        std::vector<VkSemaphore> consolWaitSems;
        std::vector<uint64_t> consolWaitValues;
        if (consolidationWaitSemaphore != VK_NULL_HANDLE) {
            consolWaitSems.push_back(consolidationWaitSemaphore);
            consolWaitValues.push_back(consolidationWaitValue);
        }
        VkFence consolFence = app->submitCommandBufferAsyncToQueue(
            consolCmd, app->getGraphicsQueue(), nullptr,
            consolWaitSems, false, {}, consolWaitValues, 0, {}, false);
        // Generation submitted: record the fence + version and return. Veg
        // draws must not reference this prefix until pollConsolidation
        // observes fence completion (same-queue FIFO + the in-CB barriers
        // above order the bytes before later cull/draw CBs; the CPU-side
        // dirty flag stays set meanwhile). Per-chunk sources are safe to
        // reference: every destroy path funnels through
        // deferDestroyUntilFence(VK_NULL_HANDLE) (drain after ALL pending
        // submits, which include this one), so no source buffer can die
        // mid-copy; the concatenated/bake targets are preallocated for the
        // app lifetime. No flag callback is registered: publishing is owned
        // exclusively by pollConsolidation (ground-truth fence query), so a
        // stalled pump can neither wedge nor duplicate it.
        consolidationFence = consolFence;
        consolidationPending = true;
        // Pair this submit wrote (the then-inactive one); publish swaps to it.
        concatSubmittedUsesAlt = !concatActiveUsesAlt;
        // Record the PRE-snapshot version (see the gate above): a mutation that
        // landed during recording must be re-submitted next frame, not skipped.
        submittedVersion_ = versionAtSnapshot;
        consolEpochAtSubmit_ = consolEpoch_;
    }
}

// Publish the in-flight consolidation generation once its fence signals.
// Idempotent per generation (second call after publish is a no-op), so both
// prepareCull and prepareCullCascades may call it every frame. Returns true
// when a NEW generation completed (caller pushes metadata). Uses the
// ground-truth fence query: unlike the old tracking-list poll, completion
// here cannot stick when the pump is not running.
bool VegetationRenderer::pollConsolidation() {
    if (!consolidationPending) return false;
    if (!appPtr) return false;
    if (consolidationFence != VK_NULL_HANDLE && !appPtr->isFenceDone(consolidationFence))
        return false; // genuinely in flight; try again next frame
    std::lock_guard<std::mutex> lk(consolidationMutex);
    if (!consolidationPending) return false; // raced with another publisher
    if (consolEpochAtSubmit_ != consolEpoch_) {
        // A reset (clearAllInstances) abandoned this generation after submit:
        // drop it instead of publishing stale rows. The dirty flag left by
        // the clear drives a fresh submit once new chunks arrive.
        consolidationPending = false;
        consolidationFence = VK_NULL_HANDLE;
        submittedSnapshot_.clear();
        return false;
    }
    consolidationPending = false;
    consolidationFence = VK_NULL_HANDLE;
    // Flicker fix: make the pair this generation wrote the ACTIVE one from
    // now on. Draws recorded after this point (same frame's veg task, which
    // waits tlCull = after prepareCull/publish) bind the new pair; the
    // now-inactive pair is free for the next consolidation copy to overwrite.
    concatActiveUsesAlt = concatSubmittedUsesAlt;
    completedSnapshot_ = submittedSnapshot_;
    vegNumChunks = static_cast<uint32_t>(completedSnapshot_.size());
    vegChunkInfoDirty = true;
    // Newer publishes need the next generation; a quiet tree converges here.
    if (chunkBufferVersion.load(std::memory_order_acquire) == submittedVersion_)
        vegConsolidationDirty = false;
    return true;
}

void VegetationRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj) {
    // Generational flow (no-vegetation fix): publish any fence-complete
    // generation first, then submit a new one only if the set moved. The
    // cull below runs on the last COMPLETED snapshot (vegNumChunks) even
    // while dirty/new work streams — the old code skipped the cull on any
    // dirt and required an exact version match to publish, which livelocked
    // during streaming (dirty never cleared) so nothing ever drew.
    const bool completedNow = pollConsolidation();
    // Consolidate if chunks have changed since the last submit.
    if (vegConsolidationDirty && !chunkBuffers.empty()) {
        consolidateChunks(appPtr);
    }
    if (vegNumChunks == 0) return;

    // Ensure the shared GPU chunk-info table exists (created lazily so the main
    // cull works even when shadows/cascade-cull are disabled) and grow it if the
    // scene has more chunks than its current capacity (overflow would corrupt
    // adjacent GPU-visible memory).
    ensureChunkInfo(appPtr);

    // Fixed-capacity compact buffers (preallocate() at startup). Growth is
    // removed: overflow is a sizing bug (assert), never a runtime realloc.
    if (vegNumChunks > vegMainCompactCapacity) {
        std::cerr << "[veg] prepareCull: compact capacity exceeded "
                  << "(need " << vegNumChunks << " cap " << vegMainCompactCapacity << ")\n";
        assert(false && "VegetationRenderer compact capacity exceeded");
        return;
    }

    // Upload the GPU chunk table (AABBs + instance counts + firstInstance
    // offsets) into the host-visible chunk-info buffer. Iteration order MUST
    // match consolidateChunks' concatenated-instance copy order. Also (re)builds
    // vegChunkInfoMap (keyed by the solid mesh id) for the merged cull.
    writeVegChunkInfo();

    // The vegetation cull is now MERGED into the SOLID IndirectRenderer's single
    // IndirectRenderer.comp dispatch. We only hand it our per-frame output buffers (so its
    // merged dispatch writes billboard/impostor commands in place) plus the
    // per-chunk veg metadata (so it can build the binding-9 table keyed by solid
    // draw entry). The actual GPU dispatch happens inside solidIR->prepareCull,
    // which runs AFTER this call in the frame (see MyApp.cpp ordering).
    if (solidIR) {
        solidIR->setVegetationCullData(compactedCmdBuffers, visibleCountBuffers,
                                       impostorCompactBuffers, impostorCountBuffers);
        // Only push the (potentially large) metadata map when a NEW
        // generation completed (i.e. after a consolidation publish), to avoid
        // forcing a solid draw-list rebuild every frame.
        if (completedNow) {
            solidIR->setVegetationChunkInfo(vegChunkInfoMap);
        }
    }
}

void VegetationRenderer::ensureChunkInfo(VulkanApp* app) {
    // Fixed-capacity table, reserved once via preallocate() (default 4096).
    // Never grown at runtime: overflow is a sizing bug (assert), so the
    // steady state issues zero vmaCreateBuffer calls.
    if (vegChunkInfoBuffer.buffer == VK_NULL_HANDLE) {
        preallocate(app);
        return;
    }
    if (vegNumChunks > vegChunkInfoCapacity) {
        std::cerr << "[veg] ensureChunkInfo: capacity exceeded "
                  << "(need " << vegNumChunks << " cap " << vegChunkInfoCapacity << ")\n";
        assert(false && "VegetationRenderer chunk-info capacity exceeded");
    }
}

// ── Cascade-aware culling for vegetation shadows ──────────────────────────────

void VegetationRenderer::initCascadeCull(VulkanApp* app) {
    if (vegCascadeCullInited) return;
    // All cascade + chunk-info buffers are owned by preallocate() (worst-case
    // 4096 chunks). Delegating here keeps the single init-time reservation
    // path; the old 1024-entry sizing is retired.
    preallocate(app);
}

void VegetationRenderer::writeVegChunkInfo() {
    if (!vegChunkInfoMapped) return;
    // M11 (perf report 22): the table content is a pure function of the
    // last COMPLETED consolidation snapshot (counts, AABBs, cumulative
    // firstInstance), so it only changes when a new generation publishes.
    // Built from completedSnapshot_ — never the live map — so publishes
    // landing between the copy submit and this table cannot skew the
    // firstInstance mapping apart from the copied regions (which would
    // draw wrong instances; the pre-generational code papered over this
    // with a version-match rule that starved the cull instead).
    if (!vegChunkInfoDirty) return;
    glm::vec4* dst = static_cast<glm::vec4*>(vegChunkInfoMapped);
    uint32_t idx = 0;
    // Record (solid mesh id, instance count) per snapshotted chunk so we can
    // also build vegChunkInfoMap (fed to the solid IndirectRenderer's merged
    // cull). Tombstoned rows (count 0) are included with 0: the cull skips
    // them (meta.x < 1) while later rows keep stable offsets.
    std::vector<std::pair<uint32_t, uint32_t>> order;
    order.reserve(completedSnapshot_.size());
    for (const auto& e : completedSnapshot_) {
        if (idx >= vegChunkInfoCapacity) {
            // Unreachable (submit-time capacity assert covers the snapshot,
            // which only shrinks afterwards via tombstones) — guard against
            // future call sites writing past the buffer.
            std::cerr << "[veg] FATAL: writeVegChunkInfo overflow cap=" << vegChunkInfoCapacity << "\n";
            break;
        }
        dst[idx * 3 + 0] = glm::vec4(e.aabbMin, 0.0f);
        dst[idx * 3 + 1] = glm::vec4(e.aabbMax, 0.0f);
        dst[idx * 3 + 2] = glm::vec4(static_cast<float>(e.count), 0.0f, 0.0f, 0.0f);
        order.push_back({static_cast<uint32_t>(e.chunkId), e.count});
        ++idx;
    }
    // firstInstance per chunk = cumulative instance count (matches the
    // submit-time copy order frozen in the snapshot).
    uint32_t instOff = 0;
    vegChunkInfoMap.clear();
    for (uint32_t i = 0; i < idx; ++i) {
        dst[i * 3 + 2].y = static_cast<float>(instOff);
        // Key by the SOLID mesh id (== uint32 chunk NodeID) so the IndirectRenderer
        // can index the binding-9 table by its own solid draw entry index.
        vegChunkInfoMap[order[i].first] = glm::vec4(static_cast<float>(order[i].second),
                                                    static_cast<float>(instOff), 0.0f, 0.0f);
        instOff += order[i].second;
    }
    vegChunkInfoDirty = false;
}

void VegetationRenderer::prepareCullCascades(VkCommandBuffer cmd,
                                              const glm::mat4 cascadeMatrices[3]) {
    if (!appPtr) return;
    if (!vegCascadeCullInited) initCascadeCull(appPtr);

    if (!solidIR) return;
    // The veg cascade streams are written by the SOLID IndirectRenderer's merged
    // IndirectRenderer.comp dispatch (doVegCascade=true) into the buffers bound via
    // setVegCascadeData. Those are indexed by the cull frame, so reuse the SAME
    // frame the solid cascade + shadow draws read back (set by the main-view cull).
    vegCullCurrentSlot = solidIR->getCurrentCullFrame();

    // NO publish/swap here (flicker fix). The shadow path runs on its own
    // worker AFTER the main prepareCull in the same frame (tlCull chain) and
    // both run under the same vegetationEnabled gate, so the main poll has
    // already published and swapped before the main cull consumed the table.
    // Publishing here could swap the active instance pair BETWEEN the main
    // cull (old-offset compact commands) and the main draws, making them read
    // the new pair with stale offsets for one frame. Consolidation SUBMITS
    // are still allowed: they only fill the inactive pair.
    if (vegConsolidationDirty && !chunkBuffers.empty()) {
        consolidateChunks(appPtr);
    }

    if (vegNumChunks == 0) return;

    // DIAG (blade-shadow investigation): confirm the cascade cull runs and
    // with what chunk/slot state. Throttled: cascade culls run every frame.
    {
        static uint32_t diagCount = 0;
        if ((diagCount++ % 300u) == 0u) {
            std::cerr << "[vegshadow-diag] prepareCullCascades: vegNumChunks=" << vegNumChunks
                      << " slot=" << vegCullCurrentSlot << std::endl;
        }
    }

    if (vegNumChunks > vegCascadeCompactCapacity) {
        std::cerr << "[veg] prepareCullCascades: cascade capacity exceeded "
                  << "(need " << vegNumChunks << " cap " << vegCascadeCompactCapacity << ")\n";
        assert(false && "VegetationRenderer cascade capacity exceeded");
        return;
    }

    // Grow the GPU chunk table if the scene exceeds the current capacity.
    // This must NEVER overflow: writeVegChunkInfo and the compute dispatch
    // both index up to vegNumChunks — writing/reading past the buffer would
    // corrupt adjacent GPU-visible memory (a GPU-hang candidate). ensureChunkInfo
    // re-points BOTH the cascade and main-cull descriptor sets so a growth
    // triggered from either path keeps the other in sync.
    ensureChunkInfo(appPtr);

    static bool vegCullStatsLogged = false;
    if (!vegCullStatsLogged) {
        vegCullStatsLogged = true;
        std::cerr << "[veg] chunk stats: vegNumChunks=" << vegNumChunks
                  << " compactCap=" << vegCascadeCompactCapacity
                  << " chunkInfoCap=" << vegChunkInfoCapacity << "\n";
    }

    // Upload the GPU chunk table (AABBs + instance counts + firstInstance offsets)
    // that the veg cascade reads via binding 24 (VegCascadeChunkInfo).
    writeVegChunkInfo();

    // Bind the per-cascade billboard/impostor command + count buffers into the SOLID
    // IR descriptor set (bindings 24..36). Done every call so re-grows above are
    // always reflected. The veg cascade reads the per-veg-chunk AABB table via
    // binding 24 (= vegChunkInfoBuffer).
    std::array<std::array<VkBuffer, 3>, VEG_CULL_FRAMES> bbCompact{}, bbCount{}, impCompact{}, impCount{};
    for (uint32_t ff = 0; ff < VEG_CULL_FRAMES; ff++) {
        for (uint32_t c = 0; c < 3; c++) {
            bbCompact[ff][c] = vegCascadeCullFrames[ff].compactBuffers[c].buffer;
            bbCount[ff][c]   = vegCascadeCullFrames[ff].countBuffers[c].buffer;
            impCompact[ff][c]= vegCascadeCullFrames[ff].impostorCompactBuffers[c].buffer;
            impCount[ff][c]  = vegCascadeCullFrames[ff].impostorCountBuffers[c].buffer;
        }
    }
    solidIR->setVegCascadeData(vegChunkInfoBuffer.buffer, bbCompact, bbCount, impCompact, impCount);

    // Cascade matrices upload, output zeroing, dispatch and publish barriers all
    // live in IndirectRenderer::prepareCull (doVegCascade=true). The main-view cull
    // is NOT re-issued here (it runs in the main pass via the merged dispatch).
    // Publishing is idempotent, so poll here too: this path must also observe
    // completions when it runs without a preceding main prepareCull.
    solidIR->prepareCull(cmd, glm::mat4(1.0f), glm::vec3(0.0f), 0.0f, 0,
                         cascadeMatrices, /*doCascade=*/false, /*doMain=*/false,
                         /*doVegCascade=*/true, /*vegChunkCount=*/vegNumChunks);
}

void VegetationRenderer::drawShadowCascade(VulkanApp* app, VkCommandBuffer& commandBuffer,
                                            VkDescriptorSet shadowDescriptorSet,
                                            const glm::vec3& cameraPos,
                                            uint32_t cascadeIndex) {
    if (cascadeIndex >= 3) return;
    if (!app || vegetationShadowPipeline == VK_NULL_HANDLE) return;
    if (chunkBuffers.empty()) return;
    // H7 (report 26) as fixed for generational consistency: the provably-empty
    // gate is zero COMPLETED chunks. A set-but-dirty consolidation means newer
    // work streams, not that nothing is drawable — the cull and the draws
    // below consume the last fence-complete snapshot (vegNumChunks), exactly
    // as the main view does. Skipping on dirt hid all grass shadows (and,
    // via the same pattern, all vegetation) for the entire streaming phase.
    // Per-cascade GPU counts stay unread (no stall); the CPU-known empty
    // state is the only one elided.
    if (vegNumChunks == 0) return;
    if (shadowDescriptorSet == VK_NULL_HANDLE) return;

    // DIAG (blade-shadow investigation): confirm per-cascade shadow draws
    // execute and which sub-paths run. Throttled: 3 cascades x every frame.
    {
        static uint32_t diagCount = 0;
        const bool bladePipeOk = (vegetationShadowPipeline != VK_NULL_HANDLE);
        const bool impPipeOk = (impostorShadowPipeline != VK_NULL_HANDLE) ||
                              (impostorDepthPipeline != VK_NULL_HANDLE);
        const bool impDescOk = (impostorDepthDescSet != VK_NULL_HANDLE);
        if ((diagCount++ % 900u) == 0u) {
            std::cerr << "[vegshadow-diag] drawShadowCascade: cascade=" << cascadeIndex
                      << " vegNumChunks=" << vegNumChunks << " slot=" << vegCullCurrentSlot
                      << " bladePipe=" << bladePipeOk
                      << " impostorShadow=" << (impPipeOk && impDescOk && impostorDistance > 0.0f)
                      << " (impPipe=" << impPipeOk << " impDesc=" << impDescOk
                      << " impDist=" << impostorDistance << ")" << std::endl;
        }
    }

    uint32_t f = vegCullCurrentSlot;
    if (vegCascadeCullFrames[f].compactBuffers[cascadeIndex].buffer == VK_NULL_HANDLE) return;

    // Raw binds (not cmdState): this records into the shadow cascade command
    // buffers, but the vegetation renderer's state tracker belongs to the
    // main/task command buffer. Eliding a bind through it could skip a
    // required bind here — beginShadowPass has already bound the SOLID shadow
    // pipeline in this command buffer (VUID-vkCmdDrawIndexedIndirectCount-
    // None-08600).
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationShadowPipeline);

    updateWindParamsUBO(cameraPos);
    // Bladed grass uses sets 0 (shadow/global) and 2 (wind) only; set 1
    // (legacy billboard textures) is unbound (shader does not declare it).
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, 1, &shadowDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 2, 1, &windParamsDescSet, 0, nullptr);

    WindPushConstants pc{};
    pc.billboardScale = billboardScale;
    pc.windEnabled = -1.0f;
    pc.windTime = windTimeSeconds;
    pc.impostorDistance = impostorDistance;

    vkCmdPushConstants(commandBuffer, shadowPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(WindPushConstants), &pc);

    vkCmdBindIndexBuffer(commandBuffer, billboardVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    if (activeConcatenatedBuffer().buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
        VkBuffer vbs[3] = { billboardVBO.vertexBuffer.buffer, activeConcatenatedBuffer().buffer, activeBakedHeightsBuffer().buffer };
        VkDeviceSize offsets[3] = { 0, 0, 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 3, vbs, offsets);
        {
            uint32_t vegMaxDraws = std::min(vegNumChunks, vegCascadeCompactCapacity);
            vkCmdDrawIndexedIndirectCount(commandBuffer,
                vegCascadeCullFrames[f].compactBuffers[cascadeIndex].buffer, 0,
                vegCascadeCullFrames[f].countBuffers[cascadeIndex].buffer, 0,
                vegMaxDraws, sizeof(VkDrawIndexedIndirectCommand));
        }
    }

    // Impostor shadow pass — uses the same per-cascade visibility data from
    // the cascade compact buffer (filled by prepareCullCascades) so impostors
    // are only rendered in cascades where the chunk is visible.
    VkPipeline impostorShadowPipe = (impostorShadowPipeline != VK_NULL_HANDLE)
                                  ? impostorShadowPipeline : impostorDepthPipeline;
    VkPipelineLayout impostorShadowLayout = (impostorShadowPipelineLayout != VK_NULL_HANDLE)
                                          ? impostorShadowPipelineLayout : impostorDepthPipelineLayout;
    if (impostorShadowPipe != VK_NULL_HANDLE &&
        impostorDepthDescSet != VK_NULL_HANDLE &&
        impostorDistance > 0.0f && impostorVBO.vertexBuffer.buffer != VK_NULL_HANDLE &&
        !chunkBuffers.empty()) {
        // Raw binds: same cross-command-buffer tracker rationale as above.
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorShadowPipe);

        VkDescriptorSet depthSets[3] = { shadowDescriptorSet, impostorDepthDescSet, windParamsDescSet };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorShadowLayout, 0, 3, depthSets, 0, nullptr);

        vkCmdPushConstants(commandBuffer, impostorShadowLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(WindPushConstants), &pc);

        vkCmdBindIndexBuffer(commandBuffer, impostorVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        VkBuffer impVbs[3] = { impostorVBO.vertexBuffer.buffer, VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkDeviceSize impOffsets[3] = { 0, 0, 0 };
        if (activeConcatenatedBuffer().buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
            impVbs[1] = activeConcatenatedBuffer().buffer;
            impVbs[2] = activeBakedHeightsBuffer().buffer;
            vkCmdBindVertexBuffers(commandBuffer, 0, 3, impVbs, impOffsets);

            // Impostor draw commands (indexCount=6) are written by the GPU
            // veg_cascade_cull.comp into the per-cascade impostor compact +
            // count buffers — no CPU readback of cull results.
            Buffer& impCompactBuf = vegCascadeCullFrames[f].impostorCompactBuffers[cascadeIndex];
            Buffer& impCountBuf   = vegCascadeCullFrames[f].impostorCountBuffers[cascadeIndex];
            if (impCompactBuf.buffer != VK_NULL_HANDLE && impCountBuf.buffer != VK_NULL_HANDLE) {
                uint32_t vegMaxImpostorDraws = std::min(vegNumChunks, vegCascadeCompactCapacity);
                {
                    vkCmdDrawIndexedIndirectCount(commandBuffer,
                        impCompactBuf.buffer, 0,
                        impCountBuf.buffer, 0,
                        vegMaxImpostorDraws, sizeof(VkDrawIndexedIndirectCommand));
                }
            }
        }
    }
}

void VegetationRenderer::setTextureArrayManager(TextureArrayManager* mgr, VulkanApp* app) {
    // Unregister old listener
    if (vegetationTextureArrayManager && vegTextureListenerId != -1) {
        vegetationTextureArrayManager->removeAllocationListener(vegTextureListenerId);
        vegTextureListenerId = -1;
    }
    vegetationTextureArrayManager = mgr;
    if (!vegetationTextureArrayManager) return;
    // Try to allocate descriptor set immediately if possible
    ensureVegDescriptorSet(app);
    // Register listener to react to future reallocations
    vegTextureListenerId = vegetationTextureArrayManager->addAllocationListener([this, app]() {
        this->onTextureArraysReallocated(app);
    });
}

void VegetationRenderer::onTextureArraysReallocated(VulkanApp* app) {
    std::cerr << "[VEGETATION] onTextureArraysReallocated: refreshing vegDescriptorSet in place" << std::endl;
    if (!app) return;
    // Stable set: no free/realloc, no deferred destruction, no versioning.
    // Just rewrite the bindings into the existing set (or allocate once if
    // the set does not exist yet). Event-driven only — never in the loop.
    if (vegDescriptorSet != VK_NULL_HANDLE) {
        refreshVegDescriptors(app);
        return;
    }
    if (ensureVegDescriptorSet(app)) {
        std::cerr << "[VEGETATION] onTextureArraysReallocated: created vegDescriptorSet=" << (void*)vegDescriptorSet << std::endl;
    } else {
        std::cerr << "[VEGETATION] onTextureArraysReallocated: descriptor still not ready" << std::endl;
    }
}

void VegetationRenderer::refreshVegDescriptors(VulkanApp* app) {
    if (!app) return;
    if (vegDescriptorSet == VK_NULL_HANDLE) return;
    if (billboardAlbedoView  == VK_NULL_HANDLE ||
        billboardNormalView  == VK_NULL_HANDLE ||
        billboardOpacityView == VK_NULL_HANDLE ||
        billboardArraySampler == VK_NULL_HANDLE) return;
    // In-place rewrite of the 3 billboard bindings. Safe while the set is
    // bound by pending work (UPDATE_AFTER_BIND layout/pool); with descriptor
    // buffers this is a plain host memory write — no allocation/free.
    DescriptorWriter writer(app->getDevice());
    writer.writeImage(vegDescriptorSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      billboardArraySampler, billboardAlbedoView,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writer.writeImage(vegDescriptorSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      billboardArraySampler, billboardNormalView,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writer.writeImage(vegDescriptorSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      billboardArraySampler, billboardOpacityView,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writer.flush();
}

bool VegetationRenderer::ensureVegDescriptorSet(VulkanApp* app) {
    if (!app) return false;
    if (descriptorSetLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION] ensureVegDescriptorSet: descriptorSetLayout not created yet, deferring allocation" << std::endl;
        return false;
    }

    if (billboardAlbedoView  == VK_NULL_HANDLE ||
        billboardNormalView  == VK_NULL_HANDLE ||
        billboardOpacityView == VK_NULL_HANDLE ||
        billboardArraySampler == VK_NULL_HANDLE) return false;

    if (vegDescriptorSet == VK_NULL_HANDLE) {
        vegDescriptorSet = app->createDescriptorSet(descriptorSetLayout);

        refreshVegDescriptors(app);
        app->registerDescriptorSet(vegDescriptorSet);
        std::cerr << "[VEGETATION] Allocated vegDescriptorSet=" << (void*)vegDescriptorSet << " (3 sampler2DArray)" << std::endl;
    }
    return vegDescriptorSet != VK_NULL_HANDLE;
}


void VegetationRenderer::init(VulkanApp* app) {
    if (!app) return;
    this->appPtr = app;
    // GPU placement is the default; VULKAN_CPU_VEGETATION=1 forces the legacy
    // CPU path (RADV iGPU storage-read fallback / A-B debugging).
    if (std::getenv("VULKAN_CPU_VEGETATION")) {
        useGpuPlacement = false;
        std::cerr << "[VegetationRenderer] VULKAN_CPU_VEGETATION set — using CPU placement path\n";
    }
    VkDevice device = app->getDevice();

    DescriptorAllocator descAlloc{device, app};

    VkDescriptorSetLayoutBinding texBindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        texBindings[i].binding         = i;
        texBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texBindings[i].descriptorCount = 1;
        texBindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        texBindings[i].pImmutableSamplers = nullptr;
    }
    descriptorSetLayout = descAlloc.createLayout(
        texBindings, 3,
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        nullptr,
        "VegetationRenderer: descriptorSetLayout");

    // vkCmdDrawIndexedIndirectCount is core since Vulkan 1.2 — called
    // directly (see VulkanApp::createLogicalDevice drawIndirectCount check).

    // ── Wind params UBO + descriptor set layout (set=2) ───────────────────
    // binding 0 = wind params.
    {
        VkDescriptorSetLayoutBinding bindings[1]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        // L14 (report 26): plain layout, no UPDATE_AFTER_BIND. This set is
        // written exactly once at init (below) and never updated — the UBO
        // contents stream via memcpy to persistently-mapped memory, which
        // needs no descriptor update. Only the billboard texture set keeps
        // UPDATE_AFTER_BIND (event-driven in-place refresh while pending).
        windParamsDescSetLayout = descAlloc.createLayout(
            bindings, 1,
            0,
            nullptr,
            "VegetationRenderer: windParamsDescSetLayout");
    }


    // Allocate wind params UBO (persistently mapped host-visible).
    {
        windParamsBuffer = app->createBuffer(sizeof(WindParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        windParamsMapped = windParamsBuffer.map(0);
        // Initialize with defaults
        WindParamsUBO params{};
        std::memcpy(windParamsMapped, &params, sizeof(params));
        // M11: the change-detection cache refers to this buffer's contents;
        // a fresh buffer invalidates it (the memset above is not the cached
        // payload).
        windParamsCacheValid = false;
    }

    // Allocate wind params descriptor set and bind the UBO.
    {
        windParamsDescSet = app->createDescriptorSet(windParamsDescSetLayout);
        DescriptorWriter(device)
            .writeBuffer(windParamsDescSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         windParamsBuffer.buffer, 0, VK_WHOLE_SIZE)
            .flush();
        app->registerDescriptorSet(windParamsDescSet);
    }

    // ── Shared wind-field UBO (set=0, binding=27) ──────────────────────────
    // Owned + packed here (wind home); SceneRenderer::init binds it into the
    // static/per-frame main sets. Persistently mapped host-visible, contents
    // stream via updateWindFieldUBO() with write-on-change memcmp.
    {
        VkBufferUsageFlags wfUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (app->useDescriptorBuffer())
            wfUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        windFieldBuffer = app->createBuffer(sizeof(WindField), wfUsage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        windFieldMapped = windFieldBuffer.map(0);
        WindField zero{};
        if (windFieldMapped) std::memcpy(windFieldMapped, &zero, sizeof(zero));
        // Fresh buffer invalidates the change-detection cache (same M11
        // reasoning as windParamsCacheValid above).
        windFieldCacheValid = false;
    }

    std::vector<VkDescriptorSetLayout> setLayouts;
    setLayouts.push_back(app->getDescriptorSetLayout());
    setLayouts.push_back(descriptorSetLayout);
    setLayouts.push_back(windParamsDescSetLayout);
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(WindPushConstants);

    // Load shaders — no geometry shader
    VkShaderModule vertShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRenderer.vert.spv");
    VkShaderModule fragShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRenderer.frag.spv");
    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    VkVertexInputBindingDescription bindingDescs[3] = {};
    bindingDescs[0].binding = 0;
    bindingDescs[0].stride = sizeof(Vertex);
    bindingDescs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescs[1].binding = 1;
    bindingDescs[1].stride = static_cast<uint32_t>(kInstanceStride);
    bindingDescs[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    // Baked heights aux (perf report 22 C2/H4): one float per instance.
    bindingDescs[2].binding = 2;
    bindingDescs[2].stride = sizeof(float);
    bindingDescs[2].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // Shared attribute descriptions: localPos at POS, tangent at COLOR, UV, plane-data at BRUSH_INDEX
    std::vector<VkVertexInputAttributeDescription> attribDescs(7);
    attribDescs[0] = { ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) };
    attribDescs[1] = { ATTR_COLOR, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };
    attribDescs[2] = { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) };
    attribDescs[3] = { ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) };
    attribDescs[4] = { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    attribDescs[5] = { ATTR_VEG_AUX, 2, VK_FORMAT_R32_SFLOAT, 0 };
    // Per-instance surface normal: second vec4 of the instance payload.
    attribDescs[6] = { ATTR_VEG_NORMAL, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec4) };

    // ── Shading pass pipeline (TRIANGLE_LIST, no geometry shader) ──
    // H4 (perf report 27): blending DISABLED. Every fragment path in
    // VegetationRenderer.frag ends with outColor.a = 1.0 (including the
    // debug early-out), and the impostor twin does the same, so
    // SRC_ALPHA/ONE_MINUS_SRC_ALPHA computed src*1 + dst*0 = src: the blend
    // unit was an identity function over the whole vegetation target. With
    // blendEnable = false the pipeline returns exactly src per pixel. Depth
    // compare (LESS_OR_EQUAL, no depth write) and the H6 discard order are
    // untouched. (The impostor color pipeline needs no change: blendEnable
    // already defaults to false and its config never enables it.)
    GraphicsPipelineConfig vegCfg{};
    vegCfg.cullMode = VK_CULL_MODE_NONE;
    vegCfg.depthWriteEnable = false;
    vegCfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    vegCfg.blendEnable = false;
    vegCfg.blendSrcColorFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    vegCfg.blendDstColorFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    vegCfg.blendSrcAlphaFactor = VK_BLEND_FACTOR_ONE;
    vegCfg.blendDstAlphaFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { stages[0], stages[1] },
        std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1], bindingDescs[2]},
        attribDescs,
        setLayouts, &pushConstantRange,
        vegCfg
    );
    vegetationPipeline = pipeline;
    pipelineLayout = layout;
    if (vegetationPipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION PIPELINE ERROR] Failed to create vegetation shading pipeline or layout!" << std::endl;
    } else {
        std::cerr << "[VEGETATION PIPELINE] Created shading pipeline=" << (void*)vegetationPipeline << " layout=" << (void*)pipelineLayout << std::endl;
    }

    // ── Depth prepass pipeline (TRIANGLE_LIST, no geometry shader) ──
    {
        VkShaderModule depthVertShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRenderer.vert.spv");
        VkShaderModule depthFragShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererDepth.frag.spv");
        VkPipelineShaderStageCreateInfo depthVertStage{};
        depthVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depthVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        depthVertStage.module = depthVertShader;
        depthVertStage.pName = "main";

        VkPipelineShaderStageCreateInfo depthFragStage{};
        depthFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depthFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        depthFragStage.module = depthFragShader;
        depthFragStage.pName = "main";

        VkPipelineShaderStageCreateInfo depthStages[] = { depthVertStage, depthFragStage };

        GraphicsPipelineConfig depthCfg{};
        depthCfg.cullMode = VK_CULL_MODE_NONE;
        depthCfg.depthCompareOp = VK_COMPARE_OP_LESS;
        depthCfg.noColorAttachment = true;
        auto [depthPipe, depthLayout] = app->createGraphicsPipeline(
            { depthStages[0], depthStages[1] },
            std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1], bindingDescs[2]},
            attribDescs,
            setLayouts,
            &pushConstantRange,
            depthCfg
        );
        vegetationDepthPipeline = depthPipe;
        vegetationDepthPipelineLayout = depthLayout;

        depthVertShader = VK_NULL_HANDLE;
        depthFragShader = VK_NULL_HANDLE;
    }

    // ── EVSM shadow pipeline (writes moments via ShadowRenderer.frag ──
    {
        VkShaderModule shadowVertShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererShadow.vert.spv");
        VkShaderModule shadowFragShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererShadow.frag.spv");
        VkPipelineShaderStageCreateInfo shadowVertStage{};
        shadowVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shadowVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        shadowVertStage.module = shadowVertShader;
        shadowVertStage.pName = "main";

        VkPipelineShaderStageCreateInfo shadowFragStage{};
        shadowFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shadowFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shadowFragStage.module = shadowFragShader;
        shadowFragStage.pName = "main";

        VkPipelineShaderStageCreateInfo shadowStages[] = { shadowVertStage, shadowFragStage };

        // Shadow vertex shader (VegetationRendererShadow.vert) omits ATTR_UV (location 2)
        // — do not include it in the attribute descriptions to avoid a PERFORMANCE warning.
        // H4/C2: aux baked height IS consumed — include it.
        // ATTR_VEG_NORMAL is consumed for the normal-aligned billboard frame.
        std::vector<VkVertexInputAttributeDescription> shadowAttribDescs = {
            attribDescs[0], // ATTR_POS
            attribDescs[1], // ATTR_COLOR
            attribDescs[3], // ATTR_BRUSH_INDEX (location 4)
            attribDescs[4], // ATTR_INSTANCE   (location 5)
            attribDescs[5], // ATTR_VEG_AUX    (location 7)
            attribDescs[6], // ATTR_VEG_NORMAL (location 8)
        };
        GraphicsPipelineConfig shadowCfg{};
        shadowCfg.cullMode = VK_CULL_MODE_NONE;
        shadowCfg.depthCompareOp = VK_COMPARE_OP_LESS;
        shadowCfg.colorFormats = { VK_FORMAT_R32G32_SFLOAT };
        shadowCfg.depthBiasEnable = true;
        auto [shadowPipeline, shadowLayout] = app->createGraphicsPipeline(
            { shadowStages[0], shadowStages[1] },
            std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1], bindingDescs[2]},
            shadowAttribDescs,
            setLayouts,
            &pushConstantRange,
            shadowCfg
        );
        vegetationShadowPipeline = shadowPipeline;
        shadowPipelineLayout = shadowLayout;

        shadowVertShader = VK_NULL_HANDLE;
        shadowFragShader = VK_NULL_HANDLE;
    }
    if (vegetationShadowPipeline == VK_NULL_HANDLE || shadowPipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION SHADOW PIPELINE ERROR] Failed to create vegetation shadow pipeline/layout" << std::endl;
    } else {
        std::cerr << "[VEGETATION SHADOW PIPELINE] Created pipeline=" << (void*)vegetationShadowPipeline << " layout=" << (void*)shadowPipelineLayout << std::endl;
    }

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertShader = VK_NULL_HANDLE;
    fragShader = VK_NULL_HANDLE;
    // Build blade-tuft base mesh: 8 blades x 3 verts (left/right base + tip)
    // = 24 verts, 8 triangles = 24 indices, TRIANGLE_LIST.
    // Encoding (decoded in VegetationRenderer.vert):
    //   position = (bladeIdx, heightT, side), color = canonical blade dir,
    //   texCoord = (side*0.5, heightT), brushIndex = (bladeIdx<<8)|corner.
    if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) {
        const uint32_t BC = RendererUtils::kBladeCount;
        std::vector<Vertex> verts;
        verts.reserve(BC * 3);
        for (uint32_t b = 0; b < BC; ++b) {
            const float ang = float(b) / float(BC) * 6.28318530718f;
            const glm::vec3 dir(std::cos(ang), 0.0f, std::sin(ang));
            auto push = [&](float heightT, float side, int corner, glm::vec2 uv) {
                Vertex v;
                v.position = glm::vec3(float(b), heightT, side);
                v.color = dir;
                v.texCoord = uv;
                v.brushIndex = (int(b) << 8) | corner;
                verts.push_back(v);
            };
            push(0.0f, -1.0f, 0, glm::vec2(-0.5f, 0.0f)); // left base
            push(0.0f, +1.0f, 1, glm::vec2(+0.5f, 0.0f)); // right base
            push(1.0f,  0.0f, 2, glm::vec2(0.0f, 1.0f));  // tip
        }
        billboardVBO.vertexBuffer = app->createVertexBuffer(verts);

        std::vector<uint32_t> idx;
        idx.reserve(BC * 3);
        for (uint32_t b = 0; b < BC; ++b) {
            idx.push_back(b * 3 + 0);
            idx.push_back(b * 3 + 1);
            idx.push_back(b * 3 + 2);
        }
        billboardVBO.indexBuffer = app->createIndexBuffer(idx);
        billboardVBO.indexCount = static_cast<uint32_t>(idx.size());
    }

    // Build impostor quad mesh: 4 vertices forming a unit-square with UV corners.
    // The vertex shader scales and orients these into camera-facing billboards.
    if (impostorVBO.vertexBuffer.buffer == VK_NULL_HANDLE) {
        std::vector<Vertex> impVerts(4);
        impVerts[0] = Vertex(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec2(0.0f, 1.0f), 0); // BL
        impVerts[1] = Vertex(glm::vec3( 1.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec2(1.0f, 1.0f), 0); // BR
        impVerts[2] = Vertex(glm::vec3(-1.0f,  1.0f, 0.0f), glm::vec3(0.0f), glm::vec2(0.0f, 0.0f), 0); // TL
        impVerts[3] = Vertex(glm::vec3( 1.0f,  1.0f, 0.0f), glm::vec3(0.0f), glm::vec2(1.0f, 0.0f), 0); // TR
        impostorVBO.vertexBuffer = app->createVertexBuffer(impVerts);

        std::vector<uint32_t> impIdx = { 0, 1, 2, 1, 3, 2 };
        impostorVBO.indexBuffer = app->createIndexBuffer(impIdx);
        impostorVBO.indexCount = 6;
    }

    // Instances are generated exclusively via compute shader; no CPU uploads
    // are performed here.
}

// CPU injection removed: instances are generated by compute shader only.

void VegetationRenderer::clearAllInstances() {
    // Copy IDs first to avoid iterator invalidation from destroyInstanceBuffer erasing during iteration.
    std::vector<NodeID> ids;
    ids.reserve(chunkBuffers.size());
    for (const auto& kv : chunkBuffers) ids.push_back(kv.first);
    for (NodeID id : ids) destroyInstanceBuffer(id, appPtr);
    // destroyInstanceBuffer already clears the map entries; these are safety no-ops.
    chunkBuffers.clear();
    chunkInstanceCounts.clear();
    // Clear any pending CPU-generation chunks to prevent stale data
    // from a previous scene from being processed after scene reset.
    {
        std::lock_guard<std::mutex> lk(pendingChunksMutex);
        pendingChunks.clear();
    }
    vegConsolidationDirty = true;
    vegNumChunks = 0;
    // Abandon consolidation state with the map: in-flight submits belong to
    // the old world — pollConsolidation drops them via the epoch mismatch
    // instead of publishing stale rows. The epoch bump also invalidates any
    // submit racing this reset. Next publishes drive fresh submits.
    submittedSnapshot_.clear();
    completedSnapshot_.clear();
    vegChunkInfoDirty = true;
    ++consolEpoch_;
    // M9-(1): the chunk set moved — read barriers re-record next call.
    if (!ids.empty()) noteChunkBuffersChanged();
}

size_t VegetationRenderer::getInstanceTotal() const {
    size_t total = 0;
    for (const auto& kv : chunkInstanceCounts) {
        total += kv.second;
    }
    return total;
}

bool VegetationRenderer::hasDrawContent() const {
    // H7 (report 26) as fixed: TRUE iff the draw streams could emit anything
    // today. The ONLY exact empty condition is "no consolidated chunks"
    // (chunkBuffers empty or vegNumChunks == 0): every draw entry point
    // already no-ops on it (drawDepth/drawColor/drawShadowCascade return
    // without binds when vegNumChunks == 0), so skipping the pass leaves
    // identical cleared targets.
    // Deliberately NOT gated on streaming backlog or consolidation dirt:
    // a non-empty backlog / dirty flag means MORE content is in flight, not
    // that nothing is drawable — the last fence-complete prefix
    // (vegNumChunks, published by pollConsolidation) draws normally, exactly
    // as the pre-H7 code drew it every frame.
    // Gating on those flags hid all vegetation for the entire streaming
    // phase (thousands of chunks at 10/frame), the reported no-vegetation
    // bug. pendingChunkCount() stays for telemetry; it must not gate.
    if (chunkBuffers.empty() || vegNumChunks == 0) return false;
    return true;
}

bool VegetationRenderer::beginVegPass(bool wantDraw, bool hasContent) {
    // M12 (perf report 27): single unified veg-pass arbiter (see the header
    // contract). Steady-empty frames past warmup return false (caller skips
    // the veg task entirely — no CB, no submit, no tlVeg signal — and water
    // waits tlCull instead, H7). Every input transition runs the full path.
    // State is atomic for overlapping callers; the only race outcome is a
    // missed skip, never a missed draw: a skip requires both inputs steady
    // AND no work, and the assert below verifies worklessness on the skip
    // path (!wantDraw forces both draw flags false; !hasContent means the
    // draw streams no-op).
    const uint32_t prev = vegInstancePrev.load(std::memory_order_acquire);
    const bool transition = (prev == 2) ||
        ((prev & 1u) != (wantDraw ? 1u : 0u)) ||
        ((prev & 2u) != (hasContent ? 2u : 0u));
    const bool hasWork = wantDraw && hasContent;
    const uint32_t warm = vegInstanceWarmup.load(std::memory_order_acquire);
    const bool skip = !hasWork && !transition && warm >= kVegSkipWarmupFrames;
    // Never-miss-a-draw invariant: a skip is only possible with no work.
    assert(!skip || !hasWork);
    const uint32_t nextPrev = (wantDraw ? 1u : 0u) | (hasContent ? 2u : 0u);
    if (!skip) {
        if (warm < kVegSkipWarmupFrames) vegInstanceWarmup.fetch_add(1, std::memory_order_acq_rel);
        vegInstancePrev.store(nextPrev, std::memory_order_release);
        return true;
    }
    vegInstancePrev.store(nextPrev, std::memory_order_release);
    return false;
}

// Pool utilization telemetry (perf report 22 C1): chunk/instance counts vs
// the fixed worst-case reservation, with a <25% warning gated on non-empty.
void VegetationRenderer::logUtilization() const {
    const size_t chunks = getChunkCount();
    const size_t inst = getInstanceTotal();
    const double cPct = kMaxVegChunks > 0 ? 100.0 * double(chunks) / double(kMaxVegChunks) : 0.0;
    const double iPct = kMaxVegInstances > 0 ? 100.0 * double(inst) / double(uint64_t(kMaxVegInstances)) : 0.0;
    std::printf("[memutil] vegetation chunks %zu/%u (%.1f%%), instances %zu/%llu (%.1f%%, %.1f of 256 MB), pending chunks %zu%s\n",
        chunks, kMaxVegChunks, cPct, inst, (unsigned long long)kMaxVegInstances, iPct,
        inst * double(kInstanceStride) / 1048576.0, pendingChunkCount(),
        (inst > 0 && iPct < 25.0) ? "  <-- LOW utilization, consider a smaller tier (report 22 C1)" : "");
}

void VegetationRenderer::recordReadBarriers(VkCommandBuffer& commandBuffer) {
    if (commandBuffer == VK_NULL_HANDLE) return;

    // M9-(1) (report 26): record once per chunk-set version, not once per
    // frame. A barrier makes PRIOR writes visible to LATER reads; with no
    // intervening write the steady-state re-record is redundant (sync
    // validation agrees: read-after-read with no write needs no barrier, so
    // eliding it is validation-clean). Every chunkBuffers mutation bumps
    // chunkBufferVersion, so equality proves no publish/destroy/clear happened
    // since the last record. Store-after-record closes the race: a mutation
    // racing the record bumps the version after our load, forcing a re-record
    // next call — a barrier is never missed, at most duplicated (harmless and
    // idempotent). Callers (shadow + vegetation workers) may record
    // concurrently on change-frames; duplicates are safe.
    const uint64_t version = chunkBufferVersion.load(std::memory_order_acquire);
    if (version == readBarrierRecordedVersion.load(std::memory_order_relaxed))
        return;

    // Frame-thread scratch used to be a shared member also touched from the
    // shadow worker — a thread-local keeps concurrent change-frame records
    // from racing on the vector. Same barrier contents as before.
    thread_local std::vector<VkBufferMemoryBarrier2> scratch;
    // Reuse capacity across calls to avoid one heap allocation per record.
    scratch.clear();
    scratch.reserve(chunkBuffers.size() * 2);
    std::vector<VkBufferMemoryBarrier2>& readBarriers = scratch;
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;

        // Instance buffers are filled either by the CPU upload path (host
        // write + staging transfer) or by the GPU placement dispatch (compute
        // shader write). Without this barrier the GPU may read uninitialized
        // billboardIndex values, producing out-of-bounds texture-array accesses
        // that cause RADV GPUVM faults (TCP read).
        VkBufferMemoryBarrier2 instanceBarrier{};
        instanceBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        instanceBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        instanceBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT |
                                        VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                        VK_ACCESS_2_SHADER_WRITE_BIT;
        instanceBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        instanceBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        instanceBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        instanceBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        instanceBarrier.buffer = buf.buffer;
        instanceBarrier.offset = 0;
        instanceBarrier.size = VK_WHOLE_SIZE;
        readBarriers.push_back(instanceBarrier);
    }
    if (readBarriers.empty()) {
        readBarrierRecordedVersion.store(version, std::memory_order_release);
        return;
    }

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(readBarriers.size());
    depInfo.pBufferMemoryBarriers = readBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    readBarrierRecordedVersion.store(version, std::memory_order_release);
}


void VegetationRenderer::setImpostorData(VulkanApp* app,
                                          VkImageView albedoArray60,
                                          VkImageView normalArray60,
                                          VkSampler sampler,
                                          VkImageView depthArray60,
                                          VkBuffer captureInvVPBuf,
                                          VkFence captureFence) {
    if (!app || albedoArray60 == VK_NULL_HANDLE || normalArray60 == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
    // M11 (perf report 27): no waitForFrameFences on the sync path. The
    // synchronous capture (ImpostorService::rewire after captureAll) already
    // fence-waits internally before publish, which covers the capture data
    // this wiring samples. The old full frame-fence wait stalled an unrelated
    // pipeline for nothing: the previous descriptor sets/pools/layouts below
    // are orphaned (TrackedHandle null-assignment never frees — the central
    // manager owns teardown), never freed or reused, so no pending command
    // buffer can collide with the freshly allocated handles; descriptor
    // contents written below are identical either way. Async callers pass the
    // capture fence, which is awaited here before first sampling (capture
    // async contract: frame fences do NOT cover it).
    VkDevice device = app->getDevice();
    if (captureFence != VK_NULL_HANDLE)
        VulkanApp::waitFence(device, captureFence);

    // Destroy any previous impostor resources (handles are tracked in the central manager).
    impostorPipeline            = VK_NULL_HANDLE;
    impostorPipelineLayout      = VK_NULL_HANDLE;
    impostorDescSetLayout       = VK_NULL_HANDLE;
    impostorDescPool            = VK_NULL_HANDLE;
    impostorDescSet             = VK_NULL_HANDLE;
    impostorDepthPipeline       = VK_NULL_HANDLE;
    impostorDepthPipelineLayout = VK_NULL_HANDLE;
    impostorDepthDescSetLayout  = VK_NULL_HANDLE;
    impostorDepthDescPool       = VK_NULL_HANDLE;
    impostorDepthDescSet        = VK_NULL_HANDLE;
    impostorShadowPipeline      = VK_NULL_HANDLE;
    impostorShadowPipelineLayout = VK_NULL_HANDLE;

    bool hasImpostorDepth = (depthArray60 != VK_NULL_HANDLE && captureInvVPBuf != VK_NULL_HANDLE);

    // ── Set 1: impostor color pipeline (albedo, normal, depth + captureInvVP) ──
    // Bindings 2-3 provide depth data so the fragment shader writes gl_FragDepth.
    // Single-pass rendering: depthWrite=true, LESS compare (no separate depth prepass).
    {
        uint32_t numBindings = hasImpostorDepth ? 4u : 2u;

        VkDescriptorSetLayoutBinding bindings[4]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        if (hasImpostorDepth) {
            bindings[2].binding         = 2;
            bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[3].binding         = 3;
            bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[3].descriptorCount = 1;
            bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        // L14 (report 26): plain layout + pool, no UPDATE_AFTER_BIND. This set
        // is written once here and never updated afterwards — sampled
        // views/buffers are immutable for the set lifetime (M11, report 27:
        // the old sets are orphaned, never freed, so pending work keeps
        // valid handles). Same for the depth-variant pair below. Only the
        // billboard texture set keeps UPDATE_AFTER_BIND.
        DescriptorAllocator descAlloc{device, app};
        impostorDescSetLayout = descAlloc.createLayout(
            bindings, numBindings,
            0,
            nullptr,
            "VegetationRenderer: impostorDescSetLayout");

        {
            std::vector<VkDescriptorPoolSize> poolSz;
            poolSz.emplace_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hasImpostorDepth ? 3u : 2u});
            if (hasImpostorDepth)
                poolSz.emplace_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u});
            impostorDescPool = descAlloc.createPool(
                poolSz.data(), static_cast<uint32_t>(poolSz.size()), 1,
                0,
                "VegetationRenderer: impostorDescPool");
        }

        impostorDescSet = descAlloc.allocateSet(impostorDescPool, impostorDescSetLayout);

        DescriptorWriter writer(device);
        writer.writeImage(impostorDescSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          sampler, albedoArray60,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        writer.writeImage(impostorDescSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          sampler, normalArray60,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (hasImpostorDepth) {
            writer.writeImage(impostorDescSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              sampler, depthArray60,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writer.writeBuffer(impostorDescSet, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               captureInvVPBuf, 0, VK_WHOLE_SIZE);
        }
        writer.flush();
    }

    // ── Set 1 (depth variant): depth array + capture inv VP buffer ──────
    if (hasImpostorDepth) {
        VkDescriptorSetLayoutBinding depthBindings[2]{};
        depthBindings[0].binding         = 0;
        depthBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        depthBindings[0].descriptorCount = 1;
        depthBindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        depthBindings[1].binding         = 1;
        depthBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        depthBindings[1].descriptorCount = 1;
        depthBindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        DescriptorAllocator depthDescAlloc{device, app};
        impostorDepthDescSetLayout = depthDescAlloc.createLayout(
            depthBindings, 2,
            0,
            nullptr,
            "VegetationRenderer: impostorDepthDescSetLayout");

        VkDescriptorPoolSize depthPoolSizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
        };
        impostorDepthDescPool = depthDescAlloc.createPool(
            depthPoolSizes, 2, 1,
            0,
            "VegetationRenderer: impostorDepthDescPool");

        impostorDepthDescSet = depthDescAlloc.allocateSet(impostorDepthDescPool, impostorDepthDescSetLayout);

        DescriptorWriter(device)
            .writeImage(impostorDepthDescSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        sampler, depthArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .writeBuffer(impostorDepthDescSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         captureInvVPBuf, 0, VK_WHOLE_SIZE)
            .flush();

        // ── Build impostor depth pipeline ────────────────────────────────
        VkShaderModule depthVertMod = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererImpostorDepth.vert.spv");
        VkShaderModule depthFragMod = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererImpostorDepth.frag.spv");

        VkPipelineShaderStageCreateInfo depthStages[2]{};
        depthStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depthStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        depthStages[0].module = depthVertMod;
        depthStages[0].pName  = "main";
        depthStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depthStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        depthStages[1].module = depthFragMod;
        depthStages[1].pName  = "main";

        VkVertexInputBindingDescription depthBindingDescs[3]{};
        depthBindingDescs[0] = { 0, sizeof(Vertex),       VK_VERTEX_INPUT_RATE_VERTEX   };
        depthBindingDescs[1] = { 1, static_cast<uint32_t>(kInstanceStride), VK_VERTEX_INPUT_RATE_INSTANCE };
        // Baked heights aux (perf report 22 C2/H4): one float per instance.
        depthBindingDescs[2] = { 2, sizeof(float),        VK_VERTEX_INPUT_RATE_INSTANCE };

        std::vector<VkDescriptorSetLayout> depthSetLayouts = {
            app->getDescriptorSetLayout(),
            impostorDepthDescSetLayout,
            windParamsDescSetLayout
        };
        VkPushConstantRange depthPCRange{};
        depthPCRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        depthPCRange.offset     = 0;
        depthPCRange.size       = sizeof(WindPushConstants);

        GraphicsPipelineConfig impDepthCfg{};
        impDepthCfg.cullMode = VK_CULL_MODE_NONE;
        impDepthCfg.colorWrite = false;
        impDepthCfg.depthCompareOp = VK_COMPARE_OP_LESS;
        impDepthCfg.noColorAttachment = true;
        auto [depthPipe, depthLayout] = app->createGraphicsPipeline(
            { depthStages[0], depthStages[1] },
            std::vector<VkVertexInputBindingDescription>{ depthBindingDescs[0], depthBindingDescs[1], depthBindingDescs[2] },
            {
                { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
                { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                              },
                { ATTR_VEG_AUX, 2, VK_FORMAT_R32_SFLOAT, 0                                          },
                { ATTR_VEG_NORMAL, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec4)                  },
            },
            depthSetLayouts,
            &depthPCRange,
            impDepthCfg
        );
        impostorDepthPipeline       = depthPipe;
        impostorDepthPipelineLayout = depthLayout;

        if (impostorDepthPipeline == VK_NULL_HANDLE)
            std::cerr << "[VegetationRenderer] WARNING: impostor depth pipeline creation failed\n";
        else
            std::cerr << "[VegetationRenderer] Impostor depth pipeline created: " << (void*)impostorDepthPipeline << "\n";

        // ── Build impostor shadow EVSM pipeline (color+depth write) ──────
        // Shadow-caster vertex VARIANT (VEG_SHADOW_PASS=1): same module
        // source as the depth VS, but the quad faces the LIGHT and the
        // capture layer is the light-facing view, so the rasterized
        // footprint covers the sun and the captured-depth silhouette is
        // exact from the light's side. The depth twin keeps the camera-
        // facing orientation for the deferred depth prepass.
        VkShaderModule shadowVertMod = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererImpostorShadow.vert.spv");
        VkShaderModule shadowFragMod = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererImpostorShadow.frag.spv");
        depthStages[0].module = shadowVertMod;
        depthStages[1].module = shadowFragMod;  // reuse depthStages array, swap frag

        VkFormat evsmColorFormat = VK_FORMAT_R32G32_SFLOAT;
        GraphicsPipelineConfig impShadowCfg{};
        impShadowCfg.cullMode = VK_CULL_MODE_NONE;
        impShadowCfg.depthCompareOp = VK_COMPARE_OP_LESS;
        impShadowCfg.colorFormats = { evsmColorFormat };
        impShadowCfg.depthBiasEnable = true;
        auto [shadowPipe, shadowLayout] = app->createGraphicsPipeline(
            { depthStages[0], depthStages[1] },
            std::vector<VkVertexInputBindingDescription>{ depthBindingDescs[0], depthBindingDescs[1], depthBindingDescs[2] },
            {
                { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
                { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                              },
                { ATTR_VEG_AUX, 2, VK_FORMAT_R32_SFLOAT, 0                                          },
                { ATTR_VEG_NORMAL, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec4)                  },
            },
            depthSetLayouts,
            &depthPCRange,
            impShadowCfg
        );
        impostorShadowPipeline       = shadowPipe;
        impostorShadowPipelineLayout = shadowLayout;

        if (impostorShadowPipeline == VK_NULL_HANDLE)
            std::cerr << "[VegetationRenderer] WARNING: impostor shadow pipeline creation failed\n";
        else
            std::cerr << "[VegetationRenderer] Impostor shadow pipeline created: " << (void*)impostorShadowPipeline << "\n";

        // Shader module is cached by VulkanApp — kept alive for app lifetime.
    }

    // ── Build impostor color pipeline ───────────────────────────────────
    VkShaderModule vertShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererImpostor.vert.spv");
    VkShaderModule fragShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/VegetationRendererImpostor.frag.spv");

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName  = "main";

    VkVertexInputBindingDescription bindingDescs[3]{};
    bindingDescs[0] = { 0, sizeof(Vertex),       VK_VERTEX_INPUT_RATE_VERTEX   };
    bindingDescs[1] = { 1, static_cast<uint32_t>(kInstanceStride), VK_VERTEX_INPUT_RATE_INSTANCE };
    // Baked heights aux (perf report 22 C2/H4): one float per instance.
    bindingDescs[2] = { 2, sizeof(float),        VK_VERTEX_INPUT_RATE_INSTANCE };

    std::vector<VkDescriptorSetLayout> impSetLayouts = {
        app->getDescriptorSetLayout(),
        impostorDescSetLayout,
        windParamsDescSetLayout
    };
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(WindPushConstants);

    GraphicsPipelineConfig impCfg{};
    impCfg.cullMode = VK_CULL_MODE_NONE;
    impCfg.depthCompareOp = VK_COMPARE_OP_LESS;
    auto [impPipeline, impLayout] = app->createGraphicsPipeline(
        { vertStage, fragStage },
        std::vector<VkVertexInputBindingDescription>{ bindingDescs[0], bindingDescs[1], bindingDescs[2] },
        {
            { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
            { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                              },
            { ATTR_VEG_AUX, 2, VK_FORMAT_R32_SFLOAT, 0                                          },
            { ATTR_VEG_NORMAL, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec4)                  },
        },
        impSetLayouts,
        &pcRange,
        impCfg
    );

    impostorPipeline       = impPipeline;
    impostorPipelineLayout = impLayout;

    if (impostorPipeline == VK_NULL_HANDLE)
        std::cerr << "[VegetationRenderer] WARNING: impostor pipeline creation failed\n";
    else
        std::cerr << "[VegetationRenderer] Impostor pipeline created: " << (void*)impostorPipeline << "\n";
}

VegetationRenderer::WindPushConstants VegetationRenderer::buildWindPushConstants() const {
    WindPushConstants pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = windSettings.enabled ? 1.0f : 0.0f;
    pc.windTime           = windTimeSeconds;
    pc.impostorDistance   = impostorDistance;
    return pc;
}

void VegetationRenderer::setWindTime(float timeSeconds) {
    windTimeSeconds = timeSeconds;
    // Per-frame shared-field upload (same cadence as the adjacent
    // SkyRenderer::update call in MyApp; write-on-change inside
    // updateWindFieldUBO keeps static scenes memcpy-free).
    updateWindFieldUBO();
}

void VegetationRenderer::updateWindFieldUBO() {
    if (!windFieldMapped) return;

    WindField packed{};
    // Ambient mirrors the vegetation wind sliders at pack time (no duplicate
    // sliders; same clamps as updateWindParamsUBO).
    glm::vec2 windDir = windSettings.direction;
    const float len2 = windDir.x * windDir.x + windDir.y * windDir.y;
    if (len2 > 1e-8f) {
        const float invLen = 1.0f / std::sqrt(len2);
        windDir *= invLen;
    }
    packed.ambientA = glm::vec4(windDir.x, windDir.y,
        std::max(0.0f, windSettings.strength), std::max(0.0f, windSettings.speed));
    packed.ambientB = glm::vec4(std::max(0.00001f, windSettings.baseFrequency),
        std::max(0.00001f, windSettings.gustFrequency),
        std::max(0.0f, windSettings.gustStrength), 0.0f);
    // Tornado list: groundY is a per-tornado parameter (default sea level).
    // No runtime CPU heightfield is reachable from this updater (all
    // HeightFunction::getHeightAt implementations are build-time terrain
    // data), so GPU heightfield grounding is future work — see the header.
    uint32_t activeCount = 0;
    for (uint32_t i = 0; i < kWindFieldMaxTornadoes; ++i) {
        const TornadoSettings& s = windFieldSettings.tornadoes[i];
        WindTornado& t = packed.tornadoes[i];
        t.a = glm::vec4(s.baseXZ.x, s.baseXZ.y, s.groundY, std::max(0.0f, s.radius));
        t.b = glm::vec4(std::max(0.0f, s.height), std::max(0.0f, s.strength),
            std::clamp(s.direction, -1.0f, 1.0f), s.phase);
        t.c = glm::vec4(std::max(0.0f, s.deltaTime), s.active ? 1.0f : 0.0f,
            std::max(0.0f, s.wanderRadius), s.wanderSpeed);
        t.d = glm::vec4(s.driftVelocity.x, s.driftVelocity.y,
            std::clamp(s.swingAmplitude, 0.0f, 1.0f), s.swingFrequency);
        if (s.active) ++activeCount;
    }
    packed.counts = glm::vec4(static_cast<float>(activeCount),
        static_cast<float>(windFieldSettings.windDebugMode),
        windFieldSettings.windDebugIso, 0.0f);

    // Write-on-change (M11 pattern): bitwise compare is exact — packed is a
    // plain float vec4 aggregate with no padding (static_asserted 304 bytes).
    if (windFieldCacheValid && std::memcmp(&windFieldCache, &packed, sizeof(packed)) == 0)
        return;
    std::memcpy(windFieldMapped, &packed, sizeof(packed));
    windFieldCache = packed;
    windFieldCacheValid = true;
}

void VegetationRenderer::updateWindParamsUBO(const glm::vec3& cameraPos) {
    if (!windParamsMapped) return;

    // M9-(3) (report 26): exact input-snapshot fast path. The 80 B payload
    // below is a pure function of (camera, windSettings, densitySettings,
    // impostorDistance): when every field compares equal to the last call,
    // the rebuild + memcmp + memcpy would all be provable no-ops, so return
    // before the rebuild (the old write-on-change memcmp already collapsed
    // the 4--5 per-frame memcpys to one; this removes the redundant rebuilds
    // too). Any differing field falls through to the existing path, so the
    // shared-buffer bytes are bit-identical to before. Field-wise == only —
    // never memcmp over the settings structs (bool padding) — erring toward
    // a rebuild on NaN, never toward a missed upload.
    {
        const WindSettings& sw = windSettings;
        const DistanceDensitySettings& sd = distanceDensitySettings;
        const WindSettings& pw = windParamsSnapWind;
        const DistanceDensitySettings& pd = windParamsSnapDensity;
        if (windParamsSnapValid && cameraPos == windParamsSnapCam &&
            impostorDistance == windParamsSnapImpostor &&
            sw.enabled == pw.enabled && sw.direction == pw.direction &&
            sw.strength == pw.strength && sw.baseFrequency == pw.baseFrequency &&
            sw.speed == pw.speed && sw.gustFrequency == pw.gustFrequency &&
            sw.gustStrength == pw.gustStrength && sw.skewAmount == pw.skewAmount &&
            sw.trunkStiffness == pw.trunkStiffness && sw.noiseScale == pw.noiseScale &&
            sw.verticalFlutter == pw.verticalFlutter && sw.turbulence == pw.turbulence &&
            sd.enabled == pd.enabled && sd.fullDensityDistance == pd.fullDensityDistance &&
            sd.minDensityDistance == pd.minDensityDistance &&
            sd.minDensityFactor == pd.minDensityFactor) {
            return;
        }
    }

    WindParamsUBO params{};

    glm::vec2 windDir = windSettings.direction;
    const float len2 = windDir.x * windDir.x + windDir.y * windDir.y;
    if (len2 > 1e-8f) {
        const float invLen = 1.0f / std::sqrt(len2);
        windDir *= invLen;
    }
    params.windDirection = windDir;
    params.windStrength = std::max(0.0f, windSettings.strength);
    params.windBaseFrequency = std::max(0.00001f, windSettings.baseFrequency);
    params.windSpeed = std::max(0.0f, windSettings.speed);
    params.gustFrequency = std::max(0.00001f, windSettings.gustFrequency);
    params.gustStrength = std::max(0.0f, windSettings.gustStrength);
    params.skewAmount = std::max(0.0f, windSettings.skewAmount);
    params.trunkStiffness = std::clamp(windSettings.trunkStiffness, 0.0f, 1.0f);
    params.noiseScale = std::max(0.001f, windSettings.noiseScale);
    params.verticalFlutter = std::max(0.0f, windSettings.verticalFlutter);
    params.turbulence = std::max(0.0f, windSettings.turbulence);
    // Distance-density thinning starts AT the billboard->impostor hand-off,
    // never before it. Starting at fullDensityDistance (default 512 m) while
    // impostorDistance is larger thinned the billboards below the hand-off
    // (the vegetation visibly faded out before the impostors appeared) and
    // made the impostors start at reduced density. The far LOD still applies
    // beyond the hand-off.
    const float nearDistance = std::max({0.0f, distanceDensitySettings.fullDensityDistance, impostorDistance});
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = (distanceDensitySettings.enabled && minFactor < 1.0f)
        ? (-std::log(safeMinFactor) / (farDistance - nearDistance)) : 0.0f;
    params.densityEnabled = distanceDensitySettings.enabled ? 1u : 0u;
    params.nearDistance = nearDistance;
    params.farDistance = farDistance;
    params.minFactor = minFactor;
    params.cameraPosition = cameraPos;
    params.densityFalloff = falloff;

    // M11 (perf report 22): skip the write when the payload is unchanged
    // (write-on-change). A frame's depth pass, color pass and up to three
    // cascade draws all provide the same camera/wind state, so 4--5 mapped
    // memcpys collapse to one. Bitwise comparison is exact here: the struct
    // is a plain aggregate with no implicit padding (static_asserted).
    if (windParamsCacheValid && std::memcmp(&windParamsCache, &params, sizeof(params)) == 0) {
        // Same no-op outcome as the fast path (payload would be re-derived
        // identically next call): refresh the input snapshot so the next
        // same-input call skips the rebuild entirely.
        windParamsSnapCam = cameraPos;
        windParamsSnapWind = windSettings;
        windParamsSnapDensity = distanceDensitySettings;
        windParamsSnapImpostor = impostorDistance;
        windParamsSnapValid = true;
        return;
    }
    std::memcpy(windParamsMapped, &params, sizeof(params));
    windParamsCache = params;
    windParamsCacheValid = true;
    // Snapshot the attempted inputs (not just the uploaded payload): identical
    // inputs rebuild to identical bytes, so the next same-input call takes the
    // fast path above with provably no output change.
    windParamsSnapCam = cameraPos;
    windParamsSnapWind = windSettings;
    windParamsSnapDensity = distanceDensitySettings;
    windParamsSnapImpostor = impostorDistance;
    windParamsSnapValid = true;
}

uint32_t VegetationRenderer::vegFrame() const {
    // The MAIN-pass veg draws read the buffers written by the SOLID IndirectRenderer's
    // merged dispatch, which writes into its currentCullFrame slot. Mirror that so the
    // draw and the dispatch agree on the frame.
    return solidIR ? solidIR->getCurrentCullFrame() : vegCullCurrentSlot;
}

void VegetationRenderer::issueVegetationDraws(VkCommandBuffer cmd, VkPipelineLayout activeLayout, VkShaderStageFlags pushConstantStages, const WindPushConstants& pc) {
    uint32_t f = vegFrame();
    vkCmdPushConstants(cmd, activeLayout, pushConstantStages, 0, sizeof(WindPushConstants), &pc);
    if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE || billboardVBO.indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[3] = { billboardVBO.vertexBuffer.buffer, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceSize offsets[3] = { 0, 0, 0 };
    vkCmdBindIndexBuffer(cmd, billboardVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    // No dirty gate: the compact stream is rebuilt from the last completed
    // consolidation snapshot on every cull, so it is valid whenever chunks
    // are consolidated (vegNumChunks > 0) — including while newer work
    // streams. Gating on dirt hid all billboards for the entire streaming
    // phase (the reported no-vegetation bug).
    if (solidIR && activeConcatenatedBuffer().buffer != VK_NULL_HANDLE && vegNumChunks > 0 &&
        solidIR->getVegBbCompact(f) != VK_NULL_HANDLE && solidIR->getVegBbCount(f) != VK_NULL_HANDLE) {
        vbs[1] = activeConcatenatedBuffer().buffer;
        vbs[2] = activeBakedHeightsBuffer().buffer;
        vkCmdBindVertexBuffers(cmd, 0, 3, vbs, offsets);
        {
            vkCmdDrawIndexedIndirectCount(cmd, solidIR->getVegBbCompact(f), 0,
                solidIR->getVegBbCount(f), 0, vegNumChunks, sizeof(VkDrawIndexedIndirectCommand));
        }
    }
}

void VegetationRenderer::issueImpostorDraws(VkCommandBuffer cmd, VkPipelineLayout activeLayout, VkShaderStageFlags pushConstantStages, const WindPushConstants& pc) {
    vkCmdPushConstants(cmd, activeLayout, pushConstantStages, 0, sizeof(WindPushConstants), &pc);
    if (impostorVBO.vertexBuffer.buffer == VK_NULL_HANDLE || impostorVBO.indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[3] = { impostorVBO.vertexBuffer.buffer, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceSize offsets[3] = { 0, 0, 0 };
    vkCmdBindIndexBuffer(cmd, impostorVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    uint32_t f = vegFrame();
    // No dirty gate (see issueVegetationDraws above): the impostor compact
    // stream is valid whenever chunks are consolidated.
    if (solidIR && activeConcatenatedBuffer().buffer != VK_NULL_HANDLE && vegNumChunks > 0 &&
        solidIR->getVegImpCompact(f) != VK_NULL_HANDLE && solidIR->getVegImpCount(f) != VK_NULL_HANDLE) {
        vbs[1] = activeConcatenatedBuffer().buffer;
        vbs[2] = activeBakedHeightsBuffer().buffer;
        vkCmdBindVertexBuffers(cmd, 0, 3, vbs, offsets);
        // Indirect draw consuming the impostor stream compacted by the merged
        // GPU vegetation cull (solid IndirectRenderer's IndirectRenderer.comp dispatch).
        uint32_t vegMaxImpostorDraws = std::min(vegNumChunks, vegMainCompactCapacity);
        {
            vkCmdDrawIndexedIndirectCount(cmd,
                solidIR->getVegImpCompact(f), 0,
                solidIR->getVegImpCount(f), 0,
                vegMaxImpostorDraws, sizeof(VkDrawIndexedIndirectCommand));
        }
    }
}

void VegetationRenderer::drawDepth(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::vec3& cameraPos) {
    if (!app) return;
    if (chunkBuffers.empty()) return;
    VkDescriptorSet globalSet = app->getMainDescriptorSet();
    if (globalSet == VK_NULL_HANDLE) return;
    // H7 (report 26): issueVegetationDraws no-ops without the merged cull
    // outputs (null solidIR) or with zero consolidated chunks, so skip the
    // wind UBO touch, binds and draw. Zero draws either way: pixel-identical.
    if (!solidIR || vegNumChunks == 0) return;
    updateWindParamsUBO(cameraPos);
    WindPushConstants pc = buildWindPushConstants();

    // Depth prepass.
    if (vegetationDepthPipeline != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, vegetationDepthPipeline);
        else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationDepthPipeline);
        // Bladed grass uses sets 0 (global) and 2 (wind) only; set 1
        // (legacy billboard textures) stays unbound (undeclared in shader).
        if (cmdState) {
            cmdState->bindGraphicsDescriptorSets(commandBuffer,
                vegetationDepthPipelineLayout, 0, 1, &globalSet, 0, nullptr);
            cmdState->bindGraphicsDescriptorSets(commandBuffer,
                vegetationDepthPipelineLayout, 2, 1, &windParamsDescSet, 0, nullptr);
        } else {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vegetationDepthPipelineLayout, 0, 1, &globalSet, 0, nullptr);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vegetationDepthPipelineLayout, 2, 1, &windParamsDescSet, 0, nullptr);
        }
        issueVegetationDraws(commandBuffer, vegetationDepthPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    }

}

void VegetationRenderer::drawColor(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::vec3& cameraPos,
                                   bool drawBillboards, bool drawImpostors) {
    if (!app) return;
    if (chunkBuffers.empty()) return;
    VkDescriptorSet globalSet = app->getMainDescriptorSet();
    if (globalSet == VK_NULL_HANDLE) return;
    // H7 (report 26): both streams below no-op without the merged cull
    // outputs (null solidIR) or with zero consolidated chunks, so skip the
    // wind UBO touch, binds and draws. Per-stream flags still gate the
    // blade vs impostor binds exactly as before. Zero draws either way.
    if (!solidIR || vegNumChunks == 0) return;
    updateWindParamsUBO(cameraPos);
    WindPushConstants pc = buildWindPushConstants();

    // Shading pass (bladed tufts; set 1 legacy textures unbound).
    if (drawBillboards && vegetationPipeline != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, vegetationPipeline);
        else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationPipeline);
        if (cmdState) {
            cmdState->bindGraphicsDescriptorSets(commandBuffer,
                pipelineLayout, 0, 1, &globalSet, 0, nullptr);
            cmdState->bindGraphicsDescriptorSets(commandBuffer,
                pipelineLayout, 2, 1, &windParamsDescSet, 0, nullptr);
        } else {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &globalSet, 0, nullptr);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 2, 1, &windParamsDescSet, 0, nullptr);
        }
        issueVegetationDraws(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    }
    // Impostor color pass: captured blade tufts beyond impostorDistance.
    if (drawImpostors && impostorPipeline != VK_NULL_HANDLE &&
        impostorDescSet != VK_NULL_HANDLE && impostorDistance > 0.0f) {
        if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, impostorPipeline);
        else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorPipeline);
        VkDescriptorSet impSets[3] = { globalSet, impostorDescSet, windParamsDescSet };
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer,
                    impostorPipelineLayout, 0, 3, impSets, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    impostorPipelineLayout, 0, 3, impSets, 0, nullptr);
        issueImpostorDraws(commandBuffer, impostorPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    }
}


// ── CPU-side instance generation ─────────────────────────────────────────────
// Mirrors the previous GPU compute path (now removed) on the CPU.  Avoids GPUVM
// faults on RADV iGPUs where TCP cannot read storage buffers from any
// memory type (device-local, host-visible, or concurrent-shared).
//
// The deterministic placement math (PRNG, hashes, biome gate, slot expansion)
// lives in utils/GrassPlacement so the Blender exporter reproduces exactly
// these tufts without a device.

namespace {
// Push constants for VegetationPlacement.comp (mirrors the shader block).
struct VegPlacementPC {
    uint32_t slotCount;
    uint32_t instancesPerTriangle;
    uint32_t billboardCount;
    uint32_t seed;
    int32_t  grassBrush;
    int32_t  vertexCount;
};
} // anonymous namespace

void VegetationRenderer::generateChunkInstancesCPU(NodeID chunkId,
                                                   const std::vector<glm::vec3>& positions,
                                                   const std::vector<glm::vec3>& normals,
                                                   const std::vector<int>& brushIndices,
                                                   const std::vector<uint32_t>& grassIndices,
                                                   const glm::vec3& chunkCenter,
                                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                                   uint32_t seed) {
    (void)app; // used later in processPendingChunks
    if (grassIndices.size() < 3 || instancesPerTriangle == 0 || positions.empty()) {
        destroyInstanceBuffer(chunkId, app);
        // True delete (no replacement follows): tombstone so completed
        // generations stop drawing this chunk until the next submit rebuilds.
        tombstoneConsolidated(chunkId);
        return;
    }
    // Enqueue for later processing — the render thread drains this queue.
    PendingChunk pc;
    pc.chunkId             = chunkId;
    pc.positions           = positions;
    pc.normals             = normals;
    pc.brushIndices        = brushIndices;
    pc.grassIndices        = grassIndices;
    pc.chunkCenter         = chunkCenter;
    pc.instancesPerTriangle = instancesPerTriangle;
    pc.seed                = seed;
    {
        std::lock_guard<std::mutex> lk(pendingChunksMutex);
        pendingChunks.push_back(std::move(pc));
    }
}

size_t VegetationRenderer::pendingChunkCount() const {
    std::lock_guard<std::mutex> lk(pendingChunksMutex);
    return pendingChunks.size();
}

void VegetationRenderer::ensurePlacementPipeline(VulkanApp* app) {
    if (!app) return;
    if (vegPlacementPipeline != VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    VkDescriptorSetLayoutBinding bindings[6] = {};
    for (int b = 0; b < 6; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 6;
    dli.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &vegPlacementDescSetLayout) != VK_SUCCESS)
        throw std::runtime_error("VegetationRenderer: placement descriptor set layout");
    app->resources.addDescriptorSetLayout(vegPlacementDescSetLayout, "VegetationRenderer: vegPlacementDescSetLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(VegPlacementPC);
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &vegPlacementDescSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &vegPlacementPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("VegetationRenderer: placement pipeline layout");
    app->resources.addPipelineLayout(vegPlacementPipelineLayout, "VegetationRenderer: vegPlacementPipelineLayout");

    VkShaderModule module = app->getOrCreateShaderModule("shaders/computation/vegetation/VegetationPlacement.comp.spv");
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.layout = vegPlacementPipelineLayout;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &cpi, nullptr, &vegPlacementPipeline) != VK_SUCCESS)
        throw std::runtime_error("VegetationRenderer: placement pipeline");
    app->resources.addPipeline(vegPlacementPipeline, "VegetationRenderer: vegPlacementPipeline");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 6 * kMaxVegPlacementSets;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    dpi.maxSets = kMaxVegPlacementSets;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device, &dpi, nullptr, &vegPlacementDescPool) != VK_SUCCESS)
        throw std::runtime_error("VegetationRenderer: placement descriptor pool");
    app->resources.addDescriptorPool(vegPlacementDescPool, "VegetationRenderer: vegPlacementDescPool");

    std::cerr << "[VegetationRenderer] GPU placement pipeline ready (VegetationPlacement.comp)\n";
}

// GPU-side instance placement. Per chunk: upload the tessellated mesh
// (positions/normals/brushes) + the expanded grass-slot list into one pooled
// host-visible staging buffer, dispatch VegetationPlacement.comp writing the
// 32-byte instance records DIRECTLY into a fresh device-local instance buffer,
// and publish the chunk (count read from the GPU counter) when the dispatch
// fence signals. The instance bytes never touch the CPU.
void VegetationRenderer::processPendingChunksGPU(VulkanApp* app, uint32_t maxChunks) {
    if (!app) return;
    ensurePlacementPipeline(app);
    const uint32_t billboardCnt = (billboardCount > 0) ? billboardCount : 1u;
    constexpr int kGrassBrush = 3; // LandBrush::grass: meadow brush
    const float maxBillboardRadius = billboardScale * 2.1f;

    struct Work {
        PendingChunk mesh;              // owns positions/normals/brushes/slots
        uint32_t vertexCount = 0;
        uint32_t slotCount = 0;
        uint32_t capacity = 0;          // worst-case instance count
        glm::vec3 aabbMin{0.0f}, aabbMax{0.0f}, center{0.0f};
        VkDeviceSize posOff = 0, nrmOff = 0, brushOff = 0, slotOff = 0;
        Buffer instBuf;
        Buffer counterBuf;
        VkDescriptorSet descSet = VK_NULL_HANDLE;
    };
    std::vector<Work> work;
    work.reserve(maxChunks);

    auto align256 = [](VkDeviceSize v) { return (v + 255u) & ~VkDeviceSize(255u); };

    for (uint32_t n = 0; n < maxChunks; ++n) {
        PendingChunk pc;
        {
            std::lock_guard<std::mutex> lk(pendingChunksMutex);
            if (pendingChunks.empty()) break;
            pc = std::move(pendingChunks.front());
            pendingChunks.pop_front();
        }
        const uint32_t slotCount = static_cast<uint32_t>(pc.grassIndices.size()) / 3u;
        if (slotCount == 0 || pc.positions.empty()) {
            destroyInstanceBuffer(pc.chunkId, app);
            tombstoneConsolidated(pc.chunkId);
            continue;
        }
        Work w;
        w.vertexCount = static_cast<uint32_t>(pc.positions.size());
        w.slotCount = slotCount;
        w.capacity = slotCount * std::max(1u, pc.instancesPerTriangle);
        w.center = pc.chunkCenter;
        w.aabbMin = glm::vec3(std::numeric_limits<float>::max());
        w.aabbMax = glm::vec3(-std::numeric_limits<float>::max());
        for (const auto& pos : pc.positions) {
            w.aabbMin = glm::min(w.aabbMin, pos);
            w.aabbMax = glm::max(w.aabbMax, pos);
        }
        w.aabbMin -= glm::vec3(maxBillboardRadius);
        w.aabbMax += glm::vec3(maxBillboardRadius);
        w.mesh = std::move(pc);
        work.push_back(std::move(w));
    }
    if (work.empty()) return;

    // Staging layout: positions | normals | brushes | slots, each region
    // 256-byte aligned (minStorageBufferOffsetAlignment is at most 256).
    VkDeviceSize cursor = 0;
    for (auto& w : work) {
        w.posOff = cursor;   cursor = align256(cursor + VkDeviceSize(w.vertexCount) * 16u);
        w.nrmOff = cursor;   cursor = align256(cursor + VkDeviceSize(w.vertexCount) * 16u);
        w.brushOff = cursor; cursor = align256(cursor + VkDeviceSize(w.vertexCount) * 4u);
        w.slotOff = cursor;  cursor = align256(cursor + VkDeviceSize(w.slotCount) * 12u);
    }
    const VkDeviceSize batchTotal = cursor;

    size_t stagingIndex = 0;
    Buffer staging = acquireStagingBuffer(app, batchTotal, stagingIndex);
    {
        char* base = static_cast<char*>(staging.map(0));
        if (!base) {
            if (stagingIndex < stagingPool.size()) stagingPool[stagingIndex].inUse = false;
            return;
        }
        for (auto& w : work) {
            const uint32_t vc = w.vertexCount;
            float* p = reinterpret_cast<float*>(base + w.posOff);
            for (uint32_t i = 0; i < vc; ++i) {
                const glm::vec3& pv = w.mesh.positions[i];
                p[i * 4 + 0] = pv.x; p[i * 4 + 1] = pv.y; p[i * 4 + 2] = pv.z; p[i * 4 + 3] = 0.0f;
            }
            float* nn = reinterpret_cast<float*>(base + w.nrmOff);
            const bool hasNormals = (w.mesh.normals.size() == w.mesh.positions.size());
            for (uint32_t i = 0; i < vc; ++i) {
                const glm::vec3 nv = hasNormals ? w.mesh.normals[i] : glm::vec3(0.0f, 1.0f, 0.0f);
                nn[i * 4 + 0] = nv.x; nn[i * 4 + 1] = nv.y; nn[i * 4 + 2] = nv.z; nn[i * 4 + 3] = 0.0f;
            }
            int* br = reinterpret_cast<int*>(base + w.brushOff);
            const bool hasBrushes = (w.mesh.brushIndices.size() == w.mesh.positions.size());
            for (uint32_t i = 0; i < vc; ++i)
                br[i] = hasBrushes ? w.mesh.brushIndices[i] : kGrassBrush;
            uint32_t* sl = reinterpret_cast<uint32_t*>(base + w.slotOff);
            const uint32_t slotWords = w.slotCount * 3u;
            for (uint32_t i = 0; i < slotWords; ++i) sl[i] = w.mesh.grassIndices[i];
            // The staging bytes are the only consumer of the CPU-side mesh;
            // release the vectors now instead of retaining them until the
            // dispatch fence fires. Scalars (chunkId/seed/instancesPerTriangle)
            // stay for the deferred publish.
            std::vector<glm::vec3>().swap(w.mesh.positions);
            std::vector<glm::vec3>().swap(w.mesh.normals);
            std::vector<int>().swap(w.mesh.brushIndices);
            std::vector<uint32_t>().swap(w.mesh.grassIndices);
        }
        staging.unmap();
    }

    VkDevice device = app->getDevice();
    VkDescriptorBufferInfo infos[6];
    for (auto& w : work) {
        w.instBuf = app->createBuffer(VkDeviceSize(w.capacity) * kInstanceStride,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /*zeroInit=*/false);
        w.counterBuf = app->createBuffer(sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (w.instBuf.buffer == VK_NULL_HANDLE || w.counterBuf.buffer == VK_NULL_HANDLE ||
            w.counterBuf.mappedData == nullptr)
            continue;
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = vegPlacementDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &vegPlacementDescSetLayout;
        if (vkAllocateDescriptorSets(device, &ai, &w.descSet) != VK_SUCCESS) {
            w.descSet = VK_NULL_HANDLE;
            continue;
        }
        infos[0] = { staging.buffer, w.posOff,   VkDeviceSize(w.vertexCount) * 16u };
        infos[1] = { staging.buffer, w.nrmOff,   VkDeviceSize(w.vertexCount) * 16u };
        infos[2] = { staging.buffer, w.brushOff, VkDeviceSize(w.vertexCount) * 4u };
        infos[3] = { staging.buffer, w.slotOff,  VkDeviceSize(w.slotCount) * 12u };
        infos[4] = { w.instBuf.buffer, 0, VK_WHOLE_SIZE };
        infos[5] = { w.counterBuf.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[6] = {};
        for (int b = 0; b < 6; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = w.descSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }

    VkFence fence = app->runSingleTimeCommandsAsync([&](VkCommandBuffer cmd) {
        // Host writes -> compute reads (the staging buffer is HOST_COHERENT,
        // but make the dependency explicit and validation-friendly).
        VkMemoryBarrier hostBar{};
        hostBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hostBar.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &hostBar, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegPlacementPipeline);
        for (auto& w : work) {
            if (w.descSet == VK_NULL_HANDLE) continue;
            VegPlacementPC pc{};
            pc.slotCount = w.slotCount;
            pc.instancesPerTriangle = std::max(1u, w.mesh.instancesPerTriangle);
            pc.billboardCount = billboardCnt;
            pc.seed = w.mesh.seed;
            pc.grassBrush = kGrassBrush;
            pc.vertexCount = static_cast<int32_t>(w.vertexCount);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegPlacementPipelineLayout,
                                    0, 1, &w.descSet, 0, nullptr);
            vkCmdPushConstants(cmd, vegPlacementPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            const uint32_t groups = (w.capacity + 63u) / 64u;
            vkCmdDispatch(cmd, groups, 1, 1);
        }
        // Compute writes -> host reads (counter readback in the deferred
        // callback). One barrier covers every dispatch in this batch.
        VkMemoryBarrier outBar{};
        outBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        outBar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBar.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &outBar, 0, nullptr, 0, nullptr);
    });
    streamingUploadFence = fence;

    app->deferDestroyUntilFence(fence, [this, app, stagingIndex, work = std::move(work)]() mutable {
        VkDevice dev = app->getDevice();
        if (stagingIndex < stagingPool.size()) stagingPool[stagingIndex].inUse = false;
        for (auto& w : work) {
            uint32_t count = (w.counterBuf.mappedData != nullptr)
                                 ? *static_cast<uint32_t*>(w.counterBuf.mappedData) : 0u;
            if (count > w.capacity) count = w.capacity;
            destroyInstanceBuffer(w.mesh.chunkId, app);
            if (w.descSet != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(dev, vegPlacementDescPool, 1, &w.descSet);
                w.descSet = VK_NULL_HANDLE;
            }
            if (w.counterBuf.buffer != VK_NULL_HANDLE) {
                app->destroyBuffer(w.counterBuf);
                w.counterBuf = {};
            }
            if (count == 0 || w.instBuf.buffer == VK_NULL_HANDLE) {
                if (w.instBuf.buffer != VK_NULL_HANDLE) {
                    app->destroyBuffer(w.instBuf);
                    w.instBuf = {};
                }
                // True delete (no replacement follows): tombstone so completed
                // generations stop drawing this chunk until the next submit.
                tombstoneConsolidated(w.mesh.chunkId);
                continue;
            }
            InstanceBuffer ibuf;
            ibuf.buffer     = w.instBuf.buffer;
            ibuf.memory     = w.instBuf.memory;
            ibuf.allocation = w.instBuf.allocation;
            ibuf.center     = w.center;
            ibuf.aabbMin    = w.aabbMin;
            ibuf.aabbMax    = w.aabbMax;
            ibuf.count      = count;
            chunkBuffers[w.mesh.chunkId] = ibuf;
            chunkInstanceCounts[w.mesh.chunkId] = count;
            vegConsolidationDirty = true;
        }
        // M9-(1): the chunk set moved — read barriers re-record next call.
        noteChunkBuffersChanged();
    });
}

void VegetationRenderer::processPendingChunks(uint32_t maxChunks) {
    if (!appPtr) return;
    VulkanApp* app = appPtr;
    if (useGpuPlacement) {
        processPendingChunksGPU(app, maxChunks);
        return;
    }
    const uint32_t billboardCnt = (billboardCount > 0) ? billboardCount : 1u;

    // M12-(1) (report 26): batch build list. Per-chunk payloads are stashed
    // here so a single staging buffer can be sized once from the batch total
    // below (one acquire, one packed memcpy chain) instead of one acquire +
    // memcpy per chunk. Device bytes, AABBs and publish order are unchanged;
    // only host-side acquire traffic is gone.
    struct BuiltChunk {
        std::vector<float> bytes; // 8 floats per instance, same layout as before
        NodeID chunkId;
        uint32_t instanceCount;
        glm::vec3 aabbMin, aabbMax, center;
    };
    std::vector<BuiltChunk> built;
    built.reserve(maxChunks);

    for (uint32_t n = 0; n < maxChunks; ++n) {
        PendingChunk pc;
        {
            std::lock_guard<std::mutex> lk(pendingChunksMutex);
            if (pendingChunks.empty()) break;
            pc = std::move(pendingChunks.front());
            pendingChunks.pop_front();
        }

        const uint32_t triCount = static_cast<uint32_t>(pc.grassIndices.size()) / 3;
        const uint32_t instanceCount = triCount * pc.instancesPerTriangle;

        const float maxBillboardRadius = billboardScale * 2.1f; // max heightScale (1.4) x max blade extent (~1.5)

        // Compute AABB from vertex positions (conservatively bounds all instance anchors)
        glm::vec3 aabbMin( std::numeric_limits<float>::max());
        glm::vec3 aabbMax(-std::numeric_limits<float>::max());
        for (const auto& pos : pc.positions) {
            aabbMin = glm::min(aabbMin, pos);
            aabbMax = glm::max(aabbMax, pos);
        }
        aabbMin -= glm::vec3(maxBillboardRadius);
        aabbMax += glm::vec3(maxBillboardRadius);

        // Reuse frame-thread scratch (clear + reserve) instead of allocating
        // a fresh vector per processed chunk.
        instanceGenScratch.clear();
        instanceGenScratch.reserve(instanceCount * 8);
        std::vector<float>& validData = instanceGenScratch;

        // Deterministic placement shared with the Blender exporter
        // (utils/GrassPlacement): same PRNG/hashes/dominant-vertex test as the
        // previous inline loop, byte-identical 8-float records. The exporter
        // re-runs this per chunk without a device.
        std::vector<grass::Instance> placed;
        placed.reserve(instanceCount);
        grass::placeInstances(pc.positions, pc.normals, pc.brushIndices,
                              pc.grassIndices, billboardCnt, pc.seed, placed);
        for (const grass::Instance& inst : placed) {
            validData.push_back(inst.position.x);
            validData.push_back(inst.position.y);
            validData.push_back(inst.position.z);
            validData.push_back(float(inst.variant) + inst.rotation);
            validData.push_back(inst.normal.x);
            validData.push_back(inst.normal.y);
            validData.push_back(inst.normal.z);
            validData.push_back(0.0f);
        }

        const uint32_t validCount = static_cast<uint32_t>(validData.size() / 8);
        if (validCount == 0) {
            destroyInstanceBuffer(pc.chunkId, app);
            continue;
        }

        // Stash this chunk's bytes for the single-staging flush below (takes
        // the scratch storage; the scratch is re-established next iteration
        // by clear() + reserve(), and the `validData` reference keeps
        // aliasing the scratch object, not its storage).
        BuiltChunk bc;
        bc.bytes.swap(instanceGenScratch);
        bc.chunkId = pc.chunkId;
        bc.instanceCount = validCount;
        bc.aabbMin = aabbMin;
        bc.aabbMax = aabbMax;
        bc.center = pc.chunkCenter;
        built.push_back(std::move(bc));
    }

    // Flush: pack the whole batch into ONE staging buffer sized once from the
    // batch total, then copy each chunk at its packed offset in a single async
    // submission. The deferred callback publishes chunks when the GPU is done.
    if (!built.empty()) {
        VkDeviceSize batchTotal = 0;
        for (const auto& bc : built)
            batchTotal += VkDeviceSize(bc.bytes.size() * sizeof(float));

        // Single staging acquire from the persistent pool (perf report 22
        // M10): no VMA allocation in steady-state streaming. Returned to the
        // pool once when this batch's copy fence signals (see the deferred
        // callback below).
        size_t stagingIndex = 0;
        Buffer batchStaging = acquireStagingBuffer(app, batchTotal, stagingIndex);
        {
            void* mapped = batchStaging.map(0);
            VkDeviceSize stagingOff = 0;
            for (auto& bc : built) {
                const VkDeviceSize chunkBytes = VkDeviceSize(bc.bytes.size() * sizeof(float));
                std::memcpy(static_cast<char*>(mapped) + stagingOff,
                            bc.bytes.data(), size_t(chunkBytes));
                stagingOff += chunkBytes;
            }
            batchStaging.unmap(); // VMA persistent mapping
        }

        // Device-local instance buffers: one per chunk with byte-identical
        // contents to the old per-chunk path (same sizes, same batch order).
        // On RADV iGPUs, vertex reads go through TCP (Texture Cache/Pipe),
        // and host-visible pages lack TCP-read permission → GPUVM fault.
        // zeroInit=false: fully overwritten by the staging copy before use.
        {
            VkDeviceSize stagingOff = 0;
            for (auto& bc : built) {
                const VkDeviceSize bufSize = VkDeviceSize(bc.bytes.size() * sizeof(float));
                Buffer instBuf = app->createBuffer(bufSize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
                PendingBatchCopy e;
                e.instBuf = instBuf;
                e.bufSize = bufSize;
                e.stagingOffset = stagingOff;
                e.chunkId = bc.chunkId;
                e.instanceCount = bc.instanceCount;
                e.aabbMin = bc.aabbMin;
                e.aabbMax = bc.aabbMax;
                e.center = bc.center;
                pendingBatch.push_back(std::move(e));
                stagingOff += bufSize;
            }
        }

        std::vector<PendingBatchCopy> batch = std::move(pendingBatch);
        // Minimal copy plan for the async record (plain handles + offsets by
        // value: sources and targets outlive the submit via the fence
        // callback below, which also owns the staging lifetime through the
        // pool entry).
        struct StagedCopy { VkBuffer src, dst; VkDeviceSize srcOffset, size; };
        std::vector<StagedCopy> copies;
        copies.reserve(batch.size());
        for (auto& c : batch)
            copies.push_back({ batchStaging.buffer, c.instBuf.buffer, c.stagingOffset, c.bufSize });
        VkFence fence = app->runSingleTimeCommandsAsync([copies](VkCommandBuffer cmd) {
            for (const auto& c : copies) {
                VkBufferCopy cr{};
                cr.srcOffset = c.srcOffset;
                cr.dstOffset = 0;
                cr.size = c.size;
                vkCmdCopyBuffer(cmd, c.src, c.dst, 1, &cr);
            }
        });
        // M9-(2): latest streaming fence, so consolidateChunks can defer until
        // these copies are device-complete (polled via isFencePending, never
        // dereferenced).
        streamingUploadFence = fence;
        app->deferDestroyUntilFence(fence, [this, app, stagingIndex,
                                             batch = std::move(batch)]() mutable {
            // Single pool release for the batch-sized staging buffer (perf
            // report 22 M10): the copy fence has signaled, so the buffer is
            // idle and safe to hand to a later batch.
            if (stagingIndex < stagingPool.size())
                stagingPool[stagingIndex].inUse = false;
            for (auto& c : batch) {
                destroyInstanceBuffer(c.chunkId, app);

                InstanceBuffer ibuf;
                ibuf.buffer     = c.instBuf.buffer;
                ibuf.memory     = c.instBuf.memory;
                ibuf.allocation = c.instBuf.allocation;
                ibuf.center     = c.center;
                ibuf.aabbMin    = c.aabbMin;
                ibuf.aabbMax    = c.aabbMax;
                ibuf.count      = c.instanceCount;
                chunkBuffers[c.chunkId] = ibuf;
                chunkInstanceCounts[c.chunkId] = c.instanceCount;
                vegConsolidationDirty = true;
            }
            // M9-(1): the chunk set moved — read barriers re-record next call.
            noteChunkBuffersChanged();
        });
    }
}

Buffer VegetationRenderer::acquireStagingBuffer(VulkanApp* app, VkDeviceSize size, size_t& outIndex) {
    // Best-fit among idle entries whose capacity suffices (the staging copy
    // uses offset 0, so a larger idle buffer serves a smaller request).
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < stagingPool.size(); ++i) {
        const StagingPoolEntry& e = stagingPool[i];
        if (e.inUse || e.capacity < size) continue;
        if (best == SIZE_MAX || e.capacity < stagingPool[best].capacity) best = i;
    }
    if (best != SIZE_MAX) {
        stagingPool[best].inUse = true;
        outIndex = best;
        return stagingPool[best].buffer;
    }
    // Create a new entry sized exactly for this request. Indices are stable
    // across push_back (no erasure ever), so captured indices stay valid.
    // STORAGE_BUFFER: the GPU vegetation placement dispatch binds these host-
    // visible buffers directly as its mesh/slot inputs (no extra copy), so the
    // pooled staging buffers must be storage-bindable. Harmless for the
    // instance-copy users (pure TRANSFER_SRC read).
    Buffer b = app->createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingPool.push_back({ b, size, true });
    outIndex = stagingPool.size() - 1;
    return b;
}

void VegetationRenderer::tombstoneConsolidated(NodeID chunkId) {
    // Zero the count for every snapshot row of this chunk in both the
    // submitted (in-flight copy targets a frozen layout — offsets must not
    // move) and completed (draws must stop referencing the dropped chunk)
    // snapshots. Rows stay in place so later offsets never shift. Marks the
    // chunk-info table dirty so the cull picks up the zeros. Called ONLY for
    // true deletes (drops with no replacement in the same batch):
    // replacements keep drawing their previous bytes until the next submit
    // refreshes them (no hole, no flicker), and resets drop the snapshots
    // wholesale instead.
    bool touched = false;
    for (auto* vec : {&submittedSnapshot_, &completedSnapshot_}) {
        for (auto& e : *vec) {
            if (e.chunkId == chunkId && e.count != 0) {
                e.count = 0;
                touched = true;
            }
        }
    }
    if (touched) vegChunkInfoDirty = true;
}

void VegetationRenderer::destroyInstanceBuffer(NodeID chunkId, VulkanApp* app, VkFence completionFence) {
    auto it = chunkBuffers.find(chunkId);
    if (it == chunkBuffers.end()) return;

    // Snatch the old Vulkan handles before clearing the map entry.
    InstanceBuffer old = it->second;
    it->second.buffer = VK_NULL_HANDLE;
    it->second.memory = VK_NULL_HANDLE;
    chunkBuffers.erase(it);
    chunkInstanceCounts.erase(chunkId);
    // NOTE: no snapshot tombstone here. The batch publish path replaces the
    // same id with new data in the same callback: the completed snapshot
    // keeps referencing the OLD concatenated ranges until the next submit
    // rebuilds from the live map, so draws show the previous bytes for a
    // frame or two (no hole, no flicker) instead of blanking. The old ranges
    // stay valid: the concatenated buffer is preallocated and only the next
    // submit overwrites them, and snapshot handles are dereferenced solely
    // at submit-copy time from the rebuilt snapshot. True deletes (drops
    // with no replacement) tombstone explicitly at their call sites, and
    // clearAllInstances drops the snapshots wholesale.
    // M9-(1): the chunk set moved — read barriers re-record next call.
    noteChunkBuffersChanged();

    if (!app) return;

    // Defer destruction until the provided fence signals (or until all
    // pending work completes if VK_NULL_HANDLE). The old instance buffer
    // may still be referenced by previously-submitted render command
    // buffers on the graphics queue — waiting only on the compute/copy
    // dispatch fence is insufficient.
    app->deferDestroyUntilFence(completionFence, [app, old]() {
        Buffer tmpBuf{};
        tmpBuf.buffer = old.buffer;
        tmpBuf.memory = old.memory;
        tmpBuf.allocation = old.allocation;
        app->destroyBuffer(tmpBuf);
    });
}

// Ensure we clear the stored app pointer on cleanup
// (cleanup() already clears handles; set appPtr to nullptr here)

void VegetationRenderer::generateForChunk(VulkanApp* app, NodeID nid, const Geometry& geom) {
    if (geom.indices.size() < 3 || geom.vertices.empty()) {
        // No CPU surface geometry crosses back from the GPU any more, so the
        // runtime publishes empty chunks here. Drop any stale instance buffer
        // for the chunk (generateChunkInstancesCPU's empty-input path). The
        // existence check keeps this O(1) per publish instead of scanning the
        // consolidation snapshots for every chunk of the scene.
        if (chunkBuffers.find(nid) != chunkBuffers.end()) {
            generateChunkInstancesCPU(nid, {}, {}, {}, {}, glm::vec3(0.0f), 0u, app);
        }
        return;
    }
    try {
        // Create tightly-packed position buffer (vec3[]) for CPU-side
        // instance generation below.
        std::vector<glm::vec3> positions;
        positions.reserve(geom.vertices.size());
        for (const auto &v : geom.vertices) positions.push_back(v.position);
        // Parallel per-vertex normals: interpolated barycentrically per instance
        // so blade tufts tilt onto the smooth surface normal.
        std::vector<glm::vec3> normals;
        normals.reserve(geom.vertices.size());
        for (const auto &v : geom.vertices) normals.push_back(v.normal);
        // Parallel per-vertex brush indices: the per-instance dominant-vertex
        // test in processPendingChunks keeps samples near grass verts only.
        std::vector<int> brushes;
        brushes.reserve(geom.vertices.size());
        for (const auto &v : geom.vertices) brushes.push_back(v.brushIndex);

        // Area-weighted virtual slots with unbiased stochastic rounding and a
        // per-chunk shuffle; shared with the Blender exporter so the offline
        // tufts match the rendered ones (utils/GrassPlacement).
        std::vector<uint32_t> grassIndices;
        grassIndices.reserve(geom.indices.size());
        grass::collectSlots(geom, static_cast<uintptr_t>(nid), grassIndices);

        // Each virtual triangle slot produces exactly 1 instance.
        uint32_t instancesPerTriangle = 1u;
        uint32_t seed = static_cast<uint32_t>(nid & 0xffffffffull);
        glm::vec3 chunkCenter(0.0f);
        for (const auto& position : positions) {
            chunkCenter += position;
        }
        if (!positions.empty()) {
            chunkCenter /= static_cast<float>(positions.size());
        }
        if (grassIndices.size() < 3) {
            // No grass triangles in this chunk; ensure old chunk vegetation is cleared.
            if (std::getenv("VULKAN_DISABLE_VEGETATION")) {
                return;
            }
            // CPU path handles the empty case (clears any previous chunk data).
            generateChunkInstancesCPU(nid, positions, normals, brushes, grassIndices,
                chunkCenter, instancesPerTriangle, app, seed);
            return;
        }

        // CPU-side instance generation — avoids RADV GPUVM faults where
        // the Texture Cache/Pipe cannot read storage buffers on iGPUs.
        generateChunkInstancesCPU(nid, positions, normals, brushes, grassIndices,
            chunkCenter, instancesPerTriangle, app, seed);
    } catch (const std::exception &e) {
        std::cerr << "[VegetationRenderer] Vegetation generation failed for node " << (unsigned long long)nid
                  << ": " << e.what() << std::endl;
    }
}

// ── Own offscreen framebuffer (decoupled from the solid pass) ──────────────────
// Vegetation is rendered to its own color+depth images so it can be drawn on a
// parallel async command buffer. The solid pass no longer shares its depth with
// vegetation; occlusion against solid geometry is resolved at composite time
// (PostProcessRenderer.frag) by testing the vegetation depth against the solid scene
// depth — mirroring how the water pass was decoupled.
void VegetationRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (vegRenderWidth == width && vegRenderHeight == height && vegColorImages[0] != VK_NULL_HANDLE) {
        return; // Already created at this size
    }

    destroyRenderTargets(app);

    vegRenderWidth = width;
    vegRenderHeight = height;

    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "VegetationRenderer: offscreen", image, allocation, memory, view);
    };

    for (uint32_t i = 0; i < VEG_FRAMES; ++i) {
        createImage(app->getSwapchainImageFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    vegColorImages[i], vegColorAllocations[i], vegColorMemories[i], vegColorImageViews[i]);
        // Transition directly to final layout (SHADER_READ_ONLY for post-process sampling)
        if (vegColorImages[i] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(vegColorImages[i], app->getSwapchainImageFormat(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(vegColorImages[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            vegColorImageLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    vegDepthImages[i], vegDepthAllocations[i], vegDepthMemories[i], vegDepthImageViews[i]);
        // Created in DEPTH_STENCIL_ATTACHMENT_OPTIMAL (the veg task renders depth into it)
        if (vegDepthImages[i] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(vegDepthImages[i], VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(vegDepthImages[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            vegDepthImageLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }

    std::cout << "[VegetationRenderer] Created offscreen render targets " << width << "x" << height << std::endl;
}

void VegetationRenderer::destroyRenderTargets(VulkanApp* app) {
    // Clear per-frame image handles; actual Vulkan destruction is performed by
    // the VulkanResourceManager (mirrors WaterRenderer::destroyRenderTargets).
    (void)app;
    for (uint32_t i = 0; i < VEG_FRAMES; ++i) {
        vegColorImages[i] = VK_NULL_HANDLE;
        vegColorAllocations[i] = VK_NULL_HANDLE;
        vegColorMemories[i] = VK_NULL_HANDLE;
        vegColorImageViews[i] = VK_NULL_HANDLE;
        vegDepthImages[i] = VK_NULL_HANDLE;
        vegDepthAllocations[i] = VK_NULL_HANDLE;
        vegDepthMemories[i] = VK_NULL_HANDLE;
        vegDepthImageViews[i] = VK_NULL_HANDLE;
        vegColorImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        vegDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}
