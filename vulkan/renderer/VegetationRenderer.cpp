#include "VegetationRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstddef>
#include "../../math/Common.hpp" // for NodeID
#include "VegetationRenderer.hpp"
#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

// GPU injection APIs removed. Instances must be generated via compute shader.


VegetationRenderer::VegetationRenderer() {}
VegetationRenderer::~VegetationRenderer() { /* caller must call cleanup(app) */ }

void VegetationRenderer::init() {

}


void VegetationRenderer::cleanup(VulkanApp* app) {
    (void)app;
    std::vector<NodeID> idsToDestroy;
    idsToDestroy.reserve(chunkBuffers.size());
    for (const auto& [id, _] : chunkBuffers) idsToDestroy.push_back(id);
    for (NodeID id : idsToDestroy) destroyInstanceBuffer(id);
    chunkBuffers.clear();
    chunkInstanceCounts.clear();
    vegDescriptorVersion = 0;
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
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f) {
        if (compactedCmdBuffers[f].buffer != VK_NULL_HANDLE) {
            compactedCmdMapped[f] = nullptr;
            appPtr->destroyBuffer(compactedCmdBuffers[f]);
            compactedCmdBuffers[f] = {};
        }
        if (visibleCountBuffers[f].buffer != VK_NULL_HANDLE) {
            visibleCountMapped[f] = nullptr;
            appPtr->destroyBuffer(visibleCountBuffers[f]);
            visibleCountBuffers[f] = {};
        }
    }
    if (consolidationFence != VK_NULL_HANDLE) {
        VulkanApp::waitFence(device, consolidationFence);
    }
    consolidationPending = false;
    vegNumChunks = 0;
    vegConsolidationDirty = true;
}

void VegetationRenderer::consolidateChunks(VulkanApp* app) {
    if (chunkBuffers.empty()) return;
    if (!app) return;
    auto device = app->getDevice();

    // ── Phase 1: Check if previous consolidation is still pending ──
    //    The deferred callback clears consolidationPending when the fence signals.
    if (consolidationPending) {
        vegConsolidationDirty = true;
        return;
    }

    // ── Phase 2: Start a new consolidation ──

    size_t totalInstances = 0;
    for (const auto& kv : chunkInstanceCounts)
        totalInstances += kv.second;
    if (totalInstances == 0) return;
    concatenatedInstanceCount_ = totalInstances;
    ++vegetationGeneration_;

    uint32_t numChunks = static_cast<uint32_t>(chunkBuffers.size());

    VkDeviceSize concatSize = totalInstances * sizeof(glm::vec4);
    {
        bool needsCreate = concatenatedInstanceBuffer.buffer == VK_NULL_HANDLE;
        if (!needsCreate) {
            VkMemoryRequirements reqs;
            vkGetBufferMemoryRequirements(device, concatenatedInstanceBuffer.buffer, &reqs);
            if (reqs.size < concatSize) needsCreate = true;
        }
        if (needsCreate) {
            Buffer old = concatenatedInstanceBuffer;
            concatenatedInstanceBuffer = {};
            concatenatedInstanceBuffer = app->createBuffer(concatSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (old.buffer != VK_NULL_HANDLE) {
                app->deferDestroyUntilAllPending([app, old]() {
                    if (old.buffer != VK_NULL_HANDLE)
                        app->resources.removeBufferVma(old.buffer, old.allocation);
                });
            }
        }
    }

    VkDeviceSize compactedSize = std::max(256u, numChunks) * sizeof(VkDrawIndexedIndirectCommand);
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f) {
        bool needsCreate = compactedCmdBuffers[f].buffer == VK_NULL_HANDLE;
        if (!needsCreate) {
            VkMemoryRequirements reqs;
            vkGetBufferMemoryRequirements(device, compactedCmdBuffers[f].buffer, &reqs);
            if (reqs.size < compactedSize) needsCreate = true;
        }
        if (needsCreate) {
            compactedCmdMapped[f] = nullptr;
            Buffer old = compactedCmdBuffers[f];
            compactedCmdBuffers[f] = {};
            compactedCmdBuffers[f] = app->createBuffer(compactedSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            compactedCmdMapped[f] = static_cast<VkDrawIndexedIndirectCommand*>(compactedCmdBuffers[f].map(0));
            if (old.buffer != VK_NULL_HANDLE) {
                app->deferDestroyUntilAllPending([app, old]() {
                    if (old.buffer != VK_NULL_HANDLE)
                        app->resources.removeBufferVma(old.buffer, old.allocation);
                });
            }
        }
        if (visibleCountBuffers[f].buffer == VK_NULL_HANDLE) {
            visibleCountBuffers[f] = app->createBuffer(sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            visibleCountMapped[f] = static_cast<uint32_t*>(visibleCountBuffers[f].map(0));
            *visibleCountMapped[f] = 0;
        }
    }

    // Count chunks with valid geometry (GPU metadata buffer removed — the CPU
    // culling paths write draw commands directly).
    uint32_t chunkIdx = 0;
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
        chunkIdx++;
    }
    vegNumChunks = chunkIdx;

    if (vegNumChunks == 0) return;

    // Copy per-chunk instance data into the concatenated buffer (GPU→GPU)
    if (concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            // WRITE-AFTER-READ/WRITE guard: the concatenated instance buffer is
            // read as vertex attributes by the previous frame's vegetation
            // draws, and its per-chunk sources were written by earlier transfer
            // submits. ALL_COMMANDS srcStage covers the full draw-pipeline span
            // sync validation attributes vertex-attribute reads to
            // (SYNC-HAZARD-WRITE-AFTER-READ reported on GPU-assisted runs).
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            mb.srcAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                               VK_ACCESS_2_TRANSFER_READ_BIT |
                               VK_ACCESS_2_TRANSFER_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cmd, &dep);

            uint32_t off = 0;
            for (const auto& [cid, buf] : chunkBuffers) {
                (void)cid;
                if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
                VkBufferCopy region{};
                region.srcOffset = 0;
                region.dstOffset = off * sizeof(glm::vec4);
                region.size = buf.count * sizeof(glm::vec4);
                vkCmdCopyBuffer(cmd, buf.buffer, concatenatedInstanceBuffer.buffer, 1, &region);
                off += buf.count;
            }
        });
    }

    vegConsolidationDirty = false;
}

void VegetationRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj) {
    vegCullCurrentSlot = vegCullFrameIndex % VEG_CULL_FRAMES;
    vegCullFrameIndex++;
    uint32_t f = vegCullCurrentSlot;
    // Consolidate if chunks have changed since last consolidation
    if (vegConsolidationDirty && !chunkBuffers.empty()) {
        consolidateChunks(appPtr);
        if (vegConsolidationDirty) return; // fence still in flight, skip cull
    }
    if (vegNumChunks == 0) return;
    if (compactedCmdBuffers[f].buffer == VK_NULL_HANDLE || visibleCountBuffers[f].buffer == VK_NULL_HANDLE) return;

    // Upload every chunk as a visible draw command directly into the
    // persistently mapped host-visible buffer (avoids vkCmdUpdateBuffer's
    // implicit full queue barrier that stalls the entire graphics pipeline).
    // Written in place, bounded by vegNumChunks (<= buffer capacity): a
    // previous fixed-size stack array overflowed once vegetation exceeded 256
    // chunks, corrupting the caller's stack frame (GPU hang on RADV / 680M).
    uint32_t count = 0;
    {
        VkDrawIndexedIndirectCommand* dst =
            static_cast<VkDrawIndexedIndirectCommand*>(compactedCmdMapped[f]);
        uint32_t instanceOff = 0;
        for (auto& [cid, buf] : chunkBuffers) {
            (void)cid;
            if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
            dst[count].indexCount    = 36;
            dst[count].instanceCount = buf.count;
            dst[count].firstIndex    = 0;
            dst[count].vertexOffset  = 0;
            dst[count].firstInstance = instanceOff;
            instanceOff += buf.count;
            count++;
            if (count >= vegNumChunks) break;
        }
    }

    // Write visible count to persistently mapped host-coherent buffer
    if (visibleCountMapped[f]) {
        *visibleCountMapped[f] = count;
    }
    // No barrier needed: host-coherent writes complete before vkQueueSubmit,
    // so the GPU sees the latest data when it processes the indirect draw.
}

// ── Cascade-aware culling for vegetation shadows ──────────────────────────────

void VegetationRenderer::initCascadeCull(VulkanApp* app) {
    if (vegCascadeCullInited) return;
    vegCascadeCullInited = true;
    VkDevice device = app->getDevice();

    // Shared GPU-side chunk table: one vec4 triple per chunk
    // ({aabbMin.xyz|pad, aabbMax.xyz|pad, instanceCount|firstInstance|pad|pad}).
    // Initial generous capacity (4096 chunks ≈ 192 KB); grown dynamically in
    // prepareCullCascades when the scene exceeds it, with descriptor re-points.
    constexpr uint32_t kChunkInfoCap = 4096;
    vegChunkInfoCapacity = kChunkInfoCap;
    vegChunkInfoBuffer = app->createBuffer(sizeof(glm::vec4) * 3 * kChunkInfoCap,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vegChunkInfoMapped = vegChunkInfoBuffer.map(0);

    // Cascade matrices storage buffer (3 mat4 = 192 bytes), host-written each frame.
    vegCascadeMatrixBuffer = app->createBuffer(sizeof(glm::mat4) * 3,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Descriptor set layout: 14 bindings
    //  0: chunk info (read)
    //  1..6:  billboard compact + count, per cascade
    //  7..12: impostor compact + count, per cascade
    // 13: cascade matrices
    std::array<VkDescriptorSetLayoutBinding, 14> bindings{};
    VkDescriptorBindingFlags bindingFlags[14];
    for (uint32_t i = 0; i < 14; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindingFlags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    }

    DescriptorAllocator descAlloc{device, app};
    vegCascadeCullDescSetLayout = descAlloc.createLayout(
        bindings.data(), 14,
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        bindingFlags,
        "VegetationRenderer: vegCascadeCullDescSetLayout");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(uint32_t); // numChunks

    VkPipelineLayoutCreateInfo plinfo{};
    plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plinfo.setLayoutCount = 1;
    plinfo.pSetLayouts = &vegCascadeCullDescSetLayout;
    plinfo.pushConstantRangeCount = 1;
    plinfo.pPushConstantRanges = &pc;

    if (vkCreatePipelineLayout(device, &plinfo, nullptr, &vegCascadeCullPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create veg cascade cull pipeline layout!");
    app->resources.addPipelineLayout(vegCascadeCullPipelineLayout, "VegetationRenderer: vegCascadeCullPipelineLayout");

    VkShaderModule compModule = app->getOrCreateShaderModule("shaders/veg_cascade_cull.comp.spv");
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = vegCascadeCullPipelineLayout;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &pipelineInfo, nullptr, &vegCascadeCullPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create veg cascade cull compute pipeline!");
    app->resources.addPipeline(vegCascadeCullPipeline, "VegetationRenderer: vegCascadeCullPipeline");

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
    vegCascadeCullDescPool = descAlloc.createPool(
        &poolSize, 1, VEG_CULL_FRAMES,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "VegetationRenderer: vegCascadeCullDescPool");

    VkDescriptorSet rawSets[VEG_CULL_FRAMES];
    descAlloc.allocateSets(vegCascadeCullDescPool, vegCascadeCullDescSetLayout,
                           VEG_CULL_FRAMES, rawSets,
                           "VegetationRenderer: vegCascadeCullDescSet");
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; f++) {
        vegCascadeCullFrames[f].descSet = rawSets[f];
    }

    // Per-frame cascade buffers (sized conservatively). The GPU cascade-cull
    // compute pipeline writes compact commands + counts via atomics; the
    // shadow draws consume them through vkCmdDrawIndexedIndirectCount.
    VkDeviceSize compactSize = sizeof(VkDrawIndexedIndirectCommand) * 1024;
    vegCascadeCompactCapacity = 1024;
    VkDeviceSize countSize = sizeof(uint32_t);
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; f++) {
        for (uint32_t c = 0; c < 3; c++) {
            vegCascadeCullFrames[f].compactBuffers[c] = app->createBuffer(compactSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* data = vegCascadeCullFrames[f].compactBuffers[c].map(0);
            if (data) {
                std::memset(data, 0, (size_t)compactSize);
                vegCascadeCullFrames[f].compactBuffers[c].unmap();
            }
            vegCascadeCullFrames[f].countBuffers[c] = app->createBuffer(countSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            uint32_t* mapped = static_cast<uint32_t*>(vegCascadeCullFrames[f].countBuffers[c].map(0));
            if (mapped) *mapped = 0;

            vegCascadeCullFrames[f].impostorCompactBuffers[c] = app->createBuffer(compactSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* impData = vegCascadeCullFrames[f].impostorCompactBuffers[c].map(0);
            if (impData) {
                std::memset(impData, 0, (size_t)compactSize);
                vegCascadeCullFrames[f].impostorCompactBuffers[c].unmap();
            }
            vegCascadeCullFrames[f].impostorCountBuffers[c] = app->createBuffer(countSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            uint32_t* impMapped = static_cast<uint32_t*>(vegCascadeCullFrames[f].impostorCountBuffers[c].map(0));
            if (impMapped) *impMapped = 0;
        }
        updateVegCascadeDescriptor(app, f);
    }
}

void VegetationRenderer::updateVegCascadeDescriptor(VulkanApp* app, uint32_t frame) {
    VkDescriptorSet ds = vegCascadeCullFrames[frame].descSet;
    if (ds == VK_NULL_HANDLE || vegChunkInfoBuffer.buffer == VK_NULL_HANDLE) return;

    DescriptorWriter writer(app->getDevice());
    writer.writeBuffer(ds, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       vegChunkInfoBuffer.buffer, 0, VK_WHOLE_SIZE);
    for (uint32_t c = 0; c < 3; c++) {
        static const uint32_t bbOutBindings[3] = {1, 3, 5};
        static const uint32_t bbCntBindings[3] = {2, 4, 6};
        static const uint32_t impOutBindings[3] = {7, 9, 11};
        static const uint32_t impCntBindings[3] = {8, 10, 12};
        writer.writeBuffer(ds, bbOutBindings[c], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegCascadeCullFrames[frame].compactBuffers[c].buffer, 0, VK_WHOLE_SIZE);
        writer.writeBuffer(ds, bbCntBindings[c], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegCascadeCullFrames[frame].countBuffers[c].buffer, 0, VK_WHOLE_SIZE);
        writer.writeBuffer(ds, impOutBindings[c], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegCascadeCullFrames[frame].impostorCompactBuffers[c].buffer, 0, VK_WHOLE_SIZE);
        writer.writeBuffer(ds, impCntBindings[c], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegCascadeCullFrames[frame].impostorCountBuffers[c].buffer, 0, VK_WHOLE_SIZE);
    }
    writer.writeBuffer(ds, 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       vegCascadeMatrixBuffer.buffer, 0, VK_WHOLE_SIZE);
    writer.flush();
}

void VegetationRenderer::writeVegChunkInfo() {
    if (!vegChunkInfoMapped) return;
    glm::vec4* dst = static_cast<glm::vec4*>(vegChunkInfoMapped);
    uint32_t idx = 0;
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
        if (idx >= vegChunkInfoCapacity) {
            // Unreachable (prepareCullCascades grows the table first) — guard
            // against future call sites writing past the buffer.
            std::cerr << "[veg] FATAL: writeVegChunkInfo overflow cap=" << vegChunkInfoCapacity << "\n";
            break;
        }
        dst[idx * 3 + 0] = glm::vec4(buf.aabbMin, 0.0f);
        dst[idx * 3 + 1] = glm::vec4(buf.aabbMax, 0.0f);
        dst[idx * 3 + 2] = glm::vec4(static_cast<float>(buf.count), 0.0f, 0.0f, 0.0f);
        ++idx;
    }
    // firstInstance per chunk = cumulative instance count (matches the
    // concatenated instance buffer order built in consolidateChunks).
    uint32_t instOff = 0;
    for (uint32_t i = 0; i < idx; ++i) {
        dst[i * 3 + 2].y = static_cast<float>(instOff);
        instOff += static_cast<uint32_t>(dst[i * 3 + 2].x);
    }
}

void VegetationRenderer::prepareCullCascades(VkCommandBuffer cmd,
                                              const glm::mat4 cascadeMatrices[3]) {
    if (!appPtr) return;
    if (!vegCascadeCullInited) initCascadeCull(appPtr);

    vegCullCurrentSlot = vegCullFrameIndex % VEG_CULL_FRAMES;
    vegCullFrameIndex++;
    uint32_t f = vegCullCurrentSlot;

    if (vegConsolidationDirty && !chunkBuffers.empty()) {
        consolidateChunks(appPtr);
        if (vegConsolidationDirty) return;
    }

    if (vegNumChunks == 0) return;

    if (vegNumChunks > vegCascadeCompactCapacity) {
        uint32_t newCap = vegNumChunks;
        VkDeviceSize newCompactSize = sizeof(VkDrawIndexedIndirectCommand) * newCap;
        VulkanApp* app = appPtr;
        for (uint32_t ff = 0; ff < VEG_CULL_FRAMES; ff++) {
            for (uint32_t cc = 0; cc < 3; cc++) {
                Buffer oldBuf = vegCascadeCullFrames[ff].compactBuffers[cc];
                vegCascadeCullFrames[ff].compactBuffers[cc] = app->createBuffer(newCompactSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                void* data = vegCascadeCullFrames[ff].compactBuffers[cc].map(0);
                if (data) {
                    std::memset(data, 0, (size_t)newCompactSize);
                    vegCascadeCullFrames[ff].compactBuffers[cc].unmap();
                }
                if (oldBuf.buffer != VK_NULL_HANDLE) {
                    app->deferDestroyUntilAllPending([app, oldBuf]() {
                        if (oldBuf.buffer != VK_NULL_HANDLE)
                            app->resources.removeBufferVma(oldBuf.buffer, oldBuf.allocation);
                    });
                }

                Buffer oldImpBuf = vegCascadeCullFrames[ff].impostorCompactBuffers[cc];
                vegCascadeCullFrames[ff].impostorCompactBuffers[cc] = app->createBuffer(newCompactSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                void* impData = vegCascadeCullFrames[ff].impostorCompactBuffers[cc].map(0);
                if (impData) {
                    std::memset(impData, 0, (size_t)newCompactSize);
                    vegCascadeCullFrames[ff].impostorCompactBuffers[cc].unmap();
                }
                if (oldImpBuf.buffer != VK_NULL_HANDLE) {
                    app->deferDestroyUntilAllPending([app, oldImpBuf]() {
                        if (oldImpBuf.buffer != VK_NULL_HANDLE)
                            app->resources.removeBufferVma(oldImpBuf.buffer, oldImpBuf.allocation);
                    });
                }
            }
        }
        vegCascadeCompactCapacity = newCap;
    }

    // Grow the GPU chunk table if the scene exceeds the current capacity.
    // This must NEVER overflow: writeVegChunkInfo and the compute dispatch
    // both index up to vegNumChunks — writing/reading past the buffer would
    // corrupt adjacent GPU-visible memory (a GPU-hang candidate).
    if (vegNumChunks > vegChunkInfoCapacity) {
        VkDeviceSize newInfoSize = sizeof(glm::vec4) * 3 * vegNumChunks;
        VulkanApp* app = appPtr;
        Buffer oldInfo = vegChunkInfoBuffer;
        vegChunkInfoBuffer = app->createBuffer(newInfoSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vegChunkInfoMapped = vegChunkInfoBuffer.map(0);
        if (oldInfo.buffer != VK_NULL_HANDLE) {
            app->deferDestroyUntilAllPending([app, oldInfo]() {
                if (oldInfo.buffer != VK_NULL_HANDLE)
                    app->resources.removeBufferVma(oldInfo.buffer, oldInfo.allocation);
            });
        }
        vegChunkInfoCapacity = vegNumChunks;
        // Binding 0 (chunk info) changed for every frame's descriptor set.
        for (uint32_t ff = 0; ff < VEG_CULL_FRAMES; ff++)
            updateVegCascadeDescriptor(appPtr, ff);
        std::cerr << "[veg] chunk-info table grew to " << vegNumChunks << " chunks (" << newInfoSize << " bytes)\n";
    }

    static bool vegCullStatsLogged = false;
    if (!vegCullStatsLogged) {
        vegCullStatsLogged = true;
        std::cerr << "[veg] chunk stats: vegNumChunks=" << vegNumChunks
                  << " compactCap=" << vegCascadeCompactCapacity
                  << " chunkInfoCap=" << vegChunkInfoCapacity << "\n";
    }

    // Re-point descriptors if any cascade buffers were recreated above.
    updateVegCascadeDescriptor(appPtr, f);

    // Upload cascade matrices to the host-visible storage buffer. Writes
    // complete before vkQueueSubmit, so no host→device barrier is required.
    {
        void* matData = vegCascadeMatrixBuffer.map(0);
        if (matData) {
            std::memcpy(matData, cascadeMatrices, sizeof(glm::mat4) * 3);
            vegCascadeMatrixBuffer.unmap();
        }
    }

    // Upload the GPU chunk table (AABBs + instance counts + firstInstance
    // offsets) into the host-visible chunk-info buffer.
    writeVegChunkInfo();

    // Barrier: drain prior draws/compute before zeroing the 12 cascade
    // output buffers (compact/count for billboards and impostors × 3 cascades).
    {
        VkBufferMemoryBarrier2 preFill[12]{};
        for (uint32_t i = 0; i < 12; i++) {
            preFill[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            preFill[i].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                      | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preFill[i].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                                       | VK_ACCESS_2_SHADER_READ_BIT
                                       | VK_ACCESS_2_SHADER_WRITE_BIT
                                       | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preFill[i].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preFill[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preFill[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preFill[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preFill[i].offset = 0;
            preFill[i].size = VK_WHOLE_SIZE;
        }
        for (uint32_t c = 0; c < 3; c++) {
            preFill[c * 2 + 0].buffer = vegCascadeCullFrames[f].compactBuffers[c].buffer;
            preFill[c * 2 + 1].buffer = vegCascadeCullFrames[f].countBuffers[c].buffer;
            preFill[6 + c * 2 + 0].buffer = vegCascadeCullFrames[f].impostorCompactBuffers[c].buffer;
            preFill[6 + c * 2 + 1].buffer = vegCascadeCullFrames[f].impostorCountBuffers[c].buffer;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 12;
        depInfo.pBufferMemoryBarriers = preFill;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Zero all compact + count buffers so GPU atomics start from a clean state.
    for (uint32_t c = 0; c < 3; c++) {
        vkCmdFillBuffer(cmd, vegCascadeCullFrames[f].compactBuffers[c].buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, vegCascadeCullFrames[f].countBuffers[c].buffer, 0, sizeof(uint32_t), 0);
        vkCmdFillBuffer(cmd, vegCascadeCullFrames[f].impostorCompactBuffers[c].buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, vegCascadeCullFrames[f].impostorCountBuffers[c].buffer, 0, sizeof(uint32_t), 0);
    }

    // Barrier: fill → compute + indirect draw. The compute shader reads the
    // counts (atomicAdd) and writes compact, so SHADER_READ|SHADER_WRITE at
    // COMPUTE stage is required; DRAW_INDIRECT access covers the billboard
    // draw that reads the count buffer even when the dispatch is skipped.
    {
        VkBufferMemoryBarrier2 barriers[12]{};
        for (uint32_t i = 0; i < 12; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barriers[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                    | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT
                                     | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].offset = 0;
            barriers[i].size = VK_WHOLE_SIZE;
        }
        for (uint32_t c = 0; c < 3; c++) {
            barriers[c * 2 + 0].buffer = vegCascadeCullFrames[f].compactBuffers[c].buffer;
            barriers[c * 2 + 1].buffer = vegCascadeCullFrames[f].countBuffers[c].buffer;
            barriers[c * 2 + 1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT
                                              | VK_ACCESS_2_SHADER_WRITE_BIT
                                              | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            barriers[6 + c * 2 + 0].buffer = vegCascadeCullFrames[f].impostorCompactBuffers[c].buffer;
            barriers[6 + c * 2 + 1].buffer = vegCascadeCullFrames[f].impostorCountBuffers[c].buffer;
            barriers[6 + c * 2 + 1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT
                                                  | VK_ACCESS_2_SHADER_WRITE_BIT
                                                  | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 12;
        depInfo.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Bind and dispatch the GPU cascade culling compute shader.
    VkDescriptorSet descSet = vegCascadeCullFrames[f].descSet;
    if (cmdState) cmdState->bindComputePipeline(cmd, vegCascadeCullPipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegCascadeCullPipeline);
    if (cmdState) cmdState->bindComputeDescriptorSets(cmd, vegCascadeCullPipelineLayout, 0, 1, &descSet, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegCascadeCullPipelineLayout, 0, 1, &descSet, 0, nullptr);

    uint32_t numChunks = vegNumChunks;
    vkCmdPushConstants(cmd, vegCascadeCullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numChunks);

    uint32_t groups = (numChunks + 63) / 64;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute shader writes → indirect draw (all 12 buffers). The
    // draw sees whichever wrote last (fill zeros or compute atomics).
    {
        VkBufferMemoryBarrier2 barriers[12]{};
        for (uint32_t i = 0; i < 12; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[i].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                    | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].offset = 0;
            barriers[i].size = VK_WHOLE_SIZE;
        }
        for (uint32_t c = 0; c < 3; c++) {
            barriers[c * 2 + 0].buffer = vegCascadeCullFrames[f].compactBuffers[c].buffer;
            barriers[c * 2 + 1].buffer = vegCascadeCullFrames[f].countBuffers[c].buffer;
            barriers[6 + c * 2 + 0].buffer = vegCascadeCullFrames[f].impostorCompactBuffers[c].buffer;
            barriers[6 + c * 2 + 1].buffer = vegCascadeCullFrames[f].impostorCountBuffers[c].buffer;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 12;
        depInfo.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
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

void VegetationRenderer::setBillboardArrayTextures(VkImageView albedoView, VkImageView normalView, VkImageView opacityView, VkSampler sampler, VulkanApp* app) {
    billboardAlbedoView   = albedoView;
    billboardNormalView   = normalView;
    billboardOpacityView  = opacityView;
    billboardArraySampler = sampler;

    if (!app || descriptorSetLayout == VK_NULL_HANDLE) return;

    if (vegDescriptorSet != VK_NULL_HANDLE) {
        // Defer destruction of the old descriptor set. The current frame's
        // command buffer may still reference it.  deferDestroyUntilAllPending
        // waits for all in-flight rendering to complete before freeing.
        VkDescriptorSet ds = vegDescriptorSet;
        VkDevice dev = app->getDevice();
        VkDescriptorPool pool = app->getDescriptorPool();
        app->deferDestroyUntilAllPending([dev, pool, ds, app]() {
            if (app->resources.removeDescriptorSet(ds))
                vkFreeDescriptorSets(dev, pool, 1, &ds);
        });
        vegDescriptorSet = VK_NULL_HANDLE;
        vegDescriptorVersion = 0;
    }

    ensureVegDescriptorSet(app);
}

void VegetationRenderer::onTextureArraysReallocated(VulkanApp* app) {
    std::cerr << "[VEGETATION] onTextureArraysReallocated: invalidating vegDescriptorSet" << std::endl;
    if (!app) return;
    if (vegDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet ds = vegDescriptorSet;
        VkDevice device = app->getDevice();
        VkDescriptorPool pool = app->getDescriptorPool();
        // Defer the free until all pending command buffers AND in-flight frame
        // fences signal — the descriptor set may still be referenced by a
        // previously-submitted frame command buffer. The remove happens inside
        // the deferred lambda so that, if this handle value is recycled for a
        // newer live set before the lambda runs, the second remove returns
        // false and we avoid a double free of an already-freed set.
        app->deferDestroyUntilAllPending([device, pool, ds, app]() {
            if (app->resources.removeDescriptorSet(ds))
                vkFreeDescriptorSets(device, pool, 1, &ds);
        });
        vegDescriptorSet = VK_NULL_HANDLE;
        vegDescriptorVersion = 0;
    }
    if (ensureVegDescriptorSet(app)) {
        std::cerr << "[VEGETATION] onTextureArraysReallocated: recreated vegDescriptorSet=" << (void*)vegDescriptorSet << std::endl;
    } else {
        std::cerr << "[VEGETATION] onTextureArraysReallocated: descriptor still not ready" << std::endl;
    }
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
        vegDescriptorVersion = 1;
        app->registerDescriptorSet(vegDescriptorSet);
        std::cerr << "[VEGETATION] Allocated vegDescriptorSet=" << (void*)vegDescriptorSet << " (3 sampler2DArray)" << std::endl;
    }
    return vegDescriptorSet != VK_NULL_HANDLE;
}


void VegetationRenderer::init(VulkanApp* app) {
    if (!app) return;
    this->appPtr = app;
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

    // Load indexed indirect draw function pointer
    cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(device, "vkCmdDrawIndexedIndirectCountKHR");
    if (!cmdDrawIndexedIndirectCount)
        cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(device, "vkCmdDrawIndexedIndirectCount");

    // ── Wind params UBO + descriptor set layout (set=2) ────────────────────
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        windParamsDescSetLayout = descAlloc.createLayout(
            &binding, 1,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
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

    // Build billboard corner mesh: 24 vertices (6 planes × 4 corners) + 36 indices
    // (12 triangles = 2 per plane) for TRIANGLE_LIST.
    if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) {
        const glm::vec3 baseTangents[6] = {
            {0,0,1}, {-1,0,0}, {0,0,-1}, {1,0,0}, {1,0,0}, {0,0,1}
        };
        const glm::vec3 outwardDirs[4] = {
            {1,0,0}, {0,0,1}, {-1,0,0}, {0,0,-1}
        };
        const glm::vec3 worldUp(0,1,0);
        constexpr float hs = 0.5f, h = 1.0f, tilt = 1.0f; // scaled in VS by billboardScale

        std::vector<Vertex> verts(24);
        for (int p = 0; p < 6; ++p) {
            glm::vec3 tangent = baseTangents[p];
            glm::vec3 outward = (p < 4) ? outwardDirs[p] : glm::vec3(0.0f);
            int base = p * 4;
            auto corner = [&](int ci, glm::vec3 off, glm::vec2 uv) {
                verts[base + ci].position = off;
                verts[base + ci].color = tangent;
                verts[base + ci].texCoord = uv;
                verts[base + ci].brushIndex = (p << 8) | ci;
            };
            corner(0, -tangent * hs,                    glm::vec2(0,1));  // BL
            corner(1,  tangent * hs,                    glm::vec2(1,1));  // BR
            corner(2, -tangent * hs + worldUp * h + outward * tilt, glm::vec2(0,0));  // TL
            corner(3,  tangent * hs + worldUp * h + outward * tilt, glm::vec2(1,0));  // TR
        }
        // Vertex/index buffers are also used as ray-tracing BLAS inputs, so they
        // need SHADER_DEVICE_ADDRESS_BIT and
        // ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR (the ray-tracing
        // renderer builds a shared cross-quad BLAS from them for alpha-tested
        // vegetation shadows). The original VERTEX/INDEX_BUFFER usage is kept.
        // Only add the RT usage flags when RT is actually supported — otherwise
        // the validation layer rejects the buffer (those usages require the
        // acceleration-structure / buffer-device-address extensions).
        VkBufferUsageFlags rtBufferUsage = app->rtSupport.any()
            ? (VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)
            : 0;
        {
            VkDeviceSize sz = sizeof(Vertex) * verts.size();
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | rtBufferUsage;
            Buffer staging = app->createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            memcpy(staging.mappedData, verts.data(), (size_t)sz);
            billboardVBO.vertexBuffer = app->createBuffer(sz, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /*zeroInit=*/false);
            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                VkBufferCopy r{}; r.size = sz;
                vkCmdCopyBuffer(cmd, staging.buffer, billboardVBO.vertexBuffer.buffer, 1, &r);
            });
            app->destroyBuffer(staging);
        }

        // 36 indices = 6 planes × 2 triangles × 3 indices
        std::vector<uint32_t> idx(36);
        for (int p = 0; p < 6; ++p) {
            int b = p * 4;
            int ib = p * 6;
            idx[ib + 0] = b + 0; idx[ib + 1] = b + 1; idx[ib + 2] = b + 2;
            idx[ib + 3] = b + 1; idx[ib + 4] = b + 3; idx[ib + 5] = b + 2;
        }
        {
            VkDeviceSize sz = sizeof(uint32_t) * idx.size();
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                | rtBufferUsage;
            Buffer staging = app->createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            memcpy(staging.mappedData, idx.data(), (size_t)sz);
            billboardVBO.indexBuffer = app->createBuffer(sz, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /*zeroInit=*/false);
            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                VkBufferCopy r{}; r.size = sz;
                vkCmdCopyBuffer(cmd, staging.buffer, billboardVBO.indexBuffer.buffer, 1, &r);
            });
            app->destroyBuffer(staging);
        }
        billboardVBO.indexCount = 36;
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
}

size_t VegetationRenderer::getInstanceTotal() const {
    size_t total = 0;
    for (const auto& kv : chunkInstanceCounts) {
        total += kv.second;
    }
    return total;
}

float VegetationRenderer::computeDensityFactor(float distanceToCamera) const {
    if (!distanceDensitySettings.enabled) {
        return 1.0f;
    }

    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    if (distanceToCamera <= nearDistance || minFactor >= 1.0f) {
        return 1.0f;
    }

    const float decayRange = farDistance - nearDistance;
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = -std::log(safeMinFactor) / decayRange;
    const float densityFactor = std::exp(-falloff * (distanceToCamera - nearDistance));
    return std::clamp(densityFactor, minFactor, 1.0f);
}

std::vector<DebugCubeRenderer::CubeWithColor> VegetationRenderer::getDensityDebugCubes(const glm::vec3& cameraPos) const {
    std::vector<DebugCubeRenderer::CubeWithColor> cubes;
    cubes.reserve(chunkBuffers.size());

    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) {
            continue;
        }

        const float densityFactor = computeDensityFactor(glm::distance(buf.center, cameraPos));
        const glm::vec3 color = glm::mix(glm::vec3(1.0f, 0.15f, 0.15f), glm::vec3(0.15f, 1.0f, 0.2f), densityFactor);
        const glm::vec3 minPoint = buf.aabbMin;
        const glm::vec3 maxPoint = buf.aabbMax;
        cubes.push_back({BoundingBox(minPoint, maxPoint), color});
    }

    return cubes;
}

float VegetationRenderer::getAverageDensityFactor(const glm::vec3& cameraPos) const {
    if (chunkBuffers.empty()) {
        return 1.0f;
    }

    float factorSum = 0.0f;
    size_t factorCount = 0;
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) {
            continue;
        }
        factorSum += computeDensityFactor(glm::distance(buf.center, cameraPos));
        ++factorCount;
    }

    return factorCount > 0 ? factorSum / static_cast<float>(factorCount) : 1.0f;
}

void VegetationRenderer::recordReadBarriers(VkCommandBuffer& commandBuffer) {
    if (commandBuffer == VK_NULL_HANDLE) return;

    // Reuse frame-thread scratch vector (clear + reserve) to avoid one heap
    // allocation per call; called twice per frame from the main thread.
    readBarrierScratch.clear();
    readBarrierScratch.reserve(chunkBuffers.size() * 2);
    std::vector<VkBufferMemoryBarrier2>& readBarriers = readBarrierScratch;
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;

        // Instance buffers are filled by the CPU (processPendingChunks writes
        // via mapped HOST_VISIBLE memory).  Without this barrier the GPU may
        // read uninitialized billboardIndex values, producing out-of-bounds
        // texture-array accesses that cause RADV GPUVM faults (TCP read).
        VkBufferMemoryBarrier2 instanceBarrier{};
        instanceBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        instanceBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        instanceBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        instanceBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        instanceBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        instanceBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        instanceBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        instanceBarrier.buffer = buf.buffer;
        instanceBarrier.offset = 0;
        instanceBarrier.size = VK_WHOLE_SIZE;
        readBarriers.push_back(instanceBarrier);

        VkBufferMemoryBarrier2 indirectBarrier{};
        indirectBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        indirectBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        indirectBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        indirectBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        indirectBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        indirectBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.buffer = buf.indirectBuffer;
        indirectBarrier.offset = 0;
        indirectBarrier.size = VK_WHOLE_SIZE;
        readBarriers.push_back(indirectBarrier);
    }
    if (readBarriers.empty()) return;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(readBarriers.size());
    depInfo.pBufferMemoryBarriers = readBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &depInfo);
}



void VegetationRenderer::setImpostorData(VulkanApp* app,
                                          VkImageView albedoArray60,
                                          VkImageView normalArray60,
                                          VkSampler sampler,
                                          VkImageView depthArray60,
                                          VkBuffer captureInvVPBuf) {
    if (!app || albedoArray60 == VK_NULL_HANDLE || normalArray60 == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
    // Wait for all in-flight graphics work to complete before recreating descriptor sets.
    // This prevents handle-reuse collisions between pending command buffers and
    // freshly-allocated descriptor set handles. Only waits on per-frame fences
    // rather than draining the entire device.
    app->waitForFrameFences();

    VkDevice device = app->getDevice();

    // Destroy any previous impostor resources (handles are tracked in the central manager).
    impostorDescSetLayout       = VK_NULL_HANDLE;
    impostorDescPool            = VK_NULL_HANDLE;
    impostorDescSet             = VK_NULL_HANDLE;
    impostorDepthDescSetLayout  = VK_NULL_HANDLE;
    impostorDepthDescPool       = VK_NULL_HANDLE;
    impostorDepthDescSet        = VK_NULL_HANDLE;

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
        DescriptorAllocator descAlloc{device, app};
        impostorDescSetLayout = descAlloc.createLayout(
            bindings, numBindings,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            nullptr,
            "VegetationRenderer: impostorDescSetLayout");

        {
            std::vector<VkDescriptorPoolSize> poolSz;
            poolSz.emplace_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hasImpostorDepth ? 3u : 2u});
            if (hasImpostorDepth)
                poolSz.emplace_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u});
            impostorDescPool = descAlloc.createPool(
                poolSz.data(), static_cast<uint32_t>(poolSz.size()), 1,
                VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
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
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            nullptr,
            "VegetationRenderer: impostorDepthDescSetLayout");

        VkDescriptorPoolSize depthPoolSizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
        };
        impostorDepthDescPool = depthDescAlloc.createPool(
            depthPoolSizes, 2, 1,
            VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            "VegetationRenderer: impostorDepthDescPool");

        impostorDepthDescSet = depthDescAlloc.allocateSet(impostorDepthDescPool, impostorDepthDescSetLayout);

        DescriptorWriter(device)
            .writeImage(impostorDepthDescSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        sampler, depthArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .writeBuffer(impostorDepthDescSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         captureInvVPBuf, 0, VK_WHOLE_SIZE)
            .flush();

    }

}


VegetationRenderer::WindPushConstants VegetationRenderer::buildWindPushConstants(const glm::vec3& cameraPos) const {
    (void)cameraPos;
    WindPushConstants pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = windSettings.enabled ? 1.0f : 0.0f;
    pc.windTime           = windTimeSeconds;
    pc.impostorDistance   = impostorDistance;
    return pc;
}

void VegetationRenderer::updateWindParamsUBO(const glm::vec3& cameraPos) {
    if (!windParamsMapped) return;

    WindParamsUBO params{};

    glm::vec2 windDir = windSettings.direction;
    const float len2 = windDir.x * windDir.x + windDir.y * windDir.y;
    if (len2 > 1e-8f) {
        const float invLen = 1.0f / std::sqrt(len2);
        windDir *= invLen;
    }
    params.windDirAndStrength = glm::vec4(windDir.x, 0.0f, windDir.y, std::max(0.0f, windSettings.strength));
    params.windNoise = glm::vec4(
        std::max(0.00001f, windSettings.baseFrequency),
        std::max(0.0f, windSettings.speed),
        std::max(0.00001f, windSettings.gustFrequency),
        std::max(0.0f, windSettings.gustStrength));
    params.windShape = glm::vec4(
        std::max(0.0f, windSettings.skewAmount),
        std::clamp(windSettings.trunkStiffness, 0.0f, 1.0f),
        std::max(0.001f, windSettings.noiseScale),
        std::max(0.0f, windSettings.verticalFlutter));
    params.windTurbulence = glm::vec4(std::max(0.0f, windSettings.turbulence), 0.0f, 0.0f, 0.0f);
    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = (distanceDensitySettings.enabled && minFactor < 1.0f)
        ? (-std::log(safeMinFactor) / (farDistance - nearDistance)) : 0.0f;
    params.densityParams = glm::vec4(distanceDensitySettings.enabled ? 1.0f : 0.0f, nearDistance, farDistance, minFactor);
    params.cameraPosAndFalloff = glm::vec4(cameraPos, falloff);

    std::memcpy(windParamsMapped, &params, sizeof(params));
}






// ── CPU-side instance generation ─────────────────────────────────────────────
// Mirrors the previous GPU compute path (now removed) on the CPU.  Avoids GPUVM
// faults on RADV iGPUs where TCP cannot read storage buffers from any
// memory type (device-local, host-visible, or concurrent-shared).

namespace {

// XorShift32 PRNG — matches compute shader behaviour.
uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(uint32_t& state) {
    return float(xorshift32(state) & 0x00FFFFFFu) / float(0x01000000u);
}

// Position hash — matches posHash in the compute shader.
uint32_t posHash(const glm::vec3& p) {
    glm::ivec3 qi = glm::ivec3(glm::round(p * 8.0f));
    uint32_t h = uint32_t(qi.x) * 1640531513u;
    h ^= uint32_t(qi.y) * 2246822519u;
    h ^= uint32_t(qi.z) * 3266489917u;
    return h;
}

// 2D cell hash — matches cellHash in the compute shader.
uint32_t cellHash(glm::ivec2 c) {
    uint32_t h = uint32_t(c.x) * 1640531513u ^ uint32_t(c.y) * 2246822519u;
    h ^= h >> 13;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

// Biome noise — matches biomeNoise in the compute shader.
float biomeNoise(const glm::vec2& xz) {
    const float kBiomeScale = 50.0f;
    glm::vec2 p = xz / kBiomeScale;
    glm::ivec2 i = glm::ivec2(glm::floor(p));
    glm::vec2 f = p - glm::vec2(i);
    glm::vec2 u = f * f * (3.0f - 2.0f * f);

    float a = float(cellHash(i + glm::ivec2(0, 0))) / 4294967295.0f;
    float b = float(cellHash(i + glm::ivec2(1, 0))) / 4294967295.0f;
    float c = float(cellHash(i + glm::ivec2(0, 1))) / 4294967295.0f;
    float d = float(cellHash(i + glm::ivec2(1, 1))) / 4294967295.0f;

    return glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y);
}

} // anonymous namespace

void VegetationRenderer::generateChunkInstancesCPU(NodeID chunkId,
                                                   const std::vector<glm::vec3>& positions,
                                                   const std::vector<uint32_t>& grassIndices,
                                                   const glm::vec3& chunkCenter,
                                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                                   uint32_t seed) {
    (void)app; // used later in processPendingChunks
    if (grassIndices.size() < 3 || instancesPerTriangle == 0 || positions.empty()) {
        destroyInstanceBuffer(chunkId, app);
        return;
    }
    // Enqueue for later processing — the render thread drains this queue.
    PendingChunk pc;
    pc.chunkId             = chunkId;
    pc.positions           = positions;
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

void VegetationRenderer::processPendingChunks(uint32_t maxChunks) {
    if (!appPtr) return;
    VulkanApp* app = appPtr;
    const uint32_t billboardCnt = (billboardCount > 0) ? billboardCount : 1u;

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

        const float maxBillboardRadius = billboardScale * 2.1f; // max heightScale (1.4) × max corner offset (1.5)

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
        instanceGenScratch.reserve(instanceCount * 4);
        std::vector<float>& validData = instanceGenScratch;

        for (uint32_t tri = 0; tri < triCount; ++tri) {
            const uint32_t tb = tri * 3;
            const uint32_t i0 = pc.grassIndices[tb + 0];
            const uint32_t i1 = pc.grassIndices[tb + 1];
            const uint32_t i2 = pc.grassIndices[tb + 2];
            if (i0 >= pc.positions.size() || i1 >= pc.positions.size() || i2 >= pc.positions.size()) continue;

            const glm::vec3 v0 = pc.positions[i0];
            const glm::vec3 v1 = pc.positions[i1];
            const glm::vec3 v2 = pc.positions[i2];

            const glm::vec3 fn = glm::cross(v1 - v0, v2 - v0);
            if (glm::abs(fn.y) <= 0.5f * glm::length(fn)) continue;

            const glm::vec3 tc = (v0 + v1 + v2) / 3.0f;
            const uint32_t tch = posHash(tc);

            for (uint32_t s = 0; s < pc.instancesPerTriangle; ++s) {
                uint32_t rng = pc.seed ^ tch ^ (tri * 2654435761u) ^ (s * 19349663u);
                float u = randFloat(rng);
                float v = randFloat(rng);
                if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                float w = 1.0f - u - v;
                glm::vec3 pos = u * v0 + v * v1 + w * v2;

                // Biome from spatially-coherent noise.
                float noise = biomeNoise(glm::vec2(pos.x, pos.z));
                if (noise < 0.40f) continue; // empty biome — skip entirely

                float remapped = (noise - 0.40f) / 0.60f;
                uint32_t bi = std::min(uint32_t(remapped * float(billboardCnt)), billboardCnt - 1u);

                uint32_t rs = pc.seed ^ posHash(pos);
                float rf = randFloat(rs);

                validData.push_back(pos.x);
                validData.push_back(pos.y);
                validData.push_back(pos.z);
                validData.push_back(float(bi) + rf);
            }
        }

        const uint32_t validCount = static_cast<uint32_t>(validData.size() / 4);
        if (validCount == 0) {
            destroyInstanceBuffer(pc.chunkId, app);
            continue;
        }

        const VkDeviceSize bufSize = validData.size() * sizeof(float);

        // Staging buffer: host-visible, filled by CPU.
        Buffer stagingInst = app->createBuffer(bufSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        mapped = stagingInst.map(0);
        std::memcpy(mapped, validData.data(), size_t(bufSize));
        stagingInst.unmap(); // VMA persistent mapping

        // Device-local instance buffer: GPU reads via vertex input.
        // On RADV iGPUs, vertex reads go through TCP (Texture Cache/Pipe),
        // and host-visible pages lack TCP-read permission → GPUVM fault.
        // zeroInit=false: fully overwritten by the staging copy before use.
        Buffer instBuf = app->createBuffer(bufSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);

        // Indirect buffer: device-local, avoids same TCP-read issue.
        Buffer indirect = app->createBuffer(sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);

        // Staging for indirect draw command.
        Buffer stagingIndirect = app->createBuffer(sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDrawIndexedIndirectCommand drawCmd{};
        drawCmd.indexCount    = 36;
        drawCmd.instanceCount = validCount;
        drawCmd.firstIndex    = 0;
        drawCmd.vertexOffset  = 0;
        drawCmd.firstInstance = 0;
        void* idata = nullptr;
        idata = stagingIndirect.map(0);
        std::memcpy(idata, &drawCmd, sizeof(VkDrawIndexedIndirectCommand));
        stagingIndirect.unmap(); // VMA persistent mapping

        pendingBatch.push_back({ stagingInst, instBuf, stagingIndirect, indirect,
                                 bufSize, pc.chunkId, validCount,
                                 aabbMin, aabbMax, pc.chunkCenter });
    }

    // Flush all batched copies in a single async submission.
    // The deferred callback publishes chunks when the GPU is done.
    if (!pendingBatch.empty()) {
        std::vector<PendingBatchCopy> batch = std::move(pendingBatch);
        VkFence fence = app->runSingleTimeCommandsAsync([&](VkCommandBuffer cmd) {
            for (auto& c : batch) {
                VkBufferCopy cr{};
                cr.size = c.bufSize;
                vkCmdCopyBuffer(cmd, c.stagingInst.buffer, c.instBuf.buffer, 1, &cr);
                VkBufferCopy icr{};
                icr.size = sizeof(VkDrawIndexedIndirectCommand);
                vkCmdCopyBuffer(cmd, c.stagingIndirect.buffer, c.indirect.buffer, 1, &icr);
            }
        });
        app->deferDestroyUntilFence(fence, [this, app,
                                             batch = std::move(batch)]() mutable {
            for (auto& c : batch) {
                app->destroyBuffer(c.stagingInst);
                app->destroyBuffer(c.stagingIndirect);

                destroyInstanceBuffer(c.chunkId, app);

                InstanceBuffer ibuf;
                ibuf.buffer         = c.instBuf.buffer;
                ibuf.memory         = c.instBuf.memory;
                ibuf.allocation     = c.instBuf.allocation;
                ibuf.indirectBuffer = c.indirect.buffer;
                ibuf.indirectMemory = c.indirect.memory;
                ibuf.indirectAllocation = c.indirect.allocation;
                ibuf.center         = c.center;
                ibuf.aabbMin        = c.aabbMin;
                ibuf.aabbMax        = c.aabbMax;
                ibuf.count          = c.instanceCount;
                chunkBuffers[c.chunkId] = ibuf;
                chunkInstanceCounts[c.chunkId] = c.instanceCount;
                vegConsolidationDirty = true;
            }
        });
    }
}

void VegetationRenderer::destroyInstanceBuffer(NodeID chunkId, VulkanApp* app, VkFence completionFence) {
    auto it = chunkBuffers.find(chunkId);
    if (it == chunkBuffers.end()) return;

    // Snatch the old Vulkan handles before clearing the map entry.
    InstanceBuffer old = it->second;
    it->second.buffer = VK_NULL_HANDLE;
    it->second.memory = VK_NULL_HANDLE;
    it->second.indirectBuffer = VK_NULL_HANDLE;
    it->second.indirectMemory = VK_NULL_HANDLE;
    chunkBuffers.erase(it);
    chunkInstanceCounts.erase(chunkId);

    if (!app) return;

    // Defer destruction until the provided fence signals (or until all
    // pending work completes if VK_NULL_HANDLE). The old instance buffer
    // may still be referenced by previously-submitted render command
    // buffers on the graphics queue — waiting only on the compute/copy
    // dispatch fence is insufficient.
    app->deferDestroyUntilFence(completionFence, [app, old]() {
        {
            Buffer tmpBuf{};
            tmpBuf.buffer = old.buffer;
            tmpBuf.memory = old.memory;
            tmpBuf.allocation = old.allocation;
            app->destroyBuffer(tmpBuf);
        }
        {
            Buffer tmpBuf{};
            tmpBuf.buffer = old.indirectBuffer;
            tmpBuf.memory = old.indirectMemory;
            tmpBuf.allocation = old.indirectAllocation;
            app->destroyBuffer(tmpBuf);
        }
    });
}

// Ensure we clear the stored app pointer on cleanup
// (cleanup() already clears handles; set appPtr to nullptr here)

void VegetationRenderer::generateForChunk(VulkanApp* app, NodeID nid, const Geometry& geom) {
    if (geom.indices.size() < 3 || geom.vertices.empty()) return;
    try {
        constexpr int kGrassBrushIndex = 3; // See LandBrush::grass
        // Instances per world-space unit² of triangle area.
        constexpr float kVegetationDensity = 0.01f;

        // Create tightly-packed position buffer (vec3[]) for the compute shader
        std::vector<glm::vec3> positions;
        positions.reserve(geom.vertices.size());
        for (const auto &v : geom.vertices) positions.push_back(v.position);

        // Build area-weighted virtual slots using unbiased stochastic rounding.
        // expected = area * density (instances per world-space unit area)
        // count = floor(expected) + Bernoulli(frac(expected))
        // This preserves area-proportional density without bias.
        std::vector<uint32_t> grassIndices;
        grassIndices.reserve(geom.indices.size());
        const uint32_t chunkSeed = static_cast<uint32_t>(nid ^ (nid >> 32)) ^ 0x9e3779b9u;
        std::mt19937 samplingRng(chunkSeed);
        std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
        for (size_t i = 0; i + 2 < geom.indices.size(); i += 3) {
            const uint32_t i0 = geom.indices[i + 0];
            const uint32_t i1 = geom.indices[i + 1];
            const uint32_t i2 = geom.indices[i + 2];
            if (i0 >= geom.vertices.size() || i1 >= geom.vertices.size() || i2 >= geom.vertices.size()) continue;
            const bool hasGrass =
                geom.vertices[i0].brushIndex == kGrassBrushIndex ||
                geom.vertices[i1].brushIndex == kGrassBrushIndex ||
                geom.vertices[i2].brushIndex == kGrassBrushIndex;
            if (!hasGrass) continue;
            const glm::vec3& v0 = geom.vertices[i0].position;
            const glm::vec3& v1 = geom.vertices[i1].position;
            const glm::vec3& v2 = geom.vertices[i2].position;
            // Skip steep / downward-facing triangles (same criterion as compute shader).
            // This avoids allocating output slots that the compute shader would discard,
            // preventing garbage uninitialized memory from reaching the draw call.
            const glm::vec3 faceNormal = glm::cross(v1 - v0, v2 - v0);
            if (glm::abs(faceNormal.y) <= 0.5f * glm::length(faceNormal)) continue;
            const float area = 0.5f * glm::length(faceNormal);
            const float expectedInstances = std::max(0.0f, area * kVegetationDensity);
            uint32_t slotCount = static_cast<uint32_t>(std::floor(expectedInstances));
            const float fractional = expectedInstances - static_cast<float>(slotCount);
            if (unitDist(samplingRng) < fractional) {
                ++slotCount;
            }
            for (uint32_t s = 0; s < slotCount; ++s) {
                grassIndices.push_back(i0);
                grassIndices.push_back(i1);
                grassIndices.push_back(i2);
            }
        }

        // Shuffle virtual triangle slots per chunk so reducing indirect instanceCount
        // keeps a random spatial subset instead of always dropping the tail.
        if (grassIndices.size() >= 6) {
            std::mt19937 shuffleRng(chunkSeed ^ 0x85ebca6bu);
            const size_t triangleCount = grassIndices.size() / 3;
            for (size_t slot = triangleCount - 1; slot > 0; --slot) {
                std::uniform_int_distribution<size_t> dist(0, slot);
                const size_t other = dist(shuffleRng);
                if (other == slot) continue;
                for (size_t component = 0; component < 3; ++component) {
                    std::swap(grassIndices[slot * 3 + component], grassIndices[other * 3 + component]);
                }
            }
        }

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
            generateChunkInstancesCPU(nid, positions, grassIndices,
                chunkCenter, instancesPerTriangle, app, seed);
            return;
        }

        // CPU-side instance generation — avoids RADV GPUVM faults where
        // the Texture Cache/Pipe cannot read storage buffers on iGPUs.
        generateChunkInstancesCPU(nid, positions, grassIndices,
            chunkCenter, instancesPerTriangle, app, seed);
    } catch (const std::exception &e) {
        std::cerr << "[VegetationRenderer] Vegetation generation failed for node " << (unsigned long long)nid
                  << ": " << e.what() << std::endl;
    }
}
