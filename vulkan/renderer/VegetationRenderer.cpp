#include "VegetationRenderer.hpp"

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
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

// GPU injection APIs removed. Instances must be generated via compute shader.


VegetationRenderer::VegetationRenderer() {}
VegetationRenderer::~VegetationRenderer() { /* caller must call cleanup(app) */ }

void VegetationRenderer::init() {

}


void VegetationRenderer::cleanup() {
    // Collect IDs first to avoid erasing while iterating
    std::vector<NodeID> idsToDestroy;
    idsToDestroy.reserve(chunkBuffers.size());
    for (const auto& [id, _] : chunkBuffers) idsToDestroy.push_back(id);
    for (NodeID id : idsToDestroy) destroyInstanceBuffer(id);
    chunkBuffers.clear();
    chunkInstanceCounts.clear();
    // Clear local handles; central manager handles destruction of Vulkan objects
    vegetationPipeline = VK_NULL_HANDLE;
    vegetationDepthPipeline = VK_NULL_HANDLE;
    vegetationDepthPipelineLayout = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    vegetationShadowPipeline = VK_NULL_HANDLE;
    shadowPipelineLayout = VK_NULL_HANDLE;
    descriptorSetLayout = VK_NULL_HANDLE;
    // Clear impostor resources (also tracked by central manager).
    impostorPipeline       = VK_NULL_HANDLE;
    impostorPipelineLayout = VK_NULL_HANDLE;
    impostorDescSetLayout  = VK_NULL_HANDLE;
    impostorDescPool       = VK_NULL_HANDLE;
    impostorDescSet        = VK_NULL_HANDLE;
    // Free and reset vegetation descriptor set handle locally
    vegDescriptorSet = VK_NULL_HANDLE;
    vegDescriptorVersion = 0;
    // Unregister allocation listener if set
    if (vegetationTextureArrayManager && vegTextureListenerId != -1) {
        vegetationTextureArrayManager->removeAllocationListener(vegTextureListenerId);
        vegTextureListenerId = -1;
    }
    // Clear stored app pointer and billboard VBO handles
    billboardAlbedoView   = VK_NULL_HANDLE;
    billboardNormalView   = VK_NULL_HANDLE;
    billboardOpacityView  = VK_NULL_HANDLE;
    billboardArraySampler = VK_NULL_HANDLE;
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
    appPtr = nullptr;
    destroyCulling();
}

void VegetationRenderer::destroyCulling() {
    auto device = appPtr ? appPtr->getDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;
    if (concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE) {
        if (appPtr->resources.removeBuffer(concatenatedInstanceBuffer.buffer))
            vkDestroyBuffer(device, concatenatedInstanceBuffer.buffer, nullptr);
        if (appPtr->resources.removeDeviceMemory(concatenatedInstanceBuffer.memory))
            vkFreeMemory(device, concatenatedInstanceBuffer.memory, nullptr);
        concatenatedInstanceBuffer = {};
    }
    if (chunkMetaBuffer.buffer != VK_NULL_HANDLE) {
        if (appPtr->resources.removeBuffer(chunkMetaBuffer.buffer))
            vkDestroyBuffer(device, chunkMetaBuffer.buffer, nullptr);
        if (appPtr->resources.removeDeviceMemory(chunkMetaBuffer.memory))
            vkFreeMemory(device, chunkMetaBuffer.memory, nullptr);
        chunkMetaBuffer = {};
    }
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f) {
        if (compactedCmdBuffers[f].buffer != VK_NULL_HANDLE) {
            if (appPtr->resources.removeBuffer(compactedCmdBuffers[f].buffer))
                vkDestroyBuffer(device, compactedCmdBuffers[f].buffer, nullptr);
            if (appPtr->resources.removeDeviceMemory(compactedCmdBuffers[f].memory))
                vkFreeMemory(device, compactedCmdBuffers[f].memory, nullptr);
            compactedCmdBuffers[f] = {};
        }
        if (visibleCountBuffers[f].buffer != VK_NULL_HANDLE) {
            if (visibleCountMapped[f] && appPtr) {
                vkUnmapMemory(device, visibleCountBuffers[f].memory);
                visibleCountMapped[f] = nullptr;
            }
            if (appPtr->resources.removeBuffer(visibleCountBuffers[f].buffer))
                vkDestroyBuffer(device, visibleCountBuffers[f].buffer, nullptr);
            if (appPtr->resources.removeDeviceMemory(visibleCountBuffers[f].memory))
                vkFreeMemory(device, visibleCountBuffers[f].memory, nullptr);
            visibleCountBuffers[f] = {};
        }
        vegCullDescSets[f] = VK_NULL_HANDLE;
    }
    if (vegCullDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, vegCullDescPool, nullptr);
        vegCullDescPool = VK_NULL_HANDLE;
    }
    // Pipeline and layouts are tracked by central manager — just clear handles
    vegCullPipeline = VK_NULL_HANDLE;
    vegCullPipelineLayout = VK_NULL_HANDLE;
    vegCullDescSetLayout = VK_NULL_HANDLE;
    if (consolidationFence != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &consolidationFence, VK_TRUE, UINT64_MAX);
        consolidationFence = VK_NULL_HANDLE;
    }
    pendingMeta.clear();
    pendingMetaSize = 0;
    consolidationPending = false;
    vegNumChunks = 0;
    vegConsolidationDirty = true;
}

void VegetationRenderer::initCulling(VulkanApp* app) {
    if (vegCullPipeline != VK_NULL_HANDLE) return;
    auto device = app->getDevice();

    auto compCode = FileReader::readFile("shaders/vegetation_cull.comp.spv");
    VkShaderModule compModule = app->createShaderModule(compCode);

    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].binding = 2;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &vegCullDescSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create vegetation cull desc set layout!");
    app->resources.addDescriptorSetLayout(vegCullDescSetLayout, "VegetationCull: descSetLayout");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(glm::mat4) + sizeof(uint32_t);

    VkPipelineLayoutCreateInfo plinfo{};
    plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plinfo.setLayoutCount = 1;
    plinfo.pSetLayouts = &vegCullDescSetLayout;
    plinfo.pushConstantRangeCount = 1;
    plinfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(device, &plinfo, nullptr, &vegCullPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create vegetation cull pipeline layout!");
    app->resources.addPipelineLayout(vegCullPipelineLayout, "VegetationCull: pipelineLayout");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = vegCullPipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vegCullPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create vegetation cull compute pipeline!");
    app->resources.addPipeline(vegCullPipeline, "VegetationCull: computePipeline");

    vkDestroyShaderModule(device, compModule, nullptr);

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = VEG_CULL_FRAMES * 3;
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets = VEG_CULL_FRAMES;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &vegCullDescPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create vegetation cull descriptor pool!");

    std::array<VkDescriptorSetLayout, VEG_CULL_FRAMES> layouts;
    layouts.fill(vegCullDescSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = vegCullDescPool;
    allocInfo.descriptorSetCount = VEG_CULL_FRAMES;
    allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &allocInfo, vegCullDescSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate vegetation cull descriptor sets!");
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f)
        std::cerr << "[VEG] vegCullDescSet[" << f << "] = " << (void*)vegCullDescSets[f] << std::endl;
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
    std::cerr << "[VEG CONSOLIDATE] Phase 2 start" << std::endl;
    initCulling(app);
    std::cerr << "[VEG CONSOLIDATE] initCulling done" << std::endl;

    size_t totalInstances = 0;
    for (const auto& kv : chunkInstanceCounts) {
        std::cerr << "[VEG CONSOLIDATE] chunkInstanceCounts: " << kv.first << " = " << kv.second << std::endl;
        totalInstances += kv.second;
    }
    std::cerr << "[VEG CONSOLIDATE] totalInstances=" << totalInstances << " numChunks=" << chunkBuffers.size() << std::endl;
    if (totalInstances == 0) return;

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
            if (concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE) {
                app->resources.removeBuffer(concatenatedInstanceBuffer.buffer);
                vkDestroyBuffer(device, concatenatedInstanceBuffer.buffer, nullptr);
                app->resources.removeDeviceMemory(concatenatedInstanceBuffer.memory);
                vkFreeMemory(device, concatenatedInstanceBuffer.memory, nullptr);
                concatenatedInstanceBuffer = {};
            }
            concatenatedInstanceBuffer = app->createBuffer(concatSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
    }

    VkDeviceSize metaSize = numChunks * sizeof(ChunkMeta);
    {
        bool needsCreate = chunkMetaBuffer.buffer == VK_NULL_HANDLE;
        if (!needsCreate) {
            VkMemoryRequirements reqs;
            vkGetBufferMemoryRequirements(device, chunkMetaBuffer.buffer, &reqs);
            if (reqs.size < metaSize) needsCreate = true;
        }
        if (needsCreate) {
            if (chunkMetaBuffer.buffer != VK_NULL_HANDLE) {
                app->resources.removeBuffer(chunkMetaBuffer.buffer);
                vkDestroyBuffer(device, chunkMetaBuffer.buffer, nullptr);
                app->resources.removeDeviceMemory(chunkMetaBuffer.memory);
                vkFreeMemory(device, chunkMetaBuffer.memory, nullptr);
                chunkMetaBuffer = {};
            }
            chunkMetaBuffer = app->createBuffer(metaSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
            if (compactedCmdBuffers[f].buffer != VK_NULL_HANDLE) {
                app->resources.removeBuffer(compactedCmdBuffers[f].buffer);
                vkDestroyBuffer(device, compactedCmdBuffers[f].buffer, nullptr);
                app->resources.removeDeviceMemory(compactedCmdBuffers[f].memory);
                vkFreeMemory(device, compactedCmdBuffers[f].memory, nullptr);
                compactedCmdBuffers[f] = {};
            }
            compactedCmdBuffers[f] = app->createBuffer(compactedSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (visibleCountBuffers[f].buffer == VK_NULL_HANDLE) {
            visibleCountBuffers[f] = app->createBuffer(sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkMapMemory(device, visibleCountBuffers[f].memory, 0, sizeof(uint32_t), 0,
                        reinterpret_cast<void**>(&visibleCountMapped[f]));
            *visibleCountMapped[f] = 0;
        }
    }

    // Build per-chunk metadata on CPU (no GPU copy — concatenated buffer unused)
    std::vector<ChunkMeta> metaArray(numChunks);
    uint32_t instanceOffset = 0;
    uint32_t chunkIdx = 0;

    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
        ChunkMeta& meta = metaArray[chunkIdx];
        meta.instanceOffset = instanceOffset;
        meta.instanceCount = static_cast<uint32_t>(buf.count);
        meta.aabbMin = buf.aabbMin;
        meta.aabbMax = buf.aabbMax;
        meta.pad0 = 0.0f;
        meta.pad1 = 0.0f;
        instanceOffset += static_cast<uint32_t>(buf.count);
        chunkIdx++;
    }
    vegNumChunks = chunkIdx;

    if (vegNumChunks == 0) return;

    // Upload metadata via staging buffer (device-local, shader-safe)
    if (chunkMetaBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
        VkDeviceSize upSize = vegNumChunks * sizeof(ChunkMeta);
        Buffer staging = app->createBuffer(upSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data;
        vkMapMemory(device, staging.memory, 0, upSize, 0, &data);
        memcpy(data, metaArray.data(), upSize);
        vkUnmapMemory(device, staging.memory);

        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.size = upSize;
            vkCmdCopyBuffer(cmd, staging.buffer, chunkMetaBuffer.buffer, 1, &region);
        });

        VkDevice dev = device;
        if (app->resources.removeBuffer(staging.buffer))
            vkDestroyBuffer(dev, staging.buffer, nullptr);
        if (app->resources.removeDeviceMemory(staging.memory))
            vkFreeMemory(dev, staging.memory, nullptr);
    }

    // Copy per-chunk instance data into the concatenated buffer (GPU→GPU)
    if (concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
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

    // Update descriptor sets
    for (uint32_t f = 0; f < VEG_CULL_FRAMES; ++f) {
        VkDescriptorBufferInfo metaBI{};
        metaBI.buffer = chunkMetaBuffer.buffer;
        metaBI.offset = 0;
        metaBI.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo compactedBI{};
        compactedBI.buffer = compactedCmdBuffers[f].buffer;
        compactedBI.offset = 0;
        compactedBI.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo countBI{};
        countBI.buffer = visibleCountBuffers[f].buffer;
        countBI.offset = 0;
        countBI.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = vegCullDescSets[f];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &metaBI;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = vegCullDescSets[f];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &compactedBI;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = vegCullDescSets[f];
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &countBI;
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
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
    if (vegCullPipeline == VK_NULL_HANDLE || vegNumChunks == 0) return;
    if (compactedCmdBuffers[f].buffer == VK_NULL_HANDLE || visibleCountBuffers[f].buffer == VK_NULL_HANDLE) return;

    // Upload every chunk as a visible draw command via vkCmdUpdateBuffer
    // (GPU-side transfer, bypasses the compute cull shader that crashes on RADV).
    uint32_t count = 0;
    {
        // Build draw commands on the stack and upload in one batch
        VkDrawIndexedIndirectCommand cmds[256];
        uint32_t instanceOff = 0;
        for (auto& [cid, buf] : chunkBuffers) {
            (void)cid;
            if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
            cmds[count].indexCount = 36;
            cmds[count].instanceCount = buf.count;
            cmds[count].firstIndex = 0;
            cmds[count].vertexOffset = 0;
            cmds[count].firstInstance = instanceOff;
            instanceOff += buf.count;
            count++;
            if (count >= vegNumChunks) break;
        }
        VkDeviceSize upSize = count * sizeof(VkDrawIndexedIndirectCommand);
        if (upSize > 0) {
            vkCmdUpdateBuffer(cmd, compactedCmdBuffers[f].buffer, 0, upSize, cmds);
        }
    }

    // Write visible count
    vkCmdUpdateBuffer(cmd, visibleCountBuffers[f].buffer, 0, sizeof(count), &count);

    // Barrier: make transfer writes visible to indirect draw
    {
        VkBufferMemoryBarrier2 barriers[2] = {};
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        barriers[0].buffer = compactedCmdBuffers[f].buffer;
        barriers[0].offset = 0;
        barriers[0].size = VK_WHOLE_SIZE;
        barriers[1] = barriers[0];
        barriers[1].buffer = visibleCountBuffers[f].buffer;
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 2;
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
        // Only free descriptor set if it was tracked by the resource manager.
        // Always defer until all pending command buffers AND in-flight frame
        // fences signal — the descriptor set may still be referenced by a
        // previously-submitted frame command buffer.
        if (app->resources.removeDescriptorSet(ds)) {
            VkDevice device = app->getDevice();
            VkDescriptorPool pool = app->getDescriptorPool();
            app->deferDestroyUntilAllPending([device, pool, ds]() {
                vkFreeDescriptorSets(device, pool, 1, &ds);
            });
        }
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

        VkDescriptorImageInfo infos[3]{};
        infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[0].imageView   = billboardAlbedoView;
        infos[0].sampler     = billboardArraySampler;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[1].imageView   = billboardNormalView;
        infos[1].sampler     = billboardArraySampler;
        infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[2].imageView   = billboardOpacityView;
        infos[2].sampler     = billboardArraySampler;

        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = vegDescriptorSet;
            writes[i].dstBinding      = i;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo      = &infos[i];
        }
        vkUpdateDescriptorSets(app->getDevice(), 3, writes, 0, nullptr);
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

    // Descriptor set layout: set=1, binding 0=albedo, 1=normal, 2=opacity
    VkDescriptorSetLayoutBinding texBindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        texBindings[i].binding         = i;
        texBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texBindings[i].descriptorCount = 1;
        texBindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        texBindings[i].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = texBindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create vegetation descriptor set layout");
    app->resources.addDescriptorSetLayout(descriptorSetLayout, "VegetationRenderer: descriptorSetLayout");

    // Load indexed indirect draw function pointer
    cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(device, "vkCmdDrawIndexedIndirectCountKHR");
    if (!cmdDrawIndexedIndirectCount)
        cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(device, "vkCmdDrawIndexedIndirectCount");

    std::vector<VkDescriptorSetLayout> setLayouts;
    setLayouts.push_back(app->getDescriptorSetLayout());
    setLayouts.push_back(descriptorSetLayout);
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(WindPushConstants);

    // Load shaders — no geometry shader
    auto vertCode = FileReader::readFile("shaders/vegetation.vert.spv");
    auto fragCode = FileReader::readFile("shaders/vegetation.frag.spv");
    VkShaderModule vertShader = app->createShaderModule(vertCode);
    VkShaderModule fragShader = app->createShaderModule(fragCode);

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

    VkVertexInputBindingDescription bindingDescs[2] = {};
    bindingDescs[0].binding = 0;
    bindingDescs[0].stride = sizeof(Vertex);
    bindingDescs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescs[1].binding = 1;
    bindingDescs[1].stride = sizeof(float) * 4;
    bindingDescs[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // Shared attribute descriptions: localPos at POS, tangent at COLOR, UV, plane-data at BRUSH_INDEX
    std::vector<VkVertexInputAttributeDescription> attribDescs(5);
    attribDescs[0] = { ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) };
    attribDescs[1] = { ATTR_COLOR, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };
    attribDescs[2] = { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) };
    attribDescs[3] = { ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) };
    attribDescs[4] = { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };

    // ── Shading pass pipeline (TRIANGLE_LIST, no geometry shader) ──
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { stages[0], stages[1] },
        std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1]},
        attribDescs,
        setLayouts, &pushConstantRange,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
        false, true, VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, {},
        VK_FORMAT_D32_SFLOAT, false
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
        auto depthVertCode = FileReader::readFile("shaders/vegetation.vert.spv");
        auto depthFragCode = FileReader::readFile("shaders/vegetation_depth.frag.spv");
        VkShaderModule depthVertShader = app->createShaderModule(depthVertCode);
        VkShaderModule depthFragShader = app->createShaderModule(depthFragCode);

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

        auto [depthPipe, depthLayout] = app->createGraphicsPipeline(
            { depthStages[0], depthStages[1] },
            std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1]},
            attribDescs,
            setLayouts,
            &pushConstantRange,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            true,  // depthWrite — depth prepass writes depth
            true,  // colorWrite (ignored since noColorAttachment=true)
            VK_COMPARE_OP_LESS,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false,
            {},
            VK_FORMAT_D32_SFLOAT,
            true,  // noColorAttachment — depth-only pass
            false  // depthBiasEnable
        );
        vegetationDepthPipeline = depthPipe;
        vegetationDepthPipelineLayout = depthLayout;

        depthVertShader = VK_NULL_HANDLE;
        depthFragShader = VK_NULL_HANDLE;
    }

    // ── EVSM shadow pipeline (writes moments via shadow_evsm.frag ──
    {
        auto shadowVertCode = FileReader::readFile("shaders/vegetation_shadow.vert.spv");
        auto shadowFragCode = FileReader::readFile("shaders/shadow_evsm.frag.spv");
        VkShaderModule shadowVertShader = app->createShaderModule(shadowVertCode);
        VkShaderModule shadowFragShader = app->createShaderModule(shadowFragCode);

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

        auto [shadowPipeline, shadowLayout] = app->createGraphicsPipeline(
            { shadowStages[0], shadowStages[1] },
            std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1]},
            attribDescs,
            setLayouts,
            &pushConstantRange,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            true,   // depthWrite
            true,   // colorWrite (EVSM moments)
            VK_COMPARE_OP_LESS,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false,
            std::vector<VkFormat>{VK_FORMAT_R32G32B32A32_SFLOAT},
            VK_FORMAT_D32_SFLOAT,
            false,  // noColorAttachment = false (EVSM moments)
            true    // depthBiasEnable
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
        billboardVBO.vertexBuffer = app->createVertexBuffer(verts);

        // 36 indices = 6 planes × 2 triangles × 3 indices
        std::vector<uint32_t> idx(36);
        for (int p = 0; p < 6; ++p) {
            int b = p * 4;
            int ib = p * 6;
            idx[ib + 0] = b + 0; idx[ib + 1] = b + 1; idx[ib + 2] = b + 2;
            idx[ib + 3] = b + 1; idx[ib + 4] = b + 3; idx[ib + 5] = b + 2;
        }
        billboardVBO.indexBuffer = app->createIndexBuffer(idx);
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
    for (auto& [id, _] : chunkBuffers) destroyInstanceBuffer(id);
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

    std::vector<VkBufferMemoryBarrier2> readBarriers;
    readBarriers.reserve(chunkBuffers.size() * 2);
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


void VegetationRenderer::drawShadow(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet shadowDescriptorSet, const glm::mat4& viewProj, const glm::vec3& cameraPos) {
    (void)viewProj; // GPU culling is dispatched by prepareCull() outside the render pass
    if (!app || vegetationShadowPipeline == VK_NULL_HANDLE) {
        if (!app) std::cerr << "[VEGETATION SHADOW DRAW ERROR] app is null!" << std::endl;
        if (vegetationShadowPipeline == VK_NULL_HANDLE) std::cerr << "[VEGETATION SHADOW DRAW ERROR] Shadow pipeline is VK_NULL_HANDLE!" << std::endl;
        return;
    }

    // Early-out when there are no chunks to draw: skip pipeline/descriptor
    // binding entirely. On RADV iGPUs, binding pipelines that reference
    // large texture arrays (via vegDescriptorSet) can trigger GPUVM faults
    // even when zero draw calls are issued.
    if (chunkBuffers.empty()) return;

    // Ensure vegetation descriptor set is present and up-to-date
    if (!ensureVegDescriptorSet(app)) {
        std::cerr << "[VEGETATION SHADOW DRAW ERROR] vegDescriptorSet not ready, skipping draw." << std::endl;
        return;
    }

    // Defensive checks: ensure both descriptor sets are valid before binding
    if (shadowDescriptorSet == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION SHADOW DRAW ERROR] shadowDescriptorSet is VK_NULL_HANDLE, skipping draw." << std::endl;
        return;
    }
    if (vegDescriptorSet == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION SHADOW DRAW ERROR] vegDescriptorSet is VK_NULL_HANDLE, skipping draw." << std::endl;
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationShadowPipeline);

    // Bind the shadow descriptor set at set 0 and vegetation descriptor set at set 1
    // shadowDescriptorSet contains the light-space UBO for shadow rendering
    VkDescriptorSet sets[2] = { shadowDescriptorSet, vegDescriptorSet };
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, 2, sets, 0, nullptr);

    // Push constants for shadow pass: same as regular draw but with wind disabled
    WindPushConstants pc{};
    pc.billboardScale = billboardScale;
    pc.windEnabled = -1.0f;  // Negative means shadow pass: disable wind and tighten impostor cutoff.
    pc.windTime = windTimeSeconds;
    pc.impostorDistance = impostorDistance; // skip far instances in shadow pass too (same as main pass)

    glm::vec2 windDir = windSettings.direction;
    const float len2 = windDir.x * windDir.x + windDir.y * windDir.y;
    if (len2 > 1e-8f) {
        const float invLen = 1.0f / std::sqrt(len2);
        windDir *= invLen;
    }

    pc.windDirAndStrength = glm::vec4(windDir.x, 0.0f, windDir.y, std::max(0.0f, windSettings.strength));
    pc.windNoise = glm::vec4(
        std::max(0.00001f, windSettings.baseFrequency),
        std::max(0.0f, windSettings.speed),
        std::max(0.00001f, windSettings.gustFrequency),
        std::max(0.0f, windSettings.gustStrength)
    );
    pc.windShape = glm::vec4(
        std::max(0.0f, windSettings.skewAmount),
        std::clamp(windSettings.trunkStiffness, 0.0f, 1.0f),
        std::max(0.001f, windSettings.noiseScale),
        std::max(0.0f, windSettings.verticalFlutter)
    );
    pc.windTurbulence = glm::vec4(std::max(0.0f, windSettings.turbulence), 0.0f, 0.0f, 0.0f);
    
    // Distance-based density: use camera position for LOD, NOT light position
    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = (distanceDensitySettings.enabled && minFactor < 1.0f)
        ? (-std::log(safeMinFactor) / (farDistance - nearDistance))
        : 0.0f;
    pc.densityParams = glm::vec4(distanceDensitySettings.enabled ? 1.0f : 0.0f, nearDistance, farDistance, minFactor);
    pc.cameraPosAndFalloff = glm::vec4(cameraPos, falloff);  // Camera pos, not light pos

    vkCmdPushConstants(
        commandBuffer,
        shadowPipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(WindPushConstants),
        &pc
    );

    // Draw consolidated via GPU culling only (no per-chunk fallback).
    // New chunks appear gradually as they are uploaded and consolidated.
    uint32_t sf = vegCullCurrentSlot;
    vkCmdBindIndexBuffer(commandBuffer, billboardVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    if (!vegConsolidationDirty && concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0 &&
        compactedCmdBuffers[sf].buffer != VK_NULL_HANDLE && visibleCountBuffers[sf].buffer != VK_NULL_HANDLE) {
        VkBuffer vbs[2] = { billboardVBO.vertexBuffer.buffer, concatenatedInstanceBuffer.buffer };
        VkDeviceSize offsets[2] = { 0, 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vbs, offsets);
        cmdDrawIndexedIndirectCount(commandBuffer, compactedCmdBuffers[sf].buffer, 0,
        visibleCountBuffers[sf].buffer, 0, vegNumChunks, sizeof(VkDrawIndexedIndirectCommand));
    }

    // ── Impostor depth pass ────────────────────────────────────────────────────
    if (impostorDepthPipeline != VK_NULL_HANDLE &&
        impostorDepthDescSet  != VK_NULL_HANDLE &&
        impostorDistance > 0.0f && impostorVBO.vertexBuffer.buffer != VK_NULL_HANDLE &&
        !chunkBuffers.empty()) {

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorDepthPipeline);

        VkDescriptorSet depthSets[2] = { shadowDescriptorSet, impostorDepthDescSet };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    impostorDepthPipelineLayout, 0, 2, depthSets, 0, nullptr);

        vkCmdPushConstants(commandBuffer, impostorDepthPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(WindPushConstants), &pc);

        vkCmdBindIndexBuffer(commandBuffer, impostorVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        VkBuffer impVbs[2] = { impostorVBO.vertexBuffer.buffer, VK_NULL_HANDLE };
        VkDeviceSize impOffsets[2] = { 0, 0 };
        if (!vegConsolidationDirty && concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
            impVbs[1] = concatenatedInstanceBuffer.buffer;
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, impVbs, impOffsets);
            uint32_t instOff = 0;
            for (auto& [chunkId, buf] : chunkBuffers) {
                (void)chunkId;
                if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
                vkCmdDrawIndexed(commandBuffer, 6, static_cast<uint32_t>(buf.count), 0, 0, instOff);
                instOff += buf.count;
            }
        }
    }
}

void VegetationRenderer::setImpostorData(VulkanApp* app,
                                          VkImageView albedoArray60,
                                          VkImageView normalArray60,
                                          VkSampler sampler,
                                          VkImageView depthArray60,
                                          VkBuffer captureInvVPBuf) {
    if (!app || albedoArray60 == VK_NULL_HANDLE || normalArray60 == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
    // Wait for all in-flight work to complete before recreating descriptor sets.
    // This prevents handle-reuse collisions between pending command buffers and
    // freshly-allocated descriptor set handles. setImpostorData is an init-time call.
    vkDeviceWaitIdle(app->getDevice());

    VkDevice device = app->getDevice();

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
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        info.bindingCount = numBindings;
        info.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &impostorDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDescSetLayout failed");
        app->resources.addDescriptorSetLayout(impostorDescSetLayout, "VegetationRenderer: impostorDescSetLayout");

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = hasImpostorDepth ? 3 : 2;
        uint32_t numPoolSizes = 1;
        if (hasImpostorDepth) {
            poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSizes[1].descriptorCount = 1;
            numPoolSizes = 2;
        }
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets       = 1;
        poolInfo.poolSizeCount = numPoolSizes;
        poolInfo.pPoolSizes    = poolSizes;
        poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &impostorDescPool) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDescPool failed");
        app->resources.addDescriptorPool(impostorDescPool, "VegetationRenderer: impostorDescPool");

        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool     = impostorDescPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts        = &impostorDescSetLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &impostorDescSet) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDescSet alloc failed");

        VkDescriptorImageInfo imgInfos[3]{};
        imgInfos[0] = { sampler, albedoArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        imgInfos[1] = { sampler, normalArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        if (hasImpostorDepth)
            imgInfos[2] = { sampler, depthArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        VkDescriptorBufferInfo bufInfo{};
        if (hasImpostorDepth) {
            bufInfo.buffer = captureInvVPBuf;
            bufInfo.offset = 0;
            bufInfo.range  = VK_WHOLE_SIZE;
        }

        VkWriteDescriptorSet ws[4]{};
        for (uint32_t i = 0; i < 2; ++i) {
            ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet          = impostorDescSet;
            ws[i].dstBinding      = i;
            ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ws[i].descriptorCount = 1;
            ws[i].pImageInfo      = &imgInfos[i];
        }
        uint32_t numWrites = 2;
        if (hasImpostorDepth) {
            ws[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[2].dstSet          = impostorDescSet;
            ws[2].dstBinding      = 2;
            ws[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ws[2].descriptorCount = 1;
            ws[2].pImageInfo      = &imgInfos[2];
            ws[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[3].dstSet          = impostorDescSet;
            ws[3].dstBinding      = 3;
            ws[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[3].descriptorCount = 1;
            ws[3].pBufferInfo     = &bufInfo;
            numWrites = 4;
        }
        vkUpdateDescriptorSets(device, numWrites, ws, 0, nullptr);
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
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        info.bindingCount = 2;
        info.pBindings    = depthBindings;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &impostorDepthDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDepthDescSetLayout failed");
        app->resources.addDescriptorSetLayout(impostorDepthDescSetLayout, "VegetationRenderer: impostorDepthDescSetLayout");

        VkDescriptorPoolSize depthPoolSizes[2]{};
        depthPoolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        depthPoolSizes[0].descriptorCount = 1;
        depthPoolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        depthPoolSizes[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets       = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes    = depthPoolSizes;
        poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &impostorDepthDescPool) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDepthDescPool failed");
        app->resources.addDescriptorPool(impostorDepthDescPool, "VegetationRenderer: impostorDepthDescPool");

        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool     = impostorDepthDescPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts        = &impostorDepthDescSetLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &impostorDepthDescSet) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDepthDescSet alloc failed");

        VkDescriptorImageInfo depthImgInfo{};
        depthImgInfo.sampler     = sampler;
        depthImgInfo.imageView   = depthArray60;
        depthImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = captureInvVPBuf;
        bufInfo.offset = 0;
        bufInfo.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet ws[2]{};
        ws[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[0].dstSet          = impostorDepthDescSet;
        ws[0].dstBinding      = 0;
        ws[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[0].descriptorCount = 1;
        ws[0].pImageInfo      = &depthImgInfo;
        ws[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[1].dstSet          = impostorDepthDescSet;
        ws[1].dstBinding      = 1;
        ws[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[1].descriptorCount = 1;
        ws[1].pBufferInfo     = &bufInfo;
        vkUpdateDescriptorSets(device, 2, ws, 0, nullptr);

        // ── Build impostor depth pipeline ────────────────────────────────
        auto depthVertCode = FileReader::readFile("shaders/impostors_depth.vert.spv");
        auto depthFragCode = FileReader::readFile("shaders/impostors_depth.frag.spv");

        VkShaderModule depthVertMod = app->createShaderModule(depthVertCode);
        VkShaderModule depthFragMod = app->createShaderModule(depthFragCode);

        VkPipelineShaderStageCreateInfo depthStages[2]{};
        depthStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depthStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        depthStages[0].module = depthVertMod;
        depthStages[0].pName  = "main";
        depthStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depthStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        depthStages[1].module = depthFragMod;
        depthStages[1].pName  = "main";

        VkVertexInputBindingDescription depthBindingDescs[2]{};
        depthBindingDescs[0] = { 0, sizeof(Vertex),       VK_VERTEX_INPUT_RATE_VERTEX   };
        depthBindingDescs[1] = { 1, sizeof(float) * 4,    VK_VERTEX_INPUT_RATE_INSTANCE };

        std::vector<VkDescriptorSetLayout> depthSetLayouts = {
            app->getDescriptorSetLayout(),
            impostorDepthDescSetLayout
        };
        VkPushConstantRange depthPCRange{};
        depthPCRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        depthPCRange.offset     = 0;
        depthPCRange.size       = sizeof(WindPushConstants);

        auto [depthPipe, depthLayout] = app->createGraphicsPipeline(
            { depthStages[0], depthStages[1] },
            std::vector<VkVertexInputBindingDescription>{ depthBindingDescs[0], depthBindingDescs[1] },
            {
                { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
                { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                              },
            },
            depthSetLayouts,
            &depthPCRange,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            true,  // depthWrite — write correct per-pixel depth to shadow map
            false, // colorWrite — depth-only pass
            VK_COMPARE_OP_LESS,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false,
            {},
            VK_FORMAT_D32_SFLOAT,
            true,  // noColorAttachment
            false  // depthBiasEnable (false to match color pass, avoiding precision mismatch with EQUAL compare)
        );
        impostorDepthPipeline       = depthPipe;
        impostorDepthPipelineLayout = depthLayout;

        if (impostorDepthPipeline == VK_NULL_HANDLE)
            std::cerr << "[VegetationRenderer] WARNING: impostor depth pipeline creation failed\n";
        else
            std::cerr << "[VegetationRenderer] Impostor depth pipeline created: " << (void*)impostorDepthPipeline << "\n";
    }

    // ── Build impostor color pipeline ───────────────────────────────────
    auto vertCode = FileReader::readFile("shaders/impostors.vert.spv");
    auto fragCode = FileReader::readFile("shaders/impostors.frag.spv");

    VkShaderModule vertShader = app->createShaderModule(vertCode);
    VkShaderModule fragShader = app->createShaderModule(fragCode);

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

    VkVertexInputBindingDescription bindingDescs[2]{};
    bindingDescs[0] = { 0, sizeof(Vertex),       VK_VERTEX_INPUT_RATE_VERTEX   };
    bindingDescs[1] = { 1, sizeof(float) * 4,    VK_VERTEX_INPUT_RATE_INSTANCE };

    std::vector<VkDescriptorSetLayout> impSetLayouts = {
        app->getDescriptorSetLayout(),
        impostorDescSetLayout
    };
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(WindPushConstants);

    auto [impPipeline, impLayout] = app->createGraphicsPipeline(
        { vertStage, fragStage },
        std::vector<VkVertexInputBindingDescription>{ bindingDescs[0], bindingDescs[1] },
        {
            { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
            { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                              },
        },
        impSetLayouts,
        &pcRange,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true,  // depthWrite — single-pass: impostor writes both color and depth
        true,  // colorWrite
        VK_COMPARE_OP_LESS, // LESS compare against depth prepass (solid + veg)
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );

    impostorPipeline       = impPipeline;
    impostorPipelineLayout = impLayout;

    if (impostorPipeline == VK_NULL_HANDLE)
        std::cerr << "[VegetationRenderer] WARNING: impostor pipeline creation failed\n";
    else
        std::cerr << "[VegetationRenderer] Impostor pipeline created: " << (void*)impostorPipeline << "\n";
}

void VegetationRenderer::draw(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet,
                              const glm::mat4& viewProj, const glm::vec3& cameraPos,
                              VkQueryPool queryPool,
                              uint32_t queryRealIndex,
                              uint32_t queryImpostorIndex) {
    (void)vegetationDescriptorSet;
    drawDepth(app, commandBuffer, viewProj, cameraPos);
    drawColor(app, commandBuffer, viewProj, cameraPos);
}

VegetationRenderer::WindPushConstants VegetationRenderer::buildWindPushConstants(const glm::vec3& cameraPos) const {
    WindPushConstants pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = windSettings.enabled ? 1.0f : 0.0f;
    pc.windTime           = windTimeSeconds;
    pc.impostorDistance   = impostorDistance;

    glm::vec2 windDir = windSettings.direction;
    const float len2 = windDir.x * windDir.x + windDir.y * windDir.y;
    if (len2 > 1e-8f) {
        const float invLen = 1.0f / std::sqrt(len2);
        windDir *= invLen;
    }
    pc.windDirAndStrength = glm::vec4(windDir.x, 0.0f, windDir.y, std::max(0.0f, windSettings.strength));
    pc.windNoise = glm::vec4(
        std::max(0.00001f, windSettings.baseFrequency),
        std::max(0.0f, windSettings.speed),
        std::max(0.00001f, windSettings.gustFrequency),
        std::max(0.0f, windSettings.gustStrength));
    pc.windShape = glm::vec4(
        std::max(0.0f, windSettings.skewAmount),
        std::clamp(windSettings.trunkStiffness, 0.0f, 1.0f),
        std::max(0.001f, windSettings.noiseScale),
        std::max(0.0f, windSettings.verticalFlutter));
    pc.windTurbulence = glm::vec4(std::max(0.0f, windSettings.turbulence), 0.0f, 0.0f, 0.0f);
    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = (distanceDensitySettings.enabled && minFactor < 1.0f)
        ? (-std::log(safeMinFactor) / (farDistance - nearDistance)) : 0.0f;
    pc.densityParams = glm::vec4(distanceDensitySettings.enabled ? 1.0f : 0.0f, nearDistance, farDistance, minFactor);
    pc.cameraPosAndFalloff = glm::vec4(cameraPos, falloff);
    return pc;
}

void VegetationRenderer::issueVegetationDraws(VkCommandBuffer cmd, VkPipelineLayout activeLayout, VkShaderStageFlags pushConstantStages, const WindPushConstants& pc) {
    uint32_t f = vegCullCurrentSlot;
    vkCmdPushConstants(cmd, activeLayout, pushConstantStages, 0, sizeof(WindPushConstants), &pc);
    if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE || billboardVBO.indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[2] = { billboardVBO.vertexBuffer.buffer, VK_NULL_HANDLE };
    VkDeviceSize offsets[2] = { 0, 0 };
    vkCmdBindIndexBuffer(cmd, billboardVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    if (!vegConsolidationDirty && concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0 &&
        compactedCmdBuffers[f].buffer != VK_NULL_HANDLE && visibleCountBuffers[f].buffer != VK_NULL_HANDLE) {
        vbs[1] = concatenatedInstanceBuffer.buffer;
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
        if (cmdDrawIndexedIndirectCount) {
            cmdDrawIndexedIndirectCount(cmd, compactedCmdBuffers[f].buffer, 0,
                visibleCountBuffers[f].buffer, 0, vegNumChunks, sizeof(VkDrawIndexedIndirectCommand));
        }
    }
}

void VegetationRenderer::issueImpostorDraws(VkCommandBuffer cmd, VkPipelineLayout activeLayout, VkShaderStageFlags pushConstantStages, const WindPushConstants& pc) {
    vkCmdPushConstants(cmd, activeLayout, pushConstantStages, 0, sizeof(WindPushConstants), &pc);
    if (impostorVBO.vertexBuffer.buffer == VK_NULL_HANDLE || impostorVBO.indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[2] = { impostorVBO.vertexBuffer.buffer, VK_NULL_HANDLE };
    VkDeviceSize offsets[2] = { 0, 0 };
    vkCmdBindIndexBuffer(cmd, impostorVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    if (!vegConsolidationDirty && concatenatedInstanceBuffer.buffer != VK_NULL_HANDLE && vegNumChunks > 0) {
        vbs[1] = concatenatedInstanceBuffer.buffer;
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
        uint32_t instOff = 0;
        for (auto& [chunkId, buf] : chunkBuffers) {
            (void)chunkId;
            if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) continue;
            vkCmdDrawIndexed(cmd, 6, static_cast<uint32_t>(buf.count), 0, 0, instOff);
            instOff += buf.count;
        }
    }
}

void VegetationRenderer::drawDepth(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::mat4& viewProj, const glm::vec3& cameraPos) {
    (void)viewProj;
    if (!app) return;
    if (chunkBuffers.empty()) return;
    if (billboardAlbedoView == VK_NULL_HANDLE || billboardNormalView == VK_NULL_HANDLE ||
        billboardOpacityView == VK_NULL_HANDLE || billboardArraySampler == VK_NULL_HANDLE) return;
    if (!ensureVegDescriptorSet(app)) return;
    VkDescriptorSet globalSet = app->getMainDescriptorSet();
    if (globalSet == VK_NULL_HANDLE || vegDescriptorSet == VK_NULL_HANDLE) return;
    WindPushConstants pc = buildWindPushConstants(cameraPos);
    VkDescriptorSet sets[2] = { globalSet, vegDescriptorSet };

    // Depth prepass
    if (vegetationDepthPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationDepthPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vegetationDepthPipelineLayout, 0, 2, sets, 0, nullptr);
        issueVegetationDraws(commandBuffer, vegetationDepthPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    }

}

void VegetationRenderer::drawColor(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::mat4& viewProj, const glm::vec3& cameraPos) {
    (void)viewProj;
    if (!app) return;
    if (chunkBuffers.empty()) return;
    if (billboardAlbedoView == VK_NULL_HANDLE || billboardNormalView == VK_NULL_HANDLE ||
        billboardOpacityView == VK_NULL_HANDLE || billboardArraySampler == VK_NULL_HANDLE) return;
    if (!ensureVegDescriptorSet(app)) return;
    VkDescriptorSet globalSet = app->getMainDescriptorSet();
    if (globalSet == VK_NULL_HANDLE || vegDescriptorSet == VK_NULL_HANDLE) return;
    WindPushConstants pc = buildWindPushConstants(cameraPos);
    VkDescriptorSet sets[2] = { globalSet, vegDescriptorSet };

    // Shading pass
    if (vegetationPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 2, sets, 0, nullptr);
        issueVegetationDraws(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    }
    // Impostor color pass
    if (impostorPipeline != VK_NULL_HANDLE &&
        impostorDescSet != VK_NULL_HANDLE && impostorDistance > 0.0f) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorPipeline);
        VkDescriptorSet impSets[2] = { globalSet, impostorDescSet };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    impostorPipelineLayout, 0, 2, impSets, 0, nullptr);
        issueImpostorDraws(commandBuffer, impostorPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    }
}

void VegetationRenderer::generateChunkInstances(NodeID chunkId,
                                               Buffer vertexBuffer, uint32_t vertexCount,
                                               Buffer indexBuffer, uint32_t indexCount,
                                               const glm::vec3& chunkCenter,
                                               uint32_t instancesPerTriangle, VulkanApp* app,
                                               uint32_t seed) {
    if (!app) return;

    if (indexCount < 3 || instancesPerTriangle == 0) {
        // No instances to generate; clear any previous chunk data.
        destroyInstanceBuffer(chunkId, app);
        return;
    }

    uint32_t triCount = indexCount / 3;
    uint32_t instanceCount = triCount * instancesPerTriangle;

    VkDevice device = app->getDevice();
    VkPhysicalDevice physicalDevice = app->getPhysicalDevice();

    // Create device-local storage/vertex buffer for instances (vec4 per-instance)
    VkBuffer instanceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    VkDeviceSize instanceBufferSize = static_cast<VkDeviceSize>(instanceCount) * sizeof(float) * 4; // vec4
    bufInfo.size = instanceBufferSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &instanceBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create instance storage buffer");
    }
    app->resources.addBuffer(instanceBuffer, "VegetationRenderer: instanceBuffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, instanceBuffer, &memReq);
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    uint32_t deviceLocalTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            deviceLocalTypeIndex = i;
            break;
        }
    }
    if (deviceLocalTypeIndex == UINT32_MAX) throw std::runtime_error("No suitable device local memory type for instance buffer");
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    {
        static constexpr VkDeviceSize kMin = 262144;
        const VkDeviceSize sz = memReq.size;
        allocInfo.allocationSize = (sz < kMin) ? kMin : (sz < 1048576 ? sz + 1 : sz);
    }
    allocInfo.memoryTypeIndex = deviceLocalTypeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &instanceMemory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate instance buffer memory");
    vkBindBufferMemory(device, instanceBuffer, instanceMemory, 0);
    app->resources.addDeviceMemory(instanceMemory, "VegetationRenderer: instanceMemory");

    Buffer indirect = app->createBuffer(sizeof(VkDrawIndexedIndirectCommand),
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkBuffer indirectBuffer = indirect.buffer;
    VkDeviceMemory indirectMemory = indirect.memory;

    VkDrawIndexedIndirectCommand drawCmd{};
    drawCmd.indexCount = 36;
    drawCmd.instanceCount = instanceCount;
    drawCmd.firstIndex = 0;
    drawCmd.vertexOffset = 0;
    drawCmd.firstInstance = 0;
    void* indirectData;
    if (vkMapMemory(device, indirectMemory, 0, sizeof(VkDrawIndexedIndirectCommand), 0, &indirectData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map vegetation indirect buffer");
    }
    std::memcpy(indirectData, &drawCmd, sizeof(VkDrawIndexedIndirectCommand));
    vkUnmapMemory(device, indirectMemory);

    // Dispatch compute to fill instanceBuffer asynchronously on vegetation queue
    VkFence fence = VK_NULL_HANDLE;
    uint32_t expected = app->generateVegetationInstancesComputeAsync(vertexBuffer.buffer, vertexCount, indexBuffer.buffer, indexCount, instancesPerTriangle, instanceBuffer, static_cast<uint32_t>(instanceBufferSize), &fence, seed, billboardCount);
    if (expected == 0 || fence == VK_NULL_HANDLE) {
        // Clean up partially created buffers if compute not dispatched
        if (app->resources.removeBuffer(instanceBuffer)) vkDestroyBuffer(device, instanceBuffer, nullptr);
        if (app->resources.removeDeviceMemory(instanceMemory)) vkFreeMemory(device, instanceMemory, nullptr);
        if (app->resources.removeBuffer(indirectBuffer)) vkDestroyBuffer(device, indirectBuffer, nullptr);
        if (app->resources.removeDeviceMemory(indirectMemory)) vkFreeMemory(device, indirectMemory, nullptr);
        return;
    }

    // Now that we have a valid fence, destroy any previous chunk buffers.
    // The fence signals after the previous frame's draw (same queue, earlier
    // submission), guaranteeing the old buffers are no longer in use.
    destroyInstanceBuffer(chunkId, app, fence);

    std::cout << "[VegetationRenderer::generateChunkInstances] async dispatched, expected = " << expected << " fence=" << (void*)fence << std::endl;

    // Wait for the compute dispatch to complete before returning.
    // RADV GPUVM faults (TCP read permission) occur when too many
    // concurrent compute dispatches are in flight, even with correct
    // synchronization.  Serializing eliminates the concurrency.
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Insert the instance buffer into visible maps immediately (fence
    // already signaled).
    InstanceBuffer ibuf;
    ibuf.buffer = instanceBuffer;
    ibuf.memory = instanceMemory;
    ibuf.indirectBuffer = indirectBuffer;
    ibuf.indirectMemory = indirectMemory;
    ibuf.center = chunkCenter;
    ibuf.count = expected;
    chunkBuffers[chunkId] = ibuf;
    chunkInstanceCounts[chunkId] = expected;
    std::cout << "[VegetationRenderer] chunk " << (unsigned long long)chunkId << " instances ready: " << expected << std::endl;

    // Transfer input buffers (vertex/index) — the fence is already signaled
    // (vkWaitForFences above), so destroy them immediately.
    VkDevice dev = device;
    if (vertexBuffer.buffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(vertexBuffer.buffer)) vkDestroyBuffer(dev, vertexBuffer.buffer, nullptr);
    }
    if (vertexBuffer.memory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(vertexBuffer.memory)) vkFreeMemory(dev, vertexBuffer.memory, nullptr);
    }
    if (indexBuffer.buffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(indexBuffer.buffer)) vkDestroyBuffer(dev, indexBuffer.buffer, nullptr);
    }
    if (indexBuffer.memory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(indexBuffer.memory)) vkFreeMemory(dev, indexBuffer.memory, nullptr);
    }
}

// ── CPU-side instance generation ─────────────────────────────────────────────
// Replicates vegetation_instance_gen.comp logic on the CPU.  Avoids GPUVM
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

        std::vector<float> instanceData(instanceCount * 4, 0.0f);

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
            if (glm::abs(fn.y) <= 0.5f * glm::length(fn)) {
                for (uint32_t s = 0; s < pc.instancesPerTriangle; ++s) {
                    const uint32_t oi = (tri * pc.instancesPerTriangle + s) * 4;
                    instanceData[oi + 3] = -1.0f;
                }
                continue;
            }

            const glm::vec3 tc = (v0 + v1 + v2) / 3.0f;
            const uint32_t tch = posHash(tc);

            for (uint32_t s = 0; s < pc.instancesPerTriangle; ++s) {
                uint32_t rng = pc.seed ^ tch ^ (tri * 2654435761u) ^ (s * 19349663u);
                float u = randFloat(rng);
                float v = randFloat(rng);
                if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                float w = 1.0f - u - v;
                glm::vec3 pos = u * v0 + v * v1 + w * v2;

                uint32_t bi = std::min(
                    uint32_t(biomeNoise(glm::vec2(pos.x, pos.z)) * float(billboardCnt)),
                    billboardCnt - 1u);

                uint32_t rs = pc.seed ^ posHash(pos);
                float rf = randFloat(rs);

                const uint32_t oi = (tri * pc.instancesPerTriangle + s) * 4;
                instanceData[oi + 0] = pos.x;
                instanceData[oi + 1] = pos.y;
                instanceData[oi + 2] = pos.z;
                instanceData[oi + 3] = float(bi) + rf;
    }

}

        const VkDeviceSize bufSize = instanceData.size() * sizeof(float);

        // Staging buffer: host-visible, filled by CPU.
        Buffer stagingInst = app->createBuffer(bufSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        vkMapMemory(app->getDevice(), stagingInst.memory, 0, bufSize, 0, &mapped);
        std::memcpy(mapped, instanceData.data(), size_t(bufSize));
        vkUnmapMemory(app->getDevice(), stagingInst.memory);

        // Device-local instance buffer: GPU reads via vertex input.
        // On RADV iGPUs, vertex reads go through TCP (Texture Cache/Pipe),
        // and host-visible pages lack TCP-read permission → GPUVM fault.
        Buffer instBuf = app->createBuffer(bufSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Indirect buffer: device-local, avoids same TCP-read issue.
        Buffer indirect = app->createBuffer(sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Staging for indirect draw command.
        Buffer stagingIndirect = app->createBuffer(sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDrawIndexedIndirectCommand drawCmd{};
        drawCmd.indexCount    = 36;
        drawCmd.instanceCount = instanceCount;
        drawCmd.firstIndex    = 0;
        drawCmd.vertexOffset  = 0;
        drawCmd.firstInstance = 0;
        void* idata = nullptr;
        vkMapMemory(app->getDevice(), stagingIndirect.memory, 0, sizeof(VkDrawIndexedIndirectCommand), 0, &idata);
        std::memcpy(idata, &drawCmd, sizeof(VkDrawIndexedIndirectCommand));
        vkUnmapMemory(app->getDevice(), stagingIndirect.memory);

        pendingBatch.push_back({ stagingInst, instBuf, stagingIndirect, indirect,
                                 bufSize, pc.chunkId, instanceCount,
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
        app->deferDestroyUntilFence(fence, [this, dev = app->getDevice(), app,
                                             batch = std::move(batch)]() {
            for (auto& c : batch) {
                if (app->resources.removeBuffer(c.stagingInst.buffer))
                    vkDestroyBuffer(dev, c.stagingInst.buffer, nullptr);
                if (app->resources.removeDeviceMemory(c.stagingInst.memory))
                    vkFreeMemory(dev, c.stagingInst.memory, nullptr);
                if (app->resources.removeBuffer(c.stagingIndirect.buffer))
                    vkDestroyBuffer(dev, c.stagingIndirect.buffer, nullptr);
                if (app->resources.removeDeviceMemory(c.stagingIndirect.memory))
                    vkFreeMemory(dev, c.stagingIndirect.memory, nullptr);

                destroyInstanceBuffer(c.chunkId, app);

                InstanceBuffer ibuf;
                ibuf.buffer         = c.instBuf.buffer;
                ibuf.memory         = c.instBuf.memory;
                ibuf.indirectBuffer = c.indirect.buffer;
                ibuf.indirectMemory = c.indirect.memory;
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

    // Destroy old buffers immediately.  The CPU-generation path runs on the
    // render thread via processPendingChunks() which is called from draw()
    // before command buffer submission, so no in-flight work references them.
    // The GPU path (generateChunkInstances) waits synchronously on the fence
    // before returning, so the GPU is done by the time we reach here.
    VkDevice dev = app->getDevice();
    if (old.buffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(old.buffer)) vkDestroyBuffer(dev, old.buffer, nullptr);
    }
    if (old.memory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(old.memory)) vkFreeMemory(dev, old.memory, nullptr);
    }
    if (old.indirectBuffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(old.indirectBuffer)) vkDestroyBuffer(dev, old.indirectBuffer, nullptr);
    }
    if (old.indirectMemory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(old.indirectMemory)) vkFreeMemory(dev, old.indirectMemory, nullptr);
    }
}

// Ensure we clear the stored app pointer on cleanup
// (cleanup() already clears handles; set appPtr to nullptr here)
