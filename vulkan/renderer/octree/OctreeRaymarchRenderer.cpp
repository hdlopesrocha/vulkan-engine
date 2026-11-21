#include "OctreeRaymarchRenderer.hpp"

#include "../RendererUtils.hpp"
#include "../DescriptorAllocator.hpp"
#include "../DescriptorWriter.hpp"
#include "../CommandBufferState.hpp"
#include "../../pipeline/ShaderStage.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace {
constexpr size_t kNodeBytes = sizeof(OctreeRaymarchNode);
constexpr size_t kPayloadBytes = sizeof(OctreeRaymarchPayload);
} // namespace

void OctreeRaymarchRenderer::init(VulkanApp* app) {
    if (!app) return;

    // Fit the node arrays to the device's storage-buffer descriptor range.
    // The shader bounds its own accesses (child indices are host-generated),
    // but the descriptor range must cover every byte the table can use.
    const size_t byRange = static_cast<size_t>(app->getMaxStorageBufferRange() / kNodeBytes);
    maxNodes_ = std::min({NODE_CAP, byRange});
    if (maxNodes_ == 0) {
        std::cerr << "[OctreeRaymarchRenderer] device storage-buffer range is too small; "
                     "SDF raymarch disabled" << std::endl;
        return;
    }
    app_ = app;

    DescriptorAllocator descAlloc{app->getDevice(), app};

    // Set=2: binding 0 = node table, binding 1 = layer roots, binding 2 =
    // derived QEF payload. Bindings 0 and 2 are read by the fragment march and
    // written by the payload derive compute pass, so their stage masks span
    // both stages; binding 1 stays fragment-only.
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t b = 0; b < 3; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorCount = 1;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[0].stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
    setLayout_ = descAlloc.createLayout(bindings, 3, 0, nullptr,
                                        "OctreeRaymarchRenderer: set2 layout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3u * RAYMARCH_FRAMES;
    VkDescriptorPool pool = descAlloc.createPool(
        &poolSize, 1, RAYMARCH_FRAMES, 0, "OctreeRaymarchRenderer: pool");
    descAlloc.allocateSets(pool, setLayout_, RAYMARCH_FRAMES, sets_.data(),
                           "OctreeRaymarchRenderer: set");

    // The per-frame node/root/payload buffers are allocated lazily in upload():
    // the raymarch mode is off by default and a slot costs ~64 MB at the full
    // cap, so nothing is reserved until a table actually needs publishing.

    // Fullscreen raymarch pipeline. Set order matches the shader: 0 = main
    // scene, 1 = brush params (PAINT/REMOVE tint), 2 = octree node table.
    if (app->getDescriptorSetLayout() == VK_NULL_HANDLE ||
        app->getBrushParamsDescriptorSetLayout() == VK_NULL_HANDLE) {
        std::cerr << "[OctreeRaymarchRenderer] scene descriptor layouts unavailable; "
                     "SDF raymarch disabled" << std::endl;
        return;
    }

    try {
        ShaderStage vertStage(
            app->getOrCreateShaderModule("shaders/renderer/Fullscreen.vert.spv"),
            VK_SHADER_STAGE_VERTEX_BIT);
        ShaderStage fragStage(
            app->getOrCreateShaderModule("shaders/renderer/octree/OctreeRaymarch.frag.spv"),
            VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayout> setLayouts = {
            app->getDescriptorSetLayout(),
            app->getBrushParamsDescriptorSetLayout(),
            setLayout_
        };

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        GraphicsPipelineConfig cfg{};
        cfg.cullMode = VK_CULL_MODE_NONE;
        cfg.depthTestEnable = true;
        cfg.depthWriteEnable = true;
        cfg.depthCompareOp = VK_COMPARE_OP_LESS;
        cfg.colorFormats = { app->getSwapchainImageFormat() };
        cfg.depthFormat = VK_FORMAT_D32_SFLOAT;

        auto [pipe, layout] = app->createGraphicsPipeline(
            { vertStage.info, fragStage.info },
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            setLayouts, &pushRange, cfg);

        pipeline_ = pipe;
        pipelineLayout_ = layout;

        // Hardware-RT twin: identical march with the inline reflection /
        // contact-shadow paths the raster solid shader uses. Built in its own
        // try/catch AFTER the base pipeline so a missing/stale RT module only
        // disables the RT variant (draw() falls back to the base one) instead
        // of disabling the whole raymarch path.
        if (app->rayTracingEnabled()) {
            try {
                ShaderStage fragRtStage(
                    app->getOrCreateShaderModule("shaders/renderer/octree/OctreeRaymarchRT.frag.spv"),
                    VK_SHADER_STAGE_FRAGMENT_BIT);
                pipelineRt_ = app->createGraphicsPipelineWithLayout(
                    { vertStage.info, fragRtStage.info },
                    std::vector<VkVertexInputBindingDescription>{},
                    std::vector<VkVertexInputAttributeDescription>{},
                    setLayouts, &pushRange, cfg, pipelineLayout_);
                fragRtStage.info.module = VK_NULL_HANDLE;
            } catch (const std::exception& e) {
                std::cerr << "[OctreeRaymarchRenderer] RT variant unavailable ("
                          << e.what() << "); using the non-RT shading fallback."
                          << std::endl;
                pipelineRt_ = VK_NULL_HANDLE;
            }
        }

        // The app owns the modules; do not destroy them with the stage holder.
        vertStage.info.module = VK_NULL_HANDLE;
        fragStage.info.module = VK_NULL_HANDLE;
    } catch (const std::exception& e) {
        std::cerr << "[OctreeRaymarchRenderer] shaders not available (" << e.what()
                  << "); SDF raymarch disabled." << std::endl;
    }

    // Once-per-revision payload derive pass (OctreeDerive.comp). It reuses the
    // set-2 layout (sets 0/1 are declared only for index compatibility with the
    // shader's set=2 binding and are never accessed) and carries an 8-byte
    // compute push constant. A missing module or pipeline is non-fatal: the
    // fragment march keeps its QEF / fixed-0.5 fallback.
    if (setLayout_ != VK_NULL_HANDLE) {
        try {
            VkShaderModule deriveModule =
                app->getOrCreateShaderModule("shaders/computation/octree/OctreeDerive.comp.spv");
            if (deriveModule == VK_NULL_HANDLE) {
                std::cerr << "[OctreeRaymarchRenderer] missing "
                             "shaders/computation/octree/OctreeDerive.comp.spv; "
                             "using the fragment fallback." << std::endl;
            } else {
                VkPushConstantRange derivePush{};
                derivePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                derivePush.offset = 0;
                derivePush.size = 8;

                std::vector<VkDescriptorSetLayout> deriveSetLayouts = {
                    app->getDescriptorSetLayout(),
                    app->getBrushParamsDescriptorSetLayout(),
                    setLayout_
                };
                VkPipelineLayoutCreateInfo layoutInfo{};
                layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                layoutInfo.setLayoutCount = static_cast<uint32_t>(deriveSetLayouts.size());
                layoutInfo.pSetLayouts = deriveSetLayouts.data();
                layoutInfo.pushConstantRangeCount = 1;
                layoutInfo.pPushConstantRanges = &derivePush;

                if (vkCreatePipelineLayout(app->getDevice(), &layoutInfo, nullptr,
                                           &deriveLayout_) != VK_SUCCESS) {
                    std::cerr << "[OctreeRaymarchRenderer] failed to create the derive "
                                 "pipeline layout; using the fragment fallback."
                              << std::endl;
                    deriveLayout_ = VK_NULL_HANDLE;
                } else {
                    app->resources.addPipelineLayout(
                        deriveLayout_, "OctreeRaymarchRenderer: deriveLayout");

                    VkComputePipelineCreateInfo info{};
                    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.stage.module = deriveModule;
                    info.stage.pName = "main";
                    info.layout = deriveLayout_;
                    if (vkCreateComputePipelines(app->getDevice(), app->getPipelineCache(),
                                                 1, &info, nullptr,
                                                 &derivePipeline_) != VK_SUCCESS) {
                        std::cerr << "[OctreeRaymarchRenderer] failed to create the derive "
                                     "pipeline; using the fragment fallback." << std::endl;
                        derivePipeline_ = VK_NULL_HANDLE;
                    } else {
                        app->resources.addPipeline(
                            derivePipeline_, "OctreeRaymarchRenderer: derivePipeline");
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[OctreeRaymarchRenderer] derive pass unavailable ("
                      << e.what() << "); using the fragment fallback." << std::endl;
            derivePipeline_ = VK_NULL_HANDLE;
        }
    }
}

void OctreeRaymarchRenderer::cleanup(VulkanApp* app) {
    if (app) {
        for (auto& buf : nodeBuffers_)
            if (buf.buffer != VK_NULL_HANDLE) app->destroyBuffer(buf);
        for (auto& buf : rootBuffers_)
            if (buf.buffer != VK_NULL_HANDLE) app->destroyBuffer(buf);
        for (auto& buf : payloadBuffers_)
            if (buf.buffer != VK_NULL_HANDLE) app->destroyBuffer(buf);
        // The derive pipeline/layout are registered with VulkanResourceManager
        // at creation; remove before destroying so shutdown does not free them
        // twice (same contract as RayTracingResources).
        if (derivePipeline_ != VK_NULL_HANDLE) {
            if (app->resources.removePipeline(derivePipeline_))
                vkDestroyPipeline(app->getDevice(), derivePipeline_, nullptr);
        }
        if (deriveLayout_ != VK_NULL_HANDLE) {
            if (app->resources.removePipelineLayout(deriveLayout_))
                vkDestroyPipelineLayout(app->getDevice(), deriveLayout_, nullptr);
        }
    }
    derivePipeline_ = VK_NULL_HANDLE;
    deriveLayout_ = VK_NULL_HANDLE;
    nodeBuffers_.fill(Buffer{});
    rootBuffers_.fill(Buffer{});
    payloadBuffers_.fill(Buffer{});
    sets_.fill(VK_NULL_HANDLE);
    uploadedVersion_.fill(0);
    rootCounts_.fill(0);
    nodeCounts_.fill(0);
    needsDerive_.fill(false);
    app_ = nullptr;
    // The layout/pool/pipeline are registered with VulkanResourceManager at
    // creation and destroyed by it at shutdown (same contract as the other
    // renderers).
    setLayout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    pipelineRt_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    rtShadingEnabled_ = false;
    maxNodes_ = 0;
}

bool OctreeRaymarchRenderer::ensureSlotBuffers(uint32_t fi) {
    if (!app_ || fi >= RAYMARCH_FRAMES) return false;
    if (nodeBuffers_[fi].buffer != VK_NULL_HANDLE &&
        rootBuffers_[fi].buffer != VK_NULL_HANDLE &&
        payloadBuffers_[fi].buffer != VK_NULL_HANDLE)
        return true;
    if (sets_[fi] == VK_NULL_HANDLE) return false;

    // Node table and roots are host-visible and persistently mapped:
    // MyApp::update() rewrites the slot after its fence wait, and draw() records
    // the HOST -> FRAGMENT barrier. The QEF payload is device-local and is
    // written only by OctreeDerive.comp; the fragment reads it after the
    // derive's COMPUTE -> FRAGMENT barrier. zeroInit=false is safe because the
    // payload flag is only set by the derive invocation that writes the entry.
    nodeBuffers_[fi] = app_->createBuffer(
        static_cast<VkDeviceSize>(maxNodes_ * kNodeBytes),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    rootBuffers_[fi] = app_->createBuffer(
        static_cast<VkDeviceSize>(ROOT_CAP * sizeof(OctreeRaymarchRoot)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Without a derive pipeline the flag is never set and the fragment keeps
    // its QEF fallback, so a one-entry dummy keeps the descriptor valid without
    // reserving the full payload.
    const size_t payloadNodes =
        (derivePipeline_ != VK_NULL_HANDLE && deriveLayout_ != VK_NULL_HANDLE)
            ? maxNodes_ : 1;
    payloadBuffers_[fi] = app_->createBuffer(
        static_cast<VkDeviceSize>(payloadNodes * kPayloadBytes),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);

    if (nodeBuffers_[fi].buffer == VK_NULL_HANDLE ||
        rootBuffers_[fi].buffer == VK_NULL_HANDLE ||
        payloadBuffers_[fi].buffer == VK_NULL_HANDLE) {
        std::cerr << "[OctreeRaymarchRenderer] failed to create per-frame table resources"
                  << std::endl;
        if (nodeBuffers_[fi].buffer != VK_NULL_HANDLE) app_->destroyBuffer(nodeBuffers_[fi]);
        if (rootBuffers_[fi].buffer != VK_NULL_HANDLE) app_->destroyBuffer(rootBuffers_[fi]);
        if (payloadBuffers_[fi].buffer != VK_NULL_HANDLE) app_->destroyBuffer(payloadBuffers_[fi]);
        nodeBuffers_[fi] = Buffer{};
        rootBuffers_[fi] = Buffer{};
        payloadBuffers_[fi] = Buffer{};
        return false;
    }

    DescriptorWriter(app_->getDevice())
        .writeBuffer(sets_[fi], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     nodeBuffers_[fi].buffer, 0,
                     app_->getClampedWholeSizeRange(nodeBuffers_[fi].buffer))
        .writeBuffer(sets_[fi], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     rootBuffers_[fi].buffer, 0,
                     app_->getClampedWholeSizeRange(rootBuffers_[fi].buffer))
        .writeBuffer(sets_[fi], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     payloadBuffers_[fi].buffer, 0,
                     app_->getClampedWholeSizeRange(payloadBuffers_[fi].buffer))
        .flush();
    return true;
}

void OctreeRaymarchRenderer::upload(uint32_t frameIndex, const OctreeRaymarchTable& table) {
    const uint32_t fi = frameIndex % RAYMARCH_FRAMES;
    if (uploadedVersion_[fi] == table.version) return;
    if (!ensureSlotBuffers(fi)) return;
    if (nodeBuffers_[fi].mappedData == nullptr || rootBuffers_[fi].mappedData == nullptr)
        return;

    const size_t nodeCount = std::min(table.nodes.size(), maxNodes_);
    if (nodeCount > 0) {
        memcpy(nodeBuffers_[fi].mappedData, table.nodes.data(), nodeCount * kNodeBytes);
        // The host records carry the raw brush/lod fields only. Ask
        // recordDerive() to run the payload pass before this slot's next march;
        // a version match returns early above and never re-derives.
        nodeCounts_[fi] = static_cast<uint32_t>(nodeCount);
        needsDerive_[fi] = true;
    } else {
        nodeCounts_[fi] = 0;
        needsDerive_[fi] = false;
    }

    const size_t rootCount = std::min(table.layerRoots.size(), ROOT_CAP);
    if (rootCount > 0) {
        memcpy(rootBuffers_[fi].mappedData, table.layerRoots.data(),
               rootCount * sizeof(OctreeRaymarchRoot));
    }

    rootCounts_[fi] = static_cast<uint32_t>(rootCount);
    uploadedVersion_[fi] = table.version;
}

bool OctreeRaymarchRenderer::hasTable(uint32_t frameIndex) const {
    const uint32_t fi = frameIndex % RAYMARCH_FRAMES;
    // Requires a usable pipeline/pipeline layout too: if shader loading failed
    // the caller must keep the raster path (and its depth prepass), otherwise
    // the solid depth target would stay cleared.
    return pipeline_ != VK_NULL_HANDLE && pipelineLayout_ != VK_NULL_HANDLE &&
           sets_[fi] != VK_NULL_HANDLE && rootCounts_[fi] > 0;
}

void OctreeRaymarchRenderer::recordHostWriteBarrier(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;
    // The node buffers are persistently mapped and rewritten on the host in
    // update(), after this frame slot's fence wait. Publish those writes to the
    // fragment shader that samples them later this frame. Coherent mapped
    // memory needs no explicit flush.
    VkMemoryBarrier2 hostBarrier{};
    hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    hostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    hostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &hostBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void OctreeRaymarchRenderer::recordDerive(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (cmd == VK_NULL_HANDLE) return;
    const uint32_t fi = frameIndex % RAYMARCH_FRAMES;

    // Fallback: no derive pass for this slot (pipeline unavailable, no new
    // revision, or an empty table). Publish the host writes straight to the
    // fragment stage exactly as before.
    if (derivePipeline_ == VK_NULL_HANDLE || deriveLayout_ == VK_NULL_HANDLE ||
        sets_[fi] == VK_NULL_HANDLE || payloadBuffers_[fi].buffer == VK_NULL_HANDLE ||
        !needsDerive_[fi] || nodeCounts_[fi] == 0) {
        recordHostWriteBarrier(cmd);
        return;
    }

    // The node table is persistently mapped and rewritten on the host in
    // update(), after this frame slot's fence wait. Coherent mapped memory
    // needs no explicit flush. Two destination scopes share the host writes:
    // the derive pass (node-table read + payload write) and the fragment
    // march, which still reads the host-written root buffer.
    VkMemoryBarrier2 hostBarriers[2]{};
    hostBarriers[0].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    hostBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostBarriers[0].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    hostBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    hostBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    hostBarriers[1].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    hostBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostBarriers[1].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    hostBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    hostBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo hostDep{};
    hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    hostDep.memoryBarrierCount = 2;
    hostDep.pMemoryBarriers = hostBarriers;
    vkCmdPipelineBarrier2(cmd, &hostDep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, derivePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, deriveLayout_,
                            2, 1, &sets_[fi], 0, nullptr);

    struct {
        uint32_t nodeCount;
        uint32_t flags;
    } derivePush{nodeCounts_[fi], 0u};
    vkCmdPushConstants(cmd, deriveLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(derivePush), &derivePush);
    vkCmdDispatch(cmd, (nodeCounts_[fi] + 63u) / 64u, 1u, 1u);

    // Publish the derived payload (meta.x / hsv1.z / meta.w) to the fragment
    // march that follows in this command buffer.
    VkMemoryBarrier2 deriveBarrier{};
    deriveBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    deriveBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    deriveBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    deriveBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    deriveBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo deriveDep{};
    deriveDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    deriveDep.memoryBarrierCount = 1;
    deriveDep.pMemoryBarriers = &deriveBarrier;
    vkCmdPipelineBarrier2(cmd, &deriveDep);

    needsDerive_[fi] = false;
}

void OctreeRaymarchRenderer::draw(VkCommandBuffer cmd, VulkanApp* app,
                                  uint32_t frameIndex, uint32_t width,
                                  uint32_t height, VkDescriptorSet mainSet,
                                  VkDescriptorSet brushSet) {
    const uint32_t fi = frameIndex % RAYMARCH_FRAMES;
    if (!app || cmd == VK_NULL_HANDLE || width == 0 || height == 0 ||
        pipeline_ == VK_NULL_HANDLE || sets_[fi] == VK_NULL_HANDLE ||
        rootCounts_[fi] == 0 || mainSet == VK_NULL_HANDLE ||
        brushSet == VK_NULL_HANDLE) {
        return;
    }

    // Note: the caller records recordDerive() before beginning the solid
    // color pass (host barriers and the derive dispatch are illegal inside a
    // render scope).

    const VkPipeline activePipeline =
        (rtShadingEnabled_ && pipelineRt_ != VK_NULL_HANDLE) ? pipelineRt_ : pipeline_;
    if (cmdState) cmdState->bindGraphicsPipeline(cmd, activePipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);

    // The solid pass helper already set this, but the sky renderer between
    // them may have changed it: pin the viewport/scissor to the solid target
    // this pass writes (the pipeline uses dynamic viewport/scissor).
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkDescriptorSet bindSets[3] = { mainSet, brushSet, sets_[fi] };
    if (cmdState) {
        cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout_, 0, 3, bindSets);
    } else {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout_, 0, 3, bindSets, 0, nullptr);
    }

    PushConstants pc{};
    pc.viewport = glm::vec4(static_cast<float>(width),
                            static_cast<float>(height), 0.0f, 0.0f);
    pc.rootCount = rootCounts_[fi];
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}
