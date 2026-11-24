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
    revisions_.reserve(4);

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

    // The revision buffers are allocated lazily in upload(): a slot costs
    // ~64 MB at the full cap, so nothing is reserved until a table needs it.

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
        for (auto& revision : revisions_)
            destroyRevisionBuffers(revision);
        revisions_.clear();
        // The derive pipeline/layout are registered with VulkanResourceManager
        // at creation; remove before destroying so shutdown does not free them
        // twice.
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
    sets_.fill(VK_NULL_HANDLE);
    slotRevision_.fill(-1);
    app_ = nullptr;
    // The layout/pool/pipeline are registered with VulkanResourceManager at
    // creation and destroyed by it at shutdown (same contract as the other
    // renderers).
    setLayout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    maxNodes_ = 0;
}

bool OctreeRaymarchRenderer::bindSlotRevision(uint32_t fi, int revision) {
    if (revision < 0 || static_cast<size_t>(revision) >= revisions_.size())
        return false;
    if (slotRevision_[fi] == revision) return true;

    SharedRevision& rev = revisions_[revision];
    if (rev.node.buffer == VK_NULL_HANDLE || rev.root.buffer == VK_NULL_HANDLE ||
        rev.payload.buffer == VK_NULL_HANDLE)
        return false;

    // Descriptor updates happen on the frame thread after this slot's fence,
    // so the old revision cannot still be read by this slot. Other slots keep
    // their own user count; the last release destroys the buffers.
    DescriptorWriter(app_->getDevice())
        .writeBuffer(sets_[fi], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     rev.node.buffer, 0,
                     app_->getClampedWholeSizeRange(rev.node.buffer))
        .writeBuffer(sets_[fi], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     rev.root.buffer, 0,
                     app_->getClampedWholeSizeRange(rev.root.buffer))
        .writeBuffer(sets_[fi], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     rev.payload.buffer, 0,
                     app_->getClampedWholeSizeRange(rev.payload.buffer))
        .flush();

    if (slotRevision_[fi] >= 0) {
        if (static_cast<size_t>(slotRevision_[fi]) < revisions_.size()) {
            SharedRevision& old = revisions_[slotRevision_[fi]];
            if (old.users > 0) --old.users;
            if (old.users == 0) destroyRevisionBuffers(old);
        }
        slotRevision_[fi] = -1;
    }
    ++rev.users;
    slotRevision_[fi] = revision;
    return true;
}

void OctreeRaymarchRenderer::releaseSlotRevision(uint32_t fi) {
    if (fi >= RAYMARCH_FRAMES || slotRevision_[fi] < 0 ||
        static_cast<size_t>(slotRevision_[fi]) >= revisions_.size())
        return;
    const int index = slotRevision_[fi];
    slotRevision_[fi] = -1;
    SharedRevision& rev = revisions_[index];
    if (rev.users > 0) --rev.users;
    if (rev.users == 0) destroyRevisionBuffers(rev);
}

int OctreeRaymarchRenderer::acquireRevision() {
    for (size_t i = 0; i < revisions_.size(); ++i) {
        if (revisions_[i].users == 0) {
            destroyRevisionBuffers(revisions_[i]);
            return static_cast<int>(i);
        }
    }
    revisions_.emplace_back();
    return static_cast<int>(revisions_.size() - 1);
}

void OctreeRaymarchRenderer::destroyRevisionBuffers(SharedRevision& revision) {
    if (revision.version != 0)
        std::cout << "[OctreeRaymarchRenderer] revision " << revision.version
                  << " released" << std::endl;
    if (app_) {
        if (revision.node.buffer != VK_NULL_HANDLE) app_->destroyBuffer(revision.node);
        if (revision.root.buffer != VK_NULL_HANDLE) app_->destroyBuffer(revision.root);
        if (revision.payload.buffer != VK_NULL_HANDLE) app_->destroyBuffer(revision.payload);
    }
    revision.node = Buffer{};
    revision.root = Buffer{};
    revision.payload = Buffer{};
    revision.version = 0;
    revision.nodeCount = 0;
    revision.rootCount = 0;
    revision.users = 0;
    revision.derived = false;
}

void OctreeRaymarchRenderer::upload(uint32_t frameIndex, const OctreeRaymarchTable& table) {
    if (!app_) return;
    const uint32_t fi = frameIndex % RAYMARCH_FRAMES;
    if (table.nodes.empty() || table.layerRoots.empty()) {
        releaseSlotRevision(fi);
        return;
    }

    const int current = slotRevision_[fi];
    if (current >= 0 && revisions_[current].version == table.version) return;

    // Reuse an immutable revision another slot already displays; otherwise
    // publish a new one. The current slot's old revision is released before
    // allocating, so a new revision never peaks above the three-slot budget.
    int revision = -1;
    for (size_t i = 0; i < revisions_.size(); ++i) {
        if (revisions_[i].version == table.version &&
            revisions_[i].node.buffer != VK_NULL_HANDLE) {
            revision = static_cast<int>(i);
            break;
        }
    }

    if (revision < 0) {
        releaseSlotRevision(fi);
        revision = acquireRevision();
        SharedRevision& rev = revisions_[revision];
        const size_t nodeCount = std::min(table.nodes.size(), maxNodes_);
        const size_t rootCount = std::min(table.layerRoots.size(), ROOT_CAP);
        // Node table and roots stay host-visible and persistently mapped; only
        // the QEF payload is device-local (written by OctreeDerive.comp).
        rev.node = app_->createBuffer(
            static_cast<VkDeviceSize>(nodeCount * kNodeBytes),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        rev.root = app_->createBuffer(
            static_cast<VkDeviceSize>(rootCount * sizeof(OctreeRaymarchRoot)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        const size_t payloadNodes =
            (derivePipeline_ != VK_NULL_HANDLE && deriveLayout_ != VK_NULL_HANDLE)
                ? nodeCount : 1;
        rev.payload = app_->createBuffer(
            static_cast<VkDeviceSize>(payloadNodes * kPayloadBytes),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
        if (rev.node.buffer == VK_NULL_HANDLE || rev.root.buffer == VK_NULL_HANDLE ||
            rev.payload.buffer == VK_NULL_HANDLE ||
            rev.node.mappedData == nullptr || rev.root.mappedData == nullptr) {
            std::cerr << "[OctreeRaymarchRenderer] failed to create revision buffers"
                      << std::endl;
            destroyRevisionBuffers(rev);
            return;
        }
        memcpy(rev.node.mappedData, table.nodes.data(), nodeCount * kNodeBytes);
        memcpy(rev.root.mappedData, table.layerRoots.data(),
               rootCount * sizeof(OctreeRaymarchRoot));
        rev.version = table.version;
        rev.nodeCount = static_cast<uint32_t>(nodeCount);
        rev.rootCount = static_cast<uint32_t>(rootCount);
        rev.users = 0;
        rev.derived = false;
        std::cout << "[OctreeRaymarchRenderer] revision " << rev.version
                  << " shared: " << rev.nodeCount << " nodes, " << rev.rootCount
                  << " roots, " << revisions_.size() << " pool entries" << std::endl;
    }

    if (!bindSlotRevision(fi, revision) &&
        revision >= 0 && static_cast<size_t>(revision) < revisions_.size() &&
        revisions_[revision].users == 0) {
        destroyRevisionBuffers(revisions_[revision]);
    }
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

    if (slotRevision_[fi] < 0 ||
        static_cast<size_t>(slotRevision_[fi]) >= revisions_.size()) {
        recordHostWriteBarrier(cmd);
        return;
    }
    SharedRevision& rev = revisions_[slotRevision_[fi]];
    if (derivePipeline_ == VK_NULL_HANDLE || deriveLayout_ == VK_NULL_HANDLE ||
        sets_[fi] == VK_NULL_HANDLE || rev.payload.buffer == VK_NULL_HANDLE ||
        rev.nodeCount == 0) {
        recordHostWriteBarrier(cmd);
        return;
    }

    // The shared node/root buffers are persistently mapped and rewritten on
    // the host in update(), after the slot fence. Two destination scopes share
    // the host writes: the derive pass (node-table read + payload write) and
    // the fragment march (node/root reads).
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
    hostBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo hostDep{};
    hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    hostDep.memoryBarrierCount = 2;
    hostDep.pMemoryBarriers = hostBarriers;
    vkCmdPipelineBarrier2(cmd, &hostDep);

    // Derive once per revision, not once per frame slot. Slots that bind an
    // already-derived revision only need the two barriers above and the
    // compute->fragment dependency below.
    if (!rev.derived) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, derivePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, deriveLayout_,
                                2, 1, &sets_[fi], 0, nullptr);

        struct {
            uint32_t nodeCount;
            uint32_t flags;
        } derivePush{rev.nodeCount, 0u};
        vkCmdPushConstants(cmd, deriveLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(derivePush), &derivePush);
        vkCmdDispatch(cmd, (rev.nodeCount + 63u) / 64u, 1u, 1u);
        rev.derived = true;
    }

    // Publish the derived payload (meta.x / hsv1.z / meta.w) to the fragment
    // march. For an already-derived revision this makes the earlier compute
    // writes visible to this slot's command buffer too.
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
}

void OctreeRaymarchRenderer::draw(VkCommandBuffer cmd, VulkanApp* app,
                                  uint32_t frameIndex, uint32_t width,
                                  uint32_t height, VkDescriptorSet mainSet,
                                  VkDescriptorSet brushSet) {
    const uint32_t fi = frameIndex % RAYMARCH_FRAMES;
    if (!app || cmd == VK_NULL_HANDLE || width == 0 || height == 0 ||
        pipeline_ == VK_NULL_HANDLE || sets_[fi] == VK_NULL_HANDLE ||
        slotRevision_[fi] < 0 ||
        static_cast<size_t>(slotRevision_[fi]) >= revisions_.size() ||
        mainSet == VK_NULL_HANDLE || brushSet == VK_NULL_HANDLE) {
        return;
    }
    const SharedRevision& rev = revisions_[slotRevision_[fi]];
    if (rev.rootCount == 0 || rev.node.buffer == VK_NULL_HANDLE ||
        rev.root.buffer == VK_NULL_HANDLE) {
        return;
    }

    // Note: the caller records recordDerive() before beginning the solid
    // color pass (host barriers and the derive dispatch are illegal inside a
    // render scope).

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline_);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

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
    pc.rootCount = rev.rootCount;
    pc.tableVersion = static_cast<uint32_t>(rev.version);
    pc.nodeCount = rev.nodeCount;
    pc.flags = 0u;
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}
