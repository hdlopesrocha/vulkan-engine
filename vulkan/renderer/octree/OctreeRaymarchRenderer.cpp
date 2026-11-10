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

    // Set=2: binding 0 = node table, binding 1 = layer roots (fragment stage).
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t b = 0; b < 2; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorCount = 1;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    setLayout_ = descAlloc.createLayout(bindings, 2, 0, nullptr,
                                        "OctreeRaymarchRenderer: set2 layout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2u * RAYMARCH_FRAMES;
    VkDescriptorPool pool = descAlloc.createPool(
        &poolSize, 1, RAYMARCH_FRAMES, 0, "OctreeRaymarchRenderer: pool");
    descAlloc.allocateSets(pool, setLayout_, RAYMARCH_FRAMES, sets_.data(),
                           "OctreeRaymarchRenderer: set");

    // The per-frame node/root buffers are allocated lazily in upload(): the
    // raymarch mode is off by default and a slot costs ~48 MB at the full cap,
    // so nothing is reserved until a table actually needs publishing.

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
}

void OctreeRaymarchRenderer::cleanup(VulkanApp* app) {
    if (app) {
        for (auto& buf : nodeBuffers_)
            if (buf.buffer != VK_NULL_HANDLE) app->destroyBuffer(buf);
        for (auto& buf : rootBuffers_)
            if (buf.buffer != VK_NULL_HANDLE) app->destroyBuffer(buf);
    }
    nodeBuffers_.fill(Buffer{});
    rootBuffers_.fill(Buffer{});
    sets_.fill(VK_NULL_HANDLE);
    uploadedVersion_.fill(0);
    rootCounts_.fill(0);
    app_ = nullptr;
    // The layout/pool/pipeline are registered with VulkanResourceManager at
    // creation and destroyed by it at shutdown (same contract as the other
    // renderers).
    setLayout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    maxNodes_ = 0;
}

bool OctreeRaymarchRenderer::ensureSlotBuffers(uint32_t fi) {
    if (!app_ || fi >= RAYMARCH_FRAMES) return false;
    if (nodeBuffers_[fi].buffer != VK_NULL_HANDLE &&
        rootBuffers_[fi].buffer != VK_NULL_HANDLE)
        return true;
    if (sets_[fi] == VK_NULL_HANDLE) return false;

    // Host-visible and persistently mapped: MyApp::update() rewrites the slot
    // after its fence wait, and draw() records the HOST -> FRAGMENT barrier.
    nodeBuffers_[fi] = app_->createBuffer(
        static_cast<VkDeviceSize>(maxNodes_ * kNodeBytes),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    rootBuffers_[fi] = app_->createBuffer(
        static_cast<VkDeviceSize>(ROOT_CAP * sizeof(OctreeRaymarchRoot)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (nodeBuffers_[fi].buffer == VK_NULL_HANDLE ||
        rootBuffers_[fi].buffer == VK_NULL_HANDLE) {
        std::cerr << "[OctreeRaymarchRenderer] failed to create per-frame table resources"
                  << std::endl;
        if (nodeBuffers_[fi].buffer != VK_NULL_HANDLE) app_->destroyBuffer(nodeBuffers_[fi]);
        if (rootBuffers_[fi].buffer != VK_NULL_HANDLE) app_->destroyBuffer(rootBuffers_[fi]);
        nodeBuffers_[fi] = Buffer{};
        rootBuffers_[fi] = Buffer{};
        return false;
    }

    DescriptorWriter(app_->getDevice())
        .writeBuffer(sets_[fi], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     nodeBuffers_[fi].buffer, 0,
                     app_->getClampedWholeSizeRange(nodeBuffers_[fi].buffer))
        .writeBuffer(sets_[fi], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     rootBuffers_[fi].buffer, 0,
                     app_->getClampedWholeSizeRange(rootBuffers_[fi].buffer))
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

    // Note: the caller records recordHostWriteBarrier() before beginning the
    // solid color pass (HOST barriers are illegal inside a render scope).

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
    pc.rootCount = rootCounts_[fi];
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}
