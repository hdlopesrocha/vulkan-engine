#include "BrushRenderer.hpp"
#include "SolidRenderer.hpp"
#include "RendererUtils.hpp"
#include "DescriptorWriter.hpp"
#include <stdexcept>
#include <iostream>

BrushRenderer::BrushRenderer() {}
BrushRenderer::~BrushRenderer() {}

void BrushRenderer::init(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;

    // Initialize the dedicated brush solid/liquid IndirectRenderers (no
    // streamer — brush meshes are small and infrequent; upload fits in the
    // legacy ring path). Both are dedicated pools: brush geometry never shares
    // the main scene IRs.
    solidIndirectRenderer.init();
    liquidIndirectRenderer.init();

    createRenderTargets(app, width, height);

    // Brush back-face renderer: uses solid shaders (no water dependency) and
    // VK_COMPARE_OP_GREATER to capture the farthest back-face depth.
    backFaceRenderer = std::make_unique<BrushBackFaceRenderer>();
    if (backFaceRenderer) {
        backFaceRenderer->init(app);
        backFaceRenderer->createPipelines(app);
        backFaceRenderer->createRenderTargets(app, width, height);
    }

    // Allocate (once) and write per-frame brush depth descriptor sets (set=1)
    for (size_t fi = 0; fi < depthDescriptorSets.size(); ++fi) {
        if (depthDescriptorSets[fi] == VK_NULL_HANDLE) {
            depthDescriptorSets[fi] = app->createDescriptorSet(app->getBrushDepthDescriptorSetLayout());
        }
    }
    writeDepthDescriptors(app);
}

void BrushRenderer::cleanup(VulkanApp* app) {
    if (backFaceRenderer && app) {
        backFaceRenderer->cleanup(app);
    }
    backFaceRenderer.reset();
    destroyRenderTargets(app);
    solidIndirectRenderer.cleanup(app);
    liquidIndirectRenderer.cleanup(app);
}

void BrushRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    VkDevice device = app->getDevice();
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkImageView& view) {
        VkDeviceMemory dummyMem = VK_NULL_HANDLE;
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "BrushRenderer: brush", image, allocation, dummyMem, view);
    };
    VkFormat colorFormat = app->getSwapchainImageFormat();
    for (uint32_t i = 0; i < BRUSH_FRAMES; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    colorImages[i], colorAllocations[i], colorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    depthImages[i], depthAllocations[i], depthImageViews[i]);
    }
    // Fresh images start in UNDEFINED layout; the first barrier of each frame
    // (or the resize path) transitions them explicitly.
    colorLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    depthLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
}

void BrushRenderer::destroyRenderTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < BRUSH_FRAMES; ++i) {
        if (colorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(colorImageViews[i]))
                vkDestroyImageView(device, colorImageViews[i], nullptr);
            colorImageViews[i] = VK_NULL_HANDLE;
        }
        if (colorImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(colorImages[i], colorAllocations[i], VK_NULL_HANDLE);
            colorImages[i] = VK_NULL_HANDLE;
            colorAllocations[i] = VK_NULL_HANDLE;
        }
        if (depthImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(depthImageViews[i]))
                vkDestroyImageView(device, depthImageViews[i], nullptr);
            depthImageViews[i] = VK_NULL_HANDLE;
        }
        if (depthImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(depthImages[i], depthAllocations[i], VK_NULL_HANDLE);
            depthImages[i] = VK_NULL_HANDLE;
            depthAllocations[i] = VK_NULL_HANDLE;
        }
    }
}

void BrushRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    destroyRenderTargets(app);
    createRenderTargets(app, width, height);
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, width, height);
    // Brush images have been destroyed and recreated; reset tracked layouts so
    // the first barrier after resize uses VK_IMAGE_LAYOUT_UNDEFINED as oldLayout.
    colorLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    depthLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    // Rewrite brush depth descriptors after recreating brush targets.
    writeDepthDescriptors(app);
}

void BrushRenderer::writeDepthDescriptors(VulkanApp* app) {
    VkSampler brushDepthSampler = depthLinearSampler;
    if (brushDepthSampler == VK_NULL_HANDLE) {
        brushDepthSampler = depthShadowSampler;
    }

    for (size_t fi = 0; fi < depthDescriptorSets.size(); ++fi) {
        VkDescriptorSet dstSet = depthDescriptorSets[fi];
        if (dstSet == VK_NULL_HANDLE) continue;
        VkImageView brushFrontView = getDepthView(static_cast<uint32_t>(fi));
        VkImageView brushBackView = VK_NULL_HANDLE;
        if (backFaceRenderer) {
            brushBackView = backFaceRenderer->getBackFaceDepthView(static_cast<uint32_t>(fi));
        }
        if (brushDepthSampler == VK_NULL_HANDLE) continue;

        DescriptorWriter writer(app->getDevice());
        if (brushFrontView != VK_NULL_HANDLE) {
            writer.writeImage(dstSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              brushDepthSampler, brushFrontView,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (brushBackView != VK_NULL_HANDLE) {
            writer.writeImage(dstSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              brushDepthSampler, brushBackView,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        writer.flush();
    }
}

void BrushRenderer::pollPendingTransfers(VulkanApp* app) {
    solidIndirectRenderer.pollPendingTransfers(app);
    liquidIndirectRenderer.pollPendingTransfers(app);
}

void BrushRenderer::setCullFrame(uint32_t frameIndex) {
    solidIndirectRenderer.setCullFrame(frameIndex);
}

void BrushRenderer::clearMeshes() {
    std::lock_guard<std::recursive_mutex> lock(solidChunksMutex);

    // Remove all brush opaque meshes from the dedicated brush solid IR.
    for (auto &entry : solidChunks) {
        if (entry.second.meshId != UINT32_MAX) {
            solidIndirectRenderer.removeMeshSlotted(entry.second.meshId);
        }
    }
    solidChunks.clear();

    // Remove brush transparent meshes from the dedicated brush liquid IR.
    for (auto &entry : transparentChunks) {
        if (entry.second.meshId != UINT32_MAX) {
            liquidIndirectRenderer.removeMeshSlotted(entry.second.meshId);
        }
    }
    transparentChunks.clear();
}

void BrushRenderer::stageOldChunks() {
    {
        std::lock_guard<std::recursive_mutex> lock(solidChunksMutex);
        for (auto& entry : solidChunks) {
            pendingOldSolidChunks[entry.first] = entry.second;
        }
        solidChunks.clear();
    }
    {
        std::lock_guard<std::recursive_mutex> lock(transparentChunksMutex);
        for (auto& entry : transparentChunks) {
            pendingOldLiquidChunks[entry.first] = entry.second;
        }
        transparentChunks.clear();
    }
}

void BrushRenderer::captureOldSlots(std::unordered_map<NodeID, uint32_t>& outSolid,
                                    std::unordered_map<NodeID, uint32_t>& outTransparent) {
    {
        std::lock_guard<std::recursive_mutex> lock(pendingOldSolidChunksMutex);
        for (auto& entry : pendingOldSolidChunks) {
            if (entry.second.meshId != UINT32_MAX)
                outSolid[entry.first] = entry.second.meshId;
        }
        pendingOldSolidChunks.clear();
    }
    {
        std::lock_guard<std::recursive_mutex> lock(pendingOldLiquidChunksMutex);
        for (auto& entry : pendingOldLiquidChunks) {
            if (entry.second.meshId != UINT32_MAX)
                outTransparent[entry.first] = entry.second.meshId;
        }
        pendingOldLiquidChunks.clear();
    }
}

void BrushRenderer::recordEarlyPass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                    SolidRenderer& solidRenderer, VkDescriptorSet mainDs) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (frameIndex >= BRUSH_FRAMES) return;

    // ── Early brush pass: render brush to its own buffers before solid/water ──
    // Brush front depth (LESS test, writes depth), backface depth (GREATER test),
    // and brush color are all written here before solid geometry touches the
    // scene depth buffer. A later overlay pass renders brush with opacity on top
    // of solid/water using the scene depth buffer for occlusion culling.
    VkImage brushColorImg = colorImages[frameIndex];
    VkImage brushDepthImg = depthImages[frameIndex];
    VkImageLayout oldColLayout = colorLayouts[frameIndex];
    VkImageLayout oldDepLayout = depthLayouts[frameIndex];
    const uint32_t width = app->getWidth();
    const uint32_t height = app->getHeight();

    // Transition brush color + depth to attachment layouts
    if (brushColorImg != VK_NULL_HANDLE && oldColLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        VkAccessFlags2 colSrcAccess = 0;
        VkPipelineStageFlags2 colSrcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        if (oldColLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            colSrcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            colSrcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        RendererUtils::transitionImageLayout(cmd, brushColorImg,
            oldColLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            colSrcAccess, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            colSrcStage, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        colorLayouts[frameIndex] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (brushDepthImg != VK_NULL_HANDLE && oldDepLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        VkAccessFlags2 depthSrcAccess = 0;
        VkPipelineStageFlags2 depthSrcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        if (oldDepLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            depthSrcAccess = 0;
            depthSrcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        } else if (oldDepLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            depthSrcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            depthSrcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        } else {
            depthSrcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depthSrcStage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        }
        RendererUtils::transitionImageLayout(cmd, brushDepthImg,
            oldDepLayout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            depthSrcAccess, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            depthSrcStage, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        depthLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    // Instance B1: brush front depth (LESS, write, no color)
    if (brushDepthImg != VK_NULL_HANDLE) {
        VkClearValue bdClear{}; bdClear.depthStencil = {1.0f, 0};
        VkRenderingAttachmentInfo bdAtt{};
        bdAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        bdAtt.imageView = depthImageViews[frameIndex];
        bdAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bdAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        bdAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        bdAtt.clearValue = bdClear;
        VkRenderingInfo bri{};
        bri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        bri.renderArea.offset = {0, 0};
        bri.renderArea.extent = {width, height};
        bri.layerCount = 1;
        bri.pDepthAttachment = &bdAtt;
        vkCmdBeginRendering(cmd, &bri);
        VkViewport vp{0,0,(float)width,(float)height,0,1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0,0},{width, height}};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        solidRenderer.drawDepthExternal(cmd, mainDs, solidIndirectRenderer);
        vkCmdEndRendering(cmd);
    }

    // Brush backface depth (GREATER, farthest depth wins) — to its own buffer
    if (backFaceRenderer) {
        backFaceRenderer->renderBackFacePass(app, cmd, frameIndex, solidIndirectRenderer, mainDs);
    }

    // Instance B2: brush color to brush color buffer (raw, no blending)
    if (brushColorImg != VK_NULL_HANDLE && brushDepthImg != VK_NULL_HANDLE) {
        VkClearValue bcClear{}; bcClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkRenderingAttachmentInfo bcColor{};
        bcColor.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        bcColor.imageView = colorImageViews[frameIndex];
        bcColor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        bcColor.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        bcColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        bcColor.clearValue = bcClear;
        VkRenderingAttachmentInfo bcDepth{};
        bcDepth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        bcDepth.imageView = depthImageViews[frameIndex];
        bcDepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bcDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        bcDepth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        VkRenderingInfo bcri{};
        bcri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        bcri.renderArea.offset = {0, 0};
        bcri.renderArea.extent = {width, height};
        bcri.layerCount = 1;
        bcri.colorAttachmentCount = 1;
        bcri.pColorAttachments = &bcColor;
        bcri.pDepthAttachment = &bcDepth;
        vkCmdBeginRendering(cmd, &bcri);
        VkViewport vp{0,0,(float)width,(float)height,0,1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0,0},{width, height}};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        solidRenderer.drawBrushColorExternal(cmd, mainDs, solidIndirectRenderer);
        vkCmdEndRendering(cmd);
    }

    // Transition brush targets to SHADER_READ_ONLY after use
    if (brushColorImg != VK_NULL_HANDLE && colorLayouts[frameIndex] == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        RendererUtils::transitionImageLayout(cmd, brushColorImg,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        colorLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (brushDepthImg != VK_NULL_HANDLE && depthLayouts[frameIndex] == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        RendererUtils::transitionImageLayout(cmd, brushDepthImg,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        depthLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

void BrushRenderer::initSlots(VulkanApp* app, uint32_t maxChunks, uint32_t vertexBytes, uint32_t indexBytes) {
    if (!app) return;
    solidIndirectRenderer.initSlots(app, maxChunks, vertexBytes, indexBytes);
    liquidIndirectRenderer.initSlots(app, maxChunks, vertexBytes, indexBytes);
}

void BrushRenderer::stopGenPools() {
    solidGenPool.stop();
    liquidGenPool.stop();
}
