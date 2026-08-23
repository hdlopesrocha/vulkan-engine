#include "BrushRenderer.hpp"
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

    // Allocate (once) and write per-frame brush depth descriptor sets (set=1)
    for (size_t fi = 0; fi < depthDescriptorSets.size(); ++fi) {
        if (depthDescriptorSets[fi] == VK_NULL_HANDLE) {
            depthDescriptorSets[fi] = app->createDescriptorSet(app->getBrushDepthDescriptorSetLayout());
        }
    }
    writeDepthDescriptors(app);
}

void BrushRenderer::cleanup(VulkanApp* app) {
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

void BrushRenderer::initSlots(VulkanApp* app, uint32_t maxChunks, uint32_t vertexBytes, uint32_t indexBytes) {
    if (!app) return;
    solidIndirectRenderer.initSlots(app, maxChunks, vertexBytes, indexBytes);
    liquidIndirectRenderer.initSlots(app, maxChunks, vertexBytes, indexBytes);
}

void BrushRenderer::stopGenPools() {
    solidGenPool.stop();
    liquidGenPool.stop();
}
