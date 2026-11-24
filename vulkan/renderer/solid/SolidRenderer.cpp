#include "SolidRenderer.hpp"
#include "../RendererUtils.hpp"

SolidRenderer::SolidRenderer() : indirectRenderer() {}
SolidRenderer::~SolidRenderer() { cleanup(nullptr); }

void SolidRenderer::init() {
    indirectRenderer.init();
}

void SolidRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    renderWidth = width;
    renderHeight = height;
    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory,
                           VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "SolidRenderer: image", image, allocation, memory, view);
    };

    for (uint32_t i = 0; i < SOLID_FRAMES; ++i) {
        createImage(app->getSwapchainImageFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    solidColorImages[i], solidColorAllocations[i], solidColorMemories[i],
                    solidColorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    solidDepthImages[i], solidDepthAllocations[i], solidDepthMemories[i],
                    solidDepthImageViews[i]);
        solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    for (uint32_t i = 0; i < SOLID_FRAMES; ++i) {
        if (solidColorImages[i] != VK_NULL_HANDLE) {
            try {
                app->transitionImageLayoutLayer(solidColorImages[i], app->getSwapchainImageFormat(),
                                                VK_IMAGE_LAYOUT_UNDEFINED,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            } catch (...) { /* best-effort init transition */ }
        }
        if (solidDepthImages[i] != VK_NULL_HANDLE) {
            try {
                app->transitionImageLayoutLayerForce(solidDepthImages[i], VK_FORMAT_D32_SFLOAT,
                                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
                app->setImageLayoutTracked(solidDepthImages[i],
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
                solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            } catch (...) { /* best-effort init transition */ }
        }
    }
}

void SolidRenderer::destroyRenderTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < SOLID_FRAMES; ++i) {
        if (solidColorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(solidColorImageViews[i]))
                vkDestroyImageView(device, solidColorImageViews[i], nullptr);
            solidColorImageViews[i] = VK_NULL_HANDLE;
        }
        app->destroyImageWithVma(solidColorImages[i], solidColorAllocations[i], solidColorMemories[i]);
        solidColorImages[i] = VK_NULL_HANDLE;
        solidColorAllocations[i] = VK_NULL_HANDLE;
        solidColorMemories[i] = VK_NULL_HANDLE;
        if (solidDepthImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(solidDepthImageViews[i]))
                vkDestroyImageView(device, solidDepthImageViews[i], nullptr);
            solidDepthImageViews[i] = VK_NULL_HANDLE;
        }
        app->destroyImageWithVma(solidDepthImages[i], solidDepthAllocations[i], solidDepthMemories[i]);
        solidDepthImages[i] = VK_NULL_HANDLE;
        solidDepthAllocations[i] = VK_NULL_HANDLE;
        solidDepthMemories[i] = VK_NULL_HANDLE;
    }
}

void SolidRenderer::cleanup(VulkanApp* app) {
    if (app == nullptr) return;
    destroyRenderTargets(app);
    indirectRenderer.cleanup(app);
}
