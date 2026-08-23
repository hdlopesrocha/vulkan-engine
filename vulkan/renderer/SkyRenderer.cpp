#include "SkyRenderer.hpp"
#include "RendererUtils.hpp"

#include "../../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

// For VBO creation
#include "../VertexBufferObjectBuilder.hpp"
#include "../../math/SphereModel.hpp"

SkyRenderer::SkyRenderer() {}

SkyRenderer::~SkyRenderer() { cleanup(nullptr); }

void SkyRenderer::init(VulkanApp* app) { (void)app; }

void SkyRenderer::cleanup(VulkanApp* app) {
    (void)app;
    if (skySphere) {
        skySphere->cleanup();
        skySphere.reset();
    }
    skyVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    skyVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    skyVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    skyVBO.indexBuffer.memory = VK_NULL_HANDLE;
    skyVBO.indexCount = 0;
}

void SkyRenderer::init(VulkanApp* app, SkySettings &settings, VkDescriptorSet descriptorSet) {
    if (!app) return;
    // Create sphere VBO if not present
    if (skyVBO.vertexBuffer.buffer == VK_NULL_HANDLE && skyVBO.indexCount == 0) {
        printf("[SkyRenderer::initSky] Creating sphere VBO...\n");
        SphereModel sphere(0.5f, 32, 16, 0);
        skyVBO = VertexBufferObjectBuilder::create(app, sphere);
        printf("[SkyRenderer::initSky] Created skyVBO: vertexBuffer=%p indexCount=%u\n", 
            (void*)skyVBO.vertexBuffer.buffer, skyVBO.indexCount);
    } else {
        printf("[SkyRenderer::initSky] Sky VBO already exists: vertexBuffer=%p indexCount=%u\n",
            (void*)skyVBO.vertexBuffer.buffer, skyVBO.indexCount);
    }

    if (descriptorSet != VK_NULL_HANDLE && !skySphere) {
        skySphere = std::make_unique<SkySphere>();
        skySphere->init(app, settings, descriptorSet);
    } else if (descriptorSet != VK_NULL_HANDLE && skySphere) {
        // SkySphere already created — write its buffer to this additional descriptor set
        skySphere->writeDescriptorSet(app, descriptorSet);
    }
}

void SkyRenderer::update(VulkanApp* app) {
    if (skySphere) skySphere->update(app);
}

Buffer SkyRenderer::getSkyUniformBuffer() const {
    if (skySphere) return skySphere->getBuffer();
    return {};
}

// ---------- Offscreen equirectangular sky rendering ----------

void SkyRenderer::createOffscreenTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    // Use fixed equirectangular resolution (2:1 aspect for full sphere)
    offscreenWidth = 2048;
    offscreenHeight = 1024;
    VkDevice device = app->getDevice();

    // Helper: create image + memory + view
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           uint32_t w, uint32_t h,
                           VkImage &image, VmaAllocation &allocation, VkDeviceMemory &memory, VkImageView &view) {
        RendererUtils::createImage2DWithVma(device, app, w, h, format, usage, aspect,
                                            "SkyRenderer: equirect image", image, allocation, memory, view);
    };

    // Remove render pass creation - using dynamic rendering
    VkFormat colorFormat = app->getSwapchainImageFormat();

    // Equirect raster pipeline removed: ray tracing computes sky procedurally.

    // --- Per-frame color images (no depth, no framebuffers needed for dynamic rendering) ---
    for (uint32_t i = 0; i < SkyRenderer::SKY_FRAMES; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    offscreenWidth, offscreenHeight,
                    skyColorImages[i], skyColorAllocations[i], skyColorMemories[i], skyColorImageViews[i]);
        skyColorLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    std::cerr << "[SkyRenderer] Created equirectangular sky targets " << offscreenWidth << "x" << offscreenHeight << std::endl;
}

void SkyRenderer::destroyOffscreenTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < SkyRenderer::SKY_FRAMES; ++i) {
        if (skyColorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(skyColorImageViews[i]))
                vkDestroyImageView(device, skyColorImageViews[i], nullptr);
            skyColorImageViews[i] = VK_NULL_HANDLE;
        }
        app->destroyImageWithVma(skyColorImages[i], skyColorAllocations[i], skyColorMemories[i]);
        skyColorImages[i] = VK_NULL_HANDLE;
        skyColorAllocations[i] = VK_NULL_HANDLE;
        skyColorMemories[i] = VK_NULL_HANDLE;
        skyColorLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}


