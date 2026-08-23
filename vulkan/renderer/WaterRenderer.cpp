
#include "WaterRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include <cstdlib>
#include "RendererUtils.hpp"
#include "BrushRenderer.hpp"

#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

// Sub-renderer accessors removed: SceneRenderer now owns back-face and 360 renderers.

// Global image layout tracking for WaterRenderer render targets
// (only the water color + water geometry depth layouts are consumed today:
// sceneColor/sceneDepth tracking had no readers left after getSceneDepthLayout
// was removed, so they were dropped).
static VkImageLayout waterDepthImageLayouts[VulkanApp::MAX_FRAMES_IN_FLIGHT] = {};
static VkImageLayout waterGeomDepthImageLayouts[VulkanApp::MAX_FRAMES_IN_FLIGHT] = {};

WaterRenderer::WaterRenderer() {}

WaterRenderer::~WaterRenderer() {}

void WaterRenderer::init(VulkanApp* app, Buffer& waterParamsBuffer_, const std::vector<WaterParams>& waterParams, uint32_t layerCount) {
    this->waterParamsBuffer = waterParamsBuffer_;
    this->appPtr = app;
    waterParamsCount = layerCount;
    waterIndirectRenderer.init();
    createSamplers(app);

    // Water params SSBO is initialized below from the provided vector. The raster
    // water pipelines were removed (ray tracing renders water directly).

    // Water render time UBO (binding 10): created here, bound into the scene
    // descriptor sets by SceneRenderer, updated per frame in renderPass().
    if (waterRenderUBO_.buffer == VK_NULL_HANDLE) {
        waterRenderUBO_ = app->createBuffer(sizeof(WaterRenderUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // Cache the env-var flag once (never read per frame)
    envDisableWaterGeom_ = (std::getenv("VULKAN_DISABLE_WATERGEOM") != nullptr);

    // Sub-renderer initialization is owned by SceneRenderer
}

void WaterRenderer::setSceneRenderers(BrushRenderer* brush) {
    brushRenderer_ = brush;
}



void WaterRenderer::updateGPUParamsForLayer(uint32_t layer, const WaterParams& p) {
    if (!appPtr) return;
    if (layer >= waterParamsCount) return;
    WaterParamsGPU gpu{};
    gpu.params1 = glm::vec4(p.refractionStrength, p.fresnelPower, p.transparency, p.reflectionStrength);
    gpu.params2 = glm::vec4(p.waterTint, p.noiseScale, static_cast<float>(p.noiseOctaves), p.noisePersistence);
    gpu.params3 = glm::vec4(p.noiseTimeSpeed, p.noiseLacunarity, p.specularIntensity, p.specularPower);
    gpu.shallowColor = glm::vec4(p.shallowColor, p.waveDepthTransition);
    gpu.deepColor = glm::vec4(p.deepColor, p.glitterIntensity);
    gpu.waveParams = glm::vec4(p.tessNoiseInfluence, 0.0f, p.bumpAmplitude, p.depthFalloff);
    gpu.reserved1 = glm::vec4(p.enableReflection ? 1.0f : 0.0f,
                              p.enableRefraction ? 1.0f : 0.0f,
                              p.enableBlur ? 1.0f : 0.0f,
                              p.blurRadius);
    gpu.reserved2 = glm::vec4(static_cast<float>(p.blurSamples), p.volumeBlurRate, p.volumeBumpRate, p.uniformReflection ? 1.0f : 0.0f);
    gpu.causticColor = glm::vec4(p.causticColor, 0.0f);
    gpu.causticParams = glm::vec4(p.causticScale, p.causticIntensity, p.causticPower, p.causticDepthScale);
    gpu.causticExtraParams = glm::vec4(p.causticLineScale, p.causticLineMix, static_cast<float>(p.causticType), p.causticVelocity);
    gpu.reserved3 = glm::vec4(cube360Available ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    gpu.tessParams = glm::vec4(p.tessNearDist, p.tessFarDist, p.tessMinLevel, p.tessMaxLevel);

    size_t offset = static_cast<size_t>(layer) * sizeof(WaterParamsGPU);
    void* data = nullptr;
    data = waterParamsBuffer.map(offset);
    memcpy(data, &gpu, sizeof(WaterParamsGPU));
    waterParamsBuffer.unmap(); // VMA persistent mapping
}

void WaterRenderer::cleanup(VulkanApp* app) {
    waterIndirectRenderer.cleanup(app);
    destroyRenderTargets(app);
    if (waterRenderUBO_.buffer != VK_NULL_HANDLE) waterRenderUBO_ = {};
}

void WaterRenderer::createSamplers(VulkanApp* app) {
    linearSampler = app->createSamplerLinearClamp("WaterRenderer: linearSampler");
    nearestSampler = app->createSamplerNearestClamp("WaterRenderer: nearestSampler");
}

void WaterRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (renderWidth == width && renderHeight == height && waterDepthImages[0] != VK_NULL_HANDLE) {
        return; // Already created at this size
    }
    
    destroyRenderTargets(app);
    
    renderWidth = width;
    renderHeight = height;
    
    VkDevice device = app->getDevice();
    
    // Helper to create image + memory + view
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "WaterRenderer: image", image, allocation, memory, view);
    };
    
    // Reset layout tracking (use file-scope static variables)
    for (uint32_t i = 0; i < FRAMES; ++i) {
        waterDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        waterGeomDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // Create per-frame scene offscreen render targets (2 sets for 2 frames in flight)
    for (uint32_t frameIdx = 0; frameIdx < FRAMES; ++frameIdx) {
        createImage(app->getSwapchainImageFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    sceneColorImages[frameIdx], sceneColorAllocations[frameIdx], sceneColorMemories[frameIdx], sceneColorImageViews[frameIdx]);

        // Transition directly to final layout (SHADER_READ_ONLY for post-process sampling)
        if (sceneColorImages[frameIdx] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(sceneColorImages[frameIdx], app->getSwapchainImageFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(sceneColorImages[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
        }

        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    sceneDepthImages[frameIdx], sceneDepthAllocations[frameIdx], sceneDepthMemories[frameIdx], sceneDepthImageViews[frameIdx]);
        std::cerr << "[WaterRenderer] sceneDepthImages[" << frameIdx << "] = " << (void*)sceneDepthImages[frameIdx] << std::endl;
        // Transition directly to final layout (SHADER_READ_ONLY for post-process sampling)
        if (sceneDepthImages[frameIdx] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayerForce(sceneDepthImages[frameIdx], VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(sceneDepthImages[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
        }
    }

    for (uint32_t frameIdx = 0; frameIdx < FRAMES; ++frameIdx) {
        createImage(VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    waterDepthImages[frameIdx], waterDepthAllocations[frameIdx], waterDepthMemories[frameIdx], waterDepthImageViews[frameIdx]);
        waterDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;

        // Transition directly to final layout (SHADER_READ_ONLY for post-process sampling)
        if (waterDepthImages[frameIdx] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayerForce(waterDepthImages[frameIdx], VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(waterDepthImages[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            waterDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        createImage(VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                waterGeomDepthImages[frameIdx], waterGeomDepthAllocations[frameIdx], waterGeomDepthMemories[frameIdx], waterGeomDepthImageViews[frameIdx]);
        waterGeomDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;
        std::cerr << "[WaterRenderer] waterGeomDepthImage[" << frameIdx << "] = " << (void*)waterGeomDepthImages[frameIdx] << std::endl;

        // Transition directly to final layout (DEPTH_STENCIL_ATTACHMENT for water geometry pass)
        if (waterGeomDepthImages[frameIdx] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayerForce(waterGeomDepthImages[frameIdx], VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(waterGeomDepthImages[frameIdx], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            waterGeomDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        // Back-face depth image is no longer produced (WaterBackFaceRenderer removed).
    }

    // Back-face depth targets are owned/created by SceneRenderer

    // NOTE: the per-frame scene-texture descriptor set (waterDepthDescriptorSets) is
    // allocated once per frame slot in prepareSceneTexturesForFrame() and
    // reused every frame — only the image-view bindings are refreshed via
    // vkUpdateDescriptorSets. The set is never updated while a pending command
    // buffer references it and never needs UPDATE_AFTER_BIND (which trips
    // GPU-assisted validation).

    std::cout << "[WaterRenderer] Created render targets (2 sets) " << width << "x" << height << std::endl;
}

void WaterRenderer::destroyRenderTargets(VulkanApp* app) {
    VkDevice device = app->getDevice();
    // Clear per-frame image handles; actual Vulkan destruction
    // will be performed by the VulkanResourceManager.
    for (uint32_t i = 0; i < FRAMES; ++i) {
        sceneColorImages[i] = VK_NULL_HANDLE;
        sceneColorAllocations[i] = VK_NULL_HANDLE;
        sceneColorMemories[i] = VK_NULL_HANDLE;
        sceneColorImageViews[i] = VK_NULL_HANDLE;
        sceneDepthImages[i] = VK_NULL_HANDLE;
        sceneDepthAllocations[i] = VK_NULL_HANDLE;
        sceneDepthMemories[i] = VK_NULL_HANDLE;
        sceneDepthImageViews[i] = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < FRAMES; ++i) {
        waterDepthImages[i] = VK_NULL_HANDLE;
        waterDepthAllocations[i] = VK_NULL_HANDLE;
        waterDepthMemories[i] = VK_NULL_HANDLE;
        waterDepthImageViews[i] = VK_NULL_HANDLE;
        waterGeomDepthImages[i] = VK_NULL_HANDLE;
        waterGeomDepthAllocations[i] = VK_NULL_HANDLE;
        waterGeomDepthMemories[i] = VK_NULL_HANDLE;
        waterGeomDepthImageViews[i] = VK_NULL_HANDLE;
    }
    // Back-face depth targets are destroyed by SceneRenderer

    // Reset descriptor pool to free descriptor sets. The old sets may still be
    // referenced by in-flight command buffers, so we must wait before resetting.
    // Scope it to the graphics queue (queueWaitIdle) instead of a whole-device
    // idle: these descriptor sets are only consumed by graphics-queue water draws
    // (the geometry/transfer queue only performs buffer copies, never binds these
    // sets), so a graphics-queue idle covers every consumer while leaving compute
    // and unrelated device work untouched. Per AGENTS.md, vkDeviceWaitIdle is
    // reserved for shutdown / major rebuilds.
    if (waterDepthDescriptorPool != VK_NULL_HANDLE && app) {
        VkResult r = app->queueWaitIdle();
        if (r == VK_SUCCESS) {
            VkResult resetResult = vkResetDescriptorPool(device, waterDepthDescriptorPool, 0);
            if (resetResult != VK_SUCCESS) {
                std::cerr << "[WaterRenderer] Failed to reset water depth descriptor pool (result=" << (int)resetResult << ")" << std::endl;
            }
        } else {
            std::cerr << "[WaterRenderer] Skipping descriptor pool reset: graphics queue not idle (result=" << (int)r << ")" << std::endl;
        }
    }
    // The per-slot scene-texture sets live in waterDepthDescriptorPool, so the
    // reset above frees them; just drop the dangling handles.
    for (uint32_t i = 0; i < FRAMES; ++i) waterDepthDescriptorSets[i] = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < FRAMES; ++i) cubemapWaterDepthDS[i] = VK_NULL_HANDLE;
    // The pool reset invalidated every set allocated from it; drop all cached
    // descriptor-write signatures so the next update always rewrites (a set
    // handle reused by the allocator must never be skipped against a stale
    // entry). The async per-task sets use their own pools and are re-created
    // lazily, so clearing here only costs one extra write for them.
    sceneTextureWriteCache.clear();
}

void WaterRenderer::clearRenderTargets(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!app) return;
    if (cmd == VK_NULL_HANDLE) return;
    if (frameIndex >= 3) return;

    // Clear the water offscreen targets using vkCmdClearColorImage /
    // vkCmdClearDepthStencilImage.  This is cheaper than a full dynamic
    // rendering begin/end pass and requires TRANSFER_DST on both images.
    VkImage colorImg = waterDepthImages[frameIndex];
    VkImage depthImg = waterGeomDepthImages[frameIndex];

    if (colorImg == VK_NULL_HANDLE && depthImg == VK_NULL_HANDLE) return;

    // Transition both images to TRANSFER_DST_OPTIMAL for the clear.
    if (colorImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, colorImg, VK_FORMAT_R32G32B32A32_SFLOAT,
            waterDepthImageLayouts[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    }
    if (depthImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT,
            waterGeomDepthImageLayouts[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    }

    // Clear water color to transparent black and depth to 1.0.
    VkClearColorValue clearValue{};
    clearValue.float32[0] = 0.0f; clearValue.float32[1] = 0.0f;
    clearValue.float32[2] = 0.0f; clearValue.float32[3] = 0.0f;
    if (colorImg != VK_NULL_HANDLE) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, colorImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
    }
    if (depthImg != VK_NULL_HANDLE) {
        VkClearDepthStencilValue depthClear{1.0f, 0};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, depthImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depthClear, 1, &range);
    }

    // Transition to SHADER_READ_ONLY_OPTIMAL for the post-process compositor.
    if (colorImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, colorImg, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        waterDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (depthImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        waterGeomDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

VkImageLayout WaterRenderer::getWaterGeomDepthLayout(uint32_t frameIndex) const {
    if (frameIndex < 3) return waterGeomDepthImageLayouts[frameIndex];
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

void WaterRenderer::setWaterGeomDepthLayout(uint32_t frameIndex, VkImageLayout layout) {
    if (frameIndex < 3) waterGeomDepthImageLayouts[frameIndex] = layout;
}




// (Water back-face raster pass removed; depth no longer captured.)



void WaterRenderer::initializeWaterParamsBuffer(const std::vector<WaterParams>& waterParams) {
    if (waterParamsBuffer.buffer == VK_NULL_HANDLE) return;

    auto makeGpu = [&](const WaterParams& p) {
        WaterParamsGPU gpu{};
        gpu.params1 = glm::vec4(p.refractionStrength, p.fresnelPower, p.transparency, p.reflectionStrength);
        gpu.params2 = glm::vec4(p.waterTint, p.noiseScale, static_cast<float>(p.noiseOctaves), p.noisePersistence);
        gpu.params3 = glm::vec4(p.noiseTimeSpeed, p.noiseLacunarity, p.specularIntensity, p.specularPower);
        gpu.shallowColor = glm::vec4(p.shallowColor, p.waveDepthTransition);
        gpu.deepColor = glm::vec4(p.deepColor, p.glitterIntensity);
        gpu.waveParams = glm::vec4(p.tessNoiseInfluence, 0.0f, p.bumpAmplitude, p.depthFalloff);
        gpu.reserved1 = glm::vec4(p.enableReflection ? 1.0f : 0.0f,
                                  p.enableRefraction ? 1.0f : 0.0f,
                                  p.enableBlur ? 1.0f : 0.0f,
                                  p.blurRadius);
        gpu.reserved2 = glm::vec4(static_cast<float>(p.blurSamples), p.volumeBlurRate, p.volumeBumpRate, p.uniformReflection ? 1.0f : 0.0f);
        gpu.causticColor = glm::vec4(p.causticColor, 0.0f);
        gpu.causticParams = glm::vec4(p.causticScale, p.causticIntensity, p.causticPower, p.causticDepthScale);
        gpu.causticExtraParams = glm::vec4(p.causticLineScale, p.causticLineMix, static_cast<float>(p.causticType), p.causticVelocity);
        gpu.reserved3 = glm::vec4(cube360Available ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
        gpu.tessParams = glm::vec4(p.tessNearDist, p.tessFarDist, p.tessMinLevel, p.tessMaxLevel);
        return gpu;
    };

    for (uint32_t i = 0; i < waterParams.size(); ++i) {
        const WaterParamsGPU gpu = (i < waterParams.size()) ? makeGpu(waterParams[i]) : WaterParamsGPU();
        memcpy(static_cast<char*>(waterParamsBuffer.mappedData) + i * sizeof(WaterParamsGPU), &gpu, sizeof(WaterParamsGPU));
    }
}


// Execute water's offscreen geometry pass on the provided command buffer.
// The caller must ensure that the solid pass has already ended on this same
// command buffer so that the scene color/depth images are available.


// (Water back-face raster pass removed.)

// Helper: create a 1x1 image with given format, initialize to black (color) or 1.0 (depth).
// Returns the image view on success, VK_NULL_HANDLE on failure. Outputs image/allocation/memory via pointers.
static VkImageView _createDummy1x1ImageView(VulkanApp* app, VkFormat fmt, VkImageAspectFlags aspect,
                                            VkImage* outImage = nullptr, VmaAllocation* outAllocation = nullptr, VkDeviceMemory* outMemory = nullptr) {
    VkDevice device = app->getDevice();
    VkImageCreateInfo img{};
    img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img.imageType = VK_IMAGE_TYPE_2D; img.format = fmt;
    img.extent = {1, 1, 1}; img.mipLevels = 1; img.arrayLayers = 1;
    img.samples = VK_SAMPLE_COUNT_1_BIT; img.tiling = VK_IMAGE_TILING_OPTIMAL;
    img.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image; VmaAllocation allocation; VkDeviceMemory mem;
    app->createImageWithVma(img, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, allocation, mem, "WaterRenderer: cubemapDummy");
    VkImageViewCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    VkImageView view;
    if (vkCreateImageView(device, &vi, nullptr, &view) != VK_SUCCESS) {
        app->destroyImageWithVma(image, allocation, mem);
        return VK_NULL_HANDLE;
    }
    app->resources.addImageView(view, "WaterRenderer: cubemapDummyView");
    if (outImage) *outImage = image;
    if (outAllocation) *outAllocation = allocation;
    if (outMemory) *outMemory = mem;
    // Initialize (clear) via 1-shot command buffer — record ALL barriers
    // and the clear into the same cb so layout transitions and clear are
    // submitted atomically. Must use recordTransitionImageLayoutLayer
    // (not transitionImageLayout) to avoid separate transient submissions.
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        VkImageSubresourceRange range{aspect, 0, 1, 0, 1};
        app->recordTransitionImageLayoutLayer(cmd, image, fmt, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
        if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
            VkClearDepthStencilValue cv{1.0f, 0};
            vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
        } else {
            VkClearColorValue cv{};
            vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
        }
        app->recordTransitionImageLayoutLayer(cmd, image, fmt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    });
    return view;
}




// Solid 360° cubemap reflection is owned and executed by SceneRenderer.

