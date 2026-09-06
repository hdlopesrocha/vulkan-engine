
#include "WaterRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include <vector>
#include "RendererUtils.hpp"
#include "BrushRenderer.hpp"
#include "WaterBackFaceRenderer.hpp"
#include "RayTracingResources.hpp"
#include "WireframeRenderer.hpp"

#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include "../ShaderStage.hpp"
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

    // Create water pipelines and initialize the water params SSBO from the provided vector.
    createWaterPipelines(app, waterParams);

    // Water render time UBO (binding 10): created here, bound into the scene
    // descriptor sets by SceneRenderer, updated per frame in renderPass().
    if (waterRenderUBO_.buffer == VK_NULL_HANDLE) {
        // Descriptor-buffer sources need a device address for vkGetDescriptorEXT.
        VkBufferUsageFlags uboUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (app->useDescriptorBuffer())
            uboUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        waterRenderUBO_ = app->createBuffer(sizeof(WaterRenderUBO), uboUsage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // Sub-renderer initialization is owned by SceneRenderer
}

void WaterRenderer::setSceneRenderers(SolidRenderer* solid, BrushRenderer* brush,
                                      WaterBackFaceRenderer* backFace,
                                      WireframeRenderer* waterWireframe) {
    solidRenderer_ = solid;
    brushRenderer_ = brush;
    backFaceRenderer_ = backFace;
    waterWireframe_ = waterWireframe;
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
    gpu.reserved3 = glm::vec4(0.0f); // legacy cubemap-available flag (removed with Solid360)
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
    // Hybrid RT dummy views (owned here, destroyed with the device alive).
    if (app) {
        VkDevice device = app->getDevice();
        if (dummyRTView_ != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(dummyRTView_)) vkDestroyImageView(device, dummyRTView_, nullptr);
            dummyRTView_ = VK_NULL_HANDLE;
        }
        if (dummyRTImage_ != VK_NULL_HANDLE) { app->destroyImageWithVma(dummyRTImage_, dummyRTAlloc_, dummyRTMem_); dummyRTImage_ = VK_NULL_HANDLE; }
        if (dummySkyView_ != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(dummySkyView_)) vkDestroyImageView(device, dummySkyView_, nullptr);
            dummySkyView_ = VK_NULL_HANDLE;
        }
        if (dummySkyImage_ != VK_NULL_HANDLE) { app->destroyImageWithVma(dummySkyImage_, dummySkyAlloc_, dummySkyMem_); dummySkyImage_ = VK_NULL_HANDLE; }
    }
    if (waterRenderUBO_.buffer != VK_NULL_HANDLE) waterRenderUBO_ = {};
}

// 1x1 dummy views so set-2 bindings are never NULL (water.frag statically
// uses every binding). RT dummy: RGBA16F GENERAL (valid=0/thickness=-1 → sky
// path). Sky dummy: RGBA8 SHADER_READ (black). Created once, device-local.
// NOTE: no app transition helper serves UNDEFINED->GENERAL, so the RT dummy
// uses a direct barrier (same as RayTracingResources::createOutputImages).
void WaterRenderer::ensureDummyViews(VulkanApp* app) {
    if (!app) return;
    if (dummyRTView_ == VK_NULL_HANDLE) {
        RendererUtils::createImage2DWithVma(app->getDevice(), app, 1, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, "WaterRenderer: dummyRT",
            dummyRTImage_, dummyRTAlloc_, dummyRTMem_, dummyRTView_);
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = dummyRTImage_;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        });
        app->setImageLayoutTracked(dummyRTImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    }
    if (dummySkyView_ == VK_NULL_HANDLE) {
        RendererUtils::createImage2DWithVma(app->getDevice(), app, 1, 1,
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, "WaterRenderer: dummySky",
            dummySkyImage_, dummySkyAlloc_, dummySkyMem_, dummySkyView_);
        // Direct barrier (no helper serves UNDEFINED->SHADER_READ for color).
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = dummySkyImage_;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        });
        app->setImageLayoutTracked(dummySkyImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
    }
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

        // Back-face depth image will be created by SceneRenderer-owned WaterBackFaceRenderer
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
    // reset above frees them; just drop the dangling handles. No write cache
    // exists to clear: bindings are rewritten unconditionally (descriptor-
    // buffer style plain writes), so a reused handle is always rewritten.
    for (uint32_t i = 0; i < FRAMES; ++i) waterDepthDescriptorSets[i] = VK_NULL_HANDLE;
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

void WaterRenderer::createWaterPipelines(VulkanApp* app, const std::vector<WaterParams>& waterParams) {
    VkDevice device = app->getDevice();

    // Idempotent: this can be (re)entered if WaterRenderer::init runs more than once
    // (e.g. scene reload). Create the layout + main pipeline exactly once and keep
    // them stable. Resizes recreate via a dedicated path.
    if (waterGeometryPipelineLayout != VK_NULL_HANDLE) {
        initializeWaterParamsBuffer(waterParams);
        return;
    }

    // Water params buffer is already assigned in init
    initializeWaterParamsBuffer(waterParams);
    std::cout << "[WaterRenderer] Initialized water params buffer from provided layer state" << std::endl;
    
    // Create descriptor set layout for scene textures (set 2).
    // The water pass is fully decoupled from the solid pass: it samples only its
    // own back-face depth (for volume thickness) plus the hybrid-RT outputs and
    // the sky equirect. It no longer reads the solid color or depth targets,
    // so it has no dependency on the solid pass and can be recorded/rendered on its
    // own command buffer in parallel with the solid pass. Occlusion against solids
    // is resolved at the composite stage (postprocess.frag).
    // Hybrid RT set-2 bindings (legacy 360 cubemap removed):
    //   0 = Water back-face depth (volume thickness, raster)
    //   1 = RT pipeline reflection output (GENERAL layout, 1-frame latency)
    //   2 = RT pipeline refraction + thickness output (GENERAL layout)
    //   3 = Sky equirect (RT miss fallback + non-RT path)
    std::array<VkDescriptorSetLayoutBinding, 4> sceneBindings{};

    // Water back-face depth (binding 0) — for water volume thickness.
    // Also sampled by the tessellation evaluation shader (VUID 07988).
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    sceneBindings[0].pImmutableSamplers = nullptr;

    // RT reflection output (binding 1) — half-res pipeline image (GENERAL).
    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[1].pImmutableSamplers = nullptr;

    // RT refraction + thickness output (binding 2) — GENERAL layout.
    sceneBindings[2].binding = 2;
    sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[2].descriptorCount = 1;
    sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[2].pImmutableSamplers = nullptr;

    // Sky equirect (binding 3) — RT miss fallback.
    sceneBindings[3].binding = 3;
    sceneBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[3].descriptorCount = 1;
    sceneBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[3].pImmutableSamplers = nullptr;

    VkDescriptorBindingFlags bindingFlags[4] = {
        0, 0, 0, 0
    };

    DescriptorAllocator descAlloc{device, app};
    waterDepthDescriptorSetLayout = descAlloc.createLayout(
        sceneBindings.data(), static_cast<uint32_t>(sceneBindings.size()),
        0,
        bindingFlags,
        "WaterRenderer: waterDepthDescriptorSetLayout");

    // Pool for the per-frame main scene-texture set.  Each frame's set is
    // allocated once and updated in-place via vkUpdateDescriptorSets (no
    // per-frame alloc/free).  The pool is bulk-reset on swapchain recreate.
    // Capacity covers the 3 main sets (4 samplers each) with headroom.
    VkDescriptorPoolSize wrPoolSize = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 50};
    waterDepthDescriptorPool = descAlloc.createPool(
        &wrPoolSize, 1, 10,
        0,
        "WaterRenderer: waterDepthDescriptorPool");

    // Descriptor sets are allocated and updated per-frame in
    // prepareSceneTexturesForFrame() after scene images are created
    
    // Create a custom pipeline layout for water that includes:
    // Set 0: Material SSBO (from app->getMaterialDescriptorSetLayout())
    // Set 1: UBO (from app->getDescriptorSetLayout())
    // Set 2: Scene depth texture (waterDepthDescriptorSetLayout)
    // Descriptor set ordering: set 0 = global UBO+samplers, set 1 = material set, set 2 = scene depth textures
    std::array<VkDescriptorSetLayout, 3> waterSetLayouts = {
        app->getDescriptorSetLayout(),           // Set 0: UBO + samplers
        app->getMaterialDescriptorSetLayout(),   // Set 1: Materials
        waterDepthDescriptorSetLayout            // Set 2: Scene depth texture
    };
    
    // No per-mesh model push-constants are used for water (shaders use identity/no model push-constant).

    std::cout << "[WaterRenderer] Created water pipeline layout with 3 descriptor sets" << std::endl;

    // Create water geometry pipeline with dedicated water shaders.
    // Hybrid RT: the RT variant samples the pipeline outputs + inline ray
    // queries (same lighting/CSM/water otherwise); non-RT hardware uses the
    // sky-equirect fallback variant.
    const char* waterFragPath = (app && app->rayTracingEnabled())
        ? "shaders/water_rt.frag.spv" : "shaders/water.frag.spv";
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/water.vert.spv");
    VkShaderModule fragModule = app->getOrCreateShaderModule(waterFragPath);
    VkShaderModule tescModule = VK_NULL_HANDLE;
    VkShaderModule teseModule = VK_NULL_HANDLE;
    bool hasTessellation = true;
    tescModule = app->getOrCreateShaderModule("shaders/water.tesc.spv");
    teseModule = app->getOrCreateShaderModule("shaders/water.tese.spv");

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";
    shaderStages.push_back(vertStage);

    if (hasTessellation) {
        VkPipelineShaderStageCreateInfo tescStage{};
        tescStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tescStage.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        tescStage.module = tescModule;
        tescStage.pName = "main";
        shaderStages.push_back(tescStage);

        VkPipelineShaderStageCreateInfo teseStage{};
        teseStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        teseStage.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        teseStage.module = teseModule;
        teseStage.pName = "main";
        shaderStages.push_back(teseStage);
    }

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    shaderStages.push_back(fragStage);

    // Vertex input (same as main pipeline)
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    auto attrDescs = vk_layouts::defaultAttributes();

    // --- Create pipeline layout manually ---
    VkPipelineLayoutCreateInfo waterLayoutInfo{};
    waterLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    waterLayoutInfo.setLayoutCount = static_cast<uint32_t>(waterSetLayouts.size());
    waterLayoutInfo.pSetLayouts = waterSetLayouts.data();
    waterLayoutInfo.pushConstantRangeCount = 0;
    waterLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device, &waterLayoutInfo, nullptr, &waterGeometryPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water geometry pipeline layout!");
    }
    app->resources.addPipelineLayout(waterGeometryPipelineLayout, "WaterRenderer: waterGeometryPipelineLayout");

    // --- Create pipeline (dynamic rendering, 1 color attachment VK_FORMAT_R32G32B32A32_SFLOAT) ---
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = hasTessellation ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;   // Enable water-against-water occlusion in this pass
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Water render pass has 1 color attachment — need a blend state for it
    std::array<VkPipelineColorBlendAttachmentState, 1> colorBlendAttachments{};
    for (auto& att : colorBlendAttachments) {
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        att.blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlending.pAttachments = colorBlendAttachments.data();

    VkPipelineTessellationStateCreateInfo tessState{};
    if (hasTessellation) {
        tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    // Dynamic rendering (no render pass)
    VkFormat waterColorFmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &waterColorFmt;
    pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    pipelineInfo.pNext = &pipelineRenderingInfo;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = waterGeometryPipelineLayout;
    pipelineInfo.subpass = 0;
    if (hasTessellation) pipelineInfo.pTessellationState = &tessState;

    if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &pipelineInfo, nullptr, &waterGeometryPipeline) != VK_SUCCESS) {
        std::cerr << "[WaterRenderer] Warning: Failed to create water geometry pipeline" << std::endl;
        waterGeometryPipeline = VK_NULL_HANDLE;
    } else {
        app->resources.addPipeline(waterGeometryPipeline, "WaterRenderer: waterGeometryPipeline");
        std::cout << "[WaterRenderer] Created water geometry pipeline (dynamic rendering, 1 color attachment)" << std::endl;
    }

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
    if (tescModule) tescModule = VK_NULL_HANDLE;
    if (teseModule) teseModule = VK_NULL_HANDLE;

    // Back-face pipeline creation moved to WaterBackFaceRenderer
}

void WaterRenderer::beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex, bool loadExisting) {
    if (waterGeometryPipeline == VK_NULL_HANDLE) return;
    if (frameIndex >= 3) return;
    if (waterDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    activeWaterFrameIndex = frameIndex;

    // Batched begin barriers (single vkCmdPipelineBarrier2 for color+depth;
    // was: one call per image). Transitions: water color SHADER_READ_ONLY →
    // COLOR_ATTACHMENT_OPTIMAL (water pipeline writes EVSM output) and water
    // geometry depth tracked → DEPTH_STENCIL_ATTACHMENT_OPTIMAL (occlusion
    // testing). Same stage/access mapping as the single transitions; entries
    // already in the target layout (e.g. depth re-entered with LOAD ops for
    // the brush-liquid overlay) resolve to no-ops inside the same call.
    {
        std::vector<VulkanApp::BatchTransition> batch;
        batch.reserve(2);
        VulkanApp::BatchTransition colorBegin{};
        colorBegin.image     = waterDepthImages[frameIndex];
        colorBegin.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
        colorBegin.oldLayout = waterDepthImageLayouts[frameIndex];
        colorBegin.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorBegin.mipLevels = 1;
        batch.push_back(colorBegin);
        if (waterGeomDepthImages[frameIndex] != VK_NULL_HANDLE) {
            VulkanApp::BatchTransition depthBegin{};
            depthBegin.image     = waterGeomDepthImages[frameIndex];
            depthBegin.format    = VK_FORMAT_D32_SFLOAT;
            depthBegin.oldLayout = waterGeomDepthImageLayouts[frameIndex];
            depthBegin.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthBegin.mipLevels = 1;
            batch.push_back(depthBegin);
        }
        appPtr->recordTransitionBatch(cmd, batch);
    }

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = waterDepthImageViews[frameIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // LOAD preserves the main water EVSM output when this pass overlays brush
    // liquid on top; CLEAR (default) starts a fresh water target.
    colorAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = waterGeomDepthImageViews[frameIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // Clear depth to 1.0 so every water fragment passes the depth test
    // (self-occlusion only). Forward-pass depth-test handles solid occlusion.
    // No scene-depth copy is needed. When LOADing, the main water geom depth is
    // preserved so the brush overlay depth-tests against it (storeOp STORE keeps
    // the overlay visible to the composite).
    depthAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    // STORE (not DONT_CARE) so the geom depth survives the pass: the composite samples
    // it (postprocess.frag binding 7) and the brush-liquid overlay re-enters this pass
    // with LOAD ops, depth-testing against the main water geometry written here.
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {renderWidth, renderHeight};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void WaterRenderer::endWaterRendering(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;
    vkCmdEndRendering(cmd);
}

void WaterRenderer::endWaterGeometryPass(VkCommandBuffer cmd) {
    endWaterRendering(cmd);

    uint32_t frameIndex = activeWaterFrameIndex;
    if (waterDepthImages[frameIndex] != VK_NULL_HANDLE && appPtr) {
        // Barrier: transition water color output from COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        // after the geometry pass so the forward swapchain pass can sample it.
        appPtr->recordTransitionImageLayoutLayer(cmd, waterDepthImages[frameIndex],
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, 0, 1);
        waterDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

void WaterRenderer::endWaterGeometryPassWithDepth(VkCommandBuffer cmd, uint32_t frameIndex) {
    endWaterRendering(cmd);
    if (!appPtr || waterDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    // Batched end barriers (single vkCmdPipelineBarrier2 for color+depth):
    // water color COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (sampled
    // by the composite) together with the water geometry depth
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (sampled by
    // the composite at postprocess binding 7). Was: endWaterGeometryPass plus
    // a second lone depth transition (two calls). Same mapping as the single
    // transitions; the geom depth shares the pass boundary, so one call covers
    // both resources.
    std::vector<VulkanApp::BatchTransition> batch;
    batch.reserve(2);
    VulkanApp::BatchTransition colorEnd{};
    colorEnd.image     = waterDepthImages[frameIndex];
    colorEnd.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
    colorEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorEnd.mipLevels = 1;
    batch.push_back(colorEnd);
    if (waterGeomDepthImages[frameIndex] != VK_NULL_HANDLE) {
        VulkanApp::BatchTransition depthEnd{};
        depthEnd.image     = waterGeomDepthImages[frameIndex];
        depthEnd.format    = VK_FORMAT_D32_SFLOAT;
        depthEnd.oldLayout = waterGeomDepthImageLayouts[frameIndex];
        depthEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthEnd.mipLevels = 1;
        batch.push_back(depthEnd);
    }
    appPtr->recordTransitionBatch(cmd, batch);
    waterDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (waterGeomDepthImages[frameIndex] != VK_NULL_HANDLE)
        waterGeomDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// Back-face pass is owned and executed by SceneRenderer via its WaterBackFaceRenderer.

void WaterRenderer::updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, uint32_t frameIndex,
                                               VkImageView backFaceDepthView,
                                               VkImageView rtReflectView, VkImageView rtRefractView,
                                               VkImageView skyView) {
    if (ds == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return;
    }
    (void)frameIndex;
    ensureDummyViews(app);
    if (nearestSampler == VK_NULL_HANDLE) return;

    // Every binding is statically used by water.frag — never leave NULL.
    // Missing RT outputs (unsupported/disabled) bind 1x1 GENERAL dummies
    // (valid flag 0 / thickness -1 → shader takes the sky/inline path);
    // missing sky binds a 1x1 SHADER_READ dummy.
    VkImageView effReflect = (rtReflectView != VK_NULL_HANDLE) ? rtReflectView : dummyRTView_;
    VkImageView effRefract = (rtRefractView != VK_NULL_HANDLE) ? rtRefractView : dummyRTView_;
    VkImageView effSky = (skyView != VK_NULL_HANDLE) ? skyView : dummySkyView_;
    if (backFaceDepthView == VK_NULL_HANDLE || effReflect == VK_NULL_HANDLE ||
        effRefract == VK_NULL_HANDLE || effSky == VK_NULL_HANDLE) {
        return;
    }

    std::array<VkDescriptorImageInfo, 4> imageInfos{};

    // Water back-face depth (binding 0). Nearest filtering: no interpolation
    // across geometry edges.
    imageInfos[0].sampler = nearestSampler;
    imageInfos[0].imageView = backFaceDepthView;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // RT pipeline outputs (bindings 1-2, GENERAL layout — matches the RT
    // pipeline's storage writes; no layout churn between dispatch + sample).
    imageInfos[1].sampler = linearSampler;
    imageInfos[1].imageView = effReflect;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[2].sampler = linearSampler;
    imageInfos[2].imageView = effRefract;
    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Sky equirect (binding 3, RT miss fallback).
    imageInfos[3].sampler = linearSampler;
    imageInfos[3].imageView = effSky;
    imageInfos[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Descriptor-buffer style update: rewrite the bindings unconditionally.
    // No write cache, no set allocation/free, no deferred destruction — the
    // per-frame set is allocated once (see prepareSceneTexturesForFrame) and
    // updated in place here. With VK_EXT_descriptor_buffer this same call
    // becomes plain host memory writes (vkGetDescriptorEXT); the classic
    // vkUpdateDescriptorSets below is the fallback until the layout carries
    // VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT.
    DescriptorWriter writer(app->getDevice());
    for (uint32_t i = 0; i < 4; ++i) {
        writer.writeImage(ds, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          imageInfos[i].sampler, imageInfos[i].imageView,
                          imageInfos[i].imageLayout);
    }
    writer.flush();
}

VkDescriptorSet WaterRenderer::prepareSceneTexturesForFrame(VulkanApp* app, uint32_t frameIndex,
                                                            VkImageView backFaceDepthView,
                                                            VkImageView rtReflectView,
                                                            VkImageView rtRefractView,
                                                            VkImageView skyView) {
    if (app == nullptr || waterDepthDescriptorPool == VK_NULL_HANDLE ||
        waterDepthDescriptorSetLayout == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    if (frameIndex >= FRAMES) return VK_NULL_HANDLE;

    // Allocate this slot's descriptor set once; subsequent frames just rebind
    // the (potentially different) image views via vkUpdateDescriptorSets.
    // The pool is reset on swapchain recreate, which drops all sets, so
    // allocation only happens after a fresh pool is available.
    if (waterDepthDescriptorSets[frameIndex] == VK_NULL_HANDLE) {
        DescriptorAllocator descAlloc{app->getDevice(), app};
        descAlloc.allocateSets(waterDepthDescriptorPool, waterDepthDescriptorSetLayout, 1,
                               &waterDepthDescriptorSets[frameIndex],
                               "WaterRenderer: waterDepthDescriptorSet");
        if (waterDepthDescriptorSets[frameIndex] == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    }

    updateSceneTexturesBinding(app, waterDepthDescriptorSets[frameIndex], frameIndex,
                               backFaceDepthView, rtReflectView, rtRefractView, skyView);
    return waterDepthDescriptorSets[frameIndex];
}

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
        gpu.reserved3 = glm::vec4(0.0f); // legacy cubemap-available flag (removed with Solid360)
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

void WaterRenderer::prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView sceneColorView, VkImageView skyView) {
    if (!app || cmd == VK_NULL_HANDLE) return;

    // Dynamic parameter updates are performed explicitly from the upper level
    // via updateGPUParamsForLayer().



    // Memory barrier: ensure COLOR_ATTACHMENT_OUTPUT + depth writes from the
    // solid/back-face passes are visible to shader reads in the water pass.
    // The solid render pass already images to SHADER_READ_ONLY_OPTIMAL via
    // explicit endPass barriers, but we need an execution + memory dependency
    // between the two command sequences on the same command buffer. The
    // tessellation evaluation shader also samples the back-face depth (set 2
    // binding 2) for volume bump modulation, so it must be included in the
    // destination stage mask alongside the fragment shader.
    VkMemoryBarrier2 memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    // The back-face pass writes its depth image at EARLY_FRAGMENT_TESTS stage
    // (depth-only pre-pass), so the source mask must include EARLY_FRAGMENT_TESTS
    // (not just LATE) to establish the dependency for that write.
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &memBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

// Back-face pass implementation moved to WaterBackFaceRenderer
void WaterRenderer::render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView sceneColorView, VkImageView skyView, IndirectRenderer* secondaryIR, VkDescriptorSet overrideWaterDs) {
    if (!app || cmd == VK_NULL_HANDLE) return;

    prepareRender(app, cmd, frameIndex, sceneColorView, skyView);

    // Back-face pre-pass is executed by SceneRenderer's WaterBackFaceRenderer
    // before calling WaterRenderer::render. No-op here.

    beginWaterGeometryPass(cmd, frameIndex);

    // Bind descriptor sets (shared between depth pre-pass and main pass)
    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    if (mainDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd,
            waterGeometryPipelineLayout, 0, 1, &mainDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            waterGeometryPipelineLayout, 0, 1, &mainDs, 0, nullptr);
    }
    // In the async path the caller owns the set-2 set (already populated on the
    // host before submission); the per-frame set is only populated in the
    // non-async path (see renderPass). Binding the per-frame set unconditionally
    // here used to leave set 2 unbound in the async path (it is never allocated
    // there), so the water pass accidentally inherited whatever the back-face
    // pass left bound — whose binding 0 is patched to the dummy depth.
    VkDescriptorSet sceneDs = (overrideWaterDs != VK_NULL_HANDLE) ? overrideWaterDs : getWaterDepthDescriptorSet(frameIndex);
    if (sceneDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd,
            waterGeometryPipelineLayout, 2, 1, &sceneDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            waterGeometryPipelineLayout, 2, 1, &sceneDs, 0, nullptr);
    }

    // Main geometry pass
    if (waterGeometryPipeline != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsPipeline(cmd, waterGeometryPipeline);
        else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterGeometryPipeline);
        waterIndirectRenderer.drawPrepared(cmd);
        // Brush liquid water: same pipeline, same descriptor sets, its own IR.
        // Drawn right after the main water so brush liquid depth/color lands in
        // the same offscreen target (self-occlusion tested by water depth).
        if (secondaryIR) secondaryIR->drawPrepared(cmd);
    }

    endWaterGeometryPass(cmd);
}

void WaterRenderer::renderBrushLiquid(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView skyView, VkDescriptorSet overrideWaterDs) {
    if (!app || cmd == VK_NULL_HANDLE || !brushRenderer_) return;
    if (frameIndex >= 3) return;
    if (waterDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    VkImageView sceneColorView = solidRenderer_ ? solidRenderer_->getColorView(frameIndex) : VK_NULL_HANDLE;
    prepareRender(app, cmd, frameIndex, sceneColorView, skyView);

    // Re-enter the water geometry pass with LOAD ops so the main water EVSM color
    // and geom depth are preserved; the brush liquid draws on top of them.
    beginWaterGeometryPass(cmd, frameIndex, /*loadExisting=*/true);

    VkPipeline waterPipe = getWaterGeometryPipeline();
    VkPipelineLayout waterLayout = getWaterGeometryPipelineLayout();
    if (waterPipe != VK_NULL_HANDLE && waterLayout != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsPipeline(cmd, waterPipe);
        else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterPipe);

        VkDescriptorSet mainDs = app->getMainDescriptorSet();
        if (mainDs != VK_NULL_HANDLE) {
            if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, waterLayout, 0, 1, &mainDs, 0, nullptr);
            else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterLayout, 0, 1, &mainDs, 0, nullptr);
        }
        VkDescriptorSet sceneDs = (overrideWaterDs != VK_NULL_HANDLE) ? overrideWaterDs : getWaterDepthDescriptorSet(frameIndex);
        if (sceneDs != VK_NULL_HANDLE) {
            if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, waterLayout, 2, 1, &sceneDs, 0, nullptr);
            else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterLayout, 2, 1, &sceneDs, 0, nullptr);
        }
        // Brush liquid water: same pipeline/descriptor sets, its own IndirectRenderer.
        brushRenderer_->getLiquidIR().drawPrepared(cmd);
    }

    // Merged end: water color + water geometry depth (sampled by the composite)
    // transition in a single barrier call (was: endWaterGeometryPass plus a
    // second lone depth transition).
    endWaterGeometryPassWithDepth(cmd, frameIndex);
}

void WaterRenderer::renderPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t frameIdx,
                               bool waterWireframeEnabled, float waterTime, VkImageView skyView,
                               VkDescriptorSet overrideWaterDs, bool drawBrushLiquid) {
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[WaterRenderer::renderPass] commandBuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    // Update the water render UBO with the active layer time value.
    if (waterRenderUBO_.buffer != VK_NULL_HANDLE) {
        WaterRenderUBO renderUbo{};
        renderUbo.timeParams = glm::vec4(waterTime, 0.0f, 0.0f, 0.0f);
        void* data = nullptr;
        data = waterRenderUBO_.map(0);
        memcpy(data, &renderUbo, sizeof(WaterRenderUBO));
        waterRenderUBO_.unmap(); // VMA persistent mapping
    }

    // Record the water offscreen work on the same command buffer so the solid
    // pass outputs are available for sampling.
    VkImageView sceneColorView = solidRenderer_ ? solidRenderer_->getColorView(frameIdx) : VK_NULL_HANDLE;
    // NOTE: Water no longer samples the solid depth image. Occlusion against
    // solids is resolved at the composite stage (postprocess.frag), so the water
    // pass has no dependency on the solid depth target and can be recorded
    // independently (e.g. on its own command buffer / queue) in parallel with
    // the solid pass.
    // When `overrideWaterDs` is provided (async path), the caller owns the set
    // and has already populated it on the host before submitting; we must NOT call
    // prepareSceneTexturesForFrame here (that would update a different, possibly
    // in-flight, descriptor set — VUID-vkUpdateDescriptorSets-None-03047).
    if (overrideWaterDs == VK_NULL_HANDLE) {
        VkImageView wBack = (backFaceRenderer_) ? backFaceRenderer_->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
        VkImageView wRefl = VK_NULL_HANDLE, wRefr = VK_NULL_HANDLE;
        if (rtResources_ && rtResources_->isSupported()) {
            wRefl = rtResources_->getReflectionView();
            wRefr = rtResources_->getRefractionView();
        }
        prepareSceneTexturesForFrame(app, frameIdx, wBack, wRefl, wRefr, skyView);
    }

    // Scene textures were already bound before the async back-face task was
    // launched (see main.cpp), so we must NOT call updateSceneTexturesBinding here.
    // Calling it after the async tasks submit their command buffers would update a
    // descriptor set that is already referenced by a pending command buffer
    // (VUID-vkUpdateDescriptorSets-None-03047).

    bool wf = waterWireframeEnabled;
    if (wf && waterWireframe_ && waterWireframe_->getPipeline() != VK_NULL_HANDLE) {
        // Wireframe path: use WaterRenderer for setup/pass management,
        // but bind the wireframe pipeline instead of the normal one.
        prepareRender(app, commandBuffer, frameIdx, sceneColorView, skyView);
        beginWaterGeometryPass(commandBuffer, frameIdx);

        // First render filled water geometry to populate the water depth
        // buffer so the wireframe can depth-test against actual water depth.
        VkPipeline waterPipe = getWaterGeometryPipeline();
        VkPipelineLayout waterLayout = getWaterGeometryPipelineLayout();
        if (waterPipe != VK_NULL_HANDLE && waterLayout != VK_NULL_HANDLE) {
            if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, waterPipe);

            VkDescriptorSet mainDs = app->getMainDescriptorSet();
            if (mainDs != VK_NULL_HANDLE) {
                if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, waterLayout, 0, 1, &mainDs, 0, nullptr);
            }

            VkDescriptorSet sceneDs = (overrideWaterDs != VK_NULL_HANDLE) ? overrideWaterDs : getWaterDepthDescriptorSet(frameIdx);
            if (sceneDs != VK_NULL_HANDLE) {
                if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, waterLayout, 2, 1, &sceneDs, 0, nullptr);
            }

            // Draw filled water geometry (will update depth buffer)
            getIndirectRenderer().drawPrepared(commandBuffer);
            if (drawBrushLiquid && brushRenderer_) brushRenderer_->getLiquidIR().drawPrepared(commandBuffer);
        }

        // Draw wireframe overlay on top, inside the same render pass,
        // reusing the depth buffer populated by the filled geometry pass.
        // Bind descriptor sets individually with null checks (same pattern
        // as the filled water pipeline) to handle missing sets gracefully.
        VkPipeline waterWfPipe = waterWireframe_->getPipeline();
        VkPipelineLayout wfLayout = waterWireframe_->getPipelineLayout();
        if (waterWfPipe != VK_NULL_HANDLE && wfLayout != VK_NULL_HANDLE) {
            if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, waterWfPipe);

            VkDescriptorSet wfMainDs = app->getMainDescriptorSet();
            if (wfMainDs != VK_NULL_HANDLE)
                if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, wfLayout, 0, 1, &wfMainDs, 0, nullptr);

            VkDescriptorSet wfDepthDs = (overrideWaterDs != VK_NULL_HANDLE) ? overrideWaterDs : getWaterDepthDescriptorSet(frameIdx);
            if (wfDepthDs != VK_NULL_HANDLE)
                if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, wfLayout, 2, 1, &wfDepthDs, 0, nullptr);

            getIndirectRenderer().drawPrepared(commandBuffer);
            if (drawBrushLiquid && brushRenderer_) brushRenderer_->getLiquidIR().drawPrepared(commandBuffer);
        }

        endWaterGeometryPass(commandBuffer);
    } else {
        render(app, commandBuffer, frameIdx, sceneColorView, skyView,
               (drawBrushLiquid && brushRenderer_) ? &brushRenderer_->getLiquidIR() : nullptr,
               overrideWaterDs);
    }

    // Post-processing runs inside the active main render pass; the caller
    // (e.g. MyApp::draw) invokes `postProcessRenderer->render` with valid
    // scene/water views when available. This function focuses on executing
    // offscreen geometry and returning control to the main pass.
}
