
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
#include <algorithm>
#include <cmath>
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
static VkImageLayout waterBodyImageLayouts[VulkanApp::MAX_FRAMES_IN_FLIGHT] = {};
static VkImageLayout waterColumnImageLayouts[VulkanApp::MAX_FRAMES_IN_FLIGHT] = {};

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

namespace {

// CPU/UI stores feature PERIODS (world units); the shader consumes spatial
// scales (features per world unit). The C++ WaterParamsGPU values therefore
// stay in PERIODS too and are converted exactly once, at the buffer upload
// boundary, by waterGpuPeriodsToScales() below.
float waterPeriodToScale(float period) {
    return period > 0.0f ? 1.0f / period : 0.0f;
}

// Convert the period-valued slots of a packed GPU block into the shader's
// spatial scales. Called only right before the buffer write.
void waterGpuPeriodsToScales(WaterParamsGPU& gpu) {
    gpu.params2.y = waterPeriodToScale(gpu.params2.y);
    gpu.waveComponent1.x = waterPeriodToScale(gpu.waveComponent1.x);
    gpu.waveComponent2.x = waterPeriodToScale(gpu.waveComponent2.x);
    gpu.waveMask.x = waterPeriodToScale(gpu.waveMask.x);
    gpu.foamNoise.x = waterPeriodToScale(gpu.foamNoise.x);
}

// Single source of truth for CPU -> GPU water parameter packing. Shared by
// the one-time buffer initialization and the runtime widget updates so both
// paths can never drift apart. Field meanings are documented in
// vulkan/ubo/WaterParamsGPU.hpp and mirrored by the GLSL struct.
WaterParamsGPU makeWaterParamsGPU(const WaterParams& p) {
    // Shore direction: 0 deg = +Z, 90 deg = +X (right-handed XZ plane).
    const float shoreAngle = glm::radians(p.shoreWaveAngle);
    const glm::vec2 shoreDir(std::sin(shoreAngle), std::cos(shoreAngle));

    WaterParamsGPU gpu{};
    gpu.params1 = glm::vec4(p.refractionStrength, p.fresnelPower, p.transparency, p.reflectionStrength);
    gpu.params2 = glm::vec4(p.waterTint, p.noisePeriod, static_cast<float>(p.noiseOctaves), p.noisePersistence);
    gpu.params3 = glm::vec4(p.noiseTimeSpeed, p.noiseLacunarity, p.specularIntensity, p.specularPower);
    gpu.glitterParams = glm::vec4(p.glitterIntensity, 0.0f, 0.0f, 0.0f);
    gpu.blurParams = glm::vec4(p.enableBlur ? 1.0f : 0.0f,
                               p.blurRadius,
                               p.blurDepthScale,
                               0.0f);
    gpu.waveParams = glm::vec4(p.tessNoiseInfluence, 0.0f, p.bumpAmplitude, p.depthFalloff);
    gpu.reserved1 = glm::vec4(p.enableReflection ? 1.0f : 0.0f,
                              p.enableRefraction ? 1.0f : 0.0f,
                              0.0f,
                              0.0f);
    gpu.reserved2 = glm::vec4(0.0f, 0.0f, 0.0f, p.uniformReflection ? 1.0f : 0.0f);
    gpu.reserved3 = glm::vec4(0.0f); // legacy cubemap-available flag (removed with Solid360)
    gpu.tessParams = glm::vec4(p.tessNearDist, p.tessFarDist, p.tessMinLevel, p.tessMaxLevel);
    gpu.causticColor = glm::vec4(p.causticColor, 0.0f);
    gpu.causticParams = glm::vec4(p.causticSoftness, p.causticIntensity, 0.0f, 0.0f);
    gpu.causticExtraParams = glm::vec4(0.0f); // wave-shape caustics: no mode/line/speed knobs
    gpu.absorptionParams = glm::vec4(p.absorption, p.absorptionScale);
    gpu.refractionParams = glm::vec4(p.ior, p.maxThickness, p.shoreFadeDepth, 0.0f);

    // Shore-wave system
    gpu.waveToggles = glm::vec4(p.enableWaves ? 1.0f : 0.0f,
                                p.enableFoam ? 1.0f : 0.0f,
                                p.enableVolumetric ? 1.0f : 0.0f, 0.0f);
    gpu.waveZones = glm::vec4(p.zoneDeepDepth, p.zoneBreakDepth, p.zoneShallowDepth, 0.0f);
    gpu.waveDirection = glm::vec4(shoreDir.x, shoreDir.y, 0.0f, 0.0f);
    gpu.waveShape = glm::vec4(p.waveSharpDeep, p.waveSharpBreak, p.waveSharpShallow, p.waveShoalGain);
    gpu.waveShoal = glm::vec4(p.waveShoalSpeed, p.waveShallowDecay, p.waveLineAmplitude, p.breakerWidth);
    gpu.waveComponent1 = glm::vec4(p.wavePeriod, p.waveSpeed, 1.0f, 0.0f);
    gpu.waveComponent2 = glm::vec4(p.crossWavePeriod, p.crossWaveSpeed, p.crossWaveAmplitude, p.crossWavePhase);
    gpu.waveBreaker = glm::vec4(p.breakerAmplitude, p.waveChopAmount, p.whitecapOnset, p.waveHeightFalloff);
    gpu.waveCurl = glm::vec4(p.breakerCurl, p.breakerCrestCurve, 0.0f, 0.0f);
    gpu.waveWarp = glm::vec4(p.waveWarpAmount, p.waveAmpVariation, p.waveRidgeStretch, p.shoreGradientStep);
    gpu.waveMask = glm::vec4(p.waveMaskPeriod, p.waveMaskThreshold, p.waveMaskSoftness, p.waveMaskSpeed);
    gpu.foamParams = glm::vec4(p.foamCrestThreshold, p.foamTrailPhase, p.foamDecay, p.foamColorAmount);
    gpu.foamNoise = glm::vec4(p.foamNoisePeriod, p.foamNoiseSpeed, p.foamNoiseAmount, p.foamShoreAmount);
    gpu.foamExtra = glm::vec4(p.foamMaskFloor, p.foamDiffuseFloor, p.foamAmbient, 0.0f);
    gpu.foamContact = glm::vec4(p.foamContactWidth, p.foamContactAmount, p.foamContactAlpha, p.foamContactFloor);
    gpu.foamShape = glm::vec4(p.foamEdge, p.foamCoverage, p.foamShoreSpeed, p.foamLagGrowth);
    gpu.foamColor = glm::vec4(p.foamColor, 0.0f);
    gpu.volumetricParams = glm::vec4(p.volumetricStrength, p.volumetricDensity, p.volumetricPhaseG, 0.0f);
    gpu.volumetricColor = glm::vec4(p.volumetricColor, 0.0f);
    gpu.regionShoreColor = glm::vec4(p.regionShoreColor, 0.0f);
    gpu.regionShallowColor = glm::vec4(p.regionShallowColor, 0.0f);
    gpu.regionBreakerColor = glm::vec4(p.regionBreakerColor, 0.0f);
    gpu.regionShoalColor = glm::vec4(p.regionShoalColor, 0.0f);
    gpu.regionDeepColor = glm::vec4(p.regionDeepColor, 0.0f);
    gpu.regionTintParams = glm::vec4(p.regionBlendSoftness,
                                     p.tintShoreFadeDepth,
                                     0.0f,
                                     0.0f);
    return gpu;
}

// Stable numeric key for a VkDescriptorSet handle (non-dispatchable handles
// are pointers on 64-bit, plain uint64_t elsewhere). Used only for the M7
// CPU-side binding caches.
uint64_t descriptorSetKey(VkDescriptorSet ds) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ds));
}

} // namespace



void WaterRenderer::refreshWaterBlurNeeded() {
    waterBlurNeeded_ = false;
    for (bool needed : layerBlurNeeded_) {
        if (needed) {
            waterBlurNeeded_ = true;
            break;
        }
    }
}

void WaterRenderer::updateGPUParamsForLayer(uint32_t layer, const WaterParams& p) {
    if (!appPtr) return;
    if (layer >= waterParamsCount) return;

    // H4: keep the per-layer blur gate in sync with the SSBO write so the
    // geometry pass can pick the single-attachment variant without a new
    // MyApp setter.
    if (static_cast<size_t>(layer) >= layerBlurNeeded_.size())
        layerBlurNeeded_.resize(static_cast<size_t>(layer) + 1, false);
    layerBlurNeeded_[layer] = p.enableBlur && p.blurRadius > 0.0f;
    refreshWaterBlurNeeded();

    WaterParamsGPU gpu = makeWaterParamsGPU(p);
    waterGpuPeriodsToScales(gpu); // periods -> shader scales, at the upload boundary

    size_t offset = static_cast<size_t>(layer) * sizeof(WaterParamsGPU);
    void* data = nullptr;
    data = waterParamsBuffer.map(offset);
    memcpy(data, &gpu, sizeof(WaterParamsGPU));
    waterParamsBuffer.unmap(); // VMA persistent mapping
}

void WaterRenderer::cleanup(VulkanApp* app) {
    waterIndirectRenderer.cleanup(app);
    destroyRenderTargets(app);
    // M7: drop every cached binding key (renderer-owned sets were already
    // erased by destroyRenderTargets; external sets end here too).
    {
        std::lock_guard<std::mutex> lock(sceneTexturesCacheMutex_);
        sceneTexturesBindingCache_.clear();
    }
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
    if (dummyWaterView_ == VK_NULL_HANDLE) {
        // Zeroed 1x1 RGBA: composite water input while water blends directly
        // into the main color target (alpha 0 → mix() keeps the base color).
        RendererUtils::createImage2DWithVma(app->getDevice(), app, 1, 1,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, "WaterRenderer: dummyWater",
            dummyWaterImage_, dummyWaterAlloc_, dummyWaterMem_, dummyWaterView_);
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            app->recordTransitionImageLayoutLayer(cmd, dummyWaterImage_, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
            VkClearColorValue clear{};
            clear.float32[0] = 0.0f; clear.float32[1] = 0.0f;
            clear.float32[2] = 0.0f; clear.float32[3] = 0.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, dummyWaterImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clear, 1, &range);
            app->recordTransitionImageLayoutLayer(cmd, dummyWaterImage_, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        });
        app->setImageLayoutTracked(dummyWaterImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
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
        waterBodyImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        waterColumnImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
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

    // Water body + column attachments (color attachments 1 and 2 of the water
    // geometry pass):
    //  * body (RGBA16F): RGB = refraction + tint body (pre-reflection),
    //    A = body weight = coverage * (1 - reflection mix).
    //  * column (RG16F): measured water depth (m) in R, per-material blur
    //    radius (px) in G.
    // The final composite blurs the body with a depth-scaled kernel and
    // re-inserts it with its stored weight, so the reflection lobe and the
    // surface effects stay sharp. Both live in SHADER_READ_ONLY between frames
    // like the water color target.
    for (uint32_t frameIdx = 0; frameIdx < FRAMES; ++frameIdx) {
        createImage(VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    waterBodyImages[frameIdx], waterBodyAllocations[frameIdx], waterBodyMemories[frameIdx], waterBodyImageViews[frameIdx]);
        waterBodyImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (waterBodyImages[frameIdx] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayerForce(waterBodyImages[frameIdx], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(waterBodyImages[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            waterBodyImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        createImage(VK_FORMAT_R16G16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    waterColumnImages[frameIdx], waterColumnAllocations[frameIdx], waterColumnMemories[frameIdx], waterColumnImageViews[frameIdx]);
        waterColumnImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (waterColumnImages[frameIdx] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayerForce(waterColumnImages[frameIdx], VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(waterColumnImages[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            waterColumnImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
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
        waterBodyImages[i] = VK_NULL_HANDLE;
        waterBodyAllocations[i] = VK_NULL_HANDLE;
        waterBodyMemories[i] = VK_NULL_HANDLE;
        waterBodyImageViews[i] = VK_NULL_HANDLE;
        waterColumnImages[i] = VK_NULL_HANDLE;
        waterColumnAllocations[i] = VK_NULL_HANDLE;
        waterColumnMemories[i] = VK_NULL_HANDLE;
        waterColumnImageViews[i] = VK_NULL_HANDLE;
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
    // reset above frees them. M7: drop their cached binding keys (and the
    // back-face binding-0 patch entries via invalidateSceneTexturesBinding)
    // BEFORE dropping the handles, so a later set that recycles a freed handle
    // value can never hit a stale cache entry. A reused handle is then always
    // rewritten because the cache has no entry for it.
    for (uint32_t i = 0; i < FRAMES; ++i)
        invalidateSceneTexturesBinding(waterDepthDescriptorSets[i]);
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
    VkImage bodyImg = waterBodyImages[frameIndex];
    VkImage columnImg = waterColumnImages[frameIndex];
    if (colorImg == VK_NULL_HANDLE && depthImg == VK_NULL_HANDLE &&
        bodyImg == VK_NULL_HANDLE && columnImg == VK_NULL_HANDLE) return;

    // Transition all images to TRANSFER_DST_OPTIMAL for the clear.
    if (colorImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, colorImg, VK_FORMAT_R32G32B32A32_SFLOAT,
            waterDepthImageLayouts[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    }
    if (depthImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT,
            waterGeomDepthImageLayouts[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    }
    if (bodyImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, bodyImg, VK_FORMAT_R16G16B16A16_SFLOAT,
            waterBodyImageLayouts[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    }
    if (columnImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, columnImg, VK_FORMAT_R16G16_SFLOAT,
            waterColumnImageLayouts[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    }

    // Clear water color + body + column to transparent black and depth to 1.0.
    VkClearColorValue clearValue{};
    clearValue.float32[0] = 0.0f; clearValue.float32[1] = 0.0f;
    clearValue.float32[2] = 0.0f; clearValue.float32[3] = 0.0f;
    if (colorImg != VK_NULL_HANDLE) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, colorImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
    }
    if (bodyImg != VK_NULL_HANDLE) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, bodyImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
    }
    if (columnImg != VK_NULL_HANDLE) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, columnImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
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
    if (bodyImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, bodyImg, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        waterBodyImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (columnImg != VK_NULL_HANDLE) {
        app->recordTransitionImageLayoutLayer(cmd, columnImg, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        waterColumnImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

VkImageLayout WaterRenderer::getWaterGeomDepthLayout(uint32_t frameIndex) const {
    if (frameIndex < 3) return waterGeomDepthImageLayouts[frameIndex];
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

void WaterRenderer::setWaterGeomDepthLayout(uint32_t frameIndex, VkImageLayout layout) {
    if (frameIndex < 3) waterGeomDepthImageLayouts[frameIndex] = layout;
}

VkImageLayout WaterRenderer::getWaterBodyLayout(uint32_t frameIndex) const {
    if (frameIndex < FRAMES) return waterBodyImageLayouts[frameIndex];
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

void WaterRenderer::setWaterBodyLayout(uint32_t frameIndex, VkImageLayout layout) {
    if (frameIndex < FRAMES) waterBodyImageLayouts[frameIndex] = layout;
}

VkImageLayout WaterRenderer::getWaterColumnLayout(uint32_t frameIndex) const {
    if (frameIndex < FRAMES) return waterColumnImageLayouts[frameIndex];
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

void WaterRenderer::setWaterColumnLayout(uint32_t frameIndex, VkImageLayout layout) {
    if (frameIndex < FRAMES) waterColumnImageLayouts[frameIndex] = layout;
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
    //   4 = Solid pass HDR color (screen-space reflection refinement)
    //   5 = Solid pass depth (SSR march + occlusion test)
    //   6 = Vegetation color (reflection lookup — grass/billboards)
    //   7 = Vegetation depth (front test for the vegetation layer)
    std::array<VkDescriptorSetLayoutBinding, 8> sceneBindings{};

    // Water back-face depth (binding 0) — for water volume thickness.
    // Also sampled by the tessellation evaluation shader and by the
    // WATER_NO_TESS vertex path (C1), which measures the same per-vertex
    // depth/shore direction (VUID 07988).
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
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

    // Solid pass HDR color (binding 4) — SSR hit color (linear, pre-tonemap).
    sceneBindings[4].binding = 4;
    sceneBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[4].descriptorCount = 1;
    sceneBindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[4].pImmutableSamplers = nullptr;

    // Solid pass depth (binding 5) — SSR march target + occlusion test, and
    // the water-depth bottom the TES samples for the shore-wave regions
    // (the water volume's own back face sits an SDF bias below the terrain).
    // The WATER_NO_TESS vertex path (C1) samples it too, for the same
    // per-vertex shore-zone depth/shore-direction measurement.
    sceneBindings[5].binding = 5;
    sceneBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[5].descriptorCount = 1;
    sceneBindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    sceneBindings[5].pImmutableSamplers = nullptr;

    // Vegetation color (binding 6) — reflection lookup over the grass layer.
    sceneBindings[6].binding = 6;
    sceneBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[6].descriptorCount = 1;
    sceneBindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[6].pImmutableSamplers = nullptr;

    // Vegetation depth (binding 7) — vegetation is used only when in front of
    // the reflected hit point.
    sceneBindings[7].binding = 7;
    sceneBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[7].descriptorCount = 1;
    sceneBindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[7].pImmutableSamplers = nullptr;

    VkDescriptorBindingFlags bindingFlags[8] = {
        0, 0, 0, 0, 0, 0, 0, 0
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

    // Create water geometry pipelines with dedicated water shaders.
    // Hybrid RT: BOTH fragment variants are built up front — the RT one
    // (pipeline outputs + inline ray queries) and the non-RT one (sky
    // fallbacks). The runtime selector binds whichever matches the enabled
    // ray paths, so a fully disabled RT configuration never pays the
    // ray-query shader's register/occupancy cost. Phase-1: all four stages
    // are the merged main.* sources built with WATER_MODE=1.
    //
    // C1 (perf report 19): a SECOND pipeline family is built for the
    // non-tessellation case (Settings::tessellationEnabled == false): the
    // same fragment modules and pipeline layout, but TRIANGLE_LIST topology,
    // the WATER_NO_TESS vertex module and no TCS/TES/tessellation state, so
    // the Minimal preset never runs the tessellator or the TES per-vertex
    // wave/depth work.
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/main_water.vert.spv");
    VkShaderModule vertNoTessModule = app->getOrCreateShaderModule("shaders/main_water_no_tess.vert.spv");
    VkShaderModule fragNoRtModule = app->getOrCreateShaderModule("shaders/main_water.frag.spv");
    VkShaderModule fragRtModule = (app && app->rayTracingEnabled())
        ? app->getOrCreateShaderModule("shaders/main_water_rt.frag.spv") : VK_NULL_HANDLE;
    // H4 single-attachment variants: same water fragment stage compiled with
    // WATER_NO_BODY, i.e. without the body/column aux outputs. Selected while
    // no layer requests the final-pass blur.
    VkShaderModule fragNoBodyModule = app->getOrCreateShaderModule("shaders/main_water_nobody.frag.spv");
    VkShaderModule fragRtNoBodyModule = (app && app->rayTracingEnabled())
        ? app->getOrCreateShaderModule("shaders/main_water_rt_nobody.frag.spv") : VK_NULL_HANDLE;
    VkShaderModule tescModule = VK_NULL_HANDLE;
    VkShaderModule teseModule = VK_NULL_HANDLE;
    // RT TES: same water TES plus the optional inline ray-query water-region
    // depth (rt.waterDepth). Used only by the RT pipeline variant; the RT
    // fragment variant and this TES are selected together.
    VkShaderModule teseRtModule = (app && app->rayTracingEnabled())
        ? app->getOrCreateShaderModule("shaders/main_water_rt.tese.spv") : VK_NULL_HANDLE;
    // Per-op profiling variant (counters + device clock in both stages). Built
    // only when the device supports VK_KHR_shader_clock.
    VkShaderModule fragRtProfModule = (app && app->rayTracingEnabled() && app->rtProfilingSupported)
        ? app->getOrCreateShaderModule("shaders/main_water_rt_prof.frag.spv") : VK_NULL_HANDLE;
    VkShaderModule teseRtProfModule = (app && app->rayTracingEnabled() && app->rtProfilingSupported)
        ? app->getOrCreateShaderModule("shaders/main_water_rt_prof.tese.spv") : VK_NULL_HANDLE;
    tescModule = app->getOrCreateShaderModule("shaders/main_water.tesc.spv");
    teseModule = app->getOrCreateShaderModule("shaders/main_water.tese.spv");

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

    // Non-tessellated family vertex input: the WATER_NO_TESS vertex shader
    // consumes only POS/NORMAL/BRUSH_INDEX/HSV (COLOR/UV are unused and are
    // pruned by the -O SPIR-V pass), so the pipeline must not declare the
    // pruned attributes or VVL reports
    // "Vertex attribute at location N not consumed by vertex shader".
    auto noTessAttrDescs = vk_layouts::defaultAttributesFiltered(
        { ATTR_POS, ATTR_NORMAL, ATTR_BRUSH_INDEX, ATTR_HSV });
    VkPipelineVertexInputStateCreateInfo noTessVertexInputInfo = vertexInputInfo;
    noTessVertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(noTessAttrDescs.size());
    noTessVertexInputInfo.pVertexAttributeDescriptions = noTessAttrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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

    // Water render pass has 3 color attachments — need a blend state for each:
    //  0 = water color (RGBA32F), 1 = water body (RGBA16F, body+weight),
    //  2 = water column (RG16F: measured depth in R, blur radius px in G).
    std::array<VkPipelineColorBlendAttachmentState, 3> colorBlendAttachments{};
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

    // Patch control points for the tessellated family; the non-tessellated
    // family leaves pTessellationState null (no tessellation stages).
    VkPipelineTessellationStateCreateInfo tessState{};
    tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tessState.patchControlPoints = 3;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    // Dynamic rendering (no render pass)
    // Attachment 0 = water color (RGBA32F), attachment 1 = water body
    // (RGBA16F: refraction+tint body + body weight), attachment 2 = water
    // column (RG16F: measured depth in R, blur radius in G). The composite blurs the
    // body with a depth-scaled kernel and re-inserts it with its weight.
    std::array<VkFormat, 3> waterColorFmts = {
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16_SFLOAT
    };
    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = static_cast<uint32_t>(waterColorFmts.size());
    pipelineRenderingInfo.pColorAttachmentFormats = waterColorFmts.data();
    pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    pipelineInfo.pNext = &pipelineRenderingInfo;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    // pStages / pInputAssemblyState / pTessellationState are family-specific
    // and filled in by createFamily() below; the shared state pointers remain
    // valid for both families.
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = waterGeometryPipelineLayout;
    pipelineInfo.subpass = 0;

    // H4 single-attachment state: the WATER_NO_BODY fragment modules declare
    // only outColor, so the no-body geometry pipelines must advertise exactly
    // one color attachment (and a matching blend-state count — dynamic
    // rendering requires pColorBlendState->attachmentCount to equal
    // colorAttachmentCount). Everything else is shared with the 3-attachment
    // variants.
    VkFormat noBodyColorFmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkPipelineRenderingCreateInfo noBodyRenderingInfo = pipelineRenderingInfo;
    noBodyRenderingInfo.colorAttachmentCount = 1;
    noBodyRenderingInfo.pColorAttachmentFormats = &noBodyColorFmt;
    std::array<VkPipelineColorBlendAttachmentState, 1> noBodyBlendAttachments = {
        colorBlendAttachments[0]
    };
    VkPipelineColorBlendStateCreateInfo noBodyColorBlending = colorBlending;
    noBodyColorBlending.attachmentCount = 1;
    noBodyColorBlending.pAttachments = noBodyBlendAttachments.data();

    // Phase-1 water-in-main blend variant: same stages/layout, but drawn into
    // the MAIN solid color/depth targets with alpha blending. Depth writes stay
    // off so water neither disturbs the solid depth (sampled downstream) nor
    // self-occludes in draw order; depth test still rejects water behind
    // terrain. Color format matches the solid pass attachment.
    VkPipelineRasterizationStateCreateInfo mainRasterizer = rasterizer;
    mainRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    mainRasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineDepthStencilStateCreateInfo mainDepthStencil = depthStencil;
    mainDepthStencil.depthWriteEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState, 1> mainBlendAttachments{};
    for (auto& att : mainBlendAttachments) {
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        att.blendEnable = VK_TRUE;
        att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.colorBlendOp = VK_BLEND_OP_ADD;
        // Premultiplied-style alpha accumulation: the destination alpha
        // becomes coverage, so later passes (brush overlay) can depth-test
        // against the blended result if needed.
        att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo mainColorBlending{};
    mainColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    mainColorBlending.logicOpEnable = VK_FALSE;
    mainColorBlending.attachmentCount = static_cast<uint32_t>(mainBlendAttachments.size());
    mainColorBlending.pAttachments = mainBlendAttachments.data();

    VkFormat mainColorFmt = app->getSwapchainImageFormat();
    VkPipelineRenderingCreateInfo mainRenderingInfo = pipelineRenderingInfo;
    // The water-in-main variant targets the single main solid color attachment
    // (the aux packing only exists on the offscreen water pass).
    mainRenderingInfo.colorAttachmentCount = 1;
    mainRenderingInfo.pColorAttachmentFormats = &mainColorFmt;
    mainRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo mainPipelineInfo = pipelineInfo;
    mainPipelineInfo.pNext = &mainRenderingInfo;
    mainPipelineInfo.pRasterizationState = &mainRasterizer;
    mainPipelineInfo.pDepthStencilState = &mainDepthStencil;
    mainPipelineInfo.pColorBlendState = &mainColorBlending;

    // Create the pipeline families. A family differs only in the input
    // assembly topology, the presence of the TCS/TES stages and the
    // tessellation state; the fragment module is swapped per variant (non-RT /
    // RT / RT prof). The RT variants additionally swap the TES to the
    // ray-query water-depth TES (tessellated family only — the non-tess VS
    // emits the fallback depth by design). All families share the pipeline
    // layout and render state, so descriptor sets and bind points are common.
    auto createFamily = [&](bool tess, VkShaderModule vert,
                            TrackedHandle<VkPipeline>& geomNoRt,
                            TrackedHandle<VkPipeline>& mainNoRt,
                            TrackedHandle<VkPipeline>& geomRt,
                            TrackedHandle<VkPipeline>& mainRt,
                            TrackedHandle<VkPipeline>& geomRtProf,
                            TrackedHandle<VkPipeline>& mainRtProf,
                            TrackedHandle<VkPipeline>& geomNoBody,
                            TrackedHandle<VkPipeline>& geomRtNoBody,
                            const char* familyLabel) {
        std::vector<VkPipelineShaderStageCreateInfo> stages;

        VkPipelineShaderStageCreateInfo vs{};
        vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vs.module = vert;
        vs.pName = "main";
        stages.push_back(vs);

        int teseStageIndex = -1; // shaderStages index of the TES stage (per-variant swap)
        if (tess) {
            VkPipelineShaderStageCreateInfo tescStage{};
            tescStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            tescStage.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
            tescStage.module = tescModule;
            tescStage.pName = "main";
            stages.push_back(tescStage);

            teseStageIndex = static_cast<int>(stages.size());
            VkPipelineShaderStageCreateInfo teseStage{};
            teseStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            teseStage.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
            teseStage.module = teseModule;
            teseStage.pName = "main";
            stages.push_back(teseStage);
        }

        // Fragment stage last; the per-variant module is set below.
        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = VK_NULL_HANDLE;
        fragStage.pName = "main";
        stages.push_back(fragStage);

        VkPipelineInputAssemblyStateCreateInfo familyInputAssembly = inputAssembly;
        familyInputAssembly.topology = tess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
                                            : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkGraphicsPipelineCreateInfo familyPipelineInfo = pipelineInfo;
        familyPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        familyPipelineInfo.pStages = stages.data();
        familyPipelineInfo.pInputAssemblyState = &familyInputAssembly;
        // The non-tess family's vertex shader consumes only a subset of the
        // default attributes (see noTessVertexInputInfo above).
        familyPipelineInfo.pVertexInputState = tess ? &vertexInputInfo : &noTessVertexInputInfo;
        // pTessellationState stays null for the non-tessellated family.
        if (tess) familyPipelineInfo.pTessellationState = &tessState;

        VkGraphicsPipelineCreateInfo familyMainPipelineInfo = mainPipelineInfo;
        familyMainPipelineInfo.stageCount = familyPipelineInfo.stageCount;
        familyMainPipelineInfo.pStages = stages.data();
        familyMainPipelineInfo.pInputAssemblyState = &familyInputAssembly;
        familyMainPipelineInfo.pVertexInputState = familyPipelineInfo.pVertexInputState;
        if (tess) familyMainPipelineInfo.pTessellationState = &tessState;

        // H4 no-body geometry-only variant: same family state, one color
        // attachment, WATER_NO_BODY fragment module. No water-in-main
        // counterpart is created (that pipeline already has one attachment).
        VkGraphicsPipelineCreateInfo familyNoBodyPipelineInfo = familyPipelineInfo;
        familyNoBodyPipelineInfo.pNext = &noBodyRenderingInfo;
        familyNoBodyPipelineInfo.pColorBlendState = &noBodyColorBlending;

        auto createVariant = [&](VkShaderModule frag, VkShaderModule tese,
                                 TrackedHandle<VkPipeline>& geomOut,
                                 TrackedHandle<VkPipeline>& mainOut,
                                 const char* variantLabel) {
            if (frag == VK_NULL_HANDLE) return;
            stages.back().module = frag;
            if (tese != VK_NULL_HANDLE && teseStageIndex >= 0)
                stages[teseStageIndex].module = tese;

            std::string geomName = std::string("WaterRenderer: waterGeometryPipeline (")
                                 + variantLabel + " " + familyLabel + ")";
            std::string mainName = std::string("WaterRenderer: waterMainPipeline (")
                                 + variantLabel + " " + familyLabel + ")";
            if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &familyPipelineInfo, nullptr, &geomOut) != VK_SUCCESS) {
                std::cerr << "[WaterRenderer] Warning: Failed to create water geometry pipeline ("
                          << variantLabel << " " << familyLabel << ")" << std::endl;
                geomOut = VK_NULL_HANDLE;
            } else {
                app->resources.addPipeline(geomOut, geomName.c_str());
                std::cout << "[WaterRenderer] Created water geometry pipeline "
                          << variantLabel << " " << familyLabel
                          << " (dynamic rendering, 1 color attachment)" << std::endl;
            }

            if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &familyMainPipelineInfo, nullptr, &mainOut) != VK_SUCCESS) {
                std::cerr << "[WaterRenderer] Warning: Failed to create water-in-main blend pipeline ("
                          << variantLabel << " " << familyLabel << ")" << std::endl;
                mainOut = VK_NULL_HANDLE;
            } else {
                app->resources.addPipeline(mainOut, mainName.c_str());
                std::cout << "[WaterRenderer] Created water-in-main blend pipeline "
                          << variantLabel << " " << familyLabel
                          << " (alpha, depth-write off)" << std::endl;
            }
        };

        // Geometry-only (no water-in-main) variant used with WATER_NO_BODY.
        // The TES is explicitly reset to the base module when no specialized
        // one is requested, so the RT swap performed by a previous variant does
        // not leak into the non-RT no-body pipeline.
        auto createNoBodyVariant = [&](VkShaderModule frag, VkShaderModule tese,
                                       TrackedHandle<VkPipeline>& geomOut,
                                       const char* variantLabel) {
            if (frag == VK_NULL_HANDLE) return;
            stages.back().module = frag;
            if (teseStageIndex >= 0)
                stages[teseStageIndex].module = (tese != VK_NULL_HANDLE) ? tese : teseModule;

            std::string geomName = std::string("WaterRenderer: waterGeometryPipeline (")
                                 + variantLabel + " " + familyLabel + ", no body)";
            if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &familyNoBodyPipelineInfo, nullptr, &geomOut) != VK_SUCCESS) {
                std::cerr << "[WaterRenderer] Warning: Failed to create water geometry pipeline ("
                          << variantLabel << " " << familyLabel << ", no body)" << std::endl;
                geomOut = VK_NULL_HANDLE;
            } else {
                app->resources.addPipeline(geomOut, geomName.c_str());
                std::cout << "[WaterRenderer] Created water geometry pipeline "
                          << variantLabel << " " << familyLabel
                          << " (dynamic rendering, 1 color attachment, no body/column)" << std::endl;
            }
        };

        createVariant(fragNoRtModule, VK_NULL_HANDLE, geomNoRt, mainNoRt, "non-RT");
        createVariant(fragRtModule, teseRtModule, geomRt, mainRt, "RT");
        // The profiling TES is only swapped in when a TES stage exists
        // (teseStageIndex >= 0); the non-tess family ignores it.
        createVariant(fragRtProfModule, teseRtProfModule, geomRtProf, mainRtProf, "RT prof");
        createNoBodyVariant(fragNoBodyModule, VK_NULL_HANDLE, geomNoBody, "non-RT");
        createNoBodyVariant(fragRtNoBodyModule, teseRtModule, geomRtNoBody, "RT");
    };

    // Tessellated family (today's pipeline, unchanged apart from the factory
    // refactor): PATCH_LIST + TCS/TES + tessellation state.
    createFamily(true, vertModule, waterGeometryPipeline, waterMainPipeline,
                 waterGeometryPipelineRt, waterMainPipelineRt,
                 waterGeometryPipelineRtProf, waterMainPipelineRtProf,
                 waterGeometryPipelineNoBody, waterGeometryPipelineRtNoBody, "tess");
    // Non-tessellated family (C1): TRIANGLE_LIST + WATER_NO_TESS VS, no TCS/TES
    // and no tessellation state.
    createFamily(false, vertNoTessModule, waterGeometryPipelineNoTess, waterMainPipelineNoTess,
                 waterGeometryPipelineRtNoTess, waterMainPipelineRtNoTess,
                 waterGeometryPipelineRtProfNoTess, waterMainPipelineRtProfNoTess,
                 waterGeometryPipelineNoTessNoBody, waterGeometryPipelineRtNoTessNoBody, "no-tess");

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertModule = VK_NULL_HANDLE;
    vertNoTessModule = VK_NULL_HANDLE;
    fragNoRtModule = VK_NULL_HANDLE;
    fragRtModule = VK_NULL_HANDLE;
    fragNoBodyModule = VK_NULL_HANDLE;
    fragRtNoBodyModule = VK_NULL_HANDLE;
    teseRtModule = VK_NULL_HANDLE;
    fragRtProfModule = VK_NULL_HANDLE;
    teseRtProfModule = VK_NULL_HANDLE;
    if (tescModule) tescModule = VK_NULL_HANDLE;
    if (teseModule) teseModule = VK_NULL_HANDLE;

    // Back-face pipeline creation moved to WaterBackFaceRenderer
}

void WaterRenderer::beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex, bool loadExisting) {
    if (frameIndex >= 3) return;

    // H4: the body/column aux attachments only exist in the blur-capable
    // pipeline variant. In no-body mode the pass requires (and touches) only
    // the water color + geometry depth attachments; the body/column images and
    // their tracked layouts stay untouched so a later blur-on frame resumes
    // from the last SHADER_READ_ONLY state. Cached before the early-outs so
    // endWaterGeometryPass always matches the selected mode.
    activePassBodyAttachments_ = geometryBodyAttachmentsActive();
    const bool useBody = activePassBodyAttachments_;

    if (getWaterGeometryPipeline() == VK_NULL_HANDLE) return;

    // The water color target is always required. The body/column images are
    // required only when the selected variant writes them.
    if (waterDepthImages[frameIndex] == VK_NULL_HANDLE) return;
    if (useBody) {
        if (waterBodyImages[frameIndex] == VK_NULL_HANDLE || waterBodyImageViews[frameIndex] == VK_NULL_HANDLE) return;
        if (waterColumnImages[frameIndex] == VK_NULL_HANDLE || waterColumnImageViews[frameIndex] == VK_NULL_HANDLE) return;
    }

    activeWaterFrameIndex = frameIndex;

    // Batched begin barriers (single vkCmdPipelineBarrier2 for
    // color+body+column+depth; was: one call per image). Transitions: water
    // color SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL (water pipeline writes
    // the composed output), water body SHADER_READ_ONLY →
    // COLOR_ATTACHMENT_OPTIMAL (refraction+tint body + weight) and water
    // column SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL (measured depth for
    // the composite blur), plus water geometry depth tracked →
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL (occlusion testing). Same stage/access
    // mapping as the single transitions; entries already in the target layout
    // (e.g. depth re-entered with LOAD ops for the brush-liquid overlay)
    // resolve to no-ops inside the same call. H4: body/column entries are
    // omitted in no-body mode.
    {
        std::vector<VulkanApp::BatchTransition> batch;
        batch.reserve(useBody ? 4 : 2);
        VulkanApp::BatchTransition colorBegin{};
        colorBegin.image     = waterDepthImages[frameIndex];
        colorBegin.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
        colorBegin.oldLayout = waterDepthImageLayouts[frameIndex];
        colorBegin.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorBegin.mipLevels = 1;
        batch.push_back(colorBegin);
        if (useBody && waterBodyImages[frameIndex] != VK_NULL_HANDLE) {
            VulkanApp::BatchTransition bodyBegin{};
            bodyBegin.image     = waterBodyImages[frameIndex];
            bodyBegin.format    = VK_FORMAT_R16G16B16A16_SFLOAT;
            bodyBegin.oldLayout = waterBodyImageLayouts[frameIndex];
            bodyBegin.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            bodyBegin.mipLevels = 1;
            batch.push_back(bodyBegin);
        }
        if (useBody && waterColumnImages[frameIndex] != VK_NULL_HANDLE) {
            VulkanApp::BatchTransition columnBegin{};
            columnBegin.image     = waterColumnImages[frameIndex];
            columnBegin.format    = VK_FORMAT_R16G16_SFLOAT;
            columnBegin.oldLayout = waterColumnImageLayouts[frameIndex];
            columnBegin.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            columnBegin.mipLevels = 1;
            batch.push_back(columnBegin);
        }
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

    // Attachment 1: water body (refraction+tint body RGB, weight A) for the
    // composite's depth-guided blur (blurred and re-inserted by weight, so the
    // reflection stays sharp). Preserved alongside the color target when the
    // brush-liquid overlay re-enters this pass with LOAD ops. Only present in
    // the blur-capable variant (H4).
    VkRenderingAttachmentInfo bodyAttachment{};
    bodyAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    bodyAttachment.imageView = useBody ? waterBodyImageViews[frameIndex] : VK_NULL_HANDLE;
    bodyAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bodyAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    bodyAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    bodyAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    // Attachment 2: water column (measured water depth in R, per-material blur
    // radius in pixels in G) driving the composite blur. Same no-body gating.
    VkRenderingAttachmentInfo columnAttachment{};
    columnAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    columnAttachment.imageView = useBody ? waterColumnImageViews[frameIndex] : VK_NULL_HANDLE;
    columnAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    columnAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    columnAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    columnAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    // H4: in no-body mode only attachment 0 is declared; the extra entries are
    // present but excluded by colorAttachmentCount.
    std::array<VkRenderingAttachmentInfo, 3> colorAttachments = {
        colorAttachment, bodyAttachment, columnAttachment
    };

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
    // H4: the no-body pipeline variant declares a single color attachment.
    renderingInfo.colorAttachmentCount = useBody ? static_cast<uint32_t>(colorAttachments.size()) : 1u;
    renderingInfo.pColorAttachments = colorAttachments.data();
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
    if (waterDepthImages[frameIndex] == VK_NULL_HANDLE || !appPtr) return;

    // Batched end barriers: water color + body + column
    // COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (sampled by the
    // forward swapchain/postprocess pass, which blurs the body using the
    // packed water column depth). H4: in no-body mode the aux attachments were
    // never transitioned/attached, so their barriers and layout updates are
    // skipped and only the color target is restored.
    std::vector<VulkanApp::BatchTransition> batch;
    batch.reserve(activePassBodyAttachments_ ? 3 : 1);
    VulkanApp::BatchTransition colorEnd{};
    colorEnd.image     = waterDepthImages[frameIndex];
    colorEnd.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
    colorEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorEnd.mipLevels = 1;
    batch.push_back(colorEnd);
    if (activePassBodyAttachments_ && waterBodyImages[frameIndex] != VK_NULL_HANDLE) {
        VulkanApp::BatchTransition bodyEnd{};
        bodyEnd.image     = waterBodyImages[frameIndex];
        bodyEnd.format    = VK_FORMAT_R16G16B16A16_SFLOAT;
        bodyEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        bodyEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bodyEnd.mipLevels = 1;
        batch.push_back(bodyEnd);
    }
    if (activePassBodyAttachments_ && waterColumnImages[frameIndex] != VK_NULL_HANDLE) {
        VulkanApp::BatchTransition columnEnd{};
        columnEnd.image     = waterColumnImages[frameIndex];
        columnEnd.format    = VK_FORMAT_R16G16_SFLOAT;
        columnEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        columnEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        columnEnd.mipLevels = 1;
        batch.push_back(columnEnd);
    }
    appPtr->recordTransitionBatch(cmd, batch);
    waterDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (activePassBodyAttachments_ && waterBodyImages[frameIndex] != VK_NULL_HANDLE)
        waterBodyImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (activePassBodyAttachments_ && waterColumnImages[frameIndex] != VK_NULL_HANDLE)
        waterColumnImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void WaterRenderer::endWaterGeometryPassWithDepth(VkCommandBuffer cmd, uint32_t frameIndex) {
    endWaterRendering(cmd);
    if (!appPtr || waterDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    // Batched end barriers (single vkCmdPipelineBarrier2 for
    // color+body+column+depth): water color COLOR_ATTACHMENT_OPTIMAL →
    // SHADER_READ_ONLY_OPTIMAL (sampled by the composite), water body +
    // column COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // (depth-guided blur inputs) together with the water geometry depth
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (sampled by
    // the composite at postprocess binding 7). Was: endWaterGeometryPass plus
    // a second lone depth transition (two calls). Same mapping as the single
    // transitions; the geom depth shares the pass boundary, so one call covers
    // all resources. H4: in no-body mode the aux attachments were never
    // transitioned/attached, so only color + depth are restored.
    std::vector<VulkanApp::BatchTransition> batch;
    batch.reserve(activePassBodyAttachments_ ? 4 : 2);
    VulkanApp::BatchTransition colorEnd{};
    colorEnd.image     = waterDepthImages[frameIndex];
    colorEnd.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
    colorEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorEnd.mipLevels = 1;
    batch.push_back(colorEnd);
    if (activePassBodyAttachments_ && waterBodyImages[frameIndex] != VK_NULL_HANDLE) {
        VulkanApp::BatchTransition bodyEnd{};
        bodyEnd.image     = waterBodyImages[frameIndex];
        bodyEnd.format    = VK_FORMAT_R16G16B16A16_SFLOAT;
        bodyEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        bodyEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bodyEnd.mipLevels = 1;
        batch.push_back(bodyEnd);
    }
    if (activePassBodyAttachments_ && waterColumnImages[frameIndex] != VK_NULL_HANDLE) {
        VulkanApp::BatchTransition columnEnd{};
        columnEnd.image     = waterColumnImages[frameIndex];
        columnEnd.format    = VK_FORMAT_R16G16_SFLOAT;
        columnEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        columnEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        columnEnd.mipLevels = 1;
        batch.push_back(columnEnd);
    }
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
    if (activePassBodyAttachments_ && waterBodyImages[frameIndex] != VK_NULL_HANDLE)
        waterBodyImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (activePassBodyAttachments_ && waterColumnImages[frameIndex] != VK_NULL_HANDLE)
        waterColumnImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (waterGeomDepthImages[frameIndex] != VK_NULL_HANDLE)
        waterGeomDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// Back-face pass is owned and executed by SceneRenderer via its WaterBackFaceRenderer.

void WaterRenderer::updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, uint32_t frameIndex,
                                               VkImageView backFaceDepthView,
                                               VkImageView rtReflectView, VkImageView rtRefractView,
                                               VkImageView skyView,
                                               VkImageView solidColorView, VkImageView solidDepthView,
                                               VkImageView vegColorView, VkImageView vegDepthView) {
    if (ds == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return;
    }
    (void)frameIndex;
    ensureDummyViews(app);
    if (nearestSampler == VK_NULL_HANDLE) return;

    // Every binding is statically used by water.frag — never leave NULL.
    // Missing RT outputs (unsupported/disabled) bind 1x1 GENERAL dummies
    // (valid flag 0 / thickness -1 → shader takes the sky/inline path);
    // missing sky binds a 1x1 SHADER_READ dummy. Missing solid targets bind
    // the same SHADER_READ dummy: depth reads as 1.0 → SSR never hits.
    VkImageView effReflect = (rtReflectView != VK_NULL_HANDLE) ? rtReflectView : dummyRTView_;
    VkImageView effRefract = (rtRefractView != VK_NULL_HANDLE) ? rtRefractView : dummyRTView_;
    VkImageView effSky = (skyView != VK_NULL_HANDLE) ? skyView : dummySkyView_;
    VkImageView effSolidColor = (solidColorView != VK_NULL_HANDLE) ? solidColorView : dummySkyView_;
    VkImageView effSolidDepth = (solidDepthView != VK_NULL_HANDLE) ? solidDepthView : dummySkyView_;
    if (backFaceDepthView == VK_NULL_HANDLE || effReflect == VK_NULL_HANDLE ||
        effRefract == VK_NULL_HANDLE || effSky == VK_NULL_HANDLE) {
        return;
    }

    std::array<VkDescriptorImageInfo, 8> imageInfos{};

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

    // Solid pass color/depth (bindings 4-5): the solid pass ends with both
    // images in SHADER_READ_ONLY_OPTIMAL (endPass barriers), so SSR samples
    // them directly. Linear sampler for the color (upscale-safe), nearest for
    // the depth (no interpolation across silhouettes).
    imageInfos[4].sampler = linearSampler;
    imageInfos[4].imageView = effSolidColor;
    imageInfos[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfos[5].sampler = nearestSampler;
    imageInfos[5].imageView = effSolidDepth;
    imageInfos[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Vegetation color/depth (bindings 6-7). Missing targets bind the 1x1 sky
    // dummy: its depth reads as 1.0, so the vegetation layer is simply absent.
    VkImageView effVegColor = (vegColorView != VK_NULL_HANDLE) ? vegColorView : dummySkyView_;
    VkImageView effVegDepth = (vegDepthView != VK_NULL_HANDLE) ? vegDepthView : dummySkyView_;
    imageInfos[6].sampler = linearSampler;
    imageInfos[6].imageView = effVegColor;
    imageInfos[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfos[7].sampler = nearestSampler;
    imageInfos[7].imageView = effVegDepth;
    imageInfos[7].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // M7: skip the write entirely when this set already holds exactly these
    // inputs (eight effective image views + samplers). Pure CPU state, no GPU
    // work; entries are dropped by invalidateSceneTexturesBinding() on any free
    // or reallocation, and by the write below for the back-face patch cache.
    const uint64_t dsKey = descriptorSetKey(ds);
    SceneTexturesBindingKey key{};
    for (uint32_t i = 0; i < 8; ++i) {
        key.imageViews[i] = imageInfos[i].imageView;
        key.samplers[i] = imageInfos[i].sampler;
    }
    {
        std::lock_guard<std::mutex> lock(sceneTexturesCacheMutex_);
        auto cached = sceneTexturesBindingCache_.find(dsKey);
        if (cached != sceneTexturesBindingCache_.end() && cached->second == key) {
            return;
        }
    }

    // Descriptor-buffer style update: the per-frame set is allocated once (see
    // prepareSceneTexturesForFrame) and updated in place here. With
    // VK_EXT_descriptor_buffer this same call becomes plain host memory writes
    // (vkGetDescriptorEXT); the classic vkUpdateDescriptorSets below is the
    // fallback until the layout carries
    // VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT.
    DescriptorWriter writer(app->getDevice());
    for (uint32_t i = 0; i < 8; ++i) {
        writer.writeImage(ds, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          imageInfos[i].sampler, imageInfos[i].imageView,
                          imageInfos[i].imageLayout);
    }
    writer.flush();
    {
        std::lock_guard<std::mutex> lock(sceneTexturesCacheMutex_);
        sceneTexturesBindingCache_[dsKey] = key;
    }

    // This write replaced binding 0 (back-face depth) of `ds`, so the
    // back-face renderer's dummy-patch cache for that set is stale: its next
    // patchBinding0() must not be skipped.
    if (backFaceRenderer_) backFaceRenderer_->invalidatePatchedBinding0(ds);
}

void WaterRenderer::invalidateSceneTexturesBinding(VkDescriptorSet ds) {
    if (ds == VK_NULL_HANDLE) return;
    {
        std::lock_guard<std::mutex> lock(sceneTexturesCacheMutex_);
        sceneTexturesBindingCache_.erase(descriptorSetKey(ds));
    }
    // The same handle may be reallocated from any pool (including one owned by
    // the caller), so drop the back-face binding-0 patch entry as well: a
    // recycled handle must never be treated as already patched.
    if (backFaceRenderer_) backFaceRenderer_->invalidatePatchedBinding0(ds);
}

VkDescriptorSet WaterRenderer::prepareSceneTexturesForFrame(VulkanApp* app, uint32_t frameIndex,
                                                            VkImageView backFaceDepthView,
                                                            VkImageView rtReflectView,
                                                            VkImageView rtRefractView,
                                                            VkImageView skyView,
                                                            VkImageView solidColorView,
                                                            VkImageView solidDepthView,
                                                            VkImageView vegColorView,
                                                            VkImageView vegDepthView) {
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
        // M7: a freshly allocated handle may recycle the value of a previously
        // freed set; drop any stale cache entry before writing the new bindings.
        invalidateSceneTexturesBinding(waterDepthDescriptorSets[frameIndex]);
    }

    updateSceneTexturesBinding(app, waterDepthDescriptorSets[frameIndex], frameIndex,
                               backFaceDepthView, rtReflectView, rtRefractView, skyView,
                               solidColorView, solidDepthView, vegColorView, vegDepthView);
    return waterDepthDescriptorSets[frameIndex];
}

void WaterRenderer::initializeWaterParamsBuffer(const std::vector<WaterParams>& waterParams) {
    // H4: track the per-layer blur gate from the full layer vector. The SSBO may
    // hold more entries than the CPU vector (waterParamsCount); entries beyond
    // the uploaded vector are never written and can never request blur, so they
    // are tracked as false.
    const size_t trackedLayers = std::max(static_cast<size_t>(waterParamsCount),
                                          waterParams.size());
    layerBlurNeeded_.assign(trackedLayers, false);
    for (uint32_t i = 0; i < waterParams.size(); ++i) {
        layerBlurNeeded_[i] = waterParams[i].enableBlur && waterParams[i].blurRadius > 0.0f;
    }
    refreshWaterBlurNeeded();

    if (waterParamsBuffer.buffer == VK_NULL_HANDLE) return;

    for (uint32_t i = 0; i < waterParams.size(); ++i) {
        WaterParamsGPU gpu = makeWaterParamsGPU(waterParams[i]);
        waterGpuPeriodsToScales(gpu); // periods -> shader scales, at the upload boundary
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
    // tessellation evaluation shader samples the back-face depth (set 2
    // binding 0) and the solid depth (binding 5) for the per-vertex shore
    // zones; the WATER_NO_TESS vertex path samples the same two bindings
    // (C1), so VERTEX must be in the destination stage mask alongside TES and
    // the fragment shader.
    VkMemoryBarrier2 memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    // The back-face pass writes its depth image at EARLY_FRAGMENT_TESTS stage
    // (depth-only pre-pass), so the source mask must include EARLY_FRAGMENT_TESTS
    // (not just LATE) to establish the dependency for that write.
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
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
    if (getWaterGeometryPipeline() != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsPipeline(cmd, getWaterGeometryPipeline());
        else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, getWaterGeometryPipeline());
        waterIndirectRenderer.drawPrepared(cmd);
        // Brush liquid water: same pipeline, same descriptor sets, its own IR.
        // Drawn right after the main water so brush liquid depth/color lands in
        // the same offscreen target (self-occlusion tested by water depth).
        if (secondaryIR) secondaryIR->drawPrepared(cmd);
    }

    endWaterGeometryPass(cmd);
}

void WaterRenderer::renderMainTargets(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                      VkImage colorImage, VkImageView colorView,
                                      VkImage depthImage, VkImageView depthView,
                                      VkImageView skyView, VkDescriptorSet overrideWaterDs) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (getWaterMainPipeline() == VK_NULL_HANDLE) return;
    if (frameIndex >= FRAMES) return;
    if (colorImage == VK_NULL_HANDLE || colorView == VK_NULL_HANDLE ||
        depthImage == VK_NULL_HANDLE || depthView == VK_NULL_HANDLE) return;

    // Water-in-main path (M7): it never calls renderPass(), so it owns the
    // flush of the feature gates stored by setRtFeatureFlags(). Only when a
    // gate changed; timeParams.x is preserved (no per-frame time source here).
    if (waterRenderUboDirty_) {
        flushWaterRenderUBO(0.0f, /*preserveTime=*/true);
    }

    // The caller (MyApp, water command buffer) has already waited on the solid
    // pass semaphore, so the main targets are complete and in SHADER_READ_ONLY.
    // Transition both to attachment layouts (batched single barrier).
    {
        std::vector<VulkanApp::BatchTransition> batch;
        batch.reserve(2);
        VulkanApp::BatchTransition colorBegin{};
        colorBegin.image     = colorImage;
        colorBegin.format    = app->getSwapchainImageFormat();
        colorBegin.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        colorBegin.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorBegin.mipLevels = 1;
        batch.push_back(colorBegin);
        VulkanApp::BatchTransition depthBegin{};
        depthBegin.image     = depthImage;
        depthBegin.format    = VK_FORMAT_D32_SFLOAT;
        depthBegin.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthBegin.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthBegin.mipLevels = 1;
        batch.push_back(depthBegin);
        app->recordTransitionBatch(cmd, batch);
    }

    // LOAD ops: the opaque scene stays; water alpha-blends on top.
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = colorView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

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

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, getWaterMainPipeline());
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, getWaterMainPipeline());

    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    if (mainDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, waterGeometryPipelineLayout, 0, 1, &mainDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            waterGeometryPipelineLayout, 0, 1, &mainDs, 0, nullptr);
    }
    VkDescriptorSet sceneDs = (overrideWaterDs != VK_NULL_HANDLE) ? overrideWaterDs : getWaterDepthDescriptorSet(frameIndex);
    if (sceneDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, waterGeometryPipelineLayout, 2, 1, &sceneDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            waterGeometryPipelineLayout, 2, 1, &sceneDs, 0, nullptr);
    }

    waterIndirectRenderer.drawPrepared(cmd);

    vkCmdEndRendering(cmd);

    // Back to SHADER_READ_ONLY: the composite samples the color target and the
    // depth target is read by later passes (brush overlay, debug widgets).
    {
        std::vector<VulkanApp::BatchTransition> batch;
        batch.reserve(2);
        VulkanApp::BatchTransition colorEnd{};
        colorEnd.image     = colorImage;
        colorEnd.format    = app->getSwapchainImageFormat();
        colorEnd.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        colorEnd.mipLevels = 1;
        batch.push_back(colorEnd);
        VulkanApp::BatchTransition depthEnd{};
        depthEnd.image     = depthImage;
        depthEnd.format    = VK_FORMAT_D32_SFLOAT;
        depthEnd.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthEnd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthEnd.mipLevels = 1;
        batch.push_back(depthEnd);
        app->recordTransitionBatch(cmd, batch);
    }

    (void)skyView;
}

void WaterRenderer::renderBrushLiquid(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView skyView, VkDescriptorSet overrideWaterDs) {    if (!app || cmd == VK_NULL_HANDLE || !brushRenderer_) return;
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

// Single writer of waterRenderUBO_ (M7). Builds the full WaterRenderUBO and
// memcpys it in one map/unmap, then clears the dirty bit. `preserveTime` keeps
// the existing timeParams.x for the water-in-main flush, which has no
// per-frame time argument and must not zero the wave clock.
void WaterRenderer::flushWaterRenderUBO(float waterTime, bool preserveTime) {
    if (waterRenderUBO_.buffer == VK_NULL_HANDLE) return;
    void* data = waterRenderUBO_.map(0);
    if (!data) return;
    const float time = preserveTime
        ? static_cast<WaterRenderUBO*>(data)->timeParams.x
        : waterTime;
    WaterRenderUBO renderUbo{};
    renderUbo.timeParams = glm::vec4(time,
                                     rtRefractionsEnabled_ ? 1.0f : 0.0f,
                                     rtReflectionsEnabled_ ? 1.0f : 0.0f,
                                     blurEnabled_ ? 1.0f : 0.0f);
    memcpy(data, &renderUbo, sizeof(WaterRenderUBO));
    waterRenderUBO_.unmap(); // VMA persistent mapping
    waterRenderUboDirty_ = false;
}

void WaterRenderer::setRtFeatureFlags(bool reflections, bool refractions, bool blur) {
    // Pure state store (M7): no UBO map here. The offscreen path folds the
    // stored flags into renderPass()'s single per-frame UBO write; the
    // water-in-main path (which never calls renderPass) flushes them lazily
    // from renderMainTargets() through the dirty bit. Setting the bit only on
    // an actual change keeps the steady state map-free for water-in-main;
    // the bit starts true so the first flush of an initially zeroed UBO
    // always happens.
    if (rtReflectionsEnabled_ == reflections &&
        rtRefractionsEnabled_ == refractions &&
        blurEnabled_ == blur) {
        return;
    }
    rtReflectionsEnabled_ = reflections;
    rtRefractionsEnabled_ = refractions;
    blurEnabled_ = blur;
    waterRenderUboDirty_ = true;
}

void WaterRenderer::renderPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t frameIdx,
                               bool waterWireframeEnabled, float waterTime, VkImageView skyView,
                               VkDescriptorSet overrideWaterDs, bool drawBrushLiquid) {
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[WaterRenderer::renderPass] commandBuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    // The only UBO write on the offscreen path (M7): the stored feature gates
    // are folded into this single write and the dirty bit is cleared inside.
    flushWaterRenderUBO(waterTime, /*preserveTime=*/false);

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
        // Non-async path: vegetation targets are not reachable from here
        // (they belong to the app's frame graph); bind null → dummy (no veg).
        prepareSceneTexturesForFrame(app, frameIdx, wBack, wRefl, wRefr, skyView,
                                     solidRenderer_ ? solidRenderer_->getColorView(frameIdx) : VK_NULL_HANDLE,
                                     solidRenderer_ ? solidRenderer_->getDepthView(frameIdx) : VK_NULL_HANDLE,
                                     VK_NULL_HANDLE, VK_NULL_HANDLE);
    }

    // Scene textures were already bound before the async back-face task was
    // launched (see MyApp.cpp), so we must NOT call updateSceneTexturesBinding here.
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
