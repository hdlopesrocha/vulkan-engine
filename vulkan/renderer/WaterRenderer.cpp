
#include "WaterRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include <vector>
#include "RendererUtils.hpp"
#include "BrushRenderer.hpp"
#include "WaterBackFaceRenderer.hpp"
#include "Solid360Renderer.hpp"
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
                                      WaterBackFaceRenderer* backFace, Solid360Renderer* solid360,
                                      WireframeRenderer* waterWireframe) {
    solidRenderer_ = solid;
    brushRenderer_ = brush;
    backFaceRenderer_ = backFace;
    solid360Renderer_ = solid360;
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
    // Dummy cube image/view destruction deferred to VulkanResourceManager.
    cubemapDummyCubeImage = VK_NULL_HANDLE;
    cubemapDummyCubeAllocation = VK_NULL_HANDLE;
    cubemapDummyCubeMemory = VK_NULL_HANDLE;
    cubemapDummyCubeView = VK_NULL_HANDLE;
    cubemapDummyCubeFormat = VK_FORMAT_UNDEFINED;
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
    // (e.g. scene reload). Recreating the pipeline layout every time thrashes its
    // handle, leaving the cubemap water pipeline (built against an earlier layout)
    // bound against an incompatible layout at draw time. Create the layout + main
    // pipeline exactly once and keep them stable. Resizes recreate via a dedicated path.
    if (waterGeometryPipelineLayout != VK_NULL_HANDLE) {
        initializeWaterParamsBuffer(waterParams);
        return;
    }

    // Water params buffer is already assigned in init
    initializeWaterParamsBuffer(waterParams);
    std::cout << "[WaterRenderer] Initialized water params buffer from provided layer state" << std::endl;
    
    // Create descriptor set layout for scene textures (set 2).
    // The water pass is fully decoupled from the solid pass: it samples only its
    // own back-face depth (for volume thickness) and the solid 360 cubemap (for
    // reflection/refraction). It no longer reads the solid color or depth targets,
    // so it has no dependency on the solid pass and can be recorded/rendered on its
    // own command buffer in parallel with the solid pass. Occlusion against solids
    // is resolved at the composite stage (postprocess.frag).
    // Binding 0: Water back-face depth texture (for volume thickness)
    // Binding 1: Optional cubemap (solid 360)
    std::array<VkDescriptorSetLayoutBinding, 2> sceneBindings{};

    // Water back-face depth (binding 0) — for water volume thickness.
    // Also sampled by the tessellation evaluation shader (VUID 07988).
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    sceneBindings[0].pImmutableSamplers = nullptr;

    // Optional cubemap sampler (binding 1) — used by water shader when solid 360 is available
    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[1].pImmutableSamplers = nullptr;

    VkDescriptorBindingFlags bindingFlags[2] = {
        0, 0
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
    // Capacity covers the 3 cubemap sets plus the 3 main sets with headroom.
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

    // Create water geometry pipeline with dedicated water shaders
    // Load water shaders (vertex, tessellation control, tessellation evaluation, fragment)
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/water.vert.spv");
    VkShaderModule fragModule = app->getOrCreateShaderModule("shaders/water.frag.spv");
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

void WaterRenderer::updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, uint32_t frameIndex, VkImageView backFaceDepthView, VkImageView cube360View) {
    if (ds == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return;
    }
    (void)frameIndex;

    // The real cubemap + back-face depth are created in SceneRenderer::init
    // (and recreated on swapchain resize) before the first frame, so they are
    // always ready before first use. No dummy fallback: if either view is
    // missing, leave the set untouched so no dummy/NULL view is ever bound
    // at draw time.

    // Determine the effective cubemap to bind:
    // - If a new explicit `cube360View` is provided, use it and remember it.
    // - If `cube360View` is VK_NULL_HANDLE, prefer the previously remembered view
    //   (so callers that update only color won't accidentally clear the cube).
    VkImageView finalCubeView = cube360View;
    if (finalCubeView == VK_NULL_HANDLE) finalCubeView = currentCube360View;

    // If an explicit cube view was provided, remember it for future updates.
    if (cube360View != VK_NULL_HANDLE) currentCube360View = cube360View;

    // Track cubemap availability and update per-layer GPU params when it changes
    bool cubeAvail = (finalCubeView != VK_NULL_HANDLE);
    if (cube360Available != cubeAvail) {
        cube360Available = cubeAvail;
        // The upper level must explicitly refresh GPU params for any layer
        // whose state should change when cubemap availability toggles.
    }

    // Both views must be valid — otherwise skip the update entirely (never
    // bind VK_NULL_HANDLE or a dummy placeholder).
    if (backFaceDepthView == VK_NULL_HANDLE || finalCubeView == VK_NULL_HANDLE) {
        return;
    }
    if (nearestSampler == VK_NULL_HANDLE) return;

    std::array<VkDescriptorImageInfo, 2> imageInfos{};

    // Water back-face depth (binding 0) — real view only.
    // Use nearest filtering so depth values are not interpolated across geometry edges.
    imageInfos[0].sampler = nearestSampler;
    imageInfos[0].imageView = backFaceDepthView;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Cubemap (binding 1) — real solid360 cube view only.
    imageInfos[1].sampler = linearSampler;
    imageInfos[1].imageView = finalCubeView;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Descriptor-buffer style update: rewrite the bindings unconditionally.
    // No write cache, no set allocation/free, no deferred destruction — the
    // per-frame set is allocated once (see prepareSceneTexturesForFrame) and
    // updated in place here. With VK_EXT_descriptor_buffer this same call
    // becomes plain host memory writes (vkGetDescriptorEXT); the classic
    // vkUpdateDescriptorSets below is the fallback until the layout carries
    // VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT.
    DescriptorWriter writer(app->getDevice());
    for (uint32_t i = 0; i < 2; ++i) {
        writer.writeImage(ds, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          imageInfos[i].sampler, imageInfos[i].imageView,
                          imageInfos[i].imageLayout);
    }
    writer.flush();
}

VkDescriptorSet WaterRenderer::prepareSceneTexturesForFrame(VulkanApp* app, uint32_t frameIndex,
                                                              VkImageView backFaceDepthView,
                                                              VkImageView cube360View) {
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
                                backFaceDepthView, cube360View);
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

// 1x1 cube-typed dummy for cubemapWaterDS binding 1 (see member comment).
// Created once per swapchain color format, cleared to black, left in
// SHADER_READ_ONLY_OPTIMAL. Actual destruction is deferred to
// VulkanResourceManager; a format change orphans the old handles.
VkImageView WaterRenderer::ensureDummyCubeView(VulkanApp* app, VkFormat format) {
    if (!app || format == VK_FORMAT_UNDEFINED) return VK_NULL_HANDLE;
    if (cubemapDummyCubeView != VK_NULL_HANDLE && cubemapDummyCubeFormat == format)
        return cubemapDummyCubeView;
    cubemapDummyCubeImage = VK_NULL_HANDLE;
    cubemapDummyCubeAllocation = VK_NULL_HANDLE;
    cubemapDummyCubeMemory = VK_NULL_HANDLE;
    cubemapDummyCubeView = VK_NULL_HANDLE;

    VkImageCreateInfo img{};
    img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    img.imageType = VK_IMAGE_TYPE_2D;
    img.format = format;
    img.extent = {1, 1, 1};
    img.mipLevels = 1;
    img.arrayLayers = 6;
    img.samples = VK_SAMPLE_COUNT_1_BIT;
    img.tiling = VK_IMAGE_TILING_OPTIMAL;
    img.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    app->createImageWithVma(img, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        cubemapDummyCubeImage, cubemapDummyCubeAllocation, cubemapDummyCubeMemory,
        "WaterRenderer: cubemapDummyCube");

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = cubemapDummyCubeImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 6;
    if (vkCreateImageView(app->getDevice(), &vi, nullptr, &cubemapDummyCubeView) != VK_SUCCESS) {
        app->destroyImageWithVma(cubemapDummyCubeImage, cubemapDummyCubeAllocation, cubemapDummyCubeMemory);
        cubemapDummyCubeImage = VK_NULL_HANDLE;
        cubemapDummyCubeView = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    app->resources.addImageView(cubemapDummyCubeView, "WaterRenderer: cubemapDummyCubeView");

    // Clear to black and park in SHADER_READ_ONLY_OPTIMAL (all 6 layers) in a
    // single synchronous submit, mirroring the old 2D-dummy helper.
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        app->recordTransitionImageLayoutLayer(cmd, cubemapDummyCubeImage, format,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 6);
        VkClearColorValue cv{};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 };
        vkCmdClearColorImage(cmd, cubemapDummyCubeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
        app->recordTransitionImageLayoutLayer(cmd, cubemapDummyCubeImage, format,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 6);
    });
    cubemapDummyCubeFormat = format;
    return cubemapDummyCubeView;
}

void WaterRenderer::ensureCubemapResources(VulkanApp* app, VkFormat colorFormat) {    // --- Cubemap-compatible water pipeline (solid 360 cube faces) ---
    // The main water geometry pipeline targets R32G32B32A32_SFLOAT, which cannot
    // render into the swapchain-format cube faces, so a second pipeline is needed
    // for the cubemap pass. Recreate it if the swapchain color format changed.
    if (cubemapWaterPipeline != VK_NULL_HANDLE && cubemapWaterPipelineFormat != colorFormat) {
        app->resources.removePipeline(cubemapWaterPipeline);
        vkDestroyPipeline(app->getDevice(), cubemapWaterPipeline, nullptr);
        cubemapWaterPipeline = VK_NULL_HANDLE;
        app->resources.removePipelineLayout(cubemapWaterPipelineLayout);
        vkDestroyPipelineLayout(app->getDevice(), cubemapWaterPipelineLayout, nullptr);
        cubemapWaterPipelineLayout = VK_NULL_HANDLE;
    }
    if (cubemapWaterPipeline == VK_NULL_HANDLE && waterDepthDescriptorSetLayout != VK_NULL_HANDLE &&
        app->getDescriptorSetLayout() != VK_NULL_HANDLE && app->getMaterialDescriptorSetLayout() != VK_NULL_HANDLE) {
        ShaderStage vertShader(app->getOrCreateShaderModule("shaders/water.vert.spv"), VK_SHADER_STAGE_VERTEX_BIT);
        ShaderStage tescShader(app->getOrCreateShaderModule("shaders/water.tesc.spv"), VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
        ShaderStage teseShader(app->getOrCreateShaderModule("shaders/water.tese.spv"), VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
        ShaderStage fragShader(app->getOrCreateShaderModule("shaders/water.frag.spv"), VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayout> setLayouts = {
            app->getDescriptorSetLayout(),         // set 0: UBO + samplers (per-face cube360 DS)
            app->getMaterialDescriptorSetLayout(), // set 1: materials (unused by water shaders)
            waterDepthDescriptorSetLayout          // set 2: back-face depth + real cubemap
        };

        GraphicsPipelineConfig cfg{};
        cfg.colorFormats = { colorFormat };
        cfg.depthFormat = VK_FORMAT_D32_SFLOAT;
        // Water is drawn after the solid+sky color pass, tested against the
        // prepassed solid depth; no depth write (nothing consumes cube depth
        // after the water pass and water-over-water order is irrelevant for a
        // reflection).
        cfg.depthTestEnable = true;
        cfg.depthWriteEnable = false;
        cfg.depthCompareOp = VK_COMPARE_OP_LESS;
        cfg.blendEnable = false; // opaque overwrite where water passes the depth test

        auto [pipeline, layout] = app->createGraphicsPipeline(
            { vertShader.info, tescShader.info, teseShader.info, fragShader.info },
            std::vector<VkVertexInputBindingDescription>{
                VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX }
            },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            cfg);
        cubemapWaterPipeline = pipeline;
        cubemapWaterPipelineLayout = layout;
        cubemapWaterPipelineFormat = colorFormat;
        vertShader.info.module = VK_NULL_HANDLE;
        tescShader.info.module = VK_NULL_HANDLE;
        teseShader.info.module = VK_NULL_HANDLE;
        fragShader.info.module = VK_NULL_HANDLE;
        std::cout << "[WaterRenderer] Created cubemap water pipeline (solid 360)" << std::endl;
    }

    // --- Set-2 descriptor set for the cubemap water pass ---
    // Bound to validation-legal resources (never sampled in capture mode):
    // - binding 0: WaterBackFaceRenderer's 1x1 far-plane dummy depth. This is
    //   the shared thin-water depth (not a cubemap dummy): the cubemap faces
    //   have no per-face back-face pass, so thickness must read far-plane.
    // - binding 1: a 1x1 cube-typed dummy (SHADER_READ_ONLY_OPTIMAL). The face
    //   UBO carries materialFlags.x == 1 (capture mode), so water.frag skips
    //   reflection/refraction and never samples the cube it is rendering
    //   into — no feedback, no hazard. The REAL in-flight cube cannot be bound
    //   here: its rendered face is a color attachment while the other layers
    //   are read-only, so no single descriptor layout matches (VUID-00344).
    // A dedicated pool keeps the set alive across the waterDepthDescriptorPool
    // reset performed by destroyRenderTargets(). When the dummy is recreated
    // (swapchain color-format change), the set is rewritten in place: ensure is
    // only called before the cubemap pass records (never while pending).
    {
        VkImageView depthView = (backFaceRenderer_ != nullptr)
            ? backFaceRenderer_->getDummyDepthView() : VK_NULL_HANDLE;
        // Readiness gate: the real cube targets must exist (the pass renders
        // into its faces) before the set is created — but the real view is
        // never bound (see above).
        VkImageView realCubeView = (solid360Renderer_ != nullptr)
            ? solid360Renderer_->getSolid360View() : VK_NULL_HANDLE;
        if (waterDepthDescriptorSetLayout == VK_NULL_HANDLE ||
            depthView == VK_NULL_HANDLE || realCubeView == VK_NULL_HANDLE ||
            linearSampler == VK_NULL_HANDLE) {
            return;
        }
        VkImageView dummyCubeView = ensureDummyCubeView(app, colorFormat);
        if (dummyCubeView == VK_NULL_HANDLE) return;
        VkDevice device = app->getDevice();
        if (cubemapWaterDescPool == VK_NULL_HANDLE) {
            DescriptorAllocator descAlloc{device, app};
            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 };
            cubemapWaterDescPool = descAlloc.createPool(&ps, 1, 1, 0,
                "WaterRenderer: cubemapWaterDescPool");
        }
        VkSampler depthSampler = (nearestSampler != VK_NULL_HANDLE) ? nearestSampler : linearSampler;
        if (cubemapWaterDS == VK_NULL_HANDLE) {
            DescriptorAllocator descAlloc{device, app};
            cubemapWaterDS = descAlloc.allocateSet(cubemapWaterDescPool, waterDepthDescriptorSetLayout,
                "WaterRenderer: cubemapWaterDS");
            if (cubemapWaterDS == VK_NULL_HANDLE) return;
            DescriptorWriter(device)
                .writeImage(cubemapWaterDS, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            depthSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(cubemapWaterDS, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            linearSampler, dummyCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .flush();
            cubemapWaterBoundDepthView = depthView;
            cubemapWaterBoundCubeView = dummyCubeView;
        } else if (cubemapWaterBoundDepthView != depthView || cubemapWaterBoundCubeView != dummyCubeView) {
            DescriptorWriter(device)
                .writeImage(cubemapWaterDS, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            depthSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(cubemapWaterDS, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            linearSampler, dummyCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .flush();
            cubemapWaterBoundDepthView = depthView;
            cubemapWaterBoundCubeView = dummyCubeView;
        }
    }
}

// Solid 360° cubemap reflection is owned and executed by SceneRenderer.

void WaterRenderer::renderWaterIntoCubemap(VkCommandBuffer cmd, VkDescriptorSet sceneDs0,
                                            VkImageView colorView, VkImageView depthView,
                                            uint32_t faceSize,
                                            VkBuffer waterCompactBuffer, VkBuffer waterVisibleCountBuffer) {
    if (!appPtr || cmd == VK_NULL_HANDLE) return;
    if (cubemapWaterPipeline == VK_NULL_HANDLE || cubemapWaterPipelineLayout == VK_NULL_HANDLE) return;
    if (sceneDs0 == VK_NULL_HANDLE || cubemapWaterDS == VK_NULL_HANDLE) return;

    // Own rendering instance: LOAD the face color + solid depth written by the
    // preceding solid color pass so water composites over it (depth-tested).
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = colorView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = depthView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.offset = {0, 0};
    ri.renderArea.extent = {faceSize, faceSize};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0.0f, 0.0f, (float)faceSize, (float)faceSize, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, {faceSize, faceSize}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Raw binds (no CommandBufferState): this function is called from the solid
    // 360 per-face command buffers, each of which uses its own tracker owned by
    // Solid360Renderer; this renderer's cmdState may point at a different task's
    // tracker, which could wrongly elide the bind in a fresh command buffer.
    // Each face CB binds this pipeline exactly once, so dedup has no value here.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapWaterPipeline);

    // Set 0: per-face main-layout DS (binding 0 = this face's cube UBO slot,
    // with materialFlags.x == 1 so water.frag skips env-map feedback). Water
    // shaders statically use only sets 0 and 2, so set 1 (materials) is never
    // bound here.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapWaterPipelineLayout, 0, 1, &sceneDs0, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapWaterPipelineLayout, 2, 1, &cubemapWaterDS, 0, nullptr);

    // Culled per-face water commands (filled by the solid 360 cull phase).
    waterIndirectRenderer.drawPreparedWithBuffers(cmd, waterCompactBuffer, waterVisibleCountBuffer);

    vkCmdEndRendering(cmd);
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
        VkImageView wCube = (solid360Renderer_) ? solid360Renderer_->getSolid360View() : VK_NULL_HANDLE;
        prepareSceneTexturesForFrame(app, frameIdx, wBack, wCube);
    }

    // Scene textures were already bound before the async back-face/solid360 tasks were
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
