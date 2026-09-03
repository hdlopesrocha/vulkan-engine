#include "Solid360Renderer.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "../../utils/FileReader.hpp"
#include "../ShaderStage.hpp"
#include "../includes/vertex_layouts.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

Solid360Renderer::Solid360Renderer() {}
Solid360Renderer::~Solid360Renderer() {}

void Solid360Renderer::init(VulkanApp* app) {
    for (uint32_t i = 0; i < STAGING_FRAMES; ++i) {
        if (stagingUBOs[i].buffer == VK_NULL_HANDLE) {
            stagingUBOs[i] = app->createBuffer(sizeof(UniformObject) * 7,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    }
    stagingFrameIndex = 0;

    // Dedicated environment-cubemap sampler (created once for the app
    // lifetime): trilinear mip filtering so roughness-driven LOD in the
    // shaders actually blurs reflections, clamp-to-edge (correct for
    // cubemaps — repeat would seam), and anisotropy when supported to keep
    // grazing-angle reflections sharp instead of shimmering. maxLod matches
    // CUBE360_MIP_LEVELS - 1 (the shared linear sampler this used to alias
    // has maxLod == 0, which would pin every cubemap fetch to mip 0 and
    // defeat all roughness blur).
    if (!cubeSamplerOwned_ && app) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.mipLodBias = 0.0f;
        si.minLod = 0.0f;
        si.maxLod = static_cast<float>(CUBE360_MIP_LEVELS - 1);
        si.compareEnable = VK_FALSE;
        si.unnormalizedCoordinates = VK_FALSE;
        si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(app->getPhysicalDevice(), &props);
        if (props.limits.maxSamplerAnisotropy > 1.0f) {
            si.anisotropyEnable = VK_TRUE;
            si.maxAnisotropy = std::min(8.0f, props.limits.maxSamplerAnisotropy);
        } else {
            si.anisotropyEnable = VK_FALSE;
            si.maxAnisotropy = 1.0f;
        }
        solid360Sampler = app->createSampler(si, "Solid360Renderer: cube sampler");
        cubeSamplerOwned_ = (solid360Sampler != VK_NULL_HANDLE);
    }
}

void Solid360Renderer::cleanup(VulkanApp* app) {
    if (app) {
        VkDevice dev = app->getDevice();
        if (depthOnlyPipeline != VK_NULL_HANDLE) {
            app->resources.removePipeline(depthOnlyPipeline);
            vkDestroyPipeline(dev, depthOnlyPipeline, nullptr);
        }
        if (depthOnlyPipelineLayout != VK_NULL_HANDLE) {
            app->resources.removePipelineLayout(depthOnlyPipelineLayout);
            vkDestroyPipelineLayout(dev, depthOnlyPipelineLayout, nullptr);
        }
        if (equalComparePipeline != VK_NULL_HANDLE) {
            app->resources.removePipeline(equalComparePipeline);
            vkDestroyPipeline(dev, equalComparePipeline, nullptr);
        }
        if (equalComparePipelineLayout != VK_NULL_HANDLE) {
            app->resources.removePipelineLayout(equalComparePipelineLayout);
            vkDestroyPipelineLayout(dev, equalComparePipelineLayout, nullptr);
        }
        if (cullQueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(dev, cullQueryPool, nullptr);
            cullQueryPool = VK_NULL_HANDLE;
        }
        if (cubeSamplerOwned_ && solid360Sampler != VK_NULL_HANDLE) {
            app->resources.removeSampler(solid360Sampler);
            vkDestroySampler(dev, solid360Sampler, nullptr);
            solid360Sampler = VK_NULL_HANDLE;
            cubeSamplerOwned_ = false;
        }
    }
    for (uint32_t i = 0; i < STAGING_FRAMES; ++i) {
        if (stagingUBOs[i].buffer != VK_NULL_HANDLE) {
            if (app) app->destroyBuffer(stagingUBOs[i]);
            stagingUBOs[i] = {};
        }
    }
    stagingFrameIndex = 0;
}
void Solid360Renderer::createSolid360Targets(VulkanApp* app, VkSampler linearSampler) {
    if (!app) return;
    VkDevice device = app->getDevice();
    VkFormat colorFormat = app->getSwapchainImageFormat();

    auto allocImage = [&](VkImageCreateInfo& imgInfo, VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory) {
        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        allocCI.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        VmaAllocationInfo allocInfo;
        if (vmaCreateImage(app->getVmaAllocator(), &imgInfo, &allocCI, &image, &allocation, &allocInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 360 image with VMA!");
        memory = allocInfo.deviceMemory;
        app->resources.addImageVma(image, allocation, "Solid360Renderer: solid360 image");
        app->resources.setImageArrayLayers(image, imgInfo.arrayLayers);
    };

    auto createView = [&](VkImage image, VkFormat format, VkImageAspectFlags aspect,
                           VkImageViewType viewType, uint32_t baseLayer, uint32_t layerCount,
                           VkImageView& view, const char* name,
                           uint32_t baseMip = 0, uint32_t levelCount = 1) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = viewType;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = baseMip;
        viewInfo.subresourceRange.levelCount = levelCount;
        viewInfo.subresourceRange.baseArrayLayer = baseLayer;
        viewInfo.subresourceRange.layerCount = layerCount;
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 360 image view!");
        app->resources.addImageView(view, name);
    };

    // Mip generation needs linear blitting on this format; without it the
    // shaders must sample mip 0 only (still correct, just sharper).
    {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(app->getPhysicalDevice(), colorFormat, &fp);
        cubeMipmapsSupported_ = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    }
    const uint32_t mipLevels = cubeMipmapsSupported_ ? CUBE360_MIP_LEVELS : 1u;

    // --- 1. Cubemap color image (6 layers, mip-chained for roughness LOD) ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = colorFormat;
        imgInfo.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, 1};
        imgInfo.mipLevels = mipLevels;
        imgInfo.arrayLayers = 6;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER bits feed the per-face blit mip generation recorded in
        // render() after each face finishes rasterizing mip 0.
        imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imgInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, cube360ColorImage, cube360ColorAllocation, cube360ColorMemory);
    }

    if (app) {
        app->setImageLayoutTracked(cube360ColorImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
        // Force transition color image to SHADER_READ_ONLY_OPTIMAL so it is
        // always valid for sampling, even when Solid360 async rendering is disabled.
        try {
            app->transitionImageLayoutLayerForce(cube360ColorImage, colorFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 6);
            app->setImageLayoutTracked(cube360ColorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 6);
        } catch (...) {
            app->setImageLayoutTracked(cube360ColorImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
        }
    }

    for (uint32_t face = 0; face < 6; ++face) {
        createView(cube360ColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_IMAGE_VIEW_TYPE_2D, face, 1,
                   cube360FaceViews[face], "Solid360Renderer: cube360 face view");
    }

    // Create a cube-type view spanning the full mip chain so shaders can
    // sample the entire cubemap with roughness-driven LOD directly
    createView(cube360ColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
               VK_IMAGE_VIEW_TYPE_CUBE, 0, 6,
               cube360CubeView, "Solid360Renderer: cube360 cube view",
               0, mipLevels);

    // Prefer the dedicated trilinear/anisotropic cube sampler created in
    // init(). Fall back to the caller's linear sampler only if init() never
    // ran (defensive; init() is always called from SceneRenderer::init).
    // NOTE: never overwrite an owned sampler here — this function runs on
    // every swapchain resize and must not leak samplers.
    if (!cubeSamplerOwned_) {
        solid360Sampler = linearSampler;
    }

    // --- 2. Depth image with per-face layers (one layer per cubemap face) ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_D32_SFLOAT;
        imgInfo.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 6; // one layer per cubemap face
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, cube360DepthImage, cube360DepthAllocation, cube360DepthMemory);
    }

    if (app) {
        // Force an initial tracked GPU layout for the cubemap depth image
        // so other command buffers that sample the cubemap see a concrete
        // layout instead of UNDEFINED. If the force transition fails,
        // fall back to leaving the tracked layout as UNDEFINED.
        try {
            app->transitionImageLayoutLayerForce(cube360DepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 6);
            app->setImageLayoutTracked(cube360DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 6);
        } catch (...) {
            app->setImageLayoutTracked(cube360DepthImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
        }
    }
    // Create a per-face 2D view that references the corresponding array layer
    for (uint32_t face = 0; face < 6; ++face) {
        createView(cube360DepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_IMAGE_VIEW_TYPE_2D, face, 1,
                   cube360DepthViews[face], "Solid360Renderer: cube360 depth view");
    }

    // Initialize per-face tracked layouts
    for (uint32_t face = 0; face < 6; ++face) {
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // NOTE: equirectangular conversion removed. Use the cubemap directly as the
    // reflection target (sample with samplerCube in shaders).
    // No dummy 1x1 cubemap: the real cubemap above is created before any
    // consumer binds it (SceneRenderer::init / onSwapchainResized run before
    // the first frame), and capture-mode shaders (materialFlags.x == 1) skip
    // env-map sampling so binding the real view during capture is feedback-free.
}

void Solid360Renderer::destroySolid360Targets(VulkanApp* app) {
    if (app && cube360DepthImage != VK_NULL_HANDLE) {
        app->setImageLayoutTracked(cube360DepthImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
    }
    cube360ColorImage = VK_NULL_HANDLE;
    cube360ColorAllocation = VK_NULL_HANDLE;
    cube360ColorMemory = VK_NULL_HANDLE;
    for (auto& v : cube360FaceViews) v = VK_NULL_HANDLE;
    cube360CubeView = VK_NULL_HANDLE;
    // The dedicated cube sampler (when owned) survives target recreation —
    // only drop the fallback alias to the caller's shared sampler.
    if (!cubeSamplerOwned_) {
        solid360Sampler = VK_NULL_HANDLE;
    }
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthAllocation = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    for (auto &dv : cube360DepthViews) dv = VK_NULL_HANDLE;

    // Reset tracked layouts
    for (uint32_t face = 0; face < 6; ++face) {
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_UNDEFINED;
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void Solid360Renderer::createSolid360Pipelines(VulkanApp* app) {
    if (!app) return;

    ShaderStage vertexShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.vert.spv"),
        VK_SHADER_STAGE_VERTEX_BIT
    );
    ShaderStage tescShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.tesc.spv"),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    );
    ShaderStage teseShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.tese.spv"),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    );
    // Depth-only pipeline uses a lightweight fragment shader
    ShaderStage depthFragmentShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/depth_only.frag.spv"),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );
    // Color pass uses the full terrain fragment shader
    ShaderStage mainFragmentShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.frag.spv"),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getDescriptorSetLayout());
    if (app->getBrushDepthDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getBrushDepthDescriptorSetLayout());

    // Depth-only pipeline: lightweight fragment shader, no color attachment, depth write, LESS compare
    {
        GraphicsPipelineConfig depthCfg{};
        depthCfg.colorWrite = false;
        depthCfg.depthCompareOp = VK_COMPARE_OP_LESS;
        depthCfg.noColorAttachment = true;
        auto [pipeline, layout] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, depthFragmentShader.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            depthCfg
        );
        depthOnlyPipeline = pipeline;
        depthOnlyPipelineLayout = layout;
    }

    // EQUAL-compare color pipeline: full fragment shader, color write, EQUAL depth compare, no depth write
    {
        GraphicsPipelineConfig colorCfg{};
        colorCfg.depthWriteEnable = false;
        colorCfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        colorCfg.colorFormats = { app->getSwapchainImageFormat() };
        auto [pipeline, layout] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, mainFragmentShader.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            colorCfg
        );
        equalComparePipeline = pipeline;
        equalComparePipelineLayout = layout;
    }

    // Clear local handles; destruction via VulkanResourceManager
    mainFragmentShader.info.module = VK_NULL_HANDLE;
    depthFragmentShader.info.module = VK_NULL_HANDLE;
    teseShader.info.module = VK_NULL_HANDLE;
    tescShader.info.module = VK_NULL_HANDLE;
    vertexShader.info.module = VK_NULL_HANDLE;
}

void Solid360Renderer::render(VulkanApp* app,
                                      SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                                      SolidRenderer* solidRenderer,
                                      VkDescriptorSet brushDepthSet,
                                      VkBuffer faceUboBuffer,
                                      const Cube360FaceResources& faceRes,
                                      const UniformObject& ubo,
                          bool renderSolid, bool renderWater,
                          VkSemaphore waitCullSolid360, uint64_t waitCullSolid360Value,
                          VkSemaphore waitShadowSolid360, uint64_t waitShadowSolid360Value,
                          VkSemaphore waitSolid360, uint64_t waitSolid360Value,
                          const VkSemaphore (&semCullFace)[6],
                                      const VkSemaphore (&semFaceDone)[6],
                                      VkSemaphore signalSolid360, uint64_t signalSolid360Value,
                                      uint32_t frameIndex) {
    if (!app || faceUboBuffer == VK_NULL_HANDLE) return;
    if (cube360FaceViews[0] == VK_NULL_HANDLE) return;

    // Advance to next staging buffer slot for frame-in-flight isolation
    stagingFrameIndex++;
    Buffer& staging = stagingUBOs[stagingFrameIndex % STAGING_FRAMES];

    glm::vec3 camPos = glm::vec3(ubo.viewPos);
    struct FaceInfo { glm::vec3 target; glm::vec3 up; };
    // Cubemap face order and orientation: +X, -X, +Y, -Y, +Z, -Z.
    // NOTE: face targets are intentionally inverted to match the convention
    // used by water.frag's reflect(refract(viewDir, ...)) which passes the
    // view direction directly (surface→eye) rather than negating it first.
    const FaceInfo faces[6] = {
        { glm::vec3(-1, 0, 0), glm::vec3(0,-1, 0) }, // +X
        { glm::vec3( 1, 0, 0), glm::vec3(0,-1, 0) }, // -X
        { glm::vec3( 0,-1, 0), glm::vec3(0, 0, 1) },  // +Y
        { glm::vec3( 0, 1, 0), glm::vec3(0, 0,-1) },  // -Y
        { glm::vec3( 0, 0,-1), glm::vec3(0,-1, 0) }, // +Z
        { glm::vec3( 0, 0, 1), glm::vec3(0,-1, 0) }, // -Z
    };

    glm::mat4 faceProj = glm::perspective(glm::radians(90.0f), 1.0f, ubo.passParams.z, ubo.passParams.w);
    faceProj[1][1] *= -1;

    // Lazily ensure water cubemap resources are ready
    if (waterRenderer) {
        waterRenderer->ensureCubemapResources(app, app->getSwapchainImageFormat());
    }

    auto beginOne = [&](VkCommandBuffer& out) {
        out = app->allocatePrimaryCommandBuffer();
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(out, &bi) != VK_SUCCESS)
            throw std::runtime_error("[solid360] failed to begin command buffer");
    };

    // ---- Phase A: parallel per-face indirect cull (6 CBs, 6 queues) ------------
    // Each face records its OWN cull command buffer: UBO slot upload + solid cull
    // + water cull, writing that face's compact/visible outputs and its OWN
    // visible-lods scratch buffer (binding 4, one per face — see
    // IndirectRenderer::ensureFaceScratchBuffers). No writable resource is shared
    // between faces, so the 6 culls dispatch concurrently on the distinct
    // graphics-family queues (app->getCubeQueue(face); graphics queues have
    // compute capability). Each CB signals its own semCullFace[face], which only
    // that face's raster CB waits on. The acquireBuffers barrier inside
    // prepareCullWithDescriptor is recorded per-CB (idempotent reads of the same
    // shared geometry — safe to repeat).
    // The culls run after the main cull AND the shadow map are ready (timeline
    // waits below), so rasterization (which samples the shadow map at set 2)
    // never races them. GPU timestamp profiling brackets each face's cull work
    // (see profileThisFrame); the non-blocking readback + log follow the loop.
    {
        // Lazily create the 12-query timestamp pool (2 queries per face) when
        // the graphics queue family supports timestamps; otherwise profiling
        // stays disabled (cullQueryPool == VK_NULL_HANDLE).
        if (cullQueryPool == VK_NULL_HANDLE && app) {
            uint32_t qCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(app->getPhysicalDevice(), &qCount, nullptr);
            std::vector<VkQueueFamilyProperties> qProps(qCount);
            if (qCount > 0)
                vkGetPhysicalDeviceQueueFamilyProperties(app->getPhysicalDevice(), &qCount, qProps.data());
            for (const auto& q : qProps) {
                if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    // Timestamp period lives in the device limits (ns per tick);
                    // per-queue-family timestampValidBits gates query support.
                    if (q.timestampValidBits > 0) {
                        VkPhysicalDeviceProperties props{};
                        vkGetPhysicalDeviceProperties(app->getPhysicalDevice(), &props);
                        cullTimestampPeriod = props.limits.timestampPeriod;
                        VkQueryPoolCreateInfo qi{};
                        qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
                        qi.queryCount = 12;
                        if (vkCreateQueryPool(app->getDevice(), &qi, nullptr, &cullQueryPool) != VK_SUCCESS)
                            cullQueryPool = VK_NULL_HANDLE;
                    }
                    break;
                }
            }
        }
        ++cullFrameCounter;
        // Profile one frame out of every 240: the next reset lands 240 frames
        // later, so the host readback below can never race a device reset of
        // the same queries (no WAIT stall, no reset-vs-read hazard).
        const bool profileThisFrame = (cullQueryPool != VK_NULL_HANDLE) && (cullFrameCounter % 240 == 1);

        // Initialise the static bindings 0..9 of each cube360 face compute set
        // exactly once (scene indirect/bounds buffers + this face's own
        // compact/visible targets + OWN scratch buffer at binding 4 + veg
        // dummies). These bindings never change for the set's lifetime, so
        // writing once avoids re-touching an in-flight set every frame
        // (VUID-vkUpdateDescriptorSets-None-03047) without needing
        // update-after-bind. prepareCullWithDescriptor fills 17..36. When
        // capacity growth reallocates a face scratch buffer, binding 4 alone is
        // re-written exactly once (tracked via faceComputeScratchBound_).
        auto writeCoreOnce = [&](VkDescriptorSet ds, IndirectRenderer& ind,
                                 VkBuffer compact, VkBuffer visible, VkBuffer scratch) {
            VkDevice dev = app->getDevice();
            VkBuffer want = (scratch != VK_NULL_HANDLE) ? scratch : ind.getVisibleLodsScratchBuffer();
            auto it = faceComputeDsInit_.find(ds);
            if (it == faceComputeDsInit_.end()) {
                DescriptorWriter(dev)
                    .writeBuffer(ds, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getIndirectBuffer().buffer, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, compact, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getBoundsBuffer().buffer, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, visible, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, want, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .flush();
                faceComputeDsInit_[ds] = true;
                faceComputeScratchBound_[ds] = want;
                return;
            }
            if (want != VK_NULL_HANDLE && faceComputeScratchBound_[ds] != want) {
                DescriptorWriter(dev)
                    .writeBuffer(ds, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, want, 0, VK_WHOLE_SIZE)
                    .flush();
                faceComputeScratchBound_[ds] = want;
            }
        };

        // One cull CB per face; each is submitted to its own queue below.
        for (uint32_t face = 0; face < 6; ++face) {
            glm::mat4 faceView = glm::lookAt(camPos, camPos + faces[face].target, faces[face].up);
            glm::mat4 faceVP = faceProj * faceView;

            UniformObject faceUBO = ubo;
            faceUBO.viewProjection = faceVP;
            faceUBO.invViewProjection = glm::inverse(faceVP);
            faceUBO.materialFlags.x = 1.0f;

            // Resolve this face's scratch buffers, ensuring they exist at the
            // current capacity (uvec2 per draw entry). Null when the renderer
            // has no capacity yet — prepareCullWithDescriptor then falls back
            // to the legacy serial scratch for that dispatch.
            VkBuffer solidScratch = VK_NULL_HANDLE;
            if (renderSolid && solidRenderer) {
                IndirectRenderer& ind = solidRenderer->getIndirectRenderer();
                const size_t cap = ind.getMeshCapacity();
                if (cap > 0) {
                    ind.ensureFaceScratchBuffers(app, static_cast<VkDeviceSize>(cap * sizeof(uint32_t) * 2));
                    solidScratch = ind.getVisibleLodsScratchBuffer(face);
                }
            }
            VkBuffer waterScratch = VK_NULL_HANDLE;
            if (renderWater && waterRenderer) {
                IndirectRenderer& ind = waterRenderer->getIndirectRenderer();
                const size_t cap = ind.getMeshCapacity();
                if (cap > 0) {
                    ind.ensureFaceScratchBuffers(app, static_cast<VkDeviceSize>(cap * sizeof(uint32_t) * 2));
                    waterScratch = ind.getVisibleLodsScratchBuffer(face);
                }
            }

            VkCommandBuffer cullCmd;
            beginOne(cullCmd);
            CommandBufferState cullState;
            cmdState = &cullState;

            // Upload face UBO into the per-face slot of the 6-slot cube UBO via a
            // host-mapped staging copy (avoids vkCmdUpdateBuffer's FULL_QUEUE barrier).
            // The memcpy runs on the CPU at record time; the 6 GPU copies read
            // disjoint ranges of the same staging buffer and write disjoint dst
            // slots, so concurrent execution is race-free.
            VkDeviceSize off = static_cast<VkDeviceSize>(face) * sizeof(UniformObject);
            memcpy(staging.map(off), &faceUBO, sizeof(UniformObject));
            VkBufferCopy copy{ off, off, sizeof(UniformObject) };
            vkCmdCopyBuffer(cullCmd, staging.buffer, faceUboBuffer, 1, &copy);

            if (profileThisFrame) {
                vkCmdResetQueryPool(cullCmd, cullQueryPool, face * 2, 2);
                vkCmdWriteTimestamp(cullCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    cullQueryPool, face * 2);
            }

            // Solid face cull → per-face compact/visible buffers + face scratch
            if (renderSolid && solidRenderer && faceRes.solidComputeDs[face] != VK_NULL_HANDLE &&
                faceRes.compact[face] != VK_NULL_HANDLE && faceRes.visible[face] != VK_NULL_HANDLE) {
                writeCoreOnce(faceRes.solidComputeDs[face], solidRenderer->getIndirectRenderer(),
                              faceRes.compact[face], faceRes.visible[face], solidScratch);
                solidRenderer->getIndirectRenderer().prepareCullWithDescriptor(
                    cullCmd, faceVP, faceRes.solidComputeDs[face],
                    faceRes.compact[face], faceRes.visible[face], camPos,
                    8.0f, 16, true, false, solidScratch);
            }
            // Water face cull → per-face compact/visible buffers + face scratch
            if (renderWater && waterRenderer && faceRes.waterComputeDs[face] != VK_NULL_HANDLE &&
                faceRes.waterCompact[face] != VK_NULL_HANDLE && faceRes.waterVisible[face] != VK_NULL_HANDLE) {
                writeCoreOnce(faceRes.waterComputeDs[face], waterRenderer->getIndirectRenderer(),
                              faceRes.waterCompact[face], faceRes.waterVisible[face], waterScratch);
                waterRenderer->getIndirectRenderer().prepareCullWithDescriptor(
                    cullCmd, faceVP, faceRes.waterComputeDs[face],
                    faceRes.waterCompact[face], faceRes.waterVisible[face], camPos,
                    8.0f, 16, true, false, waterScratch);
            }

            if (profileThisFrame) {
                vkCmdWriteTimestamp(cullCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    cullQueryPool, face * 2 + 1);
            }

            // Submit to this face's graphics-family queue (compute-capable).
            // Waits: main cull + shadow map + prior solid360 work (timeline).
            // Signals: this face's raster CB waits on semCullFace[face] only, so
            // the 6 cull→raster pairs overlap fully across the cube queues.
            app->submitCommandBufferAsyncToQueue(cullCmd, app->getCubeQueue(face), nullptr,
                { waitCullSolid360, waitShadowSolid360, waitSolid360 }, false, { semCullFace[face] },
                { waitCullSolid360Value, waitShadowSolid360Value, waitSolid360Value });
        }
        if (profileThisFrame) cullProfilePending = true;
        // Non-blocking readback of the last profiled frame's cull timestamps.
        // Skipped on the profile frame itself (results cannot be ready yet) and
        // whenever any query is still unavailable — the pool is only reset on
        // profile frames (1/240), so a pending read never races a device reset.
        // Parallel wall time should approach a single face's time, not the sum.
        if (cullProfilePending && !profileThisFrame && cullQueryPool != VK_NULL_HANDLE) {
            struct TsAvail { uint64_t ts; uint64_t avail; };
            TsAvail results[12] = {};
            VkResult qr = vkGetQueryPoolResults(app->getDevice(), cullQueryPool, 0, 12,
                sizeof(results), results, sizeof(TsAvail),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (qr == VK_SUCCESS) {
                bool allReady = true;
                for (int i = 0; i < 12; ++i) allReady &= (results[i].avail != 0);
                if (allReady) {
                    uint64_t faceNs[6] = {};
                    uint64_t sum = 0, mx = 0;
                    for (uint32_t f = 0; f < 6; ++f) {
                        const uint64_t d = results[f * 2 + 1].ts - results[f * 2].ts;
                        faceNs[f] = static_cast<uint64_t>(d * cullTimestampPeriod);
                        sum += faceNs[f];
                        mx = std::max(mx, faceNs[f]);
                    }
                    fprintf(stderr,
                        "[solid360] cull profile: face ns=[%llu,%llu,%llu,%llu,%llu,%llu] "
                        "max=%llu sum=%llu (parallel wall should approach max, not sum)\n",
                        (unsigned long long)faceNs[0], (unsigned long long)faceNs[1],
                        (unsigned long long)faceNs[2], (unsigned long long)faceNs[3],
                        (unsigned long long)faceNs[4], (unsigned long long)faceNs[5],
                        (unsigned long long)mx, (unsigned long long)sum);
                    cullProfilePending = false;
                }
            } else {
                cullProfilePending = false; // pool reset or device loss — retry next profile frame
            }
        }
        cmdState = nullptr;
    }

    // ---- Phase B: parallel per-face rasterization on distinct graphics queues ----
    // Each face writes a distinct array layer of the cube (color + depth), binds its
    // own gfx descriptor set (binding 0 = its own UBO slot) and its own compact/visible
    // indirect buffers, so the 6 rasterizations have no shared writable resource and
    // can overlap fully across the up-to-6 queues returned by getCubeQueue(face).
    for (uint32_t face = 0; face < 6; ++face) {
        VkCommandBuffer fcmd;
        beginOne(fcmd);
        CommandBufferState faceState;
        cmdState = &faceState;
        VkDescriptorSet gfxSet = faceRes.gfxDs[face];

        // Batched per-face begin barriers (single vkCmdPipelineBarrier2 for
        // this face's color + depth layers; was: one call per image): color
        // tracked → COLOR_ATTACHMENT_OPTIMAL, depth tracked →
        // DEPTH_STENCIL_ATTACHMENT_OPTIMAL. Each face writes a distinct array
        // layer of the cube images, so the 6 per-face batches stay independent
        // across the parallel cube queues. Same stage/access mapping as the
        // single transitions; already-correct layouts become no-ops.
        if (app) {
            std::vector<VulkanApp::BatchTransition> beginBatch;
            beginBatch.reserve(2);
            VulkanApp::BatchTransition colorBegin{};
            colorBegin.image          = cube360ColorImage;
            colorBegin.format         = app->getSwapchainImageFormat();
            colorBegin.oldLayout      = cube360ColorLayouts[face];
            colorBegin.newLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBegin.baseArrayLayer = face;
            colorBegin.layerCount     = 1;
            beginBatch.push_back(colorBegin);
            VulkanApp::BatchTransition depthBegin{};
            depthBegin.image          = cube360DepthImage;
            depthBegin.format         = VK_FORMAT_D32_SFLOAT;
            depthBegin.oldLayout      = cube360DepthLayouts[face];
            depthBegin.newLayout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthBegin.baseArrayLayer = face;
            depthBegin.layerCount     = 1;
            beginBatch.push_back(depthBegin);
            app->recordTransitionBatch(fcmd, beginBatch);
        }

        // ── Instance 1: Depth pre-pass (no color, lightweight depth_only.frag) ──
        {
            VkClearValue depthClear{};
            depthClear.depthStencil = {1.0f, 0};

            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = cube360DepthViews[face];
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.clearValue = depthClear;

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset = {0, 0};
            ri.renderArea.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 0;
            ri.pColorAttachments = nullptr;
            ri.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(fcmd, &ri);

            VkViewport viewport{0.0f, 0.0f, (float)CUBE360_FACE_SIZE, (float)CUBE360_FACE_SIZE, 0.0f, 1.0f};
            vkCmdSetViewport(fcmd, 0, 1, &viewport);
            VkRect2D scissor{{0, 0}, {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE}};
            vkCmdSetScissor(fcmd, 0, 1, &scissor);

            if (renderSolid && solidRenderer && depthOnlyPipeline != VK_NULL_HANDLE) {
                if (cmdState) cmdState->bindGraphicsPipeline(fcmd, depthOnlyPipeline);
                else vkCmdBindPipeline(fcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthOnlyPipeline);
                if (cmdState) cmdState->bindGraphicsDescriptorSets(fcmd, depthOnlyPipelineLayout, 0, 1, &gfxSet, 0, nullptr);
                else vkCmdBindDescriptorSets(fcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthOnlyPipelineLayout, 0, 1, &gfxSet, 0, nullptr);
                if (faceRes.compact[face] != VK_NULL_HANDLE && faceRes.visible[face] != VK_NULL_HANDLE) {
                    solidRenderer->getIndirectRenderer().drawPreparedWithBuffers(fcmd, faceRes.compact[face], faceRes.visible[face]);
                } else {
                    solidRenderer->getIndirectRenderer().drawPrepared(fcmd, 0);
                }
            }

            vkCmdEndRendering(fcmd);
        }

        // ── Instance 2: Color pass (load prepass depth, use solid renderer's pipeline) ──
        {
            VkClearValue colorClear{};
            colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

            VkRenderingAttachmentInfo colorAtt{};
            colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAtt.imageView = cube360FaceViews[face];
            colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtt.clearValue = colorClear;

            // Depth must be stored (not DONT_CARE) when the water pass follows:
            // it runs in its own rendering instance and loads this depth.
            const bool waterFollows = renderWater && waterRenderer &&
                faceRes.waterCompact[face] != VK_NULL_HANDLE && faceRes.waterVisible[face] != VK_NULL_HANDLE;
            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = cube360DepthViews[face];
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAtt.storeOp = waterFollows ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.clearValue = {1.0f, 0};

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset = {0, 0};
            ri.renderArea.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &colorAtt;
            ri.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(fcmd, &ri);

            VkViewport viewport{0.0f, 0.0f, (float)CUBE360_FACE_SIZE, (float)CUBE360_FACE_SIZE, 0.0f, 1.0f};
            vkCmdSetViewport(fcmd, 0, 1, &viewport);
            VkRect2D scissor{{0, 0}, {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE}};
            vkCmdSetScissor(fcmd, 0, 1, &scissor);

            // Sky first (background, no depth)
            if (skyRenderer) {
                VkPipeline skyPipe = (skyMode == SkySettings::Mode::Grid) ? skyRenderer->getSkyFullscreenGridPipeline() : skyRenderer->getSkyFullscreenPipeline();
                VkPipelineLayout skyLayout = (skyMode == SkySettings::Mode::Grid) ? skyRenderer->getSkyFullscreenGridPipelineLayout() : skyRenderer->getSkyFullscreenPipelineLayout();
                if (skyPipe != VK_NULL_HANDLE && skyLayout != VK_NULL_HANDLE) {
                    if (cmdState) cmdState->bindGraphicsPipeline(fcmd, skyPipe);
                    else vkCmdBindPipeline(fcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipe);
                    if (cmdState) cmdState->bindGraphicsDescriptorSets(fcmd, skyLayout, 0, 1, &gfxSet, 0, nullptr);
                    else vkCmdBindDescriptorSets(fcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyLayout, 0, 1, &gfxSet, 0, nullptr);
                    vkCmdDraw(fcmd, 3, 1, 0, 0);
                }
            }

            // Solid geometry with LESS_OR_EQUAL, depth write (redundant but harmless)
            if (renderSolid && solidRenderer) {
                VkPipeline gfxPipe = solidRenderer->getGraphicsPipeline();
                VkPipelineLayout gfxLayout = solidRenderer->getGraphicsPipelineLayout();
                if (gfxPipe != VK_NULL_HANDLE && gfxLayout != VK_NULL_HANDLE) {
                    if (cmdState) cmdState->bindGraphicsPipeline(fcmd, gfxPipe);
                    else vkCmdBindPipeline(fcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipe);
                    // gfxPipe uses main.frag which references brush depth at set=1. Bind a
                    // DS with the brush-depth layout there; fall back to the dedicated
                    // cube360 brush-depth DS when the caller passed a null set (otherwise
                    // set 1 would be left bound to the water block's materials DS).
                    VkDescriptorSet solidSet1 = (brushDepthSet != VK_NULL_HANDLE) ? brushDepthSet : faceRes.brushDepthDs;
                    VkDescriptorSet bindSets[2] = { gfxSet, solidSet1 };
                    uint32_t bindCount = (solidSet1 != VK_NULL_HANDLE) ? 2 : 1;
                    if (cmdState) cmdState->bindGraphicsDescriptorSets(fcmd, gfxLayout, 0, bindCount, bindSets, 0, nullptr);
                    else vkCmdBindDescriptorSets(fcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxLayout, 0, bindCount, bindSets, 0, nullptr);
                    if (faceRes.compact[face] != VK_NULL_HANDLE && faceRes.visible[face] != VK_NULL_HANDLE) {
                        solidRenderer->getIndirectRenderer().drawPreparedWithBuffers(fcmd, faceRes.compact[face], faceRes.visible[face]);
                    } else {
                        solidRenderer->getIndirectRenderer().drawPrepared(fcmd, 0);
                    }
                }
            }

            vkCmdEndRendering(fcmd);
        }

        // ── Instance 3: water into the cubemap face ──
        // Water IS rendered into the cube (it is what the water reflections show).
        // The main water geometry pipeline targets R32G32B32A32_SFLOAT, so the cube
        // face (swapchain format) uses WaterRenderer's cubemap-compatible pipeline.
        // renderWaterIntoCubemap opens its own rendering instance with LOAD ops,
        // compositing water over the solid+sky color and depth-testing against the
        // prepassed solid depth. The face UBO's materialFlags.x == 1 (capture mode)
        // makes water.frag skip reflection/refraction, so the pass never samples
        // the cubemap it is writing; set 2 binds the back-face dummy depth +
        // the real cube view (capture-skipped, so no feedback).
        if (renderWater && waterRenderer &&
            faceRes.waterCompact[face] != VK_NULL_HANDLE && faceRes.waterVisible[face] != VK_NULL_HANDLE) {
            waterRenderer->renderWaterIntoCubemap(fcmd, gfxSet,
                cube360FaceViews[face], cube360DepthViews[face], CUBE360_FACE_SIZE,
                faceRes.waterCompact[face], faceRes.waterVisible[face]);
        }

        // Per-face end barriers so the downstream water/composite passes can
        // sample this cubemap layer once the join semaphore fires. Each layer
        // is a distinct image subresource, so the 6 faces stay safe to run
        // concurrently on different queues.
        // Depth goes straight to SHADER_READ_ONLY_OPTIMAL via the batched
        // helper (keeps the app's layout tracker authoritative for depth).
        if (app) {
            VulkanApp::BatchTransition depthEnd{};
            depthEnd.image          = cube360DepthImage;
            depthEnd.format         = VK_FORMAT_D32_SFLOAT;
            depthEnd.oldLayout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthEnd.newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthEnd.baseArrayLayer = face;
            depthEnd.layerCount     = 1;
            app->recordTransitionBatch(fcmd, { depthEnd });
        }
        // Color mip 0 is blit-downsampled into mips 1..N-1 right here in the
        // face's own command buffer (distinct array layer per face → no cross-
        // face hazard), ending with every mip in SHADER_READ_ONLY_OPTIMAL.
        // Without this, roughness-driven textureLod() in the shaders would
        // have no mip chain to sample and reflections would stay razor sharp
        // on every material. When the format cannot be linearly blitted,
        // fall back to a plain mip-0 transition (still correct).
        if (app && cubeMipmapsSupported_) {
            const uint32_t mips = CUBE360_MIP_LEVELS;
            auto barrierMip = [&](uint32_t mip, VkImageLayout oldL, VkImageLayout newL,
                                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
                VkImageMemoryBarrier2 b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = cube360ColorImage;
                b.oldLayout = oldL;
                b.newLayout = newL;
                b.srcStageMask = srcStage;
                b.srcAccessMask = srcAccess;
                b.dstStageMask = dstStage;
                b.dstAccessMask = dstAccess;
                b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                b.subresourceRange.baseMipLevel = mip;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.baseArrayLayer = face;
                b.subresourceRange.layerCount = 1;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(fcmd, &dep);
            };
            // Why COLOR_ATTACHMENT → TRANSFER_SRC: the blit below reads mip 0
            // as its source, and TRANSFER_SRC_OPTIMAL is the only layout that
            // guarantees transfer-read visibility of attachment writes.
            barrierMip(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            int32_t w = static_cast<int32_t>(CUBE360_FACE_SIZE);
            int32_t h = static_cast<int32_t>(CUBE360_FACE_SIZE);
            for (uint32_t i = 1; i < mips; ++i) {
                // Why UNDEFINED as oldLayout: mip contents are fully
                // overwritten by the blit, and UNDEFINED is a wildcard that is
                // valid regardless of the layout the mip was left in last
                // frame (SHADER_READ) or before first use (UNDEFINED).
                barrierMip(i, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkImageBlit blit{};
                blit.srcOffsets[0] = {0, 0, 0};
                blit.srcOffsets[1] = {w, h, 1};
                blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.srcSubresource.mipLevel = i - 1;
                blit.srcSubresource.baseArrayLayer = face;
                blit.srcSubresource.layerCount = 1;
                blit.dstOffsets[0] = {0, 0, 0};
                blit.dstOffsets[1] = {std::max(1, w / 2), std::max(1, h / 2), 1};
                blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.dstSubresource.mipLevel = i;
                blit.dstSubresource.baseArrayLayer = face;
                blit.dstSubresource.layerCount = 1;
                vkCmdBlitImage(fcmd,
                               cube360ColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               cube360ColorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &blit, VK_FILTER_LINEAR);
                // The just-consumed source mip is no longer needed downstream:
                // release it to SHADER_READ so later passes can sample it.
                // The final mip instead goes DST → SHADER_READ after the loop.
                if (i < mips - 1) {
                    // Why TRANSFER_SRC → TRANSFER_SRC is NOT used: mip i must
                    // become a blit SOURCE for the next iteration, so it goes
                    // DST → SRC here; mip i-1 goes SRC → SHADER_READ.
                    barrierMip(i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                }
                barrierMip(i - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                if (w > 1) w /= 2;
                if (h > 1) h /= 2;
            }
            // Last mip was never promoted to SRC: DST → SHADER_READ.
            barrierMip(mips - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            // Manual barriers bypass the app's layout tracker, so re-sync both
            // of its channels (same pattern as SolidRenderer::endPass):
            // - the authoritative map, so later record calls resolve the
            //   correct oldLayout for mip 0;
            // - this command buffer's pending queue, so submit-time
            //   preApplyPendingLayoutsBeforeSubmit promotes SHADER_READ
            //   instead of the stale COLOR_ATTACHMENT left by beginBatch.
            //   Without the pending entry the next frame's beginBatch skips
            //   its SHADER_READ → COLOR_ATTACHMENT barrier as a no-op and the
            //   face renders into a SHADER_READ image (VUID-vkCmdDraw-None-09600).
            app->setImageLayoutTracked(cube360ColorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, face, 1);
            app->recordTrackedLayoutForCommandBuffer(fcmd, cube360ColorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, face, 1);
        } else if (app) {
            VulkanApp::BatchTransition colorEnd{};
            colorEnd.image          = cube360ColorImage;
            colorEnd.format         = app->getSwapchainImageFormat();
            colorEnd.oldLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorEnd.newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorEnd.baseArrayLayer = face;
            colorEnd.layerCount     = 1;
            app->recordTransitionBatch(fcmd, { colorEnd });
        }
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Wait on this face's cull (which itself waited on the main cull + shadow map);
        // signal done. Each face has its own cull semaphore, so the 6 rasterizations
        // overlap fully across the cube queues without sharing a binary semaphore.
        app->submitCommandBufferAsyncToQueue(fcmd, app->getCubeQueue(face), nullptr,
            { semCullFace[face] }, false, { semFaceDone[face] });
    }

    // ---- Join: wait all 6 faces, then signal semSolid360 for the water pass ------
    // The cube image (all 6 layers) is now complete and in SHADER_READ_ONLY_OPTIMAL;
    // the async water pass waits on semSolid360, so it samples a coherent cubemap.
    VkCommandBuffer joinCmd;
    beginOne(joinCmd);
    std::vector<VkSemaphore> joinWaits(semFaceDone, semFaceDone + 6);
    app->submitCommandBufferAsyncToQueue(joinCmd, app->getGraphicsQueue(), nullptr,
        joinWaits, false, { signalSolid360 }, {}, 0, { signalSolid360Value });

    cmdState = nullptr;
}
