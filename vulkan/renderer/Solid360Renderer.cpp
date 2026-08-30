#include "Solid360Renderer.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "../../utils/FileReader.hpp"
#include "../ShaderStage.hpp"
#include "../includes/vertex_layouts.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
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
                           VkImageView& view, const char* name) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = viewType;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = baseLayer;
        viewInfo.subresourceRange.layerCount = layerCount;
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 360 image view!");
        app->resources.addImageView(view, name);
    };

    // --- 1. Cubemap color image (6 layers) ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = colorFormat;
        imgInfo.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 6;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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

    // Create a cube-type view so shaders can sample the entire cubemap directly
    createView(cube360ColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
               VK_IMAGE_VIEW_TYPE_CUBE, 0, 6,
               cube360CubeView, "Solid360Renderer: cube360 cube view");

    // Keep the sampler used for solid 360 sampling, which must be clamp-to-edge.
    solid360Sampler = linearSampler;

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

    // --- 3. Dummy 1x1x6 cubemap for binding #11 during cubemap rendering ---
    // This is a SEPARATE image from cube360ColorImage, so that the sampler
    // at binding #11 of cube360GfxDs does not reference the same image as the
    // color attachment, eliminating the SYNC-HAZARD-READ-AFTER-WRITE.
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = colorFormat;
        imgInfo.extent = {1, 1, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 6;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, cube360DummyColorImage, cube360DummyColorAllocation, cube360DummyColorMemory);
    }
    if (app) {
        try {
            app->transitionImageLayoutLayerForce(cube360DummyColorImage, colorFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 6);
            app->setImageLayoutTracked(cube360DummyColorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 6);
        } catch (...) {
            app->setImageLayoutTracked(cube360DummyColorImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
        }
    }
    createView(cube360DummyColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
               VK_IMAGE_VIEW_TYPE_CUBE, 0, 6,
               cube360DummyCubeView, "Solid360Renderer: dummy cube view");

    // NOTE: equirectangular conversion removed. Use the cubemap directly as the
    // reflection target (sample with samplerCube in shaders).
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
    solid360Sampler = VK_NULL_HANDLE;
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthAllocation = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    for (auto &dv : cube360DepthViews) dv = VK_NULL_HANDLE;
    cube360DummyColorImage = VK_NULL_HANDLE;
    cube360DummyColorAllocation = VK_NULL_HANDLE;
    cube360DummyColorMemory = VK_NULL_HANDLE;
    cube360DummyCubeView = VK_NULL_HANDLE;

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

    // ---- Phase A: serial per-face indirect cull (one CB, six face signals) -------
    // prepareCullWithDescriptor shares a single visible-lods scratch buffer, so the
    // 6 face culls MUST run serially (no HW parallelism). Each face writes its OWN
    // compact/visible buffers (faceRes) so the (parallel) raster passes never race
    // with the next face's cull. We keep all 6 culls in ONE command buffer (so the
    // geometry upload inside acquireBuffers happens once, not six times) and signal
    // one dedicated semaphore per face at the end; each raster waits only its own
    // semaphore, so the 6 rasterizations still overlap fully with each other.
    {
        VkCommandBuffer cullCmd;
        beginOne(cullCmd);
        CommandBufferState cullState;
        cmdState = &cullState;

        for (uint32_t face = 0; face < 6; ++face) {
            glm::mat4 faceView = glm::lookAt(camPos, camPos + faces[face].target, faces[face].up);
            glm::mat4 faceVP = faceProj * faceView;

            UniformObject faceUBO = ubo;
            faceUBO.viewProjection = faceVP;
            faceUBO.invViewProjection = glm::inverse(faceVP);
            faceUBO.materialFlags.x = 1.0f;

            // Upload face UBO into the per-face slot of the 6-slot cube UBO via a
            // host-mapped staging copy (avoids vkCmdUpdateBuffer's FULL_QUEUE barrier).
            VkDeviceSize off = static_cast<VkDeviceSize>(face) * sizeof(UniformObject);
            memcpy(staging.map(off), &faceUBO, sizeof(UniformObject));
            VkBufferCopy copy{ off, off, sizeof(UniformObject) };
            vkCmdCopyBuffer(cullCmd, staging.buffer, faceUboBuffer, 1, &copy);

            // Initialise the static bindings 0..9 of each cube360 face compute set
            // exactly once (scene indirect/bounds/visibleLods buffers + this face's
            // own compact/visible targets + veg dummies). These bindings never change
            // for the set's lifetime, so writing once avoids re-touching an in-flight
            // set every frame (VUID-vkUpdateDescriptorSets-None-03047) without needing
            // update-after-bind. prepareCullWithDescriptor fills 17..36.
            auto writeCoreOnce = [&](VkDescriptorSet ds, IndirectRenderer& ind,
                                     VkBuffer compact, VkBuffer visible) {
                if (faceComputeDsInit_.count(ds)) return;
                VkDevice dev = app->getDevice();
                DescriptorWriter(dev)
                    .writeBuffer(ds, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getIndirectBuffer().buffer, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, compact, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getBoundsBuffer().buffer, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, visible, 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVisibleLodsScratchBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .writeBuffer(ds, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                    .flush();
                faceComputeDsInit_[ds] = true;
            };

            // Solid face cull → per-face compact/visible buffers
            if (renderSolid && faceRes.solidComputeDs[face] != VK_NULL_HANDLE &&
                faceRes.compact[face] != VK_NULL_HANDLE && faceRes.visible[face] != VK_NULL_HANDLE) {
                writeCoreOnce(faceRes.solidComputeDs[face], solidRenderer->getIndirectRenderer(),
                              faceRes.compact[face], faceRes.visible[face]);
                solidRenderer->getIndirectRenderer().prepareCullWithDescriptor(
                    cullCmd, faceVP, faceRes.solidComputeDs[face],
                    faceRes.compact[face], faceRes.visible[face], camPos);
            }
            // Water face cull → per-face compact/visible buffers
            if (renderWater && waterRenderer && faceRes.waterComputeDs[face] != VK_NULL_HANDLE &&
                faceRes.waterCompact[face] != VK_NULL_HANDLE && faceRes.waterVisible[face] != VK_NULL_HANDLE) {
                writeCoreOnce(faceRes.waterComputeDs[face], waterRenderer->getIndirectRenderer(),
                              faceRes.waterCompact[face], faceRes.waterVisible[face]);
                waterRenderer->getIndirectRenderer().prepareCullWithDescriptor(
                    cullCmd, faceVP, faceRes.waterComputeDs[face],
                    faceRes.waterCompact[face], faceRes.waterVisible[face], camPos);
            }
        }

        // Signal all 6 per-face cull semaphores; each raster waits only its own.
        // The cull must run after the main cull AND the shadow map are ready, so the
        // rasterization (which samples the shadow map at set 2) never races it.
        std::vector<VkSemaphore> cullSigns(semCullFace, semCullFace + 6);
        app->submitCommandBufferAsyncToQueue(cullCmd, app->getGraphicsQueue(), nullptr,
            { waitCullSolid360, waitShadowSolid360, waitSolid360 }, false, cullSigns,
            { waitCullSolid360Value, waitShadowSolid360Value, waitSolid360Value });
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

        // Transition color layer: tracked layout → COLOR_ATTACHMENT_OPTIMAL
        {
            VkAccessFlags2 srcAccess = (cube360ColorLayouts[face] == VK_IMAGE_LAYOUT_UNDEFINED)
                ? 0 : VK_ACCESS_2_SHADER_READ_BIT;
            RendererUtils::transitionImageLayout(
                fcmd, cube360ColorImage,
                cube360ColorLayouts[face], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                srcAccess, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, face, 1);
        }
        // Transition depth layer: tracked layout → DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        if (app) {
            app->recordTransitionImageLayoutLayer(fcmd, cube360DepthImage, VK_FORMAT_D32_SFLOAT,
                cube360DepthLayouts[face], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                1, face, 1);
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

            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = cube360DepthViews[face];
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
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

        // NOTE: Cubemap faces render solid + sky only. Water reflections are sampled
        // from the main water pipeline's output, not re-rendered into the cube (the
        // cube face uses the swapchain color format, which is incompatible with the
        // single water geometry pipeline that targets R32G32B32A32_SFLOAT). Keeping
        // exactly one water geometry pipeline avoids a second pipeline/layout and the
        // descriptor-set-1 (materials vs brush-depth) layout clash it caused.

        // Transition per-face color/depth layers → SHADER_READ_ONLY_OPTIMAL so the
        // downstream water/composite passes can sample this cubemap layer once the
        // join semaphore fires. Each layer is a distinct image subresource, so the
        // 6 transitions are safe to run concurrently on different queues.
        {
            RendererUtils::transitionImageLayout(
                fcmd, cube360ColorImage,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, face, 1);
        }
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (app) app->recordTransitionImageLayoutLayer(fcmd, cube360DepthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, face, 1);
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
