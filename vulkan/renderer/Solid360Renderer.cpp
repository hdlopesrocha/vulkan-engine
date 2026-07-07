#include "Solid360Renderer.hpp"
#include "../../utils/FileReader.hpp"
#include "../ShaderStage.hpp"
#include "../includes/vertex_layouts.hpp"
#include <stdexcept>
#include <iostream>

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
            depthOnlyPipeline = VK_NULL_HANDLE;
        }
        if (depthOnlyPipelineLayout != VK_NULL_HANDLE) {
            app->resources.removePipelineLayout(depthOnlyPipelineLayout);
            vkDestroyPipelineLayout(dev, depthOnlyPipelineLayout, nullptr);
            depthOnlyPipelineLayout = VK_NULL_HANDLE;
        }
        if (equalComparePipeline != VK_NULL_HANDLE) {
            app->resources.removePipeline(equalComparePipeline);
            vkDestroyPipeline(dev, equalComparePipeline, nullptr);
            equalComparePipeline = VK_NULL_HANDLE;
        }
        if (equalComparePipelineLayout != VK_NULL_HANDLE) {
            app->resources.removePipelineLayout(equalComparePipelineLayout);
            vkDestroyPipelineLayout(dev, equalComparePipelineLayout, nullptr);
            equalComparePipelineLayout = VK_NULL_HANDLE;
        }
    }
    for (uint32_t i = 0; i < STAGING_FRAMES; ++i) {
        if (stagingUBOs[i].buffer != VK_NULL_HANDLE) {
            if (app) app->destroyBuffer(stagingUBOs[i]);
            stagingUBOs[i] = {};
        }
    }
    stagingFrameIndex = 0;
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
        app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT
    );
    ShaderStage tescShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    );
    ShaderStage teseShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    );
    // Depth-only pipeline uses a lightweight fragment shader
    ShaderStage depthFragmentShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/depth_only.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );
    // Color pass uses the full terrain fragment shader
    ShaderStage mainFragmentShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getDescriptorSetLayout());

    // Depth-only pipeline: lightweight fragment shader, no color attachment, depth write, LESS compare
    {
        auto [pipeline, layout] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, depthFragmentShader.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
            true, false,                    // depthWrite=true, colorWrite=false
            VK_COMPARE_OP_LESS,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false,
            {},                             // no color formats
            VK_FORMAT_D32_SFLOAT,
            true                            // noColorAttachment=true
        );
        depthOnlyPipeline = pipeline;
        depthOnlyPipelineLayout = layout;
    }

    // EQUAL-compare color pipeline: full fragment shader, color write, EQUAL depth compare, no depth write
    {
        auto [pipeline, layout] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, mainFragmentShader.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
            false, true,                    // depthWrite=false, colorWrite=true
            VK_COMPARE_OP_LESS_OR_EQUAL,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false,
            { app->getSwapchainImageFormat() },
            VK_FORMAT_D32_SFLOAT
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

void Solid360Renderer::renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                                     SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                                     SolidRenderer* solidRenderer,
                                     VkDescriptorSet mainDescriptorSet,
                                     Buffer& uniformBuffer, const UniformObject& ubo,
                                     VkDescriptorSet computeDs,
                                     VkBuffer compactIndirectBuffer, VkBuffer visibleCountBuffer,
                                     VkDescriptorSet waterComputeDs,
                                     VkBuffer waterCompactIndirectBuffer, VkBuffer waterVisibleCountBuffer) {
    if (!app || cmd == VK_NULL_HANDLE) return;
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

    for (uint32_t face = 0; face < 6; ++face) {
        glm::mat4 faceView = glm::lookAt(camPos, camPos + faces[face].target, faces[face].up);
        glm::mat4 faceVP = faceProj * faceView;

        UniformObject faceUBO = ubo;
        faceUBO.viewProjection = faceVP;

        // Wait for previous face's draws to finish reading the UBO before overwriting it
        {
            VkMemoryBarrier2 preBarrier{};
            preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            preBarrier.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
            preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &preBarrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Upload face UBO via vkCmdCopyBuffer from persistently mapped staging
        // buffer (avoids vkCmdUpdateBuffer's implicit FULL_QUEUE barrier).
        VkDeviceSize faceStagingOff = static_cast<VkDeviceSize>(face) * sizeof(UniformObject);
        memcpy(staging.map(faceStagingOff), &faceUBO, sizeof(UniformObject));
        VkBufferCopy copy{ faceStagingOff, 0, sizeof(UniformObject) };
        vkCmdCopyBuffer(cmd, staging.buffer, uniformBuffer.buffer, 1, &copy);

        VkMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        // Transition color layer: tracked layout → COLOR_ATTACHMENT_OPTIMAL
        {
            VkImageMemoryBarrier2 colorBarrier{};
            colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            colorBarrier.oldLayout = cube360ColorLayouts[face];
            colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.image = cube360ColorImage;
            colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, face, 1 };
            colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            colorBarrier.srcAccessMask = (cube360ColorLayouts[face] == VK_IMAGE_LAYOUT_UNDEFINED)
                ? 0 : VK_ACCESS_2_SHADER_READ_BIT;
            colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &colorBarrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Transition depth layer: tracked layout → DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        if (app) {
            app->recordTransitionImageLayoutLayer(cmd, cube360DepthImage, VK_FORMAT_D32_SFLOAT,
                                                 cube360DepthLayouts[face], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                                 1, face, 1);
        }

        // Per-face frustum cull — must run OUTSIDE dynamic rendering
        if (computeDs != VK_NULL_HANDLE && compactIndirectBuffer != VK_NULL_HANDLE && visibleCountBuffer != VK_NULL_HANDLE) {
            auto &ind = solidRenderer->getIndirectRenderer();
            ind.prepareCullWithDescriptor(cmd, faceVP, computeDs, compactIndirectBuffer, visibleCountBuffer);
        }
        // Per-face water cull with dedicated buffers (no race with main pass)
        if (waterRenderer && waterComputeDs != VK_NULL_HANDLE && waterCompactIndirectBuffer != VK_NULL_HANDLE && waterVisibleCountBuffer != VK_NULL_HANDLE) {
            waterRenderer->getIndirectRenderer().prepareCullWithDescriptor(cmd, faceVP, waterComputeDs, waterCompactIndirectBuffer, waterVisibleCountBuffer);
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

            vkCmdBeginRendering(cmd, &ri);

            VkViewport viewport{0.0f, 0.0f, (float)CUBE360_FACE_SIZE, (float)CUBE360_FACE_SIZE, 0.0f, 1.0f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor{{0, 0}, {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            if (solidRenderer && depthOnlyPipeline != VK_NULL_HANDLE) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthOnlyPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthOnlyPipelineLayout, 0, 1, &mainDescriptorSet, 0, nullptr);
                if (compactIndirectBuffer != VK_NULL_HANDLE && visibleCountBuffer != VK_NULL_HANDLE) {
                    solidRenderer->getIndirectRenderer().drawPreparedWithBuffers(cmd, compactIndirectBuffer, visibleCountBuffer);
                } else {
                    solidRenderer->getIndirectRenderer().drawPrepared(cmd, 0);
                }
            }

            vkCmdEndRendering(cmd);
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

            vkCmdBeginRendering(cmd, &ri);

            VkViewport viewport{0.0f, 0.0f, (float)CUBE360_FACE_SIZE, (float)CUBE360_FACE_SIZE, 0.0f, 1.0f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor{{0, 0}, {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            // Sky first (background, no depth)
            if (skyRenderer) {
                VkPipeline skyPipe = (skyMode == SkySettings::Mode::Grid) ? skyRenderer->getSkyGridPipeline() : skyRenderer->getSkyPipeline();
                VkPipelineLayout skyLayout = (skyMode == SkySettings::Mode::Grid) ? skyRenderer->getSkyGridPipelineLayout() : skyRenderer->getSkyPipelineLayout();
                if (skyPipe != VK_NULL_HANDLE && skyLayout != VK_NULL_HANDLE) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipe);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyLayout, 0, 1, &mainDescriptorSet, 0, nullptr);
                    const auto& skyVBO = skyRenderer->getSkyVBO();
                    if (skyVBO.vertexBuffer.buffer != VK_NULL_HANDLE && skyVBO.indexCount > 0) {
                        VkBuffer vbs[] = {skyVBO.vertexBuffer.buffer};
                        VkDeviceSize offsets[] = {0};
                        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                        vkCmdBindIndexBuffer(cmd, skyVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                        vkCmdDrawIndexed(cmd, skyVBO.indexCount, 1, 0, 0, 0);
                    }
                }
            }

            // Solid geometry with LESS_OR_EQUAL, depth write (redundant but harmless)
            if (solidRenderer) {
                VkPipeline gfxPipe = solidRenderer->getGraphicsPipeline();
                VkPipelineLayout gfxLayout = solidRenderer->getGraphicsPipelineLayout();
                if (gfxPipe != VK_NULL_HANDLE && gfxLayout != VK_NULL_HANDLE) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipe);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxLayout, 0, 1, &mainDescriptorSet, 0, nullptr);
                    if (compactIndirectBuffer != VK_NULL_HANDLE && visibleCountBuffer != VK_NULL_HANDLE) {
                        solidRenderer->getIndirectRenderer().drawPreparedWithBuffers(cmd, compactIndirectBuffer, visibleCountBuffer);
                    } else {
                        solidRenderer->getIndirectRenderer().drawPrepared(cmd, 0);
                    }
                }
            }

            vkCmdEndRendering(cmd);
        }

        // Render water into the cubemap face (with reflection/refraction disabled via skipEnvMap flag in UBO).
        // Depth stays in ATTACHMENT_OPTIMAL; renderWaterIntoCubemap uses the dummy depth for shader sampling
        // so no layout transition is needed.
        if (waterRenderer && waterRenderer->getCubemapWaterPipeline() != VK_NULL_HANDLE) {
            waterRenderer->renderWaterIntoCubemap(cmd,
                cube360FaceViews[face], cube360DepthViews[face],
                mainDescriptorSet, app->getMaterialDescriptorSet(),
                CUBE360_FACE_SIZE,
                waterCompactIndirectBuffer, waterVisibleCountBuffer);
        }

        // Barrier: ensure previous face's draw finishes reading compact/visible
        // buffers before next face's cull overwrites them
        {
            VkBufferMemoryBarrier2 b[2]{};
            uint32_t bc = 0;
            if (compactIndirectBuffer != VK_NULL_HANDLE) {
                b[bc].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                b[bc].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
                b[bc].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
                b[bc].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b[bc].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                b[bc].buffer = compactIndirectBuffer;
                b[bc].size = VK_WHOLE_SIZE; ++bc;
            }
            if (visibleCountBuffer != VK_NULL_HANDLE) {
                b[bc].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                b[bc].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
                b[bc].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
                b[bc].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                b[bc].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                b[bc].buffer = visibleCountBuffer;
                b[bc].size = VK_WHOLE_SIZE; ++bc;
            }
            if (bc > 0) {
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.bufferMemoryBarrierCount = bc;
                dep.pBufferMemoryBarriers = b;
                vkCmdPipelineBarrier2(cmd, &dep);
            }
        }

        // Transition color: COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        {
            VkImageMemoryBarrier2 colorBarrier{};
            colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            colorBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.image = cube360ColorImage;
            colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, face, 1 };
            colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            colorBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            colorBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &colorBarrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Transition depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        if (app) app->recordTransitionImageLayoutLayer(cmd, cube360DepthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, face, 1);
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Cubemap rendering complete; cubemap image view is available for sampling.
    // Wait for all face draws to finish reading the UBO before restoring it
    {
        VkMemoryBarrier2 preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        preBarrier.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Restore UBO via vkCmdCopyBuffer from persistently mapped staging buffer.
    VkDeviceSize restoreStagingOff = 6 * sizeof(UniformObject);
    memcpy(staging.map(restoreStagingOff), &ubo, sizeof(UniformObject));
    VkBufferCopy copy{ restoreStagingOff, 0, sizeof(UniformObject) };
    vkCmdCopyBuffer(cmd, staging.buffer, uniformBuffer.buffer, 1, &copy);
    {
        VkMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Global same-layout barrier: make all per-face COLOR_ATTACHMENT_WRITEs
    // visible to subsequent SHADER_SAMPLED_READs (e.g. in the main pass on the same queue).
    {
        VkImageMemoryBarrier2 globalBarrier{};
        globalBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        globalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        globalBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        globalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        globalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        globalBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        globalBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        globalBarrier.image = cube360ColorImage;
        globalBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 };
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &globalBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}
