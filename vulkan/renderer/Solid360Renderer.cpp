#include "Solid360Renderer.hpp"
#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>

Solid360Renderer::Solid360Renderer() {}
Solid360Renderer::~Solid360Renderer() {}

void Solid360Renderer::init(VulkanApp* app) {
    (void)app;
}

void Solid360Renderer::cleanup(VulkanApp* app) {
    (void)app;
    cube360ColorImage = VK_NULL_HANDLE;
    cube360ColorMemory = VK_NULL_HANDLE;
    for (auto& v : cube360FaceViews) v = VK_NULL_HANDLE;
    cube360CubeView = VK_NULL_HANDLE;
    solid360Sampler = VK_NULL_HANDLE;
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    for (auto &dv : cube360DepthViews) dv = VK_NULL_HANDLE;
}

void Solid360Renderer::createSolid360Targets(VulkanApp* app, VkSampler linearSampler) {
    if (!app) return;
    VkDevice device = app->getDevice();
    VkFormat colorFormat = app->getSwapchainImageFormat();

    auto allocImage = [&](VkImageCreateInfo& imgInfo, VkImage& image, VkDeviceMemory& memory) {
        if (vkCreateImage(device, &imgInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 360 image!");
        app->resources.addImage(image, "Solid360Renderer: solid360 image");
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, image, &memReqs);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate 360 image memory!");
        app->resources.addDeviceMemory(memory, "Solid360Renderer: solid360 memory");
        vkBindImageMemory(device, image, memory, 0);
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
        allocImage(imgInfo, cube360ColorImage, cube360ColorMemory);
    }

    if (app) {
        app->setImageLayoutTracked(cube360ColorImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
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
        allocImage(imgInfo, cube360DepthImage, cube360DepthMemory);
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
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_UNDEFINED;
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // NOTE: equirectangular conversion removed. Use the cubemap directly as the
    // reflection target (sample with samplerCube in shaders).
}

void Solid360Renderer::destroySolid360Targets(VulkanApp* app) {
    if (app && cube360DepthImage != VK_NULL_HANDLE) {
        app->setImageLayoutTracked(cube360DepthImage, VK_IMAGE_LAYOUT_UNDEFINED, 0, 6);
    }
    cube360ColorImage = VK_NULL_HANDLE;
    cube360ColorMemory = VK_NULL_HANDLE;
    for (auto& v : cube360FaceViews) v = VK_NULL_HANDLE;
    cube360CubeView = VK_NULL_HANDLE;
    solid360Sampler = VK_NULL_HANDLE;
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    for (auto &dv : cube360DepthViews) dv = VK_NULL_HANDLE;

    // Reset tracked layouts
    for (uint32_t face = 0; face < 6; ++face) {
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_UNDEFINED;
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void Solid360Renderer::renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                                     SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                                     SolidRenderer* solidRenderer,
                                     VkDescriptorSet mainDescriptorSet,
                                     Buffer& uniformBuffer, const UniformObject& ubo,
                                     VkBuffer compactIndirectBuffer, VkBuffer visibleCountBuffer) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (cube360FaceViews[0] == VK_NULL_HANDLE) return;

    glm::vec3 camPos = glm::vec3(ubo.viewPos);
    struct FaceInfo { glm::vec3 target; glm::vec3 up; };
    // Standard cubemap face order and orientation: +X, -X, +Y, -Y, +Z, -Z.
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

    for (uint32_t face = 0; face < 6; ++face) {
        glm::mat4 faceView = glm::lookAt(camPos, camPos + faces[face].target, faces[face].up);
        glm::mat4 faceVP = faceProj * faceView;

        UniformObject faceUBO = ubo;
        faceUBO.viewProjection = faceVP;

        vkCmdUpdateBuffer(cmd, uniformBuffer.buffer, 0, sizeof(UniformObject), &faceUBO);

        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        // Transition color layer: tracked layout → COLOR_ATTACHMENT_OPTIMAL
        {
            VkImageMemoryBarrier colorBarrier{};
            colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            colorBarrier.oldLayout = cube360ColorLayouts[face];
            colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.image = cube360ColorImage;
            colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, face, 1 };
            colorBarrier.srcAccessMask = (cube360ColorLayouts[face] == VK_IMAGE_LAYOUT_UNDEFINED)
                ? 0 : VK_ACCESS_SHADER_READ_BIT;
            colorBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, 0, nullptr, 0, nullptr, 1, &colorBarrier);
        }

        // Transition depth layer: tracked layout → DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        if (app) {
            app->recordTransitionImageLayoutLayer(cmd, cube360DepthImage, VK_FORMAT_D32_SFLOAT,
                                                 cube360DepthLayouts[face], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                                 1, face, 1);
        }

        VkClearValue clears[2];
        clears[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderingAttachmentInfo colorAtt{};
        colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAtt.imageView = cube360FaceViews[face];
        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.clearValue = clears[0];

        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView = cube360DepthViews[face];
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.clearValue = clears[1];

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = {0, 0};
        renderingInfo.renderArea.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAtt;
        renderingInfo.pDepthAttachment = &depthAtt;

        vkCmdBeginRendering(cmd, &renderingInfo);

        VkViewport viewport{0.0f, 0.0f, (float)CUBE360_FACE_SIZE, (float)CUBE360_FACE_SIZE, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

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

        if (solidRenderer) {
            VkPipeline gfxPipe = solidRenderer->getGraphicsPipeline();
            VkPipelineLayout gfxLayout = solidRenderer->getGraphicsPipelineLayout();
            if (gfxPipe != VK_NULL_HANDLE && gfxLayout != VK_NULL_HANDLE) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxLayout, 0, 1, &mainDescriptorSet, 0, nullptr);
                // If caller provided per-task compact/visible buffers, use them to draw the solid renderer's visible set.
                if (compactIndirectBuffer != VK_NULL_HANDLE && visibleCountBuffer != VK_NULL_HANDLE) {
                    solidRenderer->getIndirectRenderer().drawPreparedWithBuffers(cmd, compactIndirectBuffer, visibleCountBuffer);
                } else {
                    solidRenderer->getIndirectRenderer().drawAll(cmd);
                }
            }
        }

        vkCmdEndRendering(cmd);

        // Transition color: COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        {
            VkImageMemoryBarrier colorBarrier{};
            colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            colorBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.image = cube360ColorImage;
            colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, face, 1 };
            colorBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            colorBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &colorBarrier);
        }
        cube360ColorLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Transition depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        if (app) app->recordTransitionImageLayoutLayer(cmd, cube360DepthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, face, 1);
        cube360DepthLayouts[face] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Cubemap rendering complete; cubemap image view is available for sampling.
    vkCmdUpdateBuffer(cmd, uniformBuffer.buffer, 0, sizeof(UniformObject), &ubo);
    {
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
}
