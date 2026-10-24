#include "VideoRenderer.hpp"

#include "../RendererUtils.hpp"
#include "../../pipeline/ShaderStage.hpp"
#include "../../includes/shader/locations.hpp"
#include "../../includes/shader/vertex_layouts.hpp"
#include "../../../math/geometry/Vertex.hpp"
#include "../../../services/VideoTextureService.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

VideoRenderer::VideoRenderer() = default;
VideoRenderer::~VideoRenderer() = default;

void VideoRenderer::init(VulkanApp* app) {
    (void)app;
    indirectRenderer.init();
}

void VideoRenderer::cleanup(VulkanApp* app) {
    if (!app) return;
    destroyRenderTargets(app);
    indirectRenderer.cleanup(app);
}

void VideoRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app || width == 0 || height == 0) return;
    if (renderWidth == width && renderHeight == height && colorImages[0] != VK_NULL_HANDLE) return;
    destroyRenderTargets(app);
    renderWidth = width;
    renderHeight = height;

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(app->getDevice(), app, width, height, format, usage, aspect,
                                            "VideoRenderer: image", image, allocation, memory, view);
    };

    for (uint32_t i = 0; i < VIDEO_FRAMES; ++i) {
        createImage(app->getSwapchainImageFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    colorImages[i], colorAllocations[i], colorMemories[i], colorViews[i]);
        // The composite samples the color target: start in SHADER_READ_ONLY so
        // the first frame's barriers are consistent even before any draw.
        colorLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (colorImages[i] != VK_NULL_HANDLE) {
            app->transitionImageLayoutLayer(colorImages[i], app->getSwapchainImageFormat(),
                                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(colorImages[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            colorLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    depthImages[i], depthAllocations[i], depthMemories[i], depthViews[i]);
        depthLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (depthImages[i] != VK_NULL_HANDLE) {
            app->transitionImageLayoutLayer(depthImages[i], VK_FORMAT_D32_SFLOAT,
                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(depthImages[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            depthLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }
    std::cout << "[VideoRenderer] targets: " << width << "x" << height << " x " << VIDEO_FRAMES
              << " frames (color + depth)" << std::endl;
}

void VideoRenderer::destroyRenderTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < VIDEO_FRAMES; ++i) {
        if (colorViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(colorViews[i])) vkDestroyImageView(device, colorViews[i], nullptr);
            colorViews[i] = VK_NULL_HANDLE;
        }
        if (colorImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(colorImages[i], colorAllocations[i], colorMemories[i]);
            colorImages[i] = VK_NULL_HANDLE;
            colorAllocations[i] = VK_NULL_HANDLE;
            colorMemories[i] = VK_NULL_HANDLE;
        }
        if (depthViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(depthViews[i])) vkDestroyImageView(device, depthViews[i], nullptr);
            depthViews[i] = VK_NULL_HANDLE;
        }
        if (depthImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(depthImages[i], depthAllocations[i], depthMemories[i]);
            depthImages[i] = VK_NULL_HANDLE;
            depthAllocations[i] = VK_NULL_HANDLE;
            depthMemories[i] = VK_NULL_HANDLE;
        }
    }
    renderWidth = 0;
    renderHeight = 0;
}

void VideoRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    createRenderTargets(app, width, height);
}

bool VideoRenderer::createVideoDescriptorSet(VulkanApp* app) {
    if (!app || videoSet != VK_NULL_HANDLE) return videoSet != VK_NULL_HANDLE;

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    // set (1) is written while the layer is idle (init / array reload), not
    // every frame, so no update-after-bind feature is required.
    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr,
                                    &videoSetLayout.handle) != VK_SUCCESS) {
        std::cerr << "[VideoRenderer] failed to create the video array set layout\n";
        return false;
    }
    app->resources.addDescriptorSetLayout(videoSetLayout.handle, "VideoRenderer: video set layout");

    VkDescriptorPoolSize size{};
    size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    size.descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &size;
    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &videoSetPool.handle) != VK_SUCCESS) {
        std::cerr << "[VideoRenderer] failed to create the video array descriptor pool\n";
        return false;
    }
    app->resources.addDescriptorPool(videoSetPool.handle, "VideoRenderer: video set pool");

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = videoSetPool.handle;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &videoSetLayout.handle;
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &videoSet) != VK_SUCCESS) {
        std::cerr << "[VideoRenderer] failed to allocate the video array descriptor set\n";
        return false;
    }
    app->resources.addDescriptorSet(videoSet, "VideoRenderer: video set");

    // 1x1 black dummy layer: valid descriptor before the service is wired and
    // whenever no video array exists (VULKAN_NO_VIDEO).
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {1, 1, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        app->createImageWithVma(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dummyVideo.image,
                                dummyVideo.allocation, memory, "VideoRenderer: dummy layer");
        if (dummyVideo.image == VK_NULL_HANDLE) return false;
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = dummyVideo.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &dummyVideo.view) != VK_SUCCESS) return false;
        app->resources.addImageView(dummyVideo.view, "VideoRenderer: dummy view");
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &dummySampler) != VK_SUCCESS) return false;
        app->resources.addSampler(dummySampler, "VideoRenderer: dummy sampler");
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            app->recordTransitionImageLayoutLayer(cmd, dummyVideo.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
            VkClearColorValue black{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, dummyVideo.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            app->recordTransitionImageLayoutLayer(cmd, dummyVideo.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        });
        dummyLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Neutral derived dummy: flat normal, mid bump/roughness.
        VkImageCreateInfo derivedInfo = imageInfo;
        derivedInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        app->createImageWithVma(derivedInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dummyDerived.image,
                                dummyDerived.allocation, memory, "VideoRenderer: dummy derived layer");
        if (dummyDerived.image == VK_NULL_HANDLE) return false;
        VkImageViewCreateInfo derivedViewInfo = viewInfo;
        derivedViewInfo.image = dummyDerived.image;
        if (vkCreateImageView(app->getDevice(), &derivedViewInfo, nullptr, &dummyDerived.view) != VK_SUCCESS) return false;
        app->resources.addImageView(dummyDerived.view, "VideoRenderer: dummy derived view");
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            app->recordTransitionImageLayoutLayer(cmd, dummyDerived.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
            VkClearColorValue neutral{};
            neutral.float32[0] = 0.5f;
            neutral.float32[1] = 0.5f;
            neutral.float32[2] = 0.5f;
            neutral.float32[3] = 0.5f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, dummyDerived.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &neutral, 1, &range);
            app->recordTransitionImageLayoutLayer(cmd, dummyDerived.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        });
        writeVideoDescriptor(app, dummyVideo.view, dummySampler, dummyDerived.view, dummySampler);
    }
    return true;
}

void VideoRenderer::writeVideoDescriptor(VulkanApp* app, VkImageView albedoView, VkSampler albedoSampler,
                                        VkImageView derivedView, VkSampler derivedSampler) {
    if (!app || videoSet == VK_NULL_HANDLE || albedoView == VK_NULL_HANDLE || albedoSampler == VK_NULL_HANDLE ||
        derivedView == VK_NULL_HANDLE || derivedSampler == VK_NULL_HANDLE) {
        return;
    }
    VkDescriptorImageInfo albedoInfo{};
    albedoInfo.sampler = albedoSampler;
    albedoInfo.imageView = albedoView;
    albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo derivedInfo{};
    derivedInfo.sampler = derivedSampler;
    derivedInfo.imageView = derivedView;
    derivedInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = videoSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &albedoInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = videoSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &derivedInfo;
    vkUpdateDescriptorSets(app->getDevice(), 2, writes, 0, nullptr);
    boundVideoView = albedoView;
    boundVideoSampler = albedoSampler;
    boundDerivedView = derivedView;
    boundDerivedSampler = derivedSampler;
}

void VideoRenderer::setVideoTexture(VulkanApp* app, VideoTextureService* service) {
    if (!app || !createVideoDescriptorSet(app)) return;
    if (!service) return;
    const VkImageView view = service->videoArrayView();
    const VkSampler sampler = service->videoArraySampler();
    const VkImageView derivedView = service->videoDerivedArrayView();
    const VkSampler derivedSampler = service->videoDerivedArraySampler();
    if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE ||
        derivedView == VK_NULL_HANDLE || derivedSampler == VK_NULL_HANDLE) {
        return;
    }
    if (view == boundVideoView && sampler == boundVideoSampler &&
        derivedView == boundDerivedView && derivedSampler == boundDerivedSampler) {
        return;
    }
    writeVideoDescriptor(app, view, sampler, derivedView, derivedSampler);
    std::cout << "[VideoRenderer] video array bound (" << service->slotCapacity() << " slots, "
              << service->videoArrayWidth() << "^2)" << std::endl;
}

void VideoRenderer::createPipelines(VulkanApp* app) {
    if (!app) return;
    if (!createVideoDescriptorSet(app)) return;

    ShaderStage vertexShader(app->getOrCreateShaderModule("shaders/renderer/solid/SolidRendererNoTess.vert.spv"),
                             VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragmentShader(app->getOrCreateShaderModule("shaders/renderer/video/VideoRenderer.frag.spv"),
                               VK_SHADER_STAGE_FRAGMENT_BIT);
    if (!vertexShader.info.module || !fragmentShader.info.module) {
        std::cerr << "[VideoRenderer] shader modules missing (run `make shaders`)\n";
        return;
    }

    std::vector<VkDescriptorSetLayout> setLayouts = {app->getDescriptorSetLayout(), videoSetLayout.handle};
    GraphicsPipelineConfig cfg{};
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.cullMode = VK_CULL_MODE_NONE;
    cfg.colorFormats = {app->getSwapchainImageFormat()};
    cfg.depthFormat = VK_FORMAT_D32_SFLOAT;

    auto [created, layout] = app->createGraphicsPipeline(
        {vertexShader.info, fragmentShader.info},
        std::vector<VkVertexInputBindingDescription>{VkVertexInputBindingDescription{0, sizeof(Vertex),
                                                                                    VK_VERTEX_INPUT_RATE_VERTEX}},
        // The non-tessellated solid VS prunes ATTR_COLOR; declaring it here
        // would raise the "vertex attribute not consumed" performance warning.
        vk_layouts::defaultAttributesFiltered(
            {ATTR_POS, ATTR_UV, ATTR_NORMAL, ATTR_BRUSH_INDEX, ATTR_TEXTURE_TYPE, ATTR_HSV}),
        setLayouts,
        nullptr,
        cfg);
    pipeline = created;
    pipelineLayout = layout;
}

bool VideoRenderer::beginPass(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex) {
    if (!app || cmd == VK_NULL_HANDLE || pipeline.handle == VK_NULL_HANDLE) return false;
    if (frameIndex >= VIDEO_FRAMES || colorImages[frameIndex] == VK_NULL_HANDLE) return false;

    // Offscreen target transitions (tracked per frame slot).
    app->recordTransitionImageLayoutLayer(cmd, colorImages[frameIndex], app->getSwapchainImageFormat(),
                                          colorLayouts[frameIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                          1, 0, 1);
    app->recordTransitionImageLayoutLayer(cmd, depthImages[frameIndex], VK_FORMAT_D32_SFLOAT,
                                          depthLayouts[frameIndex],
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
    colorLayouts[frameIndex] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    depthLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = colorViews[frameIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // Alpha 0 = "no video": the composite leaves the scene color untouched
    // wherever the video layer wrote nothing.
    colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthViews[frameIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = {{0, 0}, {renderWidth, renderHeight}};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(cmd, &renderingInfo);

    VkViewport viewport{0.0f, 0.0f, static_cast<float>(renderWidth), static_cast<float>(renderHeight), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {renderWidth, renderHeight}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
    return true;
}

void VideoRenderer::endPass(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex) {
    vkCmdEndRendering(cmd);
    // The post-process composite samples both targets next.
    app->recordTransitionImageLayoutLayer(cmd, colorImages[frameIndex], app->getSwapchainImageFormat(),
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    app->recordTransitionImageLayoutLayer(cmd, depthImages[frameIndex], VK_FORMAT_D32_SFLOAT,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    colorLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void VideoRenderer::draw(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex,
                         VkDescriptorSet sceneSet, const glm::mat4& viewProjection,
                         const glm::vec3& cameraPos, float lodBias, int maxTargetLod) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    // The layer's own cull pass writes the compact draw commands and applies the
    // buffer barriers; the video geometry is small so this is a single dispatch.
    indirectRenderer.setCullFrame(frameIndex);
    indirectRenderer.prepareCull(cmd, viewProjection, cameraPos, lodBias, maxTargetLod,
                                 nullptr, false, true, false, 0, 1);

    if (!beginPass(cmd, app, frameIndex)) return;
    if (sceneSet != VK_NULL_HANDLE && videoSet != VK_NULL_HANDLE) {
        VkDescriptorSet sets[2] = {sceneSet, videoSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout.handle, 0, 2, sets, 0, nullptr);
    }
    if (indirectRenderer.getMeshCount() > 0) {
        indirectRenderer.drawPrepared(cmd);
    }
    endPass(cmd, app, frameIndex);
}

