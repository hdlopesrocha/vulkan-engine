#include "BrushBackFaceRenderer.hpp"
#include "RendererUtils.hpp"
#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"
#include "../ShaderStage.hpp"

BrushBackFaceRenderer::BrushBackFaceRenderer() {}
BrushBackFaceRenderer::~BrushBackFaceRenderer() {}

void BrushBackFaceRenderer::init(VulkanApp* app) {
    appPtr = app;
}

void BrushBackFaceRenderer::cleanup(VulkanApp* app) {
    (void)app;
    appPtr = nullptr;
}

void BrushBackFaceRenderer::createPipelines(VulkanApp* app) {
    if (!app) return;
    if (backFacePipeline != VK_NULL_HANDLE) return;
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

    ShaderStage depthFrag = ShaderStage(
        app->getOrCreateShaderModule("shaders/depth_only.frag.spv"),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    // Descriptor set layouts: only set 0 (main UBO/textures)
    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getDescriptorSetLayout());

    GraphicsPipelineConfig cfg{};
    cfg.cullMode = VK_CULL_MODE_FRONT_BIT;
    cfg.depthCompareOp = VK_COMPARE_OP_GREATER;
    cfg.noColorAttachment = true;
    cfg.colorWrite = false;

    auto [pipeline, layout] = app->createGraphicsPipeline(
        { vertexShader.info, tescShader.info, teseShader.info, depthFrag.info },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        vk_layouts::defaultAttributes(),
        setLayouts,
        nullptr,
        cfg
    );
    backFacePipeline = pipeline;
    pipelineLayout = layout;

    if (backFacePipeline != VK_NULL_HANDLE) {
        app->resources.addPipeline(backFacePipeline, "BrushBackFaceRenderer: backFacePipeline");
        std::cout << "[BrushBackFaceRenderer] Created back-face depth pipeline (GREATER compare)" << std::endl;
    }

    // Clear local shader module references
    depthFrag.info.module = VK_NULL_HANDLE;
    teseShader.info.module = VK_NULL_HANDLE;
    tescShader.info.module = VK_NULL_HANDLE;
    vertexShader.info.module = VK_NULL_HANDLE;
}

void BrushBackFaceRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    appPtr = app;
    if (renderWidth == width && renderHeight == height && backFaceDepthImages[0] != VK_NULL_HANDLE) return;
    destroyRenderTargets(app);
    renderWidth = width;
    renderHeight = height;
    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "BrushBackFaceRenderer: image", image, allocation, memory, view);
    };

    for (uint32_t i = 0; i < FRAMES; ++i) {
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    backFaceDepthImages[i], backFaceDepthAllocations[i], backFaceDepthMemories[i], backFaceDepthImageViews[i]);
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (backFaceDepthImages[i] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(backFaceDepthImages[i], VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(backFaceDepthImages[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }
}

void BrushBackFaceRenderer::destroyRenderTargets(VulkanApp* app) {
    (void)app;
    for (uint32_t i = 0; i < FRAMES; ++i) {
        backFaceDepthImages[i] = VK_NULL_HANDLE;
        backFaceDepthAllocations[i] = VK_NULL_HANDLE;
        backFaceDepthMemories[i] = VK_NULL_HANDLE;
        backFaceDepthImageViews[i] = VK_NULL_HANDLE;
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void BrushBackFaceRenderer::renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                               IndirectRenderer& indirect,
                                               VkDescriptorSet mainDs,
                                               VkBuffer compactIndirectBuffer, VkBuffer visibleCountBuffer) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (backFacePipeline == VK_NULL_HANDLE) return;
    if (frameIndex >= backFaceDepthImages.size()) return;
    if (backFaceDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    // Transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL if needed
    if (backFaceDepthImageLayouts[frameIndex] != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        app->recordTransitionImageLayoutLayer(cmd, backFaceDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             1, 0, 1);
        backFaceDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    // Clear to 0.0 (near) since GREATER comparison means farthest depth wins
    VkClearValue bfClear{};
    bfClear.depthStencil = {0.0f, 0};

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = backFaceDepthImageViews[frameIndex];
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue = bfClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {renderWidth, renderHeight};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pDepthAttachment = &depthAtt;

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

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, backFacePipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, backFacePipeline);

    // Bind descriptor set 0: main UBO/textures
    if (mainDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 1, &mainDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &mainDs, 0, nullptr);
    }

    if (compactIndirectBuffer != VK_NULL_HANDLE && visibleCountBuffer != VK_NULL_HANDLE) {
        indirect.drawPreparedWithBuffers(cmd, compactIndirectBuffer, visibleCountBuffer);
    } else {
        indirect.drawPrepared(cmd);
    }

    vkCmdEndRendering(cmd);

    // Transition depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    if (app) {
        app->recordTransitionImageLayoutLayer(cmd, backFaceDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, 0, 1);
    }
    backFaceDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
