#include "WaterBackFaceRenderer.hpp"
#include "RendererUtils.hpp"
#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

WaterBackFaceRenderer::WaterBackFaceRenderer() {}
WaterBackFaceRenderer::~WaterBackFaceRenderer() {}

void WaterBackFaceRenderer::init(VulkanApp* app) {
    appPtr = app;
}

void WaterBackFaceRenderer::cleanup(VulkanApp* app) {
    (void)app;
    appPtr = nullptr;
    // VulkanResourceManager owns actual Vulkan objects; just clear handles
    backFacePipeline = VK_NULL_HANDLE;
    for (auto &v : backFaceDepthImages) v = VK_NULL_HANDLE;
    for (auto &m : backFaceDepthMemories) m = VK_NULL_HANDLE;
    for (auto &v : backFaceDepthImageViews) v = VK_NULL_HANDLE;
}

void WaterBackFaceRenderer::createPipelines(VulkanApp* app, VkPipelineLayout pipelineLayout) {
    if (!app || pipelineLayout == VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    VkShaderModule bfVert = app->getOrCreateShaderModule("shaders/water.vert.spv");
    VkShaderModule bfFrag = app->getOrCreateShaderModule("shaders/water_backface.frag.spv");
    VkShaderModule bfTesc = app->getOrCreateShaderModule("shaders/water.tesc.spv");
    VkShaderModule bfTese = app->getOrCreateShaderModule("shaders/water.tese.spv");

    std::vector<VkPipelineShaderStageCreateInfo> bfStages;
    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = bfVert;
    vs.pName = "main";
    bfStages.push_back(vs);

    { // tessellation stages (always available — getOrCreateShaderModule would throw if missing)
        VkPipelineShaderStageCreateInfo tc{};
        tc.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tc.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        tc.module = bfTesc;
        tc.pName = "main";
        bfStages.push_back(tc);

        VkPipelineShaderStageCreateInfo te{};
        te.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        te.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        te.module = bfTese;
        te.pName = "main";
        bfStages.push_back(te);
    }

    VkPipelineShaderStageCreateInfo fs{};
    fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = bfFrag;
    fs.pName = "main";
    bfStages.push_back(fs);

    // Vertex input
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    auto attrDescs = vk_layouts::defaultAttributes();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
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

    VkPipelineRasterizationStateCreateInfo bfRasterizer{};
    bfRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    bfRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    bfRasterizer.lineWidth = 1.0f;
    bfRasterizer.cullMode = VK_CULL_MODE_FRONT_BIT; // cull front faces → render back faces
    bfRasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo bfBlend{};
    bfBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    bfBlend.attachmentCount = 0;
    bfBlend.pAttachments = nullptr;

    VkPipelineTessellationStateCreateInfo tessState{};
    tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tessState.patchControlPoints = 3;

    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = 0;
    pipelineRenderingInfo.pColorAttachmentFormats = nullptr;
    pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo bfPipeInfo{};
    bfPipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    bfPipeInfo.pNext = &pipelineRenderingInfo;
    bfPipeInfo.stageCount = static_cast<uint32_t>(bfStages.size());
    bfPipeInfo.pStages = bfStages.data();
    bfPipeInfo.pVertexInputState = &vertexInputInfo;
    bfPipeInfo.pInputAssemblyState = &inputAssembly;
    bfPipeInfo.pViewportState = &viewportState;
    bfPipeInfo.pDynamicState = &dynamicState;
    bfPipeInfo.pRasterizationState = &bfRasterizer;
    bfPipeInfo.pMultisampleState = &multisampling;
    bfPipeInfo.pDepthStencilState = &depthStencil;
    bfPipeInfo.pColorBlendState = &bfBlend;
    bfPipeInfo.layout = pipelineLayout;
    bfPipeInfo.renderPass = VK_NULL_HANDLE;
    bfPipeInfo.subpass = 0;
    bfPipeInfo.pTessellationState = &tessState;

    if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &bfPipeInfo, nullptr, &backFacePipeline) != VK_SUCCESS) {
        std::cerr << "[WaterBackFaceRenderer] Warning: Failed to create back-face depth pipeline" << std::endl;
        backFacePipeline = VK_NULL_HANDLE;
    } else {
        app->resources.addPipeline(backFacePipeline, "WaterBackFaceRenderer: backFacePipeline");
        std::cout << "[WaterBackFaceRenderer] Created back-face depth pipeline" << std::endl;
    }
}

void WaterBackFaceRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    if (renderWidth == width && renderHeight == height && backFaceDepthImages[0] != VK_NULL_HANDLE) return;
    destroyRenderTargets(app);
    renderWidth = width;
    renderHeight = height;
    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "WaterBackFaceRenderer: image", image, allocation, memory, view);
    };

    for (int i = 0; i < 3; ++i) {
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    backFaceDepthImages[i], backFaceDepthAllocations[i], backFaceDepthMemories[i], backFaceDepthImageViews[i]);
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        // Ensure authoritative/tracked layout matches the render-pass initial layout
        if (backFaceDepthImages[i] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(backFaceDepthImages[i], VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(backFaceDepthImages[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }

    for (int i = 0; i < 3; ++i) {
        // Keep the layout we recorded above (do not reset to UNDEFINED).
        // Resetting here caused later callers to emit fallback transitions
        // that disagreed with the app-tracked authoritative layout.
    }

    // Create framebuffers removed - using dynamic rendering
}

void WaterBackFaceRenderer::destroyRenderTargets(VulkanApp* app) {
    (void)app;
    for (int i = 0; i < 3; ++i) {
        backFaceDepthImages[i] = VK_NULL_HANDLE;
        backFaceDepthAllocations[i] = VK_NULL_HANDLE;
        backFaceDepthMemories[i] = VK_NULL_HANDLE;
        backFaceDepthImageViews[i] = VK_NULL_HANDLE;
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void WaterBackFaceRenderer::renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                              IndirectRenderer& indirect, VkPipelineLayout pipelineLayout,
                                              VkDescriptorSet mainDs, VkDescriptorSet materialDs, VkDescriptorSet sceneDs,
                                              VkImage sceneDepthImage,
                                              VkBuffer compactIndirectBuffer, VkBuffer visibleCountBuffer) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (backFacePipeline == VK_NULL_HANDLE) return;
    if (frameIndex >= backFaceDepthImages.size()) return;
    if (backFaceDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    // No depth copy needed — the back-face shader samples the scene depth
    // texture (in SHADER_READ_ONLY_OPTIMAL) and clips fragments behind the
    // scene. The back-face depth attachment uses LOAD_OP_CLEAR (1.0).
    if (backFaceDepthImageLayouts[frameIndex] != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        if (!app) {
            throw std::runtime_error("WaterBackFaceRenderer::renderBackFacePass requires VulkanApp (no fallback allowed)");
        }
        app->recordTransitionImageLayoutLayer(cmd, backFaceDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             1, 0, 1);
        backFaceDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkClearValue bfClear{};
    bfClear.depthStencil = {1.0f, 0};

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

    if (mainDs != VK_NULL_HANDLE) {
        //printf("[BIND] WaterBackFaceRenderer::renderBackFacePass: layout=%p firstSet=0 count=1 sets=%p\n", (void*)pipelineLayout, (void*)mainDs);
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 1, &mainDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &mainDs, 0, nullptr);
    }
    if (materialDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 1, 1, &materialDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &materialDs, 0, nullptr);
    }
    if (sceneDs != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 2, 1, &sceneDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1, &sceneDs, 0, nullptr);
    }

    if (compactIndirectBuffer != VK_NULL_HANDLE && visibleCountBuffer != VK_NULL_HANDLE) {
        indirect.drawPreparedWithBuffers(cmd, compactIndirectBuffer, visibleCountBuffer);
    } else {
        indirect.drawPrepared(cmd);
    }

    vkCmdEndRendering(cmd);

    // Transition depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    if (app) {
        app->recordTransitionImageLayoutLayer(cmd, backFaceDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, 0, 1);
    }
    backFaceDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void WaterBackFaceRenderer::postRenderBarrier(VkCommandBuffer cmd, uint32_t frameIndex) {
    // Transition already performed inside renderBackFacePass; this is a no-op.
    (void)cmd;
    (void)frameIndex;
}

// getBackFaceDepthLayout is defined inline in the header (maps indices via modulo).
