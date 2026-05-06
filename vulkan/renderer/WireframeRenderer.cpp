#include "WireframeRenderer.hpp"
#include "../../utils/FileReader.hpp"
#include <iostream>
#include <array>

void WireframeRenderer::createPipeline(VulkanApp* app,
                                       const std::vector<VkFormat>& colorFormats,
                                       const std::vector<VkDescriptorSetLayout>& setLayouts,
                                       const char* vertPath,
                                       const char* fragPath,
                                       const char* tescPath,
                                       const char* tesePath,
                                       const char* label) {
    if (!app) return;
    VkDevice device = app->getDevice();
    uint32_t colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());

    // Load shaders
    auto vertCode = FileReader::readFile(vertPath);
    auto fragCode = FileReader::readFile(fragPath);
    if (vertCode.empty() || fragCode.empty()) {
        std::cerr << "[WireframeRenderer] Warning: Could not load shaders for " << label << std::endl;
        return;
    }
    VkShaderModule vertModule = app->createShaderModule(vertCode);
    VkShaderModule fragModule = app->createShaderModule(fragCode);

    VkShaderModule tescModule = VK_NULL_HANDLE;
    VkShaderModule teseModule = VK_NULL_HANDLE;
    bool hasTessellation = false;
    if (tescPath && tesePath) {
        auto tescCode = FileReader::readFile(tescPath);
        auto teseCode = FileReader::readFile(tesePath);
        if (!tescCode.empty() && !teseCode.empty()) {
            tescModule = app->createShaderModule(tescCode);
            teseModule = app->createShaderModule(teseCode);
            hasTessellation = true;
        }
    }

    // Shader stages
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vertModule;
        stage.pName = "main";
        shaderStages.push_back(stage);
    }
    if (hasTessellation) {
        VkPipelineShaderStageCreateInfo tesc{};
        tesc.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tesc.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        tesc.module = tescModule;
        tesc.pName = "main";
        shaderStages.push_back(tesc);

        VkPipelineShaderStageCreateInfo tese{};
        tese.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tese.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        tese.module = teseModule;
        tese.pName = "main";
        shaderStages.push_back(tese);
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage.module = fragModule;
        stage.pName = "main";
        shaderStages.push_back(stage);
    }

    // Vertex input (standard Vertex layout)
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 5> attrDescs{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
    attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)};
    attrDescs[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attrDescs[4] = {5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex)};

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
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;  // WIREFRAME
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
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Color blend attachments — one per color attachment in the render pass
    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments(colorAttachmentCount);
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

    // Pipeline layout
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    layoutInfo.pSetLayouts = setLayouts.data();
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &wireframePipelineLayout) != VK_SUCCESS) {
        std::cerr << "[WireframeRenderer] Failed to create pipeline layout for " << label << std::endl;
        return;
    }
    app->resources.addPipelineLayout(wireframePipelineLayout, (std::string("WireframeRenderer: ") + label + " layout").c_str());

    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
    pipelineRenderingInfo.pColorAttachmentFormats = colorFormats.empty() ? nullptr : colorFormats.data();
    pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &pipelineRenderingInfo;
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
    pipelineInfo.layout = wireframePipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;
    if (hasTessellation) pipelineInfo.pTessellationState = &tessState;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &wireframePipeline) != VK_SUCCESS) {
        std::cerr << "[WireframeRenderer] Failed to create wireframe pipeline for " << label << std::endl;
        wireframePipeline = VK_NULL_HANDLE;
    } else {
        app->resources.addPipeline(wireframePipeline, (std::string("WireframeRenderer: ") + label).c_str());
        std::cout << "[WireframeRenderer] Created wireframe pipeline: " << label << std::endl;
    }
}

void WireframeRenderer::draw(VkCommandBuffer cmd,
                             VulkanApp* app,
                             const std::vector<VkDescriptorSet>& descriptorSets,
                             IndirectRenderer& indirectRenderer) {
    if (wireframePipeline == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE || !app) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframePipeline);

    // Set dynamic viewport/scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(app->getWidth());
    viewport.height = static_cast<float>(app->getHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(app->getWidth()), static_cast<uint32_t>(app->getHeight())};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind all descriptor sets consecutively starting at set 0
    if (!descriptorSets.empty()) {
        //printf("[BIND] WireframeRenderer::draw: layout=%p firstSet=0 count=%u sets=", (void*)wireframePipelineLayout, (unsigned)descriptorSets.size());
        for (size_t i = 0; i < descriptorSets.size(); ++i) printf("%p ", (void*)descriptorSets[i]);
        printf("\n");
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            wireframePipelineLayout, 0,
            static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
            0, nullptr);
    }

    indirectRenderer.drawPrepared(cmd);
}

void WireframeRenderer::cleanup() {
    wireframePipeline = VK_NULL_HANDLE;
    wireframePipelineLayout = VK_NULL_HANDLE;
}
