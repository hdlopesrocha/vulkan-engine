#include "WaterBackFaceRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>

WaterBackFaceRenderer::WaterBackFaceRenderer() {}
WaterBackFaceRenderer::~WaterBackFaceRenderer() {}

void WaterBackFaceRenderer::init(VulkanApp* app) {
    (void)app;
}

void WaterBackFaceRenderer::cleanup(VulkanApp* app) {
    (void)app;
    // VulkanResourceManager owns actual Vulkan objects; just clear handles
    backFaceRenderPass = VK_NULL_HANDLE;
    backFacePipeline = VK_NULL_HANDLE;
    for (auto &v : backFaceDepthImages) v = VK_NULL_HANDLE;
    for (auto &m : backFaceDepthMemories) m = VK_NULL_HANDLE;
    for (auto &v : backFaceDepthImageViews) v = VK_NULL_HANDLE;
    for (auto &fb : backFaceFramebuffers) fb = VK_NULL_HANDLE;
}

void WaterBackFaceRenderer::createRenderPass(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();

    VkAttachmentDescription bfDepthAtt{};
    bfDepthAtt.format = VK_FORMAT_D32_SFLOAT;
    bfDepthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    bfDepthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    bfDepthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    bfDepthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    bfDepthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    bfDepthAtt.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    bfDepthAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference bfDepthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription bfSubpass{};
    bfSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    bfSubpass.colorAttachmentCount = 0;
    bfSubpass.pColorAttachments = nullptr;
    bfSubpass.pDepthStencilAttachment = &bfDepthRef;

    std::array<VkSubpassDependency, 2> bfDeps{};
    bfDeps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    bfDeps[0].dstSubpass = 0;
    bfDeps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    bfDeps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bfDeps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    bfDeps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    bfDeps[1].srcSubpass = 0;
    bfDeps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    bfDeps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    bfDeps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    bfDeps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    bfDeps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo bfRPInfo{};
    bfRPInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    bfRPInfo.attachmentCount = 1;
    bfRPInfo.pAttachments = &bfDepthAtt;
    bfRPInfo.subpassCount = 1;
    bfRPInfo.pSubpasses = &bfSubpass;
    bfRPInfo.dependencyCount = static_cast<uint32_t>(bfDeps.size());
    bfRPInfo.pDependencies = bfDeps.data();

    if (vkCreateRenderPass(device, &bfRPInfo, nullptr, &backFaceRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create back-face depth render pass!");
    }
    app->resources.addRenderPass(backFaceRenderPass, "WaterBackFaceRenderer: backFaceRenderPass");
}

void WaterBackFaceRenderer::createPipelines(VulkanApp* app, VkPipelineLayout pipelineLayout) {
    if (!app || pipelineLayout == VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    auto bfVertCode = FileReader::readFile("shaders/water.vert.spv");
    auto bfTescCode = FileReader::readFile("shaders/water.tesc.spv");
    auto bfTeseCode = FileReader::readFile("shaders/water.tese.spv");
    auto bfFragCode = FileReader::readFile("shaders/water_backface.frag.spv");

    if (bfVertCode.empty() || bfFragCode.empty()) {
        std::cerr << "[WaterBackFaceRenderer] Warning: missing backface shaders" << std::endl;
        return;
    }

    VkShaderModule bfVert = app->createShaderModule(bfVertCode);
    VkShaderModule bfFrag = app->createShaderModule(bfFragCode);
    VkShaderModule bfTesc = VK_NULL_HANDLE;
    VkShaderModule bfTese = VK_NULL_HANDLE;

    bool bfHasTess = !bfTescCode.empty() && !bfTeseCode.empty();
    if (bfHasTess) {
        bfTesc = app->createShaderModule(bfTescCode);
        bfTese = app->createShaderModule(bfTeseCode);
    }

    std::vector<VkPipelineShaderStageCreateInfo> bfStages;
    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = bfVert;
    vs.pName = "main";
    bfStages.push_back(vs);

    if (bfHasTess) {
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
    inputAssembly.topology = bfHasTess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
    if (bfHasTess) {
        tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3;
    }

    VkGraphicsPipelineCreateInfo bfPipeInfo{};
    bfPipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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
    bfPipeInfo.renderPass = backFaceRenderPass;
    bfPipeInfo.subpass = 0;
    if (bfHasTess) bfPipeInfo.pTessellationState = &tessState;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &bfPipeInfo, nullptr, &backFacePipeline) != VK_SUCCESS) {
        std::cerr << "[WaterBackFaceRenderer] Warning: Failed to create back-face depth pipeline" << std::endl;
        backFacePipeline = VK_NULL_HANDLE;
    } else {
        app->resources.addPipeline(backFacePipeline, "WaterBackFaceRenderer: backFacePipeline");
        std::cout << "[WaterBackFaceRenderer] Created back-face depth pipeline" << std::endl;
    }
}

void WaterBackFaceRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    if (renderWidth == width && renderHeight == height && backFaceFramebuffers[0] != VK_NULL_HANDLE) return;
    destroyRenderTargets(app);
    renderWidth = width;
    renderHeight = height;
    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create back-face image!");
        }
        app->resources.addImage(image, "WaterBackFaceRenderer: image");
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate back-face image memory!");
        }
        vkBindImageMemory(device, image, memory, 0);
        app->resources.addDeviceMemory(memory, "WaterBackFaceRenderer: memory");

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create back-face image view!");
        }
        app->resources.addImageView(view, "WaterBackFaceRenderer: view");
    };

    for (int i = 0; i < 2; ++i) {
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    backFaceDepthImages[i], backFaceDepthMemories[i], backFaceDepthImageViews[i]);
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // Transition images to SHADER_READ_ONLY_OPTIMAL
    for (int i = 0; i < 2; ++i) {
        if (backFaceDepthImages[i] == VK_NULL_HANDLE) continue;
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = backFaceDepthImageLayouts[i];
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = backFaceDepthImages[i];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        });
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Create framebuffers
    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        VkImageView bfAttachments[1] = { backFaceDepthImageViews[frameIdx] };
        VkFramebufferCreateInfo bfFbInfo{};
        bfFbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        bfFbInfo.renderPass = backFaceRenderPass;
        bfFbInfo.attachmentCount = 1;
        bfFbInfo.pAttachments = bfAttachments;
        bfFbInfo.width = width;
        bfFbInfo.height = height;
        bfFbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &bfFbInfo, nullptr, &backFaceFramebuffers[frameIdx]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create back-face depth framebuffer!");
        }
        app->resources.addFramebuffer(backFaceFramebuffers[frameIdx], "WaterBackFaceRenderer: backFaceFramebuffer");
    }
}

void WaterBackFaceRenderer::destroyRenderTargets(VulkanApp* app) {
    (void)app;
    for (int i = 0; i < 2; ++i) {
        backFaceFramebuffers[i] = VK_NULL_HANDLE;
        backFaceDepthImages[i] = VK_NULL_HANDLE;
        backFaceDepthMemories[i] = VK_NULL_HANDLE;
        backFaceDepthImageViews[i] = VK_NULL_HANDLE;
        backFaceDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void WaterBackFaceRenderer::renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                              IndirectRenderer& indirect, VkPipelineLayout pipelineLayout,
                                              VkDescriptorSet mainDs, VkDescriptorSet materialDs, VkDescriptorSet sceneDs,
                                              VkImage sceneDepthImage) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (backFaceRenderPass == VK_NULL_HANDLE || backFacePipeline == VK_NULL_HANDLE) return;
    if (frameIndex >= backFaceFramebuffers.size()) return;
    if (backFaceFramebuffers[frameIndex] == VK_NULL_HANDLE) return;

    VkClearValue bfClear{};
    bfClear.depthStencil = {1.0f, 0};

    if (sceneDepthImage != VK_NULL_HANDLE) {
        // Initialize the back-face depth buffer with scene depth so the back-face
        // render is occluded by solid geometry and only visible water volume is captured.
        VkImageMemoryBarrier copyBarriers[2]{};

        copyBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        copyBarriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        copyBarriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        copyBarriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        copyBarriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarriers[0].image = sceneDepthImage;
        copyBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        copyBarriers[0].subresourceRange.baseMipLevel = 0;
        copyBarriers[0].subresourceRange.levelCount = 1;
        copyBarriers[0].subresourceRange.baseArrayLayer = 0;
        copyBarriers[0].subresourceRange.layerCount = 1;

        copyBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        copyBarriers[1].srcAccessMask = (backFaceDepthImageLayouts[frameIndex] == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            ? VK_ACCESS_SHADER_READ_BIT
            : 0;
        copyBarriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copyBarriers[1].oldLayout = backFaceDepthImageLayouts[frameIndex];
        copyBarriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copyBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarriers[1].image = backFaceDepthImages[frameIndex];
        copyBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        copyBarriers[1].subresourceRange.baseMipLevel = 0;
        copyBarriers[1].subresourceRange.levelCount = 1;
        copyBarriers[1].subresourceRange.baseArrayLayer = 0;
        copyBarriers[1].subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 2, copyBarriers);

        VkImageCopy copyRegion{};
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        copyRegion.srcSubresource.baseArrayLayer = 0;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.srcSubresource.mipLevel = 0;
        copyRegion.srcOffset = {0, 0, 0};
        copyRegion.dstSubresource = copyRegion.srcSubresource;
        copyRegion.dstOffset = {0, 0, 0};
        copyRegion.extent = {renderWidth, renderHeight, 1};

        vkCmdCopyImage(cmd,
            sceneDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            backFaceDepthImages[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);

        VkImageMemoryBarrier loadBarrier{};
        loadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        loadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        loadBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        loadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        loadBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        loadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        loadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        loadBarrier.image = backFaceDepthImages[frameIndex];
        loadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        loadBarrier.subresourceRange.baseMipLevel = 0;
        loadBarrier.subresourceRange.levelCount = 1;
        loadBarrier.subresourceRange.baseArrayLayer = 0;
        loadBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &loadBarrier);

        backFaceDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkImageMemoryBarrier sceneRestoreBarrier{};
        sceneRestoreBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sceneRestoreBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sceneRestoreBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sceneRestoreBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sceneRestoreBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sceneRestoreBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneRestoreBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneRestoreBarrier.image = sceneDepthImage;
        sceneRestoreBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        sceneRestoreBarrier.subresourceRange.baseMipLevel = 0;
        sceneRestoreBarrier.subresourceRange.levelCount = 1;
        sceneRestoreBarrier.subresourceRange.baseArrayLayer = 0;
        sceneRestoreBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &sceneRestoreBarrier);
    } else if (backFaceDepthImageLayouts[frameIndex] != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        VkImageMemoryBarrier fallbackBarrier{};
        fallbackBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        fallbackBarrier.srcAccessMask = 0;
        fallbackBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        fallbackBarrier.oldLayout = backFaceDepthImageLayouts[frameIndex];
        fallbackBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        fallbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fallbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fallbackBarrier.image = backFaceDepthImages[frameIndex];
        fallbackBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        fallbackBarrier.subresourceRange.baseMipLevel = 0;
        fallbackBarrier.subresourceRange.levelCount = 1;
        fallbackBarrier.subresourceRange.baseArrayLayer = 0;
        fallbackBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &fallbackBarrier);
        backFaceDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkRenderPassBeginInfo bfRPInfo{};
    bfRPInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    bfRPInfo.renderPass = backFaceRenderPass;
    bfRPInfo.framebuffer = backFaceFramebuffers[frameIndex];
    bfRPInfo.renderArea.offset = {0, 0};
    bfRPInfo.renderArea.extent = {renderWidth, renderHeight};
    bfRPInfo.clearValueCount = 1;
    bfRPInfo.pClearValues = &bfClear;
    vkCmdBeginRenderPass(cmd, &bfRPInfo, VK_SUBPASS_CONTENTS_INLINE);

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

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, backFacePipeline);

    if (mainDs != VK_NULL_HANDLE) {
        //printf("[BIND] WaterBackFaceRenderer::renderBackFacePass: layout=%p firstSet=0 count=1 sets=%p\n", (void*)pipelineLayout, (void*)mainDs);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &mainDs, 0, nullptr);
    }
    if (materialDs != VK_NULL_HANDLE) {
        //printf("[BIND] WaterBackFaceRenderer::renderBackFacePass: layout=%p firstSet=1 count=1 sets=%p\n", (void*)pipelineLayout, (void*)materialDs);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &materialDs, 0, nullptr);
    }
    if (sceneDs != VK_NULL_HANDLE) {
        //printf("[BIND] WaterBackFaceRenderer::renderBackFacePass: layout=%p firstSet=2 count=1 sets=%p\n", (void*)pipelineLayout, (void*)sceneDs);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1, &sceneDs, 0, nullptr);
    }

    indirect.drawPrepared(cmd);

    vkCmdEndRenderPass(cmd);

    // Barrier to make depth writes visible to shader reads
    if (backFaceDepthImages[frameIndex] != VK_NULL_HANDLE) {
        VkImageMemoryBarrier bfBarrier{};
        bfBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bfBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        bfBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bfBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bfBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bfBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bfBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bfBarrier.image = backFaceDepthImages[frameIndex];
        bfBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        bfBarrier.subresourceRange.baseMipLevel = 0;
        bfBarrier.subresourceRange.levelCount = 1;
        bfBarrier.subresourceRange.baseArrayLayer = 0;
        bfBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &bfBarrier);
    }
}

void WaterBackFaceRenderer::postRenderBarrier(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (cmd == VK_NULL_HANDLE) return;
    if (frameIndex >= backFaceDepthImages.size()) return;
    if (backFaceDepthImages[frameIndex] == VK_NULL_HANDLE) return;

    VkImageMemoryBarrier db{};
    db.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    db.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    db.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    db.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    db.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    db.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.image = backFaceDepthImages[frameIndex];
    db.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    db.subresourceRange.baseMipLevel = 0;
    db.subresourceRange.levelCount = 1;
    db.subresourceRange.baseArrayLayer = 0;
    db.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &db);
}
