#include "SolidRenderer.hpp"
#include "../utils/FileReader.hpp"
#include "ShaderStage.hpp"
#include <glm/gtc/matrix_transform.hpp>

SolidRenderer::SolidRenderer(VulkanApp* app_) : app(app_), indirectRenderer() {}
SolidRenderer::~SolidRenderer() { cleanup(); }

void SolidRenderer::init(VulkanApp* app_) {
    if (app_) app = app_;
    if (!app) return;
    indirectRenderer.init(app);
}

void SolidRenderer::createRenderTargets(uint32_t width, uint32_t height) {
    if (!app) return;
    renderWidth = width;
    renderHeight = height;
    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create solid image!");
        }

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate solid image memory!");
        }

        vkBindImageMemory(device, image, memory, 0);

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
            throw std::runtime_error("Failed to create solid image view!");
        }
    };

    // Create simple render pass for solid offscreen (color+depth)
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = app->getSwapchainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    std::array<VkAttachmentDescription, 2> attachments{colorAttachment, depthAttachment};

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &solidRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create solid render pass!");
    }

    for (int i = 0; i < 2; ++i) {
        createImage(app->getSwapchainImageFormat(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                    solidColorImages[i], solidColorMemories[i], solidColorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                    solidDepthImages[i], solidDepthMemories[i], solidDepthImageViews[i]);

        std::array<VkImageView, 2> attachmentsViews = {solidColorImageViews[i], solidDepthImageViews[i]};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = solidRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachmentsViews.size());
        fbInfo.pAttachments = attachmentsViews.data();
        fbInfo.width = width;
        fbInfo.height = height;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &solidFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create solid framebuffer!");
        }
    }
}

void SolidRenderer::destroyRenderTargets() {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (int i = 0; i < 2; ++i) {
        if (solidFramebuffers[i] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device, solidFramebuffers[i], nullptr);
            solidFramebuffers[i] = VK_NULL_HANDLE;
        }
        if (solidColorImageViews[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device, solidColorImageViews[i], nullptr);
            solidColorImageViews[i] = VK_NULL_HANDLE;
        }
        if (solidColorImages[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device, solidColorImages[i], nullptr);
            solidColorImages[i] = VK_NULL_HANDLE;
        }
        if (solidColorMemories[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, solidColorMemories[i], nullptr);
            solidColorMemories[i] = VK_NULL_HANDLE;
        }
        if (solidDepthImageViews[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device, solidDepthImageViews[i], nullptr);
            solidDepthImageViews[i] = VK_NULL_HANDLE;
        }
        if (solidDepthImages[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device, solidDepthImages[i], nullptr);
            solidDepthImages[i] = VK_NULL_HANDLE;
        }
        if (solidDepthMemories[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, solidDepthMemories[i], nullptr);
            solidDepthMemories[i] = VK_NULL_HANDLE;
        }
    }
    if (solidRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, solidRenderPass, nullptr);
        solidRenderPass = VK_NULL_HANDLE;
    }
}

void SolidRenderer::beginPass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear) {
    if (cmd == VK_NULL_HANDLE || solidRenderPass == VK_NULL_HANDLE || solidFramebuffers[frameIndex] == VK_NULL_HANDLE) {
        fprintf(stderr, "[SolidRenderer::beginPass] Missing cmd/renderPass/framebuffer, skipping.\n");
        return;
    }
    std::array<VkClearValue, 2> clears{colorClear, depthClear};
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = solidRenderPass;
    rpInfo.framebuffer = solidFramebuffers[frameIndex];
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = {renderWidth, renderHeight};
    rpInfo.clearValueCount = static_cast<uint32_t>(clears.size());
    rpInfo.pClearValues = clears.data();
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void SolidRenderer::endPass(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;
    vkCmdEndRenderPass(cmd);
}

void SolidRenderer::createPipelines() {
    if (!app) return;

    ShaderStage vertexShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT
    );

    ShaderStage fragmentShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    ShaderStage tescShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    );
    ShaderStage teseShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    );

    // Descriptor set layouts: ensure main UBO (set 0) is first
    // Main shaders don't use set 1 (material descriptor set), so only include set 0
    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE) setLayouts.push_back(app->getDescriptorSetLayout());
    // Note: Removed material descriptor set layout since main shaders don't use set = 1

    // Push constant range (model matrix, vertex+frag stages)
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4);

    auto [pipeline, layout] = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        setLayouts,
        &pushConstantRange,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, solidRenderPass
    );
    graphicsPipeline = pipeline;
    graphicsPipelineLayout = layout;
    // Register the main graphics pipeline with the app so ShadowRenderer can use it
    if (app) {
        printf("[SolidRenderer] setAppGraphicsPipeline: pipeline=%p\n", (void*)graphicsPipeline);
        app->setAppGraphicsPipeline(graphicsPipeline);
    }

    auto [wirePipeline, wireLayout] = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        setLayouts,
        &pushConstantRange,
        VK_POLYGON_MODE_LINE, VK_CULL_MODE_BACK_BIT, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, solidRenderPass
    );
    graphicsPipelineWire = wirePipeline;
    graphicsPipelineWireLayout = wireLayout;

    auto [depthPipeline, depthLayout] = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        setLayouts,
        &pushConstantRange,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, false, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, solidRenderPass
    );
    depthPrePassPipeline = depthPipeline;
    depthPrePassPipelineLayout = depthLayout;

    // Destroy shader modules
    vkDestroyShaderModule(app->getDevice(), teseShader.info.module, nullptr);
    vkDestroyShaderModule(app->getDevice(), tescShader.info.module, nullptr);
    vkDestroyShaderModule(app->getDevice(), fragmentShader.info.module, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertexShader.info.module, nullptr);
}

void SolidRenderer::depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool) {
    if (!app) {
        fprintf(stderr, "[SolidRenderer::depthPrePass] app is nullptr, skipping.\n");
        return;
    }
    if (depthPrePassPipeline == VK_NULL_HANDLE) {
        fprintf(stderr, "[SolidRenderer::depthPrePass] depthPrePassPipeline is VK_NULL_HANDLE, skipping.\n");
        return;
    }

    //printf("[SolidRenderer] vkCmdBindPipeline: depthPrePassPipeline=%p\n", (void*)depthPrePassPipeline);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipeline);
    VkDescriptorSet matDs = app->getMaterialDescriptorSet();
    if (matDs != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipelineLayout, 0, 1, &matDs, 0, nullptr);
    }
    // Bind vertex and index buffers before issuing draw commands
    indirectRenderer.bindBuffers(commandBuffer);
    indirectRenderer.drawIndirectOnly(commandBuffer, app);
    if (queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, queryPool, 5);
    }
}

void SolidRenderer::draw(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled) {
    static int frameCount = 0;
    if (frameCount++ == 0) {
        printf("[DEBUG] SolidRenderer::draw called for the first time\n");
    }
    
    if (!app) {
        fprintf(stderr, "[SolidRenderer::draw] app is nullptr, skipping.\n");
        return;
    }
    
    static bool printedOnce = false;
    
    VkPipelineLayout usedLayout = wireframeEnabled ? graphicsPipelineWireLayout : graphicsPipelineLayout;
    if (wireframeEnabled) {
        if (graphicsPipelineWire == VK_NULL_HANDLE) {
            fprintf(stderr, "[SolidRenderer::draw] graphicsPipelineWire is VK_NULL_HANDLE, skipping.\n");
            return;
        }
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineWire);
    } else {
        if (graphicsPipeline == VK_NULL_HANDLE) {
            fprintf(stderr, "[SolidRenderer::draw] graphicsPipeline is VK_NULL_HANDLE, skipping.\n");
            return;
        }
        if (!printedOnce) {
            printf("[SolidRenderer::draw] binding pipeline=%p, layout=%p\n", (void*)graphicsPipeline, (void*)usedLayout);
        }
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    }

    // Set dynamic viewport and scissor (required since pipeline uses dynamic state)
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(app->getWidth());
    viewport.height = static_cast<float>(app->getHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    if (!printedOnce) {
        printf("[SolidRenderer::draw] viewport: x=%.1f y=%.1f w=%.1f h=%.1f depth=[%.1f,%.1f]\n",
               viewport.x, viewport.y, viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth);
    }
    
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(app->getWidth()), static_cast<uint32_t>(app->getHeight())};
    
    if (!printedOnce) {
        printf("[SolidRenderer::draw] scissor: offset=(%d,%d) extent=(%u,%u)\n",
               scissor.offset.x, scissor.offset.y, scissor.extent.width, scissor.extent.height);
    }
    
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Bind descriptor set 0: main UBO/samplers (perTextureDescriptorSet)
    // Main shaders only use set 0, no set 1 needed
    if (!printedOnce) {
        printf("[SolidRenderer::draw] perTextureDescriptorSet=%p\n", (void*)perTextureDescriptorSet);
        printedOnce = true;
    }
    
    if (perTextureDescriptorSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, usedLayout, 0, 1, &perTextureDescriptorSet, 0, nullptr);
    } else {
        fprintf(stderr, "[SolidRenderer::draw] ERROR: perTextureDescriptorSet is NULL!\n");
    }
    
    // Draw all meshes using GPU-culled indirect commands
    printf("[SolidRenderer::draw] About to call drawPrepared\n");
    indirectRenderer.drawPrepared(commandBuffer, app);
    printf("[SolidRenderer::draw] drawPrepared returned\n");
}

void SolidRenderer::cleanup() {
    if (!app) return;
    if (graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (graphicsPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(app->getDevice(), graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (graphicsPipelineWire != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), graphicsPipelineWire, nullptr);
        graphicsPipelineWire = VK_NULL_HANDLE;
    }
    if (graphicsPipelineWireLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(app->getDevice(), graphicsPipelineWireLayout, nullptr);
        graphicsPipelineWireLayout = VK_NULL_HANDLE;
    }
    if (depthPrePassPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), depthPrePassPipeline, nullptr);
        depthPrePassPipeline = VK_NULL_HANDLE;
    }
    if (depthPrePassPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(app->getDevice(), depthPrePassPipelineLayout, nullptr);
        depthPrePassPipelineLayout = VK_NULL_HANDLE;
    }

    // Remove meshes
    for (auto &entry : nodeModelVersions) {
        if (entry.second.meshId != UINT32_MAX) indirectRenderer.removeMesh(entry.second.meshId);
    }
    nodeModelVersions.clear();

    indirectRenderer.cleanup(app);
}
