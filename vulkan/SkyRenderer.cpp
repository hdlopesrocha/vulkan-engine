#include "SkyRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

// For VBO creation
#include "VertexBufferObjectBuilder.hpp"
#include "../math/SphereModel.hpp"

SkyRenderer::SkyRenderer() {}

SkyRenderer::~SkyRenderer() { cleanup(); }

void SkyRenderer::init(VulkanApp* app, VkRenderPass renderPassOverride) {
    // create sky gradient pipeline (vertex + fragment)
    skyVertModule = app->createShaderModule(FileReader::readFile("shaders/sky.vert.spv"));
    skyFragModule = app->createShaderModule(FileReader::readFile("shaders/sky.frag.spv"));
    
    ShaderStage skyVert = ShaderStage(skyVertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage skyFrag = ShaderStage(skyFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    // Use the application's main descriptor set layout (set 0 contains the shared UBO)
    std::vector<VkDescriptorSetLayout> setLayouts;
    setLayouts.push_back(app->getDescriptorSetLayout());
    // No push-constants required for sky pipeline (sky uses UBO/viewPos to position the sphere).
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { skyVert.info, skyFrag.info },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) }
        },
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, false, true, VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        renderPassOverride
    );
    skyPipeline = pipeline;
    skyPipelineLayout = layout;
    if (skyPipeline == VK_NULL_HANDLE || skyPipelineLayout == VK_NULL_HANDLE) {
        fprintf(stderr, "[SKY PIPELINE ERROR] Failed to create sky pipeline or layout!\n");
    } else {
        fprintf(stderr, "[SKY PIPELINE] Created pipeline=%p layout=%p\n", (void*)skyPipeline, (void*)skyPipelineLayout);
    }

    // create sky grid pipeline (vertex + grid fragment)
    skyGridFragModule = app->createShaderModule(FileReader::readFile("shaders/sky_grid.frag.spv"));
    ShaderStage skyGridFrag = ShaderStage(skyGridFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    auto [gridPipeline, gridLayout] = app->createGraphicsPipeline(
        { skyVert.info, skyGridFrag.info },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) }
        },
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, false, true, VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        renderPassOverride
    );
    skyGridPipeline = gridPipeline;
    skyGridPipelineLayout = gridLayout;
    if (skyGridPipeline == VK_NULL_HANDLE || skyGridPipelineLayout == VK_NULL_HANDLE) {
        fprintf(stderr, "[SKY GRID PIPELINE ERROR] Failed to create sky grid pipeline or layout!\n");
    } else {
        fprintf(stderr, "[SKY GRID PIPELINE] Created pipeline=%p layout=%p\n", (void*)skyGridPipeline, (void*)skyGridPipelineLayout);
    }
}
void SkyRenderer::render(VulkanApp* app, VkCommandBuffer &cmd, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &ubo, const glm::mat4 &viewProjection, SkySettings::Mode skyMode) {
    // Select pipeline based on sky mode
        VkPipeline activePipeline = (skyMode == SkySettings::Mode::Grid) ? skyGridPipeline : skyPipeline;
        VkPipelineLayout activeLayout = (skyMode == SkySettings::Mode::Grid) ? skyGridPipelineLayout : skyPipelineLayout;
        if (activePipeline == VK_NULL_HANDLE || activeLayout == VK_NULL_HANDLE) {
            fprintf(stderr, "[SKY RENDER ERROR] Attempted to bind VK_NULL_HANDLE pipeline or layout!\n");
            return;
        }
        //fprintf(stderr, "[SKY RENDER] Binding pipeline=%p layout=%p\n", (void*)activePipeline, (void*)activeLayout);

    // update sky uniform centered at camera
    // Use the live UBO passed in so we preserve fields like debugParams
    UniformObject skyUbo = ubo;
    glm::vec3 camPos = glm::vec3(ubo.viewPos);
    glm::mat4 model = glm::translate(glm::mat4(1.0f), camPos) * glm::scale(glm::mat4(1.0f), glm::vec3(50.0f));
    skyUbo.viewProjection = viewProjection;
    skyUbo.passParams = glm::vec4(0.0f);
    app->updateUniformBuffer(uniformBuffer, &skyUbo, sizeof(UniformObject));

    //printf("[SkyRenderer] vkCmdBindPipeline: activePipeline=%p\n", (void*)activePipeline);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);
    // Only bind the sky descriptor set (set 0)
    //fprintf(stderr, "[SKY RENDER] Binding descriptor set: skyDs=%p\n", (void*)descriptorSet);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activeLayout, 0, 1, &descriptorSet, 0, nullptr);
    // No push-constants used for sky; model is encoded into UBO/viewPos.

    // Explicitly set viewport/scissor because this pipeline relies on dynamic state
    VkExtent2D extent = app->getSwapchainExtent();
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Use internal VBO if available
    if (skyVBO.vertexBuffer.buffer != VK_NULL_HANDLE && skyVBO.indexCount > 0) {
        const VkBuffer vertexBuffers[] = { skyVBO.vertexBuffer.buffer };
        const VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, skyVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, skyVBO.indexCount, 1, 0, 0, 0);
    } else {
        fprintf(stderr, "[SkyRenderer::render] Sky VBO not available! vertexBuffer=%p indexCount=%u\n",
            (void*)skyVBO.vertexBuffer.buffer, skyVBO.indexCount);
    }
}

void SkyRenderer::cleanup() {
    // Clear local handles; actual destruction is handled by VulkanResourceManager
    skyPipeline = VK_NULL_HANDLE;
    skyGridPipeline = VK_NULL_HANDLE;
    skyVertModule = VK_NULL_HANDLE;
    skyFragModule = VK_NULL_HANDLE;
    skyGridFragModule = VK_NULL_HANDLE;

    // Offscreen equirect pipeline handles
    skyEquirectPipeline = VK_NULL_HANDLE;
    skyEquirectPipelineLayout = VK_NULL_HANDLE;
    skyEquirectVertModule = VK_NULL_HANDLE;
    skyEquirectFragModule = VK_NULL_HANDLE;

    // Cleanup sky sphere and VBO handles
    if (skySphere) {
        skySphere->cleanup();
        skySphere.reset();
    }
    skyVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    skyVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    skyVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    skyVBO.indexBuffer.memory = VK_NULL_HANDLE;
    skyVBO.indexCount = 0;
}

void SkyRenderer::init(VulkanApp* app, SkySettings &settings, VkDescriptorSet descriptorSet) {
    if (!app) return;
    // Create sphere VBO if not present
    if (skyVBO.vertexBuffer.buffer == VK_NULL_HANDLE && skyVBO.indexCount == 0) {
        printf("[SkyRenderer::initSky] Creating sphere VBO...\n");
        SphereModel sphere(0.5f, 32, 16, 0);
        skyVBO = VertexBufferObjectBuilder::create(app, sphere);
        printf("[SkyRenderer::initSky] Created skyVBO: vertexBuffer=%p indexCount=%u\n", 
            (void*)skyVBO.vertexBuffer.buffer, skyVBO.indexCount);
    } else {
        printf("[SkyRenderer::initSky] Sky VBO already exists: vertexBuffer=%p indexCount=%u\n",
            (void*)skyVBO.vertexBuffer.buffer, skyVBO.indexCount);
    }

    if (descriptorSet != VK_NULL_HANDLE && !skySphere) {
        skySphere = std::make_unique<SkySphere>();
        skySphere->init(app, settings, descriptorSet);
    }
}

void SkyRenderer::update(VulkanApp* app) {
    if (skySphere) skySphere->update(app);
}

// ---------- Offscreen equirectangular sky rendering ----------

void SkyRenderer::createOffscreenTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    // Use fixed equirectangular resolution (2:1 aspect for full sphere)
    offscreenWidth = 2048;
    offscreenHeight = 1024;
    VkDevice device = app->getDevice();

    // Helper: create image + memory + view
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           uint32_t w, uint32_t h,
                           VkImage &image, VkDeviceMemory &memory, VkImageView &view) {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = format;
        imgInfo.extent = {w, h, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = usage;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &imgInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky offscreen image!");
        app->resources.addImage(image, "SkyRenderer: equirect image");

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, image, &memReqs);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate sky offscreen image memory!");
        app->resources.addDeviceMemory(memory, "SkyRenderer: equirect memory");
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
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky offscreen image view!");
        app->resources.addImageView(view, "SkyRenderer: equirect view");
    };

    // --- Render pass: color only (no depth needed for fullscreen equirect) ---
    VkAttachmentDescription colorAtt{};
    VkFormat colorFormat = app->getSwapchainImageFormat();
    colorAtt.format = colorFormat;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = nullptr;

    // 0→EXTERNAL dependency: flush color writes so downstream shaders can sample
    VkSubpassDependency dependency{};
    dependency.srcSubpass = 0;
    dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAtt;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &skyOffscreenRenderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create sky equirect render pass!");
    app->resources.addRenderPass(skyOffscreenRenderPass, "SkyRenderer: skyEquirectRenderPass");

    // --- Load equirect shaders ---
    skyEquirectVertModule = app->createShaderModule(FileReader::readFile("shaders/fullscreen.vert.spv"));
    skyEquirectFragModule = app->createShaderModule(FileReader::readFile("shaders/sky_equirect.frag.spv"));

    // --- Create equirect pipeline (fullscreen triangle, no vertex input, no depth) ---
    {
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};
        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = skyEquirectVertModule;
        shaderStages[0].pName = "main";
        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = skyEquirectFragModule;
        shaderStages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAtt{};
        colorBlendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAtt.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAtt;

        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // Push constant for resolution (vec2)
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(float) * 2;

        VkDescriptorSetLayout setLayout = app->getDescriptorSetLayout();
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &setLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &skyEquirectPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky equirect pipeline layout!");
        app->resources.addPipelineLayout(skyEquirectPipelineLayout, "SkyRenderer: skyEquirectPipelineLayout");

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = skyEquirectPipelineLayout;
        pipelineInfo.renderPass = skyOffscreenRenderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyEquirectPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky equirect pipeline!");
        app->resources.addPipeline(skyEquirectPipeline, "SkyRenderer: skyEquirectPipeline");
    }

    // --- Per-frame color images + framebuffers (no depth) ---
    for (int i = 0; i < 2; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    offscreenWidth, offscreenHeight,
                    skyColorImages[i], skyColorMemories[i], skyColorImageViews[i]);

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = skyOffscreenRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &skyColorImageViews[i];
        fbInfo.width = offscreenWidth;
        fbInfo.height = offscreenHeight;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &skyFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky equirect framebuffer!");
        app->resources.addFramebuffer(skyFramebuffers[i], "SkyRenderer: skyEquirectFramebuffer");
    }

    fprintf(stderr, "[SkyRenderer] Created equirectangular sky targets %ux%u\n", offscreenWidth, offscreenHeight);
}

void SkyRenderer::destroyOffscreenTargets(VulkanApp* app) {
    for (int i = 0; i < 2; ++i) {
        skyFramebuffers[i] = VK_NULL_HANDLE;
        skyColorImageViews[i] = VK_NULL_HANDLE;
        skyColorImages[i] = VK_NULL_HANDLE;
        skyColorMemories[i] = VK_NULL_HANDLE;
    }
    skyOffscreenRenderPass = VK_NULL_HANDLE;
    skyEquirectPipeline = VK_NULL_HANDLE;
    skyEquirectPipelineLayout = VK_NULL_HANDLE;
    skyEquirectVertModule = VK_NULL_HANDLE;
    skyEquirectFragModule = VK_NULL_HANDLE;
}

void SkyRenderer::renderOffscreen(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                   VkDescriptorSet descriptorSet, Buffer &uniformBuffer,
                                   const UniformObject &ubo, const glm::mat4 &viewProjection,
                                   SkySettings::Mode skyMode) {
    if (skyOffscreenRenderPass == VK_NULL_HANDLE || skyFramebuffers[frameIndex] == VK_NULL_HANDLE)
        return;
    if (skyEquirectPipeline == VK_NULL_HANDLE || skyEquirectPipelineLayout == VK_NULL_HANDLE)
        return;

    // Update UBO so sky reads current lightDir, skyParams, etc.
    UniformObject skyUbo = ubo;
    skyUbo.passParams = glm::vec4(0.0f);
    app->updateUniformBuffer(uniformBuffer, &skyUbo, sizeof(UniformObject));

    // Begin render pass (single color attachment, no depth)
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = skyOffscreenRenderPass;
    rpBegin.framebuffer = skyFramebuffers[frameIndex];
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = {offscreenWidth, offscreenHeight};
    rpBegin.clearValueCount = 1;
    rpBegin.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyEquirectPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyEquirectPipelineLayout,
                            0, 1, &descriptorSet, 0, nullptr);

    // Push resolution constant
    float resolution[2] = { static_cast<float>(offscreenWidth), static_cast<float>(offscreenHeight) };
    vkCmdPushConstants(cmd, skyEquirectPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(resolution), resolution);

    VkViewport viewport{0.0f, 0.0f, (float)offscreenWidth, (float)offscreenHeight, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, {offscreenWidth, offscreenHeight}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Fullscreen triangle (3 vertices, no vertex buffer)
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}
