#include "Solid360Renderer.hpp"
#include "../utils/FileReader.hpp"
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
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    cube360DepthView = VK_NULL_HANDLE;
    for (auto& fb : cube360Framebuffers) fb = VK_NULL_HANDLE;
    equirect360Image = VK_NULL_HANDLE;
    equirect360Memory = VK_NULL_HANDLE;
    equirect360View = VK_NULL_HANDLE;
    equirect360Framebuffer = VK_NULL_HANDLE;
    equirect360RenderPass = VK_NULL_HANDLE;
    equirect360Pipeline = VK_NULL_HANDLE;
    equirect360PipelineLayout = VK_NULL_HANDLE;
    equirect360VertModule = VK_NULL_HANDLE;
    equirect360FragModule = VK_NULL_HANDLE;
    cube360DescSetLayout = VK_NULL_HANDLE;
    cube360DescPool = VK_NULL_HANDLE;
    cube360DescSet = VK_NULL_HANDLE;
}

void Solid360Renderer::createSolid360Targets(VulkanApp* app, VkRenderPass solidRenderPass, VkSampler linearSampler) {
    if (!app || solidRenderPass == VK_NULL_HANDLE) return;
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

    for (uint32_t face = 0; face < 6; ++face) {
        createView(cube360ColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_IMAGE_VIEW_TYPE_2D, face, 1,
                   cube360FaceViews[face], "Solid360Renderer: cube360 face view");
    }



    // --- 2. Shared depth image ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_D32_SFLOAT;
        imgInfo.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, cube360DepthImage, cube360DepthMemory);
    }
    createView(cube360DepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
               VK_IMAGE_VIEW_TYPE_2D, 0, 1,
               cube360DepthView, "Solid360Renderer: cube360 depth view");

    // --- 3. Per-face framebuffers (reuse solidRenderPass: color + depth) ---
    for (uint32_t face = 0; face < 6; ++face) {
        std::array<VkImageView, 2> attachments = {cube360FaceViews[face], cube360DepthView};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = solidRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments = attachments.data();
        fbInfo.width = CUBE360_FACE_SIZE;
        fbInfo.height = CUBE360_FACE_SIZE;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &cube360Framebuffers[face]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cube360 framebuffer!");
        app->resources.addFramebuffer(cube360Framebuffers[face], "Solid360Renderer: cube360 framebuffer");
    }

    // --- Equirectangular output ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = {EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.format = colorFormat;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        allocImage(imgInfo, equirect360Image, equirect360Memory);
    }

    // equirect view
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = equirect360Image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = colorFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &viewInfo, nullptr, &equirect360View) != VK_SUCCESS)
            throw std::runtime_error("Failed to create equirect image view!");
        app->resources.addImageView(equirect360View, "Solid360Renderer: equirect view");
    }

    // equirect framebuffer + renderpass + pipeline setup will be recreated below
    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = equirect360RenderPass; // created later
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments = nullptr;
    fbInfo.width = EQUIRECT360_WIDTH;
    fbInfo.height = EQUIRECT360_HEIGHT;
    fbInfo.layers = 1;

    // Create descriptor set layout & pool for cubemap → equirect conversion
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &b;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &cube360DescSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cube360 descriptor set layout!");
    app->resources.addDescriptorSetLayout(cube360DescSetLayout, "Solid360Renderer: cube360DescSetLayout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    // Allow freeing individual descriptor sets if the renderer frees them later
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &cube360DescPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cube360 descriptor pool!");
    app->resources.addDescriptorPool(cube360DescPool, "Solid360Renderer: cube360DescPool");

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = cube360DescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &cube360DescSetLayout;
    if (vkAllocateDescriptorSets(device, &allocInfo, &cube360DescSet) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate cube360 descriptor set!");

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = linearSampler;
    imgInfo.imageView = cube360CubeView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = cube360DescSet;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    // Create simple equirect renderpass and pipeline
    // Render pass
    VkAttachmentDescription att{};
    att.format = colorFormat;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &att;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dep;
    if (vkCreateRenderPass(device, &rpInfo, nullptr, &equirect360RenderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create equirect render pass!");
    app->resources.addRenderPass(equirect360RenderPass, "Solid360Renderer: equirectRenderPass");

    // equirect framebuffer
    VkImageView attv = equirect360View;
    fbInfo.pAttachments = &attv;
    fbInfo.renderPass = equirect360RenderPass;
    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &equirect360Framebuffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create equirect framebuffer!");
    app->resources.addFramebuffer(equirect360Framebuffer, "Solid360Renderer: equirectFramebuffer");

    // Load shaders for conversion
    // Use the fullscreen vertex shader and the cubemap->equirect fragment shader
    auto vertCode = FileReader::readFile("shaders/fullscreen.vert.spv");
    auto fragCode = FileReader::readFile("shaders/cubemap_to_equirect.frag.spv");
    if (vertCode.empty() || fragCode.empty()) {
        std::cerr << "[Solid360Renderer] Missing cube->equirect shaders" << std::endl;
        return;
    }
    equirect360VertModule = app->createShaderModule(vertCode);
    equirect360FragModule = app->createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = equirect360VertModule;
    vs.pName = "main";
    VkPipelineShaderStageCreateInfo fs{};
    fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = equirect360FragModule;
    fs.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vs, fs };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{0.0f, 0.0f, (float)EQUIRECT360_WIDTH, (float)EQUIRECT360_HEIGHT, 0.0f, 1.0f};
    VkRect2D scissor{{0,0},{EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT}};

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState attState{};
    attState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    attState.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &attState;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(float) * 2;

    VkPipelineLayoutCreateInfo layoutInfoPipe{};
    layoutInfoPipe.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfoPipe.setLayoutCount = 1;
    layoutInfoPipe.pSetLayouts = &cube360DescSetLayout;
    layoutInfoPipe.pushConstantRangeCount = 1;
    layoutInfoPipe.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &layoutInfoPipe, nullptr, &equirect360PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create equirect pipeline layout!");
    app->resources.addPipelineLayout(equirect360PipelineLayout, "Solid360Renderer: equirectPipelineLayout");

    // Enable dynamic viewport/scissor so we can set them at render time
    std::array<VkDynamicState, 2> dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynState.pDynamicStates = dynStates.data();

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vi;
    pipeInfo.pInputAssemblyState = &ia;
    pipeInfo.pViewportState = &vp;
    pipeInfo.pRasterizationState = &rs;
    pipeInfo.pMultisampleState = &ms;
    pipeInfo.pColorBlendState = &cb;
    pipeInfo.layout = equirect360PipelineLayout;
    pipeInfo.renderPass = equirect360RenderPass;
    pipeInfo.subpass = 0;
    pipeInfo.pDynamicState = &dynState;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &equirect360Pipeline) != VK_SUCCESS) {
        std::cerr << "[Solid360Renderer] Warning: failed to create equirect pipeline" << std::endl;
        equirect360Pipeline = VK_NULL_HANDLE;
    } else {
        app->resources.addPipeline(equirect360Pipeline, "Solid360Renderer: equirectPipeline");
    }
}

void Solid360Renderer::destroySolid360Targets(VulkanApp* app) {
    (void)app;
    cube360ColorImage = VK_NULL_HANDLE;
    cube360ColorMemory = VK_NULL_HANDLE;
    for (auto& v : cube360FaceViews) v = VK_NULL_HANDLE;
    cube360CubeView = VK_NULL_HANDLE;
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    cube360DepthView = VK_NULL_HANDLE;
    for (auto& fb : cube360Framebuffers) fb = VK_NULL_HANDLE;
    equirect360Image = VK_NULL_HANDLE;
    equirect360Memory = VK_NULL_HANDLE;
    equirect360View = VK_NULL_HANDLE;
    equirect360Framebuffer = VK_NULL_HANDLE;
    equirect360RenderPass = VK_NULL_HANDLE;
    equirect360Pipeline = VK_NULL_HANDLE;
    equirect360PipelineLayout = VK_NULL_HANDLE;
    equirect360VertModule = VK_NULL_HANDLE;
    equirect360FragModule = VK_NULL_HANDLE;
    cube360DescSetLayout = VK_NULL_HANDLE;
    cube360DescPool = VK_NULL_HANDLE;
    cube360DescSet = VK_NULL_HANDLE;
}

void Solid360Renderer::renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                                     VkRenderPass solidRenderPass,
                                     SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                                     SolidRenderer* solidRenderer,
                                     VkDescriptorSet mainDescriptorSet,
                                     Buffer& uniformBuffer, const UniformObject& ubo) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (cube360Framebuffers[0] == VK_NULL_HANDLE || equirect360Pipeline == VK_NULL_HANDLE) return;

    glm::vec3 camPos = glm::vec3(ubo.viewPos);
    struct FaceInfo { glm::vec3 target; glm::vec3 up; };
    const FaceInfo faces[6] = {
        { glm::vec3( 1, 0, 0), glm::vec3(0, 1, 0) },
        { glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0) },
        { glm::vec3( 0, 1, 0), glm::vec3(0, 0,-1) },
        { glm::vec3( 0,-1, 0), glm::vec3(0, 0, 1) },
        { glm::vec3( 0, 0, 1), glm::vec3(0, 1, 0) },
        { glm::vec3( 0, 0,-1), glm::vec3(0, 1, 0) },
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

        VkClearValue clears[2];
        clears[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = solidRenderPass;
        rpInfo.framebuffer = cube360Framebuffers[face];
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE};
        rpInfo.clearValueCount = 2;
        rpInfo.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

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
                solidRenderer->getIndirectRenderer().drawAll(cmd);
            }
        }

        vkCmdEndRenderPass(cmd);
    }

    // --- Convert cubemap → equirectangular ---
    {
        VkClearValue clear{};
        clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = equirect360RenderPass;
        rpInfo.framebuffer = equirect360Framebuffer;
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = {EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT};
        rpInfo.clearValueCount = 1;
        rpInfo.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, equirect360Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, equirect360PipelineLayout,
                                0, 1, &cube360DescSet, 0, nullptr);

        float resolution[2] = {(float)EQUIRECT360_WIDTH, (float)EQUIRECT360_HEIGHT};
        vkCmdPushConstants(cmd, equirect360PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(resolution), resolution);

        VkViewport viewport{0.0f, 0.0f, (float)EQUIRECT360_WIDTH, (float)EQUIRECT360_HEIGHT, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, {EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

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
