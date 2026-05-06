#include "SkyRenderer.hpp"
#include "RendererUtils.hpp"

#include "../../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

// For VBO creation
#include "../VertexBufferObjectBuilder.hpp"
#include "../../math/SphereModel.hpp"

SkyRenderer::SkyRenderer() {}

SkyRenderer::~SkyRenderer() { cleanup(); }

void SkyRenderer::init(VulkanApp* app) {
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
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );
    skyPipeline = pipeline;
    skyPipelineLayout = layout;
    if (skyPipeline == VK_NULL_HANDLE || skyPipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[SKY PIPELINE ERROR] Failed to create sky pipeline or layout!" << std::endl;
    } else {
        std::cerr << "[SKY PIPELINE] Created pipeline=" << (void*)skyPipeline << " layout=" << (void*)skyPipelineLayout << std::endl;
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
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );
    skyGridPipeline = gridPipeline;
    skyGridPipelineLayout = gridLayout;
    if (skyGridPipeline == VK_NULL_HANDLE || skyGridPipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[SKY GRID PIPELINE ERROR] Failed to create sky grid pipeline or layout!" << std::endl;
    } else {
        std::cerr << "[SKY GRID PIPELINE] Created pipeline=" << (void*)skyGridPipeline << " layout=" << (void*)skyGridPipelineLayout << std::endl;
    }
}
void SkyRenderer::render(VulkanApp* app, VkCommandBuffer &cmd, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &ubo, const glm::mat4 &viewProjection, SkySettings::Mode skyMode) {
    // Select pipeline based on sky mode
        VkPipeline activePipeline = (skyMode == SkySettings::Mode::Grid) ? skyGridPipeline : skyPipeline;
        VkPipelineLayout activeLayout = (skyMode == SkySettings::Mode::Grid) ? skyGridPipelineLayout : skyPipelineLayout;
        if (activePipeline == VK_NULL_HANDLE || activeLayout == VK_NULL_HANDLE) {
            std::cerr << "[SKY RENDER ERROR] Attempted to bind VK_NULL_HANDLE pipeline or layout!" << std::endl;
            return;
        }
        //std::cerr << "[SKY RENDER] Binding pipeline=" << (void*)activePipeline << " layout=" << (void*)activeLayout << std::endl;

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
    //std::cerr << "[SKY RENDER] Binding descriptor set: skyDs=" << (void*)descriptorSet << std::endl;
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
        std::cerr << "[SkyRenderer::render] Sky VBO not available! vertexBuffer=" << (void*)skyVBO.vertexBuffer.buffer
                  << " indexCount=" << skyVBO.indexCount << std::endl;
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
        RendererUtils::createImage2D(device, app, w, h, format, usage, aspect,
                                     "SkyRenderer: equirect image", image, memory, view);
    };

    // Remove render pass creation - using dynamic rendering
    VkFormat colorFormat = app->getSwapchainImageFormat();

    // --- Load equirect shaders ---
    skyEquirectVertModule = app->createShaderModule(FileReader::readFile("shaders/fullscreen.vert.spv"));
    skyEquirectFragModule = app->createShaderModule(FileReader::readFile("shaders/sky_equirect.frag.spv"));

    // --- Create equirect pipeline (fullscreen triangle, no vertex input, no depth) ---
    {
        std::vector<VkPipelineShaderStageCreateInfo> stages = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   skyEquirectVertModule, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, skyEquirectFragModule, "main", nullptr},
        };

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

        skyEquirectPipeline = RendererUtils::buildFullscreenPipeline(
            device, app, colorFormat, VK_FORMAT_UNDEFINED, skyEquirectPipelineLayout, stages,
            RendererUtils::FullscreenPipelineOpts{}, "SkyRenderer: skyEquirectPipeline");
    }

    // --- Per-frame color images (no depth, no framebuffers needed for dynamic rendering) ---
    for (int i = 0; i < 2; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    offscreenWidth, offscreenHeight,
                    skyColorImages[i], skyColorMemories[i], skyColorImageViews[i]);
        skyColorLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    std::cerr << "[SkyRenderer] Created equirectangular sky targets " << offscreenWidth << "x" << offscreenHeight << std::endl;
}

void SkyRenderer::destroyOffscreenTargets(VulkanApp* app) {
    (void)app;
    for (int i = 0; i < 2; ++i) {
        skyColorImageViews[i] = VK_NULL_HANDLE;
        skyColorImages[i] = VK_NULL_HANDLE;
        skyColorMemories[i] = VK_NULL_HANDLE;
        skyColorLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    skyEquirectPipeline = VK_NULL_HANDLE;
    skyEquirectPipelineLayout = VK_NULL_HANDLE;
    skyEquirectVertModule = VK_NULL_HANDLE;
    skyEquirectFragModule = VK_NULL_HANDLE;
}

void SkyRenderer::renderOffscreen(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                   VkDescriptorSet descriptorSet, Buffer &uniformBuffer,
                                   const UniformObject &ubo, const glm::mat4 &viewProjection,
                                   SkySettings::Mode skyMode) {
    if (skyColorImageViews[frameIndex] == VK_NULL_HANDLE)
        return;
    if (skyEquirectPipeline == VK_NULL_HANDLE || skyEquirectPipelineLayout == VK_NULL_HANDLE)
        return;

    // Update UBO so sky reads current lightDir, skyParams, etc.
    UniformObject skyUbo = ubo;
    skyUbo.passParams = glm::vec4(0.0f);
    app->updateUniformBuffer(uniformBuffer, &skyUbo, sizeof(UniformObject));

    // Transition color image: tracked layout → COLOR_ATTACHMENT_OPTIMAL
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = skyColorLayouts[frameIndex];
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = skyColorImages[frameIndex];
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = (skyColorLayouts[frameIndex] == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = skyColorImageViews[frameIndex];
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue = clear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {offscreenWidth, offscreenHeight};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAtt;

    vkCmdBeginRendering(cmd, &renderingInfo);

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

    vkCmdEndRendering(cmd);

    // Transition color: COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = skyColorImages[frameIndex];
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    skyColorLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
