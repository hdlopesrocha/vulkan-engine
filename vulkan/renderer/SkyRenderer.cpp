#include "SkyRenderer.hpp"
#include "RendererUtils.hpp"

#include "../../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

// For VBO creation
#include "../VertexBufferObjectBuilder.hpp"
#include "../../math/SphereModel.hpp"

SkyRenderer::SkyRenderer() {}

SkyRenderer::~SkyRenderer() { cleanup(nullptr); }

void SkyRenderer::init(VulkanApp* app) {
    // Use the application's main descriptor set layout (set 0 contains the shared UBO)
    std::vector<VkDescriptorSetLayout> setLayouts;
    setLayouts.push_back(app->getDescriptorSetLayout());

    // --- Fullscreen sky pipelines (no vertex input, 3-vertex triangle) ---
    // These are used for on-screen rendering. The vertex shader reconstructs
    // world-space position from clip space via invViewProjection, so no sphere
    // geometry is needed — reducing vertex processing from 3 328 vertices to 3.
    skyFullscreenVertModule = app->getOrCreateShaderModule("shaders/sky_fullscreen.vert.spv");
    skyFragModule = app->getOrCreateShaderModule("shaders/sky.frag.spv");
    skyGridFragModule = app->getOrCreateShaderModule("shaders/sky_grid.frag.spv");

    ShaderStage fsVertStage(skyFullscreenVertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fsFragStage(skyFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);
    ShaderStage fsGridFragStage(skyGridFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkFormat colorFormat = app->getSwapchainImageFormat();
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    VkDevice device = app->getDevice();

    // Fullscreen gradient pipeline
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = setLayouts.data();
        VkPipelineLayout fsLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &fsLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky fullscreen pipeline layout!");
        app->resources.addPipelineLayout(fsLayout, "SkyRenderer: fullscreenGradientPipelineLayout");
        skyFullscreenPipelineLayout = fsLayout;

        RendererUtils::FullscreenPipelineOpts opts{};
        opts.cullMode = VK_CULL_MODE_NONE;
        opts.depthTestEnable = true;
        opts.depthWriteEnable = false;
        opts.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        skyFullscreenPipeline = RendererUtils::buildFullscreenPipeline(
            device, app, colorFormat, depthFormat, fsLayout,
            { fsVertStage.info, fsFragStage.info }, opts, "SkyRenderer: fullscreen gradient");
        std::cerr << "[SKY FULLSCREEN PIPELINE] Created pipeline=" << (void*)(VkPipeline)skyFullscreenPipeline << std::endl;
    }

    // Fullscreen grid pipeline
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = setLayouts.data();
        VkPipelineLayout fsGridLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &fsGridLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sky fullscreen grid pipeline layout!");
        app->resources.addPipelineLayout(fsGridLayout, "SkyRenderer: fullscreenGridPipelineLayout");
        skyFullscreenGridPipelineLayout = fsGridLayout;

        RendererUtils::FullscreenPipelineOpts opts{};
        opts.cullMode = VK_CULL_MODE_NONE;
        opts.depthTestEnable = true;
        opts.depthWriteEnable = false;
        opts.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        skyFullscreenGridPipeline = RendererUtils::buildFullscreenPipeline(
            device, app, colorFormat, depthFormat, fsGridLayout,
            { fsVertStage.info, fsGridFragStage.info }, opts, "SkyRenderer: fullscreen grid");
        std::cerr << "[SKY FULLSCREEN GRID PIPELINE] Created pipeline=" << (void*)(VkPipeline)skyFullscreenGridPipeline << std::endl;
    }

    // --- Sphere pipelines (retained for Solid360Renderer cubemap capture) ---
    skyVertModule = app->getOrCreateShaderModule("shaders/sky.vert.spv");

    ShaderStage skyVert = ShaderStage(skyVertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage skyFrag = ShaderStage(skyFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    GraphicsPipelineConfig cfg{};
    cfg.cullMode = VK_CULL_MODE_FRONT_BIT;
    cfg.depthWriteEnable = false;
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { skyVert.info, skyFrag.info },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{ ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
        },
        setLayouts,
        nullptr,
        cfg
    );
    skyPipeline = pipeline;
    skyPipelineLayout = layout;
    if (skyPipeline == VK_NULL_HANDLE || skyPipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[SKY PIPELINE ERROR] Failed to create sky pipeline or layout!" << std::endl;
    } else {
        std::cerr << "[SKY PIPELINE] Created pipeline=" << (void*)skyPipeline << " layout=" << (void*)skyPipelineLayout << std::endl;
    }

    // Sphere grid pipeline
    ShaderStage skyGridFrag = ShaderStage(skyGridFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    GraphicsPipelineConfig gridCfg{};
    gridCfg.cullMode = VK_CULL_MODE_FRONT_BIT;
    gridCfg.depthWriteEnable = false;
    gridCfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    auto [gridPipeline, gridLayout] = app->createGraphicsPipeline(
        { skyVert.info, skyGridFrag.info },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{ ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
        },
        setLayouts,
        nullptr,
        gridCfg
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
    // Select fullscreen pipeline based on sky mode (no vertex input, 3-vertex triangle)
    VkPipeline activePipeline = (skyMode == SkySettings::Mode::Grid) ? (VkPipeline)skyFullscreenGridPipeline : (VkPipeline)skyFullscreenPipeline;
    VkPipelineLayout activeLayout = (skyMode == SkySettings::Mode::Grid) ? (VkPipelineLayout)skyFullscreenGridPipelineLayout : (VkPipelineLayout)skyFullscreenPipelineLayout;
    if (activePipeline == VK_NULL_HANDLE || activeLayout == VK_NULL_HANDLE) {
        std::cerr << "[SKY RENDER ERROR] Attempted to bind VK_NULL_HANDLE pipeline or layout!" << std::endl;
        return;
    }

    // No UBO modification needed — the fullscreen vertex shader reconstructs
    // world position from clip space via invViewProjection, and the fragment
    // shader reads sky params from the dedicated SkyUBO (binding 6).

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, activePipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, activeLayout, 0, 1, &descriptorSet, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activeLayout, 0, 1, &descriptorSet, 0, nullptr);

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

    // Fullscreen triangle: 3 vertices, no vertex buffer, no index buffer.
    // The vertex shader generates positions from gl_VertexIndex and reconstructs
    // world-space position using invViewProjection — no sphere geometry needed.
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void SkyRenderer::cleanup(VulkanApp* app) {
    (void)app;
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
    } else if (descriptorSet != VK_NULL_HANDLE && skySphere) {
        // SkySphere already created — write its buffer to this additional descriptor set
        skySphere->writeDescriptorSet(app, descriptorSet);
    }
}

void SkyRenderer::update(VulkanApp* app) {
    if (skySphere) skySphere->update(app);
}

Buffer SkyRenderer::getSkyUniformBuffer() const {
    if (skySphere) return skySphere->getBuffer();
    return {};
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
                           VkImage &image, VmaAllocation &allocation, VkDeviceMemory &memory, VkImageView &view) {
        RendererUtils::createImage2DWithVma(device, app, w, h, format, usage, aspect,
                                            "SkyRenderer: equirect image", image, allocation, memory, view);
    };

    // Remove render pass creation - using dynamic rendering
    VkFormat colorFormat = app->getSwapchainImageFormat();

    // --- Load equirect shaders ---
    skyEquirectVertModule = app->getOrCreateShaderModule("shaders/fullscreen.vert.spv");
    skyEquirectFragModule = app->getOrCreateShaderModule("shaders/sky_equirect.frag.spv");

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
    for (uint32_t i = 0; i < SkyRenderer::SKY_FRAMES; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    offscreenWidth, offscreenHeight,
                    skyColorImages[i], skyColorAllocations[i], skyColorMemories[i], skyColorImageViews[i]);
        skyColorLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    std::cerr << "[SkyRenderer] Created equirectangular sky targets " << offscreenWidth << "x" << offscreenHeight << std::endl;
}

void SkyRenderer::destroyOffscreenTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < SkyRenderer::SKY_FRAMES; ++i) {
        if (skyColorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(skyColorImageViews[i]))
                vkDestroyImageView(device, skyColorImageViews[i], nullptr);
            skyColorImageViews[i] = VK_NULL_HANDLE;
        }
        app->destroyImageWithVma(skyColorImages[i], skyColorAllocations[i], skyColorMemories[i]);
        skyColorImages[i] = VK_NULL_HANDLE;
        skyColorAllocations[i] = VK_NULL_HANDLE;
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
    if (frameIndex >= skyColorImages.size() || skyColorImageViews[frameIndex] == VK_NULL_HANDLE)
        return;
    if (skyEquirectPipeline == VK_NULL_HANDLE || skyEquirectPipelineLayout == VK_NULL_HANDLE)
        return;

    // Update UBO so sky reads current lightDir, skyParams, etc.
    UniformObject skyUbo = ubo;
    skyUbo.passParams = glm::vec4(0.0f);
    app->updateUniformBuffer(uniformBuffer, &skyUbo, sizeof(UniformObject));

    // Guard against invalid image handles (pre-existing RADV issue)
    if (skyColorImages[frameIndex] == VK_NULL_HANDLE) return;

    // Transition color image: tracked layout → COLOR_ATTACHMENT_OPTIMAL
    {
        VkAccessFlags2 srcAccess;
        VkPipelineStageFlags2 srcStage;
        if (skyColorLayouts[frameIndex] == VK_IMAGE_LAYOUT_UNDEFINED) {
            srcAccess = 0;
            srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        } else {
            srcAccess = VK_ACCESS_2_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        }
        RendererUtils::transitionImageLayout(
            cmd, skyColorImages[frameIndex],
            skyColorLayouts[frameIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcAccess, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            srcStage, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
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

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, skyEquirectPipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyEquirectPipeline);
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, skyEquirectPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyEquirectPipelineLayout,
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
        RendererUtils::transitionImageLayout(
            cmd, skyColorImages[frameIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }
    skyColorLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
