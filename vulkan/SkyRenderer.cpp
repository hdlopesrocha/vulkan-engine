#include "SkyRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

// For VBO creation
#include "VertexBufferObjectBuilder.hpp"
#include "../math/SphereModel.hpp"

SkyRenderer::SkyRenderer(VulkanApp* app_) : app(app_) {}

SkyRenderer::~SkyRenderer() { cleanup(); }

void SkyRenderer::init(VkRenderPass renderPassOverride) {
    // create sky gradient pipeline (vertex + fragment)
    skyVertModule = app->createShaderModule(FileReader::readFile("shaders/sky.vert.spv"));
    skyFragModule = app->createShaderModule(FileReader::readFile("shaders/sky.frag.spv"));
    
    ShaderStage skyVert = ShaderStage(skyVertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage skyFrag = ShaderStage(skyFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    // Use the application's main descriptor set layout (set 0 contains the shared UBO)
    std::vector<VkDescriptorSetLayout> setLayouts;
    setLayouts.push_back(app->getDescriptorSetLayout());
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4);
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { skyVert.info, skyFrag.info },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) }
        },
        setLayouts,
        &pushConstantRange,
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
        &pushConstantRange,
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
void SkyRenderer::render(VkCommandBuffer &cmd, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &ubo, const glm::mat4 &viewProjection, SkySettings::Mode skyMode) {
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
    // Push sky model matrix via push constants (visible to vertex + tessellation stages)
    // vkCmdPushConstants(cmd, app->getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &model);

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

void SkyRenderer::initSky(SkySettings &settings, VkDescriptorSet descriptorSet) {
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
        skySphere = std::make_unique<SkySphere>(app);
        skySphere->init(settings, descriptorSet);
    }
}

void SkyRenderer::update() {
    if (skySphere) skySphere->update();
}
