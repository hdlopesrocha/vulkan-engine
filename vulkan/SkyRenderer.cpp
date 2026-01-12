#include "SkyRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

SkyRenderer::SkyRenderer(VulkanApp* app_) : app(app_) {}

SkyRenderer::~SkyRenderer() { cleanup(); }

void SkyRenderer::init() {
    // create sky gradient pipeline (vertex + fragment)
    skyVertModule = app->createShaderModule(FileReader::readFile("shaders/sky.vert.spv"));
    skyFragModule = app->createShaderModule(FileReader::readFile("shaders/sky.frag.spv"));
    
    ShaderStage skyVert = ShaderStage(skyVertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage skyFrag = ShaderStage(skyFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    skyPipeline = app->createGraphicsPipeline(
        { skyVert.info, skyFrag.info },
        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, false, true
    );

    // create sky grid pipeline (vertex + grid fragment)
    skyGridFragModule = app->createShaderModule(FileReader::readFile("shaders/sky_grid.frag.spv"));
    ShaderStage skyGridFrag = ShaderStage(skyGridFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    skyGridPipeline = app->createGraphicsPipeline(
        { skyVert.info, skyGridFrag.info },
        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, false, true
    );
}

void SkyRenderer::render(VkCommandBuffer &cmd, const VertexBufferObject &vbo, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProjection, SkyMode skyMode) {
    // Select pipeline based on sky mode
    VkPipeline activePipeline = (skyMode == SkyMode::Grid) ? skyGridPipeline : skyPipeline;
    if (activePipeline == VK_NULL_HANDLE) return;

    // update sky uniform centered at camera
    UniformObject skyUbo = uboStatic;
    glm::vec3 camPos = glm::vec3(uboStatic.viewPos);
    glm::mat4 model = glm::translate(glm::mat4(1.0f), camPos) * glm::scale(glm::mat4(1.0f), glm::vec3(50.0f));
    skyUbo.viewProjection = viewProjection;
    skyUbo.passParams = glm::vec4(0.0f);
    app->updateUniformBuffer(uniformBuffer, &skyUbo, sizeof(UniformObject));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);
    // Bind material set (set 0) and sky descriptor set (set 1) if material set exists
    VkDescriptorSet matDs = app->getMaterialDescriptorSet();
    if (matDs != VK_NULL_HANDLE) {
        VkDescriptorSet sets[2] = { matDs, descriptorSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, 2, sets, 0, nullptr);
    } else {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);
    }
    // Push sky model matrix via push constants (visible to vertex + tessellation stages)
    vkCmdPushConstants(cmd, app->getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &model);

    const VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
    const VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, vbo.indexCount, 1, 0, 0, 0);
}

void SkyRenderer::cleanup() {
    if (skyPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), skyPipeline, nullptr);
        skyPipeline = VK_NULL_HANDLE;
    }
    if (skyGridPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), skyGridPipeline, nullptr);
        skyGridPipeline = VK_NULL_HANDLE;
    }
    if (skyVertModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(app->getDevice(), skyVertModule, nullptr);
        skyVertModule = VK_NULL_HANDLE;
    }
    if (skyFragModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(app->getDevice(), skyFragModule, nullptr);
        skyFragModule = VK_NULL_HANDLE;
    }
    if (skyGridFragModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(app->getDevice(), skyGridFragModule, nullptr);
        skyGridFragModule = VK_NULL_HANDLE;
    }
}
