#include "SkyRenderer.hpp"
#include "FileReader.hpp"
#include <glm/gtc/matrix_transform.hpp>

SkyRenderer::SkyRenderer(VulkanApp* app_) : app(app_) {}

SkyRenderer::~SkyRenderer() { cleanup(); }

void SkyRenderer::init() {
    // create sky pipeline (vertex + fragment)
    ShaderStage skyVert = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/sky.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT
    );
    ShaderStage skyFrag = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/sky.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    skyPipeline = app->createGraphicsPipeline(
        { skyVert.info, skyFrag.info },
        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, false
    );

    // Shader modules are owned/managed by the app helper; no need to destroy here
}

void SkyRenderer::render(VkCommandBuffer &cmd, const VertexBufferObject &vbo, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, VkDeviceSize dynamicOffset, const UniformObject &uboStatic, const glm::mat4 &projMat, const glm::mat4 &viewMat) {
    if (skyPipeline == VK_NULL_HANDLE) return;

    // update sky uniform centered at camera
    UniformObject skyUbo = uboStatic;
    glm::vec3 camPos = glm::vec3(uboStatic.viewPos);
    glm::mat4 model = glm::translate(glm::mat4(1.0f), camPos) * glm::scale(glm::mat4(1.0f), glm::vec3(50.0f));
    skyUbo.model = model;
    skyUbo.mvp = projMat * viewMat * model;
    skyUbo.passParams = glm::vec4(0.0f);
    app->updateUniformBufferRange(uniformBuffer, dynamicOffset, &skyUbo, sizeof(UniformObject));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
    // Bind descriptor set with dynamic offset for this sky instance
    uint32_t dyn = static_cast<uint32_t>(dynamicOffset);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, 1, &descriptorSet, 1, &dyn);

    const VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
    const VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, vbo.indexCount, 1, 0, 0, 0);
}

void SkyRenderer::cleanup() {
    if (skyPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), skyPipeline, nullptr);
        skyPipeline = VK_NULL_HANDLE;
    }
}
