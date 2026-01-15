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

    graphicsPipeline = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, false, true, VK_COMPARE_OP_EQUAL
    );

    app->setAppGraphicsPipeline(graphicsPipeline);

    graphicsPipelineWire = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        VK_POLYGON_MODE_LINE,
        VK_CULL_MODE_BACK_BIT,
        false,
        true,
        VK_COMPARE_OP_EQUAL
    );

    depthPrePassPipeline = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_BACK_BIT,
        true, // depthWrite
        false // colorWrite disabled
    );

    // Destroy shader modules
    vkDestroyShaderModule(app->getDevice(), teseShader.info.module, nullptr);
    vkDestroyShaderModule(app->getDevice(), tescShader.info.module, nullptr);
    vkDestroyShaderModule(app->getDevice(), fragmentShader.info.module, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertexShader.info.module, nullptr);
}

void SolidRenderer::depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool) {
    if (!app) return;
    if (depthPrePassPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipeline);
        VkDescriptorSet matDs = app->getMaterialDescriptorSet();
        if (matDs != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, 1, &matDs, 0, nullptr);
        }
        indirectRenderer.drawIndirectOnly(commandBuffer, app);
    }
    if (queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, queryPool, 5);
    }
}

void SolidRenderer::draw(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled) {
    if (!app) return;

    if (wireframeEnabled && graphicsPipelineWire != VK_NULL_HANDLE)
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineWire);
    else
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Bind descriptor sets (material + per-texture)
    VkDescriptorSet setsToBind[2];
    uint32_t bindCount = 0;
    VkDescriptorSet matDs = app->getMaterialDescriptorSet();
    if (matDs != VK_NULL_HANDLE) setsToBind[bindCount++] = matDs;
    if (perTextureDescriptorSet != VK_NULL_HANDLE) setsToBind[bindCount++] = perTextureDescriptorSet;
    if (bindCount > 0) vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, bindCount, setsToBind, 0, nullptr);

    indirectRenderer.drawIndirectOnly(commandBuffer, app);
}

void SolidRenderer::cleanup() {
    if (!app) return;
    if (graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (graphicsPipelineWire != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), graphicsPipelineWire, nullptr);
        graphicsPipelineWire = VK_NULL_HANDLE;
    }
    if (depthPrePassPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(app->getDevice(), depthPrePassPipeline, nullptr);
        depthPrePassPipeline = VK_NULL_HANDLE;
    }

    // Remove meshes
    for (auto &entry : nodeModelVersions) {
        if (entry.second.meshId != UINT32_MAX) indirectRenderer.removeMesh(entry.second.meshId);
    }
    nodeModelVersions.clear();

    indirectRenderer.cleanup(app);
}
