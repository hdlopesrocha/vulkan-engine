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
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, true, VK_COMPARE_OP_LESS_OR_EQUAL
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
        VK_POLYGON_MODE_LINE, VK_CULL_MODE_BACK_BIT, true, true, VK_COMPARE_OP_LESS_OR_EQUAL
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
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, false
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
