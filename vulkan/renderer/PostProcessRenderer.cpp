
#include "PostProcessRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "WaterRenderer.hpp"   // WaterParams, WaterUBO
#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <array>
#include <vector>
#include <cstring>

PostProcessRenderer::PostProcessRenderer() {}

PostProcessRenderer::~PostProcessRenderer() {}

void PostProcessRenderer::init(VulkanApp* app) {
    createSampler(app);
    createPipeline(app);
    createDescriptorSets(app);

    // Create uniform buffer for post-process UBO
    uniformBuffer = app->createBuffer(sizeof(WaterUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void PostProcessRenderer::cleanup(VulkanApp* app) {
    uniformBuffer = {};
}

void PostProcessRenderer::setRenderSize(uint32_t width, uint32_t height) {
    renderWidth = width;
    renderHeight = height;
}

// ─── Sampler ──────────────────────────────────────────────────────────────────

void PostProcessRenderer::createSampler(VulkanApp* app) {
    linearSampler = app->createSamplerLinearClamp("PostProcessRenderer: linearSampler");
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

void PostProcessRenderer::createPipeline(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Descriptor set layout – 9 bindings (8 image samplers + 1 UBO)
    std::array<VkDescriptorSetLayoutBinding, 9> bindings{};

    for (int i = 0; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    DescriptorAllocator descAlloc{device, app};
    descriptorSetLayout = descAlloc.createLayout(
        bindings.data(), static_cast<uint32_t>(bindings.size()),
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        nullptr,
        "PostProcessRenderer: descriptorSetLayout");

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-process pipeline layout!");
    }
    app->resources.addPipelineLayout(pipelineLayout, "PostProcessRenderer: pipelineLayout");

    // Load shaders (cached by VulkanApp)
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/fullscreen.vert.spv");
    VkShaderModule fragModule = app->getOrCreateShaderModule("shaders/postprocess.frag.spv");

    std::vector<VkPipelineShaderStageCreateInfo> stages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr},
    };

    // No vertex input (fullscreen triangle generated in shader)
    pipeline = RendererUtils::buildFullscreenPipeline(
        device, app, app->getSwapchainImageFormat(), VK_FORMAT_D32_SFLOAT, pipelineLayout, stages,
        RendererUtils::FullscreenPipelineOpts{}, "PostProcessRenderer: pipeline");

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
}

// ─── Descriptor Sets ──────────────────────────────────────────────────────────

void PostProcessRenderer::createDescriptorSets(VulkanApp* app) {
    if (descriptorSetLayout == VK_NULL_HANDLE) return;

    DescriptorAllocator descAlloc{app->getDevice(), app};

    VkDescriptorPoolSize poolSizesDesc[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 * FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * FRAMES_IN_FLIGHT}
    };
    descriptorPool = descAlloc.createPool(
        poolSizesDesc, 2, FRAMES_IN_FLIGHT,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "PostProcessRenderer: descriptorPool");

    descAlloc.allocateSets(descriptorPool, descriptorSetLayout,
                           FRAMES_IN_FLIGHT, reinterpret_cast<VkDescriptorSet*>(descriptorSets.data()),
                           "PostProcessRenderer: descriptorSet");
}

// ─── Render ───────────────────────────────────────────────────────────────────

void PostProcessRenderer::render(VulkanApp* app, VkCommandBuffer cmd,
                                  VkImageView sceneColorView, VkImageView sceneDepthView,
                                  VkImageView waterColorView,
                                  VkImageView brushColorView, VkImageView brushDepthView,
                                  VkImageView brushBackFaceDepthView,
                                  VkImageView waterGeomDepthView,
                                  float brushAlpha, float brushMode,
                                  const glm::mat4& viewProj, const glm::mat4& invViewProj,
                                  const glm::vec3& viewPos,
                                  uint32_t frameIdx,
                                  VkImageView skyView) {
    if (pipeline == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] pipeline is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (cmd == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] cmd is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    VkDevice device = app->getDevice();

    // Update water UBO
    WaterUBO ubo{};
    ubo.viewProjection = viewProj;
    ubo.invViewProjection = invViewProj;
    ubo.viewPos = glm::vec4(viewPos, 1.0f);
    ubo.screenSize = glm::vec4(renderWidth, renderHeight, 1.0f / renderWidth, 1.0f / renderHeight);
    ubo.brushAlpha = brushAlpha;
    ubo.brushMode = brushMode;

    void* data;
    data = uniformBuffer.map(0);
    memcpy(data, &ubo, sizeof(WaterUBO));
    uniformBuffer.unmap(); // VMA persistent mapping

    // Prepare image infos and only write descriptors for valid image views
    std::array<VkDescriptorImageInfo, 9> imageInfos{};
    imageInfos[0] = {linearSampler, sceneColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[1] = {linearSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[2] = {linearSampler, waterColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Brush color and depth for deferred composition (depth-tested against scene+water)
    imageInfos[3] = {linearSampler, brushColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[4] = {linearSampler, brushDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Water geometry depth buffer for accurate brush-vs-water occlusion
    imageInfos[7] = {linearSampler, waterGeomDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Brush back-face depth for PAINT mode volume test
    imageInfos[8] = {linearSampler, brushBackFaceDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkDescriptorBufferInfo bufferInfo{uniformBuffer.buffer, 0, sizeof(WaterUBO)};

    // Sky color image info (binding 6) - fallback to scene color if null
    VkDescriptorImageInfo skyImageInfo{};
    if (skyView == VK_NULL_HANDLE) {
        skyImageInfo = imageInfos[0]; // fallback to scene color
    } else {
        skyImageInfo.sampler = linearSampler;
        skyImageInfo.imageView = skyView;
        skyImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkDescriptorSet currentDs = descriptorSets[frameIdx % FRAMES_IN_FLIGHT];

    DescriptorWriter writer(device);
    for (int i = 0; i < 5; ++i) {
        if (imageInfos[i].imageView == VK_NULL_HANDLE || imageInfos[i].sampler == VK_NULL_HANDLE) {
            continue;
        }
        writer.writeImage(currentDs, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          imageInfos[i].sampler, imageInfos[i].imageView,
                          imageInfos[i].imageLayout);
    }

    if (bufferInfo.buffer != VK_NULL_HANDLE) {
        writer.writeBuffer(currentDs, 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                           bufferInfo.buffer, bufferInfo.offset, bufferInfo.range);
    } else {
        std::cerr << "[PostProcessRenderer] Skipping UBO binding: buffer is VK_NULL_HANDLE" << std::endl;
    }

    // Sky color texture (binding 6)
    if (skyImageInfo.imageView != VK_NULL_HANDLE && skyImageInfo.sampler != VK_NULL_HANDLE) {
        writer.writeImage(currentDs, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          skyImageInfo.sampler, skyImageInfo.imageView,
                          skyImageInfo.imageLayout);
    }

    // Water geometry depth (binding 7) — used for brush-vs-water occlusion
    if (imageInfos[7].imageView != VK_NULL_HANDLE && imageInfos[7].sampler != VK_NULL_HANDLE) {
        writer.writeImage(currentDs, 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          imageInfos[7].sampler, imageInfos[7].imageView,
                          imageInfos[7].imageLayout);
    }

    // Brush back-face depth (binding 8) — used for PAINT mode volume test
    if (imageInfos[8].imageView != VK_NULL_HANDLE && imageInfos[8].sampler != VK_NULL_HANDLE) {
        writer.writeImage(currentDs, 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          imageInfos[8].sampler, imageInfos[8].imageView,
                          imageInfos[8].imageLayout);
    }

    writer.flush();

    // Set viewport and scissor (safe to call inside already-open dynamic rendering scope)
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind pipeline and descriptor set
    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 1, &currentDs, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &currentDs, 0, nullptr);

    // Draw fullscreen triangle (3 vertices, no vertex buffer needed)
    vkCmdDraw(cmd, 3, 1, 0, 0);
}
