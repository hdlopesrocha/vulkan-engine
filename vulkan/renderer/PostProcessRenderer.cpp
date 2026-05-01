
#include "PostProcessRenderer.hpp"
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
    // Clear local handles; VulkanResourceManager is responsible for actual destruction
    pipeline = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    descriptorPool = VK_NULL_HANDLE;
    descriptorSetLayout = VK_NULL_HANDLE;
    linearSampler = VK_NULL_HANDLE;
    uniformBuffer = {};
}

void PostProcessRenderer::setRenderSize(uint32_t width, uint32_t height) {
    renderWidth = width;
    renderHeight = height;
}

// ─── Sampler ──────────────────────────────────────────────────────────────────

void PostProcessRenderer::createSampler(VulkanApp* app) {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &linearSampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-process linear sampler!");
    }
    std::cout << "[PostProcessRenderer] createSampler: linearSampler=" << (void*)linearSampler << std::endl;
    app->resources.addSampler(linearSampler, "PostProcessRenderer: linearSampler");
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

void PostProcessRenderer::createPipeline(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Descriptor set layout – 7 bindings (5 image samplers + 1 UBO + 1 sky sampler)
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};

    for (int i = 0; i < 5; ++i) {
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

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-process descriptor set layout!");
    }
    app->resources.addDescriptorSetLayout(descriptorSetLayout, "PostProcessRenderer: descriptorSetLayout");

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-process pipeline layout!");
    }
    app->resources.addPipelineLayout(pipelineLayout, "PostProcessRenderer: pipelineLayout");

    // Load shaders
    auto vertCode = FileReader::readFile("shaders/fullscreen.vert.spv");
    auto fragCode = FileReader::readFile("shaders/postprocess.frag.spv");

    if (vertCode.empty() || fragCode.empty()) {
        std::cerr << "[PostProcessRenderer] Warning: Could not load post-process shaders, skipping pipeline creation" << std::endl;
        return;
    }

    VkShaderModule vertModule = app->createShaderModule(vertCode);
    VkShaderModule fragModule = app->createShaderModule(fragCode);

    std::vector<VkPipelineShaderStageCreateInfo> stages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr},
    };

    // No vertex input (fullscreen triangle generated in shader)
    pipeline = RendererUtils::buildFullscreenPipeline(
        device, app, app->getSwapchainRenderPass(), pipelineLayout, stages,
        RendererUtils::FullscreenPipelineOpts{}, "PostProcessRenderer: pipeline");

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
}

// ─── Descriptor Sets ──────────────────────────────────────────────────────────

void PostProcessRenderer::createDescriptorSets(VulkanApp* app) {
    if (descriptorSetLayout == VK_NULL_HANDLE) return;

    VkDevice device = app->getDevice();

    // Create descriptor pool
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 6;  // scene color + scene depth + water depth + water normal + water mask + sky color
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-process descriptor pool!");
    }
    app->resources.addDescriptorPool(descriptorPool, "PostProcessRenderer: descriptorPool");

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate post-process descriptor set!");
    }
    app->resources.addDescriptorSet(descriptorSet, "PostProcessRenderer: descriptorSet");
}

// ─── Render ───────────────────────────────────────────────────────────────────

void PostProcessRenderer::render(VulkanApp* app, VkCommandBuffer cmd,
                                  VkFramebuffer swapchainFramebuffer,
                                  VkRenderPass swapchainRenderPass,
                                  VkImageView sceneColorView, VkImageView sceneDepthView,
                                  VkImageView waterDepthView,
                                  const glm::mat4& viewProj, const glm::mat4& invViewProj,
                                  const glm::vec3& viewPos,
                                  bool beginRenderPass, VkImageView skyView) {
    if (pipeline == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] pipeline is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (cmd == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] cmd is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (swapchainFramebuffer == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] swapchainFramebuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (swapchainRenderPass == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] swapchainRenderPass is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    VkDevice device = app->getDevice();

    // Update water UBO
    WaterUBO ubo{};
    ubo.viewProjection = viewProj;
    ubo.invViewProjection = invViewProj;
    ubo.viewPos = glm::vec4(viewPos, 1.0f);
    ubo.screenSize = glm::vec4(renderWidth, renderHeight, 1.0f / renderWidth, 1.0f / renderHeight);

    void* data;
    vkMapMemory(device, uniformBuffer.memory, 0, sizeof(WaterUBO), 0, &data);
    memcpy(data, &ubo, sizeof(WaterUBO));
    vkUnmapMemory(device, uniformBuffer.memory);

    // Prepare image infos and only write descriptors for valid image views
    std::array<VkDescriptorImageInfo, 5> imageInfos{};
    imageInfos[0] = {linearSampler, sceneColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[1] = {linearSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[2] = {linearSampler, waterDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // waterNormal and waterMask are no longer produced by WaterRenderer; leave as VK_NULL_HANDLE so writes are skipped
    imageInfos[3] = {linearSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[4] = {linearSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkDescriptorBufferInfo bufferInfo{uniformBuffer.buffer, 0, sizeof(WaterUBO)};

    // Sky color image info (binding 6) - require explicit skyView
    if (skyView == VK_NULL_HANDLE) {
        throw std::runtime_error("PostProcessRenderer::render requires a valid skyView (no fallback allowed)");
    }
    VkDescriptorImageInfo skyImageInfo{};
    skyImageInfo.sampler = linearSampler;
    skyImageInfo.imageView = skyView;
    skyImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::vector<VkWriteDescriptorSet> writes;
    for (int i = 0; i < 5; ++i) {
        if (imageInfos[i].imageView == VK_NULL_HANDLE || imageInfos[i].sampler == VK_NULL_HANDLE) {
            std::cerr << "[PostProcessRenderer] Skipping binding " << i
                      << ": imageView=" << (void*)imageInfos[i].imageView
                      << " sampler=" << (void*)imageInfos[i].sampler << std::endl;
            continue;
        }
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = i;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfos[i];
        writes.push_back(write);
    }

    if (bufferInfo.buffer != VK_NULL_HANDLE) {
        VkWriteDescriptorSet bufWrite{};
        bufWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bufWrite.dstSet = descriptorSet;
        bufWrite.dstBinding = 5;
        bufWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bufWrite.descriptorCount = 1;
        bufWrite.pBufferInfo = &bufferInfo;
        writes.push_back(bufWrite);
    } else {
        std::cerr << "[PostProcessRenderer] Skipping UBO binding: buffer is VK_NULL_HANDLE" << std::endl;
    }

    // Sky color texture (binding 6)
    if (skyImageInfo.imageView != VK_NULL_HANDLE && skyImageInfo.sampler != VK_NULL_HANDLE) {
        VkWriteDescriptorSet skyWrite{};
        skyWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        skyWrite.dstSet = descriptorSet;
        skyWrite.dstBinding = 6;
        skyWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        skyWrite.descriptorCount = 1;
        skyWrite.pImageInfo = &skyImageInfo;
        writes.push_back(skyWrite);
    }

    if (!writes.empty()) vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    if (beginRenderPass) {
        // Begin render pass (output to swapchain)
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = swapchainRenderPass;
        renderPassInfo.framebuffer = swapchainFramebuffer;
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = {renderWidth, renderHeight};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Set viewport and scissor (safe to call inside already-open render pass)
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &descriptorSet, 0, nullptr);

    // Draw fullscreen triangle (3 vertices, no vertex buffer needed)
    vkCmdDraw(cmd, 3, 1, 0, 0);
    // NOTE: Render pass is NOT ended here – caller is responsible for ending it
    // This allows ImGui or other overlays to be rendered in the same pass
}
