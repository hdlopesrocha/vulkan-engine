#include "ShadowRenderer.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include <backends/imgui_impl_vulkan.h>
#include <stdexcept>
#include <fstream>
#include <limits>

ShadowRenderer::ShadowRenderer(uint32_t shadowMapSize)
    : shadowMapSize(shadowMapSize) {}

ShadowRenderer::~ShadowRenderer() {
    // Don't call cleanup() here - it should be called explicitly before device destruction
    // The member variables are set to VK_NULL_HANDLE after cleanup, so this is safe
}

VkDescriptorSetLayout ShadowRenderer::getShadowDescriptorSetLayout(VulkanApp* app) const {
    return app->getDescriptorSetLayout();
}

void ShadowRenderer::init(VulkanApp* app) {
    createShadowMap(app);
    createShadowRenderPass(app);
    createShadowFramebuffer(app);
    createShadowPipeline(app);
}

void ShadowRenderer::cleanup(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // If asynchronous submissions are active, defer destruction of resources
    bool pending = app->hasPendingCommandBuffers();

    if (shadowMapImGuiDescSet != VK_NULL_HANDLE) {
        VkDescriptorSet ds = shadowMapImGuiDescSet;
        if (pending) {
            app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
        } else {
            ImGui_ImplVulkan_RemoveTexture(ds);
        }
        shadowMapImGuiDescSet = VK_NULL_HANDLE;
    }

    // Do not destroy Vulkan objects here; the VulkanResourceManager owns
    // destruction. Clear local handles to avoid accidental use.
    shadowFramebuffer = VK_NULL_HANDLE;
    shadowRenderPass = VK_NULL_HANDLE;
    shadowMapSampler = VK_NULL_HANDLE;
    shadowMapView = VK_NULL_HANDLE;
    shadowColorImageView = VK_NULL_HANDLE;
    shadowMapImage = VK_NULL_HANDLE;
    shadowColorImage = VK_NULL_HANDLE;
    shadowMapMemory = VK_NULL_HANDLE;
    shadowColorImageMemory = VK_NULL_HANDLE;

    // staging resources handled by centralized cleanup

}

void ShadowRenderer::createShadowMap(VulkanApp* app) {
    VkDevice device = app->getDevice();
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = shadowMapSize;
    imageInfo.extent.height = shadowMapSize;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateImage(device, &imageInfo, nullptr, &shadowMapImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow map image!");
    }
    
    fprintf(stderr, "[ShadowRenderer] shadowMapImage = %p\n", (void*)shadowMapImage);
    // Register shadow map image
    app->resources.addImage(shadowMapImage, "ShadowRenderer: shadowMapImage");
    
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, shadowMapImage, &memRequirements);

    if (memRequirements.size == 0) {
        throw std::runtime_error("failed to get valid memory requirements for shadow map image (size == 0)");
    }
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = app->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &shadowMapMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate shadow map memory!");
    }
    
    vkBindImageMemory(device, shadowMapImage, shadowMapMemory, 0);
    // Register shadow map device memory
    app->resources.addDeviceMemory(shadowMapMemory, "ShadowRenderer: shadowMapMemory");
    app->resources.addDeviceMemory(shadowMapMemory, "ShadowRenderer: shadowMapMemory");
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = shadowMapImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_D32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(device, &viewInfo, nullptr, &shadowMapView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow map image view!");
    }
    fprintf(stderr, "[ShadowRenderer] createImageView: shadowMapView=%p image=%p\n", (void*)shadowMapView, (void*)shadowMapImage);
    // Register shadow map image view
    app->resources.addImageView(shadowMapView, "ShadowRenderer: shadowMapView");
    app->resources.addImageView(shadowMapView, "ShadowRenderer: shadowMapView");
    
    // Create sampler for shadow map
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_LESS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    
    if (vkCreateSampler(device, &samplerInfo, nullptr, &shadowMapSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow map sampler!");
    }
    fprintf(stderr, "[ShadowRenderer] createSampler: shadowMapSampler=%p\n", (void*)shadowMapSampler);
    // Register sampler
    app->resources.addSampler(shadowMapSampler, "ShadowRenderer: shadowMapSampler");
    app->resources.addSampler(shadowMapSampler, "ShadowRenderer: shadowMapSampler");
    
    // Transition shadow map from UNDEFINED to READ_ONLY_OPTIMAL so the render pass
    // can start from a valid layout on the first frame
    {
        VkCommandBuffer cmd = app->beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadowMapImage;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        app->endSingleTimeCommands(cmd);
    }
    
    // Create ImGui descriptor set for shadow map visualization
    shadowMapImGuiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
        shadowMapSampler, 
        shadowMapView, 
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
    );

    // Create a transient color image so the shadow renderpass has a color attachment
    VkFormat colorFormat = app->getSwapchainImageFormat();
    VkImageCreateInfo colorInfo{};
    colorInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    colorInfo.imageType = VK_IMAGE_TYPE_2D;
    colorInfo.extent.width = shadowMapSize;
    colorInfo.extent.height = shadowMapSize;
    colorInfo.extent.depth = 1;
    colorInfo.mipLevels = 1;
    colorInfo.arrayLayers = 1;
    colorInfo.format = colorFormat;
    colorInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    colorInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    colorInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    colorInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &colorInfo, nullptr, &shadowColorImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow color image!");
    }
    fprintf(stderr, "[ShadowRenderer] createImage: shadowColorImage=%p\n", (void*)shadowColorImage);
    // Register color image
    app->resources.addImage(shadowColorImage, "ShadowRenderer: shadowColorImage");

    vkGetImageMemoryRequirements(device, shadowColorImage, &memRequirements);
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = app->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &shadowColorImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate shadow color image memory!");
    }
    vkBindImageMemory(device, shadowColorImage, shadowColorImageMemory, 0);
    // Register color image memory
    app->resources.addDeviceMemory(shadowColorImageMemory, "ShadowRenderer: shadowColorImageMemory");

    // color image view
    VkImageViewCreateInfo colorViewInfo{};
    colorViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    colorViewInfo.image = shadowColorImage;
    colorViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorViewInfo.format = colorFormat;
    colorViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorViewInfo.subresourceRange.baseMipLevel = 0;
    colorViewInfo.subresourceRange.levelCount = 1;
    colorViewInfo.subresourceRange.baseArrayLayer = 0;
    colorViewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &colorViewInfo, nullptr, &shadowColorImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow color image view!");
    }
    fprintf(stderr, "[ShadowRenderer] createImageView: shadowColorImageView=%p image=%p\n", (void*)shadowColorImageView, (void*)shadowColorImage);
    // Register color image view
    app->resources.addImageView(shadowColorImageView, "ShadowRenderer: shadowColorImageView");
}

void ShadowRenderer::createShadowRenderPass(VulkanApp* app) {
    VkDevice device = app->getDevice();
    // Create a render pass with a color + depth attachment so it's compatible with the
    // main render pass (same attachment count and formats). The color attachment is
    // unused but allows pipeline objects to be shared between passes.
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = app->getSwapchainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Use depth-stencil attachment layout while rendering, and read-only optimal when sampling from shaders
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    // Use depth/stencil attachment layout for the subpass
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow render pass!");
    }
    // Register shadow render pass
    app->resources.addRenderPass(shadowRenderPass, "ShadowRenderer: shadowRenderPass");
}

void ShadowRenderer::createShadowFramebuffer(VulkanApp* app) {
    VkDevice device = app->getDevice();
    VkImageView attachments[] = { shadowColorImageView, shadowMapView };

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = shadowRenderPass;
    framebufferInfo.attachmentCount = 2;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = shadowMapSize;
    framebufferInfo.height = shadowMapSize;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadowFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow framebuffer!");
    }
    // Register shadow framebuffer with resource manager
    app->resources.addFramebuffer(shadowFramebuffer, "ShadowRenderer: shadowFramebuffer");
}

void ShadowRenderer::createShadowPipeline(VulkanApp* /*app*/) {
    // ShadowRenderer should reuse the application's main graphics pipeline created by the app
    // (main.cpp creates the pipeline and registers it via setAppGraphicsPipeline()).
    // This function is a no-op placeholder; the pipeline is retrieved from the app in beginShadowPass.
}

void ShadowRenderer::beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, const glm::mat4& lightSpaceMatrix) {
    currentLightSpaceMatrix = lightSpaceMatrix;
    
    // Transition shadow map from READ_ONLY to DEPTH_STENCIL_ATTACHMENT_OPTIMAL before shadow pass
    // Use the SAME command buffer to record the barrier so it executes in sequence with the render pass
    if (shadowMapImage != VK_NULL_HANDLE) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadowMapImage;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    } else {
        fprintf(stderr, "[ShadowRenderer] Warning: shadowMapImage is VK_NULL_HANDLE in beginShadowPass, skipping barrier.\n");
    }
    
    // Begin shadow render pass only if both renderPass and framebuffer are valid
    if (shadowRenderPass == VK_NULL_HANDLE || shadowFramebuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[ShadowRenderer] Error: shadowRenderPass or shadowFramebuffer is VK_NULL_HANDLE in beginShadowPass, skipping render pass.\n");
        return;
    }

    VkRenderPassBeginInfo shadowRenderPassInfo{};
    shadowRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    shadowRenderPassInfo.renderPass = shadowRenderPass;
    shadowRenderPassInfo.framebuffer = shadowFramebuffer;
    shadowRenderPassInfo.renderArea.offset = {0, 0};
    shadowRenderPassInfo.renderArea.extent = {shadowMapSize, shadowMapSize};

    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    shadowRenderPassInfo.clearValueCount = 2;
    shadowRenderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffer, &shadowRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Set viewport and scissor for shadow map
    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = (float)shadowMapSize;
    shadowViewport.height = (float)shadowMapSize;
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = {shadowMapSize, shadowMapSize};
    vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

    // Use negative depth bias to pull shadow map closer, filling gaps at edges
    vkCmdSetDepthBias(commandBuffer, -1.5f, 0.0f, -2.0f);

    // Bind the shadow pipeline
    // Bind the app's main graphics pipeline (preferred) or fall back to any
    // locally stored pipeline handle.
    VkPipeline p = app->getAppGraphicsPipeline();
    if (p == VK_NULL_HANDLE) p = shadowPipeline;
    //printf("[ShadowRenderer] beginShadowPass: getAppGraphicsPipeline()=%p, shadowPipeline=%p, binding p=%p\n", (void*)app->getAppGraphicsPipeline(), (void*)shadowPipeline, (void*)p);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
}

void ShadowRenderer::renderObject(VulkanApp* app, VkCommandBuffer commandBuffer, 
                                 const VertexBufferObject& vbo, VkDescriptorSet descriptorSet) {

    // Bind descriptor sets: material set (set 0) and per-instance set (set 1) if available.
    VkDescriptorSet matDs = app->getMaterialDescriptorSet();
    VkPipelineLayout layout = app->getPipelineLayout();
    if (shadowPipelineLayout != VK_NULL_HANDLE) layout = shadowPipelineLayout;
    if (matDs != VK_NULL_HANDLE) {
        VkDescriptorSet sets[2] = { matDs, descriptorSet };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, 0, nullptr);
    } else {
        // Fallback: bind only the provided instance set at set 0 for compatibility
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSet, 0, nullptr);
    }
    
    // Bind vertex/index buffers
    VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    
    // Draw
    vkCmdDrawIndexed(commandBuffer, vbo.indexCount, 1, 0, 0, 0);
}

void ShadowRenderer::endShadowPass(VulkanApp* /*app*/, VkCommandBuffer commandBuffer) {
    // End the shadow render pass recorded on the provided command buffer
    // The render pass finalLayout (DEPTH_STENCIL_READ_ONLY_OPTIMAL) automatically
    // transitions the shadow map to the correct layout for shader sampling.
    // No additional barrier is needed here.
    vkCmdEndRenderPass(commandBuffer);
}

void ShadowRenderer::readbackShadowDepth(VulkanApp* app) {
    // Readback the shadow depth image to a host-visible buffer and write a PGM for debugging
    VkDevice device = app->getDevice();
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(shadowMapSize) * shadowMapSize * sizeof(float);

    // Create staging buffer
    Buffer stagingBuffer = app->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Record commands: transition to TRANSFER_SRC, copy image to buffer, transition back to READ_ONLY
    VkCommandBuffer cmd = app->beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = shadowMapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0,0,0};
    region.imageExtent = { shadowMapSize, shadowMapSize, 1 };

    vkCmdCopyImageToBuffer(cmd, shadowMapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer.buffer, 1, &region);

    // transition back to read-only optimal for shader sampling
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    app->endSingleTimeCommands(cmd);

    // Map and write PGM
    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, imageSize, 0, &data);
    float* depths = reinterpret_cast<float*>(data);
    size_t count = static_cast<size_t>(shadowMapSize) * shadowMapSize;

    float minD = std::numeric_limits<float>::infinity();
    float maxD = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        float v = depths[i];
        if (v < minD) minD = v;
        if (v > maxD) maxD = v;
    }
    if (!(minD <= maxD)) { minD = 0.0f; maxD = 1.0f; }

    // create normalized 8-bit image
    std::vector<unsigned char> image(count);
    float range = maxD - minD;
    for (size_t i = 0; i < count; ++i) {
        float v = depths[i];
        float n = (range > 1e-6f) ? (v - minD) / range : 0.0f;
        unsigned char c = static_cast<unsigned char>(std::min(1.0f, std::max(0.0f, n)) * 255.0f);
        image[i] = c;
    }

    // ensure bin directory exists and write PGM
    std::ofstream ofs("shadow_depth.pgm", std::ios::binary);
    if (ofs) {
        ofs << "P5\n" << shadowMapSize << " " << shadowMapSize << "\n255\n";
        ofs.write(reinterpret_cast<const char*>(image.data()), image.size());
        ofs.close();
    }

    std::cerr << "[ShadowRenderer] depth stats: min=" << minD << " max=" << maxD << "\n";

    vkUnmapMemory(device, stagingBuffer.memory);
    // Defer actual destruction to VulkanResourceManager; clear local handles
    stagingBuffer.buffer = VK_NULL_HANDLE;
    stagingBuffer.memory = VK_NULL_HANDLE;
}

void ShadowRenderer::requestWireframeReadback() {
    requestWireframeReadbackFlag = true;
}
