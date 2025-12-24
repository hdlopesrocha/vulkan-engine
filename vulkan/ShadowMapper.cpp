#include "ShadowMapper.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include <backends/imgui_impl_vulkan.h>
#include <stdexcept>
#include <fstream>
#include <limits>

ShadowMapper::ShadowMapper(VulkanApp* app, uint32_t shadowMapSize)
    : vulkanApp(app), shadowMapSize(shadowMapSize) {}

ShadowMapper::~ShadowMapper() {
    // Don't call cleanup() here - it should be called explicitly before device destruction
    // The member variables are set to VK_NULL_HANDLE after cleanup, so this is safe
}

VkDescriptorSetLayout ShadowMapper::getShadowDescriptorSetLayout() const {
    return vulkanApp->getDescriptorSetLayout();
}

void ShadowMapper::init() {
    createShadowMap();
    createShadowRenderPass();
    createShadowFramebuffer();
    createShadowPipeline();
}

void ShadowMapper::cleanup() {
    VkDevice device = vulkanApp->getDevice();
    
    if (shadowMapImGuiDescSet != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(shadowMapImGuiDescSet);
        shadowMapImGuiDescSet = VK_NULL_HANDLE;
    }
    // shadow pipeline uses the main app's pipeline layout; don't destroy it here
    // shadow uses the main app's descriptor set layout; do not destroy it here
    if (shadowFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, shadowFramebuffer, nullptr);
        shadowFramebuffer = VK_NULL_HANDLE;
    }
    if (shadowRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, shadowRenderPass, nullptr);
        shadowRenderPass = VK_NULL_HANDLE;
    }
    if (shadowMapSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, shadowMapSampler, nullptr);
        shadowMapSampler = VK_NULL_HANDLE;
    }
    if (shadowMapView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, shadowMapView, nullptr);
        shadowMapView = VK_NULL_HANDLE;
    }
    if (shadowColorImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, shadowColorImageView, nullptr);
        shadowColorImageView = VK_NULL_HANDLE;
    }
    if (shadowMapImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, shadowMapImage, nullptr);
        shadowMapImage = VK_NULL_HANDLE;
    }
    if (shadowColorImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, shadowColorImage, nullptr);
        shadowColorImage = VK_NULL_HANDLE;
    }
    if (shadowMapMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, shadowMapMemory, nullptr);
        shadowMapMemory = VK_NULL_HANDLE;
    }
    if (shadowColorImageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, shadowColorImageMemory, nullptr);
        shadowColorImageMemory = VK_NULL_HANDLE;
    }
}

void ShadowMapper::createShadowMap() {
    VkDevice device = vulkanApp->getDevice();
    
    // Create depth image for shadow map
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
    
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, shadowMapImage, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanApp->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &shadowMapMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate shadow map memory!");
    }
    
    vkBindImageMemory(device, shadowMapImage, shadowMapMemory, 0);
    
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
    
    // Create ImGui descriptor set for shadow map visualization
    shadowMapImGuiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
        shadowMapSampler, 
        shadowMapView, 
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
    );

    // Create a transient color image so the shadow renderpass has a color attachment
    VkFormat colorFormat = vulkanApp->getSwapchainImageFormat();
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

    vkGetImageMemoryRequirements(device, shadowColorImage, &memRequirements);
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanApp->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &shadowColorImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate shadow color image memory!");
    }
    vkBindImageMemory(device, shadowColorImage, shadowColorImageMemory, 0);

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
}

void ShadowMapper::createShadowRenderPass() {
    VkDevice device = vulkanApp->getDevice();
    // Create a render pass with a color + depth attachment so it's compatible with the
    // main render pass (same attachment count and formats). The color attachment is
    // unused but allows pipeline objects to be shared between passes.
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = vulkanApp->getSwapchainImageFormat();
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
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
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
}

void ShadowMapper::createShadowFramebuffer() {
    VkDevice device = vulkanApp->getDevice();
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
}

void ShadowMapper::createShadowPipeline() {
    // ShadowMapper should reuse the application's main graphics pipeline created by the app
    // (main.cpp creates the pipeline and registers it via setAppGraphicsPipeline()).
    shadowPipeline = vulkanApp->getAppGraphicsPipeline();
    // If the app hasn't set the pipeline yet, leave shadowPipeline as VK_NULL_HANDLE.
    // beginShadowPass will bind the app pipeline directly to avoid double ownership.
}

void ShadowMapper::beginShadowPass(VkCommandBuffer commandBuffer, const glm::mat4& lightSpaceMatrix) {
    currentLightSpaceMatrix = lightSpaceMatrix;
    
    // Begin shadow render pass
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
    VkPipeline p = vulkanApp->getAppGraphicsPipeline();
    if (p == VK_NULL_HANDLE) p = shadowPipeline;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
}

void ShadowMapper::renderObject(VkCommandBuffer commandBuffer, 
                                 const VertexBufferObject& vbo, VkDescriptorSet descriptorSet) {

    // Bind descriptor sets: material set (set 0) and per-instance set (set 1) if available.
    VkDescriptorSet matDs = vulkanApp->getMaterialDescriptorSet();
    if (matDs != VK_NULL_HANDLE) {
        VkDescriptorSet sets[2] = { matDs, descriptorSet };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanApp->getPipelineLayout(), 0, 2, sets, 0, nullptr);
    } else {
        // Fallback: bind only the provided instance set at set 0 for compatibility
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanApp->getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);
    }
    
    // Bind vertex/index buffers
    VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    
    // Draw
    vkCmdDrawIndexed(commandBuffer, vbo.indexCount, 1, 0, 0, 0);
}

void ShadowMapper::endShadowPass(VkCommandBuffer commandBuffer) {
    vkCmdEndRenderPass(commandBuffer);
    
    // Transition shadow map from depth attachment to shader read
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = shadowMapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void ShadowMapper::readbackShadowDepth() {
    // Readback the shadow depth image to a host-visible buffer and write a PGM for debugging
    VkDevice device = vulkanApp->getDevice();
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(shadowMapSize) * shadowMapSize * sizeof(float);

    // Create staging buffer
    Buffer stagingBuffer = vulkanApp->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Record commands: transition to TRANSFER_SRC, copy image to buffer, transition back to READ_ONLY
    VkCommandBuffer cmd = vulkanApp->beginSingleTimeCommands();

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

    vulkanApp->endSingleTimeCommands(cmd);

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

    std::cerr << "[ShadowMapper] depth stats: min=" << minD << " max=" << maxD << "\n";

    vkUnmapMemory(device, stagingBuffer.memory);
    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);
}

void ShadowMapper::requestWireframeReadback() {
    requestWireframeReadbackFlag = true;
}
