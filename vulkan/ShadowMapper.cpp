#include "ShadowMapper.hpp"
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
    if (shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, shadowPipeline, nullptr);
        shadowPipeline = VK_NULL_HANDLE;
    }
    if (shadowPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
        shadowPipelineLayout = VK_NULL_HANDLE;
    }
    if (shadowDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, shadowDescriptorSetLayout, nullptr);
        shadowDescriptorSetLayout = VK_NULL_HANDLE;
    }
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
    if (shadowMapImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, shadowMapImage, nullptr);
        shadowMapImage = VK_NULL_HANDLE;
    }
    if (shadowMapMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, shadowMapMemory, nullptr);
        shadowMapMemory = VK_NULL_HANDLE;
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
}

void ShadowMapper::createShadowRenderPass() {
    VkDevice device = vulkanApp->getDevice();
    
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 0;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &depthAttachment;
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
    
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = shadowRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &shadowMapView;
    framebufferInfo.width = shadowMapSize;
    framebufferInfo.height = shadowMapSize;
    framebufferInfo.layers = 1;
    
    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadowFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow framebuffer!");
    }
}

void ShadowMapper::createShadowPipeline() {
    VkDevice device = vulkanApp->getDevice();
    
    // Create descriptor set layout for UBO (binding 0) and height map (binding 3)
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    uboBinding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutBinding heightBinding{};
    heightBinding.binding = 3;
    heightBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    heightBinding.descriptorCount = 1;
    heightBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    heightBinding.pImmutableSamplers = nullptr;
    
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboBinding, heightBinding};
    
    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
    descriptorLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    descriptorLayoutInfo.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &shadowDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow descriptor set layout!");
    }
    
    ShaderStage shadowVertexShader = ShaderStage(
        vulkanApp->createShaderModule(FileReader::readFile("shaders/shadow.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT
    );

    // Build a list of ShaderStage objects (so their .info members remain valid)
    std::vector<ShaderStage> shaderObjs;
    shaderObjs.push_back(shadowVertexShader);

    // Attempt to load tessellation stages for shadow pipeline (displacement in shadow pass)
    bool hasTess = false;
    try {
        auto tescCode = FileReader::readFile("shaders/shadow.tesc.spv");
        auto teseCode = FileReader::readFile("shaders/shadow.tese.spv");
        if (!tescCode.empty() && !teseCode.empty()) {
            shaderObjs.emplace_back(vulkanApp->createShaderModule(tescCode), VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
            shaderObjs.emplace_back(vulkanApp->createShaderModule(teseCode), VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
            hasTess = true;
        }
    } catch (...) {
        hasTess = false;
    }

    shaderObjs.emplace_back(vulkanApp->createShaderModule(FileReader::readFile("shaders/shadow.frag.spv")), VK_SHADER_STAGE_FRAGMENT_BIT);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    // Assemble shader stage list from shaderObjs
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    for (const auto &s : shaderObjs) stages.push_back(s.info);
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    // Keep the shader stages array alive until pipeline creation
    std::vector<VkPipelineShaderStageCreateInfo> shaderStageVec = stages;
    pipelineInfo.pStages = shaderStageVec.data();
    
    // Vertex input
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions = {
        VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
        VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
        VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
        VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
        VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent) },
        VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
    };
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    
    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = hasTess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    
    // Viewport state
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    pipelineInfo.pViewportState = &viewportState;
    
    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_TRUE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;  // Back to standard culling
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    pipelineInfo.pRasterizationState = &rasterizer;
    
    // Multisample
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    pipelineInfo.pMultisampleState = &multisampling;
    
    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;
    pipelineInfo.pDepthStencilState = &depthStencil;
    
    // Color blend - no color attachments for shadow pass
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 0;
    pipelineInfo.pColorBlendState = &colorBlending;
    
    // Dynamic state
    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;
    pipelineInfo.pDynamicState = &dynamicState;
    
    VkPipelineLayoutCreateInfo shadowLayoutInfo{};
    shadowLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    shadowLayoutInfo.setLayoutCount = 1;
    shadowLayoutInfo.pSetLayouts = &shadowDescriptorSetLayout;
    shadowLayoutInfo.pushConstantRangeCount = 0; // No push constants, using UBO
    shadowLayoutInfo.pPushConstantRanges = nullptr;
    
    if (vkCreatePipelineLayout(device, &shadowLayoutInfo, nullptr, &shadowPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow pipeline layout!");
    }
    
    pipelineInfo.layout = shadowPipelineLayout;
    pipelineInfo.renderPass = shadowRenderPass;
    pipelineInfo.subpass = 0;
    VkPipelineTessellationStateCreateInfo tessState{};
    if (hasTess) {
        tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3;
        pipelineInfo.pTessellationState = &tessState;
    }

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow pipeline!");
    }

    // Destroy shader modules we created
    for (const auto &s : shaderObjs) {
        vkDestroyShaderModule(device, s.info.module, nullptr);
    }
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
    
    VkClearValue clearDepth;
    clearDepth.depthStencil = {1.0f, 0};
    shadowRenderPassInfo.clearValueCount = 1;
    shadowRenderPassInfo.pClearValues = &clearDepth;
    
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
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
}

void ShadowMapper::renderObject(VkCommandBuffer commandBuffer, const glm::mat4& modelMatrix, 
                                 const VertexBufferObject& vbo, VkDescriptorSet descriptorSet) {
    // Compute MVP for this object
    glm::mat4 mvp = currentLightSpaceMatrix * modelMatrix;
    
    // Bind descriptor set (includes UBO and height map)
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                            shadowPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    
    // Bind vertex/index buffers
    VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
    
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
    std::ofstream ofs("bin/shadow_depth.pgm", std::ios::binary);
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
