#include "ShadowRenderer.hpp"
#include "VulkanApp.hpp"
#include "ShaderStage.hpp"
#include "../utils/FileReader.hpp"
#include "../math/Vertex.hpp"
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
    createShadowMaps(app);
    createShadowRenderPass(app);
    createShadowFramebuffers(app);
    createShadowPipeline(app);
}

void ShadowRenderer::cleanup(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // If asynchronous submissions are active, defer destruction of resources
    bool pending = app->hasPendingCommandBuffers();

    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        if (cascades[i].imguiDescSet != VK_NULL_HANDLE) {
            VkDescriptorSet ds = cascades[i].imguiDescSet;
            if (pending) {
                app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
            } else {
                ImGui_ImplVulkan_RemoveTexture(ds);
            }
            cascades[i].imguiDescSet = VK_NULL_HANDLE;
        }
        cascades[i].framebuffer = VK_NULL_HANDLE;
        cascades[i].depthView = VK_NULL_HANDLE;
        cascades[i].colorView = VK_NULL_HANDLE;
        cascades[i].depthImage = VK_NULL_HANDLE;
        cascades[i].colorImage = VK_NULL_HANDLE;
        cascades[i].depthMemory = VK_NULL_HANDLE;
        cascades[i].colorMemory = VK_NULL_HANDLE;
    }

    // Do not destroy Vulkan objects here; the VulkanResourceManager owns
    // destruction. Clear local handles to avoid accidental use.
    shadowRenderPass = VK_NULL_HANDLE;
    shadowMapSampler = VK_NULL_HANDLE;
    dummyDepthView = VK_NULL_HANDLE;
    dummyDepthImage = VK_NULL_HANDLE;
    dummyDepthMemory = VK_NULL_HANDLE;
    shadowPipeline = VK_NULL_HANDLE;
    shadowPipelineLayout = VK_NULL_HANDLE;

    // staging resources handled by centralized cleanup

}

void ShadowRenderer::createShadowMaps(VulkanApp* app) {
    VkDevice device = app->getDevice();
    VkFormat colorFormat = app->getSwapchainImageFormat();

    // Create a single sampler shared by all cascades
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
    app->resources.addSampler(shadowMapSampler, "ShadowRenderer: shadowMapSampler");

    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        auto& cas = cascades[c];
        std::string tag = "ShadowRenderer cascade " + std::to_string(c);

        // --- Depth image ---
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = { shadowMapSize, shadowMapSize, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_D32_SFLOAT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &imageInfo, nullptr, &cas.depthImage) != VK_SUCCESS)
            throw std::runtime_error("failed to create shadow map depth image cascade " + std::to_string(c));
        app->resources.addImage(cas.depthImage, (tag + " depthImage").c_str());

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, cas.depthImage, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &cas.depthMemory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate shadow map memory cascade " + std::to_string(c));
        vkBindImageMemory(device, cas.depthImage, cas.depthMemory, 0);
        app->resources.addDeviceMemory(cas.depthMemory, (tag + " depthMemory").c_str());

        // Depth image view
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = cas.depthImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_D32_SFLOAT;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &viewInfo, nullptr, &cas.depthView) != VK_SUCCESS)
            throw std::runtime_error("failed to create shadow map depth view cascade " + std::to_string(c));
        app->resources.addImageView(cas.depthView, (tag + " depthView").c_str());

        // Transition depth to READ_ONLY so the render pass starts from a valid layout
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = cas.depthImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        });

        // ImGui descriptor for shadow map visualisation
        cas.imguiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            shadowMapSampler, cas.depthView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        // --- Color image (transient, unused but keeps renderpass attachment count compatible) ---
        VkImageCreateInfo colorInfo{};
        colorInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        colorInfo.imageType = VK_IMAGE_TYPE_2D;
        colorInfo.extent = { shadowMapSize, shadowMapSize, 1 };
        colorInfo.mipLevels = 1;
        colorInfo.arrayLayers = 1;
        colorInfo.format = colorFormat;
        colorInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        colorInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        colorInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &colorInfo, nullptr, &cas.colorImage) != VK_SUCCESS)
            throw std::runtime_error("failed to create shadow color image cascade " + std::to_string(c));
        app->resources.addImage(cas.colorImage, (tag + " colorImage").c_str());

        vkGetImageMemoryRequirements(device, cas.colorImage, &memReq);
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &cas.colorMemory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate shadow color memory cascade " + std::to_string(c));
        vkBindImageMemory(device, cas.colorImage, cas.colorMemory, 0);
        app->resources.addDeviceMemory(cas.colorMemory, (tag + " colorMemory").c_str());

        VkImageViewCreateInfo colorViewInfo{};
        colorViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorViewInfo.image = cas.colorImage;
        colorViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorViewInfo.format = colorFormat;
        colorViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &colorViewInfo, nullptr, &cas.colorView) != VK_SUCCESS)
            throw std::runtime_error("failed to create shadow color view cascade " + std::to_string(c));
        app->resources.addImageView(cas.colorView, (tag + " colorView").c_str());
    }

    fprintf(stderr, "[ShadowRenderer] Created %d cascade shadow maps (%ux%u each)\n",
            SHADOW_CASCADE_COUNT, shadowMapSize, shadowMapSize);
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

void ShadowRenderer::createShadowFramebuffers(VulkanApp* app) {
    VkDevice device = app->getDevice();
    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        VkImageView attachments[] = { cascades[c].colorView, cascades[c].depthView };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = shadowRenderPass;
        framebufferInfo.attachmentCount = 2;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = shadowMapSize;
        framebufferInfo.height = shadowMapSize;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &cascades[c].framebuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow framebuffer cascade " + std::to_string(c));
        }
        app->resources.addFramebuffer(cascades[c].framebuffer, ("ShadowRenderer: framebuffer cascade " + std::to_string(c)).c_str());
    }
}

void ShadowRenderer::createShadowPipeline(VulkanApp* app) {
    // Create a dedicated pipeline against the shadow render pass using the same
    // shaders as the solid pass.  This avoids render-pass incompatibility errors
    // that occur when the solid pipeline (created against solidRenderPass) is
    // used inside the shadow render pass.
    ShaderStage vertexShader(
        app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage tescShader(
        app->createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    ShaderStage teseShader(
        app->createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    ShaderStage fragmentShader(
        app->createShaderModule(FileReader::readFile("shaders/main.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getDescriptorSetLayout());

    auto [pipeline, layout] = app->createGraphicsPipeline(
        { vertexShader.info, tescShader.info, teseShader.info, fragmentShader.info },
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX }
        },
        {
            VkVertexInputAttributeDescription{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription{ 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription{ 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription{ 5, 0, VK_FORMAT_R32_SINT,         offsetof(Vertex, texIndex) }
        },
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_BACK_BIT,
        true,   // depthWrite
        false,  // colorWrite (shadow pass only needs depth)
        VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        shadowRenderPass  // <-- use the shadow render pass
    );
    shadowPipeline = pipeline;
    shadowPipelineLayout = layout;
    fprintf(stderr, "[ShadowRenderer] createShadowPipeline: pipeline=%p layout=%p\n",
            (void*)shadowPipeline, (void*)shadowPipelineLayout);

    // Clean up local shader module references (VulkanResourceManager owns them)
    vertexShader.info.module  = VK_NULL_HANDLE;
    tescShader.info.module    = VK_NULL_HANDLE;
    teseShader.info.module    = VK_NULL_HANDLE;
    fragmentShader.info.module = VK_NULL_HANDLE;

    // ── Create a tiny 1×1 D32_SFLOAT image that stays in READ_ONLY layout
    //    so the shadow descriptor set can bind it at binding 4 without a layout
    //    mismatch during the shadow pass (the real shadow map is being written).
    {
        VkDevice device = app->getDevice();

        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.extent        = { 1, 1, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.format        = VK_FORMAT_D32_SFLOAT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &imgInfo, nullptr, &dummyDepthImage) != VK_SUCCESS)
            throw std::runtime_error("failed to create dummy depth image");
        app->resources.addImage(dummyDepthImage, "ShadowRenderer: dummyDepthImage");

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, dummyDepthImage, &memReq);
        VkMemoryAllocateInfo alloc{};
        alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize  = memReq.size;
        alloc.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &alloc, nullptr, &dummyDepthMemory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate dummy depth memory");
        vkBindImageMemory(device, dummyDepthImage, dummyDepthMemory, 0);
        app->resources.addDeviceMemory(dummyDepthMemory, "ShadowRenderer: dummyDepthMemory");

        VkImageViewCreateInfo vwInfo{};
        vwInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vwInfo.image    = dummyDepthImage;
        vwInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vwInfo.format   = VK_FORMAT_D32_SFLOAT;
        vwInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &vwInfo, nullptr, &dummyDepthView) != VK_SUCCESS)
            throw std::runtime_error("failed to create dummy depth view");
        app->resources.addImageView(dummyDepthView, "ShadowRenderer: dummyDepthView");

        // Transition to DEPTH_STENCIL_READ_ONLY_OPTIMAL once at init
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = dummyDepthImage;
            b.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
            b.srcAccessMask       = 0;
            b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        });
    }
}

void ShadowRenderer::beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex, const glm::mat4& lightSpaceMatrix) {
    currentLightSpaceMatrix = lightSpaceMatrix;
    auto& cas = cascades[cascadeIndex];

    // Transition shadow map from READ_ONLY to DEPTH_STENCIL_ATTACHMENT_OPTIMAL before shadow pass
    if (cas.depthImage != VK_NULL_HANDLE) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = cas.depthImage;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    if (shadowRenderPass == VK_NULL_HANDLE || cas.framebuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[ShadowRenderer] Error: shadowRenderPass or framebuffer is VK_NULL_HANDLE for cascade %u\n", cascadeIndex);
        return;
    }

    VkRenderPassBeginInfo shadowRenderPassInfo{};
    shadowRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    shadowRenderPassInfo.renderPass = shadowRenderPass;
    shadowRenderPassInfo.framebuffer = cas.framebuffer;
    shadowRenderPassInfo.renderArea.offset = {0, 0};
    shadowRenderPassInfo.renderArea.extent = {shadowMapSize, shadowMapSize};

    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    shadowRenderPassInfo.clearValueCount = 2;
    shadowRenderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffer, &shadowRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

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

    vkCmdSetDepthBias(commandBuffer, -1.5f, 0.0f, -2.0f);

    if (shadowPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
    }
}

void ShadowRenderer::render(VulkanApp* app, VkCommandBuffer commandBuffer, 
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

void ShadowRenderer::endShadowPass(VulkanApp* /*app*/, VkCommandBuffer commandBuffer, uint32_t /*cascadeIndex*/) {
    // The render pass finalLayout (DEPTH_STENCIL_READ_ONLY_OPTIMAL) automatically
    // transitions the cascade's shadow map to the correct layout for shader sampling.
    vkCmdEndRenderPass(commandBuffer);
}

void ShadowRenderer::readbackShadowDepth(VulkanApp* app, uint32_t cascade) {
    VkImage targetImage = cascades[cascade].depthImage;
    if (targetImage == VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(shadowMapSize) * shadowMapSize * sizeof(float);

    // Create staging buffer
    Buffer stagingBuffer = app->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Record commands: transition to TRANSFER_SRC, copy image to buffer, transition back to READ_ONLY
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = targetImage;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    region.imageExtent = { shadowMapSize, shadowMapSize, 1 };

    vkCmdCopyImageToBuffer(cmd, targetImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer.buffer, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

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
    std::string filename = "shadow_depth_" + std::to_string(cascade) + ".pgm";
    std::ofstream ofs(filename, std::ios::binary);
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
