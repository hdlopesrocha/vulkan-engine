#include "ShadowRenderer.hpp"
#include "RendererUtils.hpp"

#include "../VulkanApp.hpp"
#include "../ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include "../../math/Vertex.hpp"
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
        RendererUtils::createImage2D(device, app, shadowMapSize, shadowMapSize,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            (tag + " depth").c_str(), cas.depthImage, cas.depthMemory, cas.depthView);

        // Transition depth to READ_ONLY so the render pass starts from a valid layout
        // Use the centralized helper so the authoritative layout map is updated.
        app->transitionImageLayoutLayer(cas.depthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1, 0, 1);

        // Track that this cascade's depth image is now in READ_ONLY layout
        cascadeDepthLayouts[c] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        // ImGui descriptor for shadow map visualisation
        cas.imguiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            shadowMapSampler, cas.depthView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        // --- Color image (transient, unused but keeps renderpass attachment count compatible) ---
        RendererUtils::createImage2D(device, app, shadowMapSize, shadowMapSize,
            colorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            (tag + " color").c_str(), cas.colorImage, cas.colorMemory, cas.colorView);
    }

    std::cerr << "[ShadowRenderer] Created " << SHADOW_CASCADE_COUNT
              << " cascade shadow maps (" << shadowMapSize << "x" << shadowMapSize << " each)" << std::endl;
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
    // Expect the shadow map to be in READ_ONLY when not being written.
    // Let the render pass perform the implicit transition to the
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL subpass layout when the pass begins.
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
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

VkImage ShadowRenderer::getDepthImage(uint32_t cascade) const {
    if (cascade >= SHADOW_CASCADE_COUNT) return VK_NULL_HANDLE;
    return cascades[cascade].depthImage;
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
    std::cerr << "[ShadowRenderer] createShadowPipeline: pipeline=" << (void*)shadowPipeline
              << " layout=" << (void*)shadowPipelineLayout << std::endl;

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

        RendererUtils::createImage2D(device, app, 1, 1,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            "ShadowRenderer: dummyDepth", dummyDepthImage, dummyDepthMemory, dummyDepthView);

        // Transition to DEPTH_STENCIL_READ_ONLY_OPTIMAL once at init
        app->transitionImageLayoutLayer(dummyDepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1, 0, 1);
        // Ensure authoritative tracked layout reflects the synchronous transition
        app->setImageLayoutTracked(dummyDepthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 0, 1);
    }
}

void ShadowRenderer::beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex, const glm::mat4& lightSpaceMatrix) {
    currentLightSpaceMatrix = lightSpaceMatrix;
    auto& cas = cascades[cascadeIndex];

    // No explicit barrier here: rely on the render pass implicit transition
    // to move the image into DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
    // subpass. However, record the expected tracked layout for this
    // command buffer so validation layers see the correct layout at
    // vkQueueSubmit time (we don't emit a barrier because the renderpass
    // performs it implicitly).
    if (cas.depthImage != VK_NULL_HANDLE) {
        if (app) {
            app->recordTrackedLayoutForCommandBuffer(commandBuffer, cas.depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
        }
        if (cascadeIndex < cascadeDepthLayouts.size()) cascadeDepthLayouts[cascadeIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    if (shadowRenderPass == VK_NULL_HANDLE || cas.framebuffer == VK_NULL_HANDLE) {
        std::cerr << "[ShadowRenderer] Error: shadowRenderPass or framebuffer is VK_NULL_HANDLE for cascade " << cascadeIndex << std::endl;
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

    // Bind descriptor sets: bind the provided per-instance/main descriptor set at set 0.
    // Do NOT bind the material-only descriptor set at set 0 because shadow pipelines are
    // created with only the main descriptor set layout. Binding an incompatible set
    // to set 0 causes validation errors.
    VkPipelineLayout layout = app->getPipelineLayout();
    if (shadowPipelineLayout != VK_NULL_HANDLE) layout = shadowPipelineLayout;
    if (descriptorSet != VK_NULL_HANDLE) {
        //printf("[BIND] ShadowRenderer::render: layout=%p firstSet=0 count=1 sets=%p\n", (void*)layout, (void*)descriptorSet);
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

void ShadowRenderer::endShadowPass(VulkanApp* /*app*/, VkCommandBuffer commandBuffer, uint32_t cascadeIndex) {
    // The render pass finalLayout (DEPTH_STENCIL_READ_ONLY_OPTIMAL) automatically
    // transitions the cascade's shadow map to the correct layout for shader sampling.
    vkCmdEndRenderPass(commandBuffer);
    // Update tracked layout to reflect the renderpass final layout
    if (cascadeIndex < cascadeDepthLayouts.size()) cascadeDepthLayouts[cascadeIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
}

VkImageLayout ShadowRenderer::getDepthLayout(uint32_t cascade) const {
    if (cascade >= cascadeDepthLayouts.size()) return VK_IMAGE_LAYOUT_UNDEFINED;
    return cascadeDepthLayouts[cascade];
}

void ShadowRenderer::requestWireframeReadback() {
    requestWireframeReadbackFlag = true;
}
