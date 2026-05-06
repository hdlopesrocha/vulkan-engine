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
        cascades[i].depthView = VK_NULL_HANDLE;
        cascades[i].depthImage = VK_NULL_HANDLE;
        cascades[i].depthMemory = VK_NULL_HANDLE;
    }

    // Do not destroy Vulkan objects here; the VulkanResourceManager owns
    // destruction. Clear local handles to avoid accidental use.
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
    }

    std::cerr << "[ShadowRenderer] Created " << SHADOW_CASCADE_COUNT
              << " cascade shadow maps (" << shadowMapSize << "x" << shadowMapSize << " each)" << std::endl;
}

VkImage ShadowRenderer::getDepthImage(uint32_t cascade) const {
    if (cascade >= SHADOW_CASCADE_COUNT) return VK_NULL_HANDLE;
    return cascades[cascade].depthImage;
}

void ShadowRenderer::createShadowPipeline(VulkanApp* app) {
    // Create a depth-only dynamic rendering pipeline for shadow passes.
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
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        true   // noColorAttachment: depth-only dynamic rendering
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

    // Transition depth: DEPTH_STENCIL_READ_ONLY_OPTIMAL → DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    if (cas.depthImage != VK_NULL_HANDLE) {
        VkImageLayout oldLayout = (cascadeIndex < cascadeDepthLayouts.size())
            ? cascadeDepthLayouts[cascadeIndex]
            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        if (oldLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            VkImageMemoryBarrier depthBarrier{};
            depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            depthBarrier.oldLayout = oldLayout;
            depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.image = cas.depthImage;
            depthBarrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
            depthBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            depthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0, 0, nullptr, 0, nullptr, 1, &depthBarrier);
            if (app) app->setImageLayoutTracked(cas.depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
        }
        if (cascadeIndex < cascadeDepthLayouts.size())
            cascadeDepthLayouts[cascadeIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = cas.depthView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {shadowMapSize, shadowMapSize};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pColorAttachments = nullptr;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

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

void ShadowRenderer::endShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex) {
    vkCmdEndRendering(commandBuffer);

    // Transition depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL → DEPTH_STENCIL_READ_ONLY_OPTIMAL
    auto& cas = cascades[cascadeIndex];
    if (cas.depthImage != VK_NULL_HANDLE) {
        VkImageMemoryBarrier depthBarrier{};
        depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depthBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.image = cas.depthImage;
        depthBarrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        depthBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        depthBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &depthBarrier);
        if (app) app->setImageLayoutTracked(cas.depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 0, 1);
        if (cascadeIndex < cascadeDepthLayouts.size())
            cascadeDepthLayouts[cascadeIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    }
}

VkImageLayout ShadowRenderer::getDepthLayout(uint32_t cascade) const {
    if (cascade >= cascadeDepthLayouts.size()) return VK_IMAGE_LAYOUT_UNDEFINED;
    return cascadeDepthLayouts[cascade];
}

void ShadowRenderer::setDepthLayout(uint32_t cascade, VkImageLayout layout) {
    if (cascade < cascadeDepthLayouts.size()) cascadeDepthLayouts[cascade] = layout;
}

void ShadowRenderer::recreateImGuiDescriptors() {
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        if (cascades[i].imguiDescSet != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(cascades[i].imguiDescSet);
            cascades[i].imguiDescSet = VK_NULL_HANDLE;
        }
        if (cascades[i].depthView != VK_NULL_HANDLE && shadowMapSampler != VK_NULL_HANDLE) {
            cascades[i].imguiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
                shadowMapSampler, cascades[i].depthView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        }
    }
}

void ShadowRenderer::requestWireframeReadback() {
    requestWireframeReadbackFlag = true;
}
