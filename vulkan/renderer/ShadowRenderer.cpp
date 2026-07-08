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
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

static constexpr VkFormat EVSM_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

ShadowRenderer::ShadowRenderer(uint32_t maxShadowMapSize)
    : shadowMapSizes{maxShadowMapSize, maxShadowMapSize / 2, maxShadowMapSize / 4} {}

ShadowRenderer::~ShadowRenderer() {}

VkDescriptorSetLayout ShadowRenderer::getShadowDescriptorSetLayout(VulkanApp* app) const {
    return app->getDescriptorSetLayout();
}

void ShadowRenderer::init(VulkanApp* app) {
    createShadowMaps(app);
    createShadowPipeline(app);
    createBlurResources(app);
}

void ShadowRenderer::cleanup(VulkanApp* app) {
    VkDevice device = app->getDevice();

    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        if (cascades[i].imguiDescSet != VK_NULL_HANDLE) {
            VkDescriptorSet ds = cascades[i].imguiDescSet;
            app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
            cascades[i].imguiDescSet = VK_NULL_HANDLE;
        }
        cascades[i].colorView = VK_NULL_HANDLE;
        cascades[i].colorImage = VK_NULL_HANDLE;
        cascades[i].colorAllocation = VK_NULL_HANDLE;
        cascades[i].colorMemory = VK_NULL_HANDLE;
        cascades[i].depthView = VK_NULL_HANDLE;
        cascades[i].depthImage = VK_NULL_HANDLE;
        cascades[i].depthAllocation = VK_NULL_HANDLE;
        cascades[i].depthMemory = VK_NULL_HANDLE;
    }

    shadowMapSampler = VK_NULL_HANDLE;
    dummyColorView = VK_NULL_HANDLE;
    dummyColorImage = VK_NULL_HANDLE;
    dummyColorAllocation = VK_NULL_HANDLE;
    dummyColorMemory = VK_NULL_HANDLE;
    shadowPipeline = VK_NULL_HANDLE;
    shadowPipelineLayout = VK_NULL_HANDLE;

    // Blur resources cleared; central manager handles destruction
    blurPipeline = VK_NULL_HANDLE;
    blurPipelineLayout = VK_NULL_HANDLE;
    blurDescSetLayout = VK_NULL_HANDLE;
    blurDescPool = VK_NULL_HANDLE;
    for (auto& ds : blurHorizontalDS) ds = VK_NULL_HANDLE;
    blurVerticalDS = VK_NULL_HANDLE;
    blurTempView = VK_NULL_HANDLE;
    blurTempImage = VK_NULL_HANDLE;
    blurTempAllocation = VK_NULL_HANDLE;
    blurTempMemory = VK_NULL_HANDLE;
}

void ShadowRenderer::createShadowMaps(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Sampler with LINEAR filtering for EVSM bilinear moment sampling
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
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
        throw std::runtime_error("failed to create EVSM shadow map sampler!");
    }
    app->resources.addSampler(shadowMapSampler, "ShadowRenderer: EVSM sampler");

    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        uint32_t size = shadowMapSizes[c];
        auto& cas = cascades[c];
        std::string tag = "ShadowRenderer cascade " + std::to_string(c);

        // --- EVSM color image (RGBA32F for moments) ---
        RendererUtils::createImage2DWithVma(device, app, size, size,
            EVSM_FORMAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            (tag + " EVSM color").c_str(), cas.colorImage, cas.colorAllocation, cas.colorMemory, cas.colorView);

        app->transitionImageLayoutLayer(cas.colorImage, EVSM_FORMAT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);

        // --- Depth image (for depth testing during shadow rendering) ---
        RendererUtils::createImage2DWithVma(device, app, size, size,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            (tag + " depth").c_str(), cas.depthImage, cas.depthAllocation, cas.depthMemory, cas.depthView);

        app->transitionImageLayoutLayer(cas.depthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1, 0, 1);

        cascadeDepthLayouts[c] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        // ImGui descriptor for shadow map visualisation
        cas.imguiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            shadowMapSampler, cas.colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    std::cerr << "[ShadowRenderer] Created " << SHADOW_CASCADE_COUNT
              << " cascade EVSM maps (" << shadowMapSizes[0] << "x" << shadowMapSizes[0]
              << "/" << shadowMapSizes[1] << "x" << shadowMapSizes[1]
              << "/" << shadowMapSizes[2] << "x" << shadowMapSizes[2] << ")" << std::endl;
}

VkImage ShadowRenderer::getDepthImage(uint32_t cascade) const {
    if (cascade >= SHADOW_CASCADE_COUNT) return VK_NULL_HANDLE;
    return cascades[cascade].depthImage;
}

void ShadowRenderer::createShadowPipeline(VulkanApp* app) {
    // Create an EVSM pipeline: outputs RGBA32F color moments + depth test
    ShaderStage vertexShader(
        app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage tescShader(
        app->createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    ShaderStage teseShader(
        app->createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    ShaderStage evsmFragment(
        app->createShaderModule(FileReader::readFile("shaders/shadow_evsm.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getDescriptorSetLayout());

    auto [pipeline, layout] = app->createGraphicsPipeline(
        { vertexShader.info, tescShader.info, teseShader.info, evsmFragment.info },
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX }
        },
        vk_layouts::defaultAttributes(),
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true,   // depthWrite
        true,   // colorWrite (EVSM moments)
        VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        false,
        std::vector<VkFormat>{EVSM_FORMAT},
        VK_FORMAT_D32_SFLOAT,
        false,  // noColorAttachment = false (we HAVE a color attachment)
        true    // depthBiasEnable
    );
    shadowPipeline = pipeline;
    shadowPipelineLayout = layout;
    std::cerr << "[ShadowRenderer] EVSM pipeline: " << (void*)shadowPipeline
              << " layout=" << (void*)shadowPipelineLayout << std::endl;

    vertexShader.info.module   = VK_NULL_HANDLE;
    tescShader.info.module     = VK_NULL_HANDLE;
    teseShader.info.module     = VK_NULL_HANDLE;
    evsmFragment.info.module   = VK_NULL_HANDLE;

    // ── Create a tiny 1×1 RGBA32F image + depth dummy kept in READ_ONLY layouts
    //    so the main descriptor set can bind it at 4/8/9 without a layout
    //    mismatch during the shadow pass (the real EVSM maps are being written).
    {
        VkDevice device = app->getDevice();
        RendererUtils::createImage2DWithVma(device, app, 1, 1,
            EVSM_FORMAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "ShadowRenderer: dummyColor", dummyColorImage, dummyColorAllocation, dummyColorMemory, dummyColorView);

        app->transitionImageLayoutLayer(dummyColorImage, EVSM_FORMAT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        app->setImageLayoutTracked(dummyColorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
    }
}

void ShadowRenderer::createBlurResources(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Descriptor set layout: one combined image sampler (the EVSM texture to blur)
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 1;
    dslInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &blurDescSetLayout) != VK_SUCCESS)
        throw std::runtime_error("ShadowRenderer: failed to create blur descriptor set layout");
    app->resources.addDescriptorSetLayout(blurDescSetLayout, "ShadowRenderer: blurDescSetLayout");

    // Pipeline layout
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(float); // direction: 0 = horizontal, 1 = vertical

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &blurDescSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &blurPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("ShadowRenderer: failed to create blur pipeline layout");
    app->resources.addPipelineLayout(blurPipelineLayout, "ShadowRenderer: blurPipelineLayout");

    // Fullscreen vertex + blur fragment shader
    ShaderStage vertShader(
        app->createShaderModule(FileReader::readFile("shaders/fullscreen.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragShader(
        app->createShaderModule(FileReader::readFile("shaders/evsm_blur.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT);

    RendererUtils::FullscreenPipelineOpts opts{};
    opts.colorAttachmentCount = 1;
    blurPipeline = RendererUtils::buildFullscreenPipeline(
        device, app, EVSM_FORMAT, VK_FORMAT_UNDEFINED,
        blurPipelineLayout,
        { vertShader.info, fragShader.info },
        opts, "ShadowRenderer: blurPipeline");

    vertShader.info.module = VK_NULL_HANDLE;
    fragShader.info.module = VK_NULL_HANDLE;

    // Create dedicated descriptor pool for blur (4 sets: 3 cascade h-blur + 1 v-blur)
    VkDescriptorPoolSize blurPoolSize{};
    blurPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    blurPoolSize.descriptorCount = 4;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 4;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &blurPoolSize;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &blurDescPool) != VK_SUCCESS)
        throw std::runtime_error("ShadowRenderer: failed to create blur descriptor pool");
    app->resources.addDescriptorPool(blurDescPool, "ShadowRenderer: blurDescPool");

    // Temporary image for blur ping-pong (sized for the largest cascade)
    RendererUtils::createImage2DWithVma(device, app, shadowMapSizes[0], shadowMapSizes[0],
        EVSM_FORMAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        "ShadowRenderer: blurTemp", blurTempImage, blurTempAllocation, blurTempMemory, blurTempView);

    app->transitionImageLayoutLayer(blurTempImage, EVSM_FORMAT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    app->setImageLayoutTracked(blurTempImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);

    // Allocate vertical-blur DS (always reads blurTempImage) and write it
    {
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = blurDescPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &blurDescSetLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &blurVerticalDS) != VK_SUCCESS)
            throw std::runtime_error("ShadowRenderer: failed to allocate blurVerticalDS");

        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler = shadowMapSampler;
        imgInfo.imageView = blurTempView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = blurVerticalDS;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    // Allocate one horizontal-blur DS per cascade (reads cascade color image)
    for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = blurDescPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &blurDescSetLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &blurHorizontalDS[c]) != VK_SUCCESS)
            throw std::runtime_error("ShadowRenderer: failed to allocate blurHorizontalDS");

        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler = shadowMapSampler;
        imgInfo.imageView = cascades[c].colorView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = blurHorizontalDS[c];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void ShadowRenderer::beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex, const glm::mat4& lightSpaceMatrix) {
    currentLightSpaceMatrix = lightSpaceMatrix;
    uint32_t size = shadowMapSizes[cascadeIndex];
    auto& cas = cascades[cascadeIndex];

    // Barrier: transition cascade color from SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL
    // so the shadow pipeline can write EVSM moments.  The previous pass sampled
    // this image as a texture; after the barrier it becomes a render target.
    VkImageMemoryBarrier2 beginBarriers[2]{};
    beginBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    beginBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    beginBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    beginBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    beginBarriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    beginBarriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    beginBarriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    beginBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beginBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beginBarriers[0].image = cas.colorImage;
    beginBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    // Barrier: transition cascade depth from DEPTH_STENCIL_READ_ONLY → DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    // so the shadow pipeline can perform depth testing.  The depth was left in
    // read-only layout after the previous frame's sampling.
    beginBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    beginBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    beginBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    beginBarriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    beginBarriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    beginBarriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    beginBarriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    beginBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beginBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beginBarriers[1].image = cas.depthImage;
    beginBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

    VkDependencyInfo beginDep{};
    beginDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    beginDep.imageMemoryBarrierCount = 2;
    beginDep.pImageMemoryBarriers = beginBarriers;
    vkCmdPipelineBarrier2(commandBuffer, &beginDep);

    app->setImageLayoutTracked(cas.colorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 1);
    app->setImageLayoutTracked(cas.depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
    cascadeDepthLayouts[cascadeIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Begin dynamic rendering with color + depth attachments
    VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = cas.colorView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearColor;

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
    renderingInfo.renderArea.extent = {size, size};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = (float)size;
    shadowViewport.height = (float)size;
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = {size, size};
    vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

    vkCmdSetDepthBias(commandBuffer, 1.5f, 0.0f, 2.5f);

    if (shadowPipeline != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, shadowPipeline);
        else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
    }
}

void ShadowRenderer::endShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex) {
    vkCmdEndRendering(commandBuffer);

    auto& cas = cascades[cascadeIndex];

    // Barrier: transition cascade color from COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // after the shadow pass finishes rendering EVSM moments.  The next cascade
    // (or the main scene pass) samples this as a texture.
    VkImageMemoryBarrier2 endBarriers[2]{};
    endBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    endBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    endBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    endBarriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    endBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    endBarriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    endBarriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    endBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    endBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    endBarriers[0].image = cas.colorImage;
    endBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    // Barrier: transition cascade depth from DEPTH_STENCIL_ATTACHMENT_OPTIMAL → DEPTH_STENCIL_READ_ONLY_OPTIMAL
    // so the depth can be sampled as a shadow-map texture in the main scene pass.
    endBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    endBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    endBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    endBarriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    endBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    endBarriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    endBarriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    endBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    endBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    endBarriers[1].image = cas.depthImage;
    endBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

    VkDependencyInfo endDep{};
    endDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    endDep.imageMemoryBarrierCount = 2;
    endDep.pImageMemoryBarriers = endBarriers;
    vkCmdPipelineBarrier2(commandBuffer, &endDep);

    app->setImageLayoutTracked(cas.colorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
    app->setImageLayoutTracked(cas.depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 0, 1);
    cascadeDepthLayouts[cascadeIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
}

void ShadowRenderer::blurCascade(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex) {
    uint32_t size = shadowMapSizes[cascadeIndex];
    auto& cas = cascades[cascadeIndex];

    // ── Horizontal blur: read cascade color → write to blurTemp ──
    // Transition blurTemp from SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL
    // so the horizontal blur pass can write intermediate EVSM results.
    app->recordTransitionImageLayoutLayer(commandBuffer, blurTempImage, EVSM_FORMAT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 0, 1);

    // Horizontal blur pass (reads cascade color via pre-allocated blurHorizontalDS[cascadeIndex])
    {
        VkRenderingAttachmentInfo colorAtt{};
        colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAtt.imageView = blurTempView;
        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.offset = {0, 0};
        ri.renderArea.extent = {size, size};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &colorAtt;

        vkCmdBeginRendering(commandBuffer, &ri);

        VkViewport vp{};
        vp.width = (float)size;
        vp.height = (float)size;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &vp);
        VkRect2D sc{};
        sc.extent = {size, size};
        vkCmdSetScissor(commandBuffer, 0, 1, &sc);

        if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, blurPipeline);
        else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline);
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer,
            blurPipelineLayout, 0, 1, &blurHorizontalDS[cascadeIndex], 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            blurPipelineLayout, 0, 1, &blurHorizontalDS[cascadeIndex], 0, nullptr);

        float dir = 0.0f; // horizontal
        vkCmdPushConstants(commandBuffer, blurPipelineLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &dir);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRendering(commandBuffer);
    }

    // Transition blurTemp from COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // so the vertical blur pass can sample the intermediate result.
    app->recordTransitionImageLayoutLayer(commandBuffer, blurTempImage, EVSM_FORMAT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);

    // ── Vertical blur: read blurTemp → write back to cascade color ──
    // Transition cascade color from SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL
    // so the vertical blur pass can write the final EVSM result back.
    app->recordTransitionImageLayoutLayer(commandBuffer, cas.colorImage, EVSM_FORMAT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 0, 1);

    // Vertical blur pass (reads blurTemp via pre-allocated blurVerticalDS)
    {
        VkRenderingAttachmentInfo colorAtt{};
        colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAtt.imageView = cas.colorView;
        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.offset = {0, 0};
        ri.renderArea.extent = {size, size};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &colorAtt;

        vkCmdBeginRendering(commandBuffer, &ri);
        VkViewport blurVp{0,0,(float)size,(float)size,0,1};
        vkCmdSetViewport(commandBuffer, 0, 1, &blurVp);
        VkRect2D blurSc{{0,0},{size,size}};
        vkCmdSetScissor(commandBuffer, 0, 1, &blurSc);

        if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, blurPipeline);
        else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline);
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer,
            blurPipelineLayout, 0, 1, &blurVerticalDS, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            blurPipelineLayout, 0, 1, &blurVerticalDS, 0, nullptr);

        float dir = 1.0f; // vertical
        vkCmdPushConstants(commandBuffer, blurPipelineLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &dir);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRendering(commandBuffer);
    }

    // Transition cascade color from COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // after vertical blur completes.  The next cascade (or the main scene pass)
    // samples this as a texture.  blurTemp is already in SHADER_READ_ONLY
    // (transitioned after h-blur above), so only cascade color needs a barrier.
    app->recordTransitionImageLayoutLayer(commandBuffer, cas.colorImage, EVSM_FORMAT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void ShadowRenderer::render(VulkanApp* app, VkCommandBuffer commandBuffer,
                                 const VertexBufferObject& vbo, VkDescriptorSet descriptorSet) {
    VkPipelineLayout layout = app->getPipelineLayout();
    if (shadowPipelineLayout != VK_NULL_HANDLE) layout = shadowPipelineLayout;
    if (descriptorSet != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, layout, 0, 1, &descriptorSet, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSet, 0, nullptr);
    }

    VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, vbo.indexCount, 1, 0, 0, 0);
}

VkImageLayout ShadowRenderer::getDepthLayout(uint32_t cascade) const {
    if (cascade >= cascadeDepthLayouts.size()) return VK_IMAGE_LAYOUT_UNDEFINED;
    return cascadeDepthLayouts[cascade];
}

void ShadowRenderer::setDepthLayout(uint32_t cascade, VkImageLayout layout) {
    if (cascade < cascadeDepthLayouts.size()) cascadeDepthLayouts[cascade] = layout;
}

void ShadowRenderer::freeImGuiDescriptors() {
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        if (cascades[i].imguiDescSet != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(cascades[i].imguiDescSet);
            cascades[i].imguiDescSet = VK_NULL_HANDLE;
        }
    }
}

void ShadowRenderer::recreateImGuiDescriptors() {
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        if (cascades[i].imguiDescSet != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(cascades[i].imguiDescSet);
            cascades[i].imguiDescSet = VK_NULL_HANDLE;
        }
        if (cascades[i].colorView != VK_NULL_HANDLE && shadowMapSampler != VK_NULL_HANDLE) {
            cascades[i].imguiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
                shadowMapSampler, cascades[i].colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
}

void ShadowRenderer::requestWireframeReadback() {
    requestWireframeReadbackFlag = true;
}
