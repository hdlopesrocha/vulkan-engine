#include "ShadowRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "SolidRenderer.hpp"
#include "WaterRenderer.hpp"
#include "VegetationRenderer.hpp"
#include "BrushRenderer.hpp"

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

static constexpr VkFormat EVSM_FORMAT = VK_FORMAT_R32G32_SFLOAT; // EVSM2: 2 moments (RG32F)

ShadowRenderer::ShadowRenderer(uint32_t maxShadowMapSize)
    : shadowMapSizes{maxShadowMapSize, maxShadowMapSize / 2, maxShadowMapSize / 4} {}

ShadowRenderer::~ShadowRenderer() {}

void ShadowRenderer::setSceneRenderers(SolidRenderer* solid, WaterRenderer* liquid,
                                       VegetationRenderer* vegetation, BrushRenderer* brush) {
    solidRenderer_ = solid;
    liquidRenderer_ = liquid;
    vegetationRenderer_ = vegetation;
    brushRenderer_ = brush;
}

void ShadowRenderer::createStagingBuffers(VulkanApp* app, size_t frameCount) {
    // Per-frame staging buffers for GPU-timeline UBO uploads: SHADOW_CASCADE_COUNT
    // cascade slots + 1 restore slot for the main UBO.
    const VkDeviceSize stagingSize = sizeof(UniformObject) * (SHADOW_CASCADE_COUNT + 1);
    destroyStagingBuffers();
    uboStagingBuffers_.resize(frameCount);
    for (size_t i = 0; i < frameCount; ++i) {
        uboStagingBuffers_[i] = app->createBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

void ShadowRenderer::destroyStagingBuffers() {
    // Local CPU-side handles are cleared; Vulkan objects are destroyed via
    // VulkanResourceManager.
    for (auto& b : uboStagingBuffers_) {
        if (b.buffer != VK_NULL_HANDLE) b = {};
    }
    uboStagingBuffers_.clear();
}

void ShadowRenderer::init(VulkanApp* app) {
    createShadowMaps(app);
    createShadowPipeline(app);
    createBlurResources(app);
}

void ShadowRenderer::cleanup(VulkanApp* app) {

    destroyStagingBuffers();

    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        if (cascades[i].imguiDescSet != VK_NULL_HANDLE) {
            VkDescriptorSet ds = cascades[i].imguiDescSet;
            app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
            cascades[i].imguiDescSet = VK_NULL_HANDLE;
        }
    }
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
    shadowMapSampler = app->createSampler(samplerInfo, "ShadowRenderer: EVSM sampler");

    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        uint32_t size = shadowMapSizes[c];
        auto& cas = cascades[c];
        std::string tag = "ShadowRenderer cascade " + std::to_string(c);

        // --- EVSM color image (RG32F for EVSM2 moments) ---
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
}

VkImage ShadowRenderer::getDepthImage(uint32_t cascade) const {
    if (cascade >= SHADOW_CASCADE_COUNT) return VK_NULL_HANDLE;
    return cascades[cascade].depthImage;
}

void ShadowRenderer::createShadowPipeline(VulkanApp* app) {
    // Create an EVSM2 pipeline: outputs RG32F color moments + depth test
    ShaderStage vertexShader(
        app->getOrCreateShaderModule("shaders/main.vert.spv"),
        VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage tescShader(
        app->getOrCreateShaderModule("shaders/main.tesc.spv"),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    ShaderStage teseShader(
        app->getOrCreateShaderModule("shaders/main.tese.spv"),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    ShaderStage evsmFragment(
        app->getOrCreateShaderModule("shaders/shadow_evsm.frag.spv"),
        VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
        setLayouts.push_back(app->getDescriptorSetLayout());

    GraphicsPipelineConfig cfg{};
    cfg.cullMode = VK_CULL_MODE_FRONT_BIT;
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.colorFormats = { EVSM_FORMAT };
    cfg.depthBiasEnable = true;
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { vertexShader.info, tescShader.info, teseShader.info, evsmFragment.info },
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX }
        },
        vk_layouts::defaultAttributes(),
        setLayouts,
        nullptr,
        cfg
    );
    shadowPipeline = pipeline;
    shadowPipelineLayout = layout;
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

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    DescriptorAllocator descAlloc{device, app};
    blurDescSetLayout = descAlloc.createLayout(
        &binding, 1,
        0, nullptr,
        "ShadowRenderer: blurDescSetLayout");

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
        app->getOrCreateShaderModule("shaders/fullscreen.vert.spv"),
        VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragShader(
        app->getOrCreateShaderModule("shaders/evsm_blur.frag.spv"),
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

    VkDescriptorPoolSize srPoolSize = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
    blurDescPool = descAlloc.createPool(
        &srPoolSize, 1, 4, 0,
        "ShadowRenderer: blurDescPool");

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

        DescriptorWriter(device)
            .writeImage(blurVerticalDS, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        shadowMapSampler, blurTempView,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .flush();
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

        DescriptorWriter(device)
            .writeImage(blurHorizontalDS[c], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        shadowMapSampler, cascades[c].colorView,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .flush();
    }
}

void ShadowRenderer::beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex, const glm::mat4& lightSpaceMatrix) {
    uint32_t size = shadowMapSizes[cascadeIndex];
    auto& cas = cascades[cascadeIndex];

    // Barrier: transition cascade color from SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL
    // so the shadow pipeline can write EVSM moments.  The previous pass sampled
    // this image as a texture; after the barrier it becomes a render target.
    // The blur pass that last wrote this image uses only fragment shaders, so
    // COMPUTE_SHADER_BIT is not needed in the source mask.
    VkImageMemoryBarrier2 beginBarriers[2]{};
    beginBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    beginBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
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
    // after the shadow pass finishes rendering EVSM moments.  The next consumer
    // is the EVSM blur pass (fragment shader only), followed by main-scene
    // fragment sampling.  VERTEX/TESSELLATION/GEOMETRY/COMPUTE are not needed.
    VkImageMemoryBarrier2 endBarriers[2]{};
    endBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    endBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    endBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
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

void ShadowRenderer::render(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t frameIdx,
                                      Buffer& mainUniformBuffer, const UniformObject& uboStatic,
                                      bool shadowsEnabled, bool renderSolid, bool vegetationEnabled,
                                      bool shadowTessellationEnabled, float lodBias,
                                      const glm::vec3& cameraPos) {
    if (commandBuffer == VK_NULL_HANDLE) return;
    if (!shadowsEnabled) return;

    // Render each cascade: upload light-space UBO, draw scene, restore UBO
    const glm::mat4 cascadeMatrices[SHADOW_CASCADE_COUNT] = {
        uboStatic.lightSpaceMatrix,
        uboStatic.lightSpaceMatrix1,
        uboStatic.lightSpaceMatrix2
    };

    // Single cascade-aware GPU culling pass: culls all chunks against all 3
    // cascade frustums simultaneously.  Each cascade independently receives
    // every chunk visible in its frustum — no exclusion between cascades —
    // so the fragment shader's per-cascade sampling always finds the geometry
    // it needs. The cascade cull reads the per-chunk LoD selection the main
    // pass stamped into the shared visibleLods buffer (single source of truth),
    // so shadow draws use the exact same LoD as the main pass.
    if (solidRenderer_)
        solidRenderer_->getIndirectRenderer().prepareCullCascades(commandBuffer, cascadeMatrices, cameraPos, lodBias);
    // Water shadows share the same LoD sync: the water cascade cull reads the
    // water main pass's visibleLods (the water prepareCull ran before this
    // shadow pass, so the selection is fresh for the current frame).
    if (liquidRenderer_) {
        liquidRenderer_->getIndirectRenderer().prepareCullCascades(commandBuffer, cascadeMatrices, cameraPos, lodBias);
    }

    // Acquire vegetation instance/indirect buffers before dynamic rendering
    if (vegetationEnabled && vegetationRenderer_) {
        vegetationRenderer_->recordReadBarriers(commandBuffer);
        vegetationRenderer_->prepareCullCascades(commandBuffer, cascadeMatrices);
    }

    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        glm::mat4 lsMatrix = cascadeMatrices[c];

        // Upload a shadow-specific UBO: viewProjection = cascade lightSpaceMatrix.
        // passParams.x MUST be 0 so the TES computes fragPosWorld (needed by the EVSM
        // fragment shader to produce correct moments).  With passParams.x=1 the TES
        // outputs fragPosWorld = vec3(0) → EVSM gets garbage depth → no shadows.
        UniformObject shadowUBO = uboStatic;
        shadowUBO.viewProjection = lsMatrix;
        shadowUBO.passParams.x = 0.0f;
        shadowUBO.passParams.y = shadowTessellationEnabled ? 1.0f : 0.0f;

        // Wait for previous cascade draws to finish reading the UBO
        // before overwriting it via vkCmdCopyBuffer.
        {
            VkBufferMemoryBarrier2 preBarrier{};
            preBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            preBarrier.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
            preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preBarrier.buffer = mainUniformBuffer.buffer;
            preBarrier.offset = 0;
            preBarrier.size = VK_WHOLE_SIZE;
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &preBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        }

        // Upload shadow UBO via vkCmdCopyBuffer from persistently mapped staging
        // buffer (avoids vkCmdUpdateBuffer's implicit FULL_QUEUE barrier).
        VkDeviceSize stagingOff = static_cast<VkDeviceSize>(c) * sizeof(UniformObject);
        if (frameIdx < uboStagingBuffers_.size()) {
            memcpy(uboStagingBuffers_[frameIdx].map(stagingOff), &shadowUBO, sizeof(UniformObject));
            VkBufferCopy copy{ stagingOff, 0, sizeof(UniformObject) };
            vkCmdCopyBuffer(commandBuffer, uboStagingBuffers_[frameIdx].buffer, mainUniformBuffer.buffer, 1, &copy);
        }
        {
            VkBufferMemoryBarrier2 memBarrier{};
            memBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
            memBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            memBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            memBarrier.buffer = mainUniformBuffer.buffer;
            memBarrier.offset = 0;
            memBarrier.size = VK_WHOLE_SIZE;
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &memBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        }

        // Cascade-specific draw (no per-cascade cull — already handled above)
        beginShadowPass(app, commandBuffer, c, lsMatrix);

        // Bind shadow descriptor set (uses dummy depth at bindings 4,8,9)
        VkPipelineLayout layout = getShadowPipelineLayout();
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!shadowDescriptorSets_.empty()) {
            uint32_t idx = frameIdx % static_cast<uint32_t>(shadowDescriptorSets_.size());
            ds = shadowDescriptorSets_[idx];
        }
        if (layout != VK_NULL_HANDLE && ds != VK_NULL_HANDLE) {
            if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, layout, 0, 1, &ds, 0, nullptr);
            else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &ds, 0, nullptr);
        }

        // Bind the EVSM shadow pipeline (shared by the solid and water depth
        // draws; both use the same Vertex format and indexed-indirect draws).
        VkPipeline solidShadowPipeline = getShadowPipeline();
        if (solidShadowPipeline != VK_NULL_HANDLE) {
            if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, solidShadowPipeline);
            else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, solidShadowPipeline);
        }

        // Draw solid geometry into shadow map (can be toggled off to isolate
        // vegetation shadows for debugging).
        if (renderSolid && solidRenderer_) {
            auto& shadowIR = solidRenderer_->getIndirectRenderer();
            shadowIR.bindBuffers(commandBuffer);
            shadowIR.drawCascadeOnly(commandBuffer, c);
        }

        // Draw water geometry into the shadow map so water casts shadows at the
        // same LoD as the main pass (the water cascade cull read the shared
        // visibleLods selection). Reuses the same EVSM shadow pipeline.
        if (liquidRenderer_) {
            auto& waterShadowIR = liquidRenderer_->getIndirectRenderer();
            waterShadowIR.bindBuffers(commandBuffer);
            waterShadowIR.drawCascadeOnly(commandBuffer, c);
        }

        // Vegetation shadow pass: drawn after solid so its 2-buffer vertex
        // bindings don't leak into the solid draw. Uses cascade-aware culling
        // (prepareCullCascades dispatched above).
        if (vegetationEnabled && vegetationRenderer_) {
            const glm::vec3 camPos = glm::vec3(uboStatic.viewPos);
            vegetationRenderer_->drawShadowCascade(app, commandBuffer, ds, camPos, c);
        }

        endShadowPass(app, commandBuffer, c);

        // Apply separable Gaussian blur (EVSM moment filtering) to reduce noise.
        // Skip the smallest cascade: at 512x512 the 3-tap blur is barely visible
        // and skipping it saves two fullscreen draws plus four layout transitions.
        if (c < SHADOW_CASCADE_COUNT - 1) {
            blurCascade(app, commandBuffer, c);
        }
    }

    // Restore GPU culling for the main camera frustum (was overwritten by
    // per-cascade prepareCull calls above) so drawPrepared in the main pass
    // uses the correct visible set.
    if (solidRenderer_)
        solidRenderer_->getIndirectRenderer().prepareCull(commandBuffer, uboStatic.viewProjection);
    if (brushRenderer_) {
        brushRenderer_->getSolidIR().prepareCull(commandBuffer, uboStatic.viewProjection);
    }

    // Restore the main UBO so subsequent passes see the original data.
    // Wait for all shadow cascade draws to finish reading the UBO first.
    {
        VkBufferMemoryBarrier2 preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        preBarrier.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.buffer = mainUniformBuffer.buffer;
        preBarrier.offset = 0;
        preBarrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }

    // Restore main UBO via vkCmdCopyBuffer from persistently mapped staging buffer.
    VkDeviceSize restoreOff = static_cast<VkDeviceSize>(SHADOW_CASCADE_COUNT) * sizeof(UniformObject);
    if (frameIdx < uboStagingBuffers_.size()) {
        memcpy(uboStagingBuffers_[frameIdx].map(restoreOff), &uboStatic, sizeof(UniformObject));
        VkBufferCopy copy{ restoreOff, 0, sizeof(UniformObject) };
        vkCmdCopyBuffer(commandBuffer, uboStagingBuffers_[frameIdx].buffer, mainUniformBuffer.buffer, 1, &copy);
    }
    {
        VkBufferMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        memBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memBarrier.buffer = mainUniformBuffer.buffer;
        memBarrier.offset = 0;
        memBarrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }
}
