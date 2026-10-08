#include "SolidRenderer.hpp"
#include "../RendererUtils.hpp"
#include <vector>

#include "../../../utils/FileReader.hpp"
#include "../../pipeline/ShaderStage.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "../../includes/shader/locations.hpp"
#include "../../includes/shader/vertex_layouts.hpp"

SolidRenderer::SolidRenderer() : indirectRenderer() {}
SolidRenderer::~SolidRenderer() { cleanup(nullptr); }

void SolidRenderer::init() {
    indirectRenderer.init();
}

void SolidRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    renderWidth = width;
    renderHeight = height;
    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "SolidRenderer: image", image, allocation, memory, view);
    };

    for (uint32_t i = 0; i < SolidRenderer::SOLID_FRAMES; ++i) {
        createImage(app->getSwapchainImageFormat(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                    solidColorImages[i], solidColorAllocations[i], solidColorMemories[i], solidColorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                    solidDepthImages[i], solidDepthAllocations[i], solidDepthMemories[i], solidDepthImageViews[i]);
        solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // Ensure created images have authoritative GPU/tracked layouts before
    // first use. Use VulkanApp helpers to perform transitions so the
    // app's layout-tracking map is updated consistently.
    for (uint32_t i = 0; i < SolidRenderer::SOLID_FRAMES; ++i) {
        // color image: transition UNDEFINED -> SHADER_READ_ONLY_OPTIMAL
        if (solidColorImages[i] != VK_NULL_HANDLE && app) {
            try {
                app->transitionImageLayoutLayer(solidColorImages[i], app->getSwapchainImageFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            } catch (...) { /* best-effort: avoid throwing during init */ }
        }
        // depth image: ensure a concrete GPU/tracked layout so early
        // sampling or other uses don't see UNDEFINED. Use a forced
        // synchronous transition to SHADER_READ_ONLY_OPTIMAL and update
        // the tracked layout so subsequent record-time barriers remain
        // correct. This is a conservative init that avoids submit-time
        // validation mismatches when other command buffers reference
        // the depth image before a renderpass has written it.
        solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (solidDepthImages[i] != VK_NULL_HANDLE && app) {
            try {
                // Force a GPU transition from UNDEFINED -> SHADER_READ_ONLY_OPTIMAL
                // and make the authoritative tracked layout reflect that state.
                app->transitionImageLayoutLayerForce(solidDepthImages[i], VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
                // Ensure the tracked map matches the forced transition
                app->setImageLayoutTracked(solidDepthImages[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
                solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            } catch (...) { /* best-effort */ }
        }
    }
}

void SolidRenderer::destroyRenderTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < SolidRenderer::SOLID_FRAMES; ++i) {
        if (solidColorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(solidColorImageViews[i]))
                vkDestroyImageView(device, solidColorImageViews[i], nullptr);
            solidColorImageViews[i] = VK_NULL_HANDLE;
        }
        app->destroyImageWithVma(solidColorImages[i], solidColorAllocations[i], solidColorMemories[i]);
        solidColorImages[i] = VK_NULL_HANDLE;
        solidColorAllocations[i] = VK_NULL_HANDLE;
        solidColorMemories[i] = VK_NULL_HANDLE;
        if (solidDepthImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(solidDepthImageViews[i]))
                vkDestroyImageView(device, solidDepthImageViews[i], nullptr);
            solidDepthImageViews[i] = VK_NULL_HANDLE;
        }
        app->destroyImageWithVma(solidDepthImages[i], solidDepthAllocations[i], solidDepthMemories[i]);
        solidDepthImages[i] = VK_NULL_HANDLE;
        solidDepthAllocations[i] = VK_NULL_HANDLE;
        solidDepthMemories[i] = VK_NULL_HANDLE;
    }
}

void SolidRenderer::createPipelines(VulkanApp* app) {
    if (!app) return;

    ShaderStage vertexShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/renderer/solid/SolidRenderer.vert.spv"),
        VK_SHADER_STAGE_VERTEX_BIT
    );

    // Hybrid RT: BOTH fragment variants are built up front — the RT one
    // (inline ray queries) and the non-RT one (sky-approximation fallback).
    // The runtime selector binds whichever matches the enabled solid RT paths
    // (reflections / local shadows), so a raster-only configuration never
    // pays the ray-query shader's register/occupancy cost.
    ShaderStage fragmentShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/renderer/solid/SolidRenderer.frag.spv"),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );
    ShaderStage fragmentShaderRt = ShaderStage(
        app->rayTracingEnabled()
            ? app->getOrCreateShaderModule("shaders/renderer/solid/SolidRendererRT.frag.spv") : VK_NULL_HANDLE,
        VK_SHADER_STAGE_FRAGMENT_BIT
    );
    // Per-op profiling variant (counters + device clock). Built only when the
    // device supports VK_KHR_shader_clock AND fragmentStoresAndAtomics (the
    // instrumentation atomically writes a storage buffer from the fragment
    // stage); the production variants above stay free of the clock capability.
    ShaderStage fragmentShaderRtProf = ShaderStage(
        (app->rayTracingEnabled() && app->rtProfilingSupported)
            ? app->getOrCreateShaderModule("shaders/renderer/solid/SolidRendererRTProf.frag.spv") : VK_NULL_HANDLE,
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    ShaderStage tescShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/renderer/solid/SolidRenderer.tesc.spv"),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    );
    ShaderStage teseShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/renderer/solid/SolidRenderer.tese.spv"),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    );

    // Descriptor set layouts: main textures (set 0) then brush SDF params (set 1)
    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE) setLayouts.push_back(app->getDescriptorSetLayout());
    // SolidRenderer.frag evaluates the brush SDF for PAINT/REMOVE mode at
    // set=1 (separate from the main set so the shadow pass — which uses set=0
    // only — doesn't require it).
    if (app->getBrushParamsDescriptorSetLayout() != VK_NULL_HANDLE) setLayouts.push_back(app->getBrushParamsDescriptorSetLayout());

    // No per-mesh model push-constants are used anymore (models are identity in shaders).
    GraphicsPipelineConfig cfg{};
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    auto [pipeline, layout] = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        vk_layouts::defaultAttributes(),
        setLayouts,
        nullptr,
        cfg
    );
    graphicsPipeline = pipeline;
    graphicsPipelineLayout = layout;

    // RT variant of the main solid pipeline (same config; reuses
    // graphicsPipelineLayout so no duplicate layout is created/discarded).
    if (fragmentShaderRt.info.module != VK_NULL_HANDLE) {
        graphicsPipelineRt = app->createGraphicsPipelineWithLayout(
            {
                vertexShader.info,
                tescShader.info,
                teseShader.info,
                fragmentShaderRt.info
            },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            cfg,
            graphicsPipelineLayout);
    }

    // Profiling variant of the main solid pipeline (same config; reuses
    // graphicsPipelineLayout).
    if (fragmentShaderRtProf.info.module != VK_NULL_HANDLE) {
        graphicsPipelineRtProf = app->createGraphicsPipelineWithLayout(
            {
                vertexShader.info,
                tescShader.info,
                teseShader.info,
                fragmentShaderRtProf.info
            },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts,
            nullptr,
            cfg,
            graphicsPipelineLayout);
    }

    GraphicsPipelineConfig depthCfg{};
    depthCfg.colorWrite = false;
    depthCfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    auto [depthPipeline, depthLayout] = app->createGraphicsPipeline(
        {
            vertexShader.info,
            tescShader.info,
            teseShader.info,
            fragmentShader.info
        },
        std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
        vk_layouts::defaultAttributes(),
        setLayouts,
        nullptr,
        depthCfg
    );
    depthPrePassPipeline = depthPipeline;
    depthPrePassPipelineLayout = depthLayout;

    // Deferred depth test pipelines: depth-only (no color) and color-only (no depth write, LESS_OR_EQUAL)
    {
        // Depth-only: lightweight DepthOnly.frag, no color attachment
        ShaderStage depthFrag = ShaderStage(
            app->getOrCreateShaderModule("shaders/renderer/DepthOnly.frag.spv"),
            VK_SHADER_STAGE_FRAGMENT_BIT
        );
        GraphicsPipelineConfig ddCfg{};
        ddCfg.colorWrite = false;
        ddCfg.depthCompareOp = VK_COMPARE_OP_LESS;
        ddCfg.noColorAttachment = true;
        auto [dp, dl] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, depthFrag.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts, nullptr,
            ddCfg
        );
        deferredDepthPipeline = dp;
        deferredDepthPipelineLayout = dl;
        // No-tessellation twin of the deferred depth pipeline (C1): same
        // DepthOnly.frag, TRIANGLE_LIST + SOLID_NO_TESS VS, so the prepass
        // skips TCS/TES and the TES displacement sampling entirely.
        // H7: filtered attributes (no ATTR_COLOR — the VS no longer reads it).
        {
            ShaderStage noTessVertexShader = ShaderStage(
                app->getOrCreateShaderModule("shaders/renderer/solid/SolidRendererNoTess.vert.spv"),
                VK_SHADER_STAGE_VERTEX_BIT
            );
            deferredDepthPipelineNoTess = app->createGraphicsPipelineWithLayout(
                { noTessVertexShader.info, depthFrag.info },
                std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
                vk_layouts::defaultAttributesFiltered(
                    { ATTR_POS, ATTR_UV, ATTR_NORMAL, ATTR_BRUSH_INDEX, ATTR_HSV }),
                setLayouts, nullptr,
                ddCfg,
                deferredDepthPipelineLayout);
            noTessVertexShader.info.module = VK_NULL_HANDLE;
        }
        depthFrag.info.module = VK_NULL_HANDLE;
    }
    {
        // Color-only: full SolidRenderer.frag, LESS_OR_EQUAL compare, no depth write
        GraphicsPipelineConfig dcCfg{};
        dcCfg.depthWriteEnable = false;
        dcCfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        dcCfg.colorFormats = { app->getSwapchainImageFormat() };
        auto [cp, cl] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, fragmentShader.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts, nullptr,
            dcCfg
        );
        deferredColorPipeline = cp;
        deferredColorPipelineLayout = cl;

        // RT variant of the deferred/forward color pipeline (same config;
        // reuses deferredColorPipelineLayout).
        if (fragmentShaderRt.info.module != VK_NULL_HANDLE) {
            deferredColorPipelineRt = app->createGraphicsPipelineWithLayout(
                { vertexShader.info, tescShader.info, teseShader.info, fragmentShaderRt.info },
                std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
                vk_layouts::defaultAttributes(),
                setLayouts, nullptr,
                dcCfg,
                deferredColorPipelineLayout);
        }
        // Profiling variant of the deferred/forward color pipeline (reuses
        // deferredColorPipelineLayout).
        if (fragmentShaderRtProf.info.module != VK_NULL_HANDLE) {
            deferredColorPipelineRtProf = app->createGraphicsPipelineWithLayout(
                { vertexShader.info, tescShader.info, teseShader.info, fragmentShaderRtProf.info },
                std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
                vk_layouts::defaultAttributes(),
                setLayouts, nullptr,
                dcCfg,
                deferredColorPipelineLayout);
        }
        // Depth-write twin for the gated single-pass path (perf report 21
        // C2): same SolidRenderer.frag and TCS/TES stages, depth write on with LESS
        // compare (standard forward semantics over the cleared depth target).
        {
            GraphicsPipelineConfig dwCfg = dcCfg;
            dwCfg.depthWriteEnable = true;
            dwCfg.depthCompareOp = VK_COMPARE_OP_LESS;
            deferredColorPipelineDepthWrite = app->createGraphicsPipelineWithLayout(
                { vertexShader.info, tescShader.info, teseShader.info, fragmentShader.info },
                std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
                vk_layouts::defaultAttributes(),
                setLayouts, nullptr,
                dwCfg,
                deferredColorPipelineLayout);
        }
    }
    deferredPipelinesCreated = true;

    // Clear local shader module references; destruction handled by VulkanResourceManager
    teseShader.info.module = VK_NULL_HANDLE;
    tescShader.info.module = VK_NULL_HANDLE;
    fragmentShader.info.module = VK_NULL_HANDLE;
    fragmentShaderRt.info.module = VK_NULL_HANDLE;
    vertexShader.info.module = VK_NULL_HANDLE;
}

void SolidRenderer::drawDepth(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet descSet) {
    if (!appArg || activeDeferredDepthPipeline() == VK_NULL_HANDLE) return;
    if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, activeDeferredDepthPipeline());
    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, activeDeferredDepthPipeline());
    if (descSet != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, deferredDepthPipelineLayout, 0, 1, &descSet, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredDepthPipelineLayout, 0, 1, &descSet, 0, nullptr);
    }
    indirectRenderer.drawPrepared(commandBuffer);
}

void SolidRenderer::drawColor(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet descSet, VkDescriptorSet brushParamsSet) {
    if (!appArg || activeDeferredColorPipeline() == VK_NULL_HANDLE) return;
    if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, activeDeferredColorPipeline());
    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, activeDeferredColorPipeline());
    if (descSet != VK_NULL_HANDLE) {
        VkDescriptorSet bindSets[2] = { descSet, brushParamsSet };
        uint32_t bindCount = (brushParamsSet != VK_NULL_HANDLE) ? 2 : 1;
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, deferredColorPipelineLayout, 0, bindCount, bindSets, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredColorPipelineLayout, 0, bindCount, bindSets, 0, nullptr);
    }
    indirectRenderer.drawPrepared(commandBuffer);
}

void SolidRenderer::cleanup(VulkanApp* app) {
    if (app == nullptr) return;
    wireframe.cleanup(app);
    destroyRenderTargets(app);
    deferredPipelinesCreated = false;

    indirectRenderer.cleanup(app);
}

void SolidRenderer::createWireframe(VulkanApp* app) {
    std::vector<VkDescriptorSetLayout> solidSetLayouts = { app->getDescriptorSetLayout() };
    wireframe.createPipeline(app, {app->getSwapchainImageFormat()},
        solidSetLayouts,
        "shaders/renderer/solid/SolidRenderer.vert.spv", "shaders/renderer/solid/SolidRendererWireframe.frag.spv",
        "shaders/renderer/solid/SolidRenderer.tesc.spv", "shaders/renderer/solid/SolidRenderer.tese.spv",
        "solid wireframe");
}

void SolidRenderer::drawWireframeOverlay(VkCommandBuffer& commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet) {
    wireframe.render(commandBuffer, app, {perTextureDescriptorSet}, getIndirectRenderer());
}
