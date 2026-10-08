#include "BrushSdfRenderer.hpp"
#include "../RendererUtils.hpp"
#include "../DescriptorWriter.hpp"
#include "../CommandBufferState.hpp"
#include "../../pipeline/ShaderStage.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

BrushSdfRenderer::BrushSdfRenderer() {}
BrushSdfRenderer::~BrushSdfRenderer() {}

void BrushSdfRenderer::init(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;

    createRenderTargets(app, width, height);

    // Per-frame params UBO + descriptor set (set=1 binding=0). The buffer is
    // host-visible and rewritten in place every frame (same pattern as the
    // main scene UBO) — no vkUpdateDescriptorSets in steady state.
    VkDescriptorSetLayout setLayout = app->getBrushParamsDescriptorSetLayout();
    for (uint32_t fi = 0; fi < BRUSH_FRAMES; ++fi) {
        paramsBuffers[fi] = app->createBuffer(
            sizeof(BrushSdfUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (setLayout != VK_NULL_HANDLE) {
            paramsSets[fi] = app->createDescriptorSet(setLayout);
            if (paramsSets[fi] != VK_NULL_HANDLE && paramsBuffers[fi].buffer != VK_NULL_HANDLE) {
                DescriptorWriter(app->getDevice())
                    .writeBuffer(paramsSets[fi], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                 paramsBuffers[fi].buffer, 0, sizeof(BrushSdfUBO))
                    .flush();
            }
        }
    }

    // Fullscreen raymarch pipeline. The brush pass owns its depth target
    // (front-surface hit) and writes gl_FragDepth for the composite's
    // obstacle test; hits are opaque, misses discard.
    try {
        ShaderStage vertStage(
            app->getOrCreateShaderModule("shaders/renderer/Fullscreen.vert.spv"),
            VK_SHADER_STAGE_VERTEX_BIT);
        ShaderStage fragStage(
            app->getOrCreateShaderModule("shaders/renderer/brush/BrushSdf.frag.spv"),
            VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayout> setLayouts;
        if (app->getDescriptorSetLayout() != VK_NULL_HANDLE)
            setLayouts.push_back(app->getDescriptorSetLayout());
        if (setLayout != VK_NULL_HANDLE)
            setLayouts.push_back(setLayout);

        GraphicsPipelineConfig cfg{};
        cfg.cullMode = VK_CULL_MODE_NONE;
        cfg.depthTestEnable = true;
        cfg.depthWriteEnable = true;
        cfg.depthCompareOp = VK_COMPARE_OP_LESS;
        cfg.colorFormats = { app->getSwapchainImageFormat() };
        cfg.depthFormat = VK_FORMAT_D32_SFLOAT;
        auto [pipe, layout] = app->createGraphicsPipeline(
            { vertStage.info, fragStage.info },
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            setLayouts, nullptr, cfg);
        pipeline = pipe;
        pipelineLayout = layout;
        vertStage.info.module = VK_NULL_HANDLE;
        fragStage.info.module = VK_NULL_HANDLE;
    } catch (const std::exception& e) {
        std::cerr << "[BrushSdfRenderer] shaders not available (" << e.what()
                  << "); brush preview disabled." << std::endl;
    }
}

void BrushSdfRenderer::cleanup(VulkanApp* app) {
    // Pipelines, layouts and buffers are registered with the app resource
    // manager at creation and destroyed by it at shutdown (same contract as
    // the other renderers). Only release the offscreen targets here, like the
    // old brush renderer did.
    destroyRenderTargets(app);
}

void BrushSdfRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    width = (width == 0) ? 1u : width;
    height = (height == 0) ? 1u : height;
    VkDevice device = app->getDevice();
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkImageView& view) {
        VkDeviceMemory dummyMem = VK_NULL_HANDLE;
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "BrushSdfRenderer: brush", image, allocation, dummyMem, view);
    };
    VkFormat colorFormat = app->getSwapchainImageFormat();
    for (uint32_t i = 0; i < BRUSH_FRAMES; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    colorImages[i], colorAllocations[i], colorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    depthImages[i], depthAllocations[i], depthImageViews[i]);
    }
    // Fresh images start UNDEFINED; the first barrier transitions explicitly.
    colorLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    depthLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
}

void BrushSdfRenderer::destroyRenderTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < BRUSH_FRAMES; ++i) {
        if (colorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(colorImageViews[i]))
                vkDestroyImageView(device, colorImageViews[i], nullptr);
            colorImageViews[i] = VK_NULL_HANDLE;
        }
        if (colorImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(colorImages[i], colorAllocations[i], VK_NULL_HANDLE);
            colorImages[i] = VK_NULL_HANDLE;
            colorAllocations[i] = VK_NULL_HANDLE;
        }
        if (depthImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(depthImageViews[i]))
                vkDestroyImageView(device, depthImageViews[i], nullptr);
            depthImageViews[i] = VK_NULL_HANDLE;
        }
        if (depthImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(depthImages[i], depthAllocations[i], VK_NULL_HANDLE);
            depthImages[i] = VK_NULL_HANDLE;
            depthAllocations[i] = VK_NULL_HANDLE;
        }
    }
    colorLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    depthLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
}

void BrushSdfRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    destroyRenderTargets(app);
    createRenderTargets(app, width, height);
}

void BrushSdfRenderer::updateParams(uint32_t frameIndex, const BrushEntry& entry,
                                    const glm::vec3& sweepStart,
                                    const glm::vec3& boundsCenter, float boundsRadius,
                                    uint32_t width, uint32_t height) {
    const uint32_t fi = frameIndex % BRUSH_FRAMES;
    if (paramsBuffers[fi].mappedData == nullptr) return;

    BrushSdfUBO u{};
    u.translate = glm::vec4(entry.translate, 1.0f);
    u.scale = glm::vec4(entry.scale, 0.0f);
    u.rotation = glm::vec4(entry.rot.x, entry.rot.y, entry.rot.z, entry.rot.w);

    // Primitive parameter packing, matching the CPU constructor arguments
    // (see sdf/BrushSdfFactory-style conventions in the *DistanceFunction
    // classes; documented in shaders/includes/brush/BrushSdf.glsl).
    switch (entry.sdfType) {
        case 2:  // Capsule
            u.params0 = glm::vec4(entry.capsuleA, entry.capsuleRadius);
            u.params1 = glm::vec4(entry.capsuleB, 0.0f);
            break;
        case 5:  // Torus
            u.params0 = glm::vec4(entry.torusRadii, 0.0f, 0.0f);
            break;
        case 8:  // Tapered cylinder
            u.params0 = glm::vec4(entry.taperedCylinderRadii, 0.0f, 0.0f);
            break;
        case 9:  // Tapered capsule
            u.params0 = glm::vec4(entry.capsuleA, entry.taperedCapsuleRadii.x);
            u.params1 = glm::vec4(entry.capsuleB, entry.taperedCapsuleRadii.y);
            break;
        default:
            break;
    }

    u.effect0 = glm::vec4(entry.effectAmplitude, entry.effectFrequency,
                          entry.effectThreshold, entry.effectCellSize);
    u.effect1 = glm::vec4(entry.effectBrightness, entry.effectContrast, 0.0f, 0.0f);
    u.sweepStart = glm::vec4(sweepStart, 1.0f);
    u.bounds = glm::vec4(boundsCenter, boundsRadius);
    u.flags = glm::uvec4(
        static_cast<uint32_t>(std::clamp(entry.sdfType, 0, 9)),
        entry.useEffect ? static_cast<uint32_t>(std::clamp(entry.effectType, 0, 3)) + 1u : 0u,
        static_cast<uint32_t>(entry.materialIndex < 0 ? 0 : entry.materialIndex),
        entry.sweepMode ? 1u : 0u);
    u.hsv = glm::vec4(entry.hsv, 1.0f);
    u.viewport = glm::vec4(static_cast<float>(width), static_cast<float>(height), 0.0f, 0.0f);

    memcpy(paramsBuffers[fi].mappedData, &u, sizeof(u));
}

void BrushSdfRenderer::recordPass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                  VkDescriptorSet mainDs) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    const uint32_t fi = frameIndex % BRUSH_FRAMES;
    VkImage colorImg = colorImages[fi];
    VkImage depthImg = depthImages[fi];
    if (colorImg == VK_NULL_HANDLE || depthImg == VK_NULL_HANDLE) return;

    const uint32_t width = app->getWidth();
    const uint32_t height = app->getHeight();

    // ── Layout transitions: previous sampling -> attachments ──
    if (colorLayouts[fi] != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        VkAccessFlags2 srcAccess = 0;
        VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        if (colorLayouts[fi] == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            srcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        RendererUtils::transitionImageLayout(cmd, colorImg,
            colorLayouts[fi], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcAccess, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            srcStage, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        colorLayouts[fi] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (depthLayouts[fi] != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        VkAccessFlags2 srcAccess = 0;
        VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        if (depthLayouts[fi] == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            srcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        RendererUtils::transitionImageLayout(cmd, depthImg,
            depthLayouts[fi], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            srcAccess, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            srcStage, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        depthLayouts[fi] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        // No pipeline (shaders missing): still transition the targets to the
        // sampled layout so the composite reads defined (cleared-next-frame)
        // content instead of an attachment-layout image.
        RendererUtils::transitionImageLayout(cmd, colorImg,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        colorLayouts[fi] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        RendererUtils::transitionImageLayout(cmd, depthImg,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        depthLayouts[fi] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return;
    }

    VkClearValue colorClear{};
    colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkClearValue depthClear{};
    depthClear.depthStencil = {1.0f, 0};

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = colorImageViews[fi];
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue = colorClear;

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = depthImageViews[fi];
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue = depthClear;

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.offset = {0, 0};
    ri.renderArea.extent = {width, height};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, {width, height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDescriptorSet sets[2] = { mainDs, paramsSets[fi] };
    uint32_t setCount = (sets[1] != VK_NULL_HANDLE) ? 2u : 1u;
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, setCount, sets, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, setCount, sets, 0, nullptr);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // ── Back to SHADER_READ_ONLY for the composite / debug views ──
    RendererUtils::transitionImageLayout(cmd, colorImg,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    colorLayouts[fi] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    RendererUtils::transitionImageLayout(cmd, depthImg,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);
    depthLayouts[fi] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
