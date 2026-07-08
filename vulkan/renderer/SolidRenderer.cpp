#include "SolidRenderer.hpp"
#include "RendererUtils.hpp"

#include "../../utils/FileReader.hpp"
#include "../ShaderStage.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

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

    for (int i = 0; i < 2; ++i) {
        createImage(app->getSwapchainImageFormat(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                    solidColorImages[i], solidColorAllocations[i], solidColorMemories[i], solidColorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                    solidDepthImages[i], solidDepthAllocations[i], solidDepthMemories[i], solidDepthImageViews[i]);
        solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // Ensure created images have authoritative GPU/tracked layouts before
    // first use. Use VulkanApp helpers to perform transitions so the
    // app's layout-tracking map is updated consistently.
    for (int i = 0; i < 2; ++i) {
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
    for (int i = 0; i < 2; ++i) {
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

void SolidRenderer::beginPass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear, VulkanApp* app) {
    if (cmd == VK_NULL_HANDLE) {
        std::cerr << "[SolidRenderer::beginPass] Missing cmd, skipping." << std::endl;
        return;
    }
    if (!app) throw std::runtime_error("SolidRenderer::beginPass requires valid VulkanApp");

    // Barrier: transition solid color from SHADER_READ_ONLY → COLOR_ATTACHMENT_OPTIMAL
    // so the solid pipeline can write scene color.  The image was left in read-only
    // layout after the previous frame for ImGui / debug overlay sampling.
    if (frameIndex < solidColorImages.size() && solidColorImages[frameIndex] != VK_NULL_HANDLE) {
        VkImageMemoryBarrier2 colorBarrier{};
        colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        colorBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.image = solidColorImages[frameIndex];
        colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        colorBarrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &colorBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
        app->setImageLayoutTracked(solidColorImages[frameIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 1);
    }

    // Barrier: transition solid depth from its tracked layout to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    // for depth testing during the solid geometry pass.  The depth is usually
    // in SHADER_READ_ONLY (sampled by water / debug) or read-only attachment.
    if (frameIndex < solidDepthImages.size() && solidDepthImages[frameIndex] != VK_NULL_HANDLE) {
        if (solidDepthImageLayouts[frameIndex] != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            app->recordTransitionImageLayoutLayer(cmd, solidDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT, solidDepthImageLayouts[frameIndex], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->recordTrackedLayoutForCommandBuffer(cmd, solidDepthImages[frameIndex], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            solidDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = solidColorImageViews[frameIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = colorClear;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = solidDepthImageViews[frameIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue = depthClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {renderWidth, renderHeight};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);
}

void SolidRenderer::endPass(VkCommandBuffer cmd, uint32_t frameIndex, VulkanApp* app) {
    if (cmd == VK_NULL_HANDLE) return;
    vkCmdEndRendering(cmd);

    // Barrier: transition solid color from COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // after the solid pass so water / sky / debug renderers can sample it.
    if (frameIndex < solidColorImages.size() && solidColorImages[frameIndex] != VK_NULL_HANDLE) {
        VkImageMemoryBarrier2 colorBarrier{};
        colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        colorBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.image = solidColorImages[frameIndex];
        colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        colorBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        colorBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &colorBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
        if (app) app->setImageLayoutTracked(solidColorImages[frameIndex], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
    }

    // Barrier: transition solid depth from DEPTH_STENCIL_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // so water / debug / post-process renderers can sample the depth buffer.
    if (frameIndex < solidDepthImages.size() && solidDepthImages[frameIndex] != VK_NULL_HANDLE) {
        if (solidDepthImageLayouts[frameIndex] != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            if (app) {
                app->recordTransitionImageLayoutLayer(cmd, solidDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT, solidDepthImageLayouts[frameIndex], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
                app->recordTrackedLayoutForCommandBuffer(cmd, solidDepthImages[frameIndex], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            }
            solidDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }
}

void SolidRenderer::createPipelines(VulkanApp* app) {
    if (!app) return;

    ShaderStage vertexShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.vert.spv"),
        VK_SHADER_STAGE_VERTEX_BIT
    );

    ShaderStage fragmentShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.frag.spv"),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    ShaderStage tescShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.tesc.spv"),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    );
    ShaderStage teseShader = ShaderStage(
        app->getOrCreateShaderModule("shaders/main.tese.spv"),
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    );

    // Descriptor set layouts: ensure main UBO (set 0) is first
    // Main shaders don't use set 1 (material descriptor set), so only include set 0
    std::vector<VkDescriptorSetLayout> setLayouts;
    if (app->getDescriptorSetLayout() != VK_NULL_HANDLE) setLayouts.push_back(app->getDescriptorSetLayout());
    // Note: Removed material descriptor set layout since main shaders don't use set = 1

    // No per-mesh model push-constants are used anymore (models are identity in shaders).
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
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );
    graphicsPipeline = pipeline;
    graphicsPipelineLayout = layout;
    // Register the main graphics pipeline with the app so ShadowRenderer can use it
    if (app) {
        printf("[SolidRenderer] setAppGraphicsPipeline: pipeline=%p\n", (void*)graphicsPipeline);
        app->setAppGraphicsPipeline(graphicsPipeline);
    }


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
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, false, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );
    depthPrePassPipeline = depthPipeline;
    depthPrePassPipelineLayout = depthLayout;

    // Deferred depth test pipelines: depth-only (no color) and color-only (no depth write, LESS_OR_EQUAL)
    {
        // Depth-only: lightweight depth_only.frag, no color attachment
        ShaderStage depthFrag = ShaderStage(
            app->getOrCreateShaderModule("shaders/depth_only.frag.spv"),
            VK_SHADER_STAGE_FRAGMENT_BIT
        );
        auto [dp, dl] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, depthFrag.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts, nullptr,
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
            true, false, VK_COMPARE_OP_LESS, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false, {}, VK_FORMAT_D32_SFLOAT, true
        );
        deferredDepthPipeline = dp;
        deferredDepthPipelineLayout = dl;
        depthFrag.info.module = VK_NULL_HANDLE;
    }
    {
        // Color-only: full main.frag, LESS_OR_EQUAL compare, no depth write
        auto [cp, cl] = app->createGraphicsPipeline(
            { vertexShader.info, tescShader.info, teseShader.info, fragmentShader.info },
            std::vector<VkVertexInputBindingDescription>{ VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } },
            vk_layouts::defaultAttributes(),
            setLayouts, nullptr,
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
            false, true, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            false, { app->getSwapchainImageFormat() }, VK_FORMAT_D32_SFLOAT
        );
        deferredColorPipeline = cp;
        deferredColorPipelineLayout = cl;
    }
    deferredPipelinesCreated = true;

    // Clear local shader module references; destruction handled by VulkanResourceManager
    teseShader.info.module = VK_NULL_HANDLE;
    tescShader.info.module = VK_NULL_HANDLE;
    fragmentShader.info.module = VK_NULL_HANDLE;
    vertexShader.info.module = VK_NULL_HANDLE;
}

void SolidRenderer::render(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet perTextureDescriptorSet) {
    static int frameCount = 0;
    if (frameCount++ == 0) {
        printf("[DEBUG] SolidRenderer::render called for the first time\n");
    }
    
    if (!appArg) {
        std::cerr << "[SolidRenderer::draw] appArg is nullptr, skipping." << std::endl;
        return;
    }
    
    static bool printedOnce = false;
    
    VkPipelineLayout usedLayout = graphicsPipelineLayout;
    if (graphicsPipeline == VK_NULL_HANDLE) {
        std::cerr << "[SolidRenderer::draw] graphicsPipeline is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (!printedOnce) {
        printf("[SolidRenderer::draw] binding pipeline=%p, layout=%p\n", (void*)graphicsPipeline, (void*)usedLayout);
    }
    if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, graphicsPipeline);
    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Bind descriptor set 0: main UBO/samplers (perTextureDescriptorSet)
    // Main shaders only use set 0, no set 1 needed
    if (!printedOnce) {
        printf("[SolidRenderer::draw] perTextureDescriptorSet=%p\n", (void*)perTextureDescriptorSet);
        printedOnce = true;
    }
    
    if (perTextureDescriptorSet != VK_NULL_HANDLE) {
        //printf("[BIND] SolidRenderer::draw: layout=%p firstSet=0 count=1 sets=%p\n", (void*)usedLayout, (void*)perTextureDescriptorSet);
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, usedLayout, 0, 1, &perTextureDescriptorSet, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, usedLayout, 0, 1, &perTextureDescriptorSet, 0, nullptr);
    } else {
        std::cerr << "[SolidRenderer::draw] ERROR: perTextureDescriptorSet is NULL!" << std::endl;
    }
    
    // Draw all meshes using GPU-culled indirect commands
    //printf("[SolidRenderer::draw] About to call drawPrepared\n");
    indirectRenderer.drawPrepared(commandBuffer);
    //printf("[SolidRenderer::draw] drawPrepared returned\n");
}

void SolidRenderer::renderDepthPrepass(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet perTextureDescriptorSet) {
    if (!appArg) {
        std::cerr << "[SolidRenderer::renderDepthPrepass] appArg is nullptr, skipping." << std::endl;
        return;
    }
    if (depthPrePassPipeline == VK_NULL_HANDLE) {
        std::cerr << "[SolidRenderer::renderDepthPrepass] depth pre-pass pipeline is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, depthPrePassPipeline);
    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipeline);

    // Bind descriptor set using the depth pre-pass pipeline layout
    if (perTextureDescriptorSet != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, depthPrePassPipelineLayout, 0, 1, &perTextureDescriptorSet, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipelineLayout, 0, 1, &perTextureDescriptorSet, 0, nullptr);
    }

    // Draw all meshes using GPU-culled indirect commands (depth-only)
    indirectRenderer.drawPrepared(commandBuffer);
}

void SolidRenderer::drawDepth(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet descSet) {
    if (!appArg || deferredDepthPipeline == VK_NULL_HANDLE) return;
    if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, deferredDepthPipeline);
    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredDepthPipeline);
    if (descSet != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, deferredDepthPipelineLayout, 0, 1, &descSet, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredDepthPipelineLayout, 0, 1, &descSet, 0, nullptr);
    }
    indirectRenderer.drawPrepared(commandBuffer);
}

void SolidRenderer::drawColor(VkCommandBuffer &commandBuffer, VulkanApp* appArg, VkDescriptorSet descSet) {
    if (!appArg || deferredColorPipeline == VK_NULL_HANDLE) return;
    if (cmdState) cmdState->bindGraphicsPipeline(commandBuffer, deferredColorPipeline);
    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredColorPipeline);
    if (descSet != VK_NULL_HANDLE) {
        if (cmdState) cmdState->bindGraphicsDescriptorSets(commandBuffer, deferredColorPipelineLayout, 0, 1, &descSet, 0, nullptr);
        else vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredColorPipelineLayout, 0, 1, &descSet, 0, nullptr);
    }
    indirectRenderer.drawPrepared(commandBuffer);
}

void SolidRenderer::cleanup(VulkanApp* app) {
    if (app == nullptr) return;
    // Ensure render targets (images, views, framebuffers) are released
    destroyRenderTargets(app);
    graphicsPipeline = VK_NULL_HANDLE;
    graphicsPipelineLayout = VK_NULL_HANDLE;
    depthPrePassPipeline = VK_NULL_HANDLE;
    depthPrePassPipelineLayout = VK_NULL_HANDLE;
    if (deferredPipelinesCreated) {
        VkDevice dev = app->getDevice();
        if (deferredDepthPipeline != VK_NULL_HANDLE) {
            app->resources.removePipeline(deferredDepthPipeline);
            vkDestroyPipeline(dev, deferredDepthPipeline, nullptr);
        }
        if (deferredDepthPipelineLayout != VK_NULL_HANDLE) {
            app->resources.removePipelineLayout(deferredDepthPipelineLayout);
            vkDestroyPipelineLayout(dev, deferredDepthPipelineLayout, nullptr);
        }
        if (deferredColorPipeline != VK_NULL_HANDLE) {
            app->resources.removePipeline(deferredColorPipeline);
            vkDestroyPipeline(dev, deferredColorPipeline, nullptr);
        }
        if (deferredColorPipelineLayout != VK_NULL_HANDLE) {
            app->resources.removePipelineLayout(deferredColorPipelineLayout);
            vkDestroyPipelineLayout(dev, deferredColorPipelineLayout, nullptr);
        }
        deferredPipelinesCreated = false;
    }

    // Remove meshes
    for (auto &entry : solidChunks) {
        if (entry.second.meshId != UINT32_MAX) indirectRenderer.removeMesh(entry.second.meshId);
    }
    solidChunks.clear();

    indirectRenderer.cleanup();
}
