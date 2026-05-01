#include "SolidRenderer.hpp"
#include "RendererUtils.hpp"

#include "../../utils/FileReader.hpp"
#include "../ShaderStage.hpp"
#include <glm/gtc/matrix_transform.hpp>

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

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2D(device, app, width, height, format, usage, aspect,
                                     "SolidRenderer: image", image, memory, view);
    };

    // Create simple render pass for solid offscreen (color+depth)
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = app->getSwapchainImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    std::array<VkAttachmentDescription, 2> attachments{colorAttachment, depthAttachment};

    // Subpass dependency: flush color + depth writes so that the layout
    // transition (→ SHADER_READ_ONLY_OPTIMAL) carries visible data and
    // external consumers (water pass, post-process) can sample the images.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = 0;
    dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &solidRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create solid render pass!");
    }
    // Register solid render pass
    app->resources.addRenderPass(solidRenderPass, "SolidRenderer: solidRenderPass");

    for (int i = 0; i < 2; ++i) {
        createImage(app->getSwapchainImageFormat(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                    solidColorImages[i], solidColorMemories[i], solidColorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                    solidDepthImages[i], solidDepthMemories[i], solidDepthImageViews[i]);
        // Initially unknown layout until first transition; widget will consult this
        solidDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;

        std::array<VkImageView, 2> attachmentsViews = {solidColorImageViews[i], solidDepthImageViews[i]};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = solidRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachmentsViews.size());
        fbInfo.pAttachments = attachmentsViews.data();
        fbInfo.width = width;
        fbInfo.height = height;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &solidFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create solid framebuffer!");
        }
        app->resources.addFramebuffer(solidFramebuffers[i], "SolidRenderer: framebuffer");
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
    // Clear local handles; destruction is centralized in VulkanResourceManager
    for (int i = 0; i < 2; ++i) {
        solidFramebuffers[i] = VK_NULL_HANDLE;
        solidColorImageViews[i] = VK_NULL_HANDLE;
        solidColorImages[i] = VK_NULL_HANDLE;
        solidColorMemories[i] = VK_NULL_HANDLE;
        solidDepthImageViews[i] = VK_NULL_HANDLE;
        solidDepthImages[i] = VK_NULL_HANDLE;
        solidDepthMemories[i] = VK_NULL_HANDLE;
    }
    solidRenderPass = VK_NULL_HANDLE;
}

void SolidRenderer::beginPass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear, VulkanApp* app) {
    if (cmd == VK_NULL_HANDLE || solidRenderPass == VK_NULL_HANDLE || solidFramebuffers[frameIndex] == VK_NULL_HANDLE) {
        std::cerr << "[SolidRenderer::beginPass] Missing cmd/renderPass/framebuffer, skipping." << std::endl;
        return;
    }
    // Ensure the depth image for this frame is in the depth-stencil-attachment
    // layout so the render pass clear can initialize it. Use the tracked
    // layout to record a correct barrier at record time.
    if (frameIndex < solidDepthImages.size() && solidDepthImages[frameIndex] != VK_NULL_HANDLE) {
        if (solidDepthImageLayouts[frameIndex] != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            // Use centralized app helper so recorded oldLayout matches the app's transition logic
            std::cerr << "[SolidRenderer::beginPass] transition: cmd=" << (void*)cmd << " image=" << (void*)solidDepthImages[frameIndex] << " frame=" << (unsigned)frameIndex << " trackedOld=" << (int)solidDepthImageLayouts[frameIndex] << " new=" << (int)VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL << std::endl;
            if (!app) {
                throw std::runtime_error("SolidRenderer::beginPass requires valid VulkanApp to record layout transitions");
            }
            // Pass UNDEFINED so VulkanApp can resolve the authoritative old layout
            // (including any pending tracked updates for this command buffer).
            app->recordTransitionImageLayoutLayer(cmd, solidDepthImages[frameIndex], VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            // Also record the expected render-pass layout as a tracked update for
            // this command buffer so submit-time pre-apply can make the
            // authoritative map reflect the transition the render pass will
            // perform. This avoids validation mismatches at submit time.
            app->recordTrackedLayoutForCommandBuffer(cmd, solidDepthImages[frameIndex], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            solidDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }
    std::array<VkClearValue, 2> clears{colorClear, depthClear};
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = solidRenderPass;
    rpInfo.framebuffer = solidFramebuffers[frameIndex];
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = {renderWidth, renderHeight};
    rpInfo.clearValueCount = static_cast<uint32_t>(clears.size());
    rpInfo.pClearValues = clears.data();
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void SolidRenderer::endPass(VkCommandBuffer cmd, uint32_t frameIndex, VulkanApp* app) {
    if (cmd == VK_NULL_HANDLE) return;
    vkCmdEndRenderPass(cmd);

    // After render pass ends, transition depth image to SHADER_READ_ONLY_OPTIMAL
    // and update tracked layout so external recorders see the correct state.
    if (frameIndex < solidDepthImages.size() && solidDepthImages[frameIndex] != VK_NULL_HANDLE) {
        VkImageLayout trackedOld = solidDepthImageLayouts[frameIndex];
        if (trackedOld != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // The render pass has finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            // so the layout transition is performed implicitly by the render pass.
            // Avoid emitting an extra barrier here (which can conflict with the
            // renderpass transition) — just update the authoritative tracked
            // layout so later record-time callers see the correct state.
            std::cerr << "[SolidRenderer::endPass] updating tracked layout: image=" << (void*)solidDepthImages[frameIndex] << " frame=" << (unsigned)frameIndex << " -> " << (int)VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL << std::endl;
            if (app) {
                app->recordTrackedLayoutForCommandBuffer(cmd, solidDepthImages[frameIndex], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            }
            solidDepthImageLayouts[frameIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }
}

void SolidRenderer::createPipelines(VulkanApp* app) {
    if (!app) return;

    ShaderStage vertexShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
        VK_SHADER_STAGE_VERTEX_BIT
    );

    ShaderStage fragmentShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.frag.spv")),
        VK_SHADER_STAGE_FRAGMENT_BIT
    );

    ShaderStage tescShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    );
    ShaderStage teseShader = ShaderStage(
        app->createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
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
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, solidRenderPass
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
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
        },
        setLayouts,
        nullptr,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, true, false, VK_COMPARE_OP_LESS_OR_EQUAL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, solidRenderPass
    );
    depthPrePassPipeline = depthPipeline;
    depthPrePassPipelineLayout = depthLayout;

    // Clear local shader module references; destruction handled by VulkanResourceManager
    teseShader.info.module = VK_NULL_HANDLE;
    tescShader.info.module = VK_NULL_HANDLE;
    fragmentShader.info.module = VK_NULL_HANDLE;
    vertexShader.info.module = VK_NULL_HANDLE;
}

void SolidRenderer::depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool, VulkanApp* app) {
    if (!app) {
        std::cerr << "[SolidRenderer::depthPrePass] app is nullptr, skipping." << std::endl;
        return;
    }
    if (depthPrePassPipeline == VK_NULL_HANDLE) {
        std::cerr << "[SolidRenderer::depthPrePass] depthPrePassPipeline is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    //printf("[SolidRenderer] vkCmdBindPipeline: depthPrePassPipeline=%p\n", (void*)depthPrePassPipeline);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipeline);
    // Bind the main descriptor set (UBO/samplers/materials) at set 0 for the depth pre-pass.
    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    if (mainDs != VK_NULL_HANDLE) {
            //printf("[BIND] SolidRenderer::depthPrePass: layout=%p firstSet=0 count=1 sets=%p\n", (void*)depthPrePassPipelineLayout, (void*)mainDs);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPrePassPipelineLayout, 0, 1, &mainDs, 0, nullptr);
        }
    // Bind vertex and index buffers before issuing draw commands
    indirectRenderer.bindBuffers(commandBuffer);
    indirectRenderer.drawIndirectOnly(commandBuffer, app);
    if (queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, queryPool, 5);
    }
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
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Set dynamic viewport and scissor (required since pipeline uses dynamic state)
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(appArg->getWidth());
    viewport.height = static_cast<float>(appArg->getHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    if (!printedOnce) {
        printf("[SolidRenderer::draw] viewport: x=%.1f y=%.1f w=%.1f h=%.1f depth=[%.1f,%.1f]\n",
               viewport.x, viewport.y, viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth);
    }
    
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(appArg->getWidth()), static_cast<uint32_t>(appArg->getHeight())};
    
    if (!printedOnce) {
        printf("[SolidRenderer::draw] scissor: offset=(%d,%d) extent=(%u,%u)\n",
               scissor.offset.x, scissor.offset.y, scissor.extent.width, scissor.extent.height);
    }
    
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Bind descriptor set 0: main UBO/samplers (perTextureDescriptorSet)
    // Main shaders only use set 0, no set 1 needed
    if (!printedOnce) {
        printf("[SolidRenderer::draw] perTextureDescriptorSet=%p\n", (void*)perTextureDescriptorSet);
        printedOnce = true;
    }
    
    if (perTextureDescriptorSet != VK_NULL_HANDLE) {
        //printf("[BIND] SolidRenderer::draw: layout=%p firstSet=0 count=1 sets=%p\n", (void*)usedLayout, (void*)perTextureDescriptorSet);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, usedLayout, 0, 1, &perTextureDescriptorSet, 0, nullptr);
    } else {
        std::cerr << "[SolidRenderer::draw] ERROR: perTextureDescriptorSet is NULL!" << std::endl;
    }
    
    // Draw all meshes using GPU-culled indirect commands
    //printf("[SolidRenderer::draw] About to call drawPrepared\n");
    indirectRenderer.drawPrepared(commandBuffer);
    //printf("[SolidRenderer::draw] drawPrepared returned\n");
}

void SolidRenderer::cleanup(VulkanApp* app) {
    if (app == nullptr) return;
    // Ensure render targets (images, views, framebuffers) are released
    destroyRenderTargets(app);
    graphicsPipeline = VK_NULL_HANDLE;
    graphicsPipelineLayout = VK_NULL_HANDLE;
    depthPrePassPipeline = VK_NULL_HANDLE;
    depthPrePassPipelineLayout = VK_NULL_HANDLE;

    // Remove meshes
    for (auto &entry : solidChunks) {
        if (entry.second.meshId != UINT32_MAX) indirectRenderer.removeMesh(entry.second.meshId);
    }
    solidChunks.clear();

    indirectRenderer.cleanup();
}
