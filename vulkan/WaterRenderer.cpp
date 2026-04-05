
#include "WaterRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <array>
#include <glm/gtc/matrix_transform.hpp>

// Global image layout tracking for WaterRenderer render targets
static VkImageLayout sceneColorImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout sceneDepthImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout waterDepthImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
static VkImageLayout waterNormalImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
static VkImageLayout waterMaskImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
static VkImageLayout waterGeomDepthImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

WaterRenderer::WaterRenderer() {}

WaterRenderer::~WaterRenderer() {}

void WaterRenderer::init(VulkanApp* app, Buffer& waterParamsBuffer) {
    this->waterParamsBuffer = waterParamsBuffer;
    waterIndirectRenderer.init();
    createSamplers(app);
    createWaterRenderPass(app);
    createSceneRenderPass(app);
    createWaterPipelines(app);
}

void WaterRenderer::cleanup(VulkanApp* app) {
    VkDevice device = app->getDevice();
    VulkanApp* appPtr = app;
    
    waterIndirectRenderer.cleanup();
    // Clear local handles; VulkanResourceManager is responsible for actual destruction
    destroyRenderTargets(app);
    destroySolid360Targets(app);
    waterGeometryPipeline = VK_NULL_HANDLE;
    waterDepthDescriptorPool = VK_NULL_HANDLE;
    waterDepthDescriptorSetLayout = VK_NULL_HANDLE;
    waterGeometryPipelineLayout = VK_NULL_HANDLE;
    waterRenderPass = VK_NULL_HANDLE;
    sceneRenderPass = VK_NULL_HANDLE;
    linearSampler = VK_NULL_HANDLE;
}

void WaterRenderer::createSamplers(VulkanApp* app) {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    
    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &linearSampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water linear sampler!");
    }
    fprintf(stderr, "[WaterRenderer] createSampler: linearSampler=%p\n", (void*)linearSampler);
    // Register water sampler
    app->resources.addSampler(linearSampler, "WaterRenderer: linearSampler");
}

void WaterRenderer::createSceneRenderPass(VulkanApp* app) {
    // Scene render pass outputs color + depth to offscreen targets
    std::array<VkAttachmentDescription, 2> attachments{};
    
    // Scene color - use same format as swapchain for pipeline compatibility
    VkFormat swapchainFormat = app->getSwapchainImageFormat();
    attachments[0].format = swapchainFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    // Scene depth
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    
    // Match swapchain render pass dependencies for pipeline compatibility
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    
    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &sceneRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create scene offscreen render pass!");
    }
    // Register scene render pass
    app->resources.addRenderPass(sceneRenderPass, "WaterRenderer: sceneRenderPass");
}

void WaterRenderer::beginScenePass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear) {
    if (cmd == VK_NULL_HANDLE || sceneRenderPass == VK_NULL_HANDLE || sceneFramebuffers[frameIndex] == VK_NULL_HANDLE) {
        fprintf(stderr, "[WaterRenderer::beginScenePass] Missing cmd/renderPass/framebuffer, skipping.\n");
        return;
    }
    std::array<VkClearValue, 2> clears{colorClear, depthClear};
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = sceneRenderPass;
    rpInfo.framebuffer = sceneFramebuffers[frameIndex];
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = {renderWidth, renderHeight};
    rpInfo.clearValueCount = static_cast<uint32_t>(clears.size());
    rpInfo.pClearValues = clears.data();
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void WaterRenderer::endScenePass(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;
    vkCmdEndRenderPass(cmd);
}

void WaterRenderer::createWaterRenderPass(VulkanApp* app) {
    // Water geometry pass outputs:
    // 0: Water world position (RGBA32_SFLOAT - xyz=worldPos, w=linearDepth)
    // 1: Water normals (RGBA16_SFLOAT)
    // 2: Water mask (R8_UNORM - 1 where water, 0 elsewhere)
    // 3: Depth attachment (LOADED from scene pass, not cleared)
    
    std::array<VkAttachmentDescription, 4> attachments{};
    
    // Water world position + linear depth
    attachments[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    // Water normals
    attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    // Water mask
    attachments[2].format = VK_FORMAT_R8_UNORM;
    attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    // Depth buffer - use a dedicated depth buffer for water geometry pass (do not reuse scene depth image)
    attachments[3].format = VK_FORMAT_D32_SFLOAT;
    attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // Clear depth for water pass's own depth buffer
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;  // Dedicated depth buffer
    attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    std::array<VkAttachmentReference, 3> colorRefs{};
    colorRefs[0] = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorRefs[1] = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorRefs[2] = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    
    VkAttachmentReference depthRef{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments = colorRefs.data();
    subpass.pDepthStencilAttachment = &depthRef;
    
    // Two subpass dependencies:
    // [0] EXTERNAL → 0: Ensure solid pass color/depth writes are visible
    //     before the water subpass's fragment shader samples the scene textures
    //     AND before color/depth attachment ops begin.
    // [1] 0 → EXTERNAL: Flush water color attachment writes so the post-process
    //     fragment shader (in the swapchain render pass) can read them.
    std::array<VkSubpassDependency, 2> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();
    
    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &waterRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water render pass!");
    }
    // Register water render pass
    app->resources.addRenderPass(waterRenderPass, "WaterRenderer: waterRenderPass");
}

void WaterRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (renderWidth == width && renderHeight == height && waterFramebuffers[0] != VK_NULL_HANDLE) {
        return; // Already created at this size
    }
    
    destroyRenderTargets(app);
    
    renderWidth = width;
    renderHeight = height;
    
    VkDevice device = app->getDevice();
    
    // Helper to create image + memory + view
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create water image!");
        }
        // Debug: print created image handle for leak tracing
        fprintf(stderr, "[WaterRenderer] createImage: image=%p format=%d usage=0x%x\n", (void*)image, (int)format, usage);
        // Register image in app registry
        app->resources.addImage(image, "WaterRenderer: image");
        
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate water image memory!");
        }
        
        vkBindImageMemory(device, image, memory, 0);
        // Register device memory
        app->resources.addDeviceMemory(memory, "WaterRenderer: memory");
        
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create water image view!");
        }
        fprintf(stderr, "[WaterRenderer] createImageView: view=%p image=%p format=%d\n", (void*)view, (void*)image, (int)format);
        // Register image view
        app->resources.addImageView(view, "WaterRenderer: view");
    };
    
    // Reset layout tracking (use file-scope static variables)
    sceneColorImageLayouts[0] = VK_IMAGE_LAYOUT_UNDEFINED;
    sceneColorImageLayouts[1] = VK_IMAGE_LAYOUT_UNDEFINED;
    sceneDepthImageLayouts[0] = VK_IMAGE_LAYOUT_UNDEFINED;
    sceneDepthImageLayouts[1] = VK_IMAGE_LAYOUT_UNDEFINED;
    waterDepthImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    waterNormalImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    waterMaskImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    waterGeomDepthImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Create per-frame scene offscreen render targets (2 sets for 2 frames in flight)
    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        createImage(app->getSwapchainImageFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    sceneColorImages[frameIdx], sceneColorMemories[frameIdx], sceneColorImageViews[frameIdx]);
        sceneColorImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;

        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    sceneDepthImages[frameIdx], sceneDepthMemories[frameIdx], sceneDepthImageViews[frameIdx]);
        sceneDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;
        fprintf(stderr, "[WaterRenderer] sceneDepthImages[%d] = %p\n", frameIdx, (void*)sceneDepthImages[frameIdx]);

        std::array<VkImageView, 2> sceneAttachments = {
            sceneColorImageViews[frameIdx],
            sceneDepthImageViews[frameIdx]
        };

        VkFramebufferCreateInfo sceneFbInfo{};
        sceneFbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        sceneFbInfo.renderPass = sceneRenderPass;
        sceneFbInfo.attachmentCount = static_cast<uint32_t>(sceneAttachments.size());
        sceneFbInfo.pAttachments = sceneAttachments.data();
        sceneFbInfo.width = width;
        sceneFbInfo.height = height;
        sceneFbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &sceneFbInfo, nullptr, &sceneFramebuffers[frameIdx]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create scene offscreen framebuffer!");
        }
        // Register framebuffer
        app->resources.addFramebuffer(sceneFramebuffers[frameIdx], "WaterRenderer: sceneFramebuffer");
    }

    createImage(VK_FORMAT_R32G32B32A32_SFLOAT, 
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                waterDepthImage, waterDepthMemory, waterDepthImageView);
    waterDepthImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    createImage(VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                waterNormalImage, waterNormalMemory, waterNormalImageView);
    waterNormalImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    createImage(VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                waterMaskImage, waterMaskMemory, waterMaskImageView);
    waterMaskImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    createImage(VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                waterGeomDepthImage, waterGeomDepthMemory, waterGeomDepthImageView);
    waterGeomDepthImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    fprintf(stderr, "[WaterRenderer] waterGeomDepthImage = %p\n", (void*)waterGeomDepthImage);

    // Transition all images to their required layouts
    auto transitionImageLayout = [&](VkImage image, VkImageLayout& currentLayout, VkImageLayout newLayout, VkImageAspectFlags aspect) {
        if (currentLayout != newLayout) {
            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = currentLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = aspect;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;

            VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            } else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            } else {
                barrier.dstAccessMask = 0;
                dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            }

            vkCmdPipelineBarrier(
                cmd,
                srcStage,
                dstStage,
                0, 0, nullptr, 0, nullptr, 1, &barrier
            );
            });
            currentLayout = newLayout;
        }
    };

    // Transition scene color images
    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        transitionImageLayout(sceneColorImages[frameIdx], sceneColorImageLayouts[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transitionImageLayout(sceneDepthImages[frameIdx], sceneDepthImageLayouts[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    transitionImageLayout(waterDepthImage, waterDepthImageLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    transitionImageLayout(waterNormalImage, waterNormalImageLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    transitionImageLayout(waterMaskImage, waterMaskImageLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    transitionImageLayout(waterGeomDepthImage, waterGeomDepthImageLayout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Create per-frame water framebuffers using the dedicated water geometry depth buffer
    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        std::array<VkImageView, 4> waterAttachments = {
            waterDepthImageView,
            waterNormalImageView,
            waterMaskImageView,
            waterGeomDepthImageView  // Use dedicated water geometry depth buffer
        };
        
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = waterRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(waterAttachments.size());
        fbInfo.pAttachments = waterAttachments.data();
        fbInfo.width = width;
        fbInfo.height = height;
        fbInfo.layers = 1;
        
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &waterFramebuffers[frameIdx]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create water framebuffer!");
        }
        // Register water framebuffer
        app->resources.addFramebuffer(waterFramebuffers[frameIdx], "WaterRenderer: waterFramebuffer");
    }
    
    // Allocate and update per-frame descriptor sets for scene textures
    if (waterDepthDescriptorSetLayout != VK_NULL_HANDLE && linearSampler != VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> layouts(2, waterDepthDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = waterDepthDescriptorPool;  // Use dedicated pool, not app pool
        allocInfo.descriptorSetCount = 2;
        allocInfo.pSetLayouts = layouts.data();
        
        if (vkAllocateDescriptorSets(device, &allocInfo, waterDepthDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate water depth descriptor sets!");
        }
        // Register allocated descriptor sets
        for (size_t i = 0; i < 2; ++i) {
            app->resources.addDescriptorSet(waterDepthDescriptorSets[i], "WaterRenderer: waterDepthDescriptorSet");
        }
        
        // Update both descriptor sets to bind their frame's scene images
        for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
            updateSceneTexturesBinding(app, sceneColorImageViews[frameIdx], sceneDepthImageViews[frameIdx], frameIdx, VK_NULL_HANDLE);
        }
    }
    
    std::cout << "[WaterRenderer] Created render targets (2 sets) " << width << "x" << height << std::endl;
}

void WaterRenderer::destroyRenderTargets(VulkanApp* app) {
    VkDevice device = app->getDevice();
    VulkanApp* appPtr = app;
    // Clear per-frame framebuffer/image handles; actual Vulkan destruction
    // will be performed by the VulkanResourceManager.
    for (int i = 0; i < 2; ++i) {
        waterFramebuffers[i] = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; ++i) {
        sceneFramebuffers[i] = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; ++i) {
        sceneColorImages[i] = VK_NULL_HANDLE;
        sceneColorMemories[i] = VK_NULL_HANDLE;
        sceneColorImageViews[i] = VK_NULL_HANDLE;
        sceneDepthImages[i] = VK_NULL_HANDLE;
        sceneDepthMemories[i] = VK_NULL_HANDLE;
        sceneDepthImageViews[i] = VK_NULL_HANDLE;
    }
    waterDepthImage = VK_NULL_HANDLE;
    waterDepthMemory = VK_NULL_HANDLE;
    waterDepthImageView = VK_NULL_HANDLE;
    waterNormalImage = VK_NULL_HANDLE;
    waterNormalMemory = VK_NULL_HANDLE;
    waterNormalImageView = VK_NULL_HANDLE;
    waterMaskImage = VK_NULL_HANDLE;
    waterMaskMemory = VK_NULL_HANDLE;
    waterMaskImageView = VK_NULL_HANDLE;
    waterGeomDepthImage = VK_NULL_HANDLE;
    waterGeomDepthMemory = VK_NULL_HANDLE;
    waterGeomDepthImageView = VK_NULL_HANDLE;

    // Reset descriptor pool to free descriptor sets (safe to reset even if pending)
    if (waterDepthDescriptorPool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(device, waterDepthDescriptorPool, 0);
    }
}

void WaterRenderer::createWaterPipelines(VulkanApp* app) {
    VkDevice device = app->getDevice();
    
    // Water params buffer is already assigned in init
    
    // Initialize water params buffer with default values
    WaterParamsGPU defaultWaterParams{};
    defaultWaterParams.params1 = glm::vec4(0.03f, 5.0f, 0.7f, 0.0f); // refractionStrength, fresnelPower, transparency, unused
    defaultWaterParams.params2 = glm::vec4(0.3f, 0.0f, 0.0f, 0.0f);  // waterTint, unused, unused, unused
    defaultWaterParams.shallowColor = glm::vec4(0.1f, 0.4f, 0.5f, 0.0f);
    defaultWaterParams.deepColor = glm::vec4(0.0f, 0.15f, 0.25f, 0.0f);
    
    void* data;
    vkMapMemory(device, waterParamsBuffer.memory, 0, sizeof(WaterParamsGPU), 0, &data);
    memcpy(data, &defaultWaterParams, sizeof(WaterParamsGPU));
    vkUnmapMemory(device, waterParamsBuffer.memory);
    
    std::cout << "[WaterRenderer] Initialized water params buffer with default values" << std::endl;
    
    // Create descriptor set layout for scene textures (set 2)
    // Binding 0: Scene color texture
    // Binding 1: Scene depth texture
    // Binding 2: Sky color texture
    std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings{};
    
    // Scene color (binding 0)
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[0].pImmutableSamplers = nullptr;
    
    // Scene depth (binding 1) — also accessed by TES for depth-based wave attenuation
    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    sceneBindings[1].pImmutableSamplers = nullptr;

    // Sky color (binding 2)
    sceneBindings[2].binding = 2;
    sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[2].descriptorCount = 1;
    sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[2].pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo depthLayoutInfo{};
    depthLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    depthLayoutInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
    depthLayoutInfo.pBindings = sceneBindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &depthLayoutInfo, nullptr, &waterDepthDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water depth descriptor set layout!");
    }
    // Register water depth descriptor set layout
    app->resources.addDescriptorSetLayout(waterDepthDescriptorSetLayout, "WaterRenderer: waterDepthDescriptorSetLayout");
    
    // Create descriptor pool for scene textures (color + depth), 2 sets for 2 frames
    VkDescriptorPoolSize depthPoolSize{};
    depthPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    depthPoolSize.descriptorCount = 6;  // (color + depth + sky) * 2 frames
    
    VkDescriptorPoolCreateInfo depthPoolInfo{};
    depthPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    depthPoolInfo.poolSizeCount = 1;
    depthPoolInfo.pPoolSizes = &depthPoolSize;
    depthPoolInfo.maxSets = 2;  // 2 frames in flight
    
    if (vkCreateDescriptorPool(device, &depthPoolInfo, nullptr, &waterDepthDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water depth descriptor pool!");
    }
    // Register water depth descriptor pool with resource manager
    app->resources.addDescriptorPool(waterDepthDescriptorPool, "WaterRenderer: waterDepthDescriptorPool");
    
    // Descriptor sets are allocated and updated per-frame in createRenderTargets()
    // after scene images are created
    
    // Create a custom pipeline layout for water that includes:
    // Set 0: Material SSBO (from app->getMaterialDescriptorSetLayout())
    // Set 1: UBO (from app->getDescriptorSetLayout())
    // Set 2: Scene depth texture (waterDepthDescriptorSetLayout)
    // Descriptor set ordering: set 0 = global UBO+samplers, set 1 = material set, set 2 = scene depth textures
    std::array<VkDescriptorSetLayout, 3> waterSetLayouts = {
        app->getDescriptorSetLayout(),           // Set 0: UBO + samplers
        app->getMaterialDescriptorSetLayout(),   // Set 1: Materials
        waterDepthDescriptorSetLayout            // Set 2: Scene depth texture
    };
    
    // No per-mesh model push-constants are used for water (shaders use identity/no model push-constant).

    std::cout << "[WaterRenderer] Created water pipeline layout with 3 descriptor sets" << std::endl;

    // Create water geometry pipeline with dedicated water shaders
    // Load water shaders (vertex, tessellation control, tessellation evaluation, fragment)
    auto vertCode = FileReader::readFile("shaders/water.vert.spv");
    auto tescCode = FileReader::readFile("shaders/water.tesc.spv"); 
    auto teseCode = FileReader::readFile("shaders/water.tese.spv");
    auto fragCode = FileReader::readFile("shaders/water.frag.spv");

    if (vertCode.empty() || fragCode.empty()) {
        std::cerr << "[WaterRenderer] Warning: Could not load water geometry shaders" << std::endl;
        return;
    }

    VkShaderModule vertModule = app->createShaderModule(vertCode);
    VkShaderModule fragModule = app->createShaderModule(fragCode);
    VkShaderModule tescModule = VK_NULL_HANDLE;
    VkShaderModule teseModule = VK_NULL_HANDLE;

    bool hasTessellation = !tescCode.empty() && !teseCode.empty();
    if (hasTessellation) {
        tescModule = app->createShaderModule(tescCode);
        teseModule = app->createShaderModule(teseCode);
    }

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";
    shaderStages.push_back(vertStage);

    if (hasTessellation) {
        VkPipelineShaderStageCreateInfo tescStage{};
        tescStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tescStage.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        tescStage.module = tescModule;
        tescStage.pName = "main";
        shaderStages.push_back(tescStage);

        VkPipelineShaderStageCreateInfo teseStage{};
        teseStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        teseStage.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        teseStage.module = teseModule;
        teseStage.pName = "main";
        shaderStages.push_back(teseStage);
    }

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    shaderStages.push_back(fragStage);

    // Vertex input (same as main pipeline)
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 5> attrDescs{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
    attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)};
    attrDescs[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attrDescs[4] = {5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex)};

    // --- Create pipeline layout manually ---
    VkPipelineLayoutCreateInfo waterLayoutInfo{};
    waterLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    waterLayoutInfo.setLayoutCount = static_cast<uint32_t>(waterSetLayouts.size());
    waterLayoutInfo.pSetLayouts = waterSetLayouts.data();
    waterLayoutInfo.pushConstantRangeCount = 0;
    waterLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device, &waterLayoutInfo, nullptr, &waterGeometryPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water geometry pipeline layout!");
    }
    app->resources.addPipelineLayout(waterGeometryPipelineLayout, "WaterRenderer: waterGeometryPipelineLayout");

    // --- Create pipeline manually (waterRenderPass has 3 color attachments) ---
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = hasTessellation ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;   // Enable water-against-water occlusion in this pass
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Water render pass has 3 color attachments — need a blend state for each
    std::array<VkPipelineColorBlendAttachmentState, 3> colorBlendAttachments{};
    for (auto& att : colorBlendAttachments) {
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        att.blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlending.pAttachments = colorBlendAttachments.data();

    VkPipelineTessellationStateCreateInfo tessState{};
    if (hasTessellation) {
        tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = waterGeometryPipelineLayout;
    pipelineInfo.renderPass = waterRenderPass;  // MUST match the render pass used at draw time
    pipelineInfo.subpass = 0;
    if (hasTessellation) pipelineInfo.pTessellationState = &tessState;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &waterGeometryPipeline) != VK_SUCCESS) {
        std::cerr << "[WaterRenderer] Warning: Failed to create water geometry pipeline" << std::endl;
        waterGeometryPipeline = VK_NULL_HANDLE;
    } else {
        app->resources.addPipeline(waterGeometryPipeline, "WaterRenderer: waterGeometryPipeline");
        std::cout << "[WaterRenderer] Created water geometry pipeline for waterRenderPass (3 color attachments)" << std::endl;
    }

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
    if (tescModule) tescModule = VK_NULL_HANDLE;
    if (teseModule) teseModule = VK_NULL_HANDLE;
}

void WaterRenderer::beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (waterRenderPass == VK_NULL_HANDLE) {
        fprintf(stderr, "[WaterRenderer::beginWaterGeometryPass] waterRenderPass is VK_NULL_HANDLE, skipping.\n");
        return;
    }
    if (waterFramebuffers[frameIndex] == VK_NULL_HANDLE) {
        fprintf(stderr, "[WaterRenderer::beginWaterGeometryPass] waterFramebuffers[%u] is VK_NULL_HANDLE, skipping.\n", frameIndex);
        return;
    }
    std::array<VkClearValue, 4> clearValues{};
    clearValues[0].color = {{1000.0f, 0.0f, 0.0f, 0.0f}}; // Far depth
    clearValues[1].color = {{0.0f, 1.0f, 0.0f, 0.0f}};    // Up normal
    clearValues[2].color = {{0.0f, 0.0f, 0.0f, 0.0f}};    // No water mask
    clearValues[3].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = waterRenderPass;
    renderPassInfo.framebuffer = waterFramebuffers[frameIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {renderWidth, renderHeight};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    // Set viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void WaterRenderer::endWaterGeometryPass(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) {
        fprintf(stderr, "[WaterRenderer::endWaterGeometryPass] cmd is VK_NULL_HANDLE, skipping.\n");
        return;
    }
    // Properly end the water geometry render pass recorded on the provided command buffer
    vkCmdEndRenderPass(cmd);
}

void WaterRenderer::updateSceneTexturesBinding(VulkanApp* app, VkImageView colorImageView, VkImageView depthImageView, uint32_t frameIndex, VkImageView skyImageView) {
    if (waterDepthDescriptorSets[frameIndex] == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return;
    }
    
    // Use the provided image views directly
    VkImageView colorView = colorImageView;
    VkImageView depthView = depthImageView;
    
    std::array<VkDescriptorImageInfo, 3> imageInfos{};
    
    // Scene color (binding 0)
    imageInfos[0].sampler = linearSampler;
    imageInfos[0].imageView = colorView;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    // Scene depth (binding 1)
    imageInfos[1].sampler = linearSampler;
    imageInfos[1].imageView = depthView;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Sky color (binding 2)
    imageInfos[2].sampler = linearSampler;
    imageInfos[2].imageView = (skyImageView != VK_NULL_HANDLE) ? skyImageView : colorView;  // fallback to scene color if no sky
    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    std::array<VkWriteDescriptorSet, 3> writes{};
    
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = waterDepthDescriptorSets[frameIndex];
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfos[0];
    
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = waterDepthDescriptorSets[frameIndex];
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &imageInfos[1];

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = waterDepthDescriptorSets[frameIndex];
    writes[2].dstBinding = 2;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &imageInfos[2];

    vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


// Execute water's offscreen geometry pass on the provided command buffer.
// The caller must ensure that the solid pass has already ended on this same
// command buffer so that the scene color/depth images are available.

void WaterRenderer::prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView sceneColorView, VkImageView sceneDepthView, const WaterParams& params, float waterTime, VkImageView skyView) {
    if (!app || cmd == VK_NULL_HANDLE) return;

    // Upload current water parameters to the GPU buffer (set 0, binding 7)
    {
        WaterParamsGPU gpu{};
        gpu.params1 = glm::vec4(params.refractionStrength, params.fresnelPower, params.transparency, params.reflectionStrength);
        gpu.params2 = glm::vec4(params.waterTint, params.noiseScale, static_cast<float>(params.noiseOctaves), params.noisePersistence);
        gpu.params3 = glm::vec4(params.noiseTimeSpeed, waterTime, params.specularIntensity, params.specularPower);
        gpu.shallowColor = glm::vec4(params.shallowColor, params.waveDepthTransition);
        gpu.deepColor = glm::vec4(params.deepColor, params.glitterIntensity);
        gpu.waveParams = glm::vec4(0.0f, 0.0f, params.bumpAmplitude, params.depthFalloff);
        gpu.reserved1 = glm::vec4(0.0f);
        gpu.reserved2 = glm::vec4(0.0f);

        void* data;
        vkMapMemory(app->getDevice(), waterParamsBuffer.memory, 0, sizeof(WaterParamsGPU), 0, &data);
        memcpy(data, &gpu, sizeof(WaterParamsGPU));
        vkUnmapMemory(app->getDevice(), waterParamsBuffer.memory);
    }

    // Update per-frame scene descriptors
    updateSceneTexturesBinding(app, sceneColorView, sceneDepthView, frameIndex, skyView);

    // Record the water geometry pass directly on the main command buffer.
    // The solid render pass already ended (finalLayout transitions images to
    // SHADER_READ_ONLY_OPTIMAL) but we still need an execution+memory barrier
    // so that COLOR_ATTACHMENT_OUTPUT writes from the solid pass are visible
    // to FRAGMENT_SHADER reads in the water pass.
    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &memBarrier, 0, nullptr, 0, nullptr);
}

void WaterRenderer::postRenderBarrier(VkCommandBuffer cmd) {
    // The 0→EXTERNAL render pass dependency flushes color attachment writes,
    // but the swapchain render pass's EXTERNAL→0 dependency only covers
    // EARLY_FRAGMENT_TESTS — it does NOT synchronize FRAGMENT_SHADER reads.
    // Add an explicit barrier so the post-process fragment shader can sample
    // the water output images.
    VkMemoryBarrier postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    postBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &postBarrier, 0, nullptr, 0, nullptr);
}

void WaterRenderer::render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView sceneColorView, VkImageView sceneDepthView, const WaterParams& params, float waterTime, VkImageView skyView) {
    if (!app || cmd == VK_NULL_HANDLE) return;

    prepareRender(app, cmd, frameIndex, sceneColorView, sceneDepthView, params, waterTime, skyView);

    beginWaterGeometryPass(cmd, frameIndex);

    // Bind the water geometry pipeline and descriptor sets before issuing draw commands
    if (waterGeometryPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterGeometryPipeline);

        // Bind each set individually so a VK_NULL_HANDLE material set is
        // never passed to vkCmdBindDescriptorSets (that causes a GPU hang).

        // Set 0: main UBO + texture samplers + water params
        VkDescriptorSet mainDs = app->getMainDescriptorSet();
        if (mainDs != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                waterGeometryPipelineLayout, 0, 1, &mainDs, 0, nullptr);
        }

        // Set 1: materials SSBO — only bind if available (shader doesn't use it)
        VkDescriptorSet materialDs = app->getMaterialDescriptorSet();
        if (materialDs != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                waterGeometryPipelineLayout, 1, 1, &materialDs, 0, nullptr);
        }

        // Set 2: scene depth textures (color + depth)
        VkDescriptorSet sceneDs = waterDepthDescriptorSets[frameIndex];
        if (sceneDs != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                waterGeometryPipelineLayout, 2, 1, &sceneDs, 0, nullptr);
        }
    }

    // Indirect rendering for water geometry
    waterIndirectRenderer.drawPrepared(cmd);
    endWaterGeometryPass(cmd);

    postRenderBarrier(cmd);
}

// ============================================================================
// Solid 360° cubemap reflection
// ============================================================================

void WaterRenderer::createSolid360Targets(VulkanApp* app, VkRenderPass solidRenderPass) {
    if (!app || solidRenderPass == VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();
    VkFormat colorFormat = app->getSwapchainImageFormat();

    // --- Helper: create image + memory ---
    auto allocImage = [&](VkImageCreateInfo& imgInfo, VkImage& image, VkDeviceMemory& memory) {
        if (vkCreateImage(device, &imgInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 360 image!");
        app->resources.addImage(image, "WaterRenderer: solid360 image");
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, image, &memReqs);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate 360 image memory!");
        app->resources.addDeviceMemory(memory, "WaterRenderer: solid360 memory");
        vkBindImageMemory(device, image, memory, 0);
    };

    auto createView = [&](VkImage image, VkFormat format, VkImageAspectFlags aspect,
                           VkImageViewType viewType, uint32_t baseLayer, uint32_t layerCount,
                           VkImageView& view, const char* name) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = viewType;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = baseLayer;
        viewInfo.subresourceRange.layerCount = layerCount;
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 360 image view!");
        app->resources.addImageView(view, name);
    };

    // --- 1. Cubemap color image (6 layers) ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = colorFormat;
        imgInfo.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 6;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, cube360ColorImage, cube360ColorMemory);
    }

    // Per-face 2D views (for framebuffer attachment)
    for (uint32_t face = 0; face < 6; ++face) {
        createView(cube360ColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_IMAGE_VIEW_TYPE_2D, face, 1,
                   cube360FaceViews[face], "WaterRenderer: cube360 face view");
    }

    // Cubemap view (for sampling in the conversion shader)
    createView(cube360ColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
               VK_IMAGE_VIEW_TYPE_CUBE, 0, 6,
               cube360CubeView, "WaterRenderer: cube360 cube view");

    // --- 2. Shared depth image ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_D32_SFLOAT;
        imgInfo.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, cube360DepthImage, cube360DepthMemory);
    }
    createView(cube360DepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
               VK_IMAGE_VIEW_TYPE_2D, 0, 1,
               cube360DepthView, "WaterRenderer: cube360 depth view");

    // --- 3. Per-face framebuffers (reuse solidRenderPass: color + depth) ---
    for (uint32_t face = 0; face < 6; ++face) {
        std::array<VkImageView, 2> attachments = {cube360FaceViews[face], cube360DepthView};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = solidRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments = attachments.data();
        fbInfo.width = CUBE360_FACE_SIZE;
        fbInfo.height = CUBE360_FACE_SIZE;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &cube360Framebuffers[face]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cube360 framebuffer!");
        app->resources.addFramebuffer(cube360Framebuffers[face], "WaterRenderer: cube360 framebuffer");
    }

    // --- 4. Equirectangular output image ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = colorFormat;
        imgInfo.extent = {EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allocImage(imgInfo, equirect360Image, equirect360Memory);
    }
    createView(equirect360Image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT,
               VK_IMAGE_VIEW_TYPE_2D, 0, 1,
               equirect360View, "WaterRenderer: equirect360 view");

    // --- 5. Equirect render pass (color only, no depth) ---
    {
        VkAttachmentDescription colorAtt{};
        colorAtt.format = colorFormat;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAtt;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dep;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &equirect360RenderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create equirect360 render pass!");
        app->resources.addRenderPass(equirect360RenderPass, "WaterRenderer: equirect360RenderPass");
    }

    // Equirect framebuffer
    {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = equirect360RenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &equirect360View;
        fbInfo.width = EQUIRECT360_WIDTH;
        fbInfo.height = EQUIRECT360_HEIGHT;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &equirect360Framebuffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create equirect360 framebuffer!");
        app->resources.addFramebuffer(equirect360Framebuffer, "WaterRenderer: equirect360 framebuffer");
    }

    // --- 6. Cubemap descriptor set layout + pool + set ---
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &cube360DescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cube360 descriptor set layout!");
        app->resources.addDescriptorSetLayout(cube360DescSetLayout, "WaterRenderer: cube360DescSetLayout");

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &cube360DescPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cube360 descriptor pool!");
        app->resources.addDescriptorPool(cube360DescPool, "WaterRenderer: cube360DescPool");

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = cube360DescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &cube360DescSetLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &cube360DescSet) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate cube360 descriptor set!");

        // Write cubemap sampler to the descriptor set
        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler = linearSampler;
        imgInfo.imageView = cube360CubeView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = cube360DescSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    // --- 7. Cubemap→equirect conversion pipeline ---
    {
        equirect360VertModule = app->createShaderModule(FileReader::readFile("shaders/fullscreen.vert.spv"));
        equirect360FragModule = app->createShaderModule(FileReader::readFile("shaders/cubemap_to_equirect.frag.spv"));

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = equirect360VertModule;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = equirect360FragModule;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState blendAtt{};
        blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAtt.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo blending{};
        blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blending.attachmentCount = 1;
        blending.pAttachments = &blendAtt;

        std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
        dynState.pDynamicStates = dynStates.data();

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(float) * 2;

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &cube360DescSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &equirect360PipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create equirect360 pipeline layout!");
        app->resources.addPipelineLayout(equirect360PipelineLayout, "WaterRenderer: equirect360PipelineLayout");

        VkGraphicsPipelineCreateInfo pipeInfo{};
        pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipeInfo.pStages = stages.data();
        pipeInfo.pVertexInputState = &vertexInput;
        pipeInfo.pInputAssemblyState = &inputAssembly;
        pipeInfo.pViewportState = &viewportState;
        pipeInfo.pRasterizationState = &rasterizer;
        pipeInfo.pMultisampleState = &multisample;
        pipeInfo.pDepthStencilState = &depthStencil;
        pipeInfo.pColorBlendState = &blending;
        pipeInfo.pDynamicState = &dynState;
        pipeInfo.layout = equirect360PipelineLayout;
        pipeInfo.renderPass = equirect360RenderPass;
        pipeInfo.subpass = 0;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &equirect360Pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create equirect360 pipeline!");
        app->resources.addPipeline(equirect360Pipeline, "WaterRenderer: equirect360Pipeline");
    }

    fprintf(stderr, "[WaterRenderer] Created solid 360 targets: cubemap %ux%u, equirect %ux%u\n",
            CUBE360_FACE_SIZE, CUBE360_FACE_SIZE, EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT);
}

void WaterRenderer::destroySolid360Targets(VulkanApp* app) {
    // Clear handles; VulkanResourceManager handles actual destruction
    cube360ColorImage = VK_NULL_HANDLE;
    cube360ColorMemory = VK_NULL_HANDLE;
    for (auto& v : cube360FaceViews) v = VK_NULL_HANDLE;
    cube360CubeView = VK_NULL_HANDLE;
    cube360DepthImage = VK_NULL_HANDLE;
    cube360DepthMemory = VK_NULL_HANDLE;
    cube360DepthView = VK_NULL_HANDLE;
    for (auto& fb : cube360Framebuffers) fb = VK_NULL_HANDLE;
    equirect360Image = VK_NULL_HANDLE;
    equirect360Memory = VK_NULL_HANDLE;
    equirect360View = VK_NULL_HANDLE;
    equirect360Framebuffer = VK_NULL_HANDLE;
    equirect360RenderPass = VK_NULL_HANDLE;
    equirect360Pipeline = VK_NULL_HANDLE;
    equirect360PipelineLayout = VK_NULL_HANDLE;
    equirect360VertModule = VK_NULL_HANDLE;
    equirect360FragModule = VK_NULL_HANDLE;
    cube360DescSetLayout = VK_NULL_HANDLE;
    cube360DescPool = VK_NULL_HANDLE;
    cube360DescSet = VK_NULL_HANDLE;
}

void WaterRenderer::renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                                    VkRenderPass solidRenderPass,
                                    SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                                    SolidRenderer* solidRenderer,
                                    VkDescriptorSet mainDescriptorSet,
                                    Buffer& uniformBuffer, const UniformObject& ubo) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (cube360Framebuffers[0] == VK_NULL_HANDLE || equirect360Pipeline == VK_NULL_HANDLE) return;

    // --- Build 6 face view matrices (Y-up, right-handed) ---
    // Cubemap face order: +X, -X, +Y, -Y, +Z, -Z
    glm::vec3 camPos = glm::vec3(ubo.viewPos);

    struct FaceInfo { glm::vec3 target; glm::vec3 up; };
    const FaceInfo faces[6] = {
        { glm::vec3( 1, 0, 0), glm::vec3(0,-1, 0) }, // +X
        { glm::vec3(-1, 0, 0), glm::vec3(0,-1, 0) }, // -X
        { glm::vec3( 0, 1, 0), glm::vec3(0, 0, 1) }, // +Y
        { glm::vec3( 0,-1, 0), glm::vec3(0, 0,-1) }, // -Y
        { glm::vec3( 0, 0, 1), glm::vec3(0,-1, 0) }, // +Z
        { glm::vec3( 0, 0,-1), glm::vec3(0,-1, 0) }, // -Z
    };

    // 90° FOV projection (square faces)
    glm::mat4 faceProj = glm::perspective(glm::radians(90.0f), 1.0f, ubo.passParams.z, ubo.passParams.w);
    faceProj[1][1] *= -1; // Vulkan Y-flip

    // --- Render 6 cubemap faces ---
    for (uint32_t face = 0; face < 6; ++face) {
        glm::mat4 faceView = glm::lookAt(camPos, camPos + faces[face].target, faces[face].up);
        glm::mat4 faceVP = faceProj * faceView;

        // Build per-face UBO (same as main UBO but with face VP)
        UniformObject faceUBO = ubo;
        faceUBO.viewProjection = faceVP;

        // Write per-face UBO to the buffer via command stream (GPU-side update)
        vkCmdUpdateBuffer(cmd, uniformBuffer.buffer, 0, sizeof(UniformObject), &faceUBO);

        // Barrier: transfer write → uniform read
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        // Begin render pass on this face
        VkClearValue clears[2];
        clears[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = solidRenderPass;
        rpInfo.framebuffer = cube360Framebuffers[face];
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE};
        rpInfo.clearValueCount = 2;
        rpInfo.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{0.0f, 0.0f, (float)CUBE360_FACE_SIZE, (float)CUBE360_FACE_SIZE, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, {CUBE360_FACE_SIZE, CUBE360_FACE_SIZE}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Draw sky sphere into this face
        if (skyRenderer) {
            VkPipeline skyPipe = (skyMode == SkySettings::Mode::Grid) ? skyRenderer->getSkyGridPipeline() : skyRenderer->getSkyPipeline();
            VkPipelineLayout skyLayout = (skyMode == SkySettings::Mode::Grid) ? skyRenderer->getSkyGridPipelineLayout() : skyRenderer->getSkyPipelineLayout();
            if (skyPipe != VK_NULL_HANDLE && skyLayout != VK_NULL_HANDLE) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyLayout, 0, 1, &mainDescriptorSet, 0, nullptr);
                const auto& skyVBO = skyRenderer->getSkyVBO();
                if (skyVBO.vertexBuffer.buffer != VK_NULL_HANDLE && skyVBO.indexCount > 0) {
                    VkBuffer vbs[] = {skyVBO.vertexBuffer.buffer};
                    VkDeviceSize offsets[] = {0};
                    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                    vkCmdBindIndexBuffer(cmd, skyVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(cmd, skyVBO.indexCount, 1, 0, 0, 0);
                }
            }
        }

        // Draw ALL solid meshes (no frustum cull — omnidirectional render)
        if (solidRenderer) {
            VkPipeline gfxPipe = solidRenderer->getGraphicsPipeline();
            VkPipelineLayout gfxLayout = solidRenderer->getGraphicsPipelineLayout();
            if (gfxPipe != VK_NULL_HANDLE && gfxLayout != VK_NULL_HANDLE) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxLayout, 0, 1, &mainDescriptorSet, 0, nullptr);
                solidRenderer->getIndirectRenderer().drawAll(cmd);
            }
        }

        vkCmdEndRenderPass(cmd);
    }

    // --- Transition cubemap to SHADER_READ_ONLY_OPTIMAL ---
    // After the render passes, each face layer's color is in SHADER_READ_ONLY_OPTIMAL
    // (from the solidRenderPass finalLayout). No extra barrier needed for sampling.

    // --- Convert cubemap → equirectangular ---
    {
        VkClearValue clear{};
        clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = equirect360RenderPass;
        rpInfo.framebuffer = equirect360Framebuffer;
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = {EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT};
        rpInfo.clearValueCount = 1;
        rpInfo.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, equirect360Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, equirect360PipelineLayout,
                                0, 1, &cube360DescSet, 0, nullptr);

        float resolution[2] = {(float)EQUIRECT360_WIDTH, (float)EQUIRECT360_HEIGHT};
        vkCmdPushConstants(cmd, equirect360PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(resolution), resolution);

        VkViewport viewport{0.0f, 0.0f, (float)EQUIRECT360_WIDTH, (float)EQUIRECT360_HEIGHT, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, {EQUIRECT360_WIDTH, EQUIRECT360_HEIGHT}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
        vkCmdEndRenderPass(cmd);
    }

    // Restore the main UBO via command stream so subsequent passes see the original data
    vkCmdUpdateBuffer(cmd, uniformBuffer.buffer, 0, sizeof(UniformObject), &ubo);
    {
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
}
