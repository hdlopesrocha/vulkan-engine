
#include "WaterRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include "WaterBackFaceRenderer.hpp"
#include "Solid360Renderer.hpp"

// Global image layout tracking for WaterRenderer render targets
static VkImageLayout sceneColorImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout sceneDepthImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout waterDepthImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout waterNormalImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout waterMaskImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
static VkImageLayout waterGeomDepthImageLayouts[2] = { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
// back-face image layout tracking moved to WaterBackFaceRenderer

WaterRenderer::WaterRenderer() {}

WaterRenderer::~WaterRenderer() {}

void WaterRenderer::init(VulkanApp* app, Buffer& waterParamsBuffer) {
    this->waterParamsBuffer = waterParamsBuffer;
    this->appPtr = app;
    waterIndirectRenderer.init();
    createSamplers(app);
    createWaterRenderPass(app);
    createSceneRenderPass(app);

    // Back-face renderpass must exist before creating its pipeline (created/owned by SceneRenderer)
    if (backFaceRenderer) backFaceRenderer->createRenderPass(app);

    // Create water pipelines (produces waterGeometryPipelineLayout)
    createWaterPipelines(app);

    // Let back-face renderer create its pipeline using the water pipeline layout
    if (backFaceRenderer) backFaceRenderer->createPipelines(app, waterGeometryPipelineLayout);
    if (solid360Renderer) solid360Renderer->init(app);
}

void WaterRenderer::setParamsBuffer(Buffer& buf, uint32_t count) {
    this->waterParamsBuffer = buf;
    this->waterParamsCount = count;
    // Reserve CPU-side params list and initialize with defaults (or use external vector if provided)
    if (externalParams) {
        externalParams->resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            (*externalParams)[i].time = 0.0f;
            // keep other defaults from WaterParams constructor
        }
    } else {
        paramsList.clear();
        paramsList.resize(count);
        // Initialize each entry with reasonable defaults (match previous single-default behavior)
        for (uint32_t i = 0; i < count; ++i) {
            paramsList[i].time = 0.0f;
            // keep other defaults from WaterParams constructor
        }
    }
    // Upload initial GPU data (default entries)
    if (appPtr && waterParamsBuffer.buffer != VK_NULL_HANDLE) {
        WaterParamsGPU defaultGpu{};
        defaultGpu.params1 = glm::vec4(0.03f, 5.0f, 0.7f, 0.0f);
        defaultGpu.params2 = glm::vec4(0.3f, 0.0f, 0.0f, 0.0f);
        defaultGpu.shallowColor = glm::vec4(0.1f, 0.4f, 0.5f, 0.0f);
        defaultGpu.deepColor = glm::vec4(0.0f, 0.15f, 0.25f, 0.0f);
        defaultGpu.waveParams = glm::vec4(0.0f, 0.0f, 8.0f, 0.1f);
        defaultGpu.reserved1 = glm::vec4(1.0f, 1.0f, 1.0f, 8.0f);
        defaultGpu.reserved2 = glm::vec4(4.0f, 0.004f, 0.05f, 0.0f);
        defaultGpu.reserved3 = glm::vec4(0.0f);
        defaultGpu.causticColor = glm::vec4(1.0f, 0.98f, 0.8f, 0.0f);
        defaultGpu.causticParams = glm::vec4(40.0f, 1.5f, 1.0f, 4.0f);
        defaultGpu.causticExtraParams = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f);

        // Map entire buffer and fill with defaultGpu
        void* data = nullptr;
        vkMapMemory(appPtr->getDevice(), waterParamsBuffer.memory, 0, sizeof(WaterParamsGPU) * waterParamsCount, 0, &data);
        for (uint32_t i = 0; i < waterParamsCount; ++i) {
            memcpy(static_cast<char*>(data) + i * sizeof(WaterParamsGPU), &defaultGpu, sizeof(WaterParamsGPU));
        }
        vkUnmapMemory(appPtr->getDevice(), waterParamsBuffer.memory);
    }
}

void WaterRenderer::updateGPUParamsForLayer(uint32_t layer) {
    if (!appPtr) return;
    if (layer >= waterParamsCount) return;
    // Pack CPU params into GPU struct
    WaterParamsGPU gpu{};
    WaterParams *p = nullptr;
    if (externalParams) {
        if (layer >= externalParams->size()) return;
        p = &(*externalParams)[layer];
    } else {
        if (layer >= paramsList.size()) return;
        p = &paramsList[layer];
    }
    gpu.params1 = glm::vec4(p->refractionStrength, p->fresnelPower, p->transparency, p->reflectionStrength);
    gpu.params2 = glm::vec4(p->waterTint, p->noiseScale, static_cast<float>(p->noiseOctaves), p->noisePersistence);
    gpu.params3 = glm::vec4(p->noiseTimeSpeed, p->time, p->specularIntensity, p->specularPower);
    gpu.shallowColor = glm::vec4(p->shallowColor, p->waveDepthTransition);
    gpu.deepColor = glm::vec4(p->deepColor, p->glitterIntensity);
    gpu.waveParams = glm::vec4(0.0f, 0.0f, p->bumpAmplitude, p->depthFalloff);
    gpu.reserved1 = glm::vec4(p->enableReflection ? 1.0f : 0.0f,
                              p->enableRefraction ? 1.0f : 0.0f,
                              p->enableBlur ? 1.0f : 0.0f,
                              p->blurRadius);
    gpu.reserved2 = glm::vec4(static_cast<float>(p->blurSamples), p->volumeBlurRate, p->volumeBumpRate, p->uniformReflection ? 1.0f : 0.0f);
    gpu.causticColor = glm::vec4(p->causticColor, 0.0f);
    gpu.causticParams = glm::vec4(p->causticScale, p->causticIntensity, p->causticPower, p->causticDepthScale);
    gpu.causticExtraParams = glm::vec4(p->causticLineScale, p->causticLineMix, 0.0f, 0.0f);
    gpu.reserved3 = glm::vec4((solid360Renderer && solid360Renderer->getCube360CubeView() != VK_NULL_HANDLE) ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);

    size_t offset = static_cast<size_t>(layer) * sizeof(WaterParamsGPU);
    void* data = nullptr;
    vkMapMemory(appPtr->getDevice(), waterParamsBuffer.memory, offset, sizeof(WaterParamsGPU), 0, &data);
    memcpy(data, &gpu, sizeof(WaterParamsGPU));
    vkUnmapMemory(appPtr->getDevice(), waterParamsBuffer.memory);
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
    for (int i = 0; i < 2; ++i) {
        waterDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        waterNormalImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        waterMaskImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        waterGeomDepthImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

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

    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        createImage(VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    waterDepthImages[frameIdx], waterDepthMemories[frameIdx], waterDepthImageViews[frameIdx]);
        waterDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;

        // Create an alternate image view that swizzles the alpha (linear depth)
        // into RGB channels so ImGui can display depth as a grayscale image.
        VkImageViewCreateInfo vw{};
        vw.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vw.image = waterDepthImages[frameIdx];
        vw.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vw.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        vw.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vw.subresourceRange.baseMipLevel = 0;
        vw.subresourceRange.levelCount = 1;
        vw.subresourceRange.baseArrayLayer = 0;
        vw.subresourceRange.layerCount = 1;
        vw.components.r = VK_COMPONENT_SWIZZLE_A;
        vw.components.g = VK_COMPONENT_SWIZZLE_A;
        vw.components.b = VK_COMPONENT_SWIZZLE_A;
        vw.components.a = VK_COMPONENT_SWIZZLE_A;
        if (vkCreateImageView(app->getDevice(), &vw, nullptr, &waterDepthAlphaImageViews[frameIdx]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create waterDepthAlphaImageView");
        }
        app->resources.addImageView(waterDepthAlphaImageViews[frameIdx], "WaterRenderer: waterDepthAlphaImageView");

        createImage(VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    waterNormalImages[frameIdx], waterNormalMemories[frameIdx], waterNormalImageViews[frameIdx]);
        waterNormalImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;

        createImage(VK_FORMAT_R8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    waterMaskImages[frameIdx], waterMaskMemories[frameIdx], waterMaskImageViews[frameIdx]);
        waterMaskImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;

        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    waterGeomDepthImages[frameIdx], waterGeomDepthMemories[frameIdx], waterGeomDepthImageViews[frameIdx]);
        waterGeomDepthImageLayouts[frameIdx] = VK_IMAGE_LAYOUT_UNDEFINED;
        fprintf(stderr, "[WaterRenderer] waterGeomDepthImage[%d] = %p\n", frameIdx, (void*)waterGeomDepthImages[frameIdx]);

        // Back-face depth image will be created by WaterBackFaceRenderer
    }

    // Create back-face depth targets (owned by WaterBackFaceRenderer)
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, width, height);

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
    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        transitionImageLayout(waterDepthImages[frameIdx], waterDepthImageLayouts[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transitionImageLayout(waterNormalImages[frameIdx], waterNormalImageLayouts[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transitionImageLayout(waterMaskImages[frameIdx], waterMaskImageLayouts[frameIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transitionImageLayout(waterGeomDepthImages[frameIdx], waterGeomDepthImageLayouts[frameIdx], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Create per-frame water framebuffers using the dedicated water geometry depth buffer
    for (int frameIdx = 0; frameIdx < 2; ++frameIdx) {
        std::array<VkImageView, 4> waterAttachments = {
            waterDepthImageViews[frameIdx],
            waterNormalImageViews[frameIdx],
            waterMaskImageViews[frameIdx],
            waterGeomDepthImageViews[frameIdx]  // Use dedicated water geometry depth buffer
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
    for (int i = 0; i < 2; ++i) {
        waterDepthImages[i] = VK_NULL_HANDLE;
        waterDepthMemories[i] = VK_NULL_HANDLE;
        waterDepthImageViews[i] = VK_NULL_HANDLE;
        waterDepthAlphaImageViews[i] = VK_NULL_HANDLE;

        waterNormalImages[i] = VK_NULL_HANDLE;
        waterNormalMemories[i] = VK_NULL_HANDLE;
        waterNormalImageViews[i] = VK_NULL_HANDLE;

        waterMaskImages[i] = VK_NULL_HANDLE;
        waterMaskMemories[i] = VK_NULL_HANDLE;
        waterMaskImageViews[i] = VK_NULL_HANDLE;

        waterGeomDepthImages[i] = VK_NULL_HANDLE;
        waterGeomDepthMemories[i] = VK_NULL_HANDLE;
        waterGeomDepthImageViews[i] = VK_NULL_HANDLE;
    }
    if (backFaceRenderer) backFaceRenderer->destroyRenderTargets(app);

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
    defaultWaterParams.waveParams = glm::vec4(0.0f, 0.0f, 8.0f, 0.1f);
    defaultWaterParams.reserved1 = glm::vec4(1.0f, 1.0f, 1.0f, 8.0f);
    defaultWaterParams.reserved2 = glm::vec4(4.0f, 0.004f, 0.05f, 0.0f);
    defaultWaterParams.reserved3 = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    // Caustic defaults: subtle warm caustics
    defaultWaterParams.causticColor = glm::vec4(1.0f, 0.98f, 0.8f, 0.0f);
    defaultWaterParams.causticParams = glm::vec4(40.0f, 1.5f, 1.0f, 4.0f); // w = causticDepthScale
    defaultWaterParams.causticExtraParams = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f); // lineScale=1, lineMix=1 (lines)
    
    void* data;
    vkMapMemory(device, waterParamsBuffer.memory, 0, sizeof(WaterParamsGPU), 0, &data);
    memcpy(data, &defaultWaterParams, sizeof(WaterParamsGPU));
    vkUnmapMemory(device, waterParamsBuffer.memory);
    
    std::cout << "[WaterRenderer] Initialized water params buffer with default values" << std::endl;
    
    // Create descriptor set layout for scene textures (set 2)
    // Binding 0: Scene color texture
    // Binding 1: Scene depth texture
    // Binding 2: Sky color texture
    // Binding 3: Water back-face depth texture (for volume thickness)
    std::array<VkDescriptorSetLayoutBinding, 5> sceneBindings{};
    
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

    // Water back-face depth (binding 3) — for water volume thickness
    sceneBindings[3].binding = 3;
    sceneBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[3].descriptorCount = 1;
    sceneBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[3].pImmutableSamplers = nullptr;

    // Optional cubemap sampler (binding 4) — used by water shader when solid 360 is available
    sceneBindings[4].binding = 4;
    sceneBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sceneBindings[4].descriptorCount = 1;
    sceneBindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[4].pImmutableSamplers = nullptr;
    
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
    depthPoolSize.descriptorCount = 10;  // Now including cubemap sampler (binding 4) per frame: (color + depth + sky + backfaceDepth + cube) * 2
    
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

    // Back-face pipeline creation moved to WaterBackFaceRenderer
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

void WaterRenderer::renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!app || cmd == VK_NULL_HANDLE) return;
    if (!backFaceRenderer) return;
    // Forward to the back-face renderer, providing necessary pipeline layout and descriptor sets
    backFaceRenderer->renderBackFacePass(app, cmd, frameIndex,
                                         waterIndirectRenderer,
                                         waterGeometryPipelineLayout,
                                         app->getMainDescriptorSet(),
                                         app->getMaterialDescriptorSet(),
                                         waterDepthDescriptorSets[frameIndex]);
}

void WaterRenderer::updateSceneTexturesBinding(VulkanApp* app, VkImageView colorImageView, VkImageView depthImageView, uint32_t frameIndex, VkImageView skyImageView) {
    if (waterDepthDescriptorSets[frameIndex] == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return;
    }
    
    // Use the provided image views directly
    VkImageView colorView = colorImageView;
    VkImageView depthView = depthImageView;
    
    std::array<VkDescriptorImageInfo, 5> imageInfos{};
    
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

    // Water back-face depth (binding 3) — will be in SHADER_READ_ONLY after back-face pass
    imageInfos[3].sampler = linearSampler;
    imageInfos[3].imageView = (backFaceRenderer && backFaceRenderer->getBackFaceDepthView(frameIndex) != VK_NULL_HANDLE) ? backFaceRenderer->getBackFaceDepthView(frameIndex) : depthView;
    imageInfos[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Cubemap (binding 4) — if available, bind cube360CubeView, otherwise leave as fallback to scene color
    imageInfos[4].sampler = linearSampler;
    imageInfos[4].imageView = (solid360Renderer && solid360Renderer->getCube360CubeView() != VK_NULL_HANDLE) ? solid360Renderer->getCube360CubeView() : colorView;
    imageInfos[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    std::array<VkWriteDescriptorSet, 5> writes{};
    
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

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = waterDepthDescriptorSets[frameIndex];
    writes[3].dstBinding = 3;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].descriptorCount = 1;
    writes[3].pImageInfo = &imageInfos[3];

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = waterDepthDescriptorSets[frameIndex];
    writes[4].dstBinding = 4;
    writes[4].dstArrayElement = 0;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].descriptorCount = 1;
    writes[4].pImageInfo = &imageInfos[4];
    vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


// Execute water's offscreen geometry pass on the provided command buffer.
// The caller must ensure that the solid pass has already ended on this same
// command buffer so that the scene color/depth images are available.

void WaterRenderer::prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView sceneColorView, VkImageView sceneDepthView, const WaterParams& params, float waterTime, VkImageView skyView) {
    if (!app || cmd == VK_NULL_HANDLE) return;

    // Update per-layer dynamic fields (time and cube availability) in the GPU SSBO
    if (app && waterParamsBuffer.buffer != VK_NULL_HANDLE && waterParamsCount > 0) {
        // Map entire SSBO and update time/reserved3.x for every entry
        void* mapped = nullptr;
        size_t totalSize = sizeof(WaterParamsGPU) * static_cast<size_t>(waterParamsCount);
        vkMapMemory(app->getDevice(), waterParamsBuffer.memory, 0, totalSize, 0, &mapped);
        for (uint32_t i = 0; i < waterParamsCount; ++i) {
            WaterParamsGPU gpu{};
            // Pack CPU-side paramsList[i] into gpu struct but only update dynamic fields when necessary
                    WaterParams *p = nullptr;
                    if (externalParams) {
                        p = &(*externalParams)[i];
                    } else {
                        p = &paramsList[i];
                    }
                    gpu.params1 = glm::vec4(p->refractionStrength, p->fresnelPower, p->transparency, p->reflectionStrength);
                    gpu.params2 = glm::vec4(p->waterTint, p->noiseScale, static_cast<float>(p->noiseOctaves), p->noisePersistence);
                    gpu.params3 = glm::vec4(p->noiseTimeSpeed, p->time, p->specularIntensity, p->specularPower);
                    gpu.shallowColor = glm::vec4(p->shallowColor, p->waveDepthTransition);
                    gpu.deepColor = glm::vec4(p->deepColor, p->glitterIntensity);
                    gpu.waveParams = glm::vec4(0.0f, 0.0f, p->bumpAmplitude, p->depthFalloff);
                    gpu.reserved1 = glm::vec4(p->enableReflection ? 1.0f : 0.0f,
                                              p->enableRefraction ? 1.0f : 0.0f,
                                              p->enableBlur ? 1.0f : 0.0f,
                                              p->blurRadius);
                    gpu.reserved2 = glm::vec4(static_cast<float>(p->blurSamples), p->volumeBlurRate, p->volumeBumpRate, p->uniformReflection ? 1.0f : 0.0f);
                    gpu.causticColor = glm::vec4(p->causticColor, 0.0f);
                    gpu.causticParams = glm::vec4(p->causticScale, p->causticIntensity, p->causticPower, p->causticDepthScale);
                    gpu.causticExtraParams = glm::vec4(p->causticLineScale, p->causticLineMix, 0.0f, 0.0f);
                    gpu.reserved3 = glm::vec4((solid360Renderer && solid360Renderer->getCube360CubeView() != VK_NULL_HANDLE) ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);

            memcpy(static_cast<char*>(mapped) + i * sizeof(WaterParamsGPU), &gpu, sizeof(WaterParamsGPU));
        }
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

// Back-face pass implementation moved to WaterBackFaceRenderer
void WaterRenderer::postRenderBarrier(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (cmd == VK_NULL_HANDLE) return;

    // Build image barriers for per-frame water outputs so fragment shader
    // sampling in the swapchain render pass sees completed color writes.
    std::vector<VkImageMemoryBarrier> barriers;

    auto pushColorBarrier = [&](VkImage img) {
        if (img == VK_NULL_HANDLE) return;
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = 1;
        barriers.push_back(b);
    };

    // color outputs
    pushColorBarrier(waterDepthImages[frameIndex]);
    pushColorBarrier(waterNormalImages[frameIndex]);
    pushColorBarrier(waterMaskImages[frameIndex]);

    // Depth-based back-face image (ensure depth writes are visible to shader reads)
    if (backFaceRenderer) {
        backFaceRenderer->postRenderBarrier(cmd, frameIndex);
    }

    if (!barriers.empty()) {
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data());
    }
}

void WaterRenderer::render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, VkImageView sceneColorView, VkImageView sceneDepthView, const WaterParams& params, float waterTime, VkImageView skyView) {
    if (!app || cmd == VK_NULL_HANDLE) return;

    prepareRender(app, cmd, frameIndex, sceneColorView, sceneDepthView, params, waterTime, skyView);

    // Back-face depth pre-pass (reversed winding for water volume thickness)
    if (backFaceRenderer) {
        backFaceRenderer->renderBackFacePass(app, cmd, frameIndex,
                                             waterIndirectRenderer,
                                             waterGeometryPipelineLayout,
                                             app->getMainDescriptorSet(),
                                             app->getMaterialDescriptorSet(),
                                             waterDepthDescriptorSets[frameIndex]);
    }

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

    postRenderBarrier(cmd, frameIndex);
}

// ============================================================================
// Solid 360° cubemap reflection
// ============================================================================

void WaterRenderer::createSolid360Targets(VulkanApp* app, VkRenderPass solidRenderPass) {
    if (solid360Renderer) solid360Renderer->createSolid360Targets(app, solidRenderPass, linearSampler);
}

void WaterRenderer::destroySolid360Targets(VulkanApp* app) {
    if (solid360Renderer) solid360Renderer->destroySolid360Targets(app);
}

void WaterRenderer::renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                                    VkRenderPass solidRenderPass,
                                    SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                                    SolidRenderer* solidRenderer,
                                    VkDescriptorSet mainDescriptorSet,
                                    Buffer& uniformBuffer, const UniformObject& ubo) {
    if (solid360Renderer) solid360Renderer->renderSolid360(app, cmd, solidRenderPass, skyRenderer, skyMode, solidRenderer, mainDescriptorSet, uniformBuffer, ubo);
}
