#include "WaterRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <iostream>
#include <array>

WaterRenderer::WaterRenderer(VulkanApp* app) : app(app) {}

WaterRenderer::~WaterRenderer() {}

void WaterRenderer::init() {
    waterIndirectRenderer.init(app);
    createSamplers();
    createWaterRenderPass();
    createSceneRenderPass();
    createWaterPipelines();
    createPostProcessPipeline();
    createDescriptorSets();
}

void WaterRenderer::cleanup() {
    VkDevice device = app->getDevice();
    
    waterIndirectRenderer.cleanup(app);
    
    destroyRenderTargets();
    
    if (waterGeometryPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, waterGeometryPipeline, nullptr);
        waterGeometryPipeline = VK_NULL_HANDLE;
    }
    
    if (waterPostProcessPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, waterPostProcessPipeline, nullptr);
        waterPostProcessPipeline = VK_NULL_HANDLE;
    }
    
    if (waterPostProcessPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, waterPostProcessPipelineLayout, nullptr);
        waterPostProcessPipelineLayout = VK_NULL_HANDLE;
    }
    
    if (postProcessDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, postProcessDescriptorPool, nullptr);
        postProcessDescriptorPool = VK_NULL_HANDLE;
    }
    
    if (postProcessDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, postProcessDescriptorSetLayout, nullptr);
        postProcessDescriptorSetLayout = VK_NULL_HANDLE;
    }
    
    if (waterDepthDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, waterDepthDescriptorPool, nullptr);
        waterDepthDescriptorPool = VK_NULL_HANDLE;
    }
    
    if (waterDepthDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, waterDepthDescriptorSetLayout, nullptr);
        waterDepthDescriptorSetLayout = VK_NULL_HANDLE;
    }
    
    // Destroy water pipeline layout (we create it ourselves now)
    if (waterGeometryPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, waterGeometryPipelineLayout, nullptr);
        waterGeometryPipelineLayout = VK_NULL_HANDLE;
    }
    
    if (waterRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, waterRenderPass, nullptr);
        waterRenderPass = VK_NULL_HANDLE;
    }
    
    if (sceneRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, sceneRenderPass, nullptr);
        sceneRenderPass = VK_NULL_HANDLE;
    }
    
    if (linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, linearSampler, nullptr);
        linearSampler = VK_NULL_HANDLE;
    }
    
    if (waterUniformBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, waterUniformBuffer.buffer, nullptr);
        waterUniformBuffer.buffer = VK_NULL_HANDLE;
    }
    if (waterUniformBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, waterUniformBuffer.memory, nullptr);
        waterUniformBuffer.memory = VK_NULL_HANDLE;
    }
}

void WaterRenderer::createSamplers() {
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
}

void WaterRenderer::createSceneRenderPass() {
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
}

void WaterRenderer::createWaterRenderPass() {
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
    
    // Depth buffer - LOAD from scene pass (don't clear!)
    attachments[3].format = VK_FORMAT_D32_SFLOAT;
    attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // Load scene depth!
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // Coming from scene pass
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
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;  // Scene wrote depth
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    
    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &waterRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water render pass!");
    }
}

void WaterRenderer::createRenderTargets(uint32_t width, uint32_t height) {
    if (renderWidth == width && renderHeight == height && waterFramebuffer != VK_NULL_HANDLE) {
        return; // Already created at this size
    }
    
    destroyRenderTargets();
    
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
    };
    
    // Create scene offscreen color target - use swapchain format for pipeline compatibility
    createImage(app->getSwapchainImageFormat(),
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                sceneColorImage, sceneColorMemory, sceneColorImageView);
    
    // Create scene offscreen depth target
    createImage(VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                sceneDepthImage, sceneDepthMemory, sceneDepthImageView);
    
    // Create scene framebuffer
    std::array<VkImageView, 2> sceneAttachments = {
        sceneColorImageView,
        sceneDepthImageView
    };
    
    VkFramebufferCreateInfo sceneFbInfo{};
    sceneFbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    sceneFbInfo.renderPass = sceneRenderPass;
    sceneFbInfo.attachmentCount = static_cast<uint32_t>(sceneAttachments.size());
    sceneFbInfo.pAttachments = sceneAttachments.data();
    sceneFbInfo.width = width;
    sceneFbInfo.height = height;
    sceneFbInfo.layers = 1;
    
    if (vkCreateFramebuffer(device, &sceneFbInfo, nullptr, &sceneFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create scene offscreen framebuffer!");
    }
    
    // Create water world position image (xyz=worldPos, w=linearDepth)
    createImage(VK_FORMAT_R32G32B32A32_SFLOAT, 
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                waterDepthImage, waterDepthMemory, waterDepthImageView);
    
    // Create water normal image
    createImage(VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                waterNormalImage, waterNormalMemory, waterNormalImageView);
    
    // Create water mask image
    createImage(VK_FORMAT_R8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                waterMaskImage, waterMaskMemory, waterMaskImageView);
    
    // NOTE: We do NOT create a separate depth buffer for water geometry pass
    // Instead, we reuse the scene depth buffer (sceneDepthImageView) so water
    // is properly depth-tested against the terrain
    
    // Create water framebuffer using scene depth buffer for depth testing
    std::array<VkImageView, 4> waterAttachments = {
        waterDepthImageView,
        waterNormalImageView,
        waterMaskImageView,
        sceneDepthImageView  // Use scene depth buffer!
    };
    
    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = waterRenderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(waterAttachments.size());
    fbInfo.pAttachments = waterAttachments.data();
    fbInfo.width = width;
    fbInfo.height = height;
    fbInfo.layers = 1;
    
    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &waterFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water framebuffer!");
    }
    
    // Create a copy of the depth buffer for sampling during water rendering
    // This avoids the issue of sampling from a depth attachment during the same pass
    createImage(VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                sceneDepthCopyImage, sceneDepthCopyMemory, sceneDepthCopyImageView);
    
    std::cout << "[WaterRenderer] Created render targets " << width << "x" << height << std::endl;
}

void WaterRenderer::destroyRenderTargets() {
    VkDevice device = app->getDevice();
    
    if (waterFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, waterFramebuffer, nullptr);
        waterFramebuffer = VK_NULL_HANDLE;
    }
    
    if (sceneFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, sceneFramebuffer, nullptr);
        sceneFramebuffer = VK_NULL_HANDLE;
    }
    
    auto destroyImage = [&](VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (image != VK_NULL_HANDLE) {
            vkDestroyImage(device, image, nullptr);
            image = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    };
    
    destroyImage(sceneColorImage, sceneColorMemory, sceneColorImageView);
    destroyImage(sceneDepthImage, sceneDepthMemory, sceneDepthImageView);
    destroyImage(sceneDepthCopyImage, sceneDepthCopyMemory, sceneDepthCopyImageView);
    destroyImage(waterDepthImage, waterDepthMemory, waterDepthImageView);
    destroyImage(waterNormalImage, waterNormalMemory, waterNormalImageView);
    destroyImage(waterMaskImage, waterMaskMemory, waterMaskImageView);
    destroyImage(waterGeomDepthImage, waterGeomDepthMemory, waterGeomDepthImageView);
}

void WaterRenderer::createWaterPipelines() {
    VkDevice device = app->getDevice();
    
    // Create uniform buffer for water params
    waterUniformBuffer = app->createBuffer(sizeof(WaterUBO), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Create descriptor set layout for scene depth texture (set 2, binding 0)
    VkDescriptorSetLayoutBinding depthBinding{};
    depthBinding.binding = 0;
    depthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    depthBinding.descriptorCount = 1;
    depthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    depthBinding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo depthLayoutInfo{};
    depthLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    depthLayoutInfo.bindingCount = 1;
    depthLayoutInfo.pBindings = &depthBinding;
    
    if (vkCreateDescriptorSetLayout(device, &depthLayoutInfo, nullptr, &waterDepthDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water depth descriptor set layout!");
    }
    
    // Create descriptor pool for depth texture
    VkDescriptorPoolSize depthPoolSize{};
    depthPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    depthPoolSize.descriptorCount = 1;
    
    VkDescriptorPoolCreateInfo depthPoolInfo{};
    depthPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    depthPoolInfo.poolSizeCount = 1;
    depthPoolInfo.pPoolSizes = &depthPoolSize;
    depthPoolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(device, &depthPoolInfo, nullptr, &waterDepthDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water depth descriptor pool!");
    }
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo depthAllocInfo{};
    depthAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    depthAllocInfo.descriptorPool = waterDepthDescriptorPool;
    depthAllocInfo.descriptorSetCount = 1;
    depthAllocInfo.pSetLayouts = &waterDepthDescriptorSetLayout;
    
    if (vkAllocateDescriptorSets(device, &depthAllocInfo, &waterDepthDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate water depth descriptor set!");
    }
    
    // Create a custom pipeline layout for water that includes:
    // Set 0: Material SSBO (from app->getMaterialDescriptorSetLayout())
    // Set 1: UBO (from app->getDescriptorSetLayout())
    // Set 2: Scene depth texture (waterDepthDescriptorSetLayout)
    std::array<VkDescriptorSetLayout, 3> waterSetLayouts = {
        app->getMaterialDescriptorSetLayout(),  // Set 0: Materials SSBO
        app->getDescriptorSetLayout(),           // Set 1: UBO
        waterDepthDescriptorSetLayout            // Set 2: Scene depth texture
    };
    
    // Push constants (same as main pipeline - model matrix)
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | 
                                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4);  // model matrix
    
    VkPipelineLayoutCreateInfo waterPipelineLayoutInfo{};
    waterPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    waterPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(waterSetLayouts.size());
    waterPipelineLayoutInfo.pSetLayouts = waterSetLayouts.data();
    waterPipelineLayoutInfo.pushConstantRangeCount = 1;
    waterPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    if (vkCreatePipelineLayout(device, &waterPipelineLayoutInfo, nullptr, &waterGeometryPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water pipeline layout!");
    }
    
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
    
    VkPipelineTessellationStateCreateInfo tessState{};
    tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tessState.patchControlPoints = 3;
    
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;  // No culling for water
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;   // Test against scene depth
    depthStencil.depthWriteEnable = VK_FALSE; // Don't write depth (water is transparent)
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;
    
    // Single color attachment for swapchain with alpha blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    
    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();
    
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pTessellationState = hasTessellation ? &tessState : nullptr;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = waterGeometryPipelineLayout;  // Use water pipeline layout with depth texture
    pipelineInfo.renderPass = app->getContinuationRenderPass();  // Use continuation render pass (LOAD_OP_LOAD)
    pipelineInfo.subpass = 0;
    
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &waterGeometryPipeline) != VK_SUCCESS) {
        std::cerr << "[WaterRenderer] Warning: Failed to create water geometry pipeline" << std::endl;
        waterGeometryPipeline = VK_NULL_HANDLE;
    } else {
        std::cout << "[WaterRenderer] Created water geometry pipeline" << std::endl;
    }
    
    // Cleanup shader modules
    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    if (tescModule) vkDestroyShaderModule(device, tescModule, nullptr);
    if (teseModule) vkDestroyShaderModule(device, teseModule, nullptr);
}

void WaterRenderer::createPostProcessPipeline() {
    // Post-process pipeline is no longer used - water rendering is done in a single geometry pass
    // with depth-based edge foam calculated directly in the water fragment shader
    return;
    
    VkDevice device = app->getDevice();
    
    // Create descriptor set layout for post-process
    // Bindings:
    // 0: Scene color (sampler2D)
    // 1: Scene depth (sampler2D)
    // 2: Water depth (sampler2D)
    // 3: Water normal (sampler2D)
    // 4: Water mask (sampler2D)
    // 5: Water UBO (uniform buffer)
    
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    
    for (int i = 0; i < 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &postProcessDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water post-process descriptor set layout!");
    }
    
    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &postProcessDescriptorSetLayout;
    
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &waterPostProcessPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water post-process pipeline layout!");
    }
    
    // Load shaders
    auto vertCode = FileReader::readFile("shaders/fullscreen.vert.spv");
    auto fragCode = FileReader::readFile("shaders/water_postprocess.frag.spv");
    
    if (vertCode.empty() || fragCode.empty()) {
        std::cerr << "[WaterRenderer] Warning: Could not load post-process shaders, skipping pipeline creation" << std::endl;
        return;
    }
    
    VkShaderModule vertModule = app->createShaderModule(vertCode);
    VkShaderModule fragModule = app->createShaderModule(fragCode);
    
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertModule;
    shaderStages[0].pName = "main";
    
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragModule;
    shaderStages[1].pName = "main";
    
    // No vertex input (fullscreen triangle generated in shader)
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    
    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();
    
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = waterPostProcessPipelineLayout;
    pipelineInfo.renderPass = app->getSwapchainRenderPass();  // Use swapchain render pass
    pipelineInfo.subpass = 0;
    
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &waterPostProcessPipeline) != VK_SUCCESS) {
        std::cerr << "[WaterRenderer] Warning: Failed to create post-process pipeline" << std::endl;
        waterPostProcessPipeline = VK_NULL_HANDLE;
    }
    
    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    
    if (waterPostProcessPipeline != VK_NULL_HANDLE) {
        std::cout << "[WaterRenderer] Created post-process pipeline" << std::endl;
    }
}

void WaterRenderer::createDescriptorSets() {
    if (postProcessDescriptorSetLayout == VK_NULL_HANDLE) return;
    
    VkDevice device = app->getDevice();
    
    // Create descriptor pool
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 5;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &postProcessDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create water post-process descriptor pool!");
    }
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = postProcessDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &postProcessDescriptorSetLayout;
    
    if (vkAllocateDescriptorSets(device, &allocInfo, &postProcessDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate water post-process descriptor set!");
    }
}

void WaterRenderer::initDepthCopyResources(uint32_t width, uint32_t height) {
    if (sceneDepthCopyImage != VK_NULL_HANDLE && renderWidth == width && renderHeight == height) {
        return; // Already initialized at this size
    }
    
    VkDevice device = app->getDevice();
    
    // Destroy existing resources
    if (sceneDepthCopyImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, sceneDepthCopyImageView, nullptr);
        sceneDepthCopyImageView = VK_NULL_HANDLE;
    }
    if (sceneDepthCopyImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, sceneDepthCopyImage, nullptr);
        sceneDepthCopyImage = VK_NULL_HANDLE;
    }
    if (sceneDepthCopyMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, sceneDepthCopyMemory, nullptr);
        sceneDepthCopyMemory = VK_NULL_HANDLE;
    }
    
    renderWidth = width;
    renderHeight = height;
    
    // Create depth copy image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateImage(device, &imageInfo, nullptr, &sceneDepthCopyImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth copy image!");
    }
    
    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, sceneDepthCopyImage, &memReq);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &sceneDepthCopyMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate depth copy memory!");
    }
    
    vkBindImageMemory(device, sceneDepthCopyImage, sceneDepthCopyMemory, 0);
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = sceneDepthCopyImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_D32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(device, &viewInfo, nullptr, &sceneDepthCopyImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth copy image view!");
    }
    
    // Update descriptor set with the depth copy image
    updateSceneDepthBinding(sceneDepthCopyImageView);
    
    std::cout << "[WaterRenderer] Created depth copy resources " << width << "x" << height << std::endl;
}

void WaterRenderer::beginWaterGeometryPass(VkCommandBuffer cmd) {
    if (waterFramebuffer == VK_NULL_HANDLE) return;
    
    std::array<VkClearValue, 4> clearValues{};
    clearValues[0].color = {{1000.0f, 0.0f, 0.0f, 0.0f}}; // Far depth
    clearValues[1].color = {{0.0f, 1.0f, 0.0f, 0.0f}};    // Up normal
    clearValues[2].color = {{0.0f, 0.0f, 0.0f, 0.0f}};    // No water mask
    clearValues[3].depthStencil = {1.0f, 0};
    
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = waterRenderPass;
    renderPassInfo.framebuffer = waterFramebuffer;
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
    vkCmdEndRenderPass(cmd);
}

void WaterRenderer::renderWaterPostProcess(VkCommandBuffer cmd, VkFramebuffer swapchainFramebuffer,
                                            VkRenderPass swapchainRenderPass,
                                            VkImageView sceneColorView, VkImageView sceneDepthView,
                                            const WaterParams& params,
                                            const glm::mat4& viewProj, const glm::mat4& invViewProj,
                                            const glm::vec3& viewPos, float time) {
    if (waterPostProcessPipeline == VK_NULL_HANDLE) return;
    
    VkDevice device = app->getDevice();
    
    // Update water UBO
    WaterUBO ubo{};
    ubo.viewProjection = viewProj;
    ubo.invViewProjection = invViewProj;
    ubo.viewPos = glm::vec4(viewPos, 1.0f);
    ubo.waterParams1 = glm::vec4(time, params.waveSpeed, params.waveScale, params.refractionStrength);
    ubo.waterParams2 = glm::vec4(params.fresnelPower, params.transparency, params.depthFalloff, params.noiseScale);
    ubo.shallowColor = glm::vec4(params.shallowColor, 0.0f);
    ubo.deepColor = glm::vec4(params.deepColor, static_cast<float>(params.noiseOctaves));
    ubo.screenSize = glm::vec4(renderWidth, renderHeight, 1.0f / renderWidth, 1.0f / renderHeight);
    ubo.noisePersistence = glm::vec4(params.noisePersistence, 0.0f, 0.0f, 0.0f);
    
    void* data;
    vkMapMemory(device, waterUniformBuffer.memory, 0, sizeof(WaterUBO), 0, &data);
    memcpy(data, &ubo, sizeof(WaterUBO));
    vkUnmapMemory(device, waterUniformBuffer.memory);
    
    // Update descriptor set with current images
    std::array<VkDescriptorImageInfo, 5> imageInfos{};
    imageInfos[0] = {linearSampler, sceneColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[1] = {linearSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[2] = {linearSampler, waterDepthImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[3] = {linearSampler, waterNormalImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[4] = {linearSampler, waterMaskImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    
    VkDescriptorBufferInfo bufferInfo{waterUniformBuffer.buffer, 0, sizeof(WaterUBO)};
    
    std::array<VkWriteDescriptorSet, 6> writes{};
    for (int i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = postProcessDescriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &imageInfos[i];
    }
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = postProcessDescriptorSet;
    writes[5].dstBinding = 5;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[5].descriptorCount = 1;
    writes[5].pBufferInfo = &bufferInfo;
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    // Begin render pass (output to swapchain)
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = swapchainRenderPass;
    renderPassInfo.framebuffer = swapchainFramebuffer;
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
    
    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterPostProcessPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterPostProcessPipelineLayout, 
                            0, 1, &postProcessDescriptorSet, 0, nullptr);
    
    // Draw fullscreen triangle (3 vertices, no vertex buffer needed)
    vkCmdDraw(cmd, 3, 1, 0, 0);
    
    // NOTE: Render pass is NOT ended here - caller is responsible for ending it
    // This allows ImGui or other overlays to be rendered in the same pass
}

void WaterRenderer::updateSceneDepthBinding(VkImageView depthImageView) {
    if (waterDepthDescriptorSet == VK_NULL_HANDLE || linearSampler == VK_NULL_HANDLE) {
        return;
    }
    
    // Use the depth copy image view for sampling (avoid reading from depth attachment during render)
    VkImageView viewToUse = sceneDepthCopyImageView != VK_NULL_HANDLE ? sceneDepthCopyImageView : depthImageView;
    
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = linearSampler;
    imageInfo.imageView = viewToUse;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = waterDepthDescriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    
    vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);
}

void WaterRenderer::copySceneDepth(VkCommandBuffer cmd, VkImage srcDepthImage, uint32_t width, uint32_t height) {
    if (sceneDepthCopyImage == VK_NULL_HANDLE) {
        return;
    }
    
    // Transition source depth from DEPTH_STENCIL_ATTACHMENT to TRANSFER_SRC
    VkImageMemoryBarrier srcBarrier{};
    srcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    srcBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    srcBarrier.image = srcDepthImage;
    srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    srcBarrier.subresourceRange.baseMipLevel = 0;
    srcBarrier.subresourceRange.levelCount = 1;
    srcBarrier.subresourceRange.baseArrayLayer = 0;
    srcBarrier.subresourceRange.layerCount = 1;
    srcBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    
    // Transition dest depth from UNDEFINED to TRANSFER_DST
    VkImageMemoryBarrier dstBarrier{};
    dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dstBarrier.image = sceneDepthCopyImage;
    dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    dstBarrier.subresourceRange.baseMipLevel = 0;
    dstBarrier.subresourceRange.levelCount = 1;
    dstBarrier.subresourceRange.baseArrayLayer = 0;
    dstBarrier.subresourceRange.layerCount = 1;
    dstBarrier.srcAccessMask = 0;
    dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    
    std::array<VkImageMemoryBarrier, 2> barriers = {srcBarrier, dstBarrier};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data());
    
    // Copy depth image
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcOffset = {0, 0, 0};
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.dstSubresource.mipLevel = 0;
    copyRegion.dstSubresource.baseArrayLayer = 0;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.dstOffset = {0, 0, 0};
    copyRegion.extent = {width, height, 1};
    
    vkCmdCopyImage(cmd,
        srcDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        sceneDepthCopyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copyRegion);
    
    // Transition source back to DEPTH_STENCIL_ATTACHMENT
    srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    srcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    srcBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    // Transition dest to SHADER_READ_ONLY
    dstBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dstBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dstBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    barriers = {srcBarrier, dstBarrier};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data());
}
