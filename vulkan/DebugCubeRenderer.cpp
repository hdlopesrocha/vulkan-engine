#include "DebugCubeRenderer.hpp"
#include "VertexBufferObjectBuilder.hpp"
#include "ShaderStage.hpp"
#include "../utils/FileReader.hpp"
#include "../math/BoxLineGeometry.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <stb/stb_image.h>

DebugCubeRenderer::DebugCubeRenderer(VulkanApp* app_) : app(app_) {}

DebugCubeRenderer::~DebugCubeRenderer() { cleanup(); }

void DebugCubeRenderer::init(VkRenderPass renderPassOverride) {
    // Create cube VBO
    createCubeVBO();
    
    // Load grid texture
    loadGridTexture();
    
    // Create descriptor set for grid texture
    createGridDescriptorSet();
    
    // Create shader modules
    vertModule = app->createShaderModule(FileReader::readFile("shaders/debug_cube.vert.spv"));
    fragModule = app->createShaderModule(FileReader::readFile("shaders/debug_cube.frag.spv"));
    
    ShaderStage vertStage(vertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragStage(fragModule, VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Setup descriptor set layouts: set 0 = UBO, set 1 = grid texture + instance buffer
    std::vector<VkDescriptorSetLayout> setLayouts = {
        app->getDescriptorSetLayout(),  // set 0: UBO
        gridDescriptorSetLayout          // set 1: grid texture + instance buffer
    };
    
    // Create pipeline with alpha blending enabled for transparency
    VkPipeline pipelineHandle;
    VkPipelineLayout layoutHandle;
    std::tie(pipelineHandle, layoutHandle) = app->createGraphicsPipeline(
        { vertStage.info, fragStage.info },
        std::vector<VkVertexInputBindingDescription>{ 
            VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } 
        },
        {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, texCoord) }
        },
        setLayouts,
        nullptr,  // No push constants
        VK_POLYGON_MODE_LINE,      // Wireframe mode
        VK_CULL_MODE_NONE,         // No culling for wireframe
        true,                      // Enable blending
        true,                      // Depth test enabled
        VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,  // Lines for wireframe
        renderPassOverride
    );
    
    pipeline = pipelineHandle;
    pipelineLayout = layoutHandle;
    
    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to create pipeline!\n");
    }
}

void DebugCubeRenderer::createCubeVBO() {
    // Create a unit cube with min=(0,0,0) and max=(1,1,1)
    BoundingBox unitBox(glm::vec3(0.0f), glm::vec3(1.0f));
    BoxLineGeometry cubeGeom(unitBox);
    
    cubeVBO = VertexBufferObjectBuilder::create(app, cubeGeom);
    printf("[DebugCubeRenderer] Created cube VBO: vertices=%zu indices=%zu\n", cubeGeom.vertices.size(), cubeGeom.indices.size());
}

void DebugCubeRenderer::loadGridTexture() {
    // Load grid.png texture
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load("textures/grid.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    
    if (!pixels) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to load grid texture!\n");
        return;
    }
    
    printf("[DEBUG CUBE RENDERER] Loaded grid texture: %dx%d, channels=%d\n", texWidth, texHeight, texChannels);
    
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    
    // Create staging buffer
    Buffer stagingBuffer = app->createBuffer(imageSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    void* data;
    vkMapMemory(app->getDevice(), stagingBuffer.memory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(app->getDevice(), stagingBuffer.memory);
    
    stbi_image_free(pixels);
    
    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = texWidth;
    imageInfo.extent.height = texHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateImage(app->getDevice(), &imageInfo, nullptr, &gridTextureImage) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to create grid texture image!\n");
        return;
    }
    printf("[DebugCubeRenderer] createImage: gridTextureImage=%p\n", (void*)gridTextureImage);
    // Register grid texture image
    app->resources.addImage(gridTextureImage, "DebugCubeRenderer: gridTextureImage");
    
    // Allocate memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(app->getDevice(), gridTextureImage, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = app->findMemoryType(memRequirements.memoryTypeBits, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(app->getDevice(), &allocInfo, nullptr, &gridTextureMemory) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to allocate grid texture memory!\n");
        return;
    }
    printf("[DebugCubeRenderer] allocateMemory: gridTextureMemory=%p\n", (void*)gridTextureMemory);
    
    vkBindImageMemory(app->getDevice(), gridTextureImage, gridTextureMemory, 0);
    // Register grid texture memory
    app->resources.addDeviceMemory(gridTextureMemory, "DebugCubeRenderer: gridTextureMemory");
    
    // Transition and copy
    app->transitionImageLayout(gridTextureImage, VK_FORMAT_R8G8B8A8_SRGB, 
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    app->copyBufferToImage(stagingBuffer.buffer, gridTextureImage, texWidth, texHeight);
    app->transitionImageLayout(gridTextureImage, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    // Defer actual destruction to VulkanResourceManager; clear local handles
    stagingBuffer.buffer = VK_NULL_HANDLE;
    stagingBuffer.memory = VK_NULL_HANDLE;
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = gridTextureImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &gridTextureView) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to create grid texture view!\n");
        return;
    }
    printf("[DebugCubeRenderer] createImageView: gridTextureView=%p for image=%p\n", (void*)gridTextureView, (void*)gridTextureImage);
    // Register grid texture view
    app->resources.addImageView(gridTextureView, "DebugCubeRenderer: gridTextureView");
    
    // Create sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    
    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &gridTextureSampler) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to create grid texture sampler!\n");
    }
    printf("[DebugCubeRenderer] createSampler: gridTextureSampler=%p\n", (void*)gridTextureSampler);
    // Register grid texture sampler
    app->resources.addSampler(gridTextureSampler, "DebugCubeRenderer: gridTextureSampler");
    
    printf("[DebugCubeRenderer] Loaded grid texture: %dx%d\n", texWidth, texHeight);
}

void DebugCubeRenderer::createGridDescriptorSet() {
    // Create descriptor set layout for texture (binding 1) and instance buffer (binding 2)
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    VkDescriptorSetLayoutBinding instanceBufferBinding{};
    instanceBufferBinding.binding = 2;
    instanceBufferBinding.descriptorCount = 1;
    instanceBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    instanceBufferBinding.pImmutableSamplers = nullptr;
    instanceBufferBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    
    VkDescriptorSetLayoutBinding bindings[] = { samplerLayoutBinding, instanceBufferBinding };
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    
    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &gridDescriptorSetLayout) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to create grid descriptor set layout!\n");
        return;
    }
    printf("[DebugCubeRenderer] createDescriptorSetLayout: gridDescriptorSetLayout=%p\n", (void*)gridDescriptorSetLayout);
    // Register descriptor set layout
    app->resources.addDescriptorSetLayout(gridDescriptorSetLayout, "DebugCubeRenderer: gridDescriptorSetLayout");
    
    // Create descriptor pool
    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &gridDescriptorPool) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to create grid descriptor pool!\n");
        return;
    }
    printf("[DebugCubeRenderer] createDescriptorPool: gridDescriptorPool=%p\n", (void*)gridDescriptorPool);
    // Register descriptor pool
    app->resources.addDescriptorPool(gridDescriptorPool, "DebugCubeRenderer: gridDescriptorPool");
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = gridDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &gridDescriptorSetLayout;
    
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &gridDescriptorSet) != VK_SUCCESS) {
        fprintf(stderr, "[DEBUG CUBE RENDERER ERROR] Failed to allocate grid descriptor set!\n");
        return;
    }
    // Register descriptor set for tracking
    app->resources.addDescriptorSet(gridDescriptorSet, "DebugCubeRenderer: gridDescriptorSet");
    
    // Update descriptor set with texture
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = gridTextureView;
    imageInfo.sampler = gridTextureSampler;
    
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = gridDescriptorSet;
    descriptorWrite.dstBinding = 1;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;
    
    vkUpdateDescriptorSets(app->getDevice(), 1, &descriptorWrite, 0, nullptr);
    
    // Create initial instance buffer (will be resized as needed)
    instanceBufferCapacity = 128;
    instanceBuffer = app->createBuffer(
        instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4)),  // mat4 + vec4 for alignment
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    // Update descriptor set with instance buffer
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = instanceBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet bufferWrite{};
    bufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    bufferWrite.dstSet = gridDescriptorSet;
    bufferWrite.dstBinding = 2;
    bufferWrite.dstArrayElement = 0;
    bufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;
    
    vkUpdateDescriptorSets(app->getDevice(), 1, &bufferWrite, 0, nullptr);
}

void DebugCubeRenderer::updateInstanceBuffer() {
    if (activeCubes.empty()) return;
    
    // Resize buffer if needed
    if (activeCubes.size() > instanceBufferCapacity) {
        if (instanceBuffer.buffer != VK_NULL_HANDLE) {
                // Defer destruction to VulkanResourceManager; clear local handles
                instanceBuffer = {};
            }
        
        instanceBufferCapacity = activeCubes.size() * 2;  // Allocate extra
        instanceBuffer = app->createBuffer(
            instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        
        // Update descriptor with new buffer
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = instanceBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;
        
        VkWriteDescriptorSet bufferWrite{};
        bufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bufferWrite.dstSet = gridDescriptorSet;
        bufferWrite.dstBinding = 2;
        bufferWrite.dstArrayElement = 0;
        bufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bufferWrite.descriptorCount = 1;
        bufferWrite.pBufferInfo = &bufferInfo;
        
        vkUpdateDescriptorSets(app->getDevice(), 1, &bufferWrite, 0, nullptr);
    }
    
    // Upload instance data
    struct InstanceData {
        glm::mat4 model;
        glm::vec4 color;  // vec4 for alignment
    };
    
    if (activeCubes.empty()) return;
    
    std::vector<InstanceData> instanceData;
    instanceData.reserve(activeCubes.size());
    for (const auto& cube : activeCubes) {
        InstanceData inst;
        // Build transformation matrix from cube min/max and per-axis size
        // The unit cube goes from (0,0,0) to (1,1,1), so we need to scale by per-axis lengths
        glm::vec3 min = cube.cube.getMin();
        glm::vec3 sizes = cube.cube.getLength();
        inst.model = glm::translate(glm::mat4(1.0f), min)
                   * glm::scale(glm::mat4(1.0f), sizes);
        inst.color = glm::vec4(cube.color, 1.0f);
        instanceData.push_back(inst);
    }
    
    void* data;
    vkMapMemory(app->getDevice(), instanceBuffer.memory, 0, instanceData.size() * sizeof(InstanceData), 0, &data);
    memcpy(data, instanceData.data(), instanceData.size() * sizeof(InstanceData));
    vkUnmapMemory(app->getDevice(), instanceBuffer.memory);
}

void DebugCubeRenderer::setCubes(const std::vector<CubeWithColor>& cubes) {
    activeCubes = cubes;
}

void DebugCubeRenderer::render(VkCommandBuffer& cmd, VkDescriptorSet descriptorSet) {
    if (pipeline == VK_NULL_HANDLE || activeCubes.empty() || cubeVBO.vertexBuffer.buffer == VK_NULL_HANDLE || cubeVBO.indexCount == 0) {
        return;
    }
    
    // Update instance buffer before rendering
    updateInstanceBuffer();
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    
    // Bind descriptor sets: set 0 = UBO, set 1 = grid texture + instance buffer
    VkDescriptorSet descriptorSets[] = { descriptorSet, gridDescriptorSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 
        0, 2, descriptorSets, 0, nullptr);
    
    // Bind cube VBO
    const VkBuffer vertexBuffers[] = { cubeVBO.vertexBuffer.buffer };
    const VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, cubeVBO.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    
    // Line width is specified statically in the pipeline (1.0). Do not call vkCmdSetLineWidth
    // unless the pipeline is created with VK_DYNAMIC_STATE_LINE_WIDTH and the device enables wideLines.
    
    // Single instanced draw call for all cubes
    vkCmdDrawIndexed(cmd, cubeVBO.indexCount, static_cast<uint32_t>(activeCubes.size()), 0, 0, 0);
}

void DebugCubeRenderer::cleanup() {
    // Destruction of Vulkan objects is centralized in VulkanResourceManager.
    // Clear local handles to avoid accidental use; actual destroys happen
    // during `resources.cleanup(device)`.
    pipeline = VK_NULL_HANDLE;
    instanceBuffer.buffer = VK_NULL_HANDLE;
    instanceBuffer.memory = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
    gridTextureSampler = VK_NULL_HANDLE;
    gridTextureView = VK_NULL_HANDLE;
    gridTextureImage = VK_NULL_HANDLE;
    gridTextureMemory = VK_NULL_HANDLE;
    gridDescriptorSetLayout = VK_NULL_HANDLE;
    gridDescriptorPool = VK_NULL_HANDLE;
    cubeVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    cubeVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    cubeVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    cubeVBO.indexBuffer.memory = VK_NULL_HANDLE;
    cubeVBO.indexCount = 0;
}
