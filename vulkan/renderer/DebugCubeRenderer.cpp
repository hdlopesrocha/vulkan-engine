#include "DebugCubeRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "../VertexBufferObjectBuilder.hpp"
#include "../ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include "../../math/BoxLineGeometry.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <stb/stb_image.h>
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

DebugCubeRenderer::DebugCubeRenderer() {}

DebugCubeRenderer::~DebugCubeRenderer() { cleanup(); }

void DebugCubeRenderer::init(VulkanApp* app) {
    // Create cube VBO
    createCubeVBO(app);
    
    // Load grid texture
    loadGridTexture(app);
    
    // Create descriptor set for grid texture
    createGridDescriptorSet(app);
    
    // Create shader modules
    vertModule = app->getOrCreateShaderModule("shaders/debug_cube.vert.spv");
    fragModule = app->getOrCreateShaderModule("shaders/debug_cube.frag.spv");
    
    ShaderStage vertStage(vertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragStage(fragModule, VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Setup descriptor set layouts: set 0 = UBO, set 1 = grid texture + instance buffer
    std::vector<VkDescriptorSetLayout> setLayouts = {
        app->getDescriptorSetLayout(),  // set 0: UBO
        gridDescriptorSetLayout          // set 1: grid texture + instance buffer
    };
    
    // Create pipeline with alpha blending enabled for transparency
    GraphicsPipelineConfig cfg{};
    cfg.polygonMode = VK_POLYGON_MODE_LINE;
    cfg.cullMode = VK_CULL_MODE_NONE;
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    VkPipeline pipelineHandle;
    VkPipelineLayout layoutHandle;
    std::tie(pipelineHandle, layoutHandle) = app->createGraphicsPipeline(
        { vertStage.info, fragStage.info },
        std::vector<VkVertexInputBindingDescription>{ 
            VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } 
        },
        std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{ ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription{ ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
        },
        setLayouts,
        nullptr,
        cfg
    );
    
    pipeline = pipelineHandle;
    pipelineLayout = layoutHandle;
    
    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[DEBUG CUBE RENDERER ERROR] Failed to create pipeline!" << std::endl;
    }
}

void DebugCubeRenderer::createCubeVBO(VulkanApp* app) {
    // Create a unit cube with min=(0,0,0) and max=(1,1,1)
    BoundingBox unitBox(glm::vec3(0.0f), glm::vec3(1.0f));
    BoxLineGeometry cubeGeom(unitBox);
    
    cubeVBO = VertexBufferObjectBuilder::create(app, cubeGeom);
    std::cout << "[DebugCubeRenderer] Created cube VBO: vertices=" << cubeGeom.vertices.size() << " indices=" << cubeGeom.indices.size() << std::endl;
}

void DebugCubeRenderer::loadGridTexture(VulkanApp* app) {
    // Load grid.png texture
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load("textures/grid.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    
    if (!pixels) {
        std::cerr << "[DEBUG CUBE RENDERER ERROR] Failed to load grid texture!" << std::endl;
        return;
    }
    
    printf("[DEBUG CUBE RENDERER] Loaded grid texture: %dx%d, channels=%d\n", texWidth, texHeight, texChannels);
    
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    
    // Create staging buffer
    Buffer stagingBuffer = app->createBuffer(imageSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    void* data;
    data = stagingBuffer.map(0);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    stagingBuffer.unmap(); // VMA persistent mapping
    
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
    
    app->createImageWithVma(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gridTextureImage, gridTextureAllocation, gridTextureMemory, "DebugCubeRenderer: gridTextureImage");
    
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
        std::cerr << "[DEBUG CUBE RENDERER ERROR] Failed to create grid texture view!" << std::endl;
        return;
    }
    std::cout << "[DebugCubeRenderer] createImageView: gridTextureView=" << (void*)gridTextureView << " for image=" << (void*)gridTextureImage << std::endl;
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
    gridTextureSampler = app->createSampler(samplerInfo, "DebugCubeRenderer: gridTextureSampler");
    
    std::cout << "[DebugCubeRenderer] Loaded grid texture: " << texWidth << "x" << texHeight << std::endl;
}

void DebugCubeRenderer::createGridDescriptorSet(VulkanApp* app) {
    DescriptorAllocator descAlloc{app->getDevice(), app};

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

    gridDescriptorSetLayout = descAlloc.createLayout(
        bindings, 2,
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        nullptr,
        "DebugCubeRenderer: gridDescriptorSetLayout");

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
    };
    gridDescriptorPool = descAlloc.createPool(
        poolSizes, 2, 1,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "DebugCubeRenderer: gridDescriptorPool");

    gridDescriptorSet = descAlloc.allocateSet(gridDescriptorPool, gridDescriptorSetLayout, "DebugCubeRenderer: gridDescriptorSet");
    
    DescriptorWriter(app->getDevice())
        .writeImage(gridDescriptorSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    gridTextureSampler, gridTextureView,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .flush();
    
    // Create initial instance buffer (will be resized as needed)
    instanceBufferCapacity = 128;
    instanceBuffer = app->createBuffer(
        instanceBufferCapacity * (sizeof(glm::mat4) + sizeof(glm::vec4)),  // mat4 + vec4 for alignment
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    DescriptorWriter(app->getDevice())
        .writeBuffer(gridDescriptorSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     instanceBuffer.buffer, 0, VK_WHOLE_SIZE)
        .flush();
}

void DebugCubeRenderer::updateInstanceBuffer(VulkanApp* app) {
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
        
        DescriptorWriter(app->getDevice())
            .writeBuffer(gridDescriptorSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         instanceBuffer.buffer, 0, VK_WHOLE_SIZE)
            .flush();
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
    
    memcpy(instanceBuffer.mappedData, instanceData.data(), instanceData.size() * sizeof(InstanceData));
}

void DebugCubeRenderer::setCubes(const std::vector<CubeWithColor>& cubes) {
    activeCubes = cubes;
}

void DebugCubeRenderer::render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet) {
    if (pipeline == VK_NULL_HANDLE || activeCubes.empty() || cubeVBO.vertexBuffer.buffer == VK_NULL_HANDLE || cubeVBO.indexCount == 0) {
        return;
    }
    
    // Update instance buffer before rendering
    updateInstanceBuffer(app);
    
    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    
    // Bind descriptor sets: set 0 = UBO, set 1 = grid texture + instance buffer
    VkDescriptorSet descriptorSets[] = { descriptorSet, gridDescriptorSet };
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 2, descriptorSets, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 
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
    cubeVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    cubeVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    cubeVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    cubeVBO.indexBuffer.memory = VK_NULL_HANDLE;
    cubeVBO.indexCount = 0;
    instanceBuffer = {};
}
