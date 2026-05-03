#include "VegetationRenderer.hpp"

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstddef>
#include "../../math/Common.hpp" // for NodeID
#include "VegetationRenderer.hpp"
#include "../../utils/FileReader.hpp"
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cmath>

// GPU injection APIs removed. Instances must be generated via compute shader.


VegetationRenderer::VegetationRenderer() {}
VegetationRenderer::~VegetationRenderer() { /* caller must call cleanup(app) */ }

void VegetationRenderer::init() {

}


void VegetationRenderer::cleanup() {
    // Collect IDs first to avoid erasing while iterating
    std::vector<NodeID> idsToDestroy;
    idsToDestroy.reserve(chunkBuffers.size());
    for (const auto& [id, _] : chunkBuffers) idsToDestroy.push_back(id);
    for (NodeID id : idsToDestroy) destroyInstanceBuffer(id);
    chunkBuffers.clear();
    chunkInstanceCounts.clear();
    // Clear local handles; central manager handles destruction of Vulkan objects
    vegetationPipeline = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    descriptorSetLayout = VK_NULL_HANDLE;
    // Free and reset vegetation descriptor set handle locally
    vegDescriptorSet = VK_NULL_HANDLE;
    vegDescriptorVersion = 0;
    // Unregister allocation listener if set
    if (vegetationTextureArrayManager && vegTextureListenerId != -1) {
        vegetationTextureArrayManager->removeAllocationListener(vegTextureListenerId);
        vegTextureListenerId = -1;
    }
    // Clear stored app pointer and billboard VBO handles
    billboardAlbedoView   = VK_NULL_HANDLE;
    billboardNormalView   = VK_NULL_HANDLE;
    billboardOpacityView  = VK_NULL_HANDLE;
    billboardArraySampler = VK_NULL_HANDLE;
    billboardVBO.vertexBuffer.buffer = VK_NULL_HANDLE;
    billboardVBO.vertexBuffer.memory = VK_NULL_HANDLE;
    billboardVBO.indexBuffer.buffer = VK_NULL_HANDLE;
    billboardVBO.indexBuffer.memory = VK_NULL_HANDLE;
    billboardVBO.indexCount = 0;
    appPtr = nullptr;
}

void VegetationRenderer::setTextureArrayManager(TextureArrayManager* mgr, VulkanApp* app) {
    // Unregister old listener
    if (vegetationTextureArrayManager && vegTextureListenerId != -1) {
        vegetationTextureArrayManager->removeAllocationListener(vegTextureListenerId);
        vegTextureListenerId = -1;
    }
    vegetationTextureArrayManager = mgr;
    if (!vegetationTextureArrayManager) return;
    // Try to allocate descriptor set immediately if possible
    ensureVegDescriptorSet(app);
    // Register listener to react to future reallocations
    vegTextureListenerId = vegetationTextureArrayManager->addAllocationListener([this, app]() {
        this->onTextureArraysReallocated(app);
    });
}

void VegetationRenderer::setBillboardArrayTextures(VkImageView albedoView, VkImageView normalView, VkImageView opacityView, VkSampler sampler, VulkanApp* app) {
    billboardAlbedoView   = albedoView;
    billboardNormalView   = normalView;
    billboardOpacityView  = opacityView;
    billboardArraySampler = sampler;

    if (!app || descriptorSetLayout == VK_NULL_HANDLE) return;

    if (vegDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet ds = vegDescriptorSet;
        if (app->resources.removeDescriptorSet(ds)) {
            vkFreeDescriptorSets(app->getDevice(), app->getDescriptorPool(), 1, &ds);
        }
        vegDescriptorSet = VK_NULL_HANDLE;
        vegDescriptorVersion = 0;
    }

    ensureVegDescriptorSet(app);
}

void VegetationRenderer::onTextureArraysReallocated(VulkanApp* app) {
    std::cerr << "[VEGETATION] onTextureArraysReallocated: invalidating vegDescriptorSet" << std::endl;
    if (!app) return;
    if (vegDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet ds = vegDescriptorSet;
        // Only free descriptor set if it was tracked by the resource manager
        if (app->resources.removeDescriptorSet(ds)) {
            if (app->hasPendingCommandBuffers()) {
                VkDevice device = app->getDevice();
                VkDescriptorPool pool = app->getDescriptorPool();
                app->deferDestroyUntilAllPending([device, pool, ds]() {
                    vkFreeDescriptorSets(device, pool, 1, &ds);
                });
            } else {
                vkFreeDescriptorSets(app->getDevice(), app->getDescriptorPool(), 1, &ds);
            }
        }
        vegDescriptorSet = VK_NULL_HANDLE;
        vegDescriptorVersion = 0;
    }
    if (ensureVegDescriptorSet(app)) {
        std::cerr << "[VEGETATION] onTextureArraysReallocated: recreated vegDescriptorSet=" << (void*)vegDescriptorSet << std::endl;
    } else {
        std::cerr << "[VEGETATION] onTextureArraysReallocated: descriptor still not ready" << std::endl;
    }
}

bool VegetationRenderer::ensureVegDescriptorSet(VulkanApp* app) {
    if (!app) return false;
    if (descriptorSetLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION] ensureVegDescriptorSet: descriptorSetLayout not created yet, deferring allocation" << std::endl;
        return false;
    }

    if (billboardAlbedoView  == VK_NULL_HANDLE ||
        billboardNormalView  == VK_NULL_HANDLE ||
        billboardOpacityView == VK_NULL_HANDLE ||
        billboardArraySampler == VK_NULL_HANDLE) return false;

    if (vegDescriptorSet == VK_NULL_HANDLE) {
        vegDescriptorSet = app->createDescriptorSet(descriptorSetLayout);

        VkDescriptorImageInfo infos[3]{};
        infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[0].imageView   = billboardAlbedoView;
        infos[0].sampler     = billboardArraySampler;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[1].imageView   = billboardNormalView;
        infos[1].sampler     = billboardArraySampler;
        infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[2].imageView   = billboardOpacityView;
        infos[2].sampler     = billboardArraySampler;

        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = vegDescriptorSet;
            writes[i].dstBinding      = i;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo      = &infos[i];
        }
        vkUpdateDescriptorSets(app->getDevice(), 3, writes, 0, nullptr);
        vegDescriptorVersion = 1;
        app->registerDescriptorSet(vegDescriptorSet);
        std::cerr << "[VEGETATION] Allocated vegDescriptorSet=" << (void*)vegDescriptorSet << " (3 sampler2DArray)" << std::endl;
    }
    return vegDescriptorSet != VK_NULL_HANDLE;
}


void VegetationRenderer::init(VulkanApp* app, VkRenderPass renderPassOverride) {
    if (!app) return;
    // Store the app pointer for use when generating instance buffers via compute
    this->appPtr = app;
    VkDevice device = app->getDevice();

    // Descriptor set layout: set=1, binding 0=albedo, 1=normal, 2=opacity (all sampler2DArray)
    VkDescriptorSetLayoutBinding texBindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        texBindings[i].binding         = i;
        texBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texBindings[i].descriptorCount = 1;
        texBindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        texBindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings    = texBindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create vegetation descriptor set layout");
    }
    // Register descriptor set layout
    app->resources.addDescriptorSetLayout(descriptorSetLayout, "VegetationRenderer: descriptorSetLayout");

    // Pipeline layout: set 0 = UBO, set 1 = vegetation textures
    std::vector<VkDescriptorSetLayout> setLayouts;
    setLayouts.push_back(app->getDescriptorSetLayout());
    setLayouts.push_back(descriptorSetLayout);
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(WindPushConstants);

    // Load shaders
    auto vertCode = FileReader::readFile("shaders/vegetation.vert.spv");
    auto geomCode = FileReader::readFile("shaders/vegetation.geom.spv");
    auto fragCode = FileReader::readFile("shaders/vegetation.frag.spv");
    VkShaderModule vertShader = app->createShaderModule(vertCode);
    VkShaderModule geomShader = app->createShaderModule(geomCode);
    VkShaderModule fragShader = app->createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo geomStage{};
    geomStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomStage.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    geomStage.module = geomShader;
    geomStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, geomStage, fragStage };


    // Vertex input: match shader attributes
    // binding 0: per-vertex, binding 1: per-instance
    VkVertexInputBindingDescription bindingDescs[2] = {};
    bindingDescs[0].binding = 0;
    bindingDescs[0].stride = sizeof(Vertex);
    bindingDescs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescs[1].binding = 1;
    // Compute shader writes vec4 per-instance for alignment; use 16-byte stride.
    bindingDescs[1].stride = sizeof(float) * 4; // vec4: xyz=position, w=billboardIndex
    bindingDescs[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // Attribute descriptions as initializer_list
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { stages[0], stages[1], stages[2] },
        std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1]},
        {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},   // inPosition
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},     // inNormal
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)},      // inTexCoord
            {3, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex)},           // inTexIndex
            {4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0}                          // instanceData (xyz=pos, w=billboardIndex)
        },
        setLayouts,
        &pushConstantRange,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true, // depthTest
        true, // depthWrite
        VK_COMPARE_OP_LESS,
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        renderPassOverride
    );
    vegetationPipeline = pipeline;
    pipelineLayout = layout;
    if (vegetationPipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION PIPELINE ERROR] Failed to create vegetation pipeline or layout!" << std::endl;
    } else {
        std::cerr << "[VEGETATION PIPELINE] Created pipeline=" << (void*)vegetationPipeline << " layout=" << (void*)pipelineLayout << std::endl;
    }

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertShader = VK_NULL_HANDLE;
    geomShader = VK_NULL_HANDLE;
    fragShader = VK_NULL_HANDLE;
    // shader modules are tracked by the central manager for final cleanup
    // Ensure we have a minimal base vertex buffer to feed the pipeline. The
    // pipeline draws a single base-vertex with multiple instances; the
    // instance buffer supplies world-space positions.
    if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) {
        Vertex baseVertex;
        baseVertex.position = glm::vec3(0.0f, 0.0f, 0.0f);
        baseVertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        baseVertex.texCoord = glm::vec2(0.5f, 0.5f);
        baseVertex.texIndex = 0;
        std::vector<Vertex> baseVerts = { baseVertex };
        billboardVBO.vertexBuffer = app->createVertexBuffer(baseVerts);
        billboardVBO.indexCount = 0;
    }

    // Instances are generated exclusively via compute shader; no CPU uploads
    // are performed here.
}

// CPU injection removed: instances are generated by compute shader only.

void VegetationRenderer::clearAllInstances() {
    for (auto& [id, _] : chunkBuffers) destroyInstanceBuffer(id);
    chunkBuffers.clear();
    chunkInstanceCounts.clear();
}

size_t VegetationRenderer::getInstanceTotal() const {
    size_t total = 0;
    for (const auto& kv : chunkInstanceCounts) {
        total += kv.second;
    }
    return total;
}


void VegetationRenderer::draw(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet, const glm::mat4& viewProj) {
    if (!app || vegetationPipeline == VK_NULL_HANDLE) {
        if (!app) std::cerr << "[VEGETATION DRAW ERROR] app is null!" << std::endl;
        if (vegetationPipeline == VK_NULL_HANDLE) std::cerr << "[VEGETATION DRAW ERROR] Attempted to bind VK_NULL_HANDLE pipeline!" << std::endl;
        return;
    }
    if (billboardAlbedoView  == VK_NULL_HANDLE ||
        billboardNormalView  == VK_NULL_HANDLE ||
        billboardOpacityView == VK_NULL_HANDLE ||
        billboardArraySampler == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION DRAW ERROR] billboard array textures not ready, skipping draw." << std::endl;
        return;
    }

    // Ensure vegetation descriptor set is present and up-to-date
    if (!ensureVegDescriptorSet(app)) {
        std::cerr << "[VEGETATION DRAW ERROR] vegDescriptorSet not ready, skipping draw." << std::endl;
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationPipeline);

    // Bind the persistent, already-updated global descriptor set from VulkanApp as set 0
    VkDescriptorSet globalSet = app->getMainDescriptorSet();
    if (globalSet == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION DRAW ERROR] globalSet (main descriptor set) is VK_NULL_HANDLE!" << std::endl;
        return;
    }
    VkDescriptorSet sets[2] = { globalSet, vegDescriptorSet };
    //printf("[BIND] VegetationRenderer::draw: layout=%p firstSet=0 count=2 sets=%p %p\n", (void*)pipelineLayout, (void*)sets[0], (void*)sets[1]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    WindPushConstants pc{};
    pc.billboardScale = billboardScale;
    pc.windEnabled = windSettings.enabled ? 1.0f : 0.0f;
    pc.windTime = windTimeSeconds;

    glm::vec2 windDir = windSettings.direction;
    const float len2 = windDir.x * windDir.x + windDir.y * windDir.y;
    if (len2 > 1e-8f) {
        const float invLen = 1.0f / std::sqrt(len2);
        windDir *= invLen;
    }

    pc.windDirAndStrength = glm::vec4(windDir.x, 0.0f, windDir.y, std::max(0.0f, windSettings.strength));
    pc.windNoise = glm::vec4(
        std::max(0.00001f, windSettings.baseFrequency),
        std::max(0.0f, windSettings.speed),
        std::max(0.00001f, windSettings.gustFrequency),
        std::max(0.0f, windSettings.gustStrength)
    );
    pc.windShape = glm::vec4(
        std::max(0.0f, windSettings.skewAmount),
        std::clamp(windSettings.trunkStiffness, 0.0f, 1.0f),
        std::max(0.001f, windSettings.noiseScale),
        std::max(0.0f, windSettings.verticalFlutter)
    );
    pc.windTurbulence = glm::vec4(std::max(0.0f, windSettings.turbulence), 0.0f, 0.0f, 0.0f);

    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(WindPushConstants),
        &pc
    );

    // For each chunk, bind base vertex buffer + instance buffer and draw indirect
    for (const auto& [chunkId, buf] : chunkBuffers) {
        if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;
        if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) continue;
        VkBuffer vbs[2] = { billboardVBO.vertexBuffer.buffer, buf.buffer };
        VkDeviceSize offsets[2] = { 0, 0 };
        // Bind base vertex buffer at binding 0 and instance buffer at binding 1
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vbs, offsets);
        vkCmdDrawIndirect(commandBuffer, buf.indirectBuffer, 0, 1, sizeof(VkDrawIndirectCommand));
    }
}

void VegetationRenderer::generateChunkInstances(NodeID chunkId,
                                               Buffer vertexBuffer, uint32_t vertexCount,
                                               Buffer indexBuffer, uint32_t indexCount,
                                               uint32_t instancesPerTriangle, VulkanApp* app,
                                               uint32_t seed) {
    if (!app) return;

    // Always clear any previous chunk instances before deciding whether to regenerate.
    destroyInstanceBuffer(chunkId);
    if (indexCount < 3 || instancesPerTriangle == 0) return;

    uint32_t triCount = indexCount / 3;
    uint32_t instanceCount = triCount * instancesPerTriangle;

    VkDevice device = app->getDevice();
    VkPhysicalDevice physicalDevice = app->getPhysicalDevice();

    // Create device-local storage/vertex buffer for instances (vec4 per-instance)
    VkBuffer instanceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    VkDeviceSize instanceBufferSize = static_cast<VkDeviceSize>(instanceCount) * sizeof(float) * 4; // vec4
    bufInfo.size = instanceBufferSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &instanceBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create instance storage buffer");
    }
    app->resources.addBuffer(instanceBuffer, "VegetationRenderer: instanceBuffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, instanceBuffer, &memReq);
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    uint32_t deviceLocalTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            deviceLocalTypeIndex = i;
            break;
        }
    }
    if (deviceLocalTypeIndex == UINT32_MAX) throw std::runtime_error("No suitable device local memory type for instance buffer");
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = deviceLocalTypeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &instanceMemory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate instance buffer memory");
    vkBindBufferMemory(device, instanceBuffer, instanceMemory, 0);
    app->resources.addDeviceMemory(instanceMemory, "VegetationRenderer: instanceMemory");

    // Create host-visible indirect buffer and fill draw command
    VkBuffer indirectBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indirectMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo indirectBufInfo{};
    indirectBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    indirectBufInfo.size = sizeof(VkDrawIndirectCommand);
    indirectBufInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    indirectBufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &indirectBufInfo, nullptr, &indirectBuffer) != VK_SUCCESS) throw std::runtime_error("Failed to create indirect buffer");
    app->resources.addBuffer(indirectBuffer, "VegetationRenderer: indirectBuffer");

    VkMemoryRequirements indirectMemReq;
    vkGetBufferMemoryRequirements(device, indirectBuffer, &indirectMemReq);
    uint32_t indirectTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((indirectMemReq.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            indirectTypeIndex = i;
            break;
        }
    }
    if (indirectTypeIndex == UINT32_MAX) throw std::runtime_error("No suitable host visible memory type for indirect buffer");
    VkMemoryAllocateInfo indirectAllocInfo{};
    indirectAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    indirectAllocInfo.allocationSize = indirectMemReq.size;
    indirectAllocInfo.memoryTypeIndex = indirectTypeIndex;
    if (vkAllocateMemory(device, &indirectAllocInfo, nullptr, &indirectMemory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate indirect buffer memory");
    vkBindBufferMemory(device, indirectBuffer, indirectMemory, 0);

    VkDrawIndirectCommand drawCmd{};
    drawCmd.vertexCount = 1;
    drawCmd.instanceCount = instanceCount;
    drawCmd.firstVertex = 0;
    drawCmd.firstInstance = 0;
    void* indirectData;
    vkMapMemory(device, indirectMemory, 0, sizeof(VkDrawIndirectCommand), 0, &indirectData);
    std::memcpy(indirectData, &drawCmd, sizeof(VkDrawIndirectCommand));
    vkUnmapMemory(device, indirectMemory);
    app->resources.addDeviceMemory(indirectMemory, "VegetationRenderer: indirectMemory");

    // Dispatch compute to fill instanceBuffer asynchronously on vegetation queue
    VkFence fence = VK_NULL_HANDLE;
    uint32_t expected = app->generateVegetationInstancesComputeAsync(vertexBuffer.buffer, vertexCount, indexBuffer.buffer, indexCount, instancesPerTriangle, instanceBuffer, static_cast<uint32_t>(instanceBufferSize), &fence, seed, billboardCount);
    if (expected == 0 || fence == VK_NULL_HANDLE) {
        // Clean up partially created buffers if compute not dispatched
        if (app->resources.removeBuffer(instanceBuffer)) vkDestroyBuffer(device, instanceBuffer, nullptr);
        if (app->resources.removeDeviceMemory(instanceMemory)) vkFreeMemory(device, instanceMemory, nullptr);
        if (app->resources.removeBuffer(indirectBuffer)) vkDestroyBuffer(device, indirectBuffer, nullptr);
        if (app->resources.removeDeviceMemory(indirectMemory)) vkFreeMemory(device, indirectMemory, nullptr);
        return;
    }

    std::cout << "[VegetationRenderer::generateChunkInstances] async dispatched, expected = " << expected << " fence=" << (void*)fence << std::endl;

    // Prepare instance buffer record to insert on completion
    InstanceBuffer ibuf;
    ibuf.buffer = instanceBuffer;
    ibuf.memory = instanceMemory;
    ibuf.indirectBuffer = indirectBuffer;
    ibuf.indirectMemory = indirectMemory;
    ibuf.count = expected;

    // Defer adding the instance buffer to the visible map until the GPU finished
    app->deferDestroyUntilFence(fence, [this, chunkId, ibuf, expected]() mutable {
        // insert into local maps so draw will pick it up
        this->chunkBuffers[chunkId] = ibuf;
        this->chunkInstanceCounts[chunkId] = expected;
        std::cout << "[VegetationRenderer] chunk " << (unsigned long long)chunkId << " instances ready: " << expected << std::endl;
    });

    // Defer destruction of the temporary input buffers (vertex/index) until fence signals
    VkDevice dev = device;
    Buffer vbuf = vertexBuffer;
    Buffer ib = indexBuffer;
    app->deferDestroyUntilFence(fence, [dev, vbuf, ib, app]() {
        if (vbuf.buffer != VK_NULL_HANDLE) {
            if (app->resources.removeBuffer(vbuf.buffer)) vkDestroyBuffer(dev, vbuf.buffer, nullptr);
        }
        if (vbuf.memory != VK_NULL_HANDLE) {
            if (app->resources.removeDeviceMemory(vbuf.memory)) vkFreeMemory(dev, vbuf.memory, nullptr);
        }
        if (ib.buffer != VK_NULL_HANDLE) {
            if (app->resources.removeBuffer(ib.buffer)) vkDestroyBuffer(dev, ib.buffer, nullptr);
        }
        if (ib.memory != VK_NULL_HANDLE) {
            if (app->resources.removeDeviceMemory(ib.memory)) vkFreeMemory(dev, ib.memory, nullptr);
        }
    });
}

void VegetationRenderer::destroyInstanceBuffer(NodeID chunkId) {
    auto it = chunkBuffers.find(chunkId);
    if (it != chunkBuffers.end()) {
        // Defer actual destruction to VulkanResourceManager; clear local handles
        it->second.buffer = VK_NULL_HANDLE;
        it->second.memory = VK_NULL_HANDLE;
        it->second.indirectBuffer = VK_NULL_HANDLE;
        it->second.indirectMemory = VK_NULL_HANDLE;
        chunkBuffers.erase(it);
    }
}

// Ensure we clear the stored app pointer on cleanup
// (cleanup() already clears handles; set appPtr to nullptr here)
