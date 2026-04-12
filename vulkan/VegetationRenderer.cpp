#include "VegetationRenderer.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstddef>
#include "../math/Common.hpp" // for NodeID
#include "VegetationRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <cstring>
#include <numeric>

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

void VegetationRenderer::onTextureArraysReallocated(VulkanApp* app) {
    fprintf(stderr, "[VEGETATION] onTextureArraysReallocated: invalidating vegDescriptorSet\n");
    if (!app) return;
    if (vegDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet ds = vegDescriptorSet;
        // Remove from resource registry now to avoid later double-free.
        app->resources.removeDescriptorSet(ds);
        // If there are pending command buffers, defer freeing until they're done
        if (app->hasPendingCommandBuffers()) {
            VkDevice device = app->getDevice();
            VkDescriptorPool pool = app->getDescriptorPool();
            app->deferDestroyUntilAllPending([device, pool, ds]() {
                vkFreeDescriptorSets(device, pool, 1, &ds);
            });
        } else {
            vkFreeDescriptorSets(app->getDevice(), app->getDescriptorPool(), 1, &ds);
        }
        vegDescriptorSet = VK_NULL_HANDLE;
        vegDescriptorVersion = 0;
    }
    if (ensureVegDescriptorSet(app)) {
        fprintf(stderr, "[VEGETATION] onTextureArraysReallocated: recreated vegDescriptorSet=%p\n", (void*)vegDescriptorSet);
    } else {
        fprintf(stderr, "[VEGETATION] onTextureArraysReallocated: descriptor still not ready\n");
    }
}

bool VegetationRenderer::ensureVegDescriptorSet(VulkanApp* app) {
    if (!app) return false;
    if (!vegetationTextureArrayManager) return false;
    if (descriptorSetLayout == VK_NULL_HANDLE) {
        fprintf(stderr, "[VEGETATION] ensureVegDescriptorSet: descriptorSetLayout not created yet, deferring allocation\n");
        return false;
    }
    // Need valid view and sampler
    if (vegetationTextureArrayManager->albedoArray.view == VK_NULL_HANDLE || vegetationTextureArrayManager->albedoSampler == VK_NULL_HANDLE) return false;
    uint32_t managerVersion = vegetationTextureArrayManager->getVersion();
    // If descriptor set missing or version changed, (re)create
    if (vegDescriptorSet == VK_NULL_HANDLE || vegDescriptorVersion != managerVersion) {
        // Free previous descriptor set if any
        if (vegDescriptorSet != VK_NULL_HANDLE) {
            VkDescriptorSet ds = vegDescriptorSet;
            app->resources.removeDescriptorSet(ds);
            vkFreeDescriptorSets(app->getDevice(), app->getDescriptorPool(), 1, &ds);
            vegDescriptorSet = VK_NULL_HANDLE;
            vegDescriptorVersion = 0;
        }
        // Allocate and write new descriptor set
        vegDescriptorSet = app->createDescriptorSet(descriptorSetLayout);
        VkDescriptorImageInfo texInfo{};
        texInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        texInfo.imageView = vegetationTextureArrayManager->albedoArray.view;
        texInfo.sampler = vegetationTextureArrayManager->albedoSampler;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = vegDescriptorSet;
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &texInfo;
        vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);
        vegDescriptorVersion = managerVersion;
        app->registerDescriptorSet(vegDescriptorSet);
        fprintf(stderr, "[VEGETATION] Allocated vegDescriptorSet=%p version=%u\n", (void*)vegDescriptorSet, vegDescriptorVersion);
    }
    return vegDescriptorSet != VK_NULL_HANDLE;
}


void VegetationRenderer::init(VulkanApp* app, VkRenderPass renderPassOverride) {
    if (!app || !vegetationTextureArrayManager) return;
    // Store the app pointer for use when generating instance buffers via compute
    this->appPtr = app;
    VkDevice device = app->getDevice();

    // Descriptor set layout for vegetation textures (set=1, binding=0)
    VkDescriptorSetLayoutBinding texArrayBinding{};
    texArrayBinding.binding = 0;
    texArrayBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texArrayBinding.descriptorCount = 1;
    texArrayBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    texArrayBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &texArrayBinding;
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
    pushConstantRange.size = sizeof(float); // billboardScale

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
    bindingDescs[1].stride = sizeof(float) * 4; // vec4 instancePosition packed
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
            {4, 1, VK_FORMAT_R32G32B32_SFLOAT, 0}                             // instancePosition
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
        fprintf(stderr, "[VEGETATION PIPELINE ERROR] Failed to create vegetation pipeline or layout!\n");
    } else {
        fprintf(stderr, "[VEGETATION PIPELINE] Created pipeline=%p layout=%p\n", (void*)vegetationPipeline, (void*)pipelineLayout);
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
        if (!app) fprintf(stderr, "[VEGETATION DRAW ERROR] app is null!\n");
        if (vegetationPipeline == VK_NULL_HANDLE) fprintf(stderr, "[VEGETATION DRAW ERROR] Attempted to bind VK_NULL_HANDLE pipeline!\n");
        return;
    }
    if (!vegetationTextureArrayManager || vegetationTextureArrayManager->albedoArray.view == VK_NULL_HANDLE || vegetationTextureArrayManager->albedoSampler == VK_NULL_HANDLE) {
        fprintf(stderr, "[VEGETATION DRAW ERROR] texture array not ready (view/sampler missing), skipping draw.\n");
        return;
    }

    // Ensure vegetation descriptor set is present and up-to-date
    if (!ensureVegDescriptorSet(app)) {
        fprintf(stderr, "[VEGETATION DRAW ERROR] vegDescriptorSet not ready, skipping draw.\n");
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationPipeline);

    // Bind the persistent, already-updated global descriptor set from VulkanApp as set 0
    VkDescriptorSet globalSet = app->getMainDescriptorSet();
    if (globalSet == VK_NULL_HANDLE) {
        fprintf(stderr, "[VEGETATION DRAW ERROR] globalSet (main descriptor set) is VK_NULL_HANDLE!\n");
        return;
    }
    VkDescriptorSet sets[2] = { globalSet, vegDescriptorSet };
    //printf("[BIND] VegetationRenderer::draw: layout=%p firstSet=0 count=2 sets=%p %p\n", (void*)pipelineLayout, (void*)sets[0], (void*)sets[1]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    // Push billboard scale as push constant
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &billboardScale);

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
                                               VkBuffer vertexBuffer, uint32_t vertexCount,
                                               VkBuffer indexBuffer, uint32_t indexCount,
                                               uint32_t instancesPerTriangle, VulkanApp* app,
                                               uint32_t seed) {
    if (!app) return;
    if (indexCount < 3 || instancesPerTriangle == 0) return;

    uint32_t triCount = indexCount / 3;
    uint32_t instanceCount = triCount * instancesPerTriangle;

    VkDevice device = app->getDevice();
    VkPhysicalDevice physicalDevice = app->getPhysicalDevice();

    // Destroy any existing buffer for this chunk
    destroyInstanceBuffer(chunkId);

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

    // Dispatch compute to fill instanceBuffer
    uint32_t written = app->generateVegetationInstancesCompute(vertexBuffer, vertexCount, indexBuffer, indexCount, instancesPerTriangle, instanceBuffer, static_cast<uint32_t>(instanceBufferSize), seed);
    if (written == 0) {
        // Clean up partially created buffers
        instanceBuffer = VK_NULL_HANDLE;
        instanceMemory = VK_NULL_HANDLE;
        indirectBuffer = VK_NULL_HANDLE;
        indirectMemory = VK_NULL_HANDLE;
        return;
    }
    std::cout << "[VegetationRenderer::generateChunkInstances] written = " << written << std::endl; 
    InstanceBuffer ibuf;
    ibuf.buffer = instanceBuffer;
    ibuf.memory = instanceMemory;
    ibuf.indirectBuffer = indirectBuffer;
    ibuf.indirectMemory = indirectMemory;
    ibuf.count = written;
    chunkBuffers[chunkId] = ibuf;
    chunkInstanceCounts[chunkId] = written;
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
