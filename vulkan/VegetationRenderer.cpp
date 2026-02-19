#include "VegetationRenderer.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include "../math/Common.hpp" // for NodeID
#include "VegetationRenderer.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <cstring>
#include <numeric>

// Set chunk instance buffer directly from GPU buffer (compute shader output)
void VegetationRenderer::setChunkInstanceBuffer(NodeID chunkId, VkBuffer buffer, uint32_t count) {
    destroyInstanceBuffer(chunkId);
    InstanceBuffer ibuf;
    ibuf.buffer = buffer;
    ibuf.memory = VK_NULL_HANDLE; // Ownership of memory is not managed here
    ibuf.indirectBuffer = VK_NULL_HANDLE; // Indirect draw not set up for raw GPU buffer
    ibuf.indirectMemory = VK_NULL_HANDLE;
    ibuf.count = count;
    chunkBuffers[chunkId] = ibuf;
    chunkInstanceCounts[chunkId] = count;
}


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
        app->resources.removeDescriptorSet(ds);
        vkFreeDescriptorSets(app->getDevice(), app->getDescriptorPool(), 1, &ds);
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
    bindingDescs[0].stride = sizeof(float) * 3   // inPosition
                                 + sizeof(float) * 3   // inNormal
                                 + sizeof(float) * 2   // inTexCoord
                                 + sizeof(int);        // inTexIndex
    bindingDescs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescs[1].binding = 1;
    bindingDescs[1].stride = sizeof(float) * 3; // instancePosition
    bindingDescs[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // Attribute descriptions as initializer_list
    auto [pipeline, layout] = app->createGraphicsPipeline(
        { stages[0], stages[1], stages[2] },
        std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1]},
        {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},   // inPosition
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float)*3}, // inNormal
            {2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float)*6},    // inTexCoord
            {3, 0, VK_FORMAT_R32_SINT, sizeof(float)*8},         // inTexIndex
            {4, 1, VK_FORMAT_R32G32B32_SFLOAT, 0}                // instancePosition
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
    // If there are pending CPU-side instance positions, upload them now that `app` is available
    if (!pendingChunkPositions.empty()) {
        for (auto &kv : pendingChunkPositions) {
            createInstanceBuffer(kv.first, kv.second, app);
        }
        pendingChunkPositions.clear();
    }
}

void VegetationRenderer::setChunkInstances(NodeID chunkId, const std::vector<glm::vec3>& positions) {
    // Defer GPU upload until pipelines are created (app available). Store CPU-side positions.
    pendingChunkPositions[chunkId] = positions;
    chunkInstanceCounts[chunkId] = positions.size();
}

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
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    // Push billboard scale as push constant
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &billboardScale);

    // For each chunk, bind instance buffer and draw indirect
    for (const auto& [chunkId, buf] : chunkBuffers) {
        if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &buf.buffer, offsets);
        vkCmdDrawIndirect(commandBuffer, buf.indirectBuffer, 0, 1, sizeof(VkDrawIndirectCommand));
    }
}

void VegetationRenderer::createInstanceBuffer(NodeID chunkId, const std::vector<glm::vec3>& positions, VulkanApp* app) {
    if (!app || positions.empty()) return;
    VkDevice device = app->getDevice();
    VkPhysicalDevice physicalDevice = app->getPhysicalDevice();
    VkQueue graphicsQueue = app->getGraphicsQueue();
    VkCommandPool commandPool = app->getCommandPool();

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer indirectBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indirectMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = positions.size() * sizeof(glm::vec3);
    bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkDeviceSize bufferSize = bufInfo.size;

    // 1. Create staging buffer (host visible)
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkBufferCreateInfo stagingBufInfo = bufInfo;
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(device, &stagingBufInfo, nullptr, &stagingBuffer) != VK_SUCCESS) throw std::runtime_error("Failed to create staging buffer");
    fprintf(stderr, "[VegetationRenderer] createBuffer: stagingBuffer=%p size=%llu usage=0x%x\n", (void*)stagingBuffer, (unsigned long long)stagingBufInfo.size, (unsigned)stagingBufInfo.usage);
    // Register staging buffer with resource manager
    app->resources.addBuffer(stagingBuffer, "VegetationRenderer: stagingBuffer");
    VkMemoryRequirements stagingMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemReq);
    VkMemoryAllocateInfo stagingAllocInfo{};
    stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAllocInfo.allocationSize = stagingMemReq.size;
    // Find host visible memory type
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    uint32_t stagingTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((stagingMemReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            stagingTypeIndex = i;
            break;
        }
    }
    if (stagingTypeIndex == UINT32_MAX) throw std::runtime_error("No suitable host visible memory type for staging buffer");
    stagingAllocInfo.memoryTypeIndex = stagingTypeIndex;
    if (vkAllocateMemory(device, &stagingAllocInfo, nullptr, &stagingMemory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate staging buffer memory");
    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);
    // Register staging memory
    app->resources.addDeviceMemory(stagingMemory, "VegetationRenderer: stagingMemory");
    void* stagingData;
    vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &stagingData);
    std::memcpy(stagingData, positions.data(), bufferSize);
    vkUnmapMemory(device, stagingMemory);

    // 2. Create device-local buffer
    bufInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &buffer) != VK_SUCCESS) throw std::runtime_error("Failed to create buffer");
    fprintf(stderr, "[VegetationRenderer] createBuffer: buffer=%p size=%llu usage=0x%x\n", (void*)buffer, (unsigned long long)bufInfo.size, (unsigned)bufInfo.usage);
    // Register device-local buffer
    app->resources.addBuffer(buffer, "VegetationRenderer: buffer");
        // 3. Create indirect buffer (host visible, small)
        VkBufferCreateInfo indirectBufInfo{};
        indirectBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        indirectBufInfo.size = sizeof(VkDrawIndirectCommand);
        indirectBufInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        indirectBufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &indirectBufInfo, nullptr, &indirectBuffer) != VK_SUCCESS) throw std::runtime_error("Failed to create indirect buffer");
        fprintf(stderr, "[VegetationRenderer] createBuffer: indirectBuffer=%p size=%llu usage=0x%x\n", (void*)indirectBuffer, (unsigned long long)indirectBufInfo.size, (unsigned)indirectBufInfo.usage);
        // Register indirect buffer and its memory
        app->resources.addBuffer(indirectBuffer, "VegetationRenderer: indirectBuffer");
        VkMemoryRequirements indirectMemReq;
        vkGetBufferMemoryRequirements(device, indirectBuffer, &indirectMemReq);
        VkMemoryAllocateInfo indirectAllocInfo{};
        indirectAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        indirectAllocInfo.allocationSize = indirectMemReq.size;
        // Host visible for easy update
        uint32_t indirectTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((indirectMemReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                indirectTypeIndex = i;
                break;
            }
        }
        if (indirectTypeIndex == UINT32_MAX) throw std::runtime_error("No suitable host visible memory type for indirect buffer");
        indirectAllocInfo.memoryTypeIndex = indirectTypeIndex;
        if (vkAllocateMemory(device, &indirectAllocInfo, nullptr, &indirectMemory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate indirect buffer memory");
        vkBindBufferMemory(device, indirectBuffer, indirectMemory, 0);
        // Fill indirect command
        VkDrawIndirectCommand drawCmd{};
        drawCmd.vertexCount = 1;
        drawCmd.instanceCount = static_cast<uint32_t>(positions.size());
        drawCmd.firstVertex = 0;
        drawCmd.firstInstance = 0;
        void* indirectData;
        vkMapMemory(device, indirectMemory, 0, sizeof(VkDrawIndirectCommand), 0, &indirectData);
        std::memcpy(indirectData, &drawCmd, sizeof(VkDrawIndirectCommand));
        vkUnmapMemory(device, indirectMemory);
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    // Find device local memory type
    uint32_t deviceLocalTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            deviceLocalTypeIndex = i;
            break;
        }
    }
    if (deviceLocalTypeIndex == UINT32_MAX) throw std::runtime_error("No suitable device local memory type for instance buffer");
    allocInfo.memoryTypeIndex = deviceLocalTypeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate buffer memory");
    vkBindBufferMemory(device, buffer, memory, 0);
    // Register device-local memory
    app->resources.addDeviceMemory(memory, "VegetationRenderer: memory");

    // 3. Copy from staging to device-local buffer
    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuf;
    if (vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &cmdBuf) != VK_SUCCESS) throw std::runtime_error("Failed to allocate command buffer");
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(cmdBuf, stagingBuffer, buffer, 1, &copyRegion);
    vkEndCommandBuffer(cmdBuf);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);

    // Defer actual destruction to VulkanResourceManager; clear local handles
    stagingBuffer = VK_NULL_HANDLE;
    stagingMemory = VK_NULL_HANDLE;

    InstanceBuffer ibuf;
    ibuf.buffer = buffer;
    ibuf.memory = memory;
    ibuf.indirectBuffer = indirectBuffer;
    ibuf.indirectMemory = indirectMemory;
    ibuf.count = positions.size();
    chunkBuffers[chunkId] = ibuf;
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
