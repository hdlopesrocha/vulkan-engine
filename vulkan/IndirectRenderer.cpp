#include "IndirectRenderer.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"

IndirectRenderer::IndirectRenderer() {}
IndirectRenderer::~IndirectRenderer() {}

void IndirectRenderer::init() {
}

void IndirectRenderer::cleanup() {
    // meshes no longer own per-mesh buffers; clear CPU lists
    meshes.clear();
    // Clear local handles; destruction is centralized in VulkanResourceManager.
    vertexBuffer = {};
    indexBuffer = {};
    indirectBuffer = {};
    compactIndirectBuffer = {};
    modelsBuffer = {};
    boundsBuffer = {};
    visibleCountBuffer = {};

    computePipeline = VK_NULL_HANDLE;
    computePipelineLayout = VK_NULL_HANDLE;
    computeDescriptorSetLayout = VK_NULL_HANDLE;
    computeDescriptorPool = VK_NULL_HANDLE;
}

uint32_t IndirectRenderer::addMesh(const Geometry& mesh) {
    return updateMesh(mesh, nextId++);
}

uint32_t IndirectRenderer::updateMesh(const Geometry& mesh, uint32_t customId) {
    std::lock_guard<std::mutex> guard(mutex);
    std::cout << "[IndirectRenderer::addMesh] Adding/replacing mesh ID " << customId << " with " << mesh.vertices.size() << " vertices and " << mesh.indices.size() << " indices.\n";

    MeshInfo m{};
    m.id = customId;
    m.baseVertex = static_cast<uint32_t>(mergedVertices.size());
    m.firstIndex = static_cast<uint32_t>(mergedIndices.size());
    m.indexCount = static_cast<uint32_t>(mesh.indices.size());
    m.model = glm::mat4(1.0f);
    m.active = true;

    if (mesh.vertices.empty()) {
        // Empty mesh: set degenerate zero-sized bounds at origin
        m.boundsMin = glm::vec4(0.0f);
        m.boundsMax = glm::vec4(0.0f);
    } else {
        glm::vec3 minp(FLT_MAX), maxp(-FLT_MAX);
        for (const auto& v : mesh.vertices) {
            minp = glm::min(minp, v.position);
            maxp = glm::max(maxp, v.position);
        }
        m.boundsMin = glm::vec4(minp, 0.0f);
        m.boundsMax = glm::vec4(maxp, 0.0f);
    }

    mergedVertices.insert(mergedVertices.end(), mesh.vertices.begin(), mesh.vertices.end());
    mergedIndices.insert(mergedIndices.end(), mesh.indices.begin(), mesh.indices.end());

    VkDrawIndexedIndirectCommand cmd{};
    cmd.indexCount = m.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex = m.firstIndex;
    cmd.vertexOffset = static_cast<int32_t>(m.baseVertex);
    cmd.firstInstance = static_cast<uint32_t>(indirectCommands.size());
    indirectCommands.push_back(cmd);

    meshes[m.id] = m; // insert or replace

    return customId;
}


void IndirectRenderer::removeMesh(uint32_t meshId) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return;
    it->second.active = false;
    dirty = true;
}

bool IndirectRenderer::ensureCapacity(size_t vertexCount, size_t indexCount, size_t meshCount) {
    std::lock_guard<std::mutex> guard(mutex);
    
    // Add 25% headroom for future growth
    size_t neededVertexCap = vertexCount + vertexCount / 4;
    size_t neededIndexCap = indexCount + indexCount / 4;
    size_t neededMeshCap = meshCount + meshCount / 4;
    
    bool needsRebuild = false;
    
    if (vertexBuffer.buffer == VK_NULL_HANDLE || vertexCapacity < neededVertexCap) {
        needsRebuild = true;
    }
    if (indexBuffer.buffer == VK_NULL_HANDLE || indexCapacity < neededIndexCap) {
        needsRebuild = true;
    }
    if (indirectBuffer.buffer == VK_NULL_HANDLE || meshCapacity < neededMeshCap) {
        needsRebuild = true;
    }
    
    if (needsRebuild) {
        // Set target capacities - rebuild will use these
        if (neededVertexCap > vertexCapacity) vertexCapacity = neededVertexCap;
        if (neededIndexCap > indexCapacity) indexCapacity = neededIndexCap;
        if (neededMeshCap > meshCapacity) meshCapacity = neededMeshCap;
        dirty = true;
    }
    
    return !needsRebuild;
}

bool IndirectRenderer::uploadMeshVerticesAndIndices(VulkanApp* app, uint32_t meshId) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) {
        printf("[IndirectRenderer::uploadMeshVerticesAndIndices] meshId %u not found\n", meshId);
        return false;
    }
    MeshInfo& info = it->second;
    if (!info.active) {
        printf("[IndirectRenderer::uploadMeshVerticesAndIndices] meshId %u is inactive\n", meshId);
        return false;
    }
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
        printf("[IndirectRenderer::uploadMeshVerticesAndIndices] buffers not created, need rebuild()\n");
        return false;
    }
    uint32_t maxVertexIdx = 0;
    for (size_t i = info.firstIndex; i < info.firstIndex + info.indexCount && i < mergedIndices.size(); ++i) {
        if (mergedIndices[i] > maxVertexIdx) maxVertexIdx = mergedIndices[i];
    }
    size_t meshVertexCount = maxVertexIdx + 1;
    if (info.baseVertex + meshVertexCount > vertexCapacity) {
        printf("[IndirectRenderer::uploadMeshVerticesAndIndices] vertex capacity exceeded (%u + %zu > %zu)\n", info.baseVertex, meshVertexCount, vertexCapacity);
        return false;
    }
    if (info.firstIndex + info.indexCount > indexCapacity) {
        printf("[IndirectRenderer::uploadMeshVerticesAndIndices] index capacity exceeded\n");
        return false;
    }
    VkDeviceSize vertexOffset = info.baseVertex * sizeof(Vertex);
    VkDeviceSize vertexSize = meshVertexCount * sizeof(Vertex);
    if (vertexSize > 0 && info.baseVertex < mergedVertices.size()) {
        Buffer stagingVertex = app->createBuffer(vertexSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data;
        vkMapMemory(app->getDevice(), stagingVertex.memory, 0, vertexSize, 0, &data);
        memcpy(data, &mergedVertices[info.baseVertex], vertexSize);
        vkUnmapMemory(app->getDevice(), stagingVertex.memory);
        VkCommandBuffer cmd = app->beginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = vertexOffset;
        copyRegion.size = vertexSize;
        vkCmdCopyBuffer(cmd, stagingVertex.buffer, vertexBuffer.buffer, 1, &copyRegion);
        app->endSingleTimeCommands(cmd);
        // Rely on VulkanResourceManager to destroy tracked buffers/memory later
        stagingVertex.buffer = VK_NULL_HANDLE;
        stagingVertex.memory = VK_NULL_HANDLE;
    }
    VkDeviceSize indexOffset = info.firstIndex * sizeof(uint32_t);
    VkDeviceSize indexSize = info.indexCount * sizeof(uint32_t);
    if (indexSize > 0 && info.firstIndex < mergedIndices.size()) {
        Buffer stagingIndex = app->createBuffer(indexSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data;
        vkMapMemory(app->getDevice(), stagingIndex.memory, 0, indexSize, 0, &data);
        memcpy(data, &mergedIndices[info.firstIndex], indexSize);
        vkUnmapMemory(app->getDevice(), stagingIndex.memory);
        VkCommandBuffer cmd = app->beginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = indexOffset;
        copyRegion.size = indexSize;
        vkCmdCopyBuffer(cmd, stagingIndex.buffer, indexBuffer.buffer, 1, &copyRegion);
        app->endSingleTimeCommands(cmd);
        // Rely on VulkanResourceManager to destroy tracked buffers/memory later
        stagingIndex.buffer = VK_NULL_HANDLE;
        stagingIndex.memory = VK_NULL_HANDLE;
    }
    return true;
}

bool IndirectRenderer::uploadMesh(VulkanApp* app, uint32_t meshId) {
    if (!uploadMeshVerticesAndIndices(app, meshId)) {
        return false;
    }
    uploadMeshMetaBuffers(app);
    return true;
}

// Write all mesh indirect/model/bounds buffers for all active meshes
void IndirectRenderer::uploadMeshMetaBuffers(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;
    size_t activeIdx = 0;
    for (auto& kv : meshes) {
        MeshInfo& info = kv.second;
        if (!info.active) continue;
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
        cmd.firstInstance = static_cast<uint32_t>(activeIdx);
        VkDeviceSize cmdOffset = activeIdx * sizeof(VkDrawIndexedIndirectCommand);
        VkDeviceSize cmdSize = sizeof(VkDrawIndexedIndirectCommand);
        void* data;
        vkMapMemory(app->getDevice(), indirectBuffer.memory, cmdOffset, cmdSize, 0, &data);
        memcpy(data, &cmd, cmdSize);
        vkUnmapMemory(app->getDevice(), indirectBuffer.memory);
        info.indirectOffset = cmdOffset;
        if (modelsBuffer.buffer != VK_NULL_HANDLE) {
            VkDeviceSize modelOffset = activeIdx * sizeof(glm::mat4);
            glm::mat4 model = info.model;
            vkMapMemory(app->getDevice(), modelsBuffer.memory, modelOffset, sizeof(glm::mat4), 0, &data);
            memcpy(data, &model, sizeof(glm::mat4));
            vkUnmapMemory(app->getDevice(), modelsBuffer.memory);
        }
        if (boundsBuffer.buffer != VK_NULL_HANDLE) {
            VkDeviceSize boundsOffset = activeIdx * 2 * sizeof(glm::vec4);
            glm::vec4 bounds[2] = { info.boundsMin, info.boundsMax };
            vkMapMemory(app->getDevice(), boundsBuffer.memory, boundsOffset, sizeof(bounds), 0, &data);
            memcpy(data, bounds, sizeof(bounds));
            vkUnmapMemory(app->getDevice(), boundsBuffer.memory);
        }
        ++activeIdx;
    }
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);
    
    size_t activeMeshCount = 0;
    for (const auto& kv : meshes) if (kv.second.active) ++activeMeshCount;
    printf("[IndirectRenderer::rebuild] Called. dirty=%d, meshes.size()=%zu, activeMeshCount=%zu, mergedVertices=%zu, mergedIndices=%zu\n",
        dirty, meshes.size(), activeMeshCount, mergedVertices.size(), mergedIndices.size());
    
    if (!dirty) return;
    printf("[IndirectRenderer::rebuild] dirty=true, rebuilding buffers...\n");

    // Wait for GPU to finish using current buffers before destroying/recreating them
    vkDeviceWaitIdle(app->getDevice());

    // Calculate required capacity with 25% headroom for incremental adds
    size_t neededVertexCap = mergedVertices.size() + mergedVertices.size() / 4 + 1024;
    size_t neededIndexCap = mergedIndices.size() + mergedIndices.size() / 4 + 4096;
    size_t neededMeshCap = activeMeshCount + activeMeshCount / 4 + 64;
    
    // Use max of current capacity or needed capacity (never shrink)
    if (neededVertexCap > vertexCapacity) vertexCapacity = neededVertexCap;
    if (neededIndexCap > indexCapacity) indexCapacity = neededIndexCap;
    if (neededMeshCap > meshCapacity) meshCapacity = neededMeshCap;

    // Build merged GPU-side vertex and index buffers from the CPU arrays.
    // If there are no meshes, free existing buffers.
    static bool printedBufferInfo = false;
    if (!printedBufferInfo) {
        printf("[IndirectRenderer::rebuild] mergedVertices.size()=%zu mergedIndices.size()=%zu\n", 
            mergedVertices.size(), mergedIndices.size());
        if (!mergedVertices.empty()) {
            printf("[IndirectRenderer::rebuild] Sample vertex[0]: pos=(%.2f,%.2f,%.2f)\n",
                mergedVertices[0].position.x, mergedVertices[0].position.y, mergedVertices[0].position.z);
        }
        printedBufferInfo = true;
    }
    if (mergedVertices.empty() || mergedIndices.empty()) {
        if (vertexBuffer.buffer != VK_NULL_HANDLE) {
            // Defer actual destruction to VulkanResourceManager; clear local handle
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE) {
            // Defer actual destruction to VulkanResourceManager; clear local handle
            indexBuffer = {};
        }
        vertexCapacity = 0;
        indexCapacity = 0;
    } else {
        // Recreate vertex/index buffers with capacity-based sizing
        if (vertexBuffer.buffer != VK_NULL_HANDLE) {
            // Defer actual destruction to VulkanResourceManager; clear local handle
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE) {
            // Defer actual destruction to VulkanResourceManager; clear local handle
            indexBuffer = {};
        }
        
        // Create vertex buffer with capacity (not just current size)
        VkDeviceSize vertexBufferSize = vertexCapacity * sizeof(Vertex);
        vertexBuffer = app->createBuffer(vertexBufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        // Create index buffer with capacity
        VkDeviceSize indexBufferSize = indexCapacity * sizeof(uint32_t);
        indexBuffer = app->createBuffer(indexBufferSize,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        // Upload current data via staging
        if (!mergedVertices.empty()) {
            VkDeviceSize dataSize = mergedVertices.size() * sizeof(Vertex);
            Buffer staging = app->createBuffer(dataSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* data;
            vkMapMemory(app->getDevice(), staging.memory, 0, dataSize, 0, &data);
            memcpy(data, mergedVertices.data(), dataSize);
            vkUnmapMemory(app->getDevice(), staging.memory);
            
            VkCommandBuffer cmd = app->beginSingleTimeCommands();
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = dataSize;
            vkCmdCopyBuffer(cmd, staging.buffer, vertexBuffer.buffer, 1, &copyRegion);
            app->endSingleTimeCommands(cmd);
            
            // Defer actual destruction to VulkanResourceManager; clear local handle
            staging.buffer = VK_NULL_HANDLE;
            staging.memory = VK_NULL_HANDLE;
        }
        
        if (!mergedIndices.empty()) {
            VkDeviceSize dataSize = mergedIndices.size() * sizeof(uint32_t);
            Buffer staging = app->createBuffer(dataSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* data;
            vkMapMemory(app->getDevice(), staging.memory, 0, dataSize, 0, &data);
            memcpy(data, mergedIndices.data(), dataSize);
            vkUnmapMemory(app->getDevice(), staging.memory);
            
            VkCommandBuffer cmd = app->beginSingleTimeCommands();
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = dataSize;
            vkCmdCopyBuffer(cmd, staging.buffer, indexBuffer.buffer, 1, &copyRegion);
            app->endSingleTimeCommands(cmd);
            
            // Defer actual destruction to VulkanResourceManager; clear local handle
            staging.buffer = VK_NULL_HANDLE;
            staging.memory = VK_NULL_HANDLE;
        }
    }

    // Rebuild indirect command list from active meshes so GPU-side compaction matches models/bounds
    std::vector<VkDrawIndexedIndirectCommand> cmds;
    cmds.reserve(meshes.size());
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
        cmd.firstInstance = static_cast<uint32_t>(cmds.size());
        cmds.push_back(cmd);
    }
    indirectCommands = cmds;

    // Debug: print first few commands to verify data
    static bool printedCmds = false;
    if (!printedCmds && !cmds.empty()) {
        printf("[IndirectRenderer::rebuild] Sample indirect commands:\n");
        for (size_t i = 0; i < std::min(size_t(3), cmds.size()); ++i) {
            printf("  cmd[%zu]: indexCount=%u instanceCount=%u firstIndex=%u vertexOffset=%d firstInstance=%u\n",
                i, cmds[i].indexCount, cmds[i].instanceCount, cmds[i].firstIndex, cmds[i].vertexOffset, cmds[i].firstInstance);
        }
        printedCmds = true;
    }

    // Create or update the global indirect buffer with capacity-based sizing
    // Use host-visible memory for AMD RADV driver compatibility
    VkDeviceSize indirectBufferSize = sizeof(VkDrawIndexedIndirectCommand) * meshCapacity;
    VkDeviceSize indirectDataSize = sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size();
    if (indirectBuffer.buffer != VK_NULL_HANDLE) {
        // Defer actual destruction to VulkanResourceManager; clear local handle
        indirectBuffer = {};
    }
    if (meshCapacity > 0) {
        indirectBuffer = app->createBuffer(indirectBufferSize, 
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (indirectDataSize > 0) {
            void* data;
            vkMapMemory(app->getDevice(), indirectBuffer.memory, 0, indirectDataSize, 0, &data);
            memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
            vkUnmapMemory(app->getDevice(), indirectBuffer.memory);
        }
    }

    // mark per-mesh indirect offsets (byte offsets inside indirect buffer)
    VkDeviceSize offsetCursor = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        if (!meshes[i].active) continue;
        meshes[i].indirectOffset = offsetCursor;
        offsetCursor += sizeof(VkDrawIndexedIndirectCommand);
    }

    // Build and upload a GPU-side SSBO containing model matrices in the same
    // order as the indirect commands so shaders can index by draw ID.
    std::vector<glm::mat4> models;
    models.reserve(meshes.size());
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        models.push_back(info.model);
    }
    
    static bool printedModels = false;
    if (!printedModels && !models.empty()) {
        printf("[IndirectRenderer::rebuild] Sample model matrix [0]:\n");
        printf("  [%.2f %.2f %.2f %.2f]\n", models[0][0][0], models[0][1][0], models[0][2][0], models[0][3][0]);
        printf("  [%.2f %.2f %.2f %.2f]\n", models[0][0][1], models[0][1][1], models[0][2][1], models[0][3][1]);
        printf("  [%.2f %.2f %.2f %.2f]\n", models[0][0][2], models[0][1][2], models[0][2][2], models[0][3][2]);
        printf("  [%.2f %.2f %.2f %.2f]\n", models[0][0][3], models[0][1][3], models[0][2][3], models[0][3][3]);
        printedModels = true;
    }

    VkDeviceSize modelsBufferSize = sizeof(glm::mat4) * meshCapacity;
    VkDeviceSize modelsDataSize = sizeof(glm::mat4) * models.size();
    if (modelsBuffer.buffer != VK_NULL_HANDLE) {
        // Defer actual destruction to VulkanResourceManager; clear local handle
        modelsBuffer = {};
    }
    if (meshCapacity > 0) {
        // Use host-visible memory for models - updated when meshes change
        modelsBuffer = app->createBuffer(modelsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (modelsDataSize > 0) {
            void* mdata;
            vkMapMemory(app->getDevice(), modelsBuffer.memory, 0, modelsDataSize, 0, &mdata);
            memcpy(mdata, models.data(), (size_t)modelsDataSize);
            vkUnmapMemory(app->getDevice(), modelsBuffer.memory);
        }
    }

    // Upload bounds SSBO (two vec4s per active mesh: min, max)
    std::vector<glm::vec4> boundsData;
    boundsData.reserve(meshes.size() * 2);
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        boundsData.push_back(info.boundsMin);
        boundsData.push_back(info.boundsMax);
    }
    VkDeviceSize boundsBufferSize = sizeof(glm::vec4) * meshCapacity * 2;
    VkDeviceSize boundsDataSize = sizeof(glm::vec4) * boundsData.size();
    if (boundsBuffer.buffer != VK_NULL_HANDLE) {
        // Defer actual destruction to VulkanResourceManager; clear local handle
        boundsBuffer = {};
    }
    if (meshCapacity > 0) {
        // Use host-visible memory for bounds - updated when meshes change
        boundsBuffer = app->createBuffer(boundsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (boundsDataSize > 0) {
            void* bdata;
            vkMapMemory(app->getDevice(), boundsBuffer.memory, 0, boundsDataSize, 0, &bdata);
            memcpy(bdata, boundsData.data(), (size_t)boundsDataSize);
            vkUnmapMemory(app->getDevice(), boundsBuffer.memory);
        }
    }

    // Create/resize compact indirect buffer (storage + indirect usage)
    // This buffer is written by compute shader and read by indirect draw
    VkDeviceSize compactSize = indirectBufferSize; // same capacity as full indirect buffer
    printf("[IndirectRenderer::rebuild] meshes=%zu activeCmds=%zu capacity=%zu\n", meshes.size(), cmds.size(), meshCapacity);
    if (compactIndirectBuffer.buffer != VK_NULL_HANDLE) {
        // Defer actual destruction to VulkanResourceManager; clear local handle
        compactIndirectBuffer = {};
    }
    if (compactSize > 0) {
        // Compact buffer is written by compute shader every frame - use host-coherent for compatibility
        // (device-local can cause issues on some AMD drivers when written by compute)
        compactIndirectBuffer = app->createBuffer(compactSize, 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        // Initialize compactIndirectBuffer with the same data as indirectBuffer so that
        // drawIndirectOnly() works even without running the compute cull pass
        if (indirectDataSize > 0) {
            void* data;
            vkMapMemory(app->getDevice(), compactIndirectBuffer.memory, 0, indirectDataSize, 0, &data);
            memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
            vkUnmapMemory(app->getDevice(), compactIndirectBuffer.memory);
        }
    }

    // Create or zero the visible count buffer (single uint) - host-visible for compute shader writes
    VkDeviceSize countSize = sizeof(uint32_t);
    if (visibleCountBuffer.buffer != VK_NULL_HANDLE) {
        // Defer actual destruction to VulkanResourceManager; clear local handle
        visibleCountBuffer = {};
    }
    visibleCountBuffer = app->createBuffer(countSize, 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // Initialize visibleCount with the number of active commands so drawIndirectOnly() works without culling
    {
        uint32_t initialCount = static_cast<uint32_t>(indirectCommands.size());
        void* data;
        vkMapMemory(app->getDevice(), visibleCountBuffer.memory, 0, sizeof(uint32_t), 0, &data);
        memcpy(data, &initialCount, sizeof(uint32_t));
        vkUnmapMemory(app->getDevice(), visibleCountBuffer.memory);
    }

    // Create compute pipeline + descriptor set for GPU culling if not present
    if (computePipeline == VK_NULL_HANDLE) {
        // Descriptor layout bindings: 0=inCmds,1=outCmds,2=models,3=bounds,4=visibleCount
        VkDescriptorSetLayoutBinding bindings[5] = {};
        for (uint32_t i = 0; i < 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 5;
        layoutInfo.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
        // central manager
        app->resources.addDescriptorSetLayout(computeDescriptorSetLayout, "IndirectRenderer: computeDescriptorSetLayout");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(glm::mat4) + sizeof(uint32_t);  // mat4 viewProj + uint targetLayer

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        plinfo.pSetLayouts = &computeDescriptorSetLayout;
        plinfo.pushConstantRangeCount = 1;
        plinfo.pPushConstantRanges = &pc;

        if (vkCreatePipelineLayout(app->getDevice(), &plinfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        // central manager
        app->resources.addPipelineLayout(computePipelineLayout, "IndirectRenderer: computePipelineLayout");

        auto compCode = FileReader::readFile("shaders/indirect.comp.spv");
        VkShaderModule compModule = app->createShaderModule(compCode);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = compModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stage;
        pipelineInfo.layout = computePipelineLayout;

        if (vkCreateComputePipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            // Pipeline creation failed: unregister and destroy the shader module immediately
            app->resources.removeShaderModule(compModule);
            vkDestroyShaderModule(app->getDevice(), compModule, nullptr);
            throw std::runtime_error("failed to create compute pipeline!");
        }
        // track compute pipeline
        app->resources.addPipeline(computePipeline, "IndirectRenderer: computePipeline");
        // Clear local shader module reference; manager owns destruction
        compModule = VK_NULL_HANDLE;

        // Descriptor pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 10;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 4;
        // Allow freeing descriptor sets if needed
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor pool");
        }
        // track descriptor pool in central manager
        app->resources.addDescriptorPool(computeDescriptorPool, "IndirectRenderer: computeDescriptorPool");
        app->resources.addDescriptorPool(computeDescriptorPool, "IndirectRenderer: computeDescriptorPool");
        /* duplicate registration removed */

        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = computeDescriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &computeDescriptorSetLayout;
        if (vkAllocateDescriptorSets(app->getDevice(), &alloc, &computeDescriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor set");
        }
        app->resources.addDescriptorSet(computeDescriptorSet, "IndirectRenderer: computeDescriptorSet");
    }

    // Update compute descriptor set with buffer infos
    if (computeDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo inBuf{};
        inBuf.buffer = indirectBuffer.buffer;
        inBuf.offset = 0;
        inBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo outBuf{};
        outBuf.buffer = compactIndirectBuffer.buffer;
        outBuf.offset = 0;
        outBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo modelsBuf{};
        modelsBuf.buffer = modelsBuffer.buffer;
        modelsBuf.offset = 0;
        modelsBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo boundsBuf{};
        boundsBuf.buffer = boundsBuffer.buffer;
        boundsBuf.offset = 0;
        boundsBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo countBuf{};
        countBuf.buffer = visibleCountBuffer.buffer;
        countBuf.offset = 0;
        countBuf.range = VK_WHOLE_SIZE;

        // Check all buffers are valid before updating descriptor set
        if (indirectBuffer.buffer == VK_NULL_HANDLE ||
            compactIndirectBuffer.buffer == VK_NULL_HANDLE ||
            modelsBuffer.buffer == VK_NULL_HANDLE ||
            boundsBuffer.buffer == VK_NULL_HANDLE ||
            visibleCountBuffer.buffer == VK_NULL_HANDLE) {
            fprintf(stderr, "[IndirectRenderer] Skipping compute descriptor set update: one or more buffers are VK_NULL_HANDLE\n");
        } else {
            VkWriteDescriptorSet writes[5] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = computeDescriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &inBuf;

            writes[1] = writes[0]; writes[1].dstBinding = 1; writes[1].pBufferInfo = &outBuf;
            writes[2] = writes[0]; writes[2].dstBinding = 2; writes[2].pBufferInfo = &modelsBuf;
            writes[3] = writes[0]; writes[3].dstBinding = 3; writes[3].pBufferInfo = &boundsBuf;
            writes[4] = writes[0]; writes[4].dstBinding = 4; writes[4].pBufferInfo = &countBuf;

            vkUpdateDescriptorSets(app->getDevice(), 5, writes, 0, nullptr);
        }
    }

    // Try to load optional device function for indirect-count draws
    cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCountKHR");

    // Always update the main descriptor set with the models SSBO after rebuild
    if (modelsBuffer.buffer != VK_NULL_HANDLE) {
        VkDescriptorSet mainSet = app->getMainDescriptorSet();
        if (mainSet != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer = modelsBuffer.buffer;
            bufInfo.offset = 0;
            bufInfo.range = VK_WHOLE_SIZE;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = mainSet;
            write.dstBinding = 8; // Models SSBO binding in main descriptor set
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufInfo;
            vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);
            modelsDescriptorSet = mainSet;
            static bool printedOnce = false;
            if (!printedOnce) {
                printf("[IndirectRenderer::rebuild] Updated main descriptor set binding 8 with models buffer=%p\n", (void*)modelsBuffer.buffer);
                printedOnce = true;
            }
        }
    }

    // Perform deferred descriptor update (now safe since we called vkDeviceWaitIdle above)
    if (descriptorDirty && modelsBuffer.buffer != VK_NULL_HANDLE) {
        // Update the main descriptor set (set 0) binding 8 with the models SSBO
        VkDescriptorSet mainSet = app->getMainDescriptorSet();
        if (mainSet != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer = modelsBuffer.buffer;
            bufInfo.offset = 0;
            bufInfo.range = VK_WHOLE_SIZE;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = mainSet;
            write.dstBinding = 8; // Models SSBO binding in main descriptor set
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufInfo;
            vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);
            modelsDescriptorSet = mainSet;
        }
        descriptorDirty = false;
        pendingDescriptorSet = VK_NULL_HANDLE;
    }

    dirty = false;
}

void IndirectRenderer::drawMergedWithCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;
    // Default implementation splits: prepareCull() runs compute cull (outside render pass)
    // and drawPrepared() issues the indirect draw (inside render pass). For backward
    // compatibility, drawMergedWithCull will perform both steps here (caller may
    // instead call prepareCull/drawPrepared to ensure correct render-pass placement).
    prepareCull(cmd, viewProj, maxDraws);
    // After preparing, issue draw commands into the current command buffer.
    // Bind merged geometry
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    if (cmdDrawIndexedIndirectCount) {
        cmdDrawIndexedIndirectCount(cmd, compactIndirectBuffer.buffer, 0, visibleCountBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
    } else {
        vkCmdDrawIndexedIndirect(cmd, compactIndirectBuffer.buffer, 0, static_cast<uint32_t>(indirectCommands.size()), sizeof(VkDrawIndexedIndirectCommand));
    }
}

void IndirectRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    // and all buffer modifications happen in rebuild() which does lock.
    if (computePipeline == VK_NULL_HANDLE || compactIndirectBuffer.buffer == VK_NULL_HANDLE) {
        static bool reported = false;
        if (!reported) {
            printf("[IndirectRenderer::prepareCull] SKIP: computePipeline=%p, compactIndirectBuffer=%p\n", 
                (void*)computePipeline, (void*)compactIndirectBuffer.buffer);
            reported = true;
        }
        return;
    }
    
    static bool printedOnce = false;
    if (!printedOnce) {
        uint32_t numCmds = static_cast<uint32_t>(indirectCommands.size());
        printf("[IndirectRenderer::prepareCull] RUNNING: numCmds=%u, computePipeline=%p, computeDescriptorSet=%p\n", 
            numCmds, (void*)computePipeline, (void*)computeDescriptorSet);
        printedOnce = true;
    }
    
    // Reset visible count to zero using a command (clear GPU-side counter)
    vkCmdFillBuffer(cmd, visibleCountBuffer.buffer, 0, sizeof(uint32_t), 0);

    // Bind and dispatch compute cull
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::mat4), &viewProj);

    uint32_t numCmds = static_cast<uint32_t>(indirectCommands.size());
    uint32_t groupSize = 64;
    uint32_t groups = (numCmds + groupSize - 1) / groupSize;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier to make shader writes to the compact indirect buffer and visible count visible to indirect draw
    VkBufferMemoryBarrier barriers[2] = {};
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = compactIndirectBuffer.buffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    barriers[1] = barriers[0];
    barriers[1].buffer = visibleCountBuffer.buffer;
    barriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 2, barriers, 0, nullptr);
}

void IndirectRenderer::drawPrepared(VkCommandBuffer cmd, uint32_t maxDraws) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
        static bool reported = false;
        if (!reported) {
            printf("[IndirectRenderer::drawPrepared] vertex or index buffer is NULL, skipping\n");
            reported = true;
        }
        return;
    }

    static int frameCount = 0;
    if (frameCount < 10) {
        std::lock_guard<std::mutex> lock(mutex);
        printf("[IndirectRenderer::drawPrepared] Frame %d: vertexBuffer=%p (verts=%zu), indexBuffer=%p (indices=%zu), drawCommands=%zu\n",
               frameCount, (void*)vertexBuffer.buffer, mergedVertices.size(),
               (void*)indexBuffer.buffer, mergedIndices.size(), indirectCommands.size());
        size_t activeMeshCount = 0;
        for (const auto& kv : meshes) if (kv.second.active) ++activeMeshCount;
        printf("[IndirectRenderer::drawPrepared] meshes.size()=%zu, activeMeshCount=%zu\n", meshes.size(), activeMeshCount);
        int meshPrint = 0;
        for (const auto& kv : meshes) {
            if (!kv.second.active) continue;
            printf("  Mesh id=%u: baseVertex=%u, firstIndex=%u, indexCount=%u, boundsMin=(%.2f,%.2f,%.2f), boundsMax=(%.2f,%.2f,%.2f)\n",
                kv.second.id, kv.second.baseVertex, kv.second.firstIndex, kv.second.indexCount,
                kv.second.boundsMin.x, kv.second.boundsMin.y, kv.second.boundsMin.z,
                kv.second.boundsMax.x, kv.second.boundsMax.y, kv.second.boundsMax.z);
            if (++meshPrint >= 5) break;
        }
        // Print first few indirect commands
        for (size_t i = 0; i < std::min(size_t(3), indirectCommands.size()); ++i) {
            const auto& cmd = indirectCommands[i];
            printf("  IndirectCmd[%zu]: indexCount=%u, instanceCount=%u, firstIndex=%u, vertexOffset=%d, firstInstance=%u\n",
                i, cmd.indexCount, cmd.instanceCount, cmd.firstIndex, cmd.vertexOffset, cmd.firstInstance);
        }
        // Check if using indirect count
        if (cmdDrawIndexedIndirectCount) {
            printf("[IndirectRenderer::drawPrepared] Using GPU-driven count from visibleCountBuffer=%p\n",
                   (void*)visibleCountBuffer.buffer);
        }
        frameCount++;
    }

    // Bind merged geometry
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Issue indirect-draw call; compute shader compacts only visible commands
    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    
    if (cmdDrawIndexedIndirectCount) {
        // Use indirect-count variant to let the GPU supply the visible count from compute shader
        cmdDrawIndexedIndirectCount(cmd, compactIndirectBuffer.buffer, 0, visibleCountBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
    } else {
        // Fallback: draw all commands (no GPU count available)
        vkCmdDrawIndexedIndirect(cmd, compactIndirectBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

void IndirectRenderer::bindBuffers(VkCommandBuffer cmd) {
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
}

void IndirectRenderer::drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws) {
    drawIndirectOnly(cmd, app->getPipelineLayout(), maxDraws);
}

void IndirectRenderer::drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws) {
    if (compactIndirectBuffer.buffer == VK_NULL_HANDLE) {
        static bool reported = false;
        if (!reported) {
            printf("[IndirectRenderer::drawIndirectOnly] compactIndirectBuffer is VK_NULL_HANDLE, no draws\n");
            reported = true;
        }
        return;
    }
    // Push identity matrix for model transform
    // glm::mat4 identity = glm::mat4(1.0f);
    // vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &identity);

    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    if (cmdDrawIndexedIndirectCount) {
        cmdDrawIndexedIndirectCount(cmd, compactIndirectBuffer.buffer, 0, visibleCountBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
    } else {
        vkCmdDrawIndexedIndirect(cmd, compactIndirectBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

uint32_t IndirectRenderer::readVisibleCount(VulkanApp* app) const {
    if (!app || visibleCountBuffer.buffer == VK_NULL_HANDLE) return 0;
    // Stats-only: wait for GPU to finish so the counter is coherent before mapping.
    vkDeviceWaitIdle(app->getDevice());
    uint32_t count = 0;
    void* data = nullptr;
    if (vkMapMemory(app->getDevice(), visibleCountBuffer.memory, 0, sizeof(uint32_t), 0, &data) == VK_SUCCESS && data) {
        memcpy(&count, data, sizeof(uint32_t));
        vkUnmapMemory(app->getDevice(), visibleCountBuffer.memory);
    }
    return count;
}

void IndirectRenderer::updateModelsDescriptorSet(VkDescriptorSet ds) {
    std::lock_guard<std::mutex> guard(mutex);
    // Mark descriptor as needing update; actual vkUpdateDescriptorSets deferred
    // to rebuild() which waits for GPU idle first.
    descriptorDirty = true;
    pendingDescriptorSet = ds;
    // Also mark buffers dirty to trigger rebuild() on next frame
    dirty = true;
}

IndirectRenderer::MeshInfo IndirectRenderer::getMeshInfo(uint32_t meshId) const {
    IndirectRenderer::MeshInfo empty;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return empty;
    return it->second;
}

std::vector<IndirectRenderer::MeshInfo> IndirectRenderer::getActiveMeshInfos() const {
    std::vector<MeshInfo> out;
    std::lock_guard<std::mutex> guard(mutex);
    for (const auto& kv : meshes) {
        if (kv.second.active) out.push_back(kv.second);
    }
    return out;
}
