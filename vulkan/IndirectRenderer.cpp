#include "IndirectRenderer.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"

IndirectRenderer::IndirectRenderer() {}
IndirectRenderer::~IndirectRenderer() {}

void IndirectRenderer::init(VulkanApp* app) {
    (void)app;
}

void IndirectRenderer::cleanup(VulkanApp* app) {
    // meshes no longer own per-mesh buffers; clear CPU lists
    meshes.clear();
    idToIndex.clear();
    if (vertexBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), vertexBuffer.buffer, nullptr);
        vertexBuffer = {};
    }
    if (vertexBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), vertexBuffer.memory, nullptr);
        vertexBuffer.memory = VK_NULL_HANDLE;
    }
    if (indexBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), indexBuffer.buffer, nullptr);
        indexBuffer = {};
    }
    if (indexBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), indexBuffer.memory, nullptr);
        indexBuffer.memory = VK_NULL_HANDLE;
    }
    if (indirectBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), indirectBuffer.buffer, nullptr);
        indirectBuffer = {};
    }
    if (indirectBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), indirectBuffer.memory, nullptr);
        indirectBuffer.memory = VK_NULL_HANDLE;
    }
    if (compactIndirectBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), compactIndirectBuffer.buffer, nullptr);
        compactIndirectBuffer = {};
    }
    if (compactIndirectBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), compactIndirectBuffer.memory, nullptr);
        compactIndirectBuffer.memory = VK_NULL_HANDLE;
    }
    if (modelsBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), modelsBuffer.buffer, nullptr);
        modelsBuffer = {};
    }
    if (modelsBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), modelsBuffer.memory, nullptr);
        modelsBuffer.memory = VK_NULL_HANDLE;
    }
    if (boundsBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), boundsBuffer.buffer, nullptr);
        boundsBuffer = {};
    }
    if (boundsBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), boundsBuffer.memory, nullptr);
        boundsBuffer.memory = VK_NULL_HANDLE;
    }
    if (visibleCountBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), visibleCountBuffer.buffer, nullptr);
        visibleCountBuffer = {};
    }
    if (visibleCountBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), visibleCountBuffer.memory, nullptr);
        visibleCountBuffer.memory = VK_NULL_HANDLE;
    }

    if (computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(app->getDevice(), computePipeline, nullptr);
    if (computePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(app->getDevice(), computePipelineLayout, nullptr);
    if (computeDescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(app->getDevice(), computeDescriptorSetLayout, nullptr);
    if (computeDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(app->getDevice(), computeDescriptorPool, nullptr);
}

uint32_t IndirectRenderer::addMesh(VulkanApp* app, const Geometry& mesh, const glm::mat4& model) {
    std::lock_guard<std::mutex> guard(mutex);
    MeshInfo info;
    info.id = nextId++;
    info.baseVertex = 0;
    info.firstIndex = 0;
    info.indexCount = static_cast<uint32_t>(mesh.indices.size());
    info.model = model;
    info.active = true;

    // Append vertex/index data into merged CPU-side arrays and record offsets
    info.baseVertex = static_cast<uint32_t>(mergedVertices.size());
    info.firstIndex = static_cast<uint32_t>(mergedIndices.size());

    // copy vertices
    mergedVertices.insert(mergedVertices.end(), mesh.vertices.begin(), mesh.vertices.end());

    // copy indices (keep original indices; we'll use vertexOffset in the draw command)
    for (uint32_t idx : mesh.indices) {
        mergedIndices.push_back(static_cast<uint32_t>(idx));
    }

    // compute bounding sphere in object space
    glm::vec3 minp(FLT_MAX), maxp(-FLT_MAX);
    for (auto &v : mesh.vertices) {
        minp = glm::min(minp, v.position);
        maxp = glm::max(maxp, v.position);
    }
    info.boundsMin = glm::vec4(minp, 0.0f);
    info.boundsMax = glm::vec4(maxp, 0.0f);

    // create indirect command for this mesh using merged offsets
    VkDrawIndexedIndirectCommand cmd{};
    cmd.indexCount = info.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex = info.firstIndex;
    cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
    cmd.firstInstance = 0;

    idToIndex[info.id] = meshes.size();
    meshes.push_back(info);

    indirectCommands.push_back(cmd);
    dirty = true;

    // Defer GPU buffer rebuild to main thread (rebuild will be called once per-frame).
    return info.id;
}

void IndirectRenderer::removeMesh(uint32_t meshId) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = idToIndex.find(meshId);
    if (it == idToIndex.end()) return;
    size_t idx = it->second;
    meshes[idx].active = false;
    idToIndex.erase(it);
    dirty = true;
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!dirty) return;

    // Build merged GPU-side vertex and index buffers from the CPU arrays.
    // If there are no meshes, free existing buffers.
    if (mergedVertices.empty() || mergedIndices.empty()) {
        if (vertexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(app->getDevice(), vertexBuffer.buffer, nullptr);
            vkFreeMemory(app->getDevice(), vertexBuffer.memory, nullptr);
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(app->getDevice(), indexBuffer.buffer, nullptr);
            vkFreeMemory(app->getDevice(), indexBuffer.memory, nullptr);
            indexBuffer = {};
        }
    } else {
        // recreate vertex/index buffers via app helpers (which perform staging transfers)
        if (vertexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(app->getDevice(), vertexBuffer.buffer, nullptr);
            vkFreeMemory(app->getDevice(), vertexBuffer.memory, nullptr);
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(app->getDevice(), indexBuffer.buffer, nullptr);
            vkFreeMemory(app->getDevice(), indexBuffer.memory, nullptr);
            indexBuffer = {};
        }
        vertexBuffer = app->createVertexBuffer(mergedVertices);
        indexBuffer = app->createIndexBuffer(mergedIndices);
    }

    // Rebuild indirect command list from active meshes so GPU-side compaction matches models/bounds
    std::vector<VkDrawIndexedIndirectCommand> cmds;
    cmds.reserve(meshes.size());
    for (size_t i = 0; i < meshes.size(); ++i) {
        if (!meshes[i].active) continue;
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = meshes[i].indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = meshes[i].firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(meshes[i].baseVertex);
        // Use firstInstance to carry the model index so shaders can fetch per-draw model
        cmd.firstInstance = static_cast<uint32_t>(cmds.size());
        cmds.push_back(cmd);
    }
    indirectCommands = cmds;

    // Create or update the global indirect buffer containing commands for all meshes
    VkDeviceSize indirectSize = sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size();
    if (indirectBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), indirectBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), indirectBuffer.memory, nullptr);
        indirectBuffer = {};
    }
    if (indirectSize > 0) {
        indirectBuffer = app->createBuffer(indirectSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data;
        vkMapMemory(app->getDevice(), indirectBuffer.memory, 0, indirectSize, 0, &data);
        memcpy(data, indirectCommands.data(), (size_t)indirectSize);
        vkUnmapMemory(app->getDevice(), indirectBuffer.memory);
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
    for (size_t i = 0; i < meshes.size(); ++i) {
        if (!meshes[i].active) continue;
        // Use identity matrices for all draws (caller requested identity-only transforms)
        models.push_back(glm::mat4(1.0f));
    }

    VkDeviceSize modelsSize = sizeof(glm::mat4) * models.size();
    if (modelsBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), modelsBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), modelsBuffer.memory, nullptr);
        modelsBuffer = {};
    }
    if (modelsSize > 0) {
        modelsBuffer = app->createBuffer(modelsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mdata;
        vkMapMemory(app->getDevice(), modelsBuffer.memory, 0, modelsSize, 0, &mdata);
        memcpy(mdata, models.data(), (size_t)modelsSize);
        vkUnmapMemory(app->getDevice(), modelsBuffer.memory);
    }

    // Upload bounds SSBO (two vec4s per active mesh: min, max)
    std::vector<glm::vec4> boundsData;
    boundsData.reserve(meshes.size() * 2);
    for (size_t i = 0; i < meshes.size(); ++i) {
        if (!meshes[i].active) continue;
        boundsData.push_back(meshes[i].boundsMin);
        boundsData.push_back(meshes[i].boundsMax);
    }
    VkDeviceSize boundsSize = sizeof(glm::vec4) * boundsData.size();
    if (boundsBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), boundsBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), boundsBuffer.memory, nullptr);
        boundsBuffer = {};
    }
    if (boundsSize > 0) {
        boundsBuffer = app->createBuffer(boundsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* bdata;
        vkMapMemory(app->getDevice(), boundsBuffer.memory, 0, boundsSize, 0, &bdata);
        memcpy(bdata, boundsData.data(), (size_t)boundsSize);
        vkUnmapMemory(app->getDevice(), boundsBuffer.memory);
    }

    // Create/resize compact indirect buffer (storage + indirect usage)
    VkDeviceSize compactSize = indirectSize; // same maximum capacity as full indirect buffer
    if (compactIndirectBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), compactIndirectBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), compactIndirectBuffer.memory, nullptr);
        compactIndirectBuffer = {};
    }
    if (compactSize > 0) {
        compactIndirectBuffer = app->createBuffer(compactSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // Create or zero the visible count buffer (single uint)
    VkDeviceSize countSize = sizeof(uint32_t);
    if (visibleCountBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), visibleCountBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), visibleCountBuffer.memory, nullptr);
        visibleCountBuffer = {};
    }
    visibleCountBuffer = app->createBuffer(countSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // initialize to zero
    uint32_t zero = 0;
    void* zdata;
    vkMapMemory(app->getDevice(), visibleCountBuffer.memory, 0, countSize, 0, &zdata);
    memcpy(zdata, &zero, sizeof(uint32_t));
    vkUnmapMemory(app->getDevice(), visibleCountBuffer.memory);

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

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(glm::mat4);

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        plinfo.pSetLayouts = &computeDescriptorSetLayout;
        plinfo.pushConstantRangeCount = 1;
        plinfo.pPushConstantRanges = &pc;

        if (vkCreatePipelineLayout(app->getDevice(), &plinfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

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
            vkDestroyShaderModule(app->getDevice(), compModule, nullptr);
            throw std::runtime_error("failed to create compute pipeline!");
        }
        vkDestroyShaderModule(app->getDevice(), compModule, nullptr);

        // Descriptor pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 10;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 4;
        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor pool");
        }

        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = computeDescriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &computeDescriptorSetLayout;
        if (vkAllocateDescriptorSets(app->getDevice(), &alloc, &computeDescriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor set");
        }
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

    // Try to load optional device function for indirect-count draws
    cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCountKHR");

    // Descriptor set updates are deferred to the application to ensure the
    // target descriptor pool/layout is compatible with the Models SSBO.
    // If the application wants the renderer to update a specific set it can
    // call `updateModelsDescriptorSet()` explicitly after confirming compatibility.

    dirty = false;
}

void IndirectRenderer::drawMergedWithCull(VkCommandBuffer cmd, const glm::mat4& viewProj, VulkanApp* app, uint32_t maxDraws) {
    std::lock_guard<std::mutex> guard(mutex);
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
    std::lock_guard<std::mutex> guard(mutex);
    if (computePipeline == VK_NULL_HANDLE || compactIndirectBuffer.buffer == VK_NULL_HANDLE) return;
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

void IndirectRenderer::drawPrepared(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws) {
    std::lock_guard<std::mutex> guard(mutex);
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;

    // Bind merged geometry
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    // Ensure shaders use identity model matrix for all draws
    glm::mat4 identity = glm::mat4(1.0f);
    vkCmdPushConstants(cmd, app->getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &identity);

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

void IndirectRenderer::updateModelsDescriptorSet(VulkanApp* app, VkDescriptorSet ds) {
    std::lock_guard<std::mutex> guard(mutex);
    // Update or create a material-style descriptor set for the Models SSBO.
    // If the caller provides a descriptor set (`ds`), write into it; otherwise
    // allocate a new one compatible with the app's material descriptor layout.
    if (modelsBuffer.buffer == VK_NULL_HANDLE) return; // nothing to bind yet

    VkDescriptorSet target = ds;
    if (target == VK_NULL_HANDLE) {
        // Prefer updating the application's existing global material descriptor
        // set (which already contains binding 5 -> Materials SSBO). If the app
        // hasn't created one, allocate a new set compatible with the material
        // layout as a fallback.
        VkDescriptorSet existingMat = app->getMaterialDescriptorSet();
        if (existingMat != VK_NULL_HANDLE) {
            target = existingMat;
        } else {
            VkDescriptorSetLayout layout = app->getMaterialDescriptorSetLayout();
            if (layout == VK_NULL_HANDLE) return;
            target = app->createDescriptorSet(layout);
            if (target == VK_NULL_HANDLE) return;
        }
    }

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = modelsBuffer.buffer;
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = target;
    write.dstBinding = 6; // binding chosen for Models SSBO in material set layout
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(app->getDevice(), 1, &write, 0, nullptr);

    // Store the target so drawing code can reference it if needed.
    modelsDescriptorSet = target;

    // Register only if we allocated the set here (caller will track provided sets)
    if (ds == VK_NULL_HANDLE) app->registerDescriptorSet(target);
}

IndirectRenderer::MeshInfo IndirectRenderer::getMeshInfo(uint32_t meshId) const {
    IndirectRenderer::MeshInfo empty;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = idToIndex.find(meshId);
    if (it == idToIndex.end()) return empty;
    return meshes[it->second];
}
 