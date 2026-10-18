#include "OctreeMeshCompute.hpp"
#include "../vulkan/core/VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include <cstring>
#include <vector>

namespace {
struct PushPC { uint32_t cellCount; float isoValue; };
}

OctreeMeshCompute* OctreeMeshCompute::globalInstance = nullptr;

OctreeMeshCompute::OctreeMeshCompute() {
    globalInstance = this;
}

void OctreeMeshCompute::init(VulkanApp* appInstance) {
    this->app = appInstance;
    if (pipeline != VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    VkDescriptorSetLayoutBinding bindings[7] = {};
    for (int b = 0; b < 7; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 7;
    dli.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout) != VK_SUCCESS)
        throw std::runtime_error("OctreeMeshCompute: desc layout");
    app->resources.addDescriptorSetLayout(descLayout, "OctreeMeshCompute: descLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushPC);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &descLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("OctreeMeshCompute: pipeline layout");
    app->resources.addPipelineLayout(layout, "OctreeMeshCompute: layout");

    VkShaderModule sm = app->getOrCreateShaderModule("shaders/renderer/octree/OctreeMesh.comp.spv");

    VkPipelineShaderStageCreateInfo ssi{};
    ssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssi.module = sm;
    ssi.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = ssi;
    cpi.layout = layout;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &cpi, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("OctreeMeshCompute: pipeline");
    app->resources.addPipeline(pipeline, "OctreeMeshCompute: pipeline");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 7 * 64;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    dpi.maxSets = 64;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device, &dpi, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("OctreeMeshCompute: desc pool");
    app->resources.addDescriptorPool(descPool, "OctreeMeshCompute: descPool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout;
    if (vkAllocateDescriptorSets(device, &ai, &descSet) != VK_SUCCESS)
        throw std::runtime_error("OctreeMeshCompute: alloc desc set (init)");
    app->resources.addDescriptorSet(descSet, "OctreeMeshCompute: descSet");
}

void OctreeMeshCompute::destroy(VulkanApp* appInstance) {
    (void)appInstance;
    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    descLayout = VK_NULL_HANDLE;
    descPool = VK_NULL_HANDLE;
}

OctreeMeshResult OctreeMeshCompute::compute(const std::vector<OctreeMeshCell>& cells, float isoValue) {
    std::lock_guard<std::mutex> lock(mutex);
    OctreeMeshResult res;
    if (cells.empty()) return res;
    if (!app) return res;
    if (pipeline == VK_NULL_HANDLE) init(app);

    VkDevice device = app->getDevice();
    uint32_t n = (uint32_t)cells.size();

    VkDeviceSize szVec = sizeof(float) * 4 * n;
    Buffer bSdf0 = app->createBuffer(szVec, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer bSdf1 = app->createBuffer(szVec, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer bCube = app->createBuffer(szVec, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer bBrush = app->createBuffer(sizeof(int) * n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer bPos = app->createBuffer(szVec, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer bNorm = app->createBuffer(szVec, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer bCounter = app->createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Fill inputs
    for (uint32_t i = 0; i < n; ++i) {
        float* p0 = (float*)bSdf0.mappedData;
        float* p1 = (float*)bSdf1.mappedData;
        float* pcu = (float*)bCube.mappedData;
        int* pbr = (int*)bBrush.mappedData;
        const auto& c = cells[i];
        for (int k = 0; k < 4; ++k) p0[i*4+k] = c.sdf[k];
        for (int k = 0; k < 4; ++k) p1[i*4+k] = c.sdf[4+k];
        pcu[i*4+0] = c.cubeMin.x;
        pcu[i*4+1] = c.cubeMin.y;
        pcu[i*4+2] = c.cubeMin.z;
        pcu[i*4+3] = c.cubeSize;
        pbr[i] = c.brushIndex;
    }
    // zero counter
    *(uint32_t*)bCounter.mappedData = 0;

    VkDescriptorBufferInfo infos[7];
    infos[0] = {bSdf0.buffer, 0, VK_WHOLE_SIZE};
    infos[1] = {bSdf1.buffer, 0, VK_WHOLE_SIZE};
    infos[2] = {bCube.buffer, 0, VK_WHOLE_SIZE};
    infos[3] = {bBrush.buffer, 0, VK_WHOLE_SIZE};
    infos[4] = {bPos.buffer, 0, VK_WHOLE_SIZE};
    infos[5] = {bNorm.buffer, 0, VK_WHOLE_SIZE};
    infos[6] = {bCounter.buffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[7] = {};
    for (int b = 0; b < 7; ++b) {
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = descSet;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);

    // Dispatch
    PushPC pc{ n, isoValue };
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushPC), &pc);
        uint32_t groups = (n + 63) / 64;
        vkCmdDispatch(cmd, groups, 1, 1);
        // barrier
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    });

    // Read back
    uint32_t vcount = *(uint32_t*)bCounter.mappedData;
    res.vertexCount = vcount;
    res.positions.resize(vcount);
    res.normals.resize(vcount);
    float* pp = (float*)bPos.mappedData;
    float* pn = (float*)bNorm.mappedData;
    for (uint32_t i = 0; i < vcount; ++i) {
        res.positions[i] = glm::vec4(pp[i*4], pp[i*4+1], pp[i*4+2], pp[i*4+3]);
        res.normals[i] = glm::vec4(pn[i*4], pn[i*4+1], pn[i*4+2], pn[i*4+3]);
    }

    // Cleanup
    app->destroyBuffer(bSdf0);
    app->destroyBuffer(bSdf1);
    app->destroyBuffer(bCube);
    app->destroyBuffer(bBrush);
    app->destroyBuffer(bPos);
    app->destroyBuffer(bNorm);
    app->destroyBuffer(bCounter);

    return res;
}