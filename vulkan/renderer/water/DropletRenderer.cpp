#include "DropletRenderer.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../../core/VulkanApp.hpp"

namespace {
struct DropletRenderPC {
    float particleCount = 0.0f;
    float radiusScale = 1.0f;
    float ior = 1.333f;
    float debugMode = 0.0f;
};
static_assert(sizeof(DropletRenderPC) == 16, "must match the shader push block");
} // namespace

void DropletRenderer::init(VulkanApp* vkApp) {
    if (pipeline_ != VK_NULL_HANDLE) return;
    app_ = vkApp;
    VkDevice device = vkApp->getDevice();

    // ── set 1: droplet state + sky view ────────────────────────────────────
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 2;
    dli.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &setLayout_) != VK_SUCCESS)
        throw std::runtime_error("DropletRenderer: descriptor set layout");
    vkApp->resources.addDescriptorSetLayout(setLayout_, "DropletRenderer: setLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(DropletRenderPC);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    const VkDescriptorSetLayout setLayouts[2] = {vkApp->getDescriptorSetLayout(), setLayout_};
    pli.setLayoutCount = 2;
    pli.pSetLayouts = setLayouts;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &layout_) != VK_SUCCESS)
        throw std::runtime_error("DropletRenderer: pipeline layout");
    vkApp->resources.addPipelineLayout(layout_, "DropletRenderer: layout");

    VkDescriptorPoolSize ps[2] = {};
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[0].descriptorCount = kSlots;
    ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[1].descriptorCount = kSlots;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    dpi.maxSets = kSlots;
    if (vkCreateDescriptorPool(device, &dpi, nullptr, &pool_) != VK_SUCCESS)
        throw std::runtime_error("DropletRenderer: descriptor pool");
    vkApp->resources.addDescriptorPool(pool_, "DropletRenderer: pool");

    VkDescriptorSetLayout layouts[kSlots] = {setLayout_, setLayout_, setLayout_};
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = kSlots;
    ai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(device, &ai, sets_) != VK_SUCCESS)
        throw std::runtime_error("DropletRenderer: descriptor sets");
    for (uint32_t i = 0; i < kSlots; ++i)
        vkApp->resources.addDescriptorSet(sets_[i], "DropletRenderer: set");

    // Linear clamp sampler for the sky view (mip chain sampled by the shader).
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = VK_LOD_CLAMP_NONE;
    if (vkCreateSampler(device, &si, nullptr, &sampler_) != VK_SUCCESS)
        throw std::runtime_error("DropletRenderer: sampler");
    vkApp->resources.addSampler(sampler_, "DropletRenderer: sampler");

    // ── pipeline: billboards inside the solid color pass ───────────────────
    VkShaderModule vert = vkApp->getOrCreateShaderModule("shaders/renderer/water/DropletRenderer.vert.spv");
    VkShaderModule frag = vkApp->getOrCreateShaderModule("shaders/renderer/water/DropletRenderer.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    GraphicsPipelineConfig cfg{};
    cfg.depthTestEnable = true;
    cfg.depthWriteEnable = false; // transparent: tested, never written
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.colorFormats = {vkApp->getSwapchainImageFormat()};
    cfg.depthFormat = VK_FORMAT_D32_SFLOAT;
    // Billboards are flat and camera-facing: no culling, no winding to match.
    cfg.cullMode = VK_CULL_MODE_NONE;
    cfg.blendEnable = true;
    // Premultiplied coverage (the shader outputs color * alpha, alpha): OVER is
    // dst * (1 - alpha) + color, exactly like the other coverage-composited
    // passes in the engine.
    cfg.blendSrcColorFactor = VK_BLEND_FACTOR_ONE;
    cfg.blendDstColorFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cfg.blendSrcAlphaFactor = VK_BLEND_FACTOR_ONE;
    cfg.blendDstAlphaFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    // The billboard is generated from gl_VertexIndex: no vertex input state.
    auto [pipeline, layout] = vkApp->createGraphicsPipeline(
        {stages[0], stages[1]}, {}, {}, {vkApp->getDescriptorSetLayout(), setLayout_}, &pcr, cfg);
    (void)layout; // registered by the helper; layout_ above is the owned one
    pipeline_ = pipeline;
}

void DropletRenderer::destroy(VulkanApp* vkApp) {
    (void)vkApp;
    // Pipeline/layout/pool/sets/sampler are owned by VulkanResourceManager.
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    setLayout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    sampler_ = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < kSlots; ++i) {
        sets_[i] = VK_NULL_HANDLE;
        boundState_[i] = VK_NULL_HANDLE;
        boundSky_[i] = VK_NULL_HANDLE;
    }
    app_ = nullptr;
}

void DropletRenderer::writeSet(uint32_t slot, VkBuffer state, VkImageView sky) {
    VkDescriptorBufferInfo bi{};
    bi.buffer = state;
    bi.offset = 0;
    bi.range = app_->getClampedWholeSizeRange(state);
    VkDescriptorImageInfo ii{};
    ii.sampler = sampler_;
    ii.imageView = sky;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = sets_[slot];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bi;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sets_[slot];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &ii;
    vkUpdateDescriptorSets(app_->getDevice(), 2, writes, 0, nullptr);
    boundState_[slot] = state;
    boundSky_[slot] = sky;
}

bool DropletRenderer::draw(VkCommandBuffer cmd, VulkanApp* vkApp, uint32_t frameIndex,
                           VkDescriptorSet mainSet, VkBuffer dropletState, uint32_t count,
                           float radiusScale, float ior, float debugMode, VkImageView skyView) {
    if (pipeline_ == VK_NULL_HANDLE || dropletState == VK_NULL_HANDLE || count == 0u ||
        skyView == VK_NULL_HANDLE || mainSet == VK_NULL_HANDLE) {
        return false;
    }
    const uint32_t slot = frameIndex % kSlots;
    if (boundState_[slot] != dropletState || boundSky_[slot] != skyView) {
        writeSet(slot, dropletState, skyView);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    VkDescriptorSet sets[2] = {mainSet, sets_[slot]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 2, sets, 0, nullptr);

    DropletRenderPC pc{};
    pc.particleCount = static_cast<float>(count);
    pc.radiusScale = radiusScale;
    pc.ior = ior;
    pc.debugMode = debugMode;
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    // 6 non-indexed billboard vertices, one instance per simulated droplet.
    vkCmdDraw(cmd, 6, count, 0, 0);
    return true;
}
