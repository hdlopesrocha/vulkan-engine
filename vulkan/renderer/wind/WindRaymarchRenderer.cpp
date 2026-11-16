#include "WindRaymarchRenderer.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "../DescriptorAllocator.hpp"
#include "../DescriptorWriter.hpp"
#include "../../ubo/WindField.hpp"

namespace {

// Descriptor bindings of the raymarch set (set 0 of its pipeline).
constexpr uint32_t kParamsBinding = 0;
constexpr uint32_t kSceneDepthBinding = 1;

} // namespace

void WindRaymarchRenderer::init(VulkanApp* app) {
    if (app_ != nullptr || app == nullptr) return;
    app_ = app;

    depthSampler_ = app->createSamplerLinearClamp("WindRaymarchRenderer: depthSampler");
    createDescriptors(app);
    createPipeline(app);

    std::cout << "[WindRaymarchRenderer] initialized" << std::endl;
}

void WindRaymarchRenderer::createDescriptors(VulkanApp* app) {
    for (Buffer& b : paramsBuffers_) {
        b = app->createBuffer(sizeof(WindRaymarchUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    DescriptorAllocator descAlloc{app->getDevice(), app};
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding = kParamsBinding;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = kSceneDepthBinding;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Shared wind field: same binding number as the scene set so
    // WindField.glsl is reused verbatim.
    bindings[2].binding = kWindFieldBinding;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    descriptorSetLayout_ = descAlloc.createLayout(
        bindings.data(), static_cast<uint32_t>(bindings.size()), 0, nullptr,
        "WindRaymarchRenderer: descriptorSetLayout");

    const VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * VulkanApp::MAX_FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VulkanApp::MAX_FRAMES_IN_FLIGHT},
    };
    descriptorPool_ = descAlloc.createPool(
        poolSizes, 2, VulkanApp::MAX_FRAMES_IN_FLIGHT,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        "WindRaymarchRenderer: descriptorPool");

    descAlloc.allocateSets(descriptorPool_, descriptorSetLayout_, VulkanApp::MAX_FRAMES_IN_FLIGHT,
                           descriptorSets_.data(), "WindRaymarchRenderer: descriptorSet");

    // Binding 0 is stable per slot. Bindings 1/27 are written in render() when
    // the bound scene-depth view or wind-field buffer changes.
    for (uint32_t i = 0; i < VulkanApp::MAX_FRAMES_IN_FLIGHT; ++i) {
        boundDepthView_[i] = VK_NULL_HANDLE;
        boundWindField_[i] = VK_NULL_HANDLE;
        DescriptorWriter(app->getDevice())
            .writeBuffer(descriptorSets_[i], kParamsBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         paramsBuffers_[i].buffer, 0, sizeof(WindRaymarchUBO))
            .flush();
    }
}

void WindRaymarchRenderer::createPipeline(VulkanApp* app) {
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/renderer/Fullscreen.vert.spv");
    VkShaderModule fragModule = app->getOrCreateShaderModule("shaders/renderer/wind/WindRaymarchRenderer.frag.spv");
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        std::cerr << "[WindRaymarchRenderer] missing shader modules" << std::endl;
        return;
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr},
    };

    // Alpha-blended overlay over the composed frame; the march is clipped by
    // the sampled scene depth, so no depth test/write is needed.
    GraphicsPipelineConfig cfg{};
    cfg.polygonMode = VK_POLYGON_MODE_FILL;
    cfg.cullMode = VK_CULL_MODE_NONE;
    cfg.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    cfg.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    cfg.depthTestEnable = false;
    cfg.depthWriteEnable = false;
    cfg.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    cfg.colorFormats = {app->getSwapchainImageFormat()};
    cfg.depthFormat = VK_FORMAT_D32_SFLOAT;
    cfg.blendEnable = true;
    cfg.blendSrcColorFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cfg.blendDstColorFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cfg.blendSrcAlphaFactor = VK_BLEND_FACTOR_ONE;
    cfg.blendDstAlphaFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    const std::vector<VkDescriptorSetLayout> setLayouts = {descriptorSetLayout_};
    auto pipelineAndLayout = app->createGraphicsPipeline(
        {stages[0], stages[1]},
        std::vector<VkVertexInputBindingDescription>{},
        std::vector<VkVertexInputAttributeDescription>{},
        setLayouts, nullptr, cfg);
    pipeline_ = pipelineAndLayout.first;
    pipelineLayout_ = pipelineAndLayout.second;
    if (pipeline_ == VK_NULL_HANDLE) {
        std::cerr << "[WindRaymarchRenderer] failed to create pipeline" << std::endl;
    }
}

void WindRaymarchRenderer::prepare(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                                   bool enabled, const WindRaymarchSettings& settings,
                                   const glm::mat4& invViewProjection,
                                   const glm::vec3& cameraPosition,
                                   VkImageView sceneDepthView, VkBuffer windFieldBuffer,
                                   float timeSeconds) {
    active_ = false;
    if (!enabled || app == nullptr || cmd == VK_NULL_HANDLE) return;
    if (pipeline_ == VK_NULL_HANDLE) return;
    if (sceneDepthView == VK_NULL_HANDLE || windFieldBuffer == VK_NULL_HANDLE) return;

    const uint32_t slot = frameIdx % VulkanApp::MAX_FRAMES_IN_FLIGHT;
    VkDescriptorSet descriptorSet = descriptorSets_[slot];
    if (descriptorSet == VK_NULL_HANDLE || paramsBuffers_[slot].mappedData == nullptr) return;

    if (boundDepthView_[slot] != sceneDepthView || boundWindField_[slot] != windFieldBuffer) {
        DescriptorWriter(app->getDevice())
            .writeImage(descriptorSet, kSceneDepthBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        depthSampler_, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .writeBuffer(descriptorSet, kWindFieldBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         windFieldBuffer, 0, sizeof(WindField))
            .flush();
        boundDepthView_[slot] = sceneDepthView;
        boundWindField_[slot] = windFieldBuffer;
    }

    WindRaymarchUBO ubo{};
    ubo.invViewProjection = invViewProjection;
    ubo.surface = glm::vec4(std::max(settings.targetSpeed, 0.001f),
                            std::max(settings.maxDistance, 1.0f),
                            static_cast<float>(std::clamp(settings.steps, 8, 128)),
                            std::clamp(settings.opacity, 0.0f, 1.0f));
    ubo.viewport = glm::vec4(static_cast<float>(std::max(app->getWidth(), 1)),
                             static_cast<float>(std::max(app->getHeight(), 1)),
                             std::max(timeSeconds, 0.0f), 0.0f);
    ubo.cameraPosition = glm::vec4(cameraPosition, 0.0f);
    std::memcpy(paramsBuffers_[slot].mappedData, &ubo, sizeof(ubo));

    // The parameters UBO and the shared wind field are host-written while this
    // command buffer is recorded: make them visible to the fragment stage.
    // Recorded here, outside the rendering scope (framebuffer-space stages
    // only are legal inside dynamic rendering).
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    active_ = true;
}

void WindRaymarchRenderer::render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                                  bool enabled) {
    if (!enabled || !active_ || app == nullptr || cmd == VK_NULL_HANDLE) return;
    if (pipeline_ == VK_NULL_HANDLE || pipelineLayout_ == VK_NULL_HANDLE) return;

    const uint32_t slot = frameIdx % VulkanApp::MAX_FRAMES_IN_FLIGHT;
    VkDescriptorSet descriptorSet = descriptorSets_[slot];
    if (descriptorSet == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(std::max(app->getWidth(), 1));
    viewport.height = static_cast<float>(std::max(app->getHeight(), 1));
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(viewport.width), static_cast<uint32_t>(viewport.height)};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                            0, 1, &descriptorSet, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void WindRaymarchRenderer::cleanup(VulkanApp* app) {
    if (app == nullptr) return;

    for (Buffer& b : paramsBuffers_) {
        if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
        b = {};
    }
    // Pipeline / layout / pool / sets / sampler are owned by
    // VulkanResourceManager and released by its teardown.
    descriptorSetLayout_ = VK_NULL_HANDLE;
    descriptorPool_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    depthSampler_ = VK_NULL_HANDLE;
    descriptorSets_.fill(VK_NULL_HANDLE);
    boundDepthView_.fill(VK_NULL_HANDLE);
    boundWindField_.fill(VK_NULL_HANDLE);
    active_ = false;
    app_ = nullptr;
}
