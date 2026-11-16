#include "WindRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../DescriptorAllocator.hpp"
#include "../DescriptorWriter.hpp"
#include "../../ubo/WindField.hpp"
#include "../../ubo/WindFieldConstants.hpp"

namespace {

// Descriptor bindings of the wind renderer's own set (set 0 of its pipelines).
// The shared wind field deliberately keeps the SCENE binding number (27) so the
// GLSL include (shaders/includes/vegetation/WindField.glsl) is reused verbatim.
constexpr uint32_t kParamsBinding = 0;
constexpr uint32_t kParticleBinding = 1;
constexpr uint32_t kTrailBinding = 2;
constexpr uint32_t kTrailSpeedBinding = 3;

// Must match the std430 struct in shaders/includes/wind/WindParticle.glsl
// (vec4 + uvec4 + vec4 = 48 bytes).
constexpr VkDeviceSize kParticleStride = 48;

// Compute shader workgroup size (shaders/computation/wind/WindParticles.comp).
constexpr uint32_t kWorkgroupSize = 64;

// Bound on a single frame's simulation step (s), mirrored in the shader's
// clamp: a hitch or a breakpoint resume must not teleport the particles.
constexpr float kMaxFrameStep = 0.1f;

// Widened frustum test so a particle grazing the screen edge is not respawned
// every frame; the spawn point itself is sampled well inside the view.
constexpr float kFrustumMargin = 1.05f;
// Seconds over which a freshly (re)spawned particle fades in.
constexpr float kSpawnFadeSeconds = 0.25f;

} // namespace

uint32_t WindRenderer::trailSegments(float durationSeconds) {
    const float duration = std::clamp(durationSeconds, 0.05f, 60.0f);
    const float segments = std::ceil(duration / kTrailSampleInterval);
    return std::clamp(static_cast<uint32_t>(segments), 1u, kTrailCapacity - 1u);
}

void WindRenderer::init(VulkanApp* app) {
    if (app_ != nullptr) return;
    if (app == nullptr) return;
    app_ = app;

    createBuffers(app);
    createDescriptors(app);
    createComputePipeline(app);
    createTrailPipeline(app);

    std::cout << "[WindRenderer] initialized: " << kMaxParticles << " particle capacity, "
              << kTrailCapacity << " trail samples/particle" << std::endl;
}

void WindRenderer::createBuffers(VulkanApp* app) {
    // Per-frame parameters UBO. Streamed by the host (memcpy) each dispatch;
    // the same contents are read by the compute pass and the trail draw.
    for (Buffer& b : paramsBuffers_) {
        b = app->createBuffer(sizeof(WindParticlesUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    // Particle state: zero-initialized so flags == 0 means "not spawned yet"
    // (the compute shader seeds it on first use).
    particleBuffer_ = app->createBuffer(static_cast<VkDeviceSize>(kMaxParticles) * kParticleStride,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        /*zeroInit*/ true);
    // Trail ring: never read before a sample is written (count >= 2 implies
    // both slots were stored after the last respawn), so no zero-fill pass.
    trailBuffer_ = app->createBuffer(static_cast<VkDeviceSize>(kMaxParticles) * kTrailCapacity * sizeof(glm::vec4),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                     /*zeroInit*/ false);
    // Per-sample wind speed (m/s) written next to each trail sample: drives the
    // speed->color ramp without re-sampling the field per vertex.
    trailSpeedBuffer_ = app->createBuffer(static_cast<VkDeviceSize>(kMaxParticles) * kTrailCapacity * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                          /*zeroInit*/ false);
}

void WindRenderer::createDescriptors(VulkanApp* app) {
    DescriptorAllocator descAlloc{app->getDevice(), app};

    const VkShaderStageFlags graphicsStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    bindings[0].binding = kParamsBinding;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | graphicsStages;
    bindings[1].binding = kParticleBinding;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    bindings[2].binding = kTrailBinding;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    // Per-sample speed, written by the simulation and read by the trail draw's
    // speed->color ramp.
    bindings[3].binding = kTrailSpeedBinding;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    // Shared wind field: same binding number as the scene set, compute-only.
    bindings[4].binding = kWindFieldBinding;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayout_ = descAlloc.createLayout(
        bindings.data(), static_cast<uint32_t>(bindings.size()), 0, nullptr,
        "WindRenderer: descriptorSetLayout");

    // The wind-field buffer handle is only known after VegetationRenderer::init,
    // which runs before this renderer; the binding is still (re)written on the
    // first dispatch through the change guard, so an early null handle is safe.
    const VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * VulkanApp::MAX_FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * VulkanApp::MAX_FRAMES_IN_FLIGHT},
    };
    descriptorPool_ = descAlloc.createPool(
        poolSizes, 2, VulkanApp::MAX_FRAMES_IN_FLIGHT,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, "WindRenderer: descriptorPool");

    descAlloc.allocateSets(descriptorPool_, descriptorSetLayout_, VulkanApp::MAX_FRAMES_IN_FLIGHT,
                           descriptorSets_.data(), "WindRenderer: descriptorSet");

    for (uint32_t i = 0; i < VulkanApp::MAX_FRAMES_IN_FLIGHT; ++i) {
        boundWindField_[i] = VK_NULL_HANDLE;
        DescriptorWriter(app->getDevice())
            .writeBuffer(descriptorSets_[i], kParamsBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         paramsBuffers_[i].buffer, 0, sizeof(WindParticlesUBO))
            .writeBuffer(descriptorSets_[i], kParticleBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         particleBuffer_.buffer, 0, app->getClampedWholeSizeRange(particleBuffer_.buffer))
            .writeBuffer(descriptorSets_[i], kTrailBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         trailBuffer_.buffer, 0, app->getClampedWholeSizeRange(trailBuffer_.buffer))
            .writeBuffer(descriptorSets_[i], kTrailSpeedBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         trailSpeedBuffer_.buffer, 0, app->getClampedWholeSizeRange(trailSpeedBuffer_.buffer))
            .flush();
    }
}

void WindRenderer::createComputePipeline(VulkanApp* app) {
    VkDevice device = app->getDevice();

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout_;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &computePipelineLayout_) != VK_SUCCESS) {
        std::cerr << "[WindRenderer] failed to create compute pipeline layout" << std::endl;
        computePipelineLayout_ = VK_NULL_HANDLE;
        return;
    }
    app->resources.addPipelineLayout(computePipelineLayout_, "WindRenderer: computePipelineLayout");

    VkShaderModule module = app->getOrCreateShaderModule("shaders/computation/wind/WindParticles.comp.spv");
    if (module == VK_NULL_HANDLE) {
        std::cerr << "[WindRenderer] missing shaders/computation/wind/WindParticles.comp.spv" << std::endl;
        return;
    }
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = computePipelineLayout_;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &pipelineInfo, nullptr,
                                 &computePipeline_) != VK_SUCCESS) {
        std::cerr << "[WindRenderer] failed to create compute pipeline" << std::endl;
        computePipeline_ = VK_NULL_HANDLE;
        return;
    }
    app->resources.addPipeline(computePipeline_, "WindRenderer: computePipeline");
}

void WindRenderer::createTrailPipeline(VulkanApp* app) {
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/renderer/wind/WindTrail.vert.spv");
    VkShaderModule fragModule = app->getOrCreateShaderModule("shaders/renderer/wind/WindTrail.frag.spv");
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        std::cerr << "[WindRenderer] missing WindTrail vertex/fragment shader modules" << std::endl;
        return;
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr},
    };

    // No vertex input: the trail quads are generated from gl_VertexIndex.
    // Straight-alpha blending over the composed frame; the trails are a
    // visualization overlay, so no depth test/write.
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
    trailPipeline_ = pipelineAndLayout.first;
    trailPipelineLayout_ = pipelineAndLayout.second;
    if (trailPipeline_ == VK_NULL_HANDLE) {
        std::cerr << "[WindRenderer] failed to create trail pipeline" << std::endl;
    }
}

void WindRenderer::barrierBeforeDispatch(VkCommandBuffer cmd) {
    // Three dependencies are needed before the particle step runs:
    //  a) the parameters UBO + the shared wind field stream from the host
    //     (persistently mapped, coherent) while this command buffer is
    //     recorded -> make those writes visible to the reading stages;
    //  b) the particle state/trail buffers persist ACROSS frames: the previous
    //     frame's compute wrote them (same queue, earlier submission) and this
    //     dispatch reads+writes them again;
    //  c) the previous frame's trail draw read them in the vertex shader
    //     (same queue, earlier submission) -> write-after-read.
    VkMemoryBarrier2 barriers[3]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

    barriers[1].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

    barriers[2].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barriers[2].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 3;
    dep.pMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void WindRenderer::barrierAfterDispatch(VkCommandBuffer cmd) {
    // The compute pass wrote the particle positions + trail samples; the trail
    // draw (recorded later in this same command buffer, inside the final
    // rendering scope) reads them as a storage buffer in the vertex shader.
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void WindRenderer::dispatch(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                            bool enabled, const WindParticlesSettings& settings,
                            const glm::mat4& viewProjection, const glm::mat4& invViewProjection,
                            const glm::vec3& cameraPosition, float cameraFarPlane,
                            const glm::vec2& screenSize, float timeSeconds, float deltaSeconds,
                            VkBuffer windFieldBuffer, bool windEnabled) {
    active_ = false;
    if (app == nullptr || cmd == VK_NULL_HANDLE || !enabled) return;
    if (computePipeline_ == VK_NULL_HANDLE || computePipelineLayout_ == VK_NULL_HANDLE) return;
    if (windFieldBuffer == VK_NULL_HANDLE) return;

    const uint32_t count = static_cast<uint32_t>(
        std::clamp(settings.particleCount, 0, static_cast<int>(kMaxParticles)));
    if (count == 0u) return;

    const uint32_t slot = frameIdx % VulkanApp::MAX_FRAMES_IN_FLIGHT;
    VkDescriptorSet descriptorSet = descriptorSets_[slot];
    if (descriptorSet == VK_NULL_HANDLE || paramsBuffers_[slot].buffer == VK_NULL_HANDLE) return;
    if (paramsBuffers_[slot].mappedData == nullptr) return;

    // Re-point the shared wind-field binding only when its handle changes.
    if (boundWindField_[slot] != windFieldBuffer) {
        DescriptorWriter(app->getDevice())
            .writeBuffer(descriptorSet, kWindFieldBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         windFieldBuffer, 0, sizeof(WindField))
            .flush();
        boundWindField_[slot] = windFieldBuffer;
    }

    // Particle band: the far plane is bounded by the camera far plane (the
    // simulated far cutoff AND the spawn distance upper bound), so the wind
    // particles always live inside the rendered view depth.
    const float farLimit = std::max(cameraFarPlane, 0.001f);
    const float nearDist = std::clamp(settings.spawnNearDistance, 0.0f, farLimit);
    const float farDist = std::clamp(settings.spawnFarDistance, nearDist, farLimit);

    WindParticlesUBO params{};
    params.viewProjection = viewProjection;
    params.invViewProjection = invViewProjection;
    params.timing = glm::vec4(std::max(timeSeconds, 0.0f),
                              std::clamp(deltaSeconds, 0.0f, kMaxFrameStep),
                              static_cast<float>(count),
                              static_cast<float>(++seedCounter_));
    params.trail = glm::vec4(std::clamp(settings.trailDuration, 0.05f, 60.0f),
                             kTrailSampleInterval,
                             static_cast<float>(kTrailCapacity),
                             std::max(settings.lineWidth, 0.1f));
    // Speed -> color ramp (blue -> green -> yellow -> red) plus peak alpha.
    params.style = glm::vec4(std::clamp(settings.colorSpeedMax, 0.01f, 500.0f),
                             std::clamp(settings.opacity, 0.0f, 1.0f), 0.0f, 0.0f);    params.spawn = glm::vec4(nearDist, farDist,
                             std::max(screenSize.x, 1.0f),
                             std::max(screenSize.y, 1.0f));
    params.cameraPosition = glm::vec4(cameraPosition, 0.0f);
    params.extra = glm::vec4(kFrustumMargin, kSpawnFadeSeconds, windEnabled ? 1.0f : 0.0f, 0.0f);
    std::memcpy(paramsBuffers_[slot].mappedData, &params, sizeof(params));

    barrierBeforeDispatch(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout_,
                            0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, (count + kWorkgroupSize - 1u) / kWorkgroupSize, 1, 1);

    barrierAfterDispatch(cmd);
    active_ = true;
}

void WindRenderer::render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                          bool enabled, const WindParticlesSettings& settings) {
    if (app == nullptr || cmd == VK_NULL_HANDLE || !enabled || !active_) return;
    if (trailPipeline_ == VK_NULL_HANDLE || trailPipelineLayout_ == VK_NULL_HANDLE) return;

    const uint32_t count = static_cast<uint32_t>(
        std::clamp(settings.particleCount, 0, static_cast<int>(kMaxParticles)));
    const uint32_t segments = trailSegments(settings.trailDuration);
    const uint64_t vertexCount = static_cast<uint64_t>(count) * segments * 6u;
    if (vertexCount == 0u) return;

    const uint32_t slot = frameIdx % VulkanApp::MAX_FRAMES_IN_FLIGHT;
    if (descriptorSets_[slot] == VK_NULL_HANDLE) return;

    const float width = static_cast<float>(std::max(app->getWidth(), 1));
    const float height = static_cast<float>(std::max(app->getHeight(), 1));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trailPipeline_);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trailPipelineLayout_,
                            0, 1, &descriptorSets_[slot], 0, nullptr);
    vkCmdDraw(cmd, static_cast<uint32_t>(vertexCount), 1, 0, 0);
}

void WindRenderer::cleanup(VulkanApp* app) {
    if (app == nullptr) return;

    for (Buffer& b : paramsBuffers_) {
        if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
        b = {};
    }
    if (particleBuffer_.buffer != VK_NULL_HANDLE) app->destroyBuffer(particleBuffer_);
    if (trailBuffer_.buffer != VK_NULL_HANDLE) app->destroyBuffer(trailBuffer_);
    if (trailSpeedBuffer_.buffer != VK_NULL_HANDLE) app->destroyBuffer(trailSpeedBuffer_);
    particleBuffer_ = {};
    trailBuffer_ = {};
    trailSpeedBuffer_ = {};

    // Pipeline / layout / pool / set objects are owned by VulkanResourceManager
    // and released by its teardown; only the local handles are dropped here.
    descriptorSetLayout_ = VK_NULL_HANDLE;
    descriptorPool_ = VK_NULL_HANDLE;
    computePipelineLayout_ = VK_NULL_HANDLE;
    computePipeline_ = VK_NULL_HANDLE;
    trailPipelineLayout_ = VK_NULL_HANDLE;
    trailPipeline_ = VK_NULL_HANDLE;
    descriptorSets_.fill(VK_NULL_HANDLE);
    boundWindField_.fill(VK_NULL_HANDLE);
    active_ = false;
    app_ = nullptr;
}
