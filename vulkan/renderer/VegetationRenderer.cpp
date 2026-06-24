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
#include "../includes/locations.hpp"
#include "../includes/vertex_layouts.hpp"

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
    vegetationShadowPipeline = VK_NULL_HANDLE;
    shadowPipelineLayout = VK_NULL_HANDLE;
    descriptorSetLayout = VK_NULL_HANDLE;
    // Clear impostor resources (also tracked by central manager).
    impostorPipeline       = VK_NULL_HANDLE;
    impostorPipelineLayout = VK_NULL_HANDLE;
    impostorDescSetLayout  = VK_NULL_HANDLE;
    impostorDescPool       = VK_NULL_HANDLE;
    impostorDescSet        = VK_NULL_HANDLE;
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
        // Defer destruction of the old descriptor set. The current frame's
        // command buffer may still reference it.  deferDestroyUntilAllPending
        // waits for all in-flight rendering to complete before freeing.
        VkDescriptorSet ds = vegDescriptorSet;
        VkDevice dev = app->getDevice();
        VkDescriptorPool pool = app->getDescriptorPool();
        app->deferDestroyUntilAllPending([dev, pool, ds, app]() {
            if (app->resources.removeDescriptorSet(ds))
                vkFreeDescriptorSets(dev, pool, 1, &ds);
        });
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


void VegetationRenderer::init(VulkanApp* app) {
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
        std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{ ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription{ ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription{ ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) },
            VkVertexInputAttributeDescription{ ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 },
        },
        setLayouts,
        &pushConstantRange,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true, // depthTest
        true, // depthWrite
        VK_COMPARE_OP_LESS,
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );
    vegetationPipeline = pipeline;
    pipelineLayout = layout;
    if (vegetationPipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION PIPELINE ERROR] Failed to create vegetation pipeline or layout!" << std::endl;
    } else {
        std::cerr << "[VEGETATION PIPELINE] Created pipeline=" << (void*)vegetationPipeline << " layout=" << (void*)pipelineLayout << std::endl;
    }

    auto [shadowPipeline, shadowLayout] = app->createGraphicsPipeline(
        { stages[0], stages[1], stages[2] },
        std::vector<VkVertexInputBindingDescription>{bindingDescs[0], bindingDescs[1]},
        std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{ ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription{ ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
            VkVertexInputAttributeDescription{ ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) },
            VkVertexInputAttributeDescription{ ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 },
        },
        setLayouts,
        &pushConstantRange,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true,
        true,
        VK_COMPARE_OP_LESS,
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        true,   // noColorAttachment: depth-only shadow pass
        true    // depthBiasEnable: push shadow depths away from light
    );
    vegetationShadowPipeline = shadowPipeline;    shadowPipelineLayout = shadowLayout;
    if (vegetationShadowPipeline == VK_NULL_HANDLE || shadowPipelineLayout == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION SHADOW PIPELINE ERROR] Failed to create vegetation shadow pipeline/layout" << std::endl;
    } else {
        std::cerr << "[VEGETATION SHADOW PIPELINE] Created pipeline=" << (void*)vegetationShadowPipeline << " layout=" << (void*)shadowPipelineLayout << std::endl;
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
        baseVertex.brushIndex = 0;
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
    // Clear any pending CPU-generation chunks to prevent stale data
    // from a previous scene from being processed after scene reset.
    {
        std::lock_guard<std::mutex> lk(pendingChunksMutex);
        pendingChunks.clear();
    }
}

size_t VegetationRenderer::getInstanceTotal() const {
    size_t total = 0;
    for (const auto& kv : chunkInstanceCounts) {
        total += kv.second;
    }
    return total;
}

float VegetationRenderer::computeDensityFactor(float distanceToCamera) const {
    if (!distanceDensitySettings.enabled) {
        return 1.0f;
    }

    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    if (distanceToCamera <= nearDistance || minFactor >= 1.0f) {
        return 1.0f;
    }

    const float decayRange = farDistance - nearDistance;
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = -std::log(safeMinFactor) / decayRange;
    const float densityFactor = std::exp(-falloff * (distanceToCamera - nearDistance));
    return std::clamp(densityFactor, minFactor, 1.0f);
}

std::vector<DebugCubeRenderer::CubeWithColor> VegetationRenderer::getDensityDebugCubes(const glm::vec3& cameraPos) const {
    std::vector<DebugCubeRenderer::CubeWithColor> cubes;
    cubes.reserve(chunkBuffers.size());
    const float halfExtent = std::max(8.0f, billboardScale * 0.35f);

    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) {
            continue;
        }

        const float densityFactor = computeDensityFactor(glm::distance(buf.center, cameraPos));
        const glm::vec3 color = glm::mix(glm::vec3(1.0f, 0.15f, 0.15f), glm::vec3(0.15f, 1.0f, 0.2f), densityFactor);
        const glm::vec3 minPoint = buf.center - glm::vec3(halfExtent);
        const glm::vec3 maxPoint = buf.center + glm::vec3(halfExtent);
        cubes.push_back({BoundingBox(minPoint, maxPoint), color});
    }

    return cubes;
}

float VegetationRenderer::getAverageDensityFactor(const glm::vec3& cameraPos) const {
    if (chunkBuffers.empty()) {
        return 1.0f;
    }

    float factorSum = 0.0f;
    size_t factorCount = 0;
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.count == 0) {
            continue;
        }
        factorSum += computeDensityFactor(glm::distance(buf.center, cameraPos));
        ++factorCount;
    }

    return factorCount > 0 ? factorSum / static_cast<float>(factorCount) : 1.0f;
}

void VegetationRenderer::recordReadBarriers(VkCommandBuffer& commandBuffer) {
    if (commandBuffer == VK_NULL_HANDLE) return;

    std::vector<VkBufferMemoryBarrier> readBarriers;
    readBarriers.reserve(chunkBuffers.size() * 2);
    for (const auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;

        // Instance buffers are filled by the CPU (processPendingChunks writes
        // via mapped HOST_VISIBLE memory).  Without this barrier the GPU may
        // read uninitialized billboardIndex values, producing out-of-bounds
        // texture-array accesses that cause RADV GPUVM faults (TCP read).
        VkBufferMemoryBarrier instanceBarrier{};
        instanceBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        instanceBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        instanceBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        instanceBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        instanceBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        instanceBarrier.buffer = buf.buffer;
        instanceBarrier.offset = 0;
        instanceBarrier.size = VK_WHOLE_SIZE;
        readBarriers.push_back(instanceBarrier);

        VkBufferMemoryBarrier indirectBarrier{};
        indirectBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        indirectBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        indirectBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        indirectBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.buffer = buf.indirectBuffer;
        indirectBarrier.offset = 0;
        indirectBarrier.size = VK_WHOLE_SIZE;
        readBarriers.push_back(indirectBarrier);
    }
    if (readBarriers.empty()) return;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0, 0, nullptr,
        static_cast<uint32_t>(readBarriers.size()), readBarriers.data(),
        0, nullptr);
}


void VegetationRenderer::drawShadow(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet shadowDescriptorSet, const glm::vec3& cameraPos) {
    if (!app || vegetationShadowPipeline == VK_NULL_HANDLE) {
        if (!app) std::cerr << "[VEGETATION SHADOW DRAW ERROR] app is null!" << std::endl;
        if (vegetationShadowPipeline == VK_NULL_HANDLE) std::cerr << "[VEGETATION SHADOW DRAW ERROR] Shadow pipeline is VK_NULL_HANDLE!" << std::endl;
        return;
    }

    // Early-out when there are no chunks to draw: skip pipeline/descriptor
    // binding entirely. On RADV iGPUs, binding pipelines that reference
    // large texture arrays (via vegDescriptorSet) can trigger GPUVM faults
    // even when zero draw calls are issued.
    if (chunkBuffers.empty()) return;

    // Ensure vegetation descriptor set is present and up-to-date
    if (!ensureVegDescriptorSet(app)) {
        std::cerr << "[VEGETATION SHADOW DRAW ERROR] vegDescriptorSet not ready, skipping draw." << std::endl;
        return;
    }

    // Defensive checks: ensure both descriptor sets are valid before binding
    if (shadowDescriptorSet == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION SHADOW DRAW ERROR] shadowDescriptorSet is VK_NULL_HANDLE, skipping draw." << std::endl;
        return;
    }
    if (vegDescriptorSet == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION SHADOW DRAW ERROR] vegDescriptorSet is VK_NULL_HANDLE, skipping draw." << std::endl;
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationShadowPipeline);

    // Bind the shadow descriptor set at set 0 and vegetation descriptor set at set 1
    // shadowDescriptorSet contains the light-space UBO for shadow rendering
    VkDescriptorSet sets[2] = { shadowDescriptorSet, vegDescriptorSet };
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, 2, sets, 0, nullptr);

    // Push constants for shadow pass: same as regular draw but with wind disabled
    WindPushConstants pc{};
    pc.billboardScale = billboardScale;
    pc.windEnabled = -1.0f;  // Negative means shadow pass: disable wind and tighten impostor cutoff.
    pc.windTime = windTimeSeconds;
    pc.impostorDistance = impostorDistance; // skip far instances in shadow pass too (same as main pass)

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
    
    // Distance-based density: use camera position for LOD, NOT light position
    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = (distanceDensitySettings.enabled && minFactor < 1.0f)
        ? (-std::log(safeMinFactor) / (farDistance - nearDistance))
        : 0.0f;
    pc.densityParams = glm::vec4(distanceDensitySettings.enabled ? 1.0f : 0.0f, nearDistance, farDistance, minFactor);
    pc.cameraPosAndFalloff = glm::vec4(cameraPos, falloff);  // Camera pos, not light pos

    vkCmdPushConstants(
        commandBuffer,
        shadowPipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(WindPushConstants),
        &pc
    );

    // For each chunk, bind base vertex buffer + instance buffer and draw indirect.
    // Distance-based thinning is handled in the vegetation shader using camera position.
    static int shadowDrawLogCounter = 0;
    int chunksDrawn = 0;
    for (auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;
        if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) continue;

        VkBuffer vbs[2] = { billboardVBO.vertexBuffer.buffer, buf.buffer };
        VkDeviceSize offsets[2] = { 0, 0 };
        // Bind base vertex buffer at binding 0 and instance buffer at binding 1
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vbs, offsets);
        vkCmdDrawIndirect(commandBuffer, buf.indirectBuffer, 0, 1, sizeof(VkDrawIndirectCommand));
        chunksDrawn++;
    }
    if (shadowDrawLogCounter < 5) {
        std::cerr << "[VEGETATION SHADOW DRAW] chunksDrawn=" << chunksDrawn << " totalChunks=" << chunkBuffers.size() << std::endl;
        shadowDrawLogCounter++;
    }
}

void VegetationRenderer::setImpostorData(VulkanApp* app, VkImageView albedoArray60, VkImageView normalArray60, VkSampler sampler) {
    if (!app || albedoArray60 == VK_NULL_HANDLE || normalArray60 == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
    // Wait for all in-flight work to complete before recreating descriptor sets.
    // This prevents handle-reuse collisions between pending command buffers and
    // freshly-allocated descriptor set handles. setImpostorData is an init-time call.
    vkDeviceWaitIdle(app->getDevice());

    VkDevice device = app->getDevice();

    // Destroy any previous impostor resources (handles are tracked in the central manager).
    impostorPipeline       = VK_NULL_HANDLE;
    impostorPipelineLayout = VK_NULL_HANDLE;
    impostorDescSetLayout  = VK_NULL_HANDLE;
    impostorDescPool       = VK_NULL_HANDLE;
    impostorDescSet        = VK_NULL_HANDLE;

    // Descriptor set layout: set 1 — binding 0=albedo array, binding 1=normal array.
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 2;
        info.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &impostorDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDescSetLayout failed");
        app->resources.addDescriptorSetLayout(impostorDescSetLayout, "VegetationRenderer: impostorDescSetLayout");
    }

    // Private descriptor pool (2 combined image samplers).
    {
        VkDescriptorPoolSize sz{};
        sz.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sz.descriptorCount = 2;
        VkDescriptorPoolCreateInfo info{};
        info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets       = 1;
        info.poolSizeCount = 1;
        info.pPoolSizes    = &sz;
        // Support update-after-bind allocations if needed by set layouts
        info.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        if (vkCreateDescriptorPool(device, &info, nullptr, &impostorDescPool) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDescPool failed");
        app->resources.addDescriptorPool(impostorDescPool, "VegetationRenderer: impostorDescPool");
    }

    // Allocate and write descriptor set (albedo + normal array).
    {
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool     = impostorDescPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts        = &impostorDescSetLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &impostorDescSet) != VK_SUCCESS)
            throw std::runtime_error("VegetationRenderer: impostorDescSet alloc failed");

        VkDescriptorImageInfo imgInfos[2]{};
        imgInfos[0] = { sampler, albedoArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        imgInfos[1] = { sampler, normalArray60, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet ws[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet          = impostorDescSet;
            ws[i].dstBinding      = i;
            ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ws[i].descriptorCount = 1;
            ws[i].pImageInfo      = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, 2, ws, 0, nullptr);
    }

    // Build impostor pipeline (same vertex input as vegetation, different shaders).
    auto vertCode = FileReader::readFile("shaders/impostors.vert.spv");
    auto geomCode = FileReader::readFile("shaders/impostors.geom.spv");
    auto fragCode = FileReader::readFile("shaders/impostors.frag.spv");

    VkShaderModule vertShader = app->createShaderModule(vertCode);
    VkShaderModule geomShader = app->createShaderModule(geomCode);
    VkShaderModule fragShader = app->createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo geomStage{};
    geomStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomStage.stage  = VK_SHADER_STAGE_GEOMETRY_BIT;
    geomStage.module = geomShader;
    geomStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName  = "main";

    VkVertexInputBindingDescription bindingDescs[2]{};
    bindingDescs[0] = { 0, sizeof(Vertex),       VK_VERTEX_INPUT_RATE_VERTEX   };
    bindingDescs[1] = { 1, sizeof(float) * 4,    VK_VERTEX_INPUT_RATE_INSTANCE };

    std::vector<VkDescriptorSetLayout> impSetLayouts = {
        app->getDescriptorSetLayout(),
        impostorDescSetLayout
    };
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
                       | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(WindPushConstants);

    auto [impPipeline, impLayout] = app->createGraphicsPipeline(
        { vertStage, geomStage, fragStage },
        std::vector<VkVertexInputBindingDescription>{ bindingDescs[0], bindingDescs[1] },
        {
            { ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT,    (uint32_t)offsetof(Vertex, position) },
            { ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT,    (uint32_t)offsetof(Vertex, normal)   },
            { ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
            { ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT,            (uint32_t)offsetof(Vertex, brushIndex) },
            { ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                                    },
        },
        impSetLayouts,
        &pcRange,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        false, // depthWrite — impostors drawn after opaque, don't write depth
        true,  // colorWrite — impostors must output their color
        VK_COMPARE_OP_LESS,
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        false,
        {},
        VK_FORMAT_D32_SFLOAT,
        false
    );

    impostorPipeline       = impPipeline;
    impostorPipelineLayout = impLayout;

    if (impostorPipeline == VK_NULL_HANDLE)
        std::cerr << "[VegetationRenderer] WARNING: impostor pipeline creation failed\n";
    else
        std::cerr << "[VegetationRenderer] Impostor pipeline created: " << (void*)impostorPipeline << "\n";
}

void VegetationRenderer::draw(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet, const glm::mat4& viewProj, const glm::vec3& cameraPos) {
    if (!app || vegetationPipeline == VK_NULL_HANDLE) {
        if (!app) std::cerr << "[VEGETATION DRAW ERROR] app is null!" << std::endl;
        if (vegetationPipeline == VK_NULL_HANDLE) std::cerr << "[VEGETATION DRAW ERROR] Attempted to bind VK_NULL_HANDLE pipeline!" << std::endl;
        return;
    }

    // Early-out when there are no chunks to draw.  Pending chunks are
    // drained in MyApp::update() so chunkBuffers is already populated
    // when preRenderPass records read barriers before beginPass.
    if (chunkBuffers.empty()) return;

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
    if (vegDescriptorSet == VK_NULL_HANDLE) {
        std::cerr << "[VEGETATION DRAW ERROR] vegDescriptorSet is VK_NULL_HANDLE, skipping draw." << std::endl;
        return;
    }
    VkDescriptorSet sets[2] = { globalSet, vegDescriptorSet };
    //printf("[BIND] VegetationRenderer::draw: layout=%p firstSet=0 count=2 sets=%p %p\n", (void*)pipelineLayout, (void*)sets[0], (void*)sets[1]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    WindPushConstants pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = windSettings.enabled ? 1.0f : 0.0f;
    pc.windTime           = windTimeSeconds;
    pc.impostorDistance   = impostorDistance; // 0 = impostor disabled (near geometry only)

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
    const float nearDistance = std::max(0.0f, distanceDensitySettings.fullDensityDistance);
    const float farDistance = std::max(nearDistance + 1.0f, distanceDensitySettings.minDensityDistance);
    const float minFactor = std::clamp(distanceDensitySettings.minDensityFactor, 0.0f, 1.0f);
    const float safeMinFactor = std::max(minFactor, 0.0001f);
    const float falloff = (distanceDensitySettings.enabled && minFactor < 1.0f)
        ? (-std::log(safeMinFactor) / (farDistance - nearDistance))
        : 0.0f;
    pc.densityParams = glm::vec4(distanceDensitySettings.enabled ? 1.0f : 0.0f, nearDistance, farDistance, minFactor);
    pc.cameraPosAndFalloff = glm::vec4(cameraPos, falloff);

    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(WindPushConstants),
        &pc
    );

    // For each chunk, bind base vertex buffer + instance buffer and draw indirect.
    // Distance-based thinning is handled in the vegetation shader.
    for (auto& [chunkId, buf] : chunkBuffers) {
        (void)chunkId;
        if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;
        if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) continue;

        VkBuffer vbs[2] = { billboardVBO.vertexBuffer.buffer, buf.buffer };
        VkDeviceSize offsets[2] = { 0, 0 };
        // Bind base vertex buffer at binding 0 and instance buffer at binding 1
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vbs, offsets);
        vkCmdDrawIndirect(commandBuffer, buf.indirectBuffer, 0, 1, sizeof(VkDrawIndirectCommand));
    }

    // ── Impostor pass ────────────────────────────────────────────────────────
    // Draw camera-facing impostor quads for instances beyond impostorDistance.
    // The impostor geom shader skips instances that are too close
    // (they were already handled by the vegetation pass above).
    if (impostorPipeline != VK_NULL_HANDLE &&
        impostorDescSet  != VK_NULL_HANDLE &&
        impostorDistance > 0.0f) {

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, impostorPipeline);

        VkDescriptorSet impSets[2] = { globalSet, impostorDescSet };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    impostorPipelineLayout, 0, 2, impSets, 0, nullptr);

        // Re-push the same push constants using the impostor pipeline layout.
        vkCmdPushConstants(commandBuffer, impostorPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
                           | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(WindPushConstants), &pc);

        for (auto& [chunkId, buf] : chunkBuffers) {
            (void)chunkId;
            if (buf.buffer == VK_NULL_HANDLE || buf.indirectBuffer == VK_NULL_HANDLE || buf.count == 0) continue;
            if (billboardVBO.vertexBuffer.buffer == VK_NULL_HANDLE) continue;

            VkBuffer vbs[2] = { billboardVBO.vertexBuffer.buffer, buf.buffer };
            VkDeviceSize offsets[2] = { 0, 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vbs, offsets);
            vkCmdDrawIndirect(commandBuffer, buf.indirectBuffer, 0, 1, sizeof(VkDrawIndirectCommand));
        }
    }
}

void VegetationRenderer::generateChunkInstances(NodeID chunkId,
                                               Buffer vertexBuffer, uint32_t vertexCount,
                                               Buffer indexBuffer, uint32_t indexCount,
                                               const glm::vec3& chunkCenter,
                                               uint32_t instancesPerTriangle, VulkanApp* app,
                                               uint32_t seed) {
    if (!app) return;

    if (indexCount < 3 || instancesPerTriangle == 0) {
        // No instances to generate; clear any previous chunk data.
        destroyInstanceBuffer(chunkId, app);
        return;
    }

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
    {
        static constexpr VkDeviceSize kMin = 262144;
        const VkDeviceSize sz = memReq.size;
        allocInfo.allocationSize = (sz < kMin) ? kMin : (sz < 1048576 ? sz + 1 : sz);
    }
    allocInfo.memoryTypeIndex = deviceLocalTypeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &instanceMemory) != VK_SUCCESS) throw std::runtime_error("Failed to allocate instance buffer memory");
    vkBindBufferMemory(device, instanceBuffer, instanceMemory, 0);
    app->resources.addDeviceMemory(instanceMemory, "VegetationRenderer: instanceMemory");

    Buffer indirect = app->createBuffer(sizeof(VkDrawIndirectCommand),
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkBuffer indirectBuffer = indirect.buffer;
    VkDeviceMemory indirectMemory = indirect.memory;

    VkDrawIndirectCommand drawCmd{};
    drawCmd.vertexCount = 1;
    drawCmd.instanceCount = instanceCount;
    drawCmd.firstVertex = 0;
    drawCmd.firstInstance = 0;
    void* indirectData;
    if (vkMapMemory(device, indirectMemory, 0, sizeof(VkDrawIndirectCommand), 0, &indirectData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map vegetation indirect buffer");
    }
    std::memcpy(indirectData, &drawCmd, sizeof(VkDrawIndirectCommand));
    vkUnmapMemory(device, indirectMemory);

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

    // Now that we have a valid fence, destroy any previous chunk buffers.
    // The fence signals after the previous frame's draw (same queue, earlier
    // submission), guaranteeing the old buffers are no longer in use.
    destroyInstanceBuffer(chunkId, app, fence);

    std::cout << "[VegetationRenderer::generateChunkInstances] async dispatched, expected = " << expected << " fence=" << (void*)fence << std::endl;

    // Wait for the compute dispatch to complete before returning.
    // RADV GPUVM faults (TCP read permission) occur when too many
    // concurrent compute dispatches are in flight, even with correct
    // synchronization.  Serializing eliminates the concurrency.
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Insert the instance buffer into visible maps immediately (fence
    // already signaled).
    InstanceBuffer ibuf;
    ibuf.buffer = instanceBuffer;
    ibuf.memory = instanceMemory;
    ibuf.indirectBuffer = indirectBuffer;
    ibuf.indirectMemory = indirectMemory;
    ibuf.center = chunkCenter;
    ibuf.count = expected;
    chunkBuffers[chunkId] = ibuf;
    chunkInstanceCounts[chunkId] = expected;
    std::cout << "[VegetationRenderer] chunk " << (unsigned long long)chunkId << " instances ready: " << expected << std::endl;

    // Transfer input buffers (vertex/index) — the fence is already signaled
    // (vkWaitForFences above), so destroy them immediately.
    VkDevice dev = device;
    if (vertexBuffer.buffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(vertexBuffer.buffer)) vkDestroyBuffer(dev, vertexBuffer.buffer, nullptr);
    }
    if (vertexBuffer.memory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(vertexBuffer.memory)) vkFreeMemory(dev, vertexBuffer.memory, nullptr);
    }
    if (indexBuffer.buffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(indexBuffer.buffer)) vkDestroyBuffer(dev, indexBuffer.buffer, nullptr);
    }
    if (indexBuffer.memory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(indexBuffer.memory)) vkFreeMemory(dev, indexBuffer.memory, nullptr);
    }
}

// ── CPU-side instance generation ─────────────────────────────────────────────
// Replicates vegetation_instance_gen.comp logic on the CPU.  Avoids GPUVM
// faults on RADV iGPUs where TCP cannot read storage buffers from any
// memory type (device-local, host-visible, or concurrent-shared).

namespace {

// XorShift32 PRNG — matches compute shader behaviour.
uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(uint32_t& state) {
    return float(xorshift32(state) & 0x00FFFFFFu) / float(0x01000000u);
}

// Position hash — matches posHash in the compute shader.
uint32_t posHash(const glm::vec3& p) {
    glm::ivec3 qi = glm::ivec3(glm::round(p * 8.0f));
    uint32_t h = uint32_t(qi.x) * 1640531513u;
    h ^= uint32_t(qi.y) * 2246822519u;
    h ^= uint32_t(qi.z) * 3266489917u;
    return h;
}

// 2D cell hash — matches cellHash in the compute shader.
uint32_t cellHash(glm::ivec2 c) {
    uint32_t h = uint32_t(c.x) * 1640531513u ^ uint32_t(c.y) * 2246822519u;
    h ^= h >> 13;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

// Biome noise — matches biomeNoise in the compute shader.
float biomeNoise(const glm::vec2& xz) {
    const float kBiomeScale = 50.0f;
    glm::vec2 p = xz / kBiomeScale;
    glm::ivec2 i = glm::ivec2(glm::floor(p));
    glm::vec2 f = p - glm::vec2(i);
    glm::vec2 u = f * f * (3.0f - 2.0f * f);

    float a = float(cellHash(i + glm::ivec2(0, 0))) / 4294967295.0f;
    float b = float(cellHash(i + glm::ivec2(1, 0))) / 4294967295.0f;
    float c = float(cellHash(i + glm::ivec2(0, 1))) / 4294967295.0f;
    float d = float(cellHash(i + glm::ivec2(1, 1))) / 4294967295.0f;

    return glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y);
}

} // anonymous namespace

void VegetationRenderer::generateChunkInstancesCPU(NodeID chunkId,
                                                   const std::vector<glm::vec3>& positions,
                                                   const std::vector<uint32_t>& grassIndices,
                                                   const glm::vec3& chunkCenter,
                                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                                   uint32_t seed) {
    (void)app; // used later in processPendingChunks
    if (grassIndices.size() < 3 || instancesPerTriangle == 0 || positions.empty()) {
        destroyInstanceBuffer(chunkId, app);
        return;
    }
    // Enqueue for later processing — the render thread drains this queue.
    PendingChunk pc;
    pc.chunkId             = chunkId;
    pc.positions           = positions;
    pc.grassIndices        = grassIndices;
    pc.chunkCenter         = chunkCenter;
    pc.instancesPerTriangle = instancesPerTriangle;
    pc.seed                = seed;
    {
        std::lock_guard<std::mutex> lk(pendingChunksMutex);
        pendingChunks.push_back(std::move(pc));
    }
}

size_t VegetationRenderer::pendingChunkCount() const {
    std::lock_guard<std::mutex> lk(pendingChunksMutex);
    return pendingChunks.size();
}

void VegetationRenderer::processPendingChunks(uint32_t maxChunks) {
    if (!appPtr) return;
    VulkanApp* app = appPtr;
    const uint32_t billboardCnt = (billboardCount > 0) ? billboardCount : 1u;

    for (uint32_t n = 0; n < maxChunks; ++n) {
        PendingChunk pc;
        {
            std::lock_guard<std::mutex> lk(pendingChunksMutex);
            if (pendingChunks.empty()) break;
            pc = std::move(pendingChunks.front());
            pendingChunks.pop_front();
        }

        const uint32_t triCount = static_cast<uint32_t>(pc.grassIndices.size()) / 3;
        const uint32_t instanceCount = triCount * pc.instancesPerTriangle;

        std::vector<float> instanceData(instanceCount * 4, 0.0f);

        for (uint32_t tri = 0; tri < triCount; ++tri) {
            const uint32_t tb = tri * 3;
            const uint32_t i0 = pc.grassIndices[tb + 0];
            const uint32_t i1 = pc.grassIndices[tb + 1];
            const uint32_t i2 = pc.grassIndices[tb + 2];
            if (i0 >= pc.positions.size() || i1 >= pc.positions.size() || i2 >= pc.positions.size()) continue;

            const glm::vec3 v0 = pc.positions[i0];
            const glm::vec3 v1 = pc.positions[i1];
            const glm::vec3 v2 = pc.positions[i2];

            const glm::vec3 fn = glm::cross(v1 - v0, v2 - v0);
            if (glm::abs(fn.y) <= 0.5f * glm::length(fn)) {
                for (uint32_t s = 0; s < pc.instancesPerTriangle; ++s) {
                    const uint32_t oi = (tri * pc.instancesPerTriangle + s) * 4;
                    instanceData[oi + 3] = -1.0f;
                }
                continue;
            }

            const glm::vec3 tc = (v0 + v1 + v2) / 3.0f;
            const uint32_t tch = posHash(tc);

            for (uint32_t s = 0; s < pc.instancesPerTriangle; ++s) {
                uint32_t rng = pc.seed ^ tch ^ (tri * 2654435761u) ^ (s * 19349663u);
                float u = randFloat(rng);
                float v = randFloat(rng);
                if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                float w = 1.0f - u - v;
                glm::vec3 pos = u * v0 + v * v1 + w * v2;

                uint32_t bi = std::min(
                    uint32_t(biomeNoise(glm::vec2(pos.x, pos.z)) * float(billboardCnt)),
                    billboardCnt - 1u);

                uint32_t rs = pc.seed ^ posHash(pos);
                float rf = randFloat(rs);

                const uint32_t oi = (tri * pc.instancesPerTriangle + s) * 4;
                instanceData[oi + 0] = pos.x;
                instanceData[oi + 1] = pos.y;
                instanceData[oi + 2] = pos.z;
                instanceData[oi + 3] = float(bi) + rf;
            }
        }

        const VkDeviceSize bufSize = instanceData.size() * sizeof(float);

        // Staging buffer: host-visible, filled by CPU.
        Buffer stagingInst = app->createBuffer(bufSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        vkMapMemory(app->getDevice(), stagingInst.memory, 0, bufSize, 0, &mapped);
        std::memcpy(mapped, instanceData.data(), size_t(bufSize));
        vkUnmapMemory(app->getDevice(), stagingInst.memory);

        // Device-local instance buffer: GPU reads via vertex input.
        // On RADV iGPUs, vertex reads go through TCP (Texture Cache/Pipe),
        // and host-visible pages lack TCP-read permission → GPUVM fault.
        Buffer instBuf = app->createBuffer(bufSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Indirect buffer: device-local, avoids same TCP-read issue.
        Buffer indirect = app->createBuffer(sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Staging for indirect draw command.
        Buffer stagingIndirect = app->createBuffer(sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDrawIndirectCommand drawCmd{};
        drawCmd.vertexCount   = 1;
        drawCmd.instanceCount = instanceCount;
        drawCmd.firstVertex   = 0;
        drawCmd.firstInstance = 0;
        void* idata = nullptr;
        vkMapMemory(app->getDevice(), stagingIndirect.memory, 0, sizeof(VkDrawIndirectCommand), 0, &idata);
        std::memcpy(idata, &drawCmd, sizeof(VkDrawIndirectCommand));
        vkUnmapMemory(app->getDevice(), stagingIndirect.memory);

        // Copy staging → device-local on the graphics queue, synchronous.
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.size = bufSize;
            vkCmdCopyBuffer(cmd, stagingInst.buffer, instBuf.buffer, 1, &copyRegion);

            VkBufferCopy indirectCopy{};
            indirectCopy.size = sizeof(VkDrawIndirectCommand);
            vkCmdCopyBuffer(cmd, stagingIndirect.buffer, indirect.buffer, 1, &indirectCopy);
        });

        // Destroy staging buffers immediately (copy completed synchronously).
        VkDevice dev = app->getDevice();
        if (app->resources.removeBuffer(stagingInst.buffer))
            vkDestroyBuffer(dev, stagingInst.buffer, nullptr);
        if (app->resources.removeDeviceMemory(stagingInst.memory))
            vkFreeMemory(dev, stagingInst.memory, nullptr);
        if (app->resources.removeBuffer(stagingIndirect.buffer))
            vkDestroyBuffer(dev, stagingIndirect.buffer, nullptr);
        if (app->resources.removeDeviceMemory(stagingIndirect.memory))
            vkFreeMemory(dev, stagingIndirect.memory, nullptr);

        destroyInstanceBuffer(pc.chunkId, app);

        InstanceBuffer ibuf;
        ibuf.buffer         = instBuf.buffer;
        ibuf.memory         = instBuf.memory;
        ibuf.indirectBuffer = indirect.buffer;
        ibuf.indirectMemory = indirect.memory;
        ibuf.center         = pc.chunkCenter;
        ibuf.count          = instanceCount;
        chunkBuffers[pc.chunkId] = ibuf;
        chunkInstanceCounts[pc.chunkId] = instanceCount;
    }
}

void VegetationRenderer::destroyInstanceBuffer(NodeID chunkId, VulkanApp* app, VkFence completionFence) {
    auto it = chunkBuffers.find(chunkId);
    if (it == chunkBuffers.end()) return;

    // Snatch the old Vulkan handles before clearing the map entry.
    InstanceBuffer old = it->second;
    it->second.buffer = VK_NULL_HANDLE;
    it->second.memory = VK_NULL_HANDLE;
    it->second.indirectBuffer = VK_NULL_HANDLE;
    it->second.indirectMemory = VK_NULL_HANDLE;
    chunkBuffers.erase(it);
    chunkInstanceCounts.erase(chunkId);

    if (!app) return;

    // Destroy old buffers immediately.  The CPU-generation path runs on the
    // render thread via processPendingChunks() which is called from draw()
    // before command buffer submission, so no in-flight work references them.
    // The GPU path (generateChunkInstances) waits synchronously on the fence
    // before returning, so the GPU is done by the time we reach here.
    VkDevice dev = app->getDevice();
    if (old.buffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(old.buffer)) vkDestroyBuffer(dev, old.buffer, nullptr);
    }
    if (old.memory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(old.memory)) vkFreeMemory(dev, old.memory, nullptr);
    }
    if (old.indirectBuffer != VK_NULL_HANDLE) {
        if (app->resources.removeBuffer(old.indirectBuffer)) vkDestroyBuffer(dev, old.indirectBuffer, nullptr);
    }
    if (old.indirectMemory != VK_NULL_HANDLE) {
        if (app->resources.removeDeviceMemory(old.indirectMemory)) vkFreeMemory(dev, old.indirectMemory, nullptr);
    }
}

// Ensure we clear the stored app pointer on cleanup
// (cleanup() already clears handles; set appPtr to nullptr here)
