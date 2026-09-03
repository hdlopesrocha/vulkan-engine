
#include "PostProcessRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "WaterRenderer.hpp"   // WaterParams, WaterUBO
#include "../../utils/FileReader.hpp"
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <array>
#include <vector>
#include <cstring>

PostProcessRenderer::PostProcessRenderer() {}

PostProcessRenderer::~PostProcessRenderer() {}

void PostProcessRenderer::init(VulkanApp* app) {
    createSampler(app);

    // Create uniform buffer for post-process UBO BEFORE descriptors: the
    // descriptor-buffer path stores its device address (binding 5) once via
    // vkGetDescriptorEXT, so the buffer needs SHADER_DEVICE_ADDRESS_BIT and
    // must exist before createDescriptorBuffers().
    VkBufferUsageFlags uboUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (app->useDescriptorBuffer())
        uboUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    uniformBuffer = app->createBuffer(sizeof(WaterUBO),
        uboUsage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    createPipeline(app);
    if (app->useDescriptorBuffer())
        createDescriptorBuffers(app);
    else
        createDescriptorSets(app);
}

void PostProcessRenderer::cleanup(VulkanApp* app) {
    destroyDescriptorBuffers(app);
    uniformBuffer = {};
}

void PostProcessRenderer::setRenderSize(uint32_t width, uint32_t height) {
    renderWidth = width;
    renderHeight = height;
}

// ─── Sampler ──────────────────────────────────────────────────────────────────

void PostProcessRenderer::createSampler(VulkanApp* app) {
    linearSampler = app->createSamplerLinearClamp("PostProcessRenderer: linearSampler");
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

void PostProcessRenderer::createPipeline(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Descriptor set layout – 15 bindings (14 image samplers + 1 UBO)
    std::array<VkDescriptorSetLayoutBinding, 15> bindings{};

    for (int i = 0; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Vegetation offscreen color + depth (decoupled from the solid pass)
    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // SDF debug cubes offscreen color + depth (decoupled from the solid pass)
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Mesh bounding boxes offscreen color + depth (decoupled from the solid pass)
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    DescriptorAllocator descAlloc{device, app};
    // Descriptor-buffer path: the layout must carry DESCRIPTOR_BUFFER_BIT_EXT
    // (VUID-requires it for vkGetDescriptorSetLayoutSizeEXT /
    // vkGetDescriptorSetLayoutBindingOffsetEXT and for
    // vkCmdSetDescriptorBufferOffsetsEXT binds). It must NOT be combined with
    // UPDATE_AFTER_BIND_POOL_BIT (VUID-flags-08002); host-written descriptor
    // memory needs no update-after-bind. Classic fallback keeps the original
    // UPDATE_AFTER_BIND flag for in-flight vkUpdateDescriptorSets writes.
#ifndef VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
#define VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT 0x00000010
#endif
    VkDescriptorSetLayoutCreateFlags layoutFlags =
        app->useDescriptorBuffer()
            ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            : VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    descriptorSetLayout = descAlloc.createLayout(
        bindings.data(), static_cast<uint32_t>(bindings.size()),
        layoutFlags,
        nullptr,
        "PostProcessRenderer: descriptorSetLayout");

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-process pipeline layout!");
    }
    app->resources.addPipelineLayout(pipelineLayout, "PostProcessRenderer: pipelineLayout");

    // Load shaders (cached by VulkanApp)
    VkShaderModule vertModule = app->getOrCreateShaderModule("shaders/fullscreen.vert.spv");
    VkShaderModule fragModule = app->getOrCreateShaderModule("shaders/postprocess.frag.spv");

    std::vector<VkPipelineShaderStageCreateInfo> stages = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr},
    };

    // No vertex input (fullscreen triangle generated in shader).
    // DB path: the pipeline must carry DESCRIPTOR_BUFFER_BIT_EXT, otherwise
    // vkCmdDraw with a bound descriptor buffer fails VUID-vkCmdDraw-None-08117.
    RendererUtils::FullscreenPipelineOpts pipeOpts{};
    pipeOpts.descriptorBuffer = app->useDescriptorBuffer();
    pipeline = RendererUtils::buildFullscreenPipeline(
        device, app, app->getSwapchainImageFormat(), VK_FORMAT_D32_SFLOAT, pipelineLayout, stages,
        pipeOpts, "PostProcessRenderer: pipeline");

    // Clear local shader module references; destruction handled by VulkanResourceManager
    vertModule = VK_NULL_HANDLE;
    fragModule = VK_NULL_HANDLE;
}

// ─── Descriptor Sets ──────────────────────────────────────────────────────────

void PostProcessRenderer::createDescriptorSets(VulkanApp* app) {
    if (descriptorSetLayout == VK_NULL_HANDLE) return;

    DescriptorAllocator descAlloc{app->getDevice(), app};

    VkDescriptorPoolSize poolSizesDesc[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 14 * FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * FRAMES_IN_FLIGHT}
    };
    descriptorPool = descAlloc.createPool(
        poolSizesDesc, 2, FRAMES_IN_FLIGHT,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "PostProcessRenderer: descriptorPool");

    descAlloc.allocateSets(descriptorPool, descriptorSetLayout,
                           FRAMES_IN_FLIGHT, reinterpret_cast<VkDescriptorSet*>(descriptorSets.data()),
                           "PostProcessRenderer: descriptorSet");
}

// ─── Descriptor Buffers (VK_EXT_descriptor_buffer, Phase 2) ────────────────
// 3 buffers (one per frame slot). Layout = descriptorSetLayout (15 bindings:
// 14 images + 1 UBO). Static bindings 0-4, 6-14 are stable per frame slot;
// binding 5 holds the UBO device address (written once — per-frame UBO
// contents stream via memcpy into uniformBuffer, no descriptor update).

void PostProcessRenderer::createDescriptorBuffers(VulkanApp* app) {
    destroyDescriptorBuffers(app);
    if (!app || !app->useDescriptorBuffer()) return;
    if (!app->fpGetDescriptorSetLayoutSizeEXT || !app->fpGetDescriptorSetLayoutBindingOffsetEXT ||
        !app->fpGetDescriptorEXT)
        return;
    if (descriptorSetLayout == VK_NULL_HANDLE) return;

    VkDevice device = app->getDevice();
    VkDeviceSize rawSize = 0;
    app->fpGetDescriptorSetLayoutSizeEXT(device, descriptorSetLayout, &rawSize);
    VkDeviceSize align = static_cast<VkDeviceSize>(app->descriptorBufferProps.descriptorBufferOffsetAlignment);
    if (align == 0) align = 1;
    VkDeviceSize setSize = (rawSize + align - 1) & ~(align - 1);
    if (setSize == 0) return;

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        Buffer b = app->createBuffer(setSize,
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer == VK_NULL_HANDLE || b.mappedData == nullptr) {
            destroyDescriptorBuffers(app);
            return;
        }
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = b.buffer;
        VkDeviceAddress addr = vkGetBufferDeviceAddress(device, &addrInfo);
        if (addr == 0) {
            app->destroyBuffer(b);
            destroyDescriptorBuffers(app);
            return;
        }
        descBuffers_[i] = b;
        descAddresses_[i] = addr;
    }
    for (uint32_t binding = 0; binding < 15; ++binding) {
        VkDeviceSize off = 0;
        app->fpGetDescriptorSetLayoutBindingOffsetEXT(device, descriptorSetLayout, binding, &off);
        descBindingOffsets_[binding] = off;
    }
    descSetSize_ = setSize;
    descReady_ = true;
    printf("[PostProcessRenderer] descriptor buffers: %u frames x %llu bytes\n",
           FRAMES_IN_FLIGHT, (unsigned long long)setSize);
}

void PostProcessRenderer::destroyDescriptorBuffers(VulkanApp* app) {
    bool any = false;
    for (auto& b : descBuffers_) {
        if (b.buffer != VK_NULL_HANDLE) { any = true; break; }
    }
    if (!any) {
        descAddresses_.fill(0);
        descBindingOffsets_.fill(0);
        descSetSize_ = 0;
        descReady_ = false;
        return;
    }
    if (app) {
        for (auto& b : descBuffers_) {
            if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
            else b = {};
        }
    } else {
        for (auto& b : descBuffers_) b = {};
    }
    descAddresses_.fill(0);
    descBindingOffsets_.fill(0);
    descSetSize_ = 0;
    descReady_ = false;
}

bool PostProcessRenderer::writeSlotToDescriptorBuffer(VulkanApp* app, uint32_t slot,
                                    const std::array<VkDescriptorImageInfo, 15>& imageInfos,
                                    const VkDescriptorImageInfo& skyImageInfo,
                                    const VkDescriptorBufferInfo& bufferInfo) {
    if (!app || !descReady_ || slot >= FRAMES_IN_FLIGHT) return false;
    if (!app->fpGetDescriptorEXT) return false;
    Buffer& dst = descBuffers_[slot];
    if (dst.buffer == VK_NULL_HANDLE || dst.mappedData == nullptr) return false;
    const auto& props = app->descriptorBufferProps;
    const size_t align = props.descriptorBufferOffsetAlignment ? props.descriptorBufferOffsetAlignment : 1;
    DescriptorBuffer view(app->getDevice(), app->fpGetDescriptorEXT,
                          app->fpGetDescriptorSetLayoutBindingOffsetEXT,
                          dst.mappedData, static_cast<size_t>(descSetSize_), align);
    const size_t imgSize = props.combinedImageSamplerDescriptorSize;
    const size_t uboSize = props.uniformBufferDescriptorSize;
    bool ok = true;
    auto wImg = [&](uint32_t binding, const VkDescriptorImageInfo& info) {
        if (info.imageView == VK_NULL_HANDLE || info.sampler == VK_NULL_HANDLE) return;
        if (!view.writeImage(static_cast<size_t>(descBindingOffsets_[binding]),
                             imgSize, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             info.sampler, info.imageView, info.imageLayout))
            ok = false;
    };
    // Static image bindings 0-4, 6-14 (binding 6 = sky).
    for (uint32_t i = 0; i <= 4; ++i) wImg(i, imageInfos[i]);
    wImg(6, skyImageInfo);
    wImg(7, imageInfos[7]);
    wImg(8, imageInfos[8]);
    wImg(9, imageInfos[9]);
    wImg(10, imageInfos[10]);
    wImg(11, imageInfos[11]);
    wImg(12, imageInfos[12]);
    wImg(13, imageInfos[13]);
    wImg(14, imageInfos[14]);
    // Dynamic binding 5 (UBO): address written once per slot; contents stream
    // via memcpy. vkGetDescriptorEXT forbids VK_WHOLE_SIZE, so the exact range
    // is passed.
    if (bufferInfo.buffer != VK_NULL_HANDLE && bufferInfo.range != VK_WHOLE_SIZE && bufferInfo.range != 0) {
        if (!view.writeBuffer(static_cast<size_t>(descBindingOffsets_[5]),
                              uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                              bufferInfo.buffer, bufferInfo.offset, bufferInfo.range))
            ok = false;
    }
    return ok;
}

// ─── Render ───────────────────────────────────────────────────────────────────

void PostProcessRenderer::render(VulkanApp* app, VkCommandBuffer cmd,
                                   VkImageView sceneColorView, VkImageView sceneDepthView,
                                   VkImageView waterColorView,
                                   VkImageView brushColorView, VkImageView brushDepthView,
                                   VkImageView brushBackFaceDepthView,
                                   VkImageView waterGeomDepthView,
                                   VkImageView vegColorView, VkImageView vegDepthView,
                                   VkImageView sdfColorView, VkImageView sdfDepthView,
                                   VkImageView bboxColorView, VkImageView bboxDepthView,
                                   float brushAlpha, float brushMode,
                                   const glm::mat4& viewProj, const glm::mat4& invViewProj,
                                   const glm::vec3& viewPos,
                                   uint32_t frameIdx,
                                   VkImageView skyView) {
    assert(skyView != VK_NULL_HANDLE);
    if (pipeline == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] pipeline is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (cmd == VK_NULL_HANDLE) {
        std::cerr << "[PostProcessRenderer::render] cmd is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    VkDevice device = app->getDevice();

    // Update water UBO
    WaterUBO ubo{};
    ubo.viewProjection = viewProj;
    ubo.invViewProjection = invViewProj;
    ubo.viewPos = glm::vec4(viewPos, 1.0f);
    ubo.screenSize = glm::vec4(renderWidth, renderHeight, 1.0f / renderWidth, 1.0f / renderHeight);
    ubo.brushAlpha = brushAlpha;
    ubo.brushMode = brushMode;

    void* data;
    data = uniformBuffer.map(0);
    memcpy(data, &ubo, sizeof(WaterUBO));
    uniformBuffer.unmap(); // VMA persistent mapping

    // Prepare image infos and only write descriptors for valid image views
    std::array<VkDescriptorImageInfo, 15> imageInfos{};
    {
        static bool diagPrinted = false;
        if (!diagPrinted) {
            diagPrinted = true;
            std::cerr << "[PostProcess DIAG] views: "
                << "0=" << (void*)sceneColorView
                << " 1=" << (void*)sceneDepthView
                << " 2=" << (void*)waterColorView
                << " 3=" << (void*)brushColorView
                << " 4=" << (void*)brushDepthView
                << " 7=" << (void*)waterGeomDepthView
                << " 8=" << (void*)brushBackFaceDepthView
                << " 9=" << (void*)vegColorView
                << " 10=" << (void*)vegDepthView
                << " 11=" << (void*)sdfColorView
                << " 12=" << (void*)sdfDepthView
                << " 13=" << (void*)bboxColorView
                << " 14=" << (void*)bboxDepthView
                << " sampler=" << (void*)linearSampler
                << std::endl;
        }
    }
    imageInfos[0] = {linearSampler, sceneColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[1] = {linearSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[2] = {linearSampler, waterColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Brush color and depth for deferred composition (depth-tested against scene+water)
    imageInfos[3] = {linearSampler, brushColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[4] = {linearSampler, brushDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Water geometry depth buffer for accurate brush-vs-water occlusion
    imageInfos[7] = {linearSampler, waterGeomDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Brush back-face depth for PAINT mode volume test
    imageInfos[8] = {linearSampler, brushBackFaceDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Vegetation offscreen color + depth (decoupled from the solid pass)
    imageInfos[9] = {linearSampler, vegColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[10] = {linearSampler, vegDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // SDF debug cubes offscreen color + depth
    imageInfos[11] = {linearSampler, sdfColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[12] = {linearSampler, sdfDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    // Mesh bounding boxes offscreen color + depth
    imageInfos[13] = {linearSampler, bboxColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imageInfos[14] = {linearSampler, bboxDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkDescriptorBufferInfo bufferInfo{uniformBuffer.buffer, 0, sizeof(WaterUBO)};

    // Sky color image info (binding 6) — sky offscreen targets are always
    // available (SceneRenderer::init creates them before PostProcess init).
    VkDescriptorImageInfo skyImageInfo{};
    skyImageInfo.sampler = linearSampler;
    skyImageInfo.imageView = skyView;
    skyImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const uint32_t slot = frameIdx % FRAMES_IN_FLIGHT;
    const bool useDescBuf = app->useDescriptorBuffer() && descReady_;

    // Per-frame-slot signature: offscreen views are stable per slot, so writes
    // are skipped while every input matches — steady state issues 0 descriptor
    // updates. UBO contents stream via mapped memcpy above. `valid` starts
    // false, so the first frame always writes.
    FrameDescriptorSignature sig;
    for (int i = 0; i < 15; ++i) {
        if (i == 5) continue; // binding 5 is the UBO, stored separately below
        sig.samplers[i] = imageInfos[i].sampler;
        sig.views[i] = imageInfos[i].imageView;
        sig.layouts[i] = imageInfos[i].imageLayout;
    }
    sig.samplers[6] = skyImageInfo.sampler;
    sig.views[6] = skyImageInfo.imageView;
    sig.layouts[6] = skyImageInfo.imageLayout;
    sig.uboBuffer = bufferInfo.buffer;
    sig.uboOffset = bufferInfo.offset;
    sig.uboRange = bufferInfo.range;

    FrameDescriptorSignature& cached = descriptorWriteCache[slot];
    // Descriptor-buffer path: cache miss = host vkGetDescriptorEXT writes into
    // the slot's descriptor buffer (no vkUpdateDescriptorSets, no validation).
    // Static bindings 0-4, 6-14 are stable per slot; binding 5 (UBO address)
    // is written once and its contents stream via memcpy.
    if (useDescBuf) {
        if (!cached.valid || !cached.matches(sig)) {
            cached = sig;
            cached.valid = true;
            if (!writeSlotToDescriptorBuffer(app, slot, imageInfos, skyImageInfo, bufferInfo)) {
                std::cerr << "[PostProcessRenderer] descriptor-buffer write failed for slot "
                          << slot << std::endl;
            }
        }
    } else {
    VkDescriptorSet currentDs = descriptorSets[slot];
    if (!cached.valid || !cached.matches(sig)) {
        cached = sig;
        cached.valid = true;

        DescriptorWriter writer(device);
        for (int i = 0; i < 5; ++i) {
            if (imageInfos[i].imageView == VK_NULL_HANDLE || imageInfos[i].sampler == VK_NULL_HANDLE) {
                continue;
            }
            writer.writeImage(currentDs, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[i].sampler, imageInfos[i].imageView,
                              imageInfos[i].imageLayout);
        }

        if (bufferInfo.buffer != VK_NULL_HANDLE) {
            writer.writeBuffer(currentDs, 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                               bufferInfo.buffer, bufferInfo.offset, bufferInfo.range);
        } else {
            std::cerr << "[PostProcessRenderer] Skipping UBO binding: buffer is VK_NULL_HANDLE" << std::endl;
        }

        // Sky color texture (binding 6)
        if (skyImageInfo.imageView != VK_NULL_HANDLE && skyImageInfo.sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              skyImageInfo.sampler, skyImageInfo.imageView,
                              skyImageInfo.imageLayout);
        }

        // Water geometry depth (binding 7) — used for brush-vs-water occlusion
        if (imageInfos[7].imageView != VK_NULL_HANDLE && imageInfos[7].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[7].sampler, imageInfos[7].imageView,
                              imageInfos[7].imageLayout);
        }

        // Brush back-face depth (binding 8) — used for PAINT mode volume test
        if (imageInfos[8].imageView != VK_NULL_HANDLE && imageInfos[8].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[8].sampler, imageInfos[8].imageView,
                              imageInfos[8].imageLayout);
        }

        // Vegetation offscreen color (binding 9) + depth (binding 10)
        if (imageInfos[9].imageView != VK_NULL_HANDLE && imageInfos[9].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[9].sampler, imageInfos[9].imageView,
                              imageInfos[9].imageLayout);
        }
        if (imageInfos[10].imageView != VK_NULL_HANDLE && imageInfos[10].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[10].sampler, imageInfos[10].imageView,
                              imageInfos[10].imageLayout);
        }
        // SDF debug cubes offscreen color (binding 11) + depth (binding 12)
        if (imageInfos[11].imageView != VK_NULL_HANDLE && imageInfos[11].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[11].sampler, imageInfos[11].imageView,
                              imageInfos[11].imageLayout);
        }
        if (imageInfos[12].imageView != VK_NULL_HANDLE && imageInfos[12].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[12].sampler, imageInfos[12].imageView,
                              imageInfos[12].imageLayout);
        }
        // Mesh bounding boxes offscreen color (binding 13) + depth (binding 14)
        if (imageInfos[13].imageView != VK_NULL_HANDLE && imageInfos[13].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[13].sampler, imageInfos[13].imageView,
                              imageInfos[13].imageLayout);
        }
        if (imageInfos[14].imageView != VK_NULL_HANDLE && imageInfos[14].sampler != VK_NULL_HANDLE) {
            writer.writeImage(currentDs, 14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              imageInfos[14].sampler, imageInfos[14].imageView,
                              imageInfos[14].imageLayout);
        }

        writer.flush();
    }
    } // end classic fallback

    // Set viewport and scissor (safe to call inside already-open dynamic rendering scope)
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind pipeline and descriptors.
    // DB path: single vkCmdBindDescriptorBuffersEXT +
    // vkCmdSetDescriptorBufferOffsetsEXT per frame (no sets, no pool).
    // Classic path: vkCmdBindDescriptorSets with the per-slot set.
    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (useDescBuf) {
        // Documented barrier: descriptor-buffer host writes are host-visible +
        // coherent, and this bind runs on the same queue after the writes, so
        // no explicit barrier is needed (same serialization as the classic
        // writes this path mirrors, never concurrent with GPU reads of the
        // same slot: slot = frameIdx % FRAMES_IN_FLIGHT).
        VkDescriptorBufferBindingInfoEXT bindInfo{};
        bindInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
        bindInfo.address = descAddresses_[slot];
        bindInfo.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
        app->fpCmdBindDescriptorBuffersEXT(cmd, 1, &bindInfo);
        const uint32_t bufferIndex = 0;
        const VkDeviceSize setOffset = 0;
        app->fpCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                pipelineLayout, 0, 1, &bufferIndex, &setOffset);
    } else {
        VkDescriptorSet currentDs = descriptorSets[slot];
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 1, &currentDs, 0, nullptr);
        else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                0, 1, &currentDs, 0, nullptr);
    }

    // Draw fullscreen triangle (3 vertices, no vertex buffer needed)
    vkCmdDraw(cmd, 3, 1, 0, 0);
}
