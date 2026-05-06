#include "ImpostorCapture.hpp"
#include "../VulkanApp.hpp"
#include "../../math/Vertex.hpp"
#include "../../utils/FileReader.hpp"
#include <backends/imgui_impl_vulkan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

// ─────────────────────────────────────────── Fibonacci sphere ───────────────

void ImpostorCapture::generateFibonacciDirs() {
    // Golden angle in radians: π*(3 − √5)
    const float goldenAngle = glm::pi<float>() * (3.0f - std::sqrt(5.0f));
    for (uint32_t i = 0; i < NUM_VIEWS; ++i) {
        // y sweeps from ~+1 (top) to ~−1 (bottom), offset by 0.5 for interior points
        const float y     = 1.0f - (float(i) + 0.5f) / float(NUM_VIEWS) * 2.0f;
        const float r     = std::sqrt(std::max(0.0f, 1.0f - y * y));
        const float theta = goldenAngle * float(i);
        viewDirs[i] = glm::normalize(glm::vec3(std::cos(theta) * r, y, std::sin(theta) * r));
    }
}

// ─────────────────────────────────────────── Public API ─────────────────────

void ImpostorCapture::init(VulkanApp* app) {
    if (!app || initDone) return;

    generateFibonacciDirs();

    // Determine per-view UBO stride aligned to hardware requirement.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(app->getPhysicalDevice(), &props);
        const VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
        uboStride = (sizeof(CaptureUBO) + align - 1) / align * align;
    }

    createCaptureImages(app);
    createDepth(app);
    createDescSetLayouts(app);
    createPipeline(app);
    createUBO(app);
    createCaptureBuffers(app);
    createSceneSampler(app);
    allocateDescSets(app);
    imguiDescSets.fill(VK_NULL_HANDLE);
    capturedTypes = 0;

    // Pre-transition all 60 layers UNDEFINED → SHADER_READ_ONLY_OPTIMAL
    // so the render pass (which now declares initialLayout=SHADER_READ_ONLY)
    // sees a consistent layout even before the first capture.
    app->runSingleTimeCommands([&](VkCommandBuffer cb) {
        app->recordTransitionImageLayoutLayer(cb, captureImage, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, 0, TOTAL_LAYERS);
        app->recordTransitionImageLayoutLayer(cb, captureNormalImage, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, 0, TOTAL_LAYERS);
    });

    initDone = true;
    fprintf(stderr, "[ImpostorCapture] init complete: %u billboard types × %u views = %u layers @ %ux%u\n",
            NUM_BILLBOARD_TYPES, NUM_VIEWS, TOTAL_LAYERS, TEX_SIZE, TEX_SIZE);
}

void ImpostorCapture::cleanup(VulkanApp* app) {
    if (!app || !initDone) return;
    VkDevice device = app->getDevice();

    destroyImGuiDescSets();

    if (imguiSampler != VK_NULL_HANDLE) {
        app->resources.removeSampler(imguiSampler);
        vkDestroySampler(device, imguiSampler, nullptr);
        imguiSampler = VK_NULL_HANDLE;
    }

    // Descriptor pool (frees uboDescSet + texDescSet implicitly).
    if (descriptorPool != VK_NULL_HANDLE) {
        app->resources.removeDescriptorPool(descriptorPool);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
        uboDescSet = texDescSet = VK_NULL_HANDLE;
    }

    // UBO buffer.
    if (uboMapped) { vkUnmapMemory(device, uboMemory); uboMapped = nullptr; }
    if (uboBuffer != VK_NULL_HANDLE) {
        app->resources.removeBuffer(uboBuffer);
        vkDestroyBuffer(device, uboBuffer, nullptr);
        uboBuffer = VK_NULL_HANDLE;
    }
    if (uboMemory != VK_NULL_HANDLE) {
        app->resources.removeDeviceMemory(uboMemory);
        vkFreeMemory(device, uboMemory, nullptr);
        uboMemory = VK_NULL_HANDLE;
    }

    // Capture vertex/instance buffers.
    auto destroyBuf = [&](VkBuffer& b, VkDeviceMemory& m) {
        if (b != VK_NULL_HANDLE) { app->resources.removeBuffer(b); vkDestroyBuffer(device, b, nullptr); b = VK_NULL_HANDLE; }
        if (m != VK_NULL_HANDLE) { app->resources.removeDeviceMemory(m); vkFreeMemory(device, m, nullptr); m = VK_NULL_HANDLE; }
    };
    destroyBuf(captureVertBuf, captureVertMem);
    destroyBuf(captureInstBuf, captureInstMem);

    // Framebuffers removed - using dynamic rendering

    // Render pass removed - using dynamic rendering

    // Pipeline + layout.
    if (capturePipeline != VK_NULL_HANDLE) {
        app->resources.removePipeline(capturePipeline);
        vkDestroyPipeline(device, capturePipeline, nullptr);
        capturePipeline = VK_NULL_HANDLE;
    }
    if (capturePipelineLayout != VK_NULL_HANDLE) {
        app->resources.removePipelineLayout(capturePipelineLayout);
        vkDestroyPipelineLayout(device, capturePipelineLayout, nullptr);
        capturePipelineLayout = VK_NULL_HANDLE;
    }

    // Descriptor set layouts.
    if (uboDescSetLayout != VK_NULL_HANDLE) {
        app->resources.removeDescriptorSetLayout(uboDescSetLayout);
        vkDestroyDescriptorSetLayout(device, uboDescSetLayout, nullptr);
        uboDescSetLayout = VK_NULL_HANDLE;
    }
    if (texDescSetLayout != VK_NULL_HANDLE) {
        app->resources.removeDescriptorSetLayout(texDescSetLayout);
        vkDestroyDescriptorSetLayout(device, texDescSetLayout, nullptr);
        texDescSetLayout = VK_NULL_HANDLE;
    }

    // Depth image.
    if (depthView   != VK_NULL_HANDLE) { app->resources.removeImageView(depthView);   vkDestroyImageView(device, depthView, nullptr);  depthView   = VK_NULL_HANDLE; }
    if (depthImage  != VK_NULL_HANDLE) { app->resources.removeImage(depthImage);      vkDestroyImage(device, depthImage, nullptr);     depthImage  = VK_NULL_HANDLE; }
    if (depthMemory != VK_NULL_HANDLE) { app->resources.removeDeviceMemory(depthMemory); vkFreeMemory(device, depthMemory, nullptr);    depthMemory = VK_NULL_HANDLE; }

    // Scene sampler.
    if (sceneSampler != VK_NULL_HANDLE) {
        app->resources.removeSampler(sceneSampler);
        vkDestroySampler(device, sceneSampler, nullptr);
        sceneSampler = VK_NULL_HANDLE;
    }

    // Capture image views.
    for (auto& v : captureLayerViews) {
        if (v != VK_NULL_HANDLE) { app->resources.removeImageView(v); vkDestroyImageView(device, v, nullptr); v = VK_NULL_HANDLE; }
    }
    if (captureArrayView != VK_NULL_HANDLE) {
        app->resources.removeImageView(captureArrayView);
        vkDestroyImageView(device, captureArrayView, nullptr);
        captureArrayView = VK_NULL_HANDLE;
    }
    if (captureImage  != VK_NULL_HANDLE) { app->resources.removeImage(captureImage);       vkDestroyImage(device, captureImage, nullptr);  captureImage  = VK_NULL_HANDLE; }
    if (captureMemory != VK_NULL_HANDLE) { app->resources.removeDeviceMemory(captureMemory); vkFreeMemory(device, captureMemory, nullptr); captureMemory = VK_NULL_HANDLE; }

    // Normal capture image views.
    for (auto& v : captureNormalLayerViews) {
        if (v != VK_NULL_HANDLE) { app->resources.removeImageView(v); vkDestroyImageView(device, v, nullptr); v = VK_NULL_HANDLE; }
    }
    if (captureNormalArrayView != VK_NULL_HANDLE) {
        app->resources.removeImageView(captureNormalArrayView);
        vkDestroyImageView(device, captureNormalArrayView, nullptr);
        captureNormalArrayView = VK_NULL_HANDLE;
    }
    if (captureNormalImage  != VK_NULL_HANDLE) { app->resources.removeImage(captureNormalImage);       vkDestroyImage(device, captureNormalImage, nullptr);  captureNormalImage  = VK_NULL_HANDLE; }
    if (captureNormalMemory != VK_NULL_HANDLE) { app->resources.removeDeviceMemory(captureNormalMemory); vkFreeMemory(device, captureNormalMemory, nullptr); captureNormalMemory = VK_NULL_HANDLE; }

    capturedTypes = 0;
    initDone = false;
}

void ImpostorCapture::capture(VulkanApp* app,
                               VkImageView albedoView, VkImageView normalView,
                               VkImageView opacityView, VkSampler   sampler,
                               float billboardScale, uint32_t billboardType) {
    if (!initDone || !app) return;
    if (albedoView == VK_NULL_HANDLE || normalView == VK_NULL_HANDLE ||
        opacityView == VK_NULL_HANDLE || sampler   == VK_NULL_HANDLE) {
        fprintf(stderr, "[ImpostorCapture] capture: billboard texture arrays not ready\n");
        return;
    }
    if (billboardType >= NUM_BILLBOARD_TYPES) {
        fprintf(stderr, "[ImpostorCapture] capture: billboardType=%u out of range\n", billboardType);
        return;
    }

    // Layer range for this billboard type.
    const uint32_t layerBase = billboardType * NUM_VIEWS;

    // Remove old ImGui descriptors for this type before overwriting images.
    for (uint32_t v = 0; v < NUM_VIEWS; ++v) {
        VkDescriptorSet& ds = imguiDescSets[layerBase + v];
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); ds = VK_NULL_HANDLE; }
        VkDescriptorSet& dsN = imguiNormalDescSets[layerBase + v];
        if (dsN != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(dsN); dsN = VK_NULL_HANDLE; }
    }

    // Update the texture descriptor set with current billboard arrays.
    updateTexDescSet(app->getDevice(), albedoView, normalView, opacityView, sampler);

    // Single centred instance at the origin with the correct billboard type.
    {
        const glm::vec4 inst(0.0f, 0.0f, 0.0f, float(billboardType));
        void* mapped;
        vkMapMemory(app->getDevice(), captureInstMem, 0, sizeof(inst), 0, &mapped);
        std::memcpy(mapped, &inst, sizeof(inst));
        vkUnmapMemory(app->getDevice(), captureInstMem);
    }

    // Pre-compute all 20 UBOs.
    const glm::vec3 center(0.0f, billboardScale * 0.5f, 0.0f);
    const float captureDist = billboardScale * 2.5f;
    const float nearP       = 0.5f;
    const float farP        = captureDist * 2.0f + billboardScale;
    const glm::vec4 lightDir   = glm::vec4(glm::normalize(glm::vec3(0.6f, -1.0f, 0.5f)), 0.0f);
    const glm::vec4 lightColor = glm::vec4(1.0f, 0.97f, 0.88f, 1.0f);

    for (uint32_t i = 0; i < NUM_VIEWS; ++i) {
        const glm::vec3 dir = viewDirs[i];
        const glm::vec3 eye = center + dir * captureDist;
        glm::vec3 worldUp   = glm::vec3(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(dir, worldUp)) > 0.99f)
            worldUp = glm::vec3(0.0f, 0.0f, 1.0f);

        const glm::mat4 view = glm::lookAt(eye, center, worldUp);
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, nearP, farP);
        proj[1][1] *= -1.0f;

        CaptureUBO ubo{};
        ubo.viewProjection = proj * view;
        ubo.viewPos        = glm::vec4(eye, 1.0f);
        ubo.lightDir       = lightDir;
        ubo.lightColor     = lightColor;
        std::memcpy(static_cast<uint8_t*>(uboMapped) + i * uboStride, &ubo, sizeof(CaptureUBO));
    }

    // Push constant template: no wind, density culling disabled, impostorDistance=0.
    CapturePC pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = 0.0f;
    pc.windTime           = 0.0f;
    pc.impostorDistance   = 0.0f;
    pc.windDirAndStrength = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    pc.windNoise          = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    pc.windShape          = glm::vec4(0.0f, 1.0f, 1.0f, 0.0f);
    pc.windTurbulence     = glm::vec4(0.0f);
    pc.densityParams      = glm::vec4(0.0f);
    pc.cameraPosAndFalloff = glm::vec4(center, 0.0f);

    app->runSingleTimeCommands([&](VkCommandBuffer cb) {
        VkBuffer     vbs[2]     = { captureVertBuf, captureInstBuf };
        VkDeviceSize offsets[2] = { 0, 0 };

        for (uint32_t viewIdx = 0; viewIdx < NUM_VIEWS; ++viewIdx) {
            const uint32_t layerIdx = layerBase + viewIdx;

            // Transition color layers to COLOR_ATTACHMENT_OPTIMAL
            app->recordTransitionImageLayoutLayer(cb, captureImage, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, layerIdx, 1);
            app->recordTransitionImageLayoutLayer(cb, captureNormalImage, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, layerIdx, 1);
            // Transition depth UNDEFINED → DEPTH_STENCIL_ATTACHMENT (reuse each view)
            app->recordTransitionImageLayoutLayer(cb, depthImage, VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);

            VkRenderingAttachmentInfo colorAtts[2]{};
            colorAtts[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAtts[0].imageView = captureLayerViews[layerIdx];
            colorAtts[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAtts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtts[0].clearValue.color = { 0.0f, 0.0f, 0.0f, 0.0f };
            colorAtts[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAtts[1].imageView = captureNormalLayerViews[layerIdx];
            colorAtts[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAtts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtts[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtts[1].clearValue.color = { 0.5f, 0.5f, 1.0f, 0.0f };

            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = depthView;
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.clearValue.depthStencil = { 1.0f, 0 };

            VkRenderingInfo renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            renderingInfo.renderArea.offset = {0, 0};
            renderingInfo.renderArea.extent = {TEX_SIZE, TEX_SIZE};
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 2;
            renderingInfo.pColorAttachments = colorAtts;
            renderingInfo.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(cb, &renderingInfo);

            VkViewport vp{ 0.0f, 0.0f, float(TEX_SIZE), float(TEX_SIZE), 0.0f, 1.0f };
            VkRect2D   sci{ { 0, 0 }, { TEX_SIZE, TEX_SIZE } };
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sci);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, capturePipeline);

            const uint32_t dynOffset = static_cast<uint32_t>(viewIdx * uboStride);
            VkDescriptorSet sets[2] = { uboDescSet, texDescSet };
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    capturePipelineLayout, 0, 2, sets, 1, &dynOffset);

            vkCmdPushConstants(cb, capturePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
                               | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(CapturePC), &pc);

            vkCmdBindVertexBuffers(cb, 0, 2, vbs, offsets);
            vkCmdDraw(cb, 1, NUM_INSTANCES, 0, 0);

            vkCmdEndRendering(cb);

            // Transition color layers back to SHADER_READ_ONLY_OPTIMAL
            app->recordTransitionImageLayoutLayer(cb, captureImage, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, layerIdx, 1);
            app->recordTransitionImageLayoutLayer(cb, captureNormalImage, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, layerIdx, 1);
        }
    });

    createImGuiDescSetsForType(app, billboardType);
    capturedTypes |= (1u << billboardType);
    fprintf(stderr,
            "[ImpostorCapture] Captured billboard type %u | %u views | scale=%.1f | layers %u-%u\n",
            billboardType, NUM_VIEWS, billboardScale, layerBase, layerBase + NUM_VIEWS - 1);
}

void ImpostorCapture::captureAll(VulkanApp* app,
                                   VkImageView albedoView, VkImageView normalView,
                                   VkImageView opacityView, VkSampler  sampler,
                                   float billboardScale) {
    for (uint32_t t = 0; t < NUM_BILLBOARD_TYPES; ++t) {
        capture(app, albedoView, normalView, opacityView, sampler, billboardScale, t);
    }
    fprintf(stderr, "[ImpostorCapture] captureAll complete: %u types × %u views = %u layers\n",
            NUM_BILLBOARD_TYPES, NUM_VIEWS, TOTAL_LAYERS);
}

VkDescriptorSet ImpostorCapture::getImGuiDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    if (billboardType >= NUM_BILLBOARD_TYPES || viewIdx >= NUM_VIEWS) return VK_NULL_HANDLE;
    return imguiDescSets[billboardType * NUM_VIEWS + viewIdx];
}

VkDescriptorSet ImpostorCapture::getImGuiNormalDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    if (billboardType >= NUM_BILLBOARD_TYPES || viewIdx >= NUM_VIEWS) return VK_NULL_HANDLE;
    return imguiNormalDescSets[billboardType * NUM_VIEWS + viewIdx];
}

uint32_t ImpostorCapture::closestView(const glm::vec3& dir) const {
    float    best = -2.0f;
    uint32_t idx  = 0;
    for (uint32_t i = 0; i < NUM_VIEWS; ++i) {
        const float d = glm::dot(dir, viewDirs[i]);
        if (d > best) { best = d; idx = i; }
    }
    return idx;
}

// ─────────────────────────────────────────── Private helpers ────────────────

void ImpostorCapture::createCaptureImages(VulkanApp* app) {
    VkDevice device = app->getDevice();
    const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;

    // Array image (all views).
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = fmt;
        imgInfo.extent        = { TEX_SIZE, TEX_SIZE, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = TOTAL_LAYERS;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &imgInfo, nullptr, &captureImage) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureImage creation failed");
        app->resources.addImage(captureImage, "ImpostorCapture: captureImage");

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, captureImage, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &captureMemory) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureMemory allocation failed");
        app->resources.addDeviceMemory(captureMemory, "ImpostorCapture: captureMemory");
        vkBindImageMemory(device, captureImage, captureMemory, 0);
    }

    // All-layers view (VK_IMAGE_VIEW_TYPE_2D_ARRAY).
    {
        VkImageViewCreateInfo v{};
        v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image            = captureImage;
        v.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v.format           = fmt;
        v.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, TOTAL_LAYERS };
        if (vkCreateImageView(device, &v, nullptr, &captureArrayView) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureArrayView failed");
        app->resources.addImageView(captureArrayView, "ImpostorCapture: captureArrayView");
    }

    // Per-layer views (VK_IMAGE_VIEW_TYPE_2D, used as framebuffer attachments).
    for (uint32_t i = 0; i < TOTAL_LAYERS; ++i) {
        VkImageViewCreateInfo v{};
        v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image            = captureImage;
        v.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        v.format           = fmt;
        v.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1 };
        if (vkCreateImageView(device, &v, nullptr, &captureLayerViews[i]) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureLayerView failed");
        app->resources.addImageView(captureLayerViews[i], "ImpostorCapture: captureLayerView");
    }

    // ── Normal capture image (world-space normals, same dimensions/format) ──
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = fmt;
        imgInfo.extent        = { TEX_SIZE, TEX_SIZE, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = TOTAL_LAYERS;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &imgInfo, nullptr, &captureNormalImage) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureNormalImage creation failed");
        app->resources.addImage(captureNormalImage, "ImpostorCapture: captureNormalImage");

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, captureNormalImage, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &captureNormalMemory) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureNormalMemory allocation failed");
        app->resources.addDeviceMemory(captureNormalMemory, "ImpostorCapture: captureNormalMemory");
        vkBindImageMemory(device, captureNormalImage, captureNormalMemory, 0);
    }

    // All-layers normal view.
    {
        VkImageViewCreateInfo v{};
        v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image            = captureNormalImage;
        v.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v.format           = fmt;
        v.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, TOTAL_LAYERS };
        if (vkCreateImageView(device, &v, nullptr, &captureNormalArrayView) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureNormalArrayView failed");
        app->resources.addImageView(captureNormalArrayView, "ImpostorCapture: captureNormalArrayView");
    }

    // Per-layer normal views.
    for (uint32_t i = 0; i < TOTAL_LAYERS; ++i) {
        VkImageViewCreateInfo v{};
        v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image            = captureNormalImage;
        v.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        v.format           = fmt;
        v.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1 };
        if (vkCreateImageView(device, &v, nullptr, &captureNormalLayerViews[i]) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureNormalLayerView failed");
        app->resources.addImageView(captureNormalLayerViews[i], "ImpostorCapture: captureNormalLayerView");
    }
}

void ImpostorCapture::createDepth(VulkanApp* app) {
    VkDevice device = app->getDevice();
    const VkFormat fmt = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = fmt;
    imgInfo.extent        = { TEX_SIZE, TEX_SIZE, 1 };
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &imgInfo, nullptr, &depthImage) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: depthImage creation failed");
    app->resources.addImage(depthImage, "ImpostorCapture: depthImage");

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, depthImage, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &depthMemory) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: depthMemory allocation failed");
    app->resources.addDeviceMemory(depthMemory, "ImpostorCapture: depthMemory");
    vkBindImageMemory(device, depthImage, depthMemory, 0);

    VkImageViewCreateInfo v{};
    v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.image            = depthImage;
    v.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    v.format           = fmt;
    v.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(device, &v, nullptr, &depthView) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: depthView failed");
    app->resources.addImageView(depthView, "ImpostorCapture: depthView");
}

void ImpostorCapture::createCaptureRenderPass(VulkanApp* /*app*/) {}
void ImpostorCapture::createFramebuffers(VulkanApp* /*app*/) {}

void ImpostorCapture::createDescSetLayouts(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Set 0: single DYNAMIC uniform buffer (per-view camera data).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT
                          | VK_SHADER_STAGE_GEOMETRY_BIT
                          | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1;
        info.pBindings    = &b;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &uboDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: uboDescSetLayout failed");
        app->resources.addDescriptorSetLayout(uboDescSetLayout, "ImpostorCapture: uboDescSetLayout");
    }

    // Set 1: three combined image samplers (albedo, normal, opacity arrays).
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 3;
        info.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &texDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: texDescSetLayout failed");
        app->resources.addDescriptorSetLayout(texDescSetLayout, "ImpostorCapture: texDescSetLayout");
    }
}

void ImpostorCapture::createPipeline(VulkanApp* app) {
    VkDevice device = app->getDevice();

    auto vertCode = FileReader::readFile("shaders/vegetation.vert.spv");
    auto geomCode = FileReader::readFile("shaders/vegetation.geom.spv");
    auto fragCode = FileReader::readFile("shaders/capture.frag.spv");

    VkShaderModule vertShader = app->createShaderModule(vertCode);
    VkShaderModule geomShader = app->createShaderModule(geomCode);
    VkShaderModule fragShader = app->createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[3]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShader;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_GEOMETRY_BIT;
    stages[1].module = geomShader;
    stages[1].pName  = "main";
    stages[2].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[2].module = fragShader;
    stages[2].pName  = "main";

    VkVertexInputBindingDescription bindingDescs[2]{};
    bindingDescs[0] = { 0, sizeof(Vertex),    VK_VERTEX_INPUT_RATE_VERTEX   };
    bindingDescs[1] = { 1, sizeof(float) * 4, VK_VERTEX_INPUT_RATE_INSTANCE };

    VkVertexInputAttributeDescription attrs[5]{};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    (uint32_t)offsetof(Vertex, position) };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,    (uint32_t)offsetof(Vertex, normal)   };
    attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) };
    attrs[3] = { 3, 0, VK_FORMAT_R32_SINT,            (uint32_t)offsetof(Vertex, texIndex) };
    attrs[4] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                                    };

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 2;
    vertexInput.pVertexBindingDescriptions      = bindingDescs;
    vertexInput.vertexAttributeDescriptionCount = 5;
    vertexInput.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dynStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Two color blend attachments to match the 2-attachment render pass (albedo + normal).
    VkPipelineColorBlendAttachmentState blendAtts[2]{};
    for (auto& b : blendAtts) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                         | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        b.blendEnable    = VK_FALSE;
    }
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 2;
    colorBlend.pAttachments    = blendAtts;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
                       | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.size       = sizeof(CapturePC);

    VkDescriptorSetLayout layouts[2] = { uboDescSetLayout, texDescSetLayout };
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 2;
    layoutInfo.pSetLayouts            = layouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &capturePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: pipelineLayout creation failed");
    app->resources.addPipelineLayout(capturePipelineLayout, "ImpostorCapture: capturePipelineLayout");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 3;
    pipelineInfo.pStages             = stages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pDynamicState       = &dynState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.layout              = capturePipelineLayout;

    // Dynamic rendering (no render pass)
    VkFormat colorFormats[2] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM };
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 2;
    renderingInfo.pColorAttachmentFormats = colorFormats;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    pipelineInfo.subpass             = 0;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &capturePipeline) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: pipeline creation failed");
    app->resources.addPipeline(capturePipeline, "ImpostorCapture: capturePipeline");

    app->resources.removeShaderModule(vertShader);
    vkDestroyShaderModule(device, vertShader, nullptr);
    app->resources.removeShaderModule(geomShader);
    vkDestroyShaderModule(device, geomShader, nullptr);
    app->resources.removeShaderModule(fragShader);
    vkDestroyShaderModule(device, fragShader, nullptr);
}

void ImpostorCapture::createUBO(VulkanApp* app) {
    const VkDeviceSize totalSize = NUM_VIEWS * uboStride;
    // createBuffer registers buffer and memory in resources internally.
    Buffer buf = app->createBuffer(totalSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uboBuffer = buf.buffer;
    uboMemory = buf.memory;
    vkMapMemory(app->getDevice(), uboMemory, 0, totalSize, 0, &uboMapped);
}

void ImpostorCapture::createCaptureBuffers(VulkanApp* app) {
    // Single base vertex — the geometry shader uses the instance position,
    // but a vertex buffer binding is still required by the pipeline.
    Vertex v;
    v.position = glm::vec3(0.0f);
    v.normal   = glm::vec3(0.0f, 1.0f, 0.0f);
    v.texCoord = glm::vec2(0.5f);
    v.texIndex = 0;
    Buffer vb = app->createDeviceLocalBuffer(&v, sizeof(Vertex),
                                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    captureVertBuf = vb.buffer;
    captureVertMem = vb.memory;

    // Instance buffer: 1 vec4 (xyz=world pos, w=billboardIndex), host-visible.
    Buffer ib = app->createBuffer(NUM_INSTANCES * sizeof(glm::vec4),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    captureInstBuf = ib.buffer;
    captureInstMem = ib.memory;
}

void ImpostorCapture::createSceneSampler(VulkanApp* app) {
    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.anisotropyEnable = VK_FALSE;
    if (vkCreateSampler(app->getDevice(), &si, nullptr, &sceneSampler) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: vkCreateSampler (scene) failed");
    app->resources.addSampler(sceneSampler, "ImpostorCapture: sceneSampler");
}

void ImpostorCapture::allocateDescSets(VulkanApp* app) {
    // Private descriptor pool (just enough for two sets: UBO + tex).
    {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        sizes[0].descriptorCount = 1;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = 3;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets       = 2;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes    = sizes;
        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: vkCreateDescriptorPool failed");
        app->resources.addDescriptorPool(descriptorPool, "ImpostorCapture: descriptorPool");
    }

    VkDescriptorSetLayout layouts[2] = { uboDescSetLayout, texDescSetLayout };
    VkDescriptorSet       sets[2];
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, sets) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: vkAllocateDescriptorSets failed");
    uboDescSet = sets[0];
    texDescSet = sets[1];

    // Write UBO descriptor (DYNAMIC: range = one slot, offset supplied per-draw).
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = uboBuffer;
    bufInfo.offset = 0;
    bufInfo.range  = sizeof(CaptureUBO);
    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = uboDescSet;
    w.dstBinding      = 0;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    w.descriptorCount = 1;
    w.pBufferInfo     = &bufInfo;
    vkUpdateDescriptorSets(app->getDevice(), 1, &w, 0, nullptr);
    // Texture descriptor is written in updateTexDescSet() at capture time.
}

void ImpostorCapture::updateTexDescSet(VkDevice device,
                                       VkImageView albedo, VkImageView normal,
                                       VkImageView opacity, VkSampler sampler) {
    VkDescriptorImageInfo infos[3]{};
    infos[0] = { sampler, albedo,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[1] = { sampler, normal,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[2] = { sampler, opacity, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

    VkWriteDescriptorSet ws[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = texDescSet;
        ws[i].dstBinding      = i;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[i].descriptorCount = 1;
        ws[i].pImageInfo      = &infos[i];
    }
    vkUpdateDescriptorSets(device, 3, ws, 0, nullptr);
}

void ImpostorCapture::createImGuiDescSetsForType(VulkanApp* app, uint32_t billboardType) {
    if (imguiSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(app->getDevice(), &si, nullptr, &imguiSampler);
        app->resources.addSampler(imguiSampler, "ImpostorCapture: imguiSampler");
    }
    const uint32_t layerBase = billboardType * NUM_VIEWS;
    for (uint32_t v = 0; v < NUM_VIEWS; ++v) {
        const uint32_t layerIdx = layerBase + v;
        imguiDescSets[layerIdx] = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            imguiSampler, captureLayerViews[layerIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        imguiNormalDescSets[layerIdx] = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            imguiSampler, captureNormalLayerViews[layerIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void ImpostorCapture::destroyImGuiDescSets() {
    for (auto& ds : imguiDescSets) {
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); ds = VK_NULL_HANDLE; }
    }
    for (auto& ds : imguiNormalDescSets) {
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); ds = VK_NULL_HANDLE; }
    }
}
