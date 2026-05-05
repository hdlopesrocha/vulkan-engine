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
    createCaptureRenderPass(app);
    createFramebuffers(app);
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

    // Framebuffers.
    for (auto& fb : framebuffers) {
        if (fb != VK_NULL_HANDLE) {
            app->resources.removeFramebuffer(fb);
            vkDestroyFramebuffer(device, fb, nullptr);
            fb = VK_NULL_HANDLE;
        }
    }

    // Render pass.
    if (captureRenderPass != VK_NULL_HANDLE) {
        app->resources.removeRenderPass(captureRenderPass);
        vkDestroyRenderPass(device, captureRenderPass, nullptr);
        captureRenderPass = VK_NULL_HANDLE;
    }

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
        if (ds != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(ds);
            ds = VK_NULL_HANDLE;
        }
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
        // Transition depth buffer UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        app->recordTransitionImageLayoutLayer(cb, depthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1, 0, 1);

        VkBuffer     vbs[2]     = { captureVertBuf, captureInstBuf };
        VkDeviceSize offsets[2] = { 0, 0 };

        for (uint32_t viewIdx = 0; viewIdx < NUM_VIEWS; ++viewIdx) {
            const uint32_t layerIdx = layerBase + viewIdx;

            VkClearValue clearVals[2]{};
            clearVals[0].color        = { 0.0f, 0.0f, 0.0f, 0.0f };
            clearVals[1].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo rpBegin{};
            rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpBegin.renderPass        = captureRenderPass;
            rpBegin.framebuffer       = framebuffers[layerIdx];
            rpBegin.renderArea.offset = { 0, 0 };
            rpBegin.renderArea.extent = { TEX_SIZE, TEX_SIZE };
            rpBegin.clearValueCount   = 2;
            rpBegin.pClearValues      = clearVals;

            vkCmdBeginRenderPass(cb, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

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

            vkCmdEndRenderPass(cb);
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

void ImpostorCapture::createCaptureRenderPass(VulkanApp* app) {
    // Color: SHADER_READ_ONLY_OPTIMAL → COLOR_ATTACHMENT → SHADER_READ_ONLY_OPTIMAL.
    // All 60 layers are pre-transitioned to SHADER_READ_ONLY in init() so this
    // initial layout is always valid, even on the very first capture.
    VkAttachmentDescription colorAtt{};
    colorAtt.format         = VK_FORMAT_R8G8B8A8_UNORM;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Depth: DEPTH_STENCIL_ATTACHMENT → DEPTH_STENCIL_ATTACHMENT (reused every pass).
    // The first transition (UNDEFINED → DEPTH_STENCIL) is done explicitly before the loop.
    VkAttachmentDescription depthAtt{};
    depthAtt.format         = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    // Synchronize with previous pass's shader reads of the same image layer.
    deps[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass      = 0;
    deps[0].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                            | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                            | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Make the written layer visible to subsequent shader reads.
    deps[1].srcSubpass      = 0;
    deps[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

    VkAttachmentDescription atts[2] = { colorAtt, depthAtt };
    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments    = atts;
    rpInfo.subpassCount    = 1;
    rpInfo.pSubpasses      = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies   = deps;
    if (vkCreateRenderPass(app->getDevice(), &rpInfo, nullptr, &captureRenderPass) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: vkCreateRenderPass failed");
    app->resources.addRenderPass(captureRenderPass, "ImpostorCapture: captureRenderPass");
}

void ImpostorCapture::createFramebuffers(VulkanApp* app) {
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < TOTAL_LAYERS; ++i) {
        VkImageView atts[2] = { captureLayerViews[i], depthView };
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = captureRenderPass;
        fbInfo.attachmentCount = 2;
        fbInfo.pAttachments    = atts;
        fbInfo.width           = TEX_SIZE;
        fbInfo.height          = TEX_SIZE;
        fbInfo.layers          = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: vkCreateFramebuffer failed");
        app->resources.addFramebuffer(framebuffers[i], "ImpostorCapture: framebuffer");
    }
}

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
    auto vertCode = FileReader::readFile("shaders/vegetation.vert.spv");
    auto geomCode = FileReader::readFile("shaders/vegetation.geom.spv");
    // capture.frag stores raw composite albedo without baked lighting so that
    // impostors.frag can apply real-time per-frame lighting at render time.
    auto fragCode = FileReader::readFile("shaders/capture.frag.spv");

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
    bindingDescs[0] = { 0, sizeof(Vertex),    VK_VERTEX_INPUT_RATE_VERTEX   };
    bindingDescs[1] = { 1, sizeof(float) * 4, VK_VERTEX_INPUT_RATE_INSTANCE };

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
                       | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(CapturePC);

    std::vector<VkDescriptorSetLayout> setLayouts = { uboDescSetLayout, texDescSetLayout };

    auto [pipeline, layout] = app->createGraphicsPipeline(
        { vertStage, geomStage, fragStage },
        std::vector<VkVertexInputBindingDescription>{ bindingDescs[0], bindingDescs[1] },
        {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    (uint32_t)offsetof(Vertex, position) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,    (uint32_t)offsetof(Vertex, normal)   },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, texCoord) },
            { 3, 0, VK_FORMAT_R32_SINT,            (uint32_t)offsetof(Vertex, texIndex) },
            { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0                                    },
        },
        setLayouts,
        &pcRange,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        true,   // depthWrite
        true,   // colorWrite
        VK_COMPARE_OP_LESS,
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        captureRenderPass
    );

    capturePipeline       = pipeline;
    capturePipelineLayout = layout;
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
    }
}

void ImpostorCapture::destroyImGuiDescSets() {
    for (auto& ds : imguiDescSets) {
        if (ds != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(ds);
            ds = VK_NULL_HANDLE;
        }
    }
}
