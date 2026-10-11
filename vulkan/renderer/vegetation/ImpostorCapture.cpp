#include "ImpostorCapture.hpp"
#include "../DescriptorAllocator.hpp"
#include "../DescriptorWriter.hpp"
#include "../RendererUtils.hpp"
#include "VegetationRenderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../../math/Vertex.hpp"
#include "../../../utils/FileReader.hpp"
#include <backends/imgui_impl_vulkan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include "../../includes/shader/locations.hpp"
#include "../../includes/shader/vertex_layouts.hpp"

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

void ImpostorCapture::init(VulkanApp* app, VegetationRenderer* vegRenderer) {
    if (!app || initDone) return;
    if (!vegRenderer ||
        vegRenderer->getWindParamsDescSetLayout() == VK_NULL_HANDLE ||
        vegRenderer->getWindParamsDescSet() == VK_NULL_HANDLE) {
        fprintf(stderr, "[ImpostorCapture] init: vegetation renderer wind params not ready, capture disabled\n");
        return;
    }
    sharedVegRenderer = vegRenderer;

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
    createCaptureInvVPBuffer(app);
    createSceneSampler(app);
    allocateDescSets(app);
    imguiDescSets.fill(VK_NULL_HANDLE);
    capturedTypes = 0;

    // H5 (perf report 27): no init-time pre-transition. It is redundant:
    // every capture re-transitions its layer ranges from UNDEFINED on first
    // touch inside recordCaptureViews (batched, CLEAR on load discards prior
    // bytes), so pre-transitioning all 60 layers here only costs a submit.

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
    }

    if (descriptorPool != VK_NULL_HANDLE) {
        app->resources.removeDescriptorPool(descriptorPool);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    uboMapped = nullptr;

    captureInstMapped = nullptr;

    if (capturePipeline != VK_NULL_HANDLE) {
        app->resources.removePipeline(capturePipeline);
        vkDestroyPipeline(device, capturePipeline, nullptr);
    }
    if (capturePipelineLayout != VK_NULL_HANDLE) {
        app->resources.removePipelineLayout(capturePipelineLayout);
        vkDestroyPipelineLayout(device, capturePipelineLayout, nullptr);
    }

    if (uboDescSetLayout != VK_NULL_HANDLE) {
        app->resources.removeDescriptorSetLayout(uboDescSetLayout);
        vkDestroyDescriptorSetLayout(device, uboDescSetLayout, nullptr);
    }
    if (texDescSetLayout != VK_NULL_HANDLE) {
        app->resources.removeDescriptorSetLayout(texDescSetLayout);
        vkDestroyDescriptorSetLayout(device, texDescSetLayout, nullptr);
    }

    if (depthView   != VK_NULL_HANDLE) { app->resources.removeImageView(depthView);   vkDestroyImageView(device, depthView, nullptr); }
    if (depthImage  != VK_NULL_HANDLE) { app->destroyImageWithVma(depthImage, depthAllocation, depthMemory); }

    if (sceneSampler != VK_NULL_HANDLE) {
        app->resources.removeSampler(sceneSampler);
        vkDestroySampler(device, sceneSampler, nullptr);
    }

    for (auto& v : captureLayerViews) {
        if (v != VK_NULL_HANDLE) { app->resources.removeImageView(v); vkDestroyImageView(device, v, nullptr); }
    }
    if (captureArrayView != VK_NULL_HANDLE) {
        app->resources.removeImageView(captureArrayView);
        vkDestroyImageView(device, captureArrayView, nullptr);
    }
    if (captureImage  != VK_NULL_HANDLE) { app->destroyImageWithVma(captureImage, captureAllocation, captureMemory); }

    for (auto& v : captureNormalLayerViews) {
        if (v != VK_NULL_HANDLE) { app->resources.removeImageView(v); vkDestroyImageView(device, v, nullptr); }
    }
    if (captureNormalArrayView != VK_NULL_HANDLE) {
        app->resources.removeImageView(captureNormalArrayView);
        vkDestroyImageView(device, captureNormalArrayView, nullptr);
    }
    if (captureNormalImage  != VK_NULL_HANDLE) { app->destroyImageWithVma(captureNormalImage, captureNormalAllocation, captureNormalMemory); }

    for (auto& v : captureDepthLayerViews) {
        if (v != VK_NULL_HANDLE) { app->resources.removeImageView(v); vkDestroyImageView(device, v, nullptr); }
    }
    if (captureDepthArrayView != VK_NULL_HANDLE) {
        app->resources.removeImageView(captureDepthArrayView);
        vkDestroyImageView(device, captureDepthArrayView, nullptr);
    }
    if (captureDepthImage  != VK_NULL_HANDLE) { app->destroyImageWithVma(captureDepthImage, captureDepthAllocation, captureDepthMemory); }

    captureInvVPMapped = nullptr;

    capturedTypes = 0;
    initDone = false;
}

// ────────────────── H8 async-capture helpers (perf report 26) ─────────────
// The recorded per-view command stream is identical to the old synchronous
// runSingleTimeCommands version (same transitions, clears, binds, push
// constants, draws with identical instance/UBO bytes), so captured texels are
// bit-identical — only host submit timing changes. All submits below use the
// graphics queue (2-arg runSingleTimeCommandsAsync overload): capture renders
// and can never ride the transfer queue.

void ImpostorCapture::writeCaptureCameraUbos(float billboardScale, uint32_t layerBase) {
    // Pre-compute all 20 UBOs.
    const glm::vec3 center(0.0f, billboardScale * 0.5f, 0.0f);
    const float captureDist = billboardScale * 2.5f;
    const float nearP       = 0.5f;
    const float farP        = captureDist * 2.0f + billboardScale;
    const glm::vec3 lightDir   = glm::normalize(glm::vec3(0.6f, -1.0f, 0.5f));
    const glm::vec3 lightColor = glm::vec3(1.0f, 0.97f, 0.88f);

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
        ubo.viewPosition   = eye;
        ubo.lightDirection = lightDir;
        ubo.lightColor     = lightColor;
        std::memcpy(static_cast<uint8_t*>(uboMapped) + i * uboStride, &ubo, sizeof(CaptureUBO));

        // Store inverse VP for depth reprojection in the shadow pass.
        const uint32_t layerIdx = layerBase + i;
        captureInvVP[layerIdx] = glm::inverse(proj * view);
    }

    // Upload updated inv VP data to the GPU buffer (persistently mapped).
    std::memcpy(captureInvVPMapped, captureInvVP.data(), TOTAL_LAYERS * sizeof(glm::mat4));
}

void ImpostorCapture::recordCaptureViews(VkCommandBuffer cb, VulkanApp* app,
                                         uint32_t layerBase, uint32_t viewCount,
                                         uint32_t firstInstance, const CapturePC& pc) {
    // Defensive: ensure descriptor sets are allocated before recording
    // anything (loop-invariant, so check once up front instead of per view).
    if (uboDescSet == VK_NULL_HANDLE || texDescSet == VK_NULL_HANDLE || viewCount == 0) {
        std::cerr << "[ImpostorCapture] descriptor sets not ready, skipping capture for these views." << std::endl;
        return;
    }

    // H5 (perf report 27): batched attachment transitions. Every layer is
    // CLEARed on load, so the three arrays stay in COLOR_ATTACHMENT_OPTIMAL
    // across all views of a capture: one batched UNDEFINED → ATTACHMENT
    // transition per array up front (discards prior bytes, correct for both
    // first capture and re-capture) and one batched ATTACHMENT →
    // SHADER_READ_ONLY transition per array at the end. Same per-layer
    // old/new layouts as the old per-view transitions, in a handful of
    // multi-layer calls instead of 6 per view.
    app->recordTransitionImageLayoutLayer(cb, captureImage, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, layerBase, viewCount);
    app->recordTransitionImageLayoutLayer(cb, captureNormalImage, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, layerBase, viewCount);
    app->recordTransitionImageLayoutLayer(cb, captureDepthImage, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, layerBase, viewCount);
    // H5 (perf report 27): the transient depth image has storeOp DONT_CARE,
    // so it stays in depth-attachment layout across views (and captures);
    // the helper skips the barrier when the tracked layout is already there.
    app->recordTransitionImageLayoutLayer(cb, depthImage, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);

    // H5 (perf report 27): view-invariant binds hoisted out of the per-view
    // loop — pipeline, viewport/scissor, vertex/index buffers (plus the
    // loop-invariant CapturePC push, M6 report 28). Only the dynamic UBO
    // offset changes per view, so only the descriptor bind is recorded per
    // view.
    VkViewport vp{ 0.0f, 0.0f, float(TEX_SIZE), float(TEX_SIZE), 0.0f, 1.0f };
    VkRect2D   sci{ { 0, 0 }, { TEX_SIZE, TEX_SIZE } };
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sci);

    if (cmdState) cmdState->bindGraphicsPipeline(cb, capturePipeline);
    else vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, capturePipeline);

    VkBuffer     vbs[2]     = { captureVertBuf, captureInstBuf };
    VkDeviceSize offsets[2] = { 0, 0 };
    vkCmdBindVertexBuffers(cb, 0, 2, vbs, offsets);
    vkCmdBindIndexBuffer(cb, captureIdxBuf, 0, VK_INDEX_TYPE_UINT32);

    // M6 (perf report 28): loop-invariant CapturePC push hoisted out of the
    // per-view loop. The bytes are identical for every view of the call (only
    // the dynamic UBO offset varies per view), and push-constant state
    // survives begin/endRendering, so one push records exactly the values
    // every draw previously re-pushed. The per-view dynamic-offset
    // descriptor bind stays inside the loop.
    vkCmdPushConstants(cb, capturePipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(CapturePC), &pc);

    for (uint32_t viewIdx = 0; viewIdx < viewCount; ++viewIdx) {
        const uint32_t layerIdx = layerBase + viewIdx;

        VkRenderingAttachmentInfo colorAtts[3]{};
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
        colorAtts[2].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAtts[2].imageView = captureDepthLayerViews[layerIdx];
        colorAtts[2].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtts[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtts[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtts[2].clearValue.color = { 1.0f, 0.0f, 0.0f, 0.0f }; // clear depth to far=1.0

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
        renderingInfo.colorAttachmentCount = 3;
        renderingInfo.pColorAttachments = colorAtts;
        renderingInfo.pDepthAttachment = &depthAtt;

        vkCmdBeginRendering(cb, &renderingInfo);

        const uint32_t dynOffset = static_cast<uint32_t>(viewIdx * uboStride);
        VkDescriptorSet sets[3] = { uboDescSet, texDescSet, sharedVegRenderer->getWindParamsDescSet() };
        if (cmdState) cmdState->bindGraphicsDescriptorSets(cb,
                                capturePipelineLayout, 0, 3, sets, 1, &dynOffset);
        else vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                capturePipelineLayout, 0, 3, sets, 1, &dynOffset);

        vkCmdDrawIndexed(cb, captureIdxCount, NUM_INSTANCES, 0, 0, firstInstance);

        vkCmdEndRendering(cb);
    }

    // H5 (perf report 27): one batched out-transition per array at the end
    // of the capture range (same per-layer old/new layouts as before).
    app->recordTransitionImageLayoutLayer(cb, captureImage, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, layerBase, viewCount);
    app->recordTransitionImageLayoutLayer(cb, captureNormalImage, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, layerBase, viewCount);
    app->recordTransitionImageLayoutLayer(cb, captureDepthImage, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, layerBase, viewCount);
}

VkFence ImpostorCapture::submitCaptureTypesAsync(VulkanApp* app, const CapturePC& pc,
                                                 uint32_t typeBegin, uint32_t typeCount) {
    // One async submit covering [typeBegin, typeBegin + typeCount). Views run
    // back to back on the graphics queue in type order, so the shared UBO
    // slots (type-independent content) are never rewritten mid-flight and the
    // 60 renders of captureAllAsync need no interim fence.
    return app->runSingleTimeCommandsAsync([&](VkCommandBuffer cb) {
        for (uint32_t t = typeBegin; t < typeBegin + typeCount; ++t)
            recordCaptureViews(cb, app, t * NUM_VIEWS, NUM_VIEWS, t, pc);
    });
}

void ImpostorCapture::completeCaptureType(VulkanApp* app, uint32_t billboardType,
                                          float billboardScale) {
    // Publish one captured type: retire the previous ImGui sets FIRST so a
    // re-capture (sync, or async whose creation is still deferred) can never
    // orphan a set, then create the new ones. Runs inline before return on the
    // sync path and as a fence callback (deferDestroyUntilFence, pumped by
    // processPendingCommandBuffers — the same mechanism the vegetation
    // streaming path uses to publish chunks) on the async path. Either way it
    // runs strictly after this type's capture renders have completed.
    const uint32_t layerBase = billboardType * NUM_VIEWS;
    for (uint32_t v = 0; v < NUM_VIEWS; ++v) {
        VkDescriptorSet ds = imguiDescSets[layerBase + v];
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); imguiDescSets[layerBase + v] = VK_NULL_HANDLE; }
        VkDescriptorSet dsN = imguiNormalDescSets[layerBase + v];
        if (dsN != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(dsN); imguiNormalDescSets[layerBase + v] = VK_NULL_HANDLE; }
    }
    createImGuiDescSetsForType(app, billboardType);
    capturedTypes |= (1u << billboardType);
    fprintf(stderr,
            "[ImpostorCapture] Captured billboard type %u | %u views | scale=%.1f | layers %u-%u\n",
            billboardType, NUM_VIEWS, billboardScale, layerBase, layerBase + NUM_VIEWS - 1);
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

    // Old ImGui sets for this type are retired in completeCaptureType() after
    // the GPU work finishes (inline on the sync path, fence-deferred on the
    // async path), so a re-capture can never orphan a deferred set.

    // Update the texture descriptor set with current billboard arrays.
    updateTexDescSet(app->getDevice(), albedoView, normalView, opacityView, sampler);

    // Single centred instance at the origin with the correct billboard type.
    // The instance buffer holds one slot per billboard type; the draw selects
    // it with firstInstance, so staged per-type data never aliases and the
    // instanceData values are identical to the old single-slot upload
    // (gl_InstanceIndex is unread by every vegetation shader: pixel-identical).
    {
        const glm::vec4 inst(0.0f, 0.0f, 0.0f, float(billboardType));
        std::memcpy(static_cast<uint8_t*>(captureInstMapped) + billboardType * sizeof(inst),
                    &inst, sizeof(inst));
    }

    // Camera UBOs + invVP slice for this type's layers (bytes identical to
    // the previous inline version; factored out so captureAllAsync can stage
    // all types before one submit).
    writeCaptureCameraUbos(billboardScale, layerBase);

    // Push constant: no wind, density culling disabled, impostorDistance=0.
    CapturePC pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = 0.0f;
    pc.windTime           = 0.0f;
    pc.impostorDistance   = 0.0f;

    // Async submit of this type's NUM_VIEWS renders (identical command stream;
    // see recordCaptureViews). The fence wait below preserves the synchronous
    // contract — the old code blocked inside runSingleTimeCommands — with the
    // codebase's validation-clean poll wait. Reaping of the transient command
    // buffer stays on the frame pump (processPendingCommandBuffers), as with
    // every other runSingleTimeCommandsAsync submitter.
    VkFence fence = submitCaptureTypesAsync(app, pc, billboardType, 1);
    VulkanApp::waitFence(app->getDevice(), fence);
    completeCaptureType(app, billboardType, billboardScale);
}

void ImpostorCapture::captureAll(VulkanApp* app,
                                   VkImageView albedoView, VkImageView normalView,
                                   VkImageView opacityView, VkSampler sampler,
                                   float billboardScale) {
    for (uint32_t t = 0; t < NUM_BILLBOARD_TYPES; ++t) {
        capture(app, albedoView, normalView, opacityView, sampler, billboardScale, t);
    }
    fprintf(stderr, "[ImpostorCapture] captureAll complete: %u types × %u views = %u layers\n",
            NUM_BILLBOARD_TYPES, NUM_VIEWS, TOTAL_LAYERS);
}

VkFence ImpostorCapture::captureAsync(VulkanApp* app,
                                       VkImageView albedoView, VkImageView normalView,
                                       VkImageView opacityView, VkSampler sampler,
                                       float billboardScale, uint32_t billboardType) {
    if (!initDone || !app) return VK_NULL_HANDLE;
    if (albedoView == VK_NULL_HANDLE || normalView == VK_NULL_HANDLE ||
        opacityView == VK_NULL_HANDLE || sampler   == VK_NULL_HANDLE) {
        fprintf(stderr, "[ImpostorCapture] captureAsync: billboard texture arrays not ready\n");
        return VK_NULL_HANDLE;
    }
    if (billboardType >= NUM_BILLBOARD_TYPES) {
        fprintf(stderr, "[ImpostorCapture] captureAsync: billboardType=%u out of range\n", billboardType);
        return VK_NULL_HANDLE;
    }
    const uint32_t layerBase = billboardType * NUM_VIEWS;

    updateTexDescSet(app->getDevice(), albedoView, normalView, opacityView, sampler);

    {
        const glm::vec4 inst(0.0f, 0.0f, 0.0f, float(billboardType));
        std::memcpy(static_cast<uint8_t*>(captureInstMapped) + billboardType * sizeof(inst),
                    &inst, sizeof(inst));
    }
    writeCaptureCameraUbos(billboardScale, layerBase);

    // Push constant: no wind, density culling disabled, impostorDistance=0.
    CapturePC pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = 0.0f;
    pc.windTime           = 0.0f;
    pc.impostorDistance   = 0.0f;

    VkFence fence = submitCaptureTypesAsync(app, pc, billboardType, 1);
    // Publish after the fence (see completeCaptureType); pumped by
    // processPendingCommandBuffers. `this`/`app` outlive the fence by
    // ownership (service/app lifetime), same as the vegetation streaming
    // publish callbacks.
    app->deferDestroyUntilFence(fence, [this, app, billboardType, billboardScale]() {
        completeCaptureType(app, billboardType, billboardScale);
    });
    return fence;
}

VkFence ImpostorCapture::captureAllAsync(VulkanApp* app,
                                          VkImageView albedoView, VkImageView normalView,
                                          VkImageView opacityView, VkSampler sampler,
                                          float billboardScale) {
    if (!initDone || !app) return VK_NULL_HANDLE;
    if (albedoView == VK_NULL_HANDLE || normalView == VK_NULL_HANDLE ||
        opacityView == VK_NULL_HANDLE || sampler   == VK_NULL_HANDLE) {
        fprintf(stderr, "[ImpostorCapture] captureAllAsync: billboard texture arrays not ready\n");
        return VK_NULL_HANDLE;
    }

    updateTexDescSet(app->getDevice(), albedoView, normalView, opacityView, sampler);

    for (uint32_t t = 0; t < NUM_BILLBOARD_TYPES; ++t) {
        const glm::vec4 inst(0.0f, 0.0f, 0.0f, float(t));
        std::memcpy(static_cast<uint8_t*>(captureInstMapped) + t * sizeof(inst),
                    &inst, sizeof(inst));
        writeCaptureCameraUbos(billboardScale, t * NUM_VIEWS);
    }

    // Push constant: no wind, density culling disabled, impostorDistance=0.
    CapturePC pc{};
    pc.billboardScale     = billboardScale;
    pc.windEnabled        = 0.0f;
    pc.windTime           = 0.0f;
    pc.impostorDistance   = 0.0f;

    // ONE submit for all 60 views (type order, shared type-independent UBO
    // slots, per-type instance slots via firstInstance): no mid-flight CPU
    // rewrite of shared slots, so no interim fence is needed.
    VkFence fence = submitCaptureTypesAsync(app, pc, 0, NUM_BILLBOARD_TYPES);
    app->deferDestroyUntilFence(fence, [this, app, billboardScale]() {
        for (uint32_t t = 0; t < NUM_BILLBOARD_TYPES; ++t)
            completeCaptureType(app, t, billboardScale);
        fprintf(stderr, "[ImpostorCapture] captureAllAsync complete: %u types × %u views = %u layers\n",
                NUM_BILLBOARD_TYPES, NUM_VIEWS, TOTAL_LAYERS);
    });
    return fence;
}

VkDescriptorSet ImpostorCapture::getImGuiDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    if (billboardType >= NUM_BILLBOARD_TYPES || viewIdx >= NUM_VIEWS) return VK_NULL_HANDLE;
    return imguiDescSets[billboardType * NUM_VIEWS + viewIdx];
}

VkDescriptorSet ImpostorCapture::getImGuiNormalDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    if (billboardType >= NUM_BILLBOARD_TYPES || viewIdx >= NUM_VIEWS) return VK_NULL_HANDLE;
    return imguiNormalDescSets[billboardType * NUM_VIEWS + viewIdx];
}

VkDescriptorSet ImpostorCapture::getImGuiDepthDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    if (billboardType >= NUM_BILLBOARD_TYPES || viewIdx >= NUM_VIEWS) return VK_NULL_HANDLE;
    return imguiDepthDescSets[billboardType * NUM_VIEWS + viewIdx];
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
        app->createImageWithVma(imgInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, captureImage, captureAllocation, captureMemory, "ImpostorCapture: captureImage");
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
        app->createImageWithVma(imgInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, captureNormalImage, captureNormalAllocation, captureNormalMemory, "ImpostorCapture: captureNormalImage");
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

    // ── Depth capture image (device Z, R32_SFLOAT, 60 layers) ──
    {
        const VkFormat depthFmt = VK_FORMAT_R32_SFLOAT;
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = depthFmt;
        imgInfo.extent        = { TEX_SIZE, TEX_SIZE, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = TOTAL_LAYERS;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        app->createImageWithVma(imgInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, captureDepthImage, captureDepthAllocation, captureDepthMemory, "ImpostorCapture: captureDepthImage");
    }

    // All-layers depth view (VK_IMAGE_VIEW_TYPE_2D_ARRAY).
    {
        VkImageViewCreateInfo v{};
        v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image            = captureDepthImage;
        v.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v.format           = VK_FORMAT_R32_SFLOAT;
        v.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, TOTAL_LAYERS };
        if (vkCreateImageView(device, &v, nullptr, &captureDepthArrayView) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureDepthArrayView failed");
        app->resources.addImageView(captureDepthArrayView, "ImpostorCapture: captureDepthArrayView");
    }

    // Per-layer depth views (VK_IMAGE_VIEW_TYPE_2D).
    for (uint32_t i = 0; i < TOTAL_LAYERS; ++i) {
        VkImageViewCreateInfo v{};
        v.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image            = captureDepthImage;
        v.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        v.format           = VK_FORMAT_R32_SFLOAT;
        v.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1 };
        if (vkCreateImageView(device, &v, nullptr, &captureDepthLayerViews[i]) != VK_SUCCESS)
            throw std::runtime_error("ImpostorCapture: captureDepthLayerView failed");
        app->resources.addImageView(captureDepthLayerViews[i], "ImpostorCapture: captureDepthLayerView");
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
    app->createImageWithVma(imgInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthAllocation, depthMemory, "ImpostorCapture: depthImage");

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

void ImpostorCapture::createDescSetLayouts(VulkanApp* app) {
    VkDevice device = app->getDevice();
    DescriptorAllocator descAlloc{device, app};

    // Set 0: single DYNAMIC uniform buffer (per-view camera data).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT
                          | VK_SHADER_STAGE_GEOMETRY_BIT
                          | VK_SHADER_STAGE_FRAGMENT_BIT;
        uboDescSetLayout = descAlloc.createLayout(
            &b, 1,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            nullptr,
            "ImpostorCapture: uboDescSetLayout");
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
        texDescSetLayout = descAlloc.createLayout(
            bindings, 3,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            nullptr,
            "ImpostorCapture: texDescSetLayout");
    }
}

void ImpostorCapture::createPipeline(VulkanApp* app) {
    VkDevice device = app->getDevice();

    // Use VegetationRenderer.vert for billboard expansion (same 24-corner + indexed approach).
    // VEG_CAPTURE variant (perf report 22 C2): evaluates the height scale
    // locally exactly as before the bake — the capture draws a canonical
    // single instance with no aux buffer bound, so it must not declare the
    // baked-height input (VUID-07904).
    // Shader derivation (H8, perf report 26): there is NO on-disk
    // ImpostorCapture.vert source. This module is compiled from
    // VegetationRenderer.vert with -DVEG_CAPTURE=1 (see the Makefile
    // ImpostorCapture.vert.spv rule). The former ImpostorCaptureLegacy.vert
    // (a pass-through to a nonexistent vegetation.geom, zero in-tree
    // references) was deleted, not rebuilt.
    VkShaderModule vertShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/ImpostorCapture.vert.spv");
    VkShaderModule fragShader = app->getOrCreateShaderModule("shaders/renderer/vegetation/ImpostorCapture.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShader;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription bindingDescs[2]{};
    bindingDescs[0] = { 0, sizeof(Vertex),    VK_VERTEX_INPUT_RATE_VERTEX   };
    bindingDescs[1] = { 1, sizeof(float) * 4, VK_VERTEX_INPUT_RATE_INSTANCE };

    // Attribute list matching VegetationRenderer.vert (tangent at ATTR_COLOR).
    std::vector<VkVertexInputAttributeDescription> attrs = {
        VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
        VkVertexInputAttributeDescription{ ATTR_COLOR, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
        VkVertexInputAttributeDescription{ ATTR_UV,  0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, texCoord) },
        VkVertexInputAttributeDescription{ ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) },
        VkVertexInputAttributeDescription{ ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 },
    };

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 2;
    vertexInput.pVertexBindingDescriptions      = bindingDescs;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertexInput.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

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

    // Three color blend attachments (albedo, normal, depth).
    VkPipelineColorBlendAttachmentState blendAtts[3]{};
    for (auto& b : blendAtts) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                         | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        b.blendEnable    = VK_FALSE;
    }
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 3;
    colorBlend.pAttachments    = blendAtts;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.size       = sizeof(CapturePC);

    VkDescriptorSetLayout layouts[3] = { uboDescSetLayout, texDescSetLayout, sharedVegRenderer->getWindParamsDescSetLayout() };
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 3;
    layoutInfo.pSetLayouts            = layouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &capturePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: pipelineLayout creation failed");
    app->resources.addPipelineLayout(capturePipelineLayout, "ImpostorCapture: capturePipelineLayout");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
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
    VkFormat colorFormats[3] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT };
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 3;
    renderingInfo.pColorAttachmentFormats = colorFormats;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    pipelineInfo.subpass             = 0;
    if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &pipelineInfo, nullptr, &capturePipeline) != VK_SUCCESS)
        throw std::runtime_error("ImpostorCapture: pipeline creation failed");
    app->resources.addPipeline(capturePipeline, "ImpostorCapture: capturePipeline");

    // Shader modules are cached by VulkanApp — kept alive for app lifetime.
}

void ImpostorCapture::createUBO(VulkanApp* app) {
    const VkDeviceSize totalSize = NUM_VIEWS * uboStride;
    // createBuffer registers buffer and memory in resources internally.
    Buffer buf = app->createBuffer(totalSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uboBuffer = buf.buffer;
    uboMapped = buf.mappedData;
}

void ImpostorCapture::createCaptureBuffers(VulkanApp* app) {

    // 24 corner vertices for the 6-plane billboard mesh (same layout as vegetation).
    const auto& baseTangents = RendererUtils::kBillboardBaseTangents;
    const auto& outwardDirs = RendererUtils::kBillboardOutwardDirs;
    const glm::vec3 worldUp(0,1,0);
    constexpr float hs = 0.5f, h = 1.0f, tilt = 1.0f;

    std::vector<Vertex> verts(24);
    for (int p = 0; p < 6; ++p) {
        glm::vec3 tangent = baseTangents[p];
        glm::vec3 outward = (p < 4) ? outwardDirs[p] : glm::vec3(0.0f);
        int base = p * 4;
        auto corner = [&](int ci, glm::vec3 off, glm::vec2 uv) {
            verts[base + ci].position = off;
            verts[base + ci].color = tangent;
            verts[base + ci].texCoord = uv;
            verts[base + ci].brushIndex = (p << 8) | ci;
        };
        corner(0, -tangent * hs,                    glm::vec2(0,1));
        corner(1,  tangent * hs,                    glm::vec2(1,1));
        corner(2, -tangent * hs + worldUp * h + outward * tilt, glm::vec2(0,0));
        corner(3,  tangent * hs + worldUp * h + outward * tilt, glm::vec2(1,0));
    }
    Buffer vb = app->createDeviceLocalBuffer(verts.data(), verts.size() * sizeof(Vertex),
                                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    captureVertBuf = vb.buffer;

    // 36-index triangle list.
    std::vector<uint32_t> idx(36);
    for (int p = 0; p < 6; ++p) {
        int b = p * 4;
        int ib = p * 6;
        idx[ib + 0] = b + 0; idx[ib + 1] = b + 1; idx[ib + 2] = b + 2;
        idx[ib + 3] = b + 1; idx[ib + 4] = b + 3; idx[ib + 5] = b + 2;
    }
    Buffer idxBuf = app->createDeviceLocalBuffer(idx.data(), idx.size() * sizeof(uint32_t),
                                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    captureIdxBuf = idxBuf.buffer;
    captureIdxCount = 36;

    // Instance buffer: one vec4 slot per billboard type (xyz=world pos,
    // w=billboardIndex), host-visible. Draws select the slot with firstInstance
    // (see capture()/captureAllAsync), so staged per-type data never aliases.
    // Needs VERTEX_BUFFER_BIT because it's bound as vertex buffer binding 1.
    Buffer ib = app->createBuffer(NUM_BILLBOARD_TYPES * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    captureInstBuf = ib.buffer;
    captureInstMapped = ib.mappedData;
}

void ImpostorCapture::createCaptureInvVPBuffer(VulkanApp* app) {
    const VkDeviceSize totalSize = TOTAL_LAYERS * sizeof(glm::mat4);
    Buffer buf = app->createBuffer(totalSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    captureInvVPBuffer = buf.buffer;
    captureInvVPMapped = buf.mappedData;
}

void ImpostorCapture::createSceneSampler(VulkanApp* app) {
    sceneSampler = app->createSamplerLinearClamp("ImpostorCapture: sceneSampler");
}

void ImpostorCapture::allocateDescSets(VulkanApp* app) {
    DescriptorAllocator descAlloc{app->getDevice(), app};

    VkDescriptorPoolSize poolSizesIC[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}
    };
    descriptorPool = descAlloc.createPool(
        poolSizesIC, 2, 2,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "ImpostorCapture: descriptorPool");

    VkDescriptorSetLayout layouts[2] = { uboDescSetLayout, texDescSetLayout };
    VkDescriptorSet sets[2];
    descAlloc.allocateSets(descriptorPool, 2, layouts, sets);
    uboDescSet = sets[0];
    texDescSet = sets[1];

    // Write UBO descriptor (DYNAMIC: range = one slot, offset supplied per-draw).
    DescriptorWriter(app->getDevice())
        .writeBuffer(uboDescSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                     uboBuffer, 0, sizeof(CaptureUBO))
        .flush();
    // Texture descriptor is written in updateTexDescSet() at capture time.
    // Set=2 wind params descriptor is shared with the VegetationRenderer.
}

void ImpostorCapture::updateTexDescSet(VkDevice device,
                                       VkImageView albedo, VkImageView normal,
                                       VkImageView opacity, VkSampler sampler) {
    DescriptorWriter writer(device);
    writer.writeImage(texDescSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      sampler, albedo, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writer.writeImage(texDescSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      sampler, normal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writer.writeImage(texDescSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      sampler, opacity, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writer.flush();
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
        si.anisotropyEnable = VK_FALSE;
        si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        si.unnormalizedCoordinates = VK_FALSE;
        si.compareEnable = VK_FALSE;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        imguiSampler = app->createSampler(si, "ImpostorCapture: imguiSampler");
    }
    const uint32_t layerBase = billboardType * NUM_VIEWS;
    for (uint32_t v = 0; v < NUM_VIEWS; ++v) {
        const uint32_t layerIdx = layerBase + v;
        imguiDescSets[layerIdx] = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            imguiSampler, captureLayerViews[layerIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        imguiNormalDescSets[layerIdx] = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            imguiSampler, captureNormalLayerViews[layerIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (captureDepthLayerViews[layerIdx] != VK_NULL_HANDLE) {
            imguiDepthDescSets[layerIdx] = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
                imguiSampler, captureDepthLayerViews[layerIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
}

void ImpostorCapture::recreateAllImGuiDescSets(VulkanApp* app) {
    destroyImGuiDescSets();
    for (uint32_t bt = 0; bt < NUM_BILLBOARD_TYPES; ++bt) {
        createImGuiDescSetsForType(app, bt);
    }
}

void ImpostorCapture::destroyImGuiDescSets() {
    for (auto& ds : imguiDescSets) {
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); ds = VK_NULL_HANDLE; }
    }
    for (auto& ds : imguiNormalDescSets) {
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); ds = VK_NULL_HANDLE; }
    }
    for (auto& ds : imguiDepthDescSets) {
        if (ds != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(ds); ds = VK_NULL_HANDLE; }
    }
}
