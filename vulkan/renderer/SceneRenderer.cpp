#include "SceneRenderer.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"


#include <stdexcept>
#include "../../utils/LocalScene.hpp"
#include "../includes/locations.hpp"
#include "../../math/ContainmentType.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <mutex>
#include <random>
#include <unordered_set>
#include <cstdlib>

namespace {
constexpr float kDebugSDFClip = 10.0f;

const uint32_t kDebugSDFFaces[6][4] = {
    {0, 1, 3, 2},
    {4, 6, 7, 5},
    {0, 4, 5, 1},
    {2, 3, 7, 6},
    {0, 2, 6, 4},
    {1, 5, 7, 3}
};

// Stable-slot pool capacities (pre-allocated GPU buffers, never reallocated).
//
// Every mesh-bearing octree node occupies one slot: each chunk (chunkLod 0)
// PLUS each of its ancestors up to the root (chunkLod 1..N). For a balanced
// octree the ancestor overhead over the chunk count is ~1/7 (~14%), but the
// pool must also absorb transient double-slotting: a rebuilt chunk holds its
// old slot until the new upload completes (deferred free), and editing a chunk
// re-dirties it together with all of its ancestors.
//
// Sizing is memory-bounded (target GPU: 4 GB integrated). Measured steady
// demand on the reference scene (full octree, works=658688):
//   solid  ~940 nodes (saturates a 1024 pool -> "no free slot"),
//   water  ~132 nodes (sparse),
//   brush  ~10  nodes.
// The budget is therefore REDISTRIBUTED: solid (the dense, dominant layer)
// gets generous headroom for future LoD depth / denser scenes, while water and
// brush are sized to a few times their observed peak. Per-slot cost: the
// slot's budget splits into STATIC per-level rows — the level-0 (finest) row
// keeps the full budget and each coarser level gets 1/4 of the previous
// (IndirectRenderer::initSlots), so a slot costs ~1.33x its level-0 budget:
//   solid slot: 1.332 MB vertex + 341 KB index ≈ 1.67 MB
//   water slot: same ≈ 1.67 MB
//   brush slot: 341 KB vertex + 85 KB index  ≈ 0.42 MB
//
// Measured post-trim peaks (full scene + brush rebuild, DEBUG logs):
//   solid ~416 slots, water ~160 slots, brush ~10 slots. The pools below hold
//   ~2.5x the observed peak while keeping the pre-allocated reservation under
//   1.6 GB — exceeding ~4 GB device-local caused radv to cancel the CS (device
//   lost) during the bulk chunk-upload burst on the 680M iGPU.
//   solid 1024 -> ~1.71 GB, water 192 -> ~320 MB, brush 64 -> ~27 MB
//   (total ≈ 2.05 GB, down from the 7.6 GB the per-(chunk, level) slot pools
//   reserved — that 5x oversize pool was the device-lost root cause)
//
// NOTE: slotted mode pre-allocates these buffers to capacity and never grows
// them at runtime (that is the point of the design — no global rebuilds). If a
// pool fills, "no free slot" is logged (with active/capacity) — bump the
// relevant constant. DEBUG builds log "slot peak N / capacity" as usage climbs.
// True runtime growth would require reallocating the buffers, which the design
// deliberately avoids.
constexpr uint32_t kMaxSolidChunkSlots = 1024;   // main solid (opaque) pool
constexpr uint32_t kMaxWaterChunkSlots = 192;   // main water (transparent) pool
constexpr uint32_t kMaxBrushChunkSlots = 64;     // brush preview pool

bool isDrawableSDF(float v) {
    return std::isfinite(v) && std::abs(v) <= kDebugSDFClip;
}

bool hasDrawableSDFFace(const std::array<float, 8>& sdf) {
    for (const auto& face : kDebugSDFFaces) {
        for (uint32_t corner : face) {
            if (isDrawableSDF(sdf[corner])) {
                return true;
            }
        }

        for (uint32_t edge = 0; edge < 4; ++edge) {
            const float a = sdf[face[edge]];
            const float b = sdf[face[(edge + 1) % 4]];
            if (std::isfinite(a) && std::isfinite(b) && ((a <= 0.0f && b >= 0.0f) || (a >= 0.0f && b <= 0.0f))) {
                return true;
            }
        }
    }
    return false;
}

void collectLeafSDFCubes(OctreeNode* node, const BoundingCube& cube, OctreeAllocator& allocator,
                         std::vector<DebugSDFRenderer::CubeSDF>& out) {
    if (!node) return;

    if (node->getLod() > 0u) {
        DebugSDFRenderer::CubeSDF debugCube{};
        debugCube.cube = cube;
        for (size_t i = 0; i < debugCube.sdf.size(); ++i) {
            debugCube.sdf[i] = node->sdf[i];
        }
        debugCube.brushIndex = node->vertex.brushIndex;
        if (hasDrawableSDFFace(debugCube.sdf)) {
            out.push_back(debugCube);
            return;  // Parent covers this subtree — children are redundant
        }
        // Parent is simplified but has no drawable faces; traverse children
        // in case individual child leaves still have visible SDF faces.
    }

    ChildBlock* block = node->getBlock(allocator);
    if (!block) return;
    for (uint32_t i = 0; i < 8; ++i) {
        OctreeNode* child = block->get(i, allocator);
        if (child && child != node) {
            collectLeafSDFCubes(child, cube.getChild(i), allocator, out);
        }
    }
}
}

void SceneRenderer::cleanup(VulkanApp* app) {
    // Tear down the async streaming engine FIRST, while the solid/water
    // IndirectRenderers (and their indirect/bounds buffers) are still alive:
    // UploadManager::destroy waits on in-flight transfers and fires each
    // pending onComplete, which publishes meta into those renderers.
    if (app) {
        streamer.destroy();
    }

    // Cleanup all sub-renderers to properly destroy GPU resources (app may be null)
    if (postProcessRenderer && app) {
        postProcessRenderer->cleanup(app);
    }
    if (mainLiquidRenderer && app) {
        mainLiquidRenderer->cleanup(app);
    }
    // Cleanup scene-owned water sub-renderers
    if (backFaceRenderer && app) {
        backFaceRenderer->cleanup(app);
    }
    if (brushBackFaceRenderer && app) {
        brushBackFaceRenderer->cleanup(app);
    }
    if (solid360Renderer && app) {
        solid360Renderer->cleanup(app);
    }
    destroyBrushRenderTargets(app);
    brushSolidIndirectRenderer.cleanup();
    brushLiquidIndirectRenderer.cleanup();
    if (mainSolidRenderer && app) {
        mainSolidRenderer->cleanup(app);
    }
    if (shadowMapper && app) {
        shadowMapper->cleanup(app);
    }
    if (skyRenderer) {
        skyRenderer->cleanup();
    }
    if (vegetationRenderer) {
        vegetationRenderer->cleanup();
    }
    if (debugCubeRenderer) {
        debugCubeRenderer->cleanup();
    }
    if (boundingBoxRenderer) {
        boundingBoxRenderer->cleanup();
    }
    if (debugSDFRenderer) {
        debugSDFRenderer->cleanup();
    }
    if (solidWireframe) {
        solidWireframe->cleanup();
    }
    if (waterWireframe) {
        waterWireframe->cleanup();
    }

    // Clear local CPU-side handles; Vulkan objects are destroyed via VulkanResourceManager
    for (auto &b : mainUniformBuffers) {
        if (b.buffer != VK_NULL_HANDLE) b = {};
    }
    mainUniformBuffers.clear();
    for (auto &b : uboStagingBuffers) {
        if (b.buffer != VK_NULL_HANDLE) b = {};
    }
    uboStagingBuffers.clear();

    if (mainPassUBO.buffer.buffer != VK_NULL_HANDLE) {
        mainPassUBO.buffer = {};
    }
    if (shadowPassUBO.buffer.buffer != VK_NULL_HANDLE) {
        shadowPassUBO.buffer = {};
    }
    if (waterPassUBO.buffer.buffer != VK_NULL_HANDLE) {
        waterPassUBO.buffer = {};
    }
}

void SceneRenderer::stopGenPools() {
    brushWaterGenPool.stop();
    brushSolidGenPool.stop();
    mainSolidGenPool.stop();
    mainWaterGenPool.stop();
}

void SceneRenderer::createBrushRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!app) return;
    VkDevice device = app->getDevice();
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkImageView& view) {
        VkDeviceMemory dummyMem = VK_NULL_HANDLE;
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "SceneRenderer: brush", image, allocation, dummyMem, view);
    };
    VkFormat colorFormat = app->getSwapchainImageFormat();
    for (uint32_t i = 0; i < SceneRenderer::BRUSH_FRAMES; ++i) {
        createImage(colorFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    brushColorImages[i], brushColorAllocations[i], brushColorImageViews[i]);
        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    brushDepthImages[i], brushDepthAllocations[i], brushDepthImageViews[i]);
    }
}

void SceneRenderer::destroyBrushRenderTargets(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (uint32_t i = 0; i < SceneRenderer::BRUSH_FRAMES; ++i) {
        if (brushColorImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(brushColorImageViews[i]))
                vkDestroyImageView(device, brushColorImageViews[i], nullptr);
            brushColorImageViews[i] = VK_NULL_HANDLE;
        }
        if (brushColorImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(brushColorImages[i], brushColorAllocations[i], VK_NULL_HANDLE);
            brushColorImages[i] = VK_NULL_HANDLE;
            brushColorAllocations[i] = VK_NULL_HANDLE;
        }
        if (brushDepthImageViews[i] != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(brushDepthImageViews[i]))
                vkDestroyImageView(device, brushDepthImageViews[i], nullptr);
            brushDepthImageViews[i] = VK_NULL_HANDLE;
        }
        if (brushDepthImages[i] != VK_NULL_HANDLE) {
            app->destroyImageWithVma(brushDepthImages[i], brushDepthAllocations[i], VK_NULL_HANDLE);
            brushDepthImages[i] = VK_NULL_HANDLE;
            brushDepthAllocations[i] = VK_NULL_HANDLE;
        }
    }
}

void SceneRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    // Recreate offscreen targets that depend on swapchain size
    if (mainSolidRenderer) {
        mainSolidRenderer->createRenderTargets(app, width, height);
    }
    destroyBrushRenderTargets(app);
    createBrushRenderTargets(app, width, height);
    if (mainLiquidRenderer) {
        mainLiquidRenderer->createRenderTargets(app, width, height);
        // Recreate back-face and 360 reflection targets owned by SceneRenderer
        if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, width, height);
        if (brushBackFaceRenderer) brushBackFaceRenderer->createRenderTargets(app, width, height);
        if (solid360Renderer) {
            solid360Renderer->destroySolid360Targets(app);
            solid360Renderer->createSolid360Targets(app, mainLiquidRenderer->getLinearSampler());
            solid360Renderer->createSolid360Pipelines(app);
            // Rewrite binding 11 (cubemap) in all descriptor sets since the
            // old VkImageView handles were destroyed and new ones created.
            VkImageView cubeView = solid360Renderer->getSolid360View();
            VkSampler cubeSampler = solid360Renderer->getSolid360Sampler();
            if (cubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE) {
                VkDescriptorSet staticDs = app->getStaticDescriptorSet();
                if (staticDs != VK_NULL_HANDLE) {
                    DescriptorWriter(app->getDevice())
                        .writeImage(staticDs, 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .flush();
                }
                // Propagate to per-frame descriptor sets
                for (size_t fi = 0; fi < app->getMainDescriptorSetCount(); ++fi) {
                    VkDescriptorSet dstSet = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));
                    if (dstSet == VK_NULL_HANDLE) continue;
                    VkCopyDescriptorSet c{};
                    c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
                    c.srcSet = staticDs; c.srcBinding = 11; c.srcArrayElement = 0;
                    c.dstSet = dstSet; c.dstBinding = 11; c.dstArrayElement = 0;
                    c.descriptorCount = 1;
                    vkUpdateDescriptorSets(app->getDevice(), 0, nullptr, 1, &c);
                }
                // Propagate to shadow descriptor sets
                for (size_t fi = 0; fi < shadowDescriptorSets.size(); ++fi) {
                    VkDescriptorSet ds = shadowDescriptorSets[fi];
                    if (ds == VK_NULL_HANDLE) continue;
                    DescriptorWriter(app->getDevice())
                        .writeImage(ds, 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .flush();
                }
            }
        }
    }
    if (postProcessRenderer) {
        postProcessRenderer->setRenderSize(width, height);
    }
    if (skyRenderer) {
        skyRenderer->destroyOffscreenTargets(app);
        skyRenderer->createOffscreenTargets(app, width, height);
    }
    // Brush images have been destroyed and recreated; reset tracked layouts so
    // the first barrier after resize uses VK_IMAGE_LAYOUT_UNDEFINED as oldLayout.
    brushColorLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    brushDepthLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    // Rewrite brush depth descriptors after recreating brush targets
    writeBrushDepthDescriptors(app);
}

void SceneRenderer::writeBrushDepthDescriptors(VulkanApp* app) {
    VkSampler brushDepthSampler = VK_NULL_HANDLE;
    if (mainLiquidRenderer) {
        brushDepthSampler = mainLiquidRenderer->getLinearSampler();
    }
    if (brushDepthSampler == VK_NULL_HANDLE && shadowMapper) {
        brushDepthSampler = shadowMapper->getShadowMapSampler();
    }

    for (size_t fi = 0; fi < brushDepthDescriptorSets.size(); ++fi) {
        VkDescriptorSet dstSet = brushDepthDescriptorSets[fi];
        if (dstSet == VK_NULL_HANDLE) continue;
        VkImageView brushFrontView = getBrushDepthView(static_cast<uint32_t>(fi));
        VkImageView brushBackView = VK_NULL_HANDLE;
        if (brushBackFaceRenderer) {
            brushBackView = brushBackFaceRenderer->getBackFaceDepthView(static_cast<uint32_t>(fi));
        }
        if (brushDepthSampler == VK_NULL_HANDLE) continue;

        DescriptorWriter writer(app->getDevice());
        if (brushFrontView != VK_NULL_HANDLE) {
            writer.writeImage(dstSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              brushDepthSampler, brushFrontView,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (brushBackView != VK_NULL_HANDLE) {
            writer.writeImage(dstSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              brushDepthSampler, brushBackView,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        writer.flush();
    }
}

SceneRenderer::SceneRenderer() :
    shadowMapper(std::make_unique<ShadowRenderer>(2048)),
    mainLiquidRenderer(std::make_unique<WaterRenderer>()),
    postProcessRenderer(std::make_unique<PostProcessRenderer>()),
    skyRenderer(std::make_unique<SkyRenderer>()),
    mainSolidRenderer(std::make_unique<SolidRenderer>()),
    vegetationRenderer(std::make_unique<VegetationRenderer>()),
    debugCubeRenderer(std::make_unique<DebugCubeRenderer>()),
    boundingBoxRenderer(std::make_unique<DebugCubeRenderer>()),
    debugSDFRenderer(std::make_unique<DebugSDFRenderer>()),
    solidWireframe(std::make_unique<WireframeRenderer>()),
    waterWireframe(std::make_unique<WireframeRenderer>()),
      skySettings(std::make_unique<SkySettings>())
{

}

SceneRenderer::~SceneRenderer() {
    // Do not attempt Vulkan cleanup here (app is not available). The owner
    // (MyApp) must call `sceneRenderer->cleanup(app)` before destroying the
    // VulkanApp instance.
}

void SceneRenderer::shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, Buffer &mainUniformBuffer, const UniformObject &uboStatic, bool shadowsEnabled, bool renderSolid, bool vegetationEnabled, bool shadowTessellationEnabled, float lodBias) {
    if (commandBuffer == VK_NULL_HANDLE) return;
    if (!shadowsEnabled) return;

    // Render each cascade: upload light-space UBO, draw scene, restore UBO
    const glm::mat4 cascadeMatrices[SHADOW_CASCADE_COUNT] = {
        uboStatic.lightSpaceMatrix,
        uboStatic.lightSpaceMatrix1,
        uboStatic.lightSpaceMatrix2
    };

    // Single cascade-aware GPU culling pass: culls all chunks against all 3
    // cascade frustums simultaneously.  Each cascade independently receives
    // every chunk visible in its frustum — no exclusion between cascades —
    // so the fragment shader's per-cascade sampling always finds the geometry
    // it needs. The cascade cull reads the per-chunk LoD selection the main
    // pass stamped into the shared visibleLods buffer (single source of truth),
    // so shadow draws use the exact same LoD as the main pass.
    mainSolidRenderer->getIndirectRenderer().prepareCullCascades(commandBuffer, cascadeMatrices, lastCameraPos_, lodBias);
    // Water shadows share the same LoD sync: the water cascade cull reads the
    // water main pass's visibleLods (the water prepareCull ran before this
    // shadow pass, so the selection is fresh for the current frame).
    if (mainLiquidRenderer) {
        mainLiquidRenderer->getIndirectRenderer().prepareCullCascades(commandBuffer, cascadeMatrices, lastCameraPos_, lodBias);
    }

    // Acquire vegetation instance/indirect buffers before dynamic rendering
    if (vegetationEnabled && vegetationRenderer) {
        vegetationRenderer->recordReadBarriers(commandBuffer);
        vegetationRenderer->prepareCullCascades(commandBuffer, cascadeMatrices);
    }

    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        glm::mat4 lsMatrix = cascadeMatrices[c];

        // Upload a shadow-specific UBO: viewProjection = cascade lightSpaceMatrix.
        // passParams.x MUST be 0 so the TES computes fragPosWorld (needed by the EVSM
        // fragment shader to produce correct moments).  With passParams.x=1 the TES
        // outputs fragPosWorld = vec3(0) → EVSM gets garbage depth → no shadows.
        UniformObject shadowUBO = uboStatic;
        shadowUBO.viewProjection = lsMatrix;
        shadowUBO.passParams.x = 0.0f;
        shadowUBO.passParams.y = shadowTessellationEnabled ? 1.0f : 0.0f;

        // Wait for previous cascade draws to finish reading the UBO
        // before overwriting it via vkCmdCopyBuffer.
        {
            VkBufferMemoryBarrier2 preBarrier{};
            preBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            preBarrier.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
            preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preBarrier.buffer = mainUniformBuffer.buffer;
            preBarrier.offset = 0;
            preBarrier.size = VK_WHOLE_SIZE;
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &preBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        }

        // Upload shadow UBO via vkCmdCopyBuffer from persistently mapped staging
        // buffer (avoids vkCmdUpdateBuffer's implicit FULL_QUEUE barrier).
        VkDeviceSize stagingOff = static_cast<VkDeviceSize>(c) * sizeof(UniformObject);
        if (frameIdx < uboStagingBuffers.size()) {
            memcpy(uboStagingBuffers[frameIdx].map(stagingOff), &shadowUBO, sizeof(UniformObject));
            VkBufferCopy copy{ stagingOff, 0, sizeof(UniformObject) };
            vkCmdCopyBuffer(commandBuffer, uboStagingBuffers[frameIdx].buffer, mainUniformBuffer.buffer, 1, &copy);
        }
        {
            VkBufferMemoryBarrier2 memBarrier{};
            memBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
            memBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            memBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            memBarrier.buffer = mainUniformBuffer.buffer;
            memBarrier.offset = 0;
            memBarrier.size = VK_WHOLE_SIZE;
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &memBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        }

        // Cascade-specific draw (no per-cascade cull — already handled above)
        shadowMapper->beginShadowPass(app, commandBuffer, c, lsMatrix);

        // Bind shadow descriptor set (uses dummy depth at bindings 4,8,9)
        VkPipelineLayout layout = shadowMapper->getShadowPipelineLayout();
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!shadowDescriptorSets.empty()) {
            uint32_t idx = frameIdx % static_cast<uint32_t>(shadowDescriptorSets.size());
            ds = shadowDescriptorSets[idx];
        }
        if (layout != VK_NULL_HANDLE && ds != VK_NULL_HANDLE) {
            frameCmdState.bindGraphicsDescriptorSets(commandBuffer, layout, 0, 1, &ds, 0, nullptr);
        }

        // Bind the EVSM shadow pipeline (shared by the solid and water depth
        // draws; both use the same Vertex format and indexed-indirect draws).
        VkPipeline solidShadowPipeline = shadowMapper->getShadowPipeline();
        if (solidShadowPipeline != VK_NULL_HANDLE) {
            frameCmdState.bindGraphicsPipeline(commandBuffer, solidShadowPipeline);
        }

        // Draw solid geometry into shadow map (can be toggled off to isolate
        // vegetation shadows for debugging).
        if (renderSolid) {
            auto& shadowIR = mainSolidRenderer->getIndirectRenderer();
            shadowIR.bindBuffers(commandBuffer);
            shadowIR.drawCascadeOnly(commandBuffer, c);
        }

        // Draw water geometry into the shadow map so water casts shadows at the
        // same LoD as the main pass (the water cascade cull read the shared
        // visibleLods selection). Reuses the same EVSM shadow pipeline.
        if (mainLiquidRenderer) {
            auto& waterShadowIR = mainLiquidRenderer->getIndirectRenderer();
            waterShadowIR.bindBuffers(commandBuffer);
            waterShadowIR.drawCascadeOnly(commandBuffer, c);
        }

        // Vegetation shadow pass: drawn after solid so its 2-buffer vertex
        // bindings don't leak into the solid draw. Uses cascade-aware culling
        // (prepareCullCascades dispatched above).
        if (vegetationEnabled && vegetationRenderer) {
            const glm::vec3 cameraPos = glm::vec3(uboStatic.viewPos);
            vegetationRenderer->drawShadowCascade(app, commandBuffer, ds, cameraPos, c);
        }

        shadowMapper->endShadowPass(app, commandBuffer, c);

        // Apply separable Gaussian blur (EVSM moment filtering) to reduce noise.
        // Skip the smallest cascade: at 512x512 the 3-tap blur is barely visible
        // and skipping it saves two fullscreen draws plus four layout transitions.
        if (c < SHADOW_CASCADE_COUNT - 1) {
            shadowMapper->blurCascade(app, commandBuffer, c);
        }
    }

    // Restore GPU culling for the main camera frustum (was overwritten by
    // per-cascade prepareCull calls above) so drawPrepared in the main pass
    // uses the correct visible set.
    mainSolidRenderer->getIndirectRenderer().prepareCull(commandBuffer, uboStatic.viewProjection, lastCameraPos_, lodBias);
    brushSolidIndirectRenderer.prepareCull(commandBuffer, uboStatic.viewProjection, lastCameraPos_, lodBias);

    // Restore the main UBO so subsequent passes see the original data.
    // Wait for all shadow cascade draws to finish reading the UBO first.
    {
        VkBufferMemoryBarrier2 preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        preBarrier.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.buffer = mainUniformBuffer.buffer;
        preBarrier.offset = 0;
        preBarrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }

    // Restore main UBO via vkCmdCopyBuffer from persistently mapped staging buffer.
    VkDeviceSize restoreOff = static_cast<VkDeviceSize>(SHADOW_CASCADE_COUNT) * sizeof(UniformObject);
    if (frameIdx < uboStagingBuffers.size()) {
        memcpy(uboStagingBuffers[frameIdx].map(restoreOff), &uboStatic, sizeof(UniformObject));
        VkBufferCopy copy{ restoreOff, 0, sizeof(UniformObject) };
        vkCmdCopyBuffer(commandBuffer, uboStagingBuffers[frameIdx].buffer, mainUniformBuffer.buffer, 1, &copy);
    }
    {
        VkBufferMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        memBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memBarrier.buffer = mainUniformBuffer.buffer;
        memBarrier.offset = 0;
        memBarrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }
}

void SceneRenderer::mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, bool hasWater, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool renderSolid, bool wireframeEnabled, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent) {
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[SceneRenderer::mainPass] commandBuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    if (renderSolid) {
        VkDescriptorSet brushDepthSet = getBrushDepthDescriptorSet(frameIdx);
        mainSolidRenderer->renderDepthPrepass(commandBuffer, app, perTextureDescriptorSet, brushDepthSet);
        mainSolidRenderer->render(commandBuffer, app, perTextureDescriptorSet, brushDepthSet);
        if (wireframeEnabled) {
            solidWireframe->draw(commandBuffer, app, {perTextureDescriptorSet}, mainSolidRenderer->getIndirectRenderer());
        } 
    }
    
}

void SceneRenderer::skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj) {
    SkySettings::Mode mode = skySettings->mode;
    skyRenderer->render(app, commandBuffer, perTextureDescriptorSet, mainUniformBuffer, uboStatic, viewProj, mode);
}

void SceneRenderer::drawSolidWireframeOverlay(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled) {
    solidWireframe->draw(commandBuffer, app, {perTextureDescriptorSet}, mainSolidRenderer->getIndirectRenderer());
}

void SceneRenderer::waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, bool waterWireframeEnabled, float waterTime, VkImageView skyView) {
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[SceneRenderer::waterPass] commandBuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    // Update the water render UBO with the active layer time value.
    if (waterRenderUBOBuffer_.buffer != VK_NULL_HANDLE) {
        WaterRenderUBO renderUbo{};
        renderUbo.timeParams = glm::vec4(waterTime, 0.0f, 0.0f, 0.0f);
        void* data = nullptr;
        data = waterRenderUBOBuffer_.map(0);
        memcpy(data, &renderUbo, sizeof(WaterRenderUBO));
        waterRenderUBOBuffer_.unmap(); // VMA persistent mapping
    }

    // Delegate water offscreen work to WaterRenderer — record on the same
    // command buffer so the solid pass outputs are available for sampling.
    VkImageView sceneColorView = mainSolidRenderer->getColorView(frameIdx);
    VkImageView sceneDepthView = mainSolidRenderer->getDepthView(frameIdx);
    // (Re)allocate and update this slot's scene-texture descriptor set (set 2,
    // binding 1 = sceneDepthTex) here on the main command buffer, immediately
    // before any draw that binds it. The async back-face task uses its OWN
    // per-task set, so this slot's set is only ever referenced by the main
    // command buffer and is freed/reallocated once its in-flight fence signals
    // (no VUID-03047, and GPU-assisted validation sees it populated — no
    // UPDATE_AFTER_BIND needed).
    {
        VkImageView wBack = (backFaceRenderer) ? backFaceRenderer->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
        VkImageView wCube = (solid360Renderer) ? solid360Renderer->getSolid360View() : VK_NULL_HANDLE;
        mainLiquidRenderer->prepareSceneTexturesForFrame(app, frameIdx, sceneColorView, sceneDepthView,
                                                    skyView, wBack, wCube);
    }

    bool _wg_env_skip = envDisableWaterGeom;

    // Scene textures were already bound before the async back-face/solid360 tasks were
    // launched (see main.cpp), so we must NOT call updateSceneTexturesBinding here.
    // Calling it after the async tasks submit their command buffers would update a
    // descriptor set that is already referenced by a pending command buffer
    // (VUID-vkUpdateDescriptorSets-None-03047).

    bool wf = waterWireframeEnabled;
    if (wf && waterWireframe && waterWireframe->getPipeline() != VK_NULL_HANDLE) {
        // Wireframe path: use WaterRenderer for setup/pass management,
        // but bind the wireframe pipeline instead of the normal one.
        if (!_wg_env_skip) {
            mainLiquidRenderer->prepareRender(app, commandBuffer, frameIdx, sceneColorView, sceneDepthView, skyView);
            mainLiquidRenderer->beginWaterGeometryPass(commandBuffer, frameIdx);

            // First render filled water geometry to populate the water depth
            // buffer so the wireframe can depth-test against actual water depth.
            VkPipeline waterPipe = mainLiquidRenderer->getWaterGeometryPipeline();
            VkPipelineLayout waterLayout = mainLiquidRenderer->getWaterGeometryPipelineLayout();
            if (waterPipe != VK_NULL_HANDLE && waterLayout != VK_NULL_HANDLE) {
                frameCmdState.bindGraphicsPipeline(commandBuffer, waterPipe);

                VkDescriptorSet mainDs = app->getMainDescriptorSet();
                if (mainDs != VK_NULL_HANDLE) {
                    frameCmdState.bindGraphicsDescriptorSets(commandBuffer, waterLayout, 0, 1, &mainDs, 0, nullptr);
                }

                VkDescriptorSet sceneDs = mainLiquidRenderer->getWaterDepthDescriptorSet(frameIdx);
                if (sceneDs != VK_NULL_HANDLE) {
                    frameCmdState.bindGraphicsDescriptorSets(commandBuffer, waterLayout, 2, 1, &sceneDs, 0, nullptr);
                }

                // Draw filled water geometry (will update depth buffer)
                mainLiquidRenderer->getIndirectRenderer().drawPrepared(commandBuffer);
                brushLiquidIndirectRenderer.drawPrepared(commandBuffer);
            }

            // Draw wireframe overlay on top, inside the same render pass,
            // reusing the depth buffer populated by the filled geometry pass.
            // Bind descriptor sets individually with null checks (same pattern
            // as the filled water pipeline) to handle missing sets gracefully.
            VkPipeline waterWfPipe = waterWireframe->getPipeline();
            VkPipelineLayout wfLayout = waterWireframe->getPipelineLayout();
            if (waterWfPipe != VK_NULL_HANDLE && wfLayout != VK_NULL_HANDLE) {
                frameCmdState.bindGraphicsPipeline(commandBuffer, waterWfPipe);

                VkDescriptorSet wfMainDs = app->getMainDescriptorSet();
                if (wfMainDs != VK_NULL_HANDLE)
                    frameCmdState.bindGraphicsDescriptorSets(commandBuffer, wfLayout, 0, 1, &wfMainDs, 0, nullptr);

                VkDescriptorSet wfDepthDs = mainLiquidRenderer->getWaterDepthDescriptorSet(frameIdx);
                if (wfDepthDs != VK_NULL_HANDLE)
                    frameCmdState.bindGraphicsDescriptorSets(commandBuffer, wfLayout, 2, 1, &wfDepthDs, 0, nullptr);

                mainLiquidRenderer->getIndirectRenderer().drawPrepared(commandBuffer);
                brushLiquidIndirectRenderer.drawPrepared(commandBuffer);
            }

            mainLiquidRenderer->endWaterGeometryPass(commandBuffer);
        } else {
            // Skipping water geometry operations as requested by env guard
        }
    } else {
        if (!_wg_env_skip) {
            mainLiquidRenderer->render(app, commandBuffer, frameIdx, sceneColorView, sceneDepthView, skyView,
                                  &brushLiquidIndirectRenderer);
        } else {
            // Skipping waterRenderer::render due to VULKAN_DISABLE_WATERGEOM
        }
    }

    // Post-processing should run inside the active main render pass; caller (e.g. MyApp::draw) should invoke
    // `postProcessRenderer->render` with valid scene/water views when available. Keep this function focused
    // on executing offscreen geometry and returning control to the main pass.
}

void SceneRenderer::init(VulkanApp* app, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterParams>& waterParams) {
    if (!app) {
        std::cerr << "[SceneRenderer::init] app is nullptr!" << std::endl;
        return;
    }

    // Cache env-var flags once at startup instead of per-frame getenv() calls
    envDisableWaterGeom = (std::getenv("VULKAN_DISABLE_WATERGEOM") != nullptr);

    // Initialize the async streaming orchestrator. It is now the real transfer
    // engine: solid/water incremental chunk uploads route through it (K
    // concurrent staging slots, no per-frame cap) instead of the single-slot
    // IndirectRenderer pendingTransfer path. slotSize = chunkVertexBytes +
    // chunkIndexBytes; a chunk mesh larger than one slot falls back to the
    // renderer's legacy ring-backed staging path automatically.
    streamer.init(app,
                  /*chunkVertexBytes*/ 1u << 20,
                  /*chunkIndexBytes*/  1u << 20,
                  /*stagingSlots*/     16,
                  /*initialChunkSlots*/ 8,
                  /*workersPerCategory*/ 2);

    // Route solid/water IndirectRenderer incremental copies through the manager.
    mainSolidRenderer->getIndirectRenderer().setUploadManager(
        &streamer.uploadManager(), streaming::StreamCategory::Solid);
    mainLiquidRenderer->getIndirectRenderer().setUploadManager(
        &streamer.uploadManager(), streaming::StreamCategory::Water);
    // Initialize the separate brush solid/liquid IndirectRenderers (no
    // streamer — brush meshes are small and infrequent; upload fits in the
    // legacy ring path). Both are dedicated pools: brush geometry never shares
    // the main scene IRs.
    brushSolidIndirectRenderer.init();
    brushLiquidIndirectRenderer.init();

    // skySettingsRef was initialized at construction and must be valid
    

    // Bind external texture arrays if provided; allocation/initialization should be done by the application
    if (vegetationRenderer) {
        if (textureArrayManager) {
            vegetationRenderer->setTextureArrayManager(textureArrayManager, app);
            vegetationRenderer->init();
        } else {
            std::cerr << "[SceneRenderer::init] No TextureArrayManager provided — vegetation renderer initialization deferred" << std::endl;
        }
    }
    
    mainSolidRenderer->init();
    mainSolidRenderer->destroyRenderTargets(app);
    mainSolidRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    mainSolidRenderer->createPipelines(app);

    // Create pipelines for all renderers (solid renderer now has its render pass ready)
    skyRenderer->init(app);
    // Create offscreen sky targets (destroy old first to prevent handle leak)
    skyRenderer->destroyOffscreenTargets(app);
    skyRenderer->createOffscreenTargets(app, app->getWidth(), app->getHeight());
    createBrushRenderTargets(app, app->getWidth(), app->getHeight());
    shadowMapper->init(app);
    vegetationRenderer->init(app);

    // Initialize debug cube renderer
    if (debugCubeRenderer) {
        debugCubeRenderer->init(app);
    }
    // Initialize bounding box renderer (reuses cube wireframe pipeline)
    if (boundingBoxRenderer) {
        boundingBoxRenderer->init(app);
    }
    if (debugSDFRenderer) {
        debugSDFRenderer->init(app);
    }
    
    // Create per-frame main uniform buffers (TRANSFER_DST for vkCmdCopyBuffer from staging)
    size_t dsCount = app->getMainDescriptorSetCount();
    if (dsCount == 0) dsCount = 1;
    mainUniformBuffers.clear();
    mainUniformBuffers.resize(dsCount);
    for (size_t i = 0; i < dsCount; ++i) {
        mainUniformBuffers[i] = app->createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    // Per-frame staging buffers for GPU-timeline UBO uploads
    VkDeviceSize stagingSize = sizeof(UniformObject) * (SHADOW_CASCADE_COUNT + 1);
    uboStagingBuffers.clear();
    uboStagingBuffers.resize(dsCount);
    for (size_t i = 0; i < dsCount; ++i) {
        uboStagingBuffers[i] = app->createBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    VkDescriptorSet mainDs = app->getMainDescriptorSetForFrame(0);

    // Initialize sky renderer with our owned settings now that descriptor sets are ready.
    // Write Sky UBO to the static descriptor set once; all per-frame descriptor sets
    // will get binding 6 via the copy loop below.
    if (skyRenderer) {
        VkDescriptorSet staticDs = app->getStaticDescriptorSet();
        if (staticDs != VK_NULL_HANDLE) {
            skyRenderer->init(app, *skySettings, staticDs);
        } else {
            skyRenderer->init(app, *skySettings, mainDs);
            for (uint32_t i = 1; i < static_cast<uint32_t>(app->getMainDescriptorSetCount()); ++i) {
                VkDescriptorSet ds = app->getMainDescriptorSetForFrame(i);
                if (ds != VK_NULL_HANDLE) {
                    skyRenderer->init(app, *skySettings, ds);
                }
            }
        }
    }
    
    // Bind texture arrays, shadow maps, materials, sky, water params (bindings 1-13)
    // These static bindings are written once to the static descriptor set and then
    // copied into per-frame descriptor sets. Only binding 0 (per-frame UBO) is
    // written individually per frame.
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> writesImg;
    std::vector<VkDescriptorBufferInfo> writesBuf;
    writesImg.reserve(9);  // max image descriptors: 5 texture arrays + 3 shadow maps + 1 cubemap
    writesBuf.reserve(3);  // materials SSBO + water params + water render UBO

    // Helper to add image write if valid. dstSet is set to the static descriptor set
    // so the accumulated writes serve as a template for the static set.
    VkDescriptorSet staticDs = app->getStaticDescriptorSet();
    auto addImageWrite = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
        if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
            std::cerr << "[SceneRenderer::init] Skipping descriptor binding " << binding
                      << ": imageView=" << (void*)view
                      << " sampler=" << (void*)sampler << std::endl;
            return;
        }
        VkDescriptorImageInfo& info = writesImg.emplace_back();
        info.sampler = sampler;
        info.imageView = view;
        info.imageLayout = layout;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = staticDs;
        w.dstBinding = binding;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &info;
        writes.push_back(w);
    };

    if (textureArrayManager) {
        addImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImageWrite(12, textureArrayManager->roughnessSampler, textureArrayManager->roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImageWrite(13, textureArrayManager->aoSampler, textureArrayManager->aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        std::cerr << "[SceneRenderer::init] No TextureArrayManager set — skipping texture array descriptor writes" << std::endl;
    }
    addImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(8, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(9, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Create and bind Materials SSBO at binding 5. Require an external MaterialManager.
    materialManagerPtr = materialManager;
    if (!materialManager) {
        throw std::runtime_error("SceneRenderer::init requires a valid MaterialManager");
    }
    materialsBuffer = materialManager->getBuffer();
    if (materialsBuffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("MaterialManager provided but materials buffer is not allocated");
    }
    VkDescriptorBufferInfo& materialsInfo = writesBuf.emplace_back(materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
    VkWriteDescriptorSet materialsWrite{};
    materialsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    materialsWrite.dstSet = staticDs;
    materialsWrite.dstBinding = 5;
    materialsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    materialsWrite.descriptorCount = 1;
    materialsWrite.pBufferInfo = &materialsInfo;
    writes.push_back(materialsWrite);

    // Initialize WaterRenderer early and allocate a params SSBO sized to texture layers.
    // Use the passed vector of WaterParams as the source of truth for layer count.
    // Do not fall back to texture-array sizes; require explicit water parameters.
    uint32_t layerCount = waterParams.size();
    if (layerCount == 0) {
        throw std::runtime_error("SceneRenderer::init requires at least one WaterParams entry (no fallback allowed)");
    }
    
    size_t paramsBufferSize = sizeof(WaterParamsGPU) * static_cast<size_t>(layerCount);
    waterParamsBuffer_ = app->createBuffer(paramsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Create scene-owned water sub-renderers. Back-face renderpass must exist
    // before water pipelines are created, so create it first.
    backFaceRenderer = std::make_unique<WaterBackFaceRenderer>();
    brushBackFaceRenderer = std::make_unique<BrushBackFaceRenderer>();
    solid360Renderer = std::make_unique<Solid360Renderer>();

    // Initialize WaterRenderer (creates its pipeline layout and initializes the param SSBO)
    mainLiquidRenderer->init(app, waterParamsBuffer_, waterParams, layerCount);

    // Now that WaterRenderer has created its pipeline layout, allow the
    // back-face renderer to create pipelines that depend on it.
    if (backFaceRenderer) backFaceRenderer->createPipelines(app, mainLiquidRenderer->getWaterGeometryPipelineLayout());
    // Create back-face render targets early so their image views are
    // available before the first frame's water pass attempts to bind them.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    // Create brush back-face renderer: uses solid shaders (no water dependency)
    // and VK_COMPARE_OP_GREATER to capture the farthest back-face depth.
    if (brushBackFaceRenderer) {
        brushBackFaceRenderer->init(app);
        brushBackFaceRenderer->createPipelines(app);
        brushBackFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    }
    if (solid360Renderer) {
        solid360Renderer->init(app);
        solid360Renderer->setWaterRenderer(mainLiquidRenderer.get());
        // Create cubemap targets now so the image view is available for
        // the environment-map descriptor binding (binding 11) below.
        solid360Renderer->createSolid360Targets(app, mainLiquidRenderer->getLinearSampler());
        solid360Renderer->createSolid360Pipelines(app);
        // Binding 11: environment cubemap for solid-shader reflections
        VkImageView cubeView = solid360Renderer->getSolid360View();
        VkSampler cubeSampler = solid360Renderer->getSolid360Sampler();
        if (cubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE) {
            addImageWrite(11, cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    // Bind water params SSBO to binding 7 of main descriptor set.
    VkDescriptorBufferInfo& waterParamsInfo = writesBuf.emplace_back(waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);
    VkWriteDescriptorSet waterParamsWrite{};
    waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    waterParamsWrite.dstSet = staticDs;
    waterParamsWrite.dstBinding = 7;
    waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    waterParamsWrite.descriptorCount = 1;
    waterParamsWrite.pBufferInfo = &waterParamsInfo;
    writes.push_back(waterParamsWrite);

    // Bind water render UBO to binding 10 of main descriptor set
    waterRenderUBOBuffer_ = app->createBuffer(sizeof(WaterRenderUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDescriptorBufferInfo& waterRenderUBOInfo = writesBuf.emplace_back(waterRenderUBOBuffer_.buffer, 0, sizeof(WaterRenderUBO));
    VkWriteDescriptorSet waterRenderUBOWrite{};
    waterRenderUBOWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    waterRenderUBOWrite.dstSet = staticDs;
    waterRenderUBOWrite.dstBinding = 10;
    waterRenderUBOWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    waterRenderUBOWrite.descriptorCount = 1;
    waterRenderUBOWrite.pBufferInfo = &waterRenderUBOInfo;
    writes.push_back(waterRenderUBOWrite);

    // ── Static descriptor set ──
    // Write bindings 1-13 to the static descriptor set once. These resources
    // rarely change (texture arrays, materials, shadow maps, sky, water params).
    // Per-frame descriptor sets will copy these via VkCopyDescriptorSet.
    {
        VkDescriptorSet staticSet = app->getStaticDescriptorSet();
        if (staticSet != VK_NULL_HANDLE) {
            DescriptorWriter staticWriter(app->getDevice());
            // Replay accumulated image/buffer writes into the static set
            for (auto &w : writes) {
                if (w.dstBinding == 0) continue; // binding 0 is per-frame
                if (w.pImageInfo) {
                    staticWriter.writeImage(staticSet, w.dstBinding, w.descriptorType,
                                            w.pImageInfo[0].sampler, w.pImageInfo[0].imageView,
                                            w.pImageInfo[0].imageLayout, w.descriptorCount);
                } else if (w.pBufferInfo) {
                    staticWriter.writeBuffer(staticSet, w.dstBinding, w.descriptorType,
                                             w.pBufferInfo[0].buffer, w.pBufferInfo[0].offset,
                                             w.pBufferInfo[0].range, w.descriptorCount);
                }
            }
            staticWriter.flush();
        }
    }

    // ── Per-frame descriptor sets ──
    // For each frame, copy bindings 1-13 from the static set and write binding 0
    // (per-frame UBO) separately using DescriptorWriter.
    {
        VkDescriptorSet staticSet = app->getStaticDescriptorSet();
        for (size_t fi = 0; fi < mainUniformBuffers.size(); ++fi) {
            VkDescriptorSet dstSet = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));

            // Collect all static bindings (1-13) for copy from staticDs.
            // The writes template excludes binding 6 (Sky UBO was written by
            // skyRenderer->init), so we enumerate the union explicitly.
            std::vector<VkCopyDescriptorSet> copies;
            auto addCopy = [&](uint32_t binding, uint32_t count = 1) {
                VkCopyDescriptorSet c{};
                c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
                c.srcSet = staticSet; c.srcBinding = binding; c.srcArrayElement = 0;
                c.dstSet = dstSet; c.dstBinding = binding; c.dstArrayElement = 0;
                c.descriptorCount = count;
                copies.push_back(c);
            };
            for (auto &w : writes) {
                if (w.dstBinding == 0) continue;
                addCopy(w.dstBinding, w.descriptorCount);
            }
            addCopy(6); // Sky UBO — written to staticSet by skyRenderer->init

            if (!copies.empty()) {
                vkUpdateDescriptorSets(app->getDevice(), 0, nullptr,
                                       static_cast<uint32_t>(copies.size()), copies.data());
            }

            // Write per-frame UBO (binding 0) using DescriptorWriter
            DescriptorWriter writer(app->getDevice());
            writer.writeBuffer(dstSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                               mainUniformBuffers[fi].buffer, 0, sizeof(UniformObject));

            writer.flush();
        }
    }

    // ── Allocate (once) and write per-frame brush depth descriptor sets (set=1) ──
    for (size_t fi = 0; fi < brushDepthDescriptorSets.size(); ++fi) {
        if (brushDepthDescriptorSets[fi] == VK_NULL_HANDLE) {
            brushDepthDescriptorSets[fi] = app->createDescriptorSet(app->getBrushDepthDescriptorSetLayout());
        }
    }
    writeBrushDepthDescriptors(app);

    // ── Allocate (once) and write shadow-specific descriptor sets per-frame ──
    shadowDescriptorSets.resize(mainUniformBuffers.size());
    for (size_t fi = 0; fi < shadowDescriptorSets.size(); ++fi) {
        VkDescriptorSet ds = shadowDescriptorSets[fi];
        if (ds == VK_NULL_HANDLE) {
            ds = app->createDescriptorSet(app->getDescriptorSetLayout());
            shadowDescriptorSets[fi] = ds;
        }

        DescriptorWriter wr(app->getDevice());
        wr.writeBuffer(ds, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                       mainUniformBuffers[fi].buffer, 0, sizeof(UniformObject));

        auto addImg = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
            if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            wr.writeImage(ds, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          sampler, view, layout);
        };

        if (textureArrayManager) {
            addImg(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(12, textureArrayManager->roughnessSampler, textureArrayManager->roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(13, textureArrayManager->aoSampler, textureArrayManager->aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        addImg(4, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImg(8, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImg(9, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        if (solid360Renderer) {
            VkImageView cubeView = solid360Renderer->getSolid360View();
            VkSampler cubeSampler = solid360Renderer->getSolid360Sampler();
            if (cubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE) {
                addImg(11, cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }

        wr.writeBuffer(ds, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
        wr.writeBuffer(ds, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);
        wr.writeBuffer(ds, 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                       waterRenderUBOBuffer_.buffer, 0, sizeof(WaterRenderUBO));
        wr.flush();
    }

    // Register listener so we update the main descriptor set when texture arrays are allocated later
    if (textureArrayManager) {
        textureArrayManager->addAllocationListener([this, app, textureArrayManager]() {
            this->updateTextureDescriptorSet(app, textureArrayManager);
        });
    }
    mainLiquidRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

    // Ensure back-face render targets are created as well so the
    // `backFaceDepthView` is valid before the first frame's water pass.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

    // Create wireframe pipelines for solid and water passes
    if (solidWireframe) {
        std::vector<VkDescriptorSetLayout> solidSetLayouts = { app->getDescriptorSetLayout() };
        solidWireframe->createPipeline(app, {app->getSwapchainImageFormat()},
            solidSetLayouts,
            "shaders/main.vert.spv", "shaders/wireframe.frag.spv",
            "shaders/main.tesc.spv", "shaders/main.tese.spv",
            "solid wireframe");
    }
    if (waterWireframe) {
        std::vector<VkDescriptorSetLayout> waterSetLayouts = {
            app->getDescriptorSetLayout(),
            app->getMaterialDescriptorSetLayout(),
            mainLiquidRenderer->getWaterDepthDescriptorSetLayout()
        };
        waterWireframe->createPipeline(app, {VK_FORMAT_R32G32B32A32_SFLOAT},
            waterSetLayouts,
            "shaders/water.vert.spv", "shaders/water_wireframe.frag.spv",
            "shaders/water.tesc.spv", "shaders/water.tese.spv",
            "water wireframe");
    }

    // Initialize post-process renderer (composites scene + water into swapchain)
    postProcessRenderer->init(app);
    postProcessRenderer->setRenderSize(app->getWidth(), app->getHeight());
    
    // Finalize sky renderer with sphere VBO. Write Sky UBO to static descriptor set
    // so all per-frame sets inherit it via the static copy (already done above).
    VkDescriptorSet finalStaticDs = app->getStaticDescriptorSet();
    if (finalStaticDs != VK_NULL_HANDLE) {
        skyRenderer->init(app, *skySettings, finalStaticDs);
    } else {
        skyRenderer->init(app, *skySettings, mainDs);
        for (uint32_t i = 1; i < static_cast<uint32_t>(app->getMainDescriptorSetCount()); ++i) {
            VkDescriptorSet ds = app->getMainDescriptorSetForFrame(i);
            if (ds != VK_NULL_HANDLE) {
                skyRenderer->init(app, *skySettings, ds);
            }
        }
    }
    

    // Activate the stable-slot indirect rendering pipeline (no global rebuilds).
    // One slot PER CHUNK: the chunk's LoD levels share the slot, each in its
    // own static row (IndirectRenderer::initSlots splits the per-slot budget
    // into per-level rows, ~1.33x the level-0 budget total). Only the indirect
    // command list is per-level (chunk * kMaxChunkLevels draw entries), so GPU
    // memory stays at the pre-refactor per-chunk footprint.
    // Must be called after all sub-renderers are initialized, before scene loading.
    initSlottedMode(app, 
        kMaxSolidChunkSlots,
        kMaxWaterChunkSlots,
        1u << 20,  // 1 MB vertex data per slot (level-0 row)
        1u << 18
    ); // 256 KB index data per slot (level-0 row)

    // Initialize brush solid/liquid IndirectRenderers with their own slot pools
    // (smaller — brush preview rarely exceeds a few dozen meshes). Brush geometry
    // no longer shares the main scene slot pools.
    brushSolidIndirectRenderer.initSlots(app, kMaxBrushChunkSlots,
                                         1u << 18,  // 256 KB vertex data per slot
                                         1u << 16); // 64 KB index data per slot
    brushLiquidIndirectRenderer.initSlots(app, kMaxBrushChunkSlots,
                                          1u << 18,  // 256 KB vertex data per slot
                                          1u << 16); // 64 KB index data per slot
}

// Update only the static bindings (textures, materials, water params) in the
// static descriptor set, then propagate to all per-frame descriptor sets via
// VkCopyDescriptorSet. This avoids re-writing identical descriptors per frame.
void SceneRenderer::updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager) {
    if (!app) return;

    VkDescriptorSet staticDs = app->getStaticDescriptorSet();
    if (staticDs == VK_NULL_HANDLE) return;

    // 1. Write updated bindings to the static descriptor set
    {
        DescriptorWriter writer(app->getDevice());

        auto addImg = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
            if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            writer.writeImage(staticDs, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              sampler, view, layout);
        };

        addImg(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImg(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImg(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImg(12, textureArrayManager->roughnessSampler, textureArrayManager->roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImg(13, textureArrayManager->aoSampler, textureArrayManager->aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Shadow map samplers (bindings 4, 8, 9) for all cascades
        if (shadowMapper) {
            addImg(4, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(8, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(9, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        // Materials SSBO (binding 5) — refresh from MaterialManager in case the buffer
        // was allocated after SceneRenderer::init (setupTextures may run on a separate thread)
        if (materialManagerPtr && materialManagerPtr->getBuffer().buffer != VK_NULL_HANDLE) {
            materialsBuffer = materialManagerPtr->getBuffer();
        }
        if (materialsBuffer.buffer != VK_NULL_HANDLE)
            writer.writeBuffer(staticDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
        else
            std::cerr << "[SceneRenderer::updateTextureDescriptorSet] materials buffer not available — skipping binding 5\n";

        if (waterParamsBuffer_.buffer != VK_NULL_HANDLE)
            writer.writeBuffer(staticDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);

        writer.flush();
    }

    // 2. Propagate static bindings (1-13) to all per-frame descriptor sets
    const size_t setCount = app->getMainDescriptorSetCount();
    for (size_t s = 0; s < setCount; ++s) {
        VkDescriptorSet mainDs = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(s));
        if (mainDs == VK_NULL_HANDLE) continue;

        // Build copy descriptors for all bindings 1-13
        std::vector<VkCopyDescriptorSet> copies;
        // Binding 1..4, 8, 9, 11 (textures)
        for (uint32_t b : {1u, 2u, 3u, 4u, 8u, 9u, 11u, 12u, 13u}) {
            VkCopyDescriptorSet c{};
            c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
            c.srcSet = staticDs; c.srcBinding = b; c.srcArrayElement = 0;
            c.dstSet = mainDs; c.dstBinding = b; c.dstArrayElement = 0;
            c.descriptorCount = 1;
            copies.push_back(c);
        }
        // Binding 5, 7 (storage buffers), 6 (Sky UBO), 10 (Water render UBO)
        for (uint32_t b : {5u, 6u, 7u, 10u}) {
            VkCopyDescriptorSet c{};
            c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
            c.srcSet = staticDs; c.srcBinding = b; c.srcArrayElement = 0;
            c.dstSet = mainDs; c.dstBinding = b; c.dstArrayElement = 0;
            c.descriptorCount = 1;
            copies.push_back(c);
        }

        if (!copies.empty()) {
            vkUpdateDescriptorSets(app->getDevice(), 0, nullptr,
                                   static_cast<uint32_t>(copies.size()), copies.data());
        }
    }

    // ── Also update shadow descriptor sets (bindings 1-3, 5, 7) with new textures/materials.
    // Shadow sets use dummy depth views for bindings 4, 8, 9, 11 so they cannot simply copy
    // from the static set; we write them individually.
    if (!shadowDescriptorSets.empty() && textureArrayManager) {
        for (size_t si = 0; si < shadowDescriptorSets.size(); ++si) {
            VkDescriptorSet ds = shadowDescriptorSets[si];
            DescriptorWriter sw(app->getDevice());
            auto addImg = [&](uint32_t b, VkSampler sm, VkImageView vw, VkImageLayout ly) {
                if (vw == VK_NULL_HANDLE || sm == VK_NULL_HANDLE) return;
                sw.writeImage(ds, b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sm, vw, ly);
            };
            addImg(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(12, textureArrayManager->roughnessSampler, textureArrayManager->roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(13, textureArrayManager->aoSampler, textureArrayManager->aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (materialsBuffer.buffer != VK_NULL_HANDLE)
                sw.writeBuffer(ds, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
            else
                std::cerr << "[SceneRenderer::updateTextureDescriptorSet] shadow materials buffer not available — skipping shadow binding 5\n";
            if (waterParamsBuffer_.buffer != VK_NULL_HANDLE)
                sw.writeBuffer(ds, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);
            sw.flush();
        }
    }

    // Also rewrite brush depth descriptors for all per-frame main sets
    writeBrushDepthDescriptors(app);
}


// Drain whatever CPU-generated mesh data the background loading thread has
// queued since the last frame, and perform the actual Vulkan GPU uploads.
// Must be called from the main (render) thread each frame.
size_t SceneRenderer::publishPendingMeshes(
    VulkanApp* app,
    std::deque<PendingMeshData>& batch,
    IndirectRenderer& opaqueIR,
    IndirectRenderer& brushOpaqueIR,
    IndirectRenderer& waterIR,
    IndirectRenderer& brushWaterIR,
    const std::function<uint32_t(Layer layer, NodeID nid, bool isBrush)>& takeOldSlot,
    const std::function<void(Layer layer, NodeID nid, uint32_t slotIdx, uint32_t version, bool isBrush)>& onChunkPublished,
    const std::function<void(NodeID nid, const Geometry& geom, bool isBrush)>& onFinestPublished)
{
    // One-slot-per-chunk publish: NO ladder reassembly, NO running sub-offsets.
    // Each queue entry is a self-contained geometry chunk at a single LoD
    // level. A chunk's levels share ONE slot: every level's data lands in its
    // own static row of the slot (fixed per-level budget — see
    // IndirectRenderer::initSlots), and every level publishes its own draw
    // entry (drawIndex = slot * kMaxChunkLevels + level) so the GPU cull
    // shader band-tests entries independently. addMeshSlotted is keyed by the
    // BASE chunk id, so all levels of a chunk resolve to the same slot (the
    // first allocates, later ones update in place) — including across frames
    // and across edits of the same chunk.
    //
    // takeOldSlot is consumed once per chunk: the first level that encounters
    // a pending-delete entry captures the old slot and frees it after ITS
    // upload completes (old geometry stays resident until the new data is
    // valid on GPU).
    //
    // Band coverage: NO publish-time clamp. Each entry's meta carries the
    // chunk's true ladder depth, and the cull shader computes the chunk's
    // published finest/coarsest range itself (scanning the 5 entries'
    // indexCount), clamping the band selection into it — so partial or
    // level-0-missing ladders still render at every distance. End-of-drain,
    // draw entries ABOVE this drain's accepted coarsest level are zeroed: a
    // chunk whose ladder shrank on republish would otherwise keep its old
    // entries' meta (still band-passing) and draw stale geometry.
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> acceptedCoarsest; // key -> {slotIdx, coarsest accepted this drain}
    size_t slotsPublished = 0;
    for (auto& item : batch) {
        const Layer layer = item.layer;
        const NodeID nid = item.nid;
        const Octree::LoDMesh& lod = item.lodMesh;
        const uint8_t level = lod.lod;
        const bool isBrush = item.isBrush;

        // Per-entry routing: solid main → opaqueIR, solid brush → brushOpaqueIR,
        // transparent main → waterIR, transparent brush → brushWaterIR. This is
        // the ONLY stream distinction the publish core makes; everything
        // downstream is one shared codepath.
        IndirectRenderer* ir = (layer == LAYER_OPAQUE)
            ? (isBrush ? &brushOpaqueIR : &opaqueIR)
            : (isBrush ? &brushWaterIR : &waterIR);
        if (!ir) continue;

        const ChunkManager::ChunkId base = static_cast<ChunkManager::ChunkId>(nid);

        // Resolve (and consume) any pending-delete slot for this chunk: the
        // old geometry stays resident until the new upload completes. Only the
        // first level of the chunk sees the entry (consumed on first hit).
        uint32_t oldSlot = takeOldSlot(layer, nid, isBrush);

        if (lod.geom.vertices.empty() || lod.geom.indices.empty()) continue;

        const glm::vec3 cubeMin = item.nodeData.cube.getMin();
        const glm::vec3 cubeMax = item.nodeData.cube.getMax();

        // Publish this level into its static row of the chunk's slot. The
        // level's sub-offsets are its row base (fixed per level). The band
        // meta maxLevel is the chunk's true ladder depth (unclamped — the GPU
        // shader clamps against the chunk's actually-published range).
        const uint32_t lv = level;
        // Composite key: bit0 = transparent, bit1 = brush stream. Unique across
        // streams so the end-of-drain cleanup resolves the owning IR.
        const uint32_t key = (static_cast<uint32_t>(base) << 2) |
                             (layer == LAYER_OPAQUE ? 0u : 1u) |
                             (isBrush ? 2u : 0u);
        const uint32_t slotIdx = ir->addMeshSlotted(lod.geom, static_cast<uint32_t>(base), (int)lv,
                                                    UINT32_MAX,
                                                    ir->getSlotLevelVertexOffset((int)lv),
                                                    ir->getSlotLevelIndexOffset((int)lv),
                                                    lod.cellSize,
                                                    lod.maxLevel,
                                                    &cubeMin,
                                                    &cubeMax);
        if (slotIdx == UINT32_MAX) continue; // no free slot in the pool / row budget exceeded

        // addMeshSlotted re-publishes the existing chunk slot in place when
        // one is already resident (updateMeshSlotted path); then oldSlot ==
        // slotIdx and there is nothing to free.
        if (oldSlot != UINT32_MAX && oldSlot == slotIdx) oldSlot = UINT32_MAX;

        // Register the chunk's slot for the erase path before the upload
        // starts; the deferred completion frees any replaced old slot and
        // (level 0 -- the mesh the ChunkManager build state tracks) promotes
        // the chunk to ReadyToSwap once resident.
        if (!isBrush && world_)
            world_->chunkManager().setSlotIndex(base, slotIdx);

        const bool trackChunkManager = !isBrush;
        ir->uploadSlot(app, slotIdx, (int)lv, 0.0f,
            [ir, oldSlot, this, base, lv, trackChunkManager]() {
                if (oldSlot != UINT32_MAX) ir->removeMeshSlotted(oldSlot);
                if (trackChunkManager && this->world_ && lv == 0)
                    this->world_->chunkManager().finishUpload(base);
            });

        onChunkPublished(layer, nid, slotIdx, lod.version, isBrush);

        // Generate vegetation instances for grass chunks using the level-0
        // (finest) geometry only. Coarse levels (chunkLod > 0) never drive
        // vegetation.
        if (layer == LAYER_OPAQUE && vegetationRenderer && lv == 0 &&
            !lod.geom.vertices.empty()) {
            onFinestPublished(nid, lod.geom, isBrush);
        }

        auto& acc = acceptedCoarsest[key];
        acc.first = slotIdx;
        acc.second = std::max<uint32_t>(acc.second, lv);
        ++slotsPublished;
    }

    // End-of-drain stale-entry cleanup: zero every draw entry of each
    // published chunk above the coarsest level ACCEPTED this drain. Entries
    // above it have no pending upload completion that could re-write them, so
    // the immediate host write is safe.
    for (auto& [key, v] : acceptedCoarsest) {
        if (v.second + 1 >= kMaxChunkLevels) continue; // full ladder, nothing stale
        const bool transparent = (key & 1u) != 0;
        const bool brush = (key & 2u) != 0;
        IndirectRenderer* ir = transparent ? (brush ? &brushWaterIR : &waterIR)
                               : brush ? &brushOpaqueIR
                               : &opaqueIR;
        ir->clearSlotLevelsFrom(v.first, v.second + 1);
    }

    return slotsPublished;
}

void SceneRenderer::drainPendingMeshes(std::deque<PendingMeshData>& out, size_t maxCount) {
    // Drain the shared queue (main scene + brush scene) at a CONTROLLED rate.
    // A full drain would burst hundreds of (chunk, level) uploads into one
    // frame when a large map finishes tessellating at once — saturating the
    // shared iGPU's command queue for seconds, tripping the amdgpu watchdog
    // (GPU reset, observed on Radeon 680M with 64-470 chunk batches) and
    // killing every GPU context on the machine. maxCount bounds the burst;
    // leftover entries stay queued. Chunks appear progressively as their CPU
    // tessellation completes. One shared budget spans every stream.
    std::lock_guard<std::mutex> lock(pendingMeshMutex);
    size_t taken = 0;
    for (auto it = pendingMeshQueue.begin();
         it != pendingMeshQueue.end() && taken < maxCount; ) {
        out.push_back(std::move(it->second));
        it = pendingMeshQueue.erase(it);
        ++taken;
    }
}

void SceneRenderer::processPendingMeshes(VulkanApp* app, glm::vec3 cameraPos, std::deque<PendingMeshData>& batch) {
    if (!mainLiquidRenderer) {
        std::cerr << "[processPendingMeshes] FATAL: waterRenderer is null!" << std::endl;
        return;
    }
    // Cache the camera position for the shadow pass (which culls with the
    // same camPos/lodBias so shadow draws match the main pass LoD selection).
    lastCameraPos_ = cameraPos;
    mainSolidRenderer->getIndirectRenderer().pollPendingTransfers(app);
    mainLiquidRenderer->getIndirectRenderer().pollPendingTransfers(app);
    brushSolidIndirectRenderer.pollPendingTransfers(app);
    brushLiquidIndirectRenderer.pollPendingTransfers(app);

    if (batch.empty()) {
        // No new geometry yet (brush tessellation may still be running). Keep
        // old geometry visible — don't free anything. On the next rebuild,
        // stageOldBrushChunks will re-capture the same slots. Still age out the
        // main stream's orphaned pending-delete entries (genuine deletions with
        // no replacement) so a mid-stream erase never leaks a slot.
        uint32_t curFrame = app ? app->getCurrentFrame() : 0;
        ageOutPendingDeletes(curFrame, mainSolidRenderer->getIndirectRenderer(), mainLiquidRenderer->getIndirectRenderer());
        processChunkSwapQueue(app);
        return;
    }

    // ── Deferred old-slot staging (brush) ─────────────────────────────────────
    // Instead of freeing old staged slots BEFORE allocating new ones (which
    // creates a window where neither old nor new geometry is valid on GPU), we
    // capture old slot indices now and free them AFTER each new slot's vertex
    // upload completes. This keeps the old geometry visible until the new data
    // is resident on the GPU, eliminating the 1-2 frame transient where the
    // brush disappears or renders garbage.
    //
    // addMeshSlotted may reuse the same slot when the same NodeID exists
    // (updateMeshSlotted path). In that case oldSlot == slotIdx and we must NOT
    // free the old slot — it was updated in-place, not replaced.
    //
    // Old slots whose NodeID no longer appears in the new set are orphans: their
    // chunk was removed in the rebuild and the stale geometry is freed
    // immediately after all new slots are allocated.
    std::unordered_map<NodeID, uint32_t> oldSolidSlots;
    std::unordered_map<NodeID, uint32_t> oldTransparentSlots;
    {
        std::lock_guard<std::recursive_mutex> lock(pendingOldBrushSolidChunksMutex);
        for (auto& entry : pendingOldBrushSolidChunks) {
            if (entry.second.meshId != UINT32_MAX)
                oldSolidSlots[entry.first] = entry.second.meshId;
        }
        pendingOldBrushSolidChunks.clear();
    }
    {
        std::lock_guard<std::recursive_mutex> lock(pendingOldBrushLiquidChunksMutex);
        for (auto& entry : pendingOldBrushLiquidChunks) {
            if (entry.second.meshId != UINT32_MAX)
                oldTransparentSlots[entry.first] = entry.second.meshId;
        }
        pendingOldBrushLiquidChunks.clear();
    }

    // ── UNIFIED publish pass ──────────────────────────────────────────────────
    // Both the main scene (solid/water) and the brush scene flow through the
    // SAME publish core. Each entry's (layer, isBrush) tag routes it to the
    // right IndirectRenderer, ChunkManager tracking and deferred-slot source:
    // solid/water geometry is processed exactly the same way as brush geometry.
    std::unordered_set<NodeID> matchedNids;
    [[maybe_unused]] size_t chunksPublished = publishPendingMeshes(app, batch, mainSolidRenderer->getIndirectRenderer(), brushSolidIndirectRenderer, mainLiquidRenderer->getIndirectRenderer(), brushLiquidIndirectRenderer,
        // takeOldSlot: resolve+consume the old slot for a chunk (one slot per
        // chunk — its LoD rows share it), or UINT32_MAX when none. The main
        // stream reads its pending-delete entry (one-frame grace); the brush
        // stream reads the deferred slots staged above.
        [this, &matchedNids, &oldSolidSlots, &oldTransparentSlots](Layer layer, NodeID nid, bool isBrush) -> uint32_t {
            if (isBrush) {
                auto& oldMap = (layer == LAYER_OPAQUE) ? oldSolidSlots : oldTransparentSlots;
                auto it = oldMap.find(nid);
                if (it == oldMap.end()) return UINT32_MAX;
                const uint32_t slot = it->second;
                oldMap.erase(it);
                matchedNids.insert(nid);
                return slot;
            }
            auto& deleteMap = (layer == LAYER_OPAQUE)
                ? this->pendingDeleteSolidSlots : this->pendingDeleteWaterSlots;
            auto it = deleteMap.find(nid);
            if (it == deleteMap.end()) return UINT32_MAX;
            const uint32_t slot = it->second.slotIndex;
            deleteMap.erase(it);
            return slot;
        },
        // onChunkPublished: the brush stream records the published slot/version
        // in its chunk maps (so erasure can free it later). The main stream's
        // slot registry update runs inside the core (ChunkManager) — nothing
        // extra to record here.
        [this](Layer layer, NodeID nid, uint32_t slotIdx, uint32_t version, bool isBrush) {
            if (!isBrush) return;
            auto& chunkMap = (layer == LAYER_OPAQUE)
                ? this->brushSolidChunks : this->brushTransparentChunks;
            chunkMap[nid] = Model3DVersion{slotIdx, version};
        },
        // onFinestPublished: grass chunks (main scene only) drive vegetation
        // from their level-0 (finest) geometry.
        [this, app](NodeID nid, const Geometry& geom, bool isBrush) {
            if (!isBrush) this->generateVegetationForNode(app, nid, geom);
        });

    // ── Orphan + grace sweeps (one sweep for ALL old slots) ──────────────────
    // Brush orphans: staged old slots whose NodeID no longer appears in the new
    // set. These chunks were removed in the rebuild and have stale geometry at
    // the old brush position — they must not linger as visible garbage.
    for (auto& [nid, oldSlot] : oldSolidSlots) {
        if (!matchedNids.count(nid) && brushSolidChunks.find(nid) == brushSolidChunks.end())
            brushSolidIndirectRenderer.removeMeshSlotted(oldSlot);
    }
    for (auto& [nid, oldSlot] : oldTransparentSlots) {
        if (!matchedNids.count(nid) && brushTransparentChunks.find(nid) == brushTransparentChunks.end())
            brushLiquidIndirectRenderer.removeMeshSlotted(oldSlot);
    }
    // Main stream: age out pending-delete entries that have been waiting longer
    // than MAX_FRAMES_IN_FLIGHT. For solid/water the octree node is reused with
    // the same NodeID, so a matching entry is normally consumed within 1 frame.
    // Entries that age out are genuine deletions (no replacement).
    uint32_t curFrame = app ? app->getCurrentFrame() : 0;
    ageOutPendingDeletes(curFrame, mainSolidRenderer->getIndirectRenderer(), mainLiquidRenderer->getIndirectRenderer());
    
    // Every frame, process the chunk swap queue (slotted mode).
    // This swaps in newly-built RenderProxies and retires old ones.
    processChunkSwapQueue(app);
}

void SceneRenderer::ageOutPendingDeletes(uint32_t curFrame, IndirectRenderer& solidIR, IndirectRenderer& waterIR) {
    auto ageOut = [curFrame](std::unordered_map<NodeID, PendingDeleteEntry>& deleteMap, IndirectRenderer& ir) {
        for (auto it = deleteMap.begin(); it != deleteMap.end(); ) {
            if (curFrame - it->second.birthFrame > VulkanApp::MAX_FRAMES_IN_FLIGHT) {
                ir.removeMeshSlotted(it->second.slotIndex);
                it = deleteMap.erase(it);
            } else {
                ++it;
            }
        }
    };
    ageOut(pendingDeleteSolidSlots, solidIR);
    ageOut(pendingDeleteWaterSlots, waterIR);
#ifdef DEBUG
    // DIAG: pending-delete backlog growth per second (user-reported draw-cmd
    // accumulation). Entries are consumed by matching publishes or aged out
    // after MAX_FRAMES_IN_FLIGHT; steady growth here means the erase path
    // outpaces publishes (unmatched deletions).
    static std::chrono::steady_clock::time_point lastDiag{};
    auto nowD = std::chrono::steady_clock::now();
    if (nowD - lastDiag >= std::chrono::seconds(1)) {
        lastDiag = nowD;
        std::cout << "[SceneRenderer::diag] pendingDelSolid=" << pendingDeleteSolidSlots.size()
                  << " pendingDelWater=" << pendingDeleteWaterSlots.size()
                  << " curFrame=" << curFrame << std::endl;
    }
#endif
}

// ── Slotted mode chunk processing ──────────────────────────────────────────

void SceneRenderer::initSlottedMode(VulkanApp* app, uint32_t maxSolidChunks,
                                    uint32_t maxWaterChunks,
                                    uint32_t vertexBytesPerChunk,
                                    uint32_t indexBytesPerChunk)
{
    mainSolidRenderer->getIndirectRenderer().initSlots(app, maxSolidChunks, vertexBytesPerChunk, indexBytesPerChunk);
    mainLiquidRenderer->getIndirectRenderer().initSlots(app, maxWaterChunks, vertexBytesPerChunk, indexBytesPerChunk);

    std::cout << "[SceneRenderer] slotted pools: solid=" << maxSolidChunks
              << " water=" << maxWaterChunks
              << " (per-slot " << (vertexBytesPerChunk >> 10) << " KB vertex + "
              << (indexBytesPerChunk >> 10) << " KB index, ~"
              << ((uint64_t)(maxSolidChunks + maxWaterChunks) * (vertexBytesPerChunk + indexBytesPerChunk) >> 20)
              << " MB device-local)" << std::endl;

}

bool SceneRenderer::processChunkSlotted(Layer layer, NodeID nid,
                                         const OctreeNodeData& nd,
                                         const Geometry& geom, uint32_t version)
{

    // Queue the geometry for main-thread GPU upload.
    // NOTE: markDirty + beginBuild were already called in the change handler
    // BEFORE tessellation was dispatched. The chunk state is already
    // UploadingGPU (from finishBuild). processPendingMeshes will call
    // addMeshSlotted + uploadSlot, then the upload completion callback calls
    // finishUpload → ReadyToSwap → processChunkSwapQueue atomically swaps.
    {
        // All streams share ONE pending queue (main solid/water + brush
        // solid/water); entries are tagged isBrush=false here since this path
        // feeds the main scene.
        std::lock_guard<std::mutex> lock(pendingMeshMutex);
        Octree::LoDMesh lod = {geom, /*lod*/ 0, /*version*/ version, nd.cube.getLength().x, /*maxLevel*/ 0};
        pendingMeshQueue[nid] = {layer, nid, std::move(lod), nd, /*isBrush=*/false};
    }

    return true;
}

void SceneRenderer::processChunkSwapQueue(VulkanApp* app)
{
    // Drain the swap queue: mark each ready chunk's new mesh version as
    // current. GPU slot data was already installed by the upload completion
    // callback, so no resource cleanup is needed here.
    if (world_) world_->chunkManager().processSwapQueue();
}

void SceneRenderer::processNodeLayer(Scene& scene, Layer layer, NodeID nid, OctreeNodeData& nodeData, GeometryHandler onGeometry, float minSize, ThreadPool* poolOverride) {

    // Only CHUNKS (stored chunkLod == 1 in the +1-shifted uint8_t space)
    // generate meshes. Coarse ancestor meshes (stored chunkLod > 1) are
    // disabled — tessellating an ancestor as one big
    // Surface-Nets cell samples the SDF at the coarse corners and misses
    // interior surface, so those meshes come out empty. Instead each chunk
    // publishes its own LoD LADDER: level 0 is the full-detail frontier mesh,
    // levels 1..N are the ancestor cells' real Surface-Nets meshes (cell size
    // frontierCell*2^lvl, sampled from each node's own stored corner SDFs) —
    // the tesselator walks the whole ladder in ONE pass, so far chunks draw
    // a fraction of the triangles.
    //
    // The GPU band test keeps ONE entry per chunk: entryLevel k covers
    // dist in [k, k+1) * chunkBase * lodBias. Every level uses the chunk base
    // cellSize (not its own density) so the bands TILE distance without gaps,
    // and the coarser density degrades with distance.
    const uint8_t chunkLod = nodeData.node ? nodeData.node->getChunkLod() : 0;
    if (chunkLod != 1) return;

    const float cubeLength = nodeData.cube.getLength().x;
    const uint8_t maxLevel = scene.maxChunkLod(layer, minSize);

    scene.requestModel3D(layer, nodeData, [&layer,&nid,&nodeData,&onGeometry,cubeLength,maxLevel](const Geometry& geo, uint8_t lod, uint version, uintptr_t emittingNodeId) {
        Octree::LoDMesh lm;
        lm.geom = geo;
        lm.lod = lod;
        lm.version = version;
        lm.cellSize = cubeLength;
        lm.maxLevel = maxLevel;
        onGeometry(layer, reinterpret_cast<NodeID>(emittingNodeId), lm);
    }, poolOverride);


}


void SceneRenderer::generateVegetationForNode(VulkanApp* app, NodeID nid, const Geometry& geom) {
    if (!vegetationRenderer) return;
    if (geom.indices.size() < 3 || geom.vertices.empty()) return;
    try {
        constexpr int kGrassBrushIndex = 3; // See LandBrush::grass
        // Instances per world-space unit² of triangle area.
        constexpr float kVegetationDensity = 0.01f;

        // Create tightly-packed position buffer (vec3[]) for the compute shader
        std::vector<glm::vec3> positions;
        positions.reserve(geom.vertices.size());
        for (const auto &v : geom.vertices) positions.push_back(v.position);

        // Build area-weighted virtual slots using unbiased stochastic rounding.
        // expected = area * density (instances per world-space unit area)
        // count = floor(expected) + Bernoulli(frac(expected))
        // This preserves area-proportional density without bias.
        std::vector<uint32_t> grassIndices;
        grassIndices.reserve(geom.indices.size());
        const uint32_t chunkSeed = static_cast<uint32_t>(nid ^ (nid >> 32)) ^ 0x9e3779b9u;
        std::mt19937 samplingRng(chunkSeed);
        std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
        for (size_t i = 0; i + 2 < geom.indices.size(); i += 3) {
            const uint32_t i0 = geom.indices[i + 0];
            const uint32_t i1 = geom.indices[i + 1];
            const uint32_t i2 = geom.indices[i + 2];
            if (i0 >= geom.vertices.size() || i1 >= geom.vertices.size() || i2 >= geom.vertices.size()) continue;
            const bool hasGrass =
                geom.vertices[i0].brushIndex == kGrassBrushIndex ||
                geom.vertices[i1].brushIndex == kGrassBrushIndex ||
                geom.vertices[i2].brushIndex == kGrassBrushIndex;
            if (!hasGrass) continue;
            const glm::vec3& v0 = geom.vertices[i0].position;
            const glm::vec3& v1 = geom.vertices[i1].position;
            const glm::vec3& v2 = geom.vertices[i2].position;
            // Skip steep / downward-facing triangles (same criterion as compute shader).
            // This avoids allocating output slots that the compute shader would discard,
            // preventing garbage uninitialized memory from reaching the draw call.
            const glm::vec3 faceNormal = glm::cross(v1 - v0, v2 - v0);
            if (glm::abs(faceNormal.y) <= 0.5f * glm::length(faceNormal)) continue;
            const float area = 0.5f * glm::length(faceNormal);
            const float expectedInstances = std::max(0.0f, area * kVegetationDensity);
            uint32_t slotCount = static_cast<uint32_t>(std::floor(expectedInstances));
            const float fractional = expectedInstances - static_cast<float>(slotCount);
            if (unitDist(samplingRng) < fractional) {
                ++slotCount;
            }
            for (uint32_t s = 0; s < slotCount; ++s) {
                grassIndices.push_back(i0);
                grassIndices.push_back(i1);
                grassIndices.push_back(i2);
            }
        }

        // Shuffle virtual triangle slots per chunk so reducing indirect instanceCount
        // keeps a random spatial subset instead of always dropping the tail.
        if (grassIndices.size() >= 6) {
            std::mt19937 shuffleRng(chunkSeed ^ 0x85ebca6bu);
            const size_t triangleCount = grassIndices.size() / 3;
            for (size_t slot = triangleCount - 1; slot > 0; --slot) {
                std::uniform_int_distribution<size_t> dist(0, slot);
                const size_t other = dist(shuffleRng);
                if (other == slot) continue;
                for (size_t component = 0; component < 3; ++component) {
                    std::swap(grassIndices[slot * 3 + component], grassIndices[other * 3 + component]);
                }
            }
        }

        // Each virtual triangle slot produces exactly 1 instance.
        uint32_t instancesPerTriangle = 1u;
        uint32_t seed = static_cast<uint32_t>(nid & 0xffffffffull);
        glm::vec3 chunkCenter(0.0f);
        for (const auto& position : positions) {
            chunkCenter += position;
        }
        if (!positions.empty()) {
            chunkCenter /= static_cast<float>(positions.size());
        }
        if (grassIndices.size() < 3) {
            // No grass triangles in this chunk; ensure old chunk vegetation is cleared.
            if (std::getenv("VULKAN_DISABLE_VEGETATION")) {
                return;
            }
            // CPU path handles the empty case (clears any previous chunk data).
            vegetationRenderer->generateChunkInstancesCPU(nid, positions, grassIndices,
                chunkCenter, instancesPerTriangle, app, seed);
            return;
        }

        // CPU-side instance generation — avoids RADV GPUVM faults where
        // the Texture Cache/Pipe cannot read storage buffers on iGPUs.
        vegetationRenderer->generateChunkInstancesCPU(nid, positions, grassIndices,
            chunkCenter, instancesPerTriangle, app, seed);
    } catch (const std::exception &e) {
        std::cerr << "[SceneRenderer] Vegetation generation failed for node " << (unsigned long long)nid
                  << ": " << e.what() << std::endl;
    }
}

size_t SceneRenderer::getTransparentModelCount() {
    return mainLiquidChunks.size();
}

bool SceneRenderer::hasModelForNode(Layer layer, NodeID nid) const {
    if (layer == LAYER_OPAQUE) {
        return mainSolidChunks.find(nid) != mainSolidChunks.end();
    } else {
        return mainLiquidChunks.find(nid) != mainLiquidChunks.end();
    }
}

void SceneRenderer::updateDebugSDFCubesForChunk(NodeID nid, const OctreeNodeData& nd, const Octree& tree) {
    if (!debugSDFRenderer || !nd.node || !tree.allocator) return;

    std::vector<DebugSDFRenderer::CubeSDF> cubes;
    collectLeafSDFCubes(nd.node, nd.cube, *tree.allocator, cubes);

    std::lock_guard<std::recursive_mutex> lock(debugSDFCubesMutex);
    if (cubes.empty()) {
        nodeDebugSDFCubes.erase(nid);
    } else {
        nodeDebugSDFCubes[nid] = std::move(cubes);
    }
}

void SceneRenderer::removeDebugSDFCubesForNode(NodeID id) {
    std::lock_guard<std::recursive_mutex> lock(debugSDFCubesMutex);
    nodeDebugSDFCubes.erase(id);
}

void SceneRenderer::clearDebugSDFCubes() {
    std::lock_guard<std::recursive_mutex> lock(debugSDFCubesMutex);
    nodeDebugSDFCubes.clear();
}

std::vector<DebugSDFRenderer::CubeSDF> SceneRenderer::getDebugSDFCubes() {
    std::lock_guard<std::recursive_mutex> lock(debugSDFCubesMutex);
    std::vector<DebugSDFRenderer::CubeSDF> out;
    size_t total = 0;
    for (const auto& entry : nodeDebugSDFCubes) {
        total += entry.second.size();
    }
    out.reserve(total);
    for (const auto& entry : nodeDebugSDFCubes) {
        out.insert(out.end(), entry.second.begin(), entry.second.end());
    }
    return out;
}

void SceneRenderer::addDebugCubeForGeometry(Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
    if (!debugCubeRenderer) return;
    DebugCubeRenderer::CubeWithColor c;
    // Compute world-space AABB from geometry vertices using same model as used for mesh
    glm::vec3 minp(nd.cube.getMax()), maxp(nd.cube.getMin());
    for (const auto &v : geom.vertices) {
        minp = glm::min(minp, v.position);
        maxp = glm::max(maxp, v.position);
    }
    if (minp.x == FLT_MAX) {
        throw std::runtime_error("SceneRenderer::addDebugCubeForGeometry requires non-empty geometry (no fallback allowed)");
    }
    c.cube = BoundingBox(minp, maxp);
    c.color = (layer == LAYER_OPAQUE) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, 0.5f, 1.0f);
    addDebugCubeForNode(nid, c);
}


void SceneRenderer::clearBrushMeshes() {
    std::lock_guard<std::recursive_mutex> lock(brushSolidChunksMutex);

    // Drain any stale entries from a previous rebuild that haven't been
    // processed yet (processPendingMeshes runs once per frame in update()).
    {
        // Drain any stale brush entries from the shared queue (main-scene
        // entries are unaffected — this only touches isBrush-tagged ones).
        // processPendingMeshes runs once per frame in update().
        std::lock_guard<std::mutex> pqLock(pendingMeshMutex);
        for (auto it = pendingMeshQueue.begin(); it != pendingMeshQueue.end(); ) {
            if (it->second.isBrush) {
                it = pendingMeshQueue.erase(it);
            } else {
                ++it;
            }
        }
    }

    IndirectRenderer& brushIR = brushSolidIndirectRenderer;
    IndirectRenderer& brushWaterIR = brushLiquidIndirectRenderer;

    // Remove all brush opaque meshes from the dedicated brush solid IR.
    for (auto &entry : brushSolidChunks) {
        if (entry.second.meshId != UINT32_MAX) {
            brushIR.removeMeshSlotted(entry.second.meshId);
        }
    }
    brushSolidChunks.clear();

    // Remove brush transparent meshes from the dedicated brush liquid IR.
    for (auto &entry : brushTransparentChunks) {
        if (entry.second.meshId != UINT32_MAX) {
            brushWaterIR.removeMeshSlotted(entry.second.meshId);
        }
    }
    brushTransparentChunks.clear();
}

void SceneRenderer::stageOldBrushChunks() {
    {
        std::lock_guard<std::recursive_mutex> lock(brushSolidChunksMutex);
        for (auto& entry : brushSolidChunks) {
            pendingOldBrushSolidChunks[entry.first] = entry.second;
        }
        brushSolidChunks.clear();
    }
    {
        std::lock_guard<std::recursive_mutex> lock(brushTransparentChunksMutex);
        for (auto& entry : brushTransparentChunks) {
            pendingOldBrushLiquidChunks[entry.first] = entry.second;
        }
        brushTransparentChunks.clear();
    }
}
