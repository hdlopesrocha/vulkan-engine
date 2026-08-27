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
//   solid  ~1084 nodes (1024 pool was too small -> "no free slot" dropped chunks;
//          raised to 1536 to fit the reference scene with headroom),
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
constexpr uint32_t kMaxSolidChunkSlots = 1536;   // main solid (opaque) pool
constexpr uint32_t kMaxWaterChunkSlots = 192;    // main water (transparent) pool
constexpr uint32_t kMaxBrushChunkSlots = 64;     // brush preview pool

// Per-chunk (per-slot) geometry ceilings used to size the TOTAL packed pools
// (total = chunkCount * perChunk). Allocation is packed/variable-size, so a chunk
// only consumes what its mesh needs; this is the worst-case footprint ceiling.
// Smaller = less VRAM, but a chunk whose mesh exceeds the per-chunk ceiling can
// only fit if a large enough free span exists in the shared pool.
constexpr uint32_t kVertexBytesPerChunk = 1u << 19;  // 512 KB per chunk
constexpr uint32_t kIndexBytesPerChunk  = 1u << 17;  // 128 KB per chunk

} // namespace

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
    if (brushRenderer && app) {
        brushRenderer->cleanup(app);
    }
    if (solid360Renderer && app) {
        solid360Renderer->cleanup(app);
    }
    if (mainSolidRenderer && app) {
        mainSolidRenderer->cleanup(app);
    }
    if (shadowMapper && app) {
        shadowMapper->cleanup(app);
    }
    if (skyRenderer) {
        skyRenderer->cleanup(app);
    }
    if (vegetationRenderer) {
        vegetationRenderer->cleanup(app);
    }
    if (debugCubeRenderer) {
        debugCubeRenderer->cleanup(app);
    }
    if (boundingBoxRenderer) {
        boundingBoxRenderer->cleanup(app);
    }
    if (debugSDFRenderer) {
        debugSDFRenderer->cleanup(app);
    }
    if (waterWireframe) {
        waterWireframe->cleanup(app);
    }

    // Clear local CPU-side handles; Vulkan objects are destroyed via VulkanResourceManager
    for (auto &b : mainUniformBuffers) {
        if (b.buffer != VK_NULL_HANDLE) b = {};
    }
    mainUniformBuffers.clear();
}

// Propagate the shared per-frame command state tracker to every renderer that
// only records on the main thread (mirrors the pre-interface wiring in
// main.cpp). backFaceRenderer and the water IndirectRenderer stay unwired:
// the async back-face task records them on a separate thread and keeping
// cmdState=nullptr avoids a data race on frameCmdState.
void SceneRenderer::setCmdState(CommandBufferState* state) {
    if (shadowMapper) shadowMapper->setCmdState(state);
    if (mainSolidRenderer) mainSolidRenderer->setCmdState(state);
    if (skyRenderer) skyRenderer->setCmdState(state);
    if (vegetationRenderer) vegetationRenderer->setCmdState(state);
    if (postProcessRenderer) postProcessRenderer->setCmdState(state);
    if (debugCubeRenderer) debugCubeRenderer->setCmdState(state);
    if (boundingBoxRenderer) boundingBoxRenderer->setCmdState(state);
    if (debugSDFRenderer) debugSDFRenderer->setCmdState(state);
    if (waterWireframe) waterWireframe->setCmdState(state);
    if (solid360Renderer) solid360Renderer->setCmdState(state);
    if (mainLiquidRenderer) mainLiquidRenderer->setCmdState(state);
    if (brushRenderer) brushRenderer->setCmdState(state);
}

void SceneRenderer::stopGenPools() {
    if (brushRenderer) brushRenderer->stopGenPools();
    mainSolidGenPool.stop();
    mainWaterGenPool.stop();
}

void SceneRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    // Recreate offscreen targets that depend on swapchain size
    if (mainSolidRenderer) {
        mainSolidRenderer->createRenderTargets(app, width, height);
    }
    if (brushRenderer) {
        brushRenderer->onSwapchainResized(app, width, height);
    }
    if (mainLiquidRenderer) {
        mainLiquidRenderer->createRenderTargets(app, width, height);
        // Recreate back-face and 360 reflection targets owned by SceneRenderer
        if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, width, height);
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
}

SceneRenderer::SceneRenderer() :
    skyRenderer(std::make_unique<SkyRenderer>()),
    shadowMapper(std::make_unique<ShadowRenderer>(2048)),
    postProcessRenderer(std::make_unique<PostProcessRenderer>()),
    mainSolidRenderer(std::make_unique<SolidRenderer>()),
    mainLiquidRenderer(std::make_unique<WaterRenderer>()),
    vegetationRenderer(std::make_unique<VegetationRenderer>()),
    brushRenderer(std::make_unique<BrushRenderer>()),
    debugCubeRenderer(std::make_unique<DebugCubeRenderer>()),
    boundingBoxRenderer(std::make_unique<DebugCubeRenderer>()),
    debugSDFRenderer(std::make_unique<DebugSDFRenderer>()),
    waterWireframe(std::make_unique<WireframeRenderer>()),
    skySettings(std::make_unique<SkySettings>())
{
    // Vegetation cull is MERGED into the solid IndirectRenderer's single
    // indirect.comp dispatch, so the vegetation renderer must share the solid
    // IndirectRenderer (it supplies the per-frame veg output buffers + metadata).
    vegetationRenderer->setSolidIndirectRenderer(&mainSolidRenderer->getIndirectRenderer());
}

SceneRenderer::~SceneRenderer() {
    // Do not attempt Vulkan cleanup here (app is not available). The owner
    // (MyApp) must call `sceneRenderer->cleanup(app)` before destroying the
    // VulkanApp instance.
}

void SceneRenderer::init(VulkanApp* app, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterParams>& waterParams) {
    if (!app) {
        std::cerr << "[SceneRenderer::init] app is nullptr!" << std::endl;
        return;
    }

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
    // Per-frame staging buffers for shadow-pass UBO uploads are owned by
    // ShadowRenderer (see createStagingBuffers).
    shadowMapper->createStagingBuffers(app, dsCount);

    VkDescriptorSet mainDs = app->getMainDescriptorSetForFrame(0);

    // Initialize sky renderer now that descriptor sets are ready. Write the Sky
    // UBO (binding 6) to the static descriptor set FIRST — the per-frame copy
    // loop below propagates it to every per-frame set, and the copy must not
    // run before binding 6 exists.
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
    solid360Renderer = std::make_unique<Solid360Renderer>();

    // Initialize WaterRenderer (creates its pipeline layout and initializes the param SSBO)
    mainLiquidRenderer->init(app, waterParamsBuffer_, waterParams, layerCount);

    // Now that WaterRenderer has created its pipeline layout, allow the
    // back-face renderer to create pipelines that depend on it.
    if (backFaceRenderer) backFaceRenderer->createPipelines(app, mainLiquidRenderer->getWaterGeometryPipelineLayout());
    // Create back-face render targets early so their image views are
    // available before the first frame's water pass attempts to bind them.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
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

    // Bind water render UBO to binding 10 of main descriptor set (the buffer
    // itself is created and updated by WaterRenderer).
    VkDescriptorBufferInfo& waterRenderUBOInfo = writesBuf.emplace_back(mainLiquidRenderer->getWaterRenderUBO().buffer, 0, sizeof(WaterRenderUBO));
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

    // ── Initialize the brush renderer ──
    // Wire the samplers used by the brush depth descriptor writes (from the
    // water and shadow renderers, which are initialized above), then create
    // everything brush-related: offscreen targets, back-face renderer,
    // per-frame brush depth descriptor sets (set=1) and the dedicated brush
    // IndirectRenderers.
    if (brushRenderer) {
        brushRenderer->setDepthSamplers(
            mainLiquidRenderer ? mainLiquidRenderer->getLinearSampler() : VK_NULL_HANDLE,
            shadowMapper ? shadowMapper->getShadowMapSampler() : VK_NULL_HANDLE);
        brushRenderer->init(app, app->getWidth(), app->getHeight());
    }

    // ── Wire scene sub-renderers into the pass orchestrators ──
    // The shadow pass draws solid/water/vegetation/brush geometry and the
    // water pass samples solid offscreen targets + brush liquid geometry,
    // so each orchestrator caches the pointers it needs.
    if (shadowMapper) {
        shadowMapper->setSceneRenderers(mainSolidRenderer.get(), mainLiquidRenderer.get(),
                                        vegetationRenderer.get(), brushRenderer.get());
    }
    if (mainLiquidRenderer) {
        mainLiquidRenderer->setSceneRenderers(mainSolidRenderer.get(), brushRenderer.get(),
                                              backFaceRenderer.get(), solid360Renderer.get(),
                                              waterWireframe.get());
    }

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
                       mainLiquidRenderer->getWaterRenderUBO().buffer, 0, sizeof(WaterRenderUBO));
        wr.flush();
    }
    // Shadow descriptor set handles are stable after init (subsequent writes
    // only update them in place), so ShadowRenderer can cache them once.
    if (shadowMapper) {
        shadowMapper->setShadowDescriptorSets(shadowDescriptorSets);
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

    // Create the solid wireframe pipeline (owned by SolidRenderer) and the
    // water wireframe pipeline
    mainSolidRenderer->createWireframe(app);
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
    
    // Activate the stable-slot indirect rendering pipeline (no global rebuilds).
    // One draw entry PER CHUNK: each chunk's vertex/index data is packed into
    // the shared element pools (PackedSpaceAllocator — no fixed per-level
    // rows), and the indirect command list has one entry per chunk. GPU memory
    // is bounded by the element pools' TOTAL byte budgets passed to
    // initSlottedMode (per-chunk budgets x chunk count — the packed model's
    // ceiling, actual usage is data-driven).
    // Must be called after all sub-renderers are initialized, before scene loading.
    initSlottedMode(app,
        kMaxSolidChunkSlots,
        kMaxWaterChunkSlots,
        kVertexBytesPerChunk,  // per-chunk vertex ceiling (total = chunks x this)
        kIndexBytesPerChunk    // per-chunk index ceiling (total = chunks x this)
    );

    // Initialize brush solid/liquid IndirectRenderers with their own packed
    // element pools (smaller — brush preview rarely exceeds a few dozen
    // meshes). Brush geometry no longer shares the main scene slot pools.
    // The byte budgets are TOTAL shared pool budgets now (packed slots): each
    // chunk consumes only what its mesh actually uses.
    if (brushRenderer) {
        brushRenderer->initSlots(app, kMaxBrushChunkSlots,
                                 kMaxBrushChunkSlots * (1u << 18),  // total vertex pool
                                 kMaxBrushChunkSlots * (1u << 16)); // total index pool
    }
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
    if (brushRenderer) brushRenderer->writeDepthDescriptors(app);
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
    // One-slot-per-chunk publish. Each queue entry is a self-contained
    // geometry chunk (one mesh per chunk — chunks arrive one by one). The
    // mesh is packed into its own span of the shared element pools
    // (PackedSpaceAllocator — see IndirectRenderer::initSlots) and publishes
    // its single draw entry. addMeshSlotted is keyed by the chunk id, so
    // edits of the same chunk resolve to the same slot — re-publishing
    // allocates a NEW span and frees the old one once the replacement upload
    // completes.
    //
    // takeOldSlot is consumed once per chunk: a pending-delete entry captures
    // the old slot and frees it after ITS upload completes (old geometry
    // stays resident until the new data is valid on GPU).
    size_t slotsPublished = 0;
    static std::atomic<int> lvlHist[8] = {};
    static std::atomic<int> pubTotal{0};
    for (auto& item : batch) {
        const Layer layer = item.layer;
        const NodeID nid = item.nid;
        const Octree::LoDMesh& lod = item.lodMesh;
        const bool isBrush = item.isBrush;

        if (lod.lod >= 0 && lod.lod < 8) lvlHist[lod.lod]++;
        int t = ++pubTotal;
        if (t == 1 || t % 3000 == 0) {
            fprintf(stderr, "[PUBLISH-DIAG] total=%d levels(lod):", t);
            for (int i = 0; i < 8; ++i) fprintf(stderr, " %d:%d", i, lvlHist[i].load());
            fprintf(stderr, "  cubeMin-sample=(%.1f,%.1f,%.1f)\n",
                lod.boundsMin.x, lod.boundsMin.y, lod.boundsMin.z);
        }

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
        // old geometry stays resident until the new upload completes.
        uint32_t oldSlot = takeOldSlot(layer, nid, isBrush);

        if (lod.geom.vertices.empty() || lod.geom.indices.empty()) continue;

        const glm::vec3 cubeMin = lod.boundsMin;
        const glm::vec3 cubeMax = lod.boundsMax;

        // Publish the mesh into its single draw entry slot. The slot index is
        // the chunk's stable slot (one draw entry per chunk); `lod.lod` is the
        // chunk's 0-based LoD level (0 = frontier), published in the bounds
        // meta for the GPU's per-chunk distance band test.
        const uint32_t slotIdx = ir->addMeshSlotted(lod.geom, static_cast<uint32_t>(base),
                                                    &cubeMin,
                                                    &cubeMax,
                                                    lod.lod,
                                                    &lod.boundsBase);
        if (slotIdx == UINT32_MAX) continue; // no free block / element pool exhausted

        // addMeshSlotted re-publishes the existing chunk slot in place when
        // one is already resident (it allocates a new packed span); then
        // oldSlot == slotIdx and there is nothing to free.
        if (oldSlot != UINT32_MAX && oldSlot == slotIdx) oldSlot = UINT32_MAX;

        // Register the frontier chunk's slot for the erase path before the
        // upload starts; the deferred completion frees any replaced old slot
        // and promotes the chunk to ReadyToSwap once resident. Coarse
        // ancestor cells (level > 0) are not tracked by the ChunkManager.
        const bool frontier = (lod.lod == 0);
        if (!isBrush && world_ && frontier)
            world_->chunkManager().setSlotIndex(base, slotIdx);

        const bool trackChunkManager = !isBrush && frontier;
        ir->uploadSlot(app, slotIdx, 0.0f,
            [ir, oldSlot, this, base, trackChunkManager]() {
                if (oldSlot != UINT32_MAX) ir->removeMeshSlotted(oldSlot);
                if (trackChunkManager && this->world_)
                    this->world_->chunkManager().finishUpload(base);
            });

        onChunkPublished(layer, nid, slotIdx, lod.version, isBrush);

        // Generate vegetation instances for grass chunks using the frontier
        // (finest) geometry only. Coarse ancestor cells (level > 0) never
        // drive vegetation.
        if (layer == LAYER_OPAQUE && vegetationRenderer && frontier &&
            !lod.geom.vertices.empty()) {
            onFinestPublished(nid, lod.geom, isBrush);
        }

        ++slotsPublished;
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
    if (brushRenderer) brushRenderer->pollPendingTransfers(app);

    // Keep the GPU LoD band meta in sync with the tree (self-correcting once the
    // scene is loaded). chunkCellSize must be the GLOBAL Octree::chunkSize so the
    // band anchors align across all chunks; maxLodLevel the tree's real ladder
    // depth. Per-chunk values here would mis-align the distance bands and cull
    // most rungs (holes across the terrain).
    if (world_) {
        const float ms = 30.0f;
        mainSolidRenderer->getIndirectRenderer().setMaxLodLevel(world_->scene().maxChunkLod(LAYER_OPAQUE, ms));
        mainLiquidRenderer->getIndirectRenderer().setMaxLodLevel(world_->scene().maxChunkLod(LAYER_TRANSPARENT, ms));
        if (brushRenderer) {
            if (world_->brushScene()) {
                brushRenderer->getSolidIR().setMaxLodLevel(world_->brushScene()->maxChunkLod(LAYER_OPAQUE, ms));
                brushRenderer->getLiquidIR().setMaxLodLevel(world_->brushScene()->maxChunkLod(LAYER_TRANSPARENT, ms));
            }
        }
    }

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
    // addMeshSlotted may reuse the same block when the same NodeID exists
    // (in-place republish). In that case oldSlot == slotIdx and we must NOT
    // free the old slot — it was updated in-place, not replaced.
    //
    // Old slots whose NodeID no longer appears in the new set are orphans: their
    // chunk was removed in the rebuild and the stale geometry is freed
    // immediately after all new slots are allocated.
    std::unordered_map<NodeID, uint32_t> oldSolidSlots;
    std::unordered_map<NodeID, uint32_t> oldTransparentSlots;
    if (brushRenderer) brushRenderer->captureOldSlots(oldSolidSlots, oldTransparentSlots);

    // ── UNIFIED publish pass ──────────────────────────────────────────────────
    // Both the main scene (solid/water) and the brush scene flow through the
    // SAME publish core. Each entry's (layer, isBrush) tag routes it to the
    // right IndirectRenderer, ChunkManager tracking and deferred-slot source:
    // solid/water geometry is processed exactly the same way as brush geometry.
    std::unordered_set<NodeID> matchedNids;
    [[maybe_unused]] size_t chunksPublished = publishPendingMeshes(app, batch, mainSolidRenderer->getIndirectRenderer(), brushRenderer->getSolidIR(), mainLiquidRenderer->getIndirectRenderer(), brushRenderer->getLiquidIR(),
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
        // in its chunk maps (so erasure can free it later); the main stream
        // records the same in the scene chunk maps — frontier chunks are also
        // tracked by the ChunkManager, but coarse ancestor cells are not, so
        // their slot is resolved through this map when the cell is deleted.
        [this](Layer layer, NodeID nid, uint32_t slotIdx, uint32_t version, bool isBrush) {
            if (isBrush) {
                auto& chunkMap = (layer == LAYER_OPAQUE)
                    ? this->brushRenderer->solidChunks : this->brushRenderer->transparentChunks;
                chunkMap[nid] = Model3DVersion{slotIdx, version};
            } else {
                auto& chunkMap = (layer == LAYER_OPAQUE)
                    ? this->mainSolidChunks : this->mainLiquidChunks;
                chunkMap[nid] = Model3DVersion{slotIdx, version};
            }
        },
        // onFinestPublished: grass chunks (main scene only) drive vegetation
        // from their level-0 (finest) geometry.
        [this, app](NodeID nid, const Geometry& geom, bool isBrush) {
            if (!isBrush && this->vegetationRenderer) this->vegetationRenderer->generateForChunk(app, nid, geom);
        });

    // ── Orphan + grace sweeps (one sweep for ALL old slots) ──────────────────
    // Brush orphans: staged old slots whose NodeID no longer appears in the new
    // set. These chunks were removed in the rebuild and have stale geometry at
    // the old brush position — they must not linger as visible garbage.
    for (auto& [nid, oldSlot] : oldSolidSlots) {
        if (!matchedNids.count(nid) && brushRenderer->solidChunks.find(nid) == brushRenderer->solidChunks.end())
            brushRenderer->getSolidIR().removeMeshSlotted(oldSlot);
    }
    for (auto& [nid, oldSlot] : oldTransparentSlots) {
        if (!matchedNids.count(nid) && brushRenderer->transparentChunks.find(nid) == brushRenderer->transparentChunks.end())
            brushRenderer->getLiquidIR().removeMeshSlotted(oldSlot);
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
    // Packed pools: the TOTAL shared element budget is the per-chunk ceiling
    // times the chunk count. Actual allocation is data-driven (chunks pack
    // into free spans), so this is the worst-case footprint.
    const uint64_t solidVertBytes = static_cast<uint64_t>(maxSolidChunks) * vertexBytesPerChunk;
    const uint64_t solidIdxBytes  = static_cast<uint64_t>(maxSolidChunks) * indexBytesPerChunk;
    const uint64_t waterVertBytes = static_cast<uint64_t>(maxWaterChunks) * vertexBytesPerChunk;
    const uint64_t waterIdxBytes  = static_cast<uint64_t>(maxWaterChunks) * indexBytesPerChunk;

    mainSolidRenderer->getIndirectRenderer().initSlots(app, maxSolidChunks,
                                                       static_cast<uint32_t>(solidVertBytes),
                                                       static_cast<uint32_t>(solidIdxBytes));
    mainLiquidRenderer->getIndirectRenderer().initSlots(app, maxWaterChunks,
                                                        static_cast<uint32_t>(waterVertBytes),
                                                        static_cast<uint32_t>(waterIdxBytes));

    std::cout << "[SceneRenderer] packed pools: solid=" << maxSolidChunks
              << " blocks / water=" << maxWaterChunks
              << " blocks (total " << (solidVertBytes >> 20) << " MB + "
              << (solidIdxBytes >> 20) << " MB index solid, "
              << (waterVertBytes >> 20) << " MB + " << (waterIdxBytes >> 20)
              << " MB index water, device-local)" << std::endl;

}

bool SceneRenderer::processChunkSlotted(Layer layer, NodeID nid,
                                         const OctreeNodeData& nd,
                                         const Geometry& geom, uint32_t version)
{
    {
        // All streams share ONE pending queue (main solid/water + brush
        // solid/water); entries are tagged isBrush=false here since this path
        // feeds the main scene.
        std::lock_guard<std::mutex> lock(pendingMeshMutex);
        Octree::LoDMesh lod = {geom, /*lod*/ 0, /*version*/ version, nd.cube.getLength().x,
                               nd.cube.getMin(), nd.cube.getMax()};
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

    // Every cell with a chunkLod (stored 1..5, the +1-shifted uint8_t space)
    // publishes its mesh — each chunk carries its own level and the GPU cull
    // keeps only the chunk level matching the camera distance band. Coarse
    // ancestor cells that come back empty (no zero crossing at that
    // resolution) simply never reach the publisher (LocalScene's walk and the
    // handler below filter empty geometry).
    const uint8_t chunkLod = nodeData.node ? nodeData.node->getChunkLod() : 0;
    if (chunkLod < 1) return;

    // Shared clipmap column anchor: every rung (chunk) of a column must use the
    // SAME base so the clipmap anchor (rootSide) nests and exactly one rung is
    // selected per region (no overdraw across LODs). The base is the FINEST
    // chunk's min corner of the column, i.e. the chunk center snapped to the
    // finest-chunk grid. (cellSize / 2^(chunkLod-1) is the finest chunk size,
    // since chunkLod==1 is the finest published rung; rootSide is invariant in
    // the gate, so only base needs to be column-consistent.)
    const float chunkCell = nodeData.cube.getLengthX();
    const float finestChunkSize = chunkCell / glm::exp2(static_cast<float>(chunkLod) - 1.0f);
    const glm::vec3 colCenter = nodeData.cube.getCenter();
    const glm::vec3 columnBase = glm::floor(colCenter / finestChunkSize) * finestChunkSize;

    // NOTE: the walk emits one callback per cell on the root path (each
    // ancestor at its own level); the cube passed is the EMITTING cell's own
    // cube — the band center (boundsMin/Max) and the meta cellSize must come
    // from it. The clipmap column anchor (boundsBase) is the shared finest-chunk
    // min of the column, computed once above, never the emitting cell's own min
    // (otherwise parent/child rungs would compute different anchors and overlap).
    scene.requestModel3D(layer, nodeData, [&layer,&onGeometry,&nodeData,&columnBase](const Geometry& geo, uint8_t lod, uint version, uintptr_t emittingNodeId, const BoundingCube& cube, const BoundingCube& /*baseCube*/) {
        Octree::LoDMesh lm;
        lm.geom = geo;
        lm.lod = lod;
        lm.version = version;
        lm.cellSize = cube.getLength().x;
        lm.boundsMin = cube.getMin();
        lm.boundsMax = cube.getMax();
        lm.boundsBase = columnBase;
        onGeometry(layer, reinterpret_cast<NodeID>(emittingNodeId), lm);
    }, poolOverride);

    // SDF debug cubes: collected through the SAME node walk as the solid meshes
    // (requestSDFCubes walks the chunk subtree and emits lod==1 nodes that carry a
    // drawable SDF face). Accumulate them for this chunk and publish to the
    // DebugSDFRenderer, which renders them via the terrain IndirectRenderer's
    // folded SDF draw stream.
    if (debugSDFRenderer) {
        std::vector<DebugSDFRenderer::CubeSDF> sdfCubes;
        std::mutex sdfMtx;
        scene.requestSDFCubes(layer, nodeData,
            [&sdfCubes, &sdfMtx, &nodeData, &columnBase](const BoundingCube& cube, const std::array<float, 8>& sdf,
                                 uint8_t /*lod*/, uint /*version*/, uintptr_t /*emittingNodeId*/, uint32_t brushIndex) {
                DebugSDFRenderer::CubeSDF c;
                c.cube = cube;
                c.sdf = sdf;
                c.brushIndex = static_cast<int>(brushIndex);
                // LoD meta for the SDF cull's clipmap band gate (mirrors the solid chunk
                // entry): cellSize = chunk cube side, level = chunkLod rung, base = chunk min
                // corner (shared column anchor). The SDF cubes are emitted at the chunk's
                // finest surface rung, but the gate selects the rung by the CHUNK's LoD so
                // parent/child chunk SDF cubes never overlap.
                c.cellSize = nodeData.cube.getLengthX();
                c.level    = static_cast<int>(nodeData.node->getChunkLod());
                c.base     = columnBase;
                std::lock_guard<std::mutex> lk(sdfMtx);
                sdfCubes.push_back(std::move(c));
            }, poolOverride);
        debugSDFRenderer->updateCubesForChunk(nid, sdfCubes);
    }

    // Mesh bounding boxes: collected through the SAME node walk as the solid
    // meshes (requestBoundingBoxes walks the chunk subtree and emits every
    // surface node whose ladder level equals its chunk LoD, so the debug overlay
    // shows all node boxes at the chunk's current resolution, not one chunk-sized
    // box). Accumulate them per chunk and publish to the bounding-box renderer,
    // which renders them via the terrain IndirectRenderer's folded bbox stream.
    if (boundingBoxRenderer) {
        std::vector<DebugCubeRenderer::CubeWithColor> bbCubes;
        std::mutex bbMtx;
        scene.requestBoundingBoxes(layer, nodeData,
            [&bbCubes, &bbMtx, layer, &nodeData, &columnBase](const BoundingCube& cube) {
                DebugCubeRenderer::CubeWithColor c;
                c.cube = BoundingBox(cube.getMin(), cube.getMax());
                c.color = (layer == LAYER_OPAQUE)
                    ? glm::vec3(0.0f, 1.0f, 0.0f)
                    : glm::vec3(0.0f, 0.5f, 1.0f);
                // LoD meta for the bbox cull's clipmap band gate (mirrors the solid
                // chunk entry): cellSize = node cube side, level = chunkLod rung,
                // base = column finest-chunk min (shared column anchor).
                c.cellSize = cube.getLengthX();
                c.level    = static_cast<int>(nodeData.node->getChunkLod());
                c.base     = columnBase;
                std::lock_guard<std::mutex> lk(bbMtx);
                bbCubes.push_back(std::move(c));
            }, poolOverride);
        boundingBoxRenderer->setBoundingBoxesForChunk(nid, bbCubes);
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

