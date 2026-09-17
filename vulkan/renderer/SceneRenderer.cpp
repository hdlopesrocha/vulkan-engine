#include "SceneRenderer.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "SceneDescriptorLayout.hpp"
#include "../ubo/SkyUniform.hpp"
#include "../../utils/Settings.hpp"


#include <stdexcept>
#include "../../utils/LocalScene.hpp"
#include "../includes/locations.hpp"
#include "../../math/ContainmentType.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
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
    destroySSRSamplers(app);
    // Hybrid RT teardown (acceleration structures + pipeline + outputs) while
    // the device is alive. RT descriptors in set 0 dangle after this, but the
    // device is being torn down anyway (shutdown path only).
    if (rayTracing) {
        rayTracing->cleanup(app);
        rayTracing.reset();
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
    destroyDescriptorBuffers(app);
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
    if (vegetationRenderer) {
        vegetationRenderer->createRenderTargets(app, width, height);
    }
    if (brushRenderer) {
        brushRenderer->onSwapchainResized(app, width, height);
    }
    if (mainLiquidRenderer) {
        mainLiquidRenderer->createRenderTargets(app, width, height);
        // Recreate back-face targets owned by SceneRenderer (the 360 cubemap
        // path is removed — reflections are hardware ray tracing now).
        if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, width, height);
        if (debugSDFRenderer) debugSDFRenderer->createRenderTargets(app, width, height);
        if (boundingBoxRenderer) boundingBoxRenderer->createRenderTargets(app, width, height);
        // Hybrid RT outputs are swapchain-sized (half-res): recreate + re-point
        // bindings 15/16 everywhere (views are new handles). TLAS/params/meta
        // handles are stable across resizes (no rewrite needed).
        if (rayTracing && rayTracing->isSupported()) {
            rayTracing->onSwapchainResized(app, width, height);
            VkImageView reflView = rayTracing->getReflectionView();
            VkImageView refrView = rayTracing->getRefractionView();
            VkSampler rtSampler = rayTracing->getLinearSampler();
            if (reflView != VK_NULL_HANDLE && refrView != VK_NULL_HANDLE &&
                rtSampler != VK_NULL_HANDLE) {
                auto rewriteRTView = [&](VkDescriptorSet ds, uint32_t binding, VkImageView view) {
                    if (ds == VK_NULL_HANDLE) return;
                    DescriptorWriter(app->getDevice())
                        .writeImage(ds, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    rtSampler, view, VK_IMAGE_LAYOUT_GENERAL)
                        .flush();
                };
                rewriteRTView(app->getStaticDescriptorSet(), 15, reflView);
                rewriteRTView(app->getStaticDescriptorSet(), 16, refrView);
                for (size_t fi = 0; fi < app->getMainDescriptorSetCount(); ++fi)
                    rewriteRTView(app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi)), 15, reflView),
                    rewriteRTView(app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi)), 16, refrView);
                for (auto ds : shadowDescriptorSets)
                    rewriteRTView(ds, 15, reflView), rewriteRTView(ds, 16, refrView);
            }
            refreshRTSceneViews(app);
        }
    }
    if (postProcessRenderer) {
        postProcessRenderer->setRenderSize(width, height);
    }
    if (skyRenderer) {
        skyRenderer->destroyOffscreenTargets(app);
        skyRenderer->createOffscreenTargets(app, width, height);
        // Sky views feed the RT pipeline miss shader (per-slot arrays) — the
        // handles changed above, so refresh the RT scene views too.
        if (rayTracing && rayTracing->isSupported()) refreshRTSceneViews(app);
    }
    // Solid SSR sources are new handles after the solid render targets above
    // were recreated; re-point bindings 19/20 (resize is idle-safe).
    writeSSRBindings(app);
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
    // Keep the texture arrays for RT proxy albedo lookups (per-layer averages).
    textureArrays_ = textureArrayManager;

    // Representative water surface tint for reflected water: computed with the
    // water's OWN params (shallow/deep colors, caustic depth scale, max
    // thickness, absorption) using the exact water.frag formula, so reflected
    // water carries the real water color — no hardcoded values.
    {
        const WaterParams& wp = waterParams.empty() ? WaterParams{} : waterParams[0];
        const float thickness = std::max(wp.maxThickness, 0.0f);
        const float tintDepthScale = std::max(wp.causticDepthScale, 0.0001f);
        const float volumeFactor = 1.0f - std::exp(-thickness / tintDepthScale);
        glm::vec3 waterTintColor = glm::mix(wp.shallowColor, wp.deepColor, volumeFactor);
        // Beer-Lambert absorption (water.frag): the tint seen through the
        // water column is attenuated.
        const glm::vec3 transmittance = glm::exp(-glm::min(
            wp.absorption * std::max(thickness * wp.absorptionScale, 0.0f),
            glm::vec3(2.5f)));
        waterReflectionTint_ = waterTintColor * transmittance;
    }

    // Initialize the async streaming orchestrator. It is the ONLY transfer
    // engine: solid/water incremental chunk uploads route through it (32
    // concurrent 4 MiB staging slots, no per-frame cap). slotSize =
    // chunkVertexBytes + chunkIndexBytes = 2 MiB + 2 MiB = 4 MiB, covering the
    // largest chunk (512 KB vertex + 128 KB index worst case with headroom);
    // each chunk uploads in a single slot (IndirectRenderer::uploadSlot
    // rejects chunks larger than one slot).
    streamer.init(app,
                  /*chunkVertexBytes*/ 2u << 20,
                  /*chunkIndexBytes*/  2u << 20,
                  /*stagingSlots*/     32,
                  /*initialChunkSlots*/ 8,
                  /*workersPerCategory*/ 2);

    // Route solid/water IndirectRenderer incremental copies through the manager.
    mainSolidRenderer->getIndirectRenderer().setUploadManager(
        &streamer.uploadManager(), streaming::StreamCategory::Solid);
    mainLiquidRenderer->getIndirectRenderer().setUploadManager(
        &streamer.uploadManager(), streaming::StreamCategory::Water);
    if (brushRenderer) {
        brushRenderer->getSolidIR().setUploadManager(
            &streamer.uploadManager(), streaming::StreamCategory::Solid);
        brushRenderer->getLiquidIR().setUploadManager(
            &streamer.uploadManager(), streaming::StreamCategory::Water);
    }

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
    // Own offscreen framebuffer for vegetation (decoupled from the solid pass so
    // it can be rendered on a parallel async command buffer).
    if (vegetationRenderer) {
        vegetationRenderer->destroyRenderTargets(app);
        vegetationRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    }

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

    // Own offscreen framebuffers for the debug SDF cubes and mesh bounding boxes so
    // they can be rendered on their own parallel async command buffers (composited
    // by postprocess.frag against the solid scene depth).
    if (debugSDFRenderer) debugSDFRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (boundingBoxRenderer) boundingBoxRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

    // Create per-frame main uniform buffers (TRANSFER_DST for vkCmdCopyBuffer from staging)
    size_t dsCount = app->getMainDescriptorSetCount();
    if (dsCount == 0) dsCount = 1;
    mainUniformBuffers.clear();
    mainUniformBuffers.resize(dsCount);
    // Descriptor-buffer sources need a device address for vkGetDescriptorEXT.
    VkBufferUsageFlags uboUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (app->useDescriptorBuffer())
        uboUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    for (size_t i = 0; i < dsCount; ++i) {
        mainUniformBuffers[i] = app->createBuffer(sizeof(UniformObject), uboUsage,
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
    writesImg.reserve(12);  // max image descriptors: 5 texture arrays + 3 shadow maps + 2 RT outputs (+1 spare; MUST exceed the emplace count — writes[] stores raw pImageInfo pointers into this vector, so any reallocation dangles them)
    writesBuf.reserve(8);  // materials SSBO + water params + water render UBO + RT params + RT meta + scene prim bases + scene albedo (+1 spare; same no-realloc requirement as writesImg)

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
    VkBufferUsageFlags waterParamsUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (app->useDescriptorBuffer())
        waterParamsUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    waterParamsBuffer_ = app->createBuffer(paramsBufferSize, waterParamsUsage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    waterParamsBufferSize_ = static_cast<VkDeviceSize>(paramsBufferSize);
    // Create scene-owned water sub-renderers. Back-face renderpass must exist
    // before water pipelines are created, so create it first.
    backFaceRenderer = std::make_unique<WaterBackFaceRenderer>();

    // Initialize WaterRenderer (creates its pipeline layout and initializes the param SSBO)
    mainLiquidRenderer->init(app, waterParamsBuffer_, waterParams, layerCount);

    // Now that WaterRenderer has created its pipeline layout, allow the
    // back-face renderer to create pipelines that depend on it.
    if (backFaceRenderer) backFaceRenderer->createPipelines(app, mainLiquidRenderer->getWaterGeometryPipelineLayout());
    // Create back-face render targets early so their image views are
    // available before the first frame's water pass attempts to bind them.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (debugSDFRenderer) debugSDFRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (boundingBoxRenderer) boundingBoxRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    // Hybrid RT: no cubemap targets — init the proxy BLAS/TLAS + water RT
    // pipeline instead. Unsupported devices get a null-safe stub (raster +
    // CSM fallback). Must run before the set-0 RT bindings below are written.
    rayTracing = std::make_unique<RayTracingResources>();
    rayTracing->init(app, app->getWidth(), app->getHeight());

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

    // ── Hybrid RT set-0 bindings (written once; handles stable) ──────────
    // 14 = TLAS (valid object from init; contents built on first
    //      buildIfNeeded — shaders gate sampling on rt.debug.y == tlasReady).
    // 15/16 = water RT reflection / refraction+thickness outputs (sampled by
    //      water.frag; GENERAL layout shared with the RT pipeline's writes —
    //      sampled descriptors use GENERAL to match, avoiding layout churn).
    // 17 = RT params UBO (contents stream per frame via updateRTParams).
    // 18 = RT proxy metadata (hit shading for inline ray queries).
    // When RT is unsupported the views/buffers are NULL and the writes are
    // skipped — shaders fall back to sky/CSM (tlasReady stays 0).
    if (rayTracing && rayTracing->isSupported() && rayTracing->getTLAS() != VK_NULL_HANDLE) {
        // The TLAS object is created once and never recreated (in-place
        // rebuilds), so this handle stays valid for app lifetime. The static
        // replay loop below only handles image/buffer writes; the AS write is
        // applied explicitly right after it (see tlasMirror_ + writeTlasBinding).
        // Pushed here so the per-frame copy enumeration picks up binding 14.
        tlasMirror_ = rayTracing->getTLAS();
        VkWriteDescriptorSet tlasWrite{};
        tlasWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tlasWrite.dstSet = staticDs;
        tlasWrite.dstBinding = 14;
        tlasWrite.descriptorCount = 1;
        tlasWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes.push_back(tlasWrite);
    }
    auto addRTImageWrite = [&](uint32_t binding, VkImageView view) {
        if (!rayTracing || !rayTracing->isSupported() || view == VK_NULL_HANDLE) return;
        VkDescriptorImageInfo& info = writesImg.emplace_back();
        info.sampler = rayTracing->getLinearSampler();
        info.imageView = view;
        info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = staticDs;
        w.dstBinding = binding;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &info;
        writes.push_back(w);
    };
    if (rayTracing && rayTracing->isSupported()) {
        addRTImageWrite(15, rayTracing->getReflectionView());
        addRTImageWrite(16, rayTracing->getRefractionView());
        if (rayTracing->getParamsBuffer() != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo& rtParamsInfo = writesBuf.emplace_back(
                rayTracing->getParamsBuffer(), 0, sizeof(RayTracingParams));
            VkWriteDescriptorSet rtParamsWrite{};
            rtParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            rtParamsWrite.dstSet = staticDs;
            rtParamsWrite.dstBinding = 17;
            rtParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            rtParamsWrite.descriptorCount = 1;
            rtParamsWrite.pBufferInfo = &rtParamsInfo;
            writes.push_back(rtParamsWrite);
        }
        if (rayTracing->getMetaBuffer() != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo& rtMetaInfo = writesBuf.emplace_back(
                rayTracing->getMetaBuffer(), 0, VK_WHOLE_SIZE);
            VkWriteDescriptorSet rtMetaWrite{};
            rtMetaWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            rtMetaWrite.dstSet = staticDs;
            rtMetaWrite.dstBinding = 18;
            rtMetaWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            rtMetaWrite.descriptorCount = 1;
            rtMetaWrite.pBufferInfo = &rtMetaInfo;
            writes.push_back(rtMetaWrite);
        }
        // Real scene-geometry reflection lookups (bindings 21/22): contents are
        // host-updated per rebuild; the descriptors are stable (preallocated
        // buffers, never resized).
        if (rayTracing->getSceneGeomPrimBaseBuffer() != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo& primBaseInfo = writesBuf.emplace_back(
                rayTracing->getSceneGeomPrimBaseBuffer(), 0, VK_WHOLE_SIZE);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = staticDs;
            w.dstBinding = 21;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &primBaseInfo;
            writes.push_back(w);
        }
        if (rayTracing->getSceneGeomMetaBuffer() != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo& sceneMetaInfo = writesBuf.emplace_back(
                rayTracing->getSceneGeomMetaBuffer(), 0, VK_WHOLE_SIZE);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = staticDs;
            w.dstBinding = 22;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &sceneMetaInfo;
            writes.push_back(w);
        }
        if (rayTracing->getSceneGeomInfoBuffer() != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo& sceneGeomInfo = writesBuf.emplace_back(
                rayTracing->getSceneGeomInfoBuffer(), 0, VK_WHOLE_SIZE);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = staticDs;
            w.dstBinding = 23;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &sceneGeomInfo;
            writes.push_back(w);
        }
    }

    // ── Static descriptor set ──
    // Write scene-static bindings once. These resources rarely change
    // (texture arrays, materials, shadow maps, sky, water params, RT handles).
    // Per-frame descriptor sets will copy these via VkCopyDescriptorSet.
    {
        VkDescriptorSet staticSet = app->getStaticDescriptorSet();
        if (staticSet != VK_NULL_HANDLE) {
            DescriptorWriter staticWriter(app->getDevice());
            // Replay accumulated image/buffer writes into the static set
            for (auto &w : writes) {
                if (w.dstBinding == 0) continue; // binding 0 is per-frame
                if (w.dstBinding == 14) continue; // TLAS: explicit AS write below
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
            // TLAS binding 14: acceleration-structure write (pNext chain —
            // DescriptorWriter only handles image/buffer infos).
            if (tlasMirror_ != VK_NULL_HANDLE) writeTlasBinding(app, staticSet);
        }
    }

    // ── Per-frame descriptor sets ──
    // For each frame, copy bindings 1-13 from the static set and write binding 0
    // (per-frame UBO) separately using DescriptorWriter. Init-time only: after
    // this point the per-frame sets are never touched in the render loop —
    // per-frame UBO contents stream via host memcpy into the already-bound
    // buffers. Batched into one vkUpdateDescriptorSets call for all frames.
    {
        VkDescriptorSet staticSet = app->getStaticDescriptorSet();
        std::vector<VkCopyDescriptorSet> copies;
        copies.reserve(mainUniformBuffers.size() * 14);
        for (size_t fi = 0; fi < mainUniformBuffers.size(); ++fi) {
            VkDescriptorSet dstSet = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));
            if (dstSet == VK_NULL_HANDLE) continue;

            // Collect all static bindings (1-13) for copy from staticDs.
            // The writes template excludes binding 6 (Sky UBO was written by
            // skyRenderer->init), so we enumerate the union explicitly.
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
        }
        if (!copies.empty()) {
            DescriptorUpdateStats::noteUpdate(copies.size());
            vkUpdateDescriptorSets(app->getDevice(), 0, nullptr,
                                   static_cast<uint32_t>(copies.size()), copies.data());
        }
        for (size_t fi = 0; fi < mainUniformBuffers.size(); ++fi) {
            VkDescriptorSet dstSet = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));
            if (dstSet == VK_NULL_HANDLE) continue;

            // Write per-frame UBO (binding 0) using DescriptorWriter
            DescriptorWriter writer(app->getDevice());
            writer.writeBuffer(dstSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                               mainUniformBuffers[fi].buffer, 0, sizeof(UniformObject));

            writer.flush();
        }
    }

    // ── Solid SSR source bindings (19/20) ─────────────────────────────────
    // Per-frame main sets sample the previous slot's solid color/depth for
    // precise screen-space reflections. Written after the static→main copies
    // above so the copies (which don't carry 19/20) cannot clobber them.
    initSSRSamplers(app);
    writeSSRBindings(app);

    // ── Descriptor buffers (Phase 1, step 1–2) ────────────────────────────
    // Allocate 3 host-visible descriptor buffers and populate every static
    // set-0 binding via vkGetDescriptorEXT. No-op when the device lacks
    // VK_EXT_descriptor_buffer (classic sets above stay authoritative).
    // Snapshot the signature so the first allocation-listener callback with
    // unchanged resources is a no-op (0 vkUpdateDescriptorSets in steady state).
    initDescriptorBuffers(app);
    writeStaticDescriptorsToBuffers(app, textureArrayManager);
    if (hasDescriptorBuffers())
        lastStaticSignature_ = currentStaticSignature(textureArrayManager);

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
                                              backFaceRenderer.get(), waterWireframe.get());
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

        // Hybrid RT mirror (shadow pass declares the same set-0 RT bindings via
        // main.frag; the shadow fast-path early-outs before sampling, but the
        // descriptors must still be valid). No cubemap binding 11 (removed).
        if (rayTracing && rayTracing->isSupported()) {
            if (rayTracing->getLinearSampler() != VK_NULL_HANDLE) {
                addImg(15, rayTracing->getLinearSampler(), rayTracing->getReflectionView(), VK_IMAGE_LAYOUT_GENERAL);
                addImg(16, rayTracing->getLinearSampler(), rayTracing->getRefractionView(), VK_IMAGE_LAYOUT_GENERAL);
            }
            if (rayTracing->getParamsBuffer() != VK_NULL_HANDLE)
                wr.writeBuffer(ds, 17, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                               rayTracing->getParamsBuffer(), 0, sizeof(RayTracingParams));
            if (rayTracing->getMetaBuffer() != VK_NULL_HANDLE)
                wr.writeBuffer(ds, 18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               rayTracing->getMetaBuffer(), 0, VK_WHOLE_SIZE);
        }

        wr.writeBuffer(ds, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
        wr.writeBuffer(ds, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);
        wr.writeBuffer(ds, 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                       mainLiquidRenderer->getWaterRenderUBO().buffer, 0, sizeof(WaterRenderUBO));
        wr.flush();
        // TLAS binding 14 (pNext chain — outside DescriptorWriter).
        if (tlasMirror_ != VK_NULL_HANDLE) writeTlasBinding(app, ds);
    }
    // Shadow descriptor set handles are stable after init (subsequent writes
    // only update them in place), so ShadowRenderer can cache them once.
    // Pre-create all parallel cascade resources now (UBOs, cascade descriptor
    // sets, internal semaphores) so renderParallel never falls back — it
    // asserts cascadeSetsBuilt_ on every frame.
    if (shadowMapper) {
        shadowMapper->setShadowDescriptorSets(shadowDescriptorSets);
        shadowMapper->ensureShadowParallelResources(app);
    }

    // Register listener so we update the main descriptor set when texture arrays are allocated later
    if (textureArrayManager) {
        textureArrayManager->addAllocationListener([this, app, textureArrayManager]() {
            this->updateTextureDescriptorSet(app, textureArrayManager);
            // Texture (re)allocation changed the per-layer albedo averages:
            // refresh RT proxy albedos on the render thread (flag consumed in
            // processPendingMeshes; setProxies dedupes when nothing changed).
            this->proxyAlbedoRefresh_.store(true, std::memory_order_relaxed);
        });
    }
    mainLiquidRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

    // Ensure back-face render targets are created as well so the
    // `backFaceDepthView` is valid before the first frame's water pass.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (debugSDFRenderer) debugSDFRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (boundingBoxRenderer) boundingBoxRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

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

    // Initialize post-process renderer (composites scene + water into swapchain).
    // Sky offscreen targets are created above (before this point) and always
    // available from here on, so PostProcessRenderer assumes a valid skyView.
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

    // Hybrid RT final wiring: water's non-async prepare path needs the RT
    // outputs, and the RT pipeline's per-slot sets need the water-depth + sky
    // views created above (water render targets + sky offscreen targets).
    if (mainLiquidRenderer) mainLiquidRenderer->setRTResources(rayTracing.get());
    refreshRTSceneViews(app);
}

// ── Descriptor-buffer migration (Phase 1) ───────────────────────────────────
// Step 1: allocate 3 host-visible descriptor buffers (one per frame). Size =
// the driver's set-0 layout size (vkGetDescriptorSetLayoutSizeEXT on the
// DESCRIPTOR_BUFFER_BIT query layout), aligned to
// descriptorBufferOffsetAlignment. Per-binding offsets come from
// vkGetDescriptorSetLayoutBindingOffsetEXT on the same layout.
void SceneRenderer::initDescriptorBuffers(VulkanApp* app) {
    destroyDescriptorBuffers(app);
    if (!app || !app->useDescriptorBuffer()) return;
    if (!app->fpGetDescriptorSetLayoutSizeEXT || !app->fpGetDescriptorSetLayoutBindingOffsetEXT)
        return;
    if (!app->sceneDescriptorLayout) return;
    VkDescriptorSetLayout query = app->sceneDescriptorLayout->descriptorBufferQueryLayout();
    if (query == VK_NULL_HANDLE) return;

    VkDeviceSize rawSize = 0;
    app->fpGetDescriptorSetLayoutSizeEXT(app->getDevice(), query, &rawSize);
    VkDeviceSize align = static_cast<VkDeviceSize>(app->descriptorBufferProps.descriptorBufferOffsetAlignment);
    if (align == 0) align = 1;
    VkDeviceSize setSize = (rawSize + align - 1) & ~(align - 1);
    if (setSize == 0) return;

    const uint32_t frames = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    descBuffers_.buffers.reserve(frames);
    descBuffers_.addresses.reserve(frames);
    for (uint32_t i = 0; i < frames; ++i) {
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
        VkDeviceAddress addr = vkGetBufferDeviceAddress(app->getDevice(), &addrInfo);
        if (addr == 0) {
            app->destroyBuffer(b);
            destroyDescriptorBuffers(app);
            return;
        }
        descBuffers_.buffers.push_back(b);
        descBuffers_.addresses.push_back(addr);
    }
    for (uint32_t binding = 0; binding < 19; ++binding) {
        // Binding 11 was removed (legacy cubemap); the query layout carries no
        // entry for it — skip so the offset query never touches a missing
        // binding (VUID-vkGetDescriptorSetLayoutBindingOffsetEXT-binding-08021).
        if (binding == 11) { descBuffers_.bindingOffsets[binding] = 0; continue; }
        VkDeviceSize off = 0;
        app->fpGetDescriptorSetLayoutBindingOffsetEXT(app->getDevice(), query, binding, &off);
        descBuffers_.bindingOffsets[binding] = off;
    }
    descBuffers_.setSize = setSize;
    descBuffers_.ready = true;
    printf("[SceneRenderer] descriptor buffers: %u frames x %llu bytes (set 0)\n",
           frames, (unsigned long long)setSize);
}

void SceneRenderer::destroyDescriptorBuffers(VulkanApp* app) {
    if (descBuffers_.buffers.empty()) {
        descBuffers_.addresses.clear();
        descBuffers_.setSize = 0;
        descBuffers_.ready = false;
        return;
    }
    if (app) {
        for (auto& b : descBuffers_.buffers) {
            if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
            else b = {};
        }
    } else {
        for (auto& b : descBuffers_.buffers) b = {};
    }
    descBuffers_.buffers.clear();
    descBuffers_.addresses.clear();
    descBuffers_.setSize = 0;
    descBuffers_.ready = false;
}

// Step 2/4: populate every frame's descriptor buffer with the static set-0
// bindings (1–13) plus the per-frame UBO address (binding 0) via
// DescriptorBuffer::writeBuffer / writeImage (vkGetDescriptorEXT host writes —
// no vkUpdateDescriptorSets, no driver validation work on this path).
// Host-visible + coherent memory makes the writes visible without explicit
// barriers; this runs at init / on resource change (same serialization as the
// classic writes it mirrors), never in the render loop.
void SceneRenderer::writeStaticDescriptorsToBuffers(VulkanApp* app, TextureArrayManager* textureArrayManager) {
    if (!app || !descBuffers_.ready || !app->useDescriptorBuffer()) return;
    if (!app->fpGetDescriptorEXT) return;
    const auto& props = app->descriptorBufferProps;
    const size_t align = props.descriptorBufferOffsetAlignment ? props.descriptorBufferOffsetAlignment : 1;

    Buffer skyUBO{};
    if (skyRenderer) skyUBO = skyRenderer->getSkyUniformBuffer();
    Buffer waterRenderUBO{};
    if (mainLiquidRenderer) waterRenderUBO = mainLiquidRenderer->getWaterRenderUBO();
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkImageView shadowViews[3] = {};
    if (shadowMapper) {
        shadowSampler = shadowMapper->getShadowMapSampler();
        shadowViews[0] = shadowMapper->getShadowMapView(0);
        shadowViews[1] = shadowMapper->getShadowMapView(1);
        shadowViews[2] = shadowMapper->getShadowMapView(2);
    }

    uint32_t failures = 0;
    for (size_t fi = 0; fi < descBuffers_.buffers.size(); ++fi) {
        Buffer& dst = descBuffers_.buffers[fi];
        if (dst.buffer == VK_NULL_HANDLE || dst.mappedData == nullptr) { ++failures; continue; }
        DescriptorBuffer view(app->getDevice(), app->fpGetDescriptorEXT,
                              app->fpGetDescriptorSetLayoutBindingOffsetEXT,
                              dst.mappedData, static_cast<size_t>(descBuffers_.setSize), align);
        auto imgSize = props.combinedImageSamplerDescriptorSize;
        auto uboSize = props.uniformBufferDescriptorSize;
        auto ssboSize = props.storageBufferDescriptorSize;
        auto wImg = [&](uint32_t binding, VkSampler sampler, VkImageView view_, VkImageLayout layout) {
            if (view_ == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            if (!view.writeImage(static_cast<size_t>(descBuffers_.bindingOffsets[binding]),
                                 imgSize, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 sampler, view_, layout))
                ++failures;
        };
        auto wBuf = [&](uint32_t binding, size_t dataSize, VkDescriptorType type,
                        VkBuffer src, VkDeviceSize range) {
            if (src == VK_NULL_HANDLE) return;
            if (!view.writeBuffer(static_cast<size_t>(descBuffers_.bindingOffsets[binding]),
                                  dataSize, type, src, 0, range))
                ++failures;
        };

        // Binding 0: this frame's UBO address (contents stream via memcpy).
        if (fi < mainUniformBuffers.size())
            wBuf(0, uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 mainUniformBuffers[fi].buffer, sizeof(UniformObject));
        // Bindings 1–3, 12–13: texture arrays.
        if (textureArrayManager) {
            wImg(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wImg(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wImg(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wImg(12, textureArrayManager->roughnessSampler, textureArrayManager->roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wImg(13, textureArrayManager->aoSampler, textureArrayManager->aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        // Bindings 4, 8, 9: shadow cascades.
        wImg(4, shadowSampler, shadowViews[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        wImg(8, shadowSampler, shadowViews[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        wImg(9, shadowSampler, shadowViews[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        // Binding 5: materials SSBO. Binding 7: water params SSBO.
        // NOTE: vkGetDescriptorEXT forbids VK_WHOLE_SIZE ranges
        // (VUID-VkDescriptorAddressInfoEXT-nullDescriptor-08939), so exact
        // byte sizes are passed; a zero/unknown size skips the write.
        VkDeviceSize materialsSize = (materialManagerPtr) ? static_cast<VkDeviceSize>(materialManagerPtr->bufferSize()) : 0;
        if (materialsSize > 0)
            wBuf(5, ssboSize, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, materialsBuffer.buffer, materialsSize);
        if (waterParamsBufferSize_ > 0)
            wBuf(7, ssboSize, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, waterParamsBuffer_.buffer, waterParamsBufferSize_);
        // Binding 6: sky UBO. Binding 10: water render UBO. (Binding 11 removed:
        // legacy cubemap — reflections are hardware ray tracing now.)
        wBuf(6, uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, skyUBO.buffer, sizeof(SkyUniform));
        wBuf(10, uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, waterRenderUBO.buffer, sizeof(WaterRenderUBO));
        // Hybrid RT mirrors (classic sets stay authoritative while the
        // descriptor-buffer bind path is inactive). Binding 14 (TLAS) has no
        // mirror (acceleration structures stay on the classic write path).
        if (rayTracing && rayTracing->isSupported()) {
            wImg(15, rayTracing->getLinearSampler(), rayTracing->getReflectionView(), VK_IMAGE_LAYOUT_GENERAL);
            wImg(16, rayTracing->getLinearSampler(), rayTracing->getRefractionView(), VK_IMAGE_LAYOUT_GENERAL);
            wBuf(17, uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 rayTracing->getParamsBuffer(), sizeof(RayTracingParams));
            VkDeviceSize metaSize = VkDeviceSize(sizeof(RTProxyMeta)) * RayTracingResources::kMaxProxies;
            if (metaSize > 0)
                wBuf(18, ssboSize, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     rayTracing->getMetaBuffer(), metaSize);
        }
    }
    if (failures > 0)
        std::cerr << "[SceneRenderer] descriptor-buffer static writes: " << failures << " skipped/failed (missing source or address)\n";
}

// Step 3: per-frame bind of set 0 from the descriptor buffer. Activates with
// the main-layout DESCRIPTOR_BUFFER_BIT cutover (all set-0 classic binds must
// be removed at the same time: VUID-08010 forbids classic binds of sets from
// a DESCRIPTOR_BUFFER_BIT layout, and SetDescriptorBufferOffsets invalidates
// classic set-1/set-2 binds in the same draw). Until then this is the
// documented bind point — kept uncalled so the classic path stays valid.
void SceneRenderer::bindSet0ForFrame(VkCommandBuffer cmd, VulkanApp* app,
                                     VkPipelineLayout layout, uint32_t frameIndex) const {
    if (!app || cmd == VK_NULL_HANDLE || layout == VK_NULL_HANDLE) return;
    if (!descriptorBufferBindActive(app)) return;
    if (descBuffers_.buffers.empty() || descBuffers_.addresses.empty()) return;
    const uint32_t idx = frameIndex % static_cast<uint32_t>(descBuffers_.buffers.size());
    VkDescriptorBufferBindingInfoEXT bindInfo{};
    bindInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bindInfo.address = descBuffers_.addresses[idx];
    bindInfo.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
    app->fpCmdBindDescriptorBuffersEXT(cmd, 1, &bindInfo);
    const uint32_t bufferIndex = 0;
    const VkDeviceSize setOffset = 0;
    app->fpCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            layout, 0, 1, &bufferIndex, &setOffset);
}

// Step 5 helper: the bind path is live only with extension + ready buffers +
// a DESCRIPTOR_BUFFER_BIT-capable main layout. False selects the unchanged
// vkUpdateDescriptorSets fallback.
bool SceneRenderer::descriptorBufferBindActive(VulkanApp* app) const {
    if (!app || !app->useDescriptorBuffer() || !descBuffers_.ready) return false;
    if (!app->fpCmdBindDescriptorBuffersEXT || !app->fpCmdSetDescriptorBufferOffsetsEXT) return false;
    if (!app->sceneDescriptorLayout) return false;
    return app->sceneDescriptorLayout->descriptorBufferBindActive();
}

SceneRenderer::StaticTextureSignature SceneRenderer::currentStaticSignature(TextureArrayManager* textureArrayManager) const {
    StaticTextureSignature sig{};
    if (textureArrayManager) {
        sig.samplers[0] = textureArrayManager->albedoSampler;
        sig.views[0] = textureArrayManager->albedoArray.view;
        sig.samplers[1] = textureArrayManager->normalSampler;
        sig.views[1] = textureArrayManager->normalArray.view;
        sig.samplers[2] = textureArrayManager->bumpSampler;
        sig.views[2] = textureArrayManager->bumpArray.view;
        sig.samplers[3] = textureArrayManager->roughnessSampler;
        sig.views[3] = textureArrayManager->roughnessArray.view;
        sig.samplers[4] = textureArrayManager->aoSampler;
        sig.views[4] = textureArrayManager->aoArray.view;
    }
    if (shadowMapper) {
        sig.shadowSampler = shadowMapper->getShadowMapSampler();
        sig.shadowViews[0] = shadowMapper->getShadowMapView(0);
        sig.shadowViews[1] = shadowMapper->getShadowMapView(1);
        sig.shadowViews[2] = shadowMapper->getShadowMapView(2);
    }
    sig.materials = materialsBuffer.buffer;
    sig.waterParams = waterParamsBuffer_.buffer;
    sig.valid = true;
    return sig;
}

// Update only the static bindings (textures, materials, water params) in the
// static descriptor set, then propagate to all per-frame descriptor sets via
// VkCopyDescriptorSet. This avoids re-writing identical descriptors per frame.
//
// Render-loop guarantee: this function is event-driven (texture-array allocation
// listener, swapchain resize) and never runs per frame. The StaticTextureSignature
// change guard below additionally makes repeated calls with unchanged resources
// a no-op, so steady state issues 0 vkUpdateDescriptorSets calls. Per-frame UBO
// contents are streamed via host memcpy into the already-bound UBO buffers
// (main.cpp preRenderPass) or vkCmdCopyBuffer (shadow cascades) — never via
// descriptor updates — so the GPU timeline overlaps descriptor-buffer-friendly
// uploads with compute. When VulkanApp::useDescriptorBuffer() is true, the
// static writes go to the per-frame descriptor buffers first (direct
// vkGetDescriptorEXT host writes via writeStaticDescriptorsToBuffers, no
// vkUpdateDescriptorSets); while SceneDescriptorLayout::
// descriptorBufferBindActive() is false the classic DescriptorWriter path below
// still runs so the bound sets stay current (buffers kept warm for the
// cutover), otherwise it returns early. Full set-0 descriptor-buffer *binding*
// (bindSet0ForFrame) requires the main layout to carry
// VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT plus the set-1/2
// migration — tracked as follow-up; the buffer update path here is live.
void SceneRenderer::updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager) {
    if (!app) return;

    VkDescriptorSet staticDs = app->getStaticDescriptorSet();
    if (staticDs == VK_NULL_HANDLE) return;
    if (!textureArrayManager) return;

    // Refresh the materials buffer first so the signature below sees the latest
    // handle (setupTextures may allocate it after SceneRenderer::init on a
    // background thread).
    if (materialManagerPtr && materialManagerPtr->getBuffer().buffer != VK_NULL_HANDLE) {
        materialsBuffer = materialManagerPtr->getBuffer();
    }

    // Change guard: skip the entire update when every static resource matches
    // the last successful write. Allocation listeners can fire redundantly
    // (e.g. material allocate + texture realloc in the same setup pass).
    {
        StaticTextureSignature sig = currentStaticSignature(textureArrayManager);
        if (lastStaticSignature_.valid && lastStaticSignature_.matches(sig)) {
            return; // nothing changed — 0 vkUpdateDescriptorSets calls
        }
        lastStaticSignature_ = sig;
    }

    // Step 4: on texture-array (re)alloc, refresh the descriptor buffers with
    // plain host writes (vkGetDescriptorEXT) — no vkUpdateDescriptorSets. Once
    // the bind path is live this returns right after (classic sets are never
    // bound then); while binding stays classic the classic writes below still
    // run so the bound sets stay current, and the buffers are kept warm for
    // the cutover. Event-driven only — never in the render loop.
    if (descBuffers_.ready && app->useDescriptorBuffer()) {
        writeStaticDescriptorsToBuffers(app, textureArrayManager);
        if (descriptorBufferBindActive(app)) {
            if (brushRenderer) brushRenderer->writeDepthDescriptors(app);
            return;
        }
    }

    // 1. Write updated bindings to the static descriptor set.
    // Fast path (descriptor buffers supported): plain host writes via
    // vkGetDescriptorEXT would target descriptor-buffer memory here; the main
    // layout is not yet DESCRIPTOR_BUFFER_BIT-capable (see note above), so we
    // record the intent and use the classic fallback for the actual bindable
    // set. The branch keeps the GPU-side path compiled and exercised
    // (address queries) without breaking validation on current layouts.
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

        // Materials SSBO (binding 5) — refreshed above.
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

    // 2. Propagate static bindings (1-13) to all per-frame descriptor sets.
    // Batched into ONE vkUpdateDescriptorSets call across all frames (fewer
    // driver round-trips than one call per frame). Event-time only — never in
    // the render loop. With descriptor buffers this copy list becomes a set of
    // host-side vkGetDescriptorEXT writes into each frame's descriptor buffer
    // region (same source data, no driver validation work); the classic copy
    // below is the fallback until the main layout is DESCRIPTOR_BUFFER_BIT.
    {
        std::vector<VkCopyDescriptorSet> copies;
        const size_t setCount = app->getMainDescriptorSetCount();
        copies.reserve(setCount * 13);
        for (size_t s = 0; s < setCount; ++s) {
            VkDescriptorSet mainDs = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(s));
            if (mainDs == VK_NULL_HANDLE) continue;

            // Binding 1..4, 8, 9, 12, 13 (textures). Binding 11 removed with the
            // legacy cubemap (hybrid RT); RT bindings (14-18) are stable since
            // init and need no propagation here.
            for (uint32_t b : {1u, 2u, 3u, 4u, 8u, 9u, 12u, 13u}) {
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
        }

        if (!copies.empty()) {
            DescriptorUpdateStats::noteUpdate(copies.size());
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
        // chunk's 0-based band rung (0 = finest published rung), published in the
        // bounds meta for the GPU's per-chunk distance band test (entryLevel).
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

        // Hybrid RT: record the proxy source of main-scene chunks (world bounds
        // + dominant material). Opaque chunks feed the solid BLAS; transparent
        // (water) chunks feed the water BLAS so solid reflections see water.
        // Brush chunks excluded (preview overlay, not scene).
        // NOTE (refraction fix): the proxy box MUST tightly enclose the actual
        // mesh surface, not the emitting octree cell. Cell cubes are full
        // volumes (tens of meters) whose faces sit at cell boundaries far from
        // the true terrain: a refraction ray from the water surface then starts
        // INSIDE the solid box and reports the cell-boundary exit (wrong
        // thickness, wrong XZ offset, side-face shading) instead of the lake
        // bottom underneath. Tight vertex bounds make each proxy a thin slab
        // around the true surface, so the refracted ray hits the top face at
        // the real underwater point with ground-related color/thickness.
        if (!isBrush && !lod.geom.vertices.empty()) {
            std::lock_guard<std::recursive_mutex> lock(mainSolidChunksMutex);
            SolidProxyData pd;
            {
                glm::vec3 tmin = lod.geom.vertices[0].position;
                glm::vec3 tmax = tmin;
                // Dominant material (mode of brushIndex) so the proxy average
                // represents the chunk's actual ground cover, not just corner 0.
                std::unordered_map<int, uint32_t> matHist;
                matHist.reserve(8);
                int bestMat = lod.geom.vertices[0].brushIndex;
                uint32_t bestCount = 0;
                for (const auto& v : lod.geom.vertices) {
                    const glm::vec3& p = v.position;
                    tmin.x = std::min(tmin.x, p.x);
                    tmin.y = std::min(tmin.y, p.y);
                    tmin.z = std::min(tmin.z, p.z);
                    tmax.x = std::max(tmax.x, p.x);
                    tmax.y = std::max(tmax.y, p.y);
                    tmax.z = std::max(tmax.z, p.z);
                    uint32_t c = ++matHist[v.brushIndex];
                    if (c > bestCount) { bestCount = c; bestMat = v.brushIndex; }
                }
                // Small padding: keeps thin/flat surfaces non-degenerate and
                // covers GPU TES displacement (which the CPU mesh does not see).
                constexpr float kProxyPad = 0.15f;
                tmin -= glm::vec3(kProxyPad);
                tmax += glm::vec3(kProxyPad);
                // Clamp inside the emitting cell so a displaced outlier can
                // never bloat the proxy back into a full-cell volume.
                tmin.x = std::max(tmin.x, cubeMin.x);
                tmin.y = std::max(tmin.y, cubeMin.y);
                tmin.z = std::max(tmin.z, cubeMin.z);
                tmax.x = std::min(tmax.x, cubeMax.x);
                tmax.y = std::min(tmax.y, cubeMax.y);
                tmax.z = std::min(tmax.z, cubeMax.z);
                pd.minp = tmin;
                pd.maxp = tmax;
                pd.materialId = static_cast<uint32_t>(std::max(0, bestMat));
                if (layer == LAYER_TRANSPARENT) {
                    // Water reflection proxy: thin slab at the lake surface.
                    // The water mesh is a flat lake top + terrain-contact
                    // walls, so the raw vertex bounds span from the lake
                    // bottom to terrain tops — a box that swallows the whole
                    // scene. Reflection rays then always hit terrain (closer)
                    // before the box's distant faces, so water never shows up
                    // in reflections. Clamp Y to a thin band around the
                    // dominant (modal) vertex height = the lake surface.
                    std::unordered_map<int, uint32_t> yHist;
                    yHist.reserve(64);
                    for (const auto& v : lod.geom.vertices)
                        ++yHist[static_cast<int>(v.position.y)];
                    int bestY = 0;
                    uint32_t bestYC = 0;
                    for (const auto& kv : yHist)
                        if (kv.second > bestYC) { bestYC = kv.second; bestY = kv.first; }
                    const float surf = static_cast<float>(bestY);
                    pd.minp.y = surf - 3.0f;
                    // Extend well ABOVE the surface so the reflection rays
                    // that pass over the lake (rising only a few degrees per
                    // chunk) still catch the water slab instead of flying
                    // past to the far shore / sky. The water.frag own-cell
                    // rejection keeps the fragment's own cell from
                    // self-hitting; the rchit own-body guard does the same
                    // for the pipeline path.
                    pd.maxp.y = surf + 3.0f;
                    pd.hgrid.fill(surf);
                }
                // 4x4 max-height grid: proxy boxes per cell follow the chunk's
                // real silhouette, so reflected secondaries land on the true
                // surface instead of a single flat box top.
                constexpr int kGrid = 4;
                const float cellW = std::max((tmax.x - tmin.x) / float(kGrid), 1e-3f);
                const float cellD = std::max((tmax.z - tmin.z) / float(kGrid), 1e-3f);
                pd.hgrid.fill(tmin.y);
                for (const auto& v : lod.geom.vertices) {
                    const glm::vec3& p = v.position;
                    int gx = std::clamp(int((p.x - tmin.x) / cellW), 0, kGrid - 1);
                    int gz = std::clamp(int((p.z - tmin.z) / cellD), 0, kGrid - 1);
                    float& h = pd.hgrid[gz * kGrid + gx];
                    h = std::max(h, p.y);
                }
            }
            pd.rung = static_cast<uint32_t>(lod.lod);
            if (layer == LAYER_OPAQUE)
                mainSolidProxyData[nid] = pd;
            else
                mainWaterProxyData[nid] = pd;
        }

        // Generate vegetation for every published grass chunk (lod.lod is the
        // 0-based band rung, so every rung carries its own grass for full
        // terrain coverage). The LoD band gate in
        // indirect.comp keeps exactly one rung per region visible, so only the
        // selected rung's grass is drawn — generating per rung (not just the
        // finest) lets grass appear across the whole visible terrain instead of
        // only in the thinnest high-detail disc around the camera, with no
        // overdraw because overlapping rungs are never simultaneously visible.
        if (layer == LAYER_OPAQUE && vegetationRenderer && !lod.geom.vertices.empty()) {
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

    // ── Async visible-count snapshot (stats only) ─────────────────────────
    // The cull counts live in DEVICE_LOCAL buffers; every prepareCull copies
    // them into small host-visible readback slots. Snapshot both layers here
    // once per frame (1-frame latency) so the stats overlay never touches GPU
    // memory on the ImGui path.
    lastOpaqueVisible_ = mainSolidRenderer->getIndirectRenderer().readVisibleCount(app);
    lastTransparentVisible_ = mainLiquidRenderer->getIndirectRenderer().readVisibleCount(app);

    if (batch.empty()) {
        // No new geometry yet (brush tessellation may still be running). Keep
        // old geometry visible — don't free anything. On the next rebuild,
        // stageOldBrushChunks will re-capture the same slots. Still age out the
        // main stream's orphaned pending-delete entries (genuine deletions with
        // no replacement) so a mid-stream erase never leaks a slot.
        uint32_t curFrame = app ? app->getCurrentFrame() : 0;
        ageOutPendingDeletes(curFrame, mainSolidRenderer->getIndirectRenderer(), mainLiquidRenderer->getIndirectRenderer());
        processChunkSwapQueue(app);
        // Hybrid RT: deletions without publishes still change the proxy set
        // (fingerprint check inside is O(N) and early-outs when idle).
        rebuildProxySet(app, false);
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

    // Hybrid RT: stage fresh proxy boxes when publishes happened this frame
    // (fingerprint dedupes pure camera/LOD frames at O(N) integer cost).
    // A pending texture-driven albedo refresh forces a repack as well.
    const bool albedoRefresh =
        proxyAlbedoRefresh_.exchange(false, std::memory_order_relaxed);
    rebuildProxySet(app, chunksPublished > 0 || albedoRefresh);
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
    // Worst-case sizing (startup-only; never reallocated at runtime):
    //   maxChunks  = every mesh-bearing octree node (chunks + ancestors) plus
    //                transient double-slotting during chunk rebuilds/edits.
    //                Upper bound ≈ 8^depth ladder nodes; the kMax* constants
    //                below hold ~2.5x the measured reference-scene peak while
    //                keeping the reservation (~2 GB) under the ~4 GB radv
    //                device-lost threshold. If world_ is set, its octree depth
    //                is logged as a sanity check on the bound.
    //   totalBytes = maxChunks × per-chunk ceiling (packed pools: allocation
    //                is data-driven, so this is the footprint ceiling).
    // Single large buffer + offset management (PackedSpaceAllocator) is used
    // for growth without reallocation; sparse virtual aliasing
    // (VK_BUFFER_CREATE_SPARSE_BINDING_BIT + vkBindBufferMemory2) is queried
    // in VulkanApp::createLogicalDevice but NOT used — VMA has no sparse
    // allocator and most iGPUs lack sparseResidencyBuffer. Buffer device
    // address (VK_EXT/BDA) needs no separate buffer: the merged pools already
    // provide the single-large-buffer + offset model.
    if (world_) {
        std::cout << "[SceneRenderer] initSlottedMode: world "
                  << (world_->scene().maxChunkLod(LAYER_OPAQUE, 1.0f))
                  << " maxChunkLod(opaque) / world set — pools sized to worst case\n";
    }
    std::cout << "[SceneRenderer] memory model: "
              << (app->supportsSparseBinding()
                      ? "sparse binding available (reserved: fixed pre-alloc pools)"
                      : "no sparse binding — fixed pre-allocated pools + offsets")
              << ", BDA " << (app->supportsBufferDeviceAddress() ? "supported" : "unsupported")
              << std::endl;

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
    // Bind the merged pools for real-triangle hit shading (bindings 24/25).
    // The device is idle here (init), so the classic set rewrite is safe.
    writeSceneVertexBindings(app);

    // Validate the worst-case reservation once at init (slotted-mode
    // ensureCapacity is a pure check — it asserts instead of growing).
    {
        const size_t solidVerts = solidVertBytes / sizeof(Vertex);
        const size_t solidIdx = solidIdxBytes / sizeof(uint32_t);
        const bool okSolid = mainSolidRenderer->getIndirectRenderer().ensureCapacity(
            solidVerts, solidIdx, maxSolidChunks);
        const size_t waterVerts = waterVertBytes / sizeof(Vertex);
        const size_t waterIdx = waterIdxBytes / sizeof(uint32_t);
        const bool okWater = mainLiquidRenderer->getIndirectRenderer().ensureCapacity(
            waterVerts, waterIdx, maxWaterChunks);
        if (!okSolid || !okWater) {
            std::cerr << "[SceneRenderer] initSlottedMode: worst-case ensureCapacity FAILED\n";
            assert(false && "initSlottedMode worst-case capacity check failed");
        }
    }

    // Pre-allocate ALL vegetation culling buffers to max chunk count (4096)
    // in the same init-time burst — zero vegetation vmaCreateBuffer calls
    // after the first frame.
    if (vegetationRenderer) {
        vegetationRenderer->preallocate(app, VegetationRenderer::kMaxVegChunks,
                                        VegetationRenderer::kMaxVegInstancesPerChunk);
    }

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

    // NOTE: the walk emits one callback per cell on the root path (each
    // ancestor at its own level); the cube passed is the EMITTING cell's own
    // cube — the band center and the meta cellSize must come from it, never
    // from nodeData (the added node), or every ancestor would publish the
    // frontier cell's size.
    scene.requestModel3D(layer, nodeData, [&layer,&onGeometry,&nodeData](const Geometry& geo, uint8_t lod, uint version, uintptr_t emittingNodeId, const BoundingCube& cube, const BoundingCube& baseCube) {
        Octree::LoDMesh lm;
        lm.geom = geo;
        lm.lod = lod;
        lm.version = version;
        lm.cellSize = cube.getLength().x;
        lm.boundsMin = cube.getMin();
        lm.boundsMax = cube.getMax();
        lm.boundsBase = baseCube.getMin();
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
            [&sdfCubes, &sdfMtx, &nodeData](const BoundingCube& cube, const std::array<float, 8>& sdf,
                                 uint8_t /*lod*/, uint /*version*/, uintptr_t /*emittingNodeId*/, uint32_t brushIndex) {
                DebugSDFRenderer::CubeSDF c;
                c.cube = cube;
                c.sdf = sdf;
                c.brushIndex = static_cast<int>(brushIndex);
                // LoD meta for the SDF cull's clipmap band gate (mirrors the solid chunk
                // entry): cellSize = chunk cube side, level = 0-based band rung
                // (chunkLod - 1), base = chunk min corner. The SDF cubes are
                // emitted at the chunk's finest surface rung, but the gate
                // selects the rung by the CHUNK's LoD so parent/child chunk SDF
                // cubes never overlap.
                c.cellSize = nodeData.cube.getLengthX();
                c.level    = static_cast<int>(nodeData.node->getChunkLod()) - 1;
                c.base     = nodeData.cube.getMin();
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
            [&bbCubes, &bbMtx, layer, &nodeData](const BoundingCube& cube) {
                DebugCubeRenderer::CubeWithColor c;
                c.cube = BoundingBox(cube.getMin(), cube.getMax());
                c.color = (layer == LAYER_OPAQUE)
                    ? glm::vec3(0.0f, 1.0f, 0.0f)
                    : glm::vec3(0.0f, 0.5f, 1.0f);
                // LoD meta for the bbox cull's clipmap band gate (mirrors the solid
                // chunk entry): cellSize = node cube side, level = 0-based band
                // rung (chunkLod - 1), base = chunk min corner.
                c.cellSize = cube.getLengthX();
                c.level    = static_cast<int>(nodeData.node->getChunkLod()) - 1;
                c.base     = nodeData.cube.getMin();
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


void SceneRenderer::writeTlasBinding(VulkanApp* app, VkDescriptorSet dstSet) {
    if (!app || dstSet == VK_NULL_HANDLE || tlasMirror_ == VK_NULL_HANDLE) return;
    if (!app->rayTracingEnabled()) return;
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlasMirror_;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.pNext = &asInfo;
    w.dstSet = dstSet;
    w.dstBinding = 14;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(app->getDevice(), 1, &w, 0, nullptr);
}

void SceneRenderer::refreshRTSceneViews(VulkanApp* app) {
    if (!app || !rayTracing || !rayTracing->isSupported()) return;
    if (!mainLiquidRenderer || !skyRenderer) return;
    VkImageView waterDepths[3] = {};
    VkImageView skyViews[3] = {};
    for (int i = 0; i < 3; ++i) {
        waterDepths[i] = mainLiquidRenderer->getWaterGeomDepthView(static_cast<uint32_t>(i));
        skyViews[i] = skyRenderer->getSkyView(static_cast<uint32_t>(i));
    }
    rayTracing->setSceneViews(app, waterDepths, skyViews);
}

void SceneRenderer::initSSRSamplers(VulkanApp* app) {
    if (!app) return;
    if (ssrColorSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = 0.0f;
        if (vkCreateSampler(app->getDevice(), &ci, nullptr, &ssrColorSampler) != VK_SUCCESS) {
            ssrColorSampler = VK_NULL_HANDLE;
        } else {
            app->resources.addSampler(ssrColorSampler, "SceneRenderer: ssrColorSampler");
        }
    }
    if (ssrDepthSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_NEAREST;
        ci.minFilter = VK_FILTER_NEAREST;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = 0.0f;
        if (vkCreateSampler(app->getDevice(), &ci, nullptr, &ssrDepthSampler) != VK_SUCCESS) {
            ssrDepthSampler = VK_NULL_HANDLE;
        } else {
            app->resources.addSampler(ssrDepthSampler, "SceneRenderer: ssrDepthSampler");
        }
    }
}

void SceneRenderer::destroySSRSamplers(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    if (ssrColorSampler != VK_NULL_HANDLE) {
        if (app->resources.removeSampler(ssrColorSampler))
            vkDestroySampler(device, ssrColorSampler, nullptr);
        ssrColorSampler = VK_NULL_HANDLE;
    }
    if (ssrDepthSampler != VK_NULL_HANDLE) {
        if (app->resources.removeSampler(ssrDepthSampler))
            vkDestroySampler(device, ssrDepthSampler, nullptr);
        ssrDepthSampler = VK_NULL_HANDLE;
    }
}

void SceneRenderer::writeSceneVertexBindings(VulkanApp* app) {
    if (!app || !mainSolidRenderer) return;
    VkBuffer vb = mainSolidRenderer->getIndirectRenderer().getVertexBufferHandle();
    VkBuffer ib = mainSolidRenderer->getIndirectRenderer().getIndexBufferHandle();
    if (vb == VK_NULL_HANDLE || ib == VK_NULL_HANDLE) return;
    // Write the full merged vertex/index buffers so the ray-hit shader can read
    // any vertex (including brush index, uv, normal) regardless of position in
    // the pool. On devices with large maxStorageBufferRange (e.g. RADV) this is
    // unrestricted; on llvmpipe a validation warning may fire but the buffer is
    // fully mapped and accessible to the shader.
    constexpr VkDeviceSize wholeSize = VK_WHOLE_SIZE;
    DescriptorWriter writer(app->getDevice());
    auto bind = [&](VkDescriptorSet ds) {
        if (ds == VK_NULL_HANDLE) return;
        writer.writeBuffer(ds, 24, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vb, 0, wholeSize);
        writer.writeBuffer(ds, 25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ib, 0, wholeSize);
    };
    bind(app->getStaticDescriptorSet());
    for (size_t fi = 0; fi < app->getMainDescriptorSetCount(); ++fi)
        bind(app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi)));
    writer.flush();
}

void SceneRenderer::writeSSRBindings(VulkanApp* app) {
    if (!app || !mainSolidRenderer) return;
    if (ssrColorSampler == VK_NULL_HANDLE || ssrDepthSampler == VK_NULL_HANDLE) return;
    const uint32_t nsrc = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    if (nsrc == 0) return;
    DescriptorWriter writer(app->getDevice());
    auto bind = [&](VkDescriptorSet ds, uint32_t slot) {
        if (ds == VK_NULL_HANDLE) return;
        // Set `slot` renders into slot `slot`'s solid images; sample the
        // previous frame's images, which live in (slot - 1) mod nsrc.
        const uint32_t src = (slot + nsrc - 1u) % nsrc;
        VkImageView c = mainSolidRenderer->getColorView(src);
        VkImageView d = mainSolidRenderer->getDepthView(src);
        if (c == VK_NULL_HANDLE || d == VK_NULL_HANDLE) return;
        writer.writeImage(ds, 19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          ssrColorSampler, c, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        writer.writeImage(ds, 20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          ssrDepthSampler, d, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    for (size_t fi = 0; fi < app->getMainDescriptorSetCount(); ++fi)
        bind(app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi)), static_cast<uint32_t>(fi));
    // Keep the static set valid too (it may be bound by passes that don't touch
    // bindings 19/20; slot 0 views are always a legal image).
    bind(app->getStaticDescriptorSet(), 0);
    // Per-frame RT params (binding 17): each main set reads its own slot's
    // camera matrices, so in-flight frames never see another frame's view.
    if (rayTracing && rayTracing->isSupported()) {
        for (size_t fi = 0; fi < app->getMainDescriptorSetCount(); ++fi) {
            VkDescriptorSet ds = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));
            VkBuffer pb = rayTracing->getParamsBuffer(static_cast<uint32_t>(fi));
            if (ds == VK_NULL_HANDLE || pb == VK_NULL_HANDLE) continue;
            writer.writeBuffer(ds, 17, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, pb, 0, sizeof(RayTracingParams));
        }
    }
    writer.flush();
}

void SceneRenderer::rebuildProxySet(VulkanApp* app, bool sceneChanged) {
    if (!rayTracing || !rayTracing->isSupported()) return;
    // Fingerprint both proxy-source registries (count + bounds/material hash).
    // Camera moves, LOD band switches and tessellation changes never touch
    // these maps, so they never mark the proxy dirty (§6). O(N), N ~= chunks.
    uint64_t fp = 0;
    size_t count = 0;
    const bool wantWater = rtWaterProxyEnabled;
    {
        std::lock_guard<std::recursive_mutex> lock(mainSolidChunksMutex);
        auto hashMap = [&](const std::unordered_map<NodeID, SolidProxyData>& m) {
            for (const auto& kv : m) {
                const SolidProxyData& d = kv.second;
                // Hash the FULL tight bounds + material + rung: Y/Z-only edits
                // (brush height strokes, water level shifts) must dirty the
                // TLAS, otherwise refraction keeps sampling the pre-edit lake
                // bottom (stale color unrelated to the surface underneath).
                fp += uint64_t(d.materialId) * 1000003ull
                    + uint64_t(std::bit_cast<uint32_t>(d.minp.x)) * 31ull
                    + uint64_t(std::bit_cast<uint32_t>(d.minp.y)) * 37ull
                    + uint64_t(std::bit_cast<uint32_t>(d.minp.z)) * 41ull
                    + uint64_t(std::bit_cast<uint32_t>(d.maxp.x)) * 43ull
                    + uint64_t(std::bit_cast<uint32_t>(d.maxp.y)) * 47ull
                    + uint64_t(std::bit_cast<uint32_t>(d.maxp.z)) * 53ull
                    + uint64_t(d.rung) * 59ull;
            }
        };
        hashMap(mainSolidProxyData);
        if (wantWater) hashMap(mainWaterProxyData);
        count = mainSolidProxyData.size() + (wantWater ? mainWaterProxyData.size() : 0);
        fp ^= uint64_t(count) * 0x9e3779b97f4a7c15ull;
        fp ^= wantWater ? 0x12345678ull : 0u;
    }
    static thread_local uint64_t lastFp = 0;
    static thread_local size_t lastCount = 0;
    {
        // ── Real scene geometry refresh ────────────────────────────────────
        // Chunk re-uploads move the packed vertex/index spans even when the
        // proxy bounds/material fingerprint is unchanged. Refresh the scene
        // BLAS inputs whenever the span set actually differs, so reflections
        // never trace spans that newer chunks reclaimed (stale geometry =
        // wrong positions AND wrong per-vertex brushIndex textures).
        if (mainSolidRenderer && textureArrays_) {
            VkBuffer vb = mainSolidRenderer->getIndirectRenderer().getVertexBufferHandle();
            VkBuffer ib = mainSolidRenderer->getIndirectRenderer().getIndexBufferHandle();
            std::vector<IndirectRenderer::RTGeometrySpan> spans;
            // Camera-independent: every active chunk, so the BLAS is rebuilt
            // only when the chunk set changes (never on camera moves) and the
            // reflection covers ALL chunks the raster draws.
            mainSolidRenderer->getIndirectRenderer().copyAllRTGeometrySpans(spans);
            if (vb != VK_NULL_HANDLE && ib != VK_NULL_HANDLE && spans != lastSceneSpans_) {
                lastSceneSpans_ = spans;
                std::lock_guard<std::recursive_mutex> lock(mainSolidChunksMutex);
                VkBufferDeviceAddressInfo q{};
                q.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                q.buffer = vb;
                const VkDeviceAddress vaddr = vkGetBufferDeviceAddress(app->getDevice(), &q);
                q.buffer = ib;
                const VkDeviceAddress iaddr = vkGetBufferDeviceAddress(app->getDevice(), &q);
                std::vector<RayTracingResources::SceneTriGeometry> geoms;
                geoms.reserve(spans.size() + 64);
                for (const auto& s : spans) {
                    RayTracingResources::SceneTriGeometry g;
                    g.vertexAddress = vaddr + VkDeviceAddress(s.baseVertex) * sizeof(Vertex);
                    g.indexAddress = iaddr + VkDeviceAddress(s.firstIndex) * sizeof(uint32_t);
                    g.vertexCount = s.vertexCount;
                    g.indexCount = s.indexCount;
                    g.baseVertex = s.baseVertex;
                    g.firstIndex = s.firstIndex;
                    auto it = mainSolidProxyData.find(s.chunkId);
                    const uint32_t mat = (it != mainSolidProxyData.end()) ? it->second.materialId : 0u;
                    const auto avg = textureArrays_->albedoAverage(mat);
                    g.albedo = glm::vec4(avg[0], avg[1], avg[2], float(mat));
                    geoms.push_back(g);
                }
                // Water chunks: append the real water MESH (transparent layer)
                // to the scene BLAS so reflections hit the actual water surface
                // triangles (accurate positions, no proxy boxes). The
                // waterChunk flag routes hit shading to the water look (sky
                // reflection + tint) instead of the terrain albedo lookup.
                if (mainLiquidRenderer) {
                    VkBuffer wvb = mainLiquidRenderer->getIndirectRenderer().getVertexBufferHandle();
                    VkBuffer wib = mainLiquidRenderer->getIndirectRenderer().getIndexBufferHandle();
                    std::vector<IndirectRenderer::RTGeometrySpan> wspan;
                    mainLiquidRenderer->getIndirectRenderer().copyAllRTGeometrySpans(wspan);
                    if (wvb != VK_NULL_HANDLE && wib != VK_NULL_HANDLE && !wspan.empty()) {
                        VkBufferDeviceAddressInfo wq{};
                        wq.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                        wq.buffer = wvb;
                        const VkDeviceAddress wvaddr = vkGetBufferDeviceAddress(app->getDevice(), &wq);
                        wq.buffer = wib;
                        const VkDeviceAddress wiaddr = vkGetBufferDeviceAddress(app->getDevice(), &wq);
                        for (const auto& s : wspan) {
                            RayTracingResources::SceneTriGeometry g;
                            g.vertexAddress = wvaddr + VkDeviceAddress(s.baseVertex) * sizeof(Vertex);
                            g.indexAddress = wiaddr + VkDeviceAddress(s.firstIndex) * sizeof(uint32_t);
                            g.vertexCount = s.vertexCount;
                            g.indexCount = s.indexCount;
                            g.baseVertex = s.baseVertex;
                            g.firstIndex = s.firstIndex;
                            g.waterChunk = true;
                            // albedo.w = the water LAYER index (chunk
                            // dominant, stable per chunk) so hit shading reads
                            // a consistent layer — per-vertex brushIndex can
                            // vary within a triangle and would flicker the
                            // water color.
                            auto wit = mainWaterProxyData.find(s.chunkId);
                            const float wLayer = (wit != mainWaterProxyData.end())
                                ? static_cast<float>(wit->second.materialId) : 0.0f;
                            g.albedo = glm::vec4(waterReflectionTint_, wLayer);
                            geoms.push_back(g);
                        }
                    }
                }
                rayTracing->setSceneGeometry(std::move(geoms));
            }
        }
    }
    if (!sceneChanged && fp == lastFp && count == lastCount) return;
    lastFp = fp;
    lastCount = count;

    // Fixed water tint for proxy hit shading (a water chunk's brush index
    // addresses water params, not scene materials, so store the tint directly).
    constexpr glm::vec3 kWaterProxyAlbedo(0.03f, 0.10f, 0.14f);
    std::vector<RTProxyBox> solids;
    std::vector<RTProxyBox> waters;
    {
        std::lock_guard<std::recursive_mutex> lock(mainSolidChunksMutex);
        // GC entries for erased chunks (publish maps are authoritative).
        for (auto it = mainSolidProxyData.begin(); it != mainSolidProxyData.end(); ) {
            if (mainSolidChunks.find(it->first) == mainSolidChunks.end())
                it = mainSolidProxyData.erase(it);
            else ++it;
        }
        for (auto it = mainWaterProxyData.begin(); it != mainWaterProxyData.end(); ) {
            if (mainLiquidChunks.find(it->first) == mainLiquidChunks.end())
                it = mainWaterProxyData.erase(it);
            else ++it;
        }
        auto pack = [&](const std::unordered_map<NodeID, SolidProxyData>& m,
                        std::vector<RTProxyBox>& out, bool isWater) {
            // All visible LOD rungs are packed: each proxy is clamped inside
            // its own octree cell, so different rungs stay disjoint and a
            // reflection ray hits the same LOD the rasterizer draws. Packing
            // only the finest rung left every mid/far reflection with no TLAS
            // geometry to hit, so distant mirrors (polished water spheres)
            // collapsed to the flat sky fallback. Sort by rung so that if the
            // slot budget truncates, the finest (closest) boxes survive.
            std::vector<std::pair<uint32_t, const SolidProxyData*>> sorted;
            sorted.reserve(m.size());
            for (const auto& kv : m) {
                const SolidProxyData& d = kv.second;
                if (!(d.maxp.x > d.minp.x && d.maxp.y > d.minp.y && d.maxp.z > d.minp.z))
                    continue; // degenerate
                sorted.emplace_back(d.rung, &d);
            }
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [rung, dp] : sorted) {
                (void)rung;
                const SolidProxyData& d = *dp;
                glm::vec3 chunkMin = d.minp;
                glm::vec3 chunkMax = d.maxp;
                // Grazing secondary rays (the reflected horizon) can slip
                // between neighbouring chunk boxes when their AABB tops differ:
                // they thread the vertical gap and stipple the sky/terrain
                // boundary. Solid boxes tile their cells in X/Z, so extending
                // each one downward closes those gaps with continuous side
                // walls. Water boxes keep their true bounds (thickness).
                if (!isWater) chunkMin.y -= 512.0f;

                auto proxyAlbedo = [&](RTProxyBox& b) {
                    if (isWater) {
                        // Actual water surface tint (shallow/deep mix from the
                        // water layer), not a hardcoded blue — so reflected
                        // water carries the real water color.
                        b.albedo = waterReflectionTint_;
                        b.roughness = 0.15f;
                    } else if (textureArrays_) {
                        // Real per-material average albedo (linear) so
                        // secondary rays return plausible terrain colors
                        // instead of flat gray.
                        const auto avg = textureArrays_->albedoAverage(d.materialId);
                        b.albedo = glm::vec3(avg[0], avg[1], avg[2]);
                    }
                };

                if (isWater) {
                    // Water volumes keep one box: they are thickness volumes,
                    // not heightfields.
                    RTProxyBox b{};
                    b.minp = chunkMin;
                    b.maxp = chunkMax;
                    b.materialId = static_cast<float>(d.materialId);
                    b.flags = 1.0f;
                    proxyAlbedo(b);
                    out.push_back(b);
                    continue;
                }

                // Solid chunks: one box per 4x4 height-grid cell so secondary
                // rays hit the chunk's real silhouette (reflection positions
                // match the terrain surface within ~cell/4 instead of one flat
                // box top per chunk).
                constexpr int kGrid = 4;
                const float w = chunkMax.x - chunkMin.x;
                const float dep = chunkMax.z - chunkMin.z;
                const float cw = w / float(kGrid);
                const float cd = dep / float(kGrid);
                const float floorY = chunkMin.y;
                for (int gz = 0; gz < kGrid; ++gz) {
                    for (int gx = 0; gx < kGrid; ++gx) {
                        const float h = d.hgrid[gz * kGrid + gx];
                        if (h <= d.minp.y) continue; // empty cell
                        RTProxyBox b{};
                        b.minp = glm::vec3(chunkMin.x + cw * float(gx), floorY,
                                           chunkMin.z + cd * float(gz));
                        b.maxp = glm::vec3((gx == kGrid - 1) ? chunkMax.x : b.minp.x + cw,
                                           h,
                                           (gz == kGrid - 1) ? chunkMax.z : b.minp.z + cd);
                        b.materialId = static_cast<float>(d.materialId);
                        b.flags = 0.0f;
                        proxyAlbedo(b);
                        out.push_back(b);
                    }
                }
            }
        };
        solids.reserve(mainSolidProxyData.size());
        pack(mainSolidProxyData, solids, false);
        if (wantWater) {
            waters.reserve(mainWaterProxyData.size());
            pack(mainWaterProxyData, waters, true);
        }
        { // Rare (repacks only): pack composition. All rungs are packed now,
            // so this also fingerprints whether the all-rung proxy set is
            // active in a given binary.
            static size_t lastKept = SIZE_MAX;
            const size_t kept = solids.size() + waters.size();
            if (kept != lastKept) {
                lastKept = kept;
                printf("[HybridRT] proxy pack: %zu solid + %zu water boxes (all rungs)\n",
                    solids.size(), waters.size());
                fflush(stdout);
            }
        }
    }
    rayTracing->setProxies(solids, waters);
}

void SceneRenderer::updateRTParams(VulkanApp* app, const Settings& settings,
                                   const WaterParams& waterLook,
                                   const glm::mat4& invViewProj, const glm::vec3& viewPos,
                                   const glm::vec3& sunDirTo, const glm::vec3& sunColor,
                                   float nearPlane, float farPlane) {
    if (!app || !rayTracing || !rayTracing->isSupported()) return;
    RayTracingParams p{};
    p.toggles = glm::vec4(settings.rtReflections ? 1.0f : 0.0f,
                           settings.rtRefractions ? 1.0f : 0.0f,
                           settings.rtThickness ? 1.0f : 0.0f,
                           settings.rtLocalShadows ? 1.0f : 0.0f);
    p.distances = glm::vec4(settings.rtMaxReflectDist, settings.rtMaxRefractDist,
                             settings.rtMaxShadowDist, settings.rtRoughnessThreshold);
    // Pipeline-path water look mirrors water layer 0 (rgen has no layer id).
    // The sampled inline path reads each fragment's own WaterParams instead.
    p.water = glm::vec4(waterLook.ior, waterLook.maxThickness,
                         settings.rtCoarseBoxSize, 0.0f);
    p.absorption = glm::vec4(waterLook.absorption[0], waterLook.absorption[1],
                             waterLook.absorption[2], waterLook.absorptionScale);
    p.debug = glm::vec4(static_cast<float>(settings.rtDebugView),
                        rayTracing->tlasBuilt() ? 1.0f : 0.0f,
                        settings.rtSelfSkipDist,
                        settings.rtWaterPipeline ? 1.0f : 0.0f);
    p.invViewProj = invViewProj;
    // Temporal SSR: expose the previous frame's view-projection so solid
    // reflections can reproject into the previous frame's color/depth instead
    // of hammering it with the current camera (which stipples while moving).
    p.prevViewProj = prevViewProj_;
    prevViewProj_ = glm::inverse(invViewProj);
    // LoD band inputs for the scene-geometry snapshot (same values the raster
    // cull uses; consumed on the next rebuild).
    lastBandCamPos_ = viewPos;
    lastBandLodBias_ = settings.lodBias;
    lastBandMaxLod_ = settings.maxTargetLod;
    p.viewPos = glm::vec4(viewPos, 1.0f);
    p.rtResolution = glm::vec4(0.0f);
    p.clipPlanes = glm::vec4(nearPlane, farPlane, 0.0f, 0.0f);
    p.sunDir = glm::vec4(sunDirTo, 0.0f);
    p.sunColor = glm::vec4(sunColor, 1.0f);
    rayTracing->updateParams(p, app->getCurrentFrame());
}
