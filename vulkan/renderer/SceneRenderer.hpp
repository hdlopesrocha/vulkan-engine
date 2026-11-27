#pragma once

// Forward declarations for change handler types
class Octree;
class World;

#include <vulkan/vulkan.h>
#include "Renderer.hpp"
#include "../core/VulkanApp.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include "../resources/TextureArrayManager.hpp"
#include "../resources/MaterialManager.hpp"
#include "../pipeline/ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include "../../math/geometry/Vertex.hpp"
#include "../../math/geometry/Geometry.hpp"
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <deque>
#include <vector>
#include "../../space/Model3DVersion.hpp"
#include "../../space/ThreadPool.hpp"
#include "../../services/OctreeTopologyTypes.hpp"
#include "solid/SolidRenderer.hpp"
#include "video/VideoRenderer.hpp"
#include "vegetation/VegetationRenderer.hpp"
#include "water/WaterRenderer.hpp"
#include "post/PostProcessRenderer.hpp"
#include "sky/SkyRenderer.hpp"
#include "debug/DebugCubeRenderer.hpp"
#include "debug/DebugSDFRenderer.hpp"
#include "sdf/SdfRenderer.hpp"
#include "wind/WindRenderer.hpp"
#include "wind/WindRaymarchRenderer.hpp"
#include "brush/BrushSdfRenderer.hpp"
#include "octree/OctreeRaymarchRenderer.hpp"
#include "indirect/IndirectRenderer.hpp"
#include "../../widgets/CloudSettings.hpp"
#include "../streaming/UploadManager.hpp"   // TerrainStreamer: async streaming orchestration
#include "../../world/World.hpp"

#include "CommandBufferState.hpp"
#include "../LodMesh.hpp"
#include "../PendingMeshData.hpp"
#include "PendingDeleteEntry.hpp"
#include "SceneLayerState.hpp"
#include "StreamStats.hpp"
#include "StreamCounters.hpp"
#include "StaticTextureSignature.hpp"
#include "Set0DescriptorBuffers.hpp"
#include "ChunkMeshProfile.hpp"

class SceneRenderer : public Renderer {
public:
    // Main uniform buffers (one per frame-in-flight)
    std::vector<Buffer> mainUniformBuffers;

    // Materials SSBO
    Buffer materialsBuffer;
    // Per-brush image source table (binding 28, owned by MaterialManager).
    Buffer brushMapBuffer_;
    MaterialManager* materialManagerPtr = nullptr;

    // Water params UBO (binding 7) — stored for descriptor rebinding
    Buffer waterParamsBuffer_;
    // Byte size of waterParamsBuffer_ (vkGetDescriptorEXT forbids WHOLE_SIZE).
    VkDeviceSize waterParamsBufferSize_ = 0;

    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<PostProcessRenderer> postProcessRenderer;
    std::unique_ptr<SolidRenderer> solidRenderer;
    std::unique_ptr<WaterRenderer> waterRenderer;
    // Video layer (MaterialKind::Video geometry): rendered after solid/water
    // into its own target and blended by the post-process composite.
    std::unique_ptr<VideoRenderer> videoRenderer;
    std::unique_ptr<VegetationRenderer> vegetationRenderer;
    std::unique_ptr<BrushSdfRenderer> brushRenderer;
    // The ONLY terrain renderer: raymarches the octree's stored SDFs of EVERY
    // terrain layer (Solid + Water). Owns the set-2 node-table descriptors and
    // the fullscreen raymarch pipeline.
    std::unique_ptr<OctreeRaymarchRenderer> octreeRaymarchRenderer;
    std::unique_ptr<DebugCubeRenderer> debugCubeRenderer;
    std::unique_ptr<DebugCubeRenderer> boundingBoxRenderer;
    std::unique_ptr<DebugSDFRenderer> debugSDFRenderer;
    // Generic GPU-driven SDF renderer (surface/volume/emissive/transparent).
    // Independent from the terrain octree; fire is its first consumer (volumetric
    // material + instanced flame SDF + spatial grid). Own offscreen targets,
    // composited by PostProcess after vegetation/water.
    std::unique_ptr<SdfRenderer> sdfRenderer;
    // Wind-trail particle visualization (Settings::windParticlesEnabled):
    // GPU particles advected by the shared wind field (binding 27) with fading
    // trails. Simulated in preRenderPass, drawn as an overlay after the
    // composite. See WindRenderer.
    std::unique_ptr<WindRenderer> windRenderer;
    // Wind-ray-march mode: renders |wind| == targetSpeed of the shared field as
    // a raymarched isosurface overlay (Settings::windRaymarchEnabled).
    std::unique_ptr<WindRaymarchRenderer> windRaymarchRenderer;

    // See setVegetationRenderScale().
    float vegetationRenderScale_ = 1.0f;
    // Sky settings owned by this renderer
    std::unique_ptr<SkySettings> skySettings;
    SkySettings& getSkySettings() { return *skySettings; }
    // Volumetric cloud settings (three altitude tiers). Shares the SkyUBO.
    std::unique_ptr<CloudSettings> cloudSettings;
    CloudSettings& getCloudSettings() { return *cloudSettings; }

    SceneRenderer();
    ~SceneRenderer();

    // Cleanup and resource destruction (accepts app for Vulkan operations)
    void cleanup(VulkanApp* app) override;

    // Propagate the shared per-frame command state tracker to every renderer
    // that records on the main thread.
    void setCmdState(CommandBufferState* state) override;

    // Drain generation pools.  Must be called after all Octree pools are
    // stopped (their workers may still be enqueuing tasks to these pools).
    void stopGenPools();
   
    // Pending change queues (thread-safe)

    // ── Dynamic per-layer GPU state ──
    // One entry per Scene layer (index = Layer::Id). Each layer tracks its own
    // chunk registry + deferred-delete slots + generation pool, while the two
    // shared IndirectRenderers (solid/water) are selected per layer via the
    // Scene's LayerRendererType (LayersWidget). States are heap-allocated so
    // PublishTarget references stay stable across vector growth.

    // Texture arrays (owned by main). Set once in init(); read on the render
    // thread for memory reporting.
    TextureArrayManager* textureArrays_ = nullptr;

    // ── Chunk tracking (per Scene layer, index = Layer::Id) ──
    // Slots whose chunks were erased but may be replaced (same NodeID, new
    // version). For solid/water the octree node is reused on edit, so NodeID
    // stays stable — addMeshSlotted finds the existing entry and republishes
    // it in place (new packed span per level, old span freed after the
    // replacement upload completes). The old slot must survive until the new
    // upload completes to avoid a 1-frame hole. Entries are matched by NodeID
    // and aged out after MAX_FRAMES_IN_FLIGHT frames past their birth frame.
    // ── World reference (separates world logic from rendering) ──
    // The World owns chunks, octrees, and the ChunkManager state machine.
    // The SceneRenderer only reads chunk state and produces/consumes
    // per-level meshes. Call setWorld() before scene loading.
    void setWorld(World* world) { world_ = world; }
    World* world() { return world_; }
    const World* world() const { return world_; }

    // ── Dynamic per-layer state access ──
    // ensureLayerStates grows the vector to at least n entries (creating
    // generation pools on demand). All accessors auto-grow.
    void ensureLayerStates(size_t n);
    std::unordered_map<NodeID, Model3DVersion>& layerChunks(Layer::Id layer);
    std::recursive_mutex& layerMutex(Layer::Id layer);
    std::unordered_map<NodeID, PendingDeleteEntry>& layerPendingDeletes(Layer::Id layer);
    ThreadPool& layerGenPool(Layer::Id layer);
    size_t getLayerModelCount(Layer::Id layer) const;
    // Renderer routing for a layer (reads the active Scene's LayerRendererType;
    // falls back to layer 1 = water, others = solid when no world/scene).
    LayerRendererType layerRendererType(Layer::Id layer) const;
    IndirectRenderer* indirectForLayer(Layer::Id layer);
    // Back-compat: total solid (opaque) count across Solid-mapped layers.
    size_t getRegisteredModelCount() const;
    // Streaming progress: active mesh count across the solid, water and video
    // indirect pools (one per published chunk entry). Main thread only.
    size_t getStreamedMeshCount();
    // Entries waiting for the main-thread drain; zero means the workers have
    // pushed every result they can for the current frame.
    size_t getPendingMeshCount() const;

    // Chunk-streaming counters (diagnostic; VULKAN_STREAM_DIAG prints them):
    // how the GPU topology path resolved every node it was asked to build.
    StreamStats streamStats() const;

    // Per-chunk GPU mesh-build profile: cross-edge count, reserved capacities,
    // actual counts, overflow/growth events. The snapshot is versioned and
    // cached inside the registry, so the UI can call this every frame.
    std::shared_ptr<const ChunkMeshProfileSnapshot> chunkMeshProfileSnapshot() const;
    // Aggregate summary (always) plus one line per chunk when
    // VULKAN_CHUNK_PROFILE is set. Called after scene load/generate through
    // logMemoryUtilization(), and periodically by the stream diagnostic.
    void logChunkMeshProfiles() const;

    // Relaxed atomics written from the build path (worker/main threads).
    StreamCounters streamStats_;

    // Per-chunk build profile (mutex-protected registry; see
    // ChunkMeshProfile.hpp). Lives here rather than on OctreeNode because it is
    // renderer telemetry, not scene topology.
    ChunkMeshProfileRegistry chunkMeshProfiles_;

    // Remove all registered meshes for one layer (from both IRs — the layer
    // may have been remapped since its slots were published) and clear it.
    void removeAllLayerMeshes(Layer::Id layer);
    // Remove every layer's meshes (clears both IRs).
    void removeAllMeshes();

    // Pool utilization telemetry (perf report 22 C1): logs used vs committed
    // for every mesh pool plus vegetation/debug streams. Called after scene
    // load/generation; steady-state cost is zero (console output only).
    void logMemoryUtilization();

    // ── Staged bring-up (S1) ─────────────────────────────────────────────
    // beginInit() runs the serial prologue and starts the four heavy renderer
    // groups on worker threads, then returns. pollInit() is a non-blocking
    // "are the groups done?" check; finishInit() joins them and performs the
    // main-thread wiring (vegetation, descriptors, pools).
    void beginInit(VulkanApp* app_, TextureArrayManager* textureArrayManager);
    bool pollInit();
    void finishInit(VulkanApp* app_, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterSettings>& waterParams);
    bool initStarted() const { return initStarted_; }
    // Re-update main descriptor set when texture arrays are (re)allocated
    void updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager);

    // ── Descriptor-buffer migration (Phase 1) ─────────────────────────────
    // Step 1: allocate 3 host-visible descriptor buffers (one per frame).
    // No-op (leaves descBuffers_.ready false) when !app->useDescriptorBuffer().
    void initDescriptorBuffers(VulkanApp* app);
    // Step 2/4: (re)write all static set-0 bindings into every frame's
    // descriptor buffer via DescriptorBuffer::writeBuffer/writeImage.
    // No-op when buffers are not ready. Sources that are not yet allocated
    // are skipped.
    void writeStaticDescriptorsToBuffers(VulkanApp* app, TextureArrayManager* textureArrayManager);
    bool hasDescriptorBuffers() const { return descBuffers_.ready; }
    // Step 3: bind frame N's descriptor buffer + set-0 offset. Activates once
    // the main layout carries DESCRIPTOR_BUFFER_BIT (see
    // SceneDescriptorLayout::descriptorBufferBindActive); until then this is
    // the documented per-frame bind point for the cutover, kept uncalled so
    // the classic set-0/set-1/set-2 binds stay valid.
    void bindSet0ForFrame(VkCommandBuffer cmd, VulkanApp* app,
                          VkPipelineLayout layout, uint32_t frameIndex) const;
    // Step 5 helper: true only when the descriptor-buffer bind path is live
    // (extension supported + buffers ready + layout capable). False keeps the
    // existing vkUpdateDescriptorSets path unchanged.
    bool descriptorBufferBindActive(VulkanApp* app) const;
    void destroyDescriptorBuffers(VulkanApp* app);
    // cleanup declared above (accepts VulkanApp*)

    using GeometryHandler = const std::function<void(Layer::Id, NodeID, const LoDMesh&)>&;

    // Process nodes from a generic per-layer NodeID->OctreeNodeData map
    // Process nodes for a single Layer (nodeMap maps NodeID->OctreeNodeData)
    void processNodeLayer(
        Scene& scene, 
        Layer::Id layer, 
        NodeID nid, 
        OctreeNodeData& nodeData, 
        GeometryHandler onGeometry, 
        ThreadPool* poolOverride = nullptr
    );

    // ── H3 (perf report 29): gated debug collection ─────────────────────────
    // Runs the SDF-cube walk and/or the mesh-bounding-box walk for one chunk,
    // each only when its renderer's enabled bit is set (DebugSDFRenderer /
    // DebugCubeRenderer::isEnabled). The bit is the ONLY gate — the draw path
    // still checks Settings::showSDFDebug / showBoundingBoxes, so this never
    // changes what is drawn; it only stops the walk and the retained per-chunk
    // records when the overlay is off. processNodeLayer calls it for every
    // built/up-to-date chunk; MyApp::recollectDebugOverlays calls it once per
    // live chunk when an overlay is switched on mid-session.
    void collectDebugForChunk(Scene& scene, Layer::Id layer, NodeID nid,
                              OctreeNodeData& nodeData, ThreadPool* poolOverride = nullptr);

    // ── GPU-resident chunk build ─────────────────────────────────────────────
    // Runs the GPU topology pass for one chunk and hands the publisher an
    // LoDMesh that already lives in the renderer's packed pools
    // (gpuSlot/gpuVertexCount/gpuIndexCount set). There is no host-visible
    // vertex mirror: the Cartesian stream stays device-local and is never read
    // back. Returns true when this function took ownership of the node
    // (published or deliberately skipped); false when the GPU pass is
    // unavailable, in which case the chunk is skipped (no CPU fallback).
    // `published` is set when the revision is settled (geometry installed, or
    // legitimately empty); it stays false on a failed build so the caller does
    // not mark the version up-to-date and a later re-emit retries.
    bool buildChunkResident(Layer::Id layer, NodeID nid, OctreeNodeData& nodeData,
                            uint8_t chunkLod, uint32_t version, OctreeTopologyInput& topoInput,
                            IndirectRenderer& ir, GeometryHandler onGeometry, bool& published);

    // ── Slotted-mode chunk processing ────────────────────────────────────────
    // Initialize the slotted rendering pipeline for solid and water layers.
    // Creates the slot pools, GPU buffers, and switches the renderers to
    // stable-slot mode. Solid and water pools are sized independently (solid is
    // the dense layer, water is sparse). Must be called once on the main thread.
    void initSlottedMode(VulkanApp* app, uint32_t maxSolidChunks,
                         uint32_t vertexBytesPerChunk = 1u << 20,
                         uint32_t indexBytesPerChunk  = 1u << 19);

    // Drain the swap queue and atomically swap in new RenderProxies.
    // Also retires old proxies via deferred destruction.
    // Call once per frame from processPendingMeshes.
    void processChunkSwapQueue();

    // Runtime introspection helpers for UI/debug
    size_t getTransparentModelCount();
    // ── Solid SSR (bindings 19/20) ────────────────────────────────────────
    // SolidRenderer.frag resolves solid reflections against the real rendered scene by
    // marching the *previous* frame's solid color/depth. Each per-frame main
    // descriptor set is bound once to the solid views of the slot it does NOT
    // render into, so the feedback loop (sampling the color attachment being
    // written) never happens and no per-frame descriptor updates are needed.
    // The solid pass waits on the previous frame's tlSolid value, so the source
    // images are complete and visible before the ray march samples them.
    VkSampler ssrColorSampler = VK_NULL_HANDLE;
    VkSampler ssrDepthSampler = VK_NULL_HANDLE;
    void initSSRSamplers(VulkanApp* app);
    void writeSSRBindings(VulkanApp* app);
    void destroySSRSamplers(VulkanApp* app);

    // The solid/water space-change lambdas are constructed by the app
    // (MyApp.cpp) — they need world state, chunk management and debug
    // markers. No make* handler factories remain here.

    // Resize offscreen resources when the swapchain changes
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    // Vegetation offscreen render scale (Settings::vegetationRenderScale): the
    // vegetation color + depth targets are created at width*scale x
    // height*scale. Same idle-recreate contract as the water targets (the veg
    // pass runs on its own queue, so a graphics-queue wait would not cover it).
    void setVegetationRenderScale(float scale) { vegetationRenderScale_ = scale; }
    float vegetationRenderScale() const { return vegetationRenderScale_; }
    void recreateVegetationTargets(VulkanApp* app, uint32_t width, uint32_t height);

    // ── Parallel scene loading ─────────────────────────────────────────────────
    // Drains the shared pending mesh queue (main thread) into a caller-provided
    // batch, then processes it. Call once per frame from update() before
    // recording command buffers. One shared per-frame budget spans both
    // streams (solid + water).
    void drainPendingMeshes(std::deque<PendingMeshData>& out, size_t maxCount);

    // Unify the given pending geometry batch into the GPU. One common path for
    // solid and water: each entry routes itself to the right IndirectRenderer
    // and deferred-slot bookkeeping by layer.
    void processPendingMeshes(VulkanApp* app, std::deque<PendingMeshData>& batch);

    // Async streaming orchestrator (parallel per-category pools, lock-free
    // queues, drop-in staging upload manager). Currently scaffolded: its
    // update()/frame-sync runs each frame; terrain/water GPU copies still
    // flow through IndirectRenderer until that path is migrated to use it.
    streaming::TerrainStreamer streamer;

private:
    // Single publish core for a pending mesh batch — every layer behaves
    // identically. Publishes each generated geometry chunk AS RECEIVED: every
    // PendingMeshData is ONE self-contained mesh (no ladder structures). Each
    // chunk occupies a slot; the mesh lands in the slot row for its LoD level
    // and publishes its own draw entry. No per-chunk reassembly. Returns the
    // number of slots published.
    //
    // The core selects, per-entry, by the Scene's LayerRendererType, which
    // IndirectRenderer and per-layer slot bookkeeping apply. Each chunk's
    // build state runs through the ChunkManager. Solid chunks also drive the
    // GPU-resident effect consumers (vegetation placement span + lava anchors).
    size_t publishPendingMeshes(
        VulkanApp* app,
        std::deque<PendingMeshData>& batch);

    // Age out pending-delete slots older than MAX_FRAMES_IN_FLIGHT
    // (genuine deletions with no replacement chunk). Routes each layer's
    // entries to its current IndirectRenderer.
    void ageOutPendingDeletes(uint32_t curFrame);

    // World reference (null until setWorld is called).
    // The World owns ChunkManager and all chunk state.
    World* world_ = nullptr;

    // Descriptor-buffer migration: change guard for updateTextureDescriptorSet().
    // Static bindings (texture arrays, materials, water params) are
    // written once at init and re-written only when the underlying resources
    // actually change (texture realloc, resize). updateTextureDescriptorSet() is
    // event-driven (allocation listener / swapchain resize) — never called from
    // the render loop — and this signature makes repeated invocations with
    // unchanged resources a no-op, guaranteeing 0 vkUpdateDescriptorSets calls
    // in steady state. When VulkanApp::useDescriptorBuffer() is true the same
    // guard selects the GPU-side path (direct vkGetDescriptorEXT host writes
    // into the descriptor buffer, overlapped with compute) instead of the
    // classic vkUpdateDescriptorSets fallback below.
    StaticTextureSignature lastStaticSignature_{};

    // Build the StaticTextureSignature for the current resources (shared by
    // init and updateTextureDescriptorSet so both agree on "unchanged").
    StaticTextureSignature currentStaticSignature(TextureArrayManager* textureArrayManager) const;

    // ── Descriptor-buffer migration (Phase 1: static set-0 bindings) ──────
    // Three host-visible descriptor buffers (one per frame-in-flight) holding
    // the full set-0 image/buffer descriptors, written via vkGetDescriptorEXT
    // (DescriptorBuffer wrapper) instead of vkUpdateDescriptorSets. Binding 0
    // (per-frame UBO) stores the UBO device address once — per-frame UBO
    // *contents* still stream via memcpy into the already-bound UBO buffers.
    // While SceneDescriptorLayout::descriptorBufferBindActive() is false the
    // buffers are warm mirrors (classic sets stay authoritative); once the
    // main layout flips to DESCRIPTOR_BUFFER_BIT, bindSet0ForFrame() becomes
    // the bind path and classic set-0 writes stop.
    // Offsets cover the set-0 bindings in use (binding 11 stays absent —
    // legacy cubemap removed).
    Set0DescriptorBuffers descBuffers_;

    // Async visible-count snapshot for stats (updated once per frame in
    // processPendingMeshes from the IRs' host-visible readback buffers;
    // 1-frame latency, never touches GPU memory). The overlay reads these.
    uint32_t lastOpaqueVisible_ = 0;
    uint32_t lastTransparentVisible_ = 0;

public:
    uint32_t getLastOpaqueVisible() const { return lastOpaqueVisible_; }
    uint32_t getLastTransparentVisible() const { return lastTransparentVisible_; }

    // Thread-safe mesh queue fed by tessellation on the generation pools;
    // drained on the main thread by processPendingMeshes(). ONE shared queue
    // for all scene layers.
    mutable std::mutex                          pendingMeshMutex;
    std::unordered_map<NodeID, PendingMeshData> pendingMeshQueue;

    CommandBufferState frameCmdState;

private:
    // Per-layer GPU state (index = Layer::Id). Guarded by layerStatesMutex_ for
    // add/remove; each entry's own mutex guards its chunk maps.
    mutable std::mutex layerStatesMutex_;
    std::vector<std::unique_ptr<SceneLayerState>> layerStates_;

    // ── M7 (perf report 29): debug-walk instrumentation ────────────────────
    // Enabled once from VULKAN_SETUP_DIAG in the constructor. When false the
    // collection path reads one bool and touches no clock, atomic or log; when
    // true it accumulates walks/cubes/boxes and the walk time and prints one
    // [setup-diag] line every 256 chunks plus a final flush in cleanup().
    // Relaxed atomics: counters only, no ordering requirement (workers update).
    bool debugWalkDiag_ = false;
    std::atomic<uint64_t> dbgWalks_{0};        // subtree walks (SDF + bbox)
    std::atomic<uint64_t> dbgWalkCubes_{0};    // SDF cubes collected
    std::atomic<uint64_t> dbgWalkBoxes_{0};    // bounding boxes collected
    std::atomic<uint64_t> dbgWalkTotalUs_{0};  // cumulative walk time
    std::atomic<uint64_t> dbgWalkWindowUs_{0}; // walk time since the last print
    std::atomic<uint64_t> dbgWalkWindowWalks_{0};
    std::atomic<uint32_t> dbgWalkChunks_{0};   // chunks that ran >= 1 walk
    void logDebugWalkStats(bool force);

    // ── Staged bring-up (S1) ─────────────────────────────────────────────
    // The four renderer groups started by beginInit() and joined by
    // finishInit(). They must be members because beginInit() returns while
    // they are still running; lambdas capture app/size by value.
    std::future<void> initFutureSolid_;
    std::future<void> initFutureVideo_;
    std::future<void> initFutureSky_;
    std::future<void> initFutureDebug_;
    bool initStarted_ = false;
    uint32_t initWidth_ = 0;
    uint32_t initHeight_ = 0;
};

// ...existing code...
