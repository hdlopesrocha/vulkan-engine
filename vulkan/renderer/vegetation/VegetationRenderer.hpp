#pragma once
#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../core/TrackedHandle.hpp"
#include "../../resources/TextureArrayManager.hpp"
#include "../../../math/geometry/Vertex.hpp"
#include "../../../math/geometry/Geometry.hpp"
#include "../debug/DebugCubeRenderer.hpp"
#include "../../resources/VertexBufferObject.hpp"
#include "../../../utils/Scene.hpp" // for NodeID
#include "../../ubo/WindParamsUBO.hpp"
#include "../../ubo/WindField.hpp"
#include <vector>
#include <deque>
#include <unordered_map>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <glm/glm.hpp>
#include <glm/gtc/round.hpp>
#include "../CommandBufferState.hpp"

class IndirectRenderer; // merged-cull integration (forward decl)

// Per-chunk bladed-grass instance buffer and renderer (triangle tufts,
// procedural color, no billboards, no SDF). Impostor quads capture the blades
// for the far LOD.
class VegetationRenderer : public Renderer {
public:
    struct WindSettings {
        bool enabled = true;
        glm::vec2 direction = glm::vec2(1.0f, 0.0f);
        float strength = 4.0f;
        float baseFrequency = 0.003f;
        float speed = 0.75f;
        float gustFrequency = 0.012f;
        float gustStrength = 0.45f;
        float skewAmount = 1.75f;
        float trunkStiffness = 0.70f;
        float noiseScale = 1.0f;
        float verticalFlutter = 0.20f;
        float turbulence = 0.60f;
    };

    struct WindPushConstants {
        float billboardScale = 1.0f;
        float windEnabled = 1.0f;
        float windTime = 0.0f;
        float impostorDistance = 0.0f;
    };
    static_assert(sizeof(WindPushConstants) == 16, "WindPushConstants expected 16 bytes");

    // Distance-density thinning for the impostor (far) range. Blades never
    // thin: the near field stays dense. Defaults keep far meadows reading
    // as grassland (half density at 4 km) instead of decaying to ~10% and
    // exposing bare terrain — that mismatch against full-density blades is
    // what reads as an impostor/blade color edge in wide vistas. Tune live
    // in the Wind widget (Distance Density section).
    struct DistanceDensitySettings {
        bool enabled = true;
        float fullDensityDistance = 1200.0f;
        float minDensityDistance = 4096.0f;
        float minDensityFactor = 0.50f;
    };

    float billboardScale = 10.0f;
    uint32_t billboardCount = 3; // blade variants: 0=meadow, 1=lawn, 2=dry (40% of instances are empty sentinel)
    explicit VegetationRenderer();
    ~VegetationRenderer();

    void setTextureArrayManager(TextureArrayManager* mgr, VulkanApp* app);
    // Register the SOLID IndirectRenderer whose merged IndirectRenderer.comp dispatch now
    // also emits the blade/impostor commands. VegetationRenderer supplies its
    // per-frame output buffers + per-chunk veg metadata to that renderer.
    void setSolidIndirectRenderer(IndirectRenderer* ir) { solidIR = ir; }
    void onTextureArraysReallocated(VulkanApp* app);
    void init();
    void cleanup(VulkanApp* app) override;
    void init(VulkanApp* app);
    // CPU-side instance generation — avoids GPUVM faults on RADV iGPUs where
    // the Texture Cache/Pipe cannot read from device-local or host-visible
    // storage buffers.  Enqueues the chunk and processes up to maxPerFrame
    // chunks each frame via processPendingChunks().  With no grass
    // triangles the chunk's previous instance data is cleared instead.
    void generateChunkInstancesCPU(NodeID chunkId,
                                   const std::vector<glm::vec3>& positions,
                                   const std::vector<glm::vec3>& normals,
                                   const std::vector<int>& brushIndices,
                                   const std::vector<uint32_t>& grassIndices,
                                   const glm::vec3& chunkCenter,
                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                   uint32_t seed = 1337);
    // CPU-side per-chunk vegetation generation (moved from SceneRenderer).
    // Samples grass-touching triangles from the chunk's tessellated
    // geometry. Builds area-weighted virtual triangle
    // slots with unbiased stochastic rounding (area-proportional density
    // without bias), shuffles the slots per chunk (so reducing the indirect
    // instanceCount keeps a random spatial subset), then hands the result to
    // generateChunkInstancesCPU. Per-instance placement keeps only samples
    // whose dominant barycentric vertex is grass-brushed, so blades hug
    // meadows at any triangle size without spilling onto other brushes.
    // With no grass triangles the
    // chunk's previous instance data is cleared instead.
    void generateForChunk(VulkanApp* app, NodeID nid, const Geometry& geom);
    // GPU-resident chunk publish: the chunk's Cartesian mesh already lives in
    // the solid renderer's packed pools, written by OctreeTriangles.comp. The
    // placement dispatch reads that span directly and writes the instance
    // records into a fresh device-local buffer; only the resulting count comes
    // back (after the dispatch fence) to publish the chunk.
    void generateForGpuChunk(VulkanApp* app, NodeID nid,
                             uint32_t baseVertex, uint32_t vertexCount,
                             uint32_t firstIndex, uint32_t indexCount,
                             const glm::vec3& boundsMin, const glm::vec3& boundsMax);
    // Drain up to maxChunks from the pending queue.  Call every frame from
    // draw() so chunks trickle in at a controlled rate. Dispatches to the GPU
    // placement path (processPendingChunksGPU) when available, else the CPU
    // path.
    void processPendingChunks(uint32_t maxChunks);
    // Number of chunks still waiting in the queue.
    size_t pendingChunkCount() const;
    void clearAllInstances();

    // (The combined render() entry point was removed with the dead code sweep,
    // perf report 22 M9: all callers use drawDepth/drawColor directly.)
    // Deferred depth test: draw vegetation + impostor depth only (no color)
    void drawDepth(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::vec3& cameraPos);
    // Deferred depth test: draw blades + impostor color only (LESS_OR_EQUAL, no depth write).
    // drawBillboards=false skips the blade pass; drawImpostors gates the captured-blade pass.
    void drawColor(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::vec3& cameraPos,
                   bool drawBillboards = true, bool drawImpostors = true);
    void recordReadBarriers(VkCommandBuffer& commandBuffer);
    // H7 (report 26): provably-empty gate for the veg block + draw entry
    // points. True iff a fence-COMPLETE consolidation exists (non-empty
    // chunk map AND completed vegNumChunks > 0): the cull and the draws
    // consume only completed snapshots, so in-flight/streaming work never
    // makes this false while drawable data exists. False ⟹ drawDepth/
    // drawColor currently issue zero indirect draws, so skipping the
    // render-pass instances is pixel-identical. Reads plain renderer members
    // also read by the existing draw paths (same cross-thread class).
    bool hasDrawContent() const;
    // H7 (report 26, unified by M12 in report 27): veg-pass skip arbiter
    // with transition + warmup guards (mirrors the report-25 M11 SDF
    // elision). Returns false only for steady-empty frames (no work, no
    // transition, past warmup); every input transition runs the full path.
    // Thread-safe for overlapping callers; worst case is a missed skip,
    // never a missed draw: a skip requires both inputs steady AND no work.
    //
    // M12 (perf report 27): UNIFIED veg-pass arbiter. The old main-thread
    // duplicate (MyApp statics vegPrevWantDraw/vegWarmupRuns over the
    // settings-level vegWantDraw formula) is removed; the single decision
    // lives here over both inputs: wantDraw (settings: billboards and/or
    // impostors enabled) and hasContent (hasDrawContent()). The transition
    // memory tracks EACH input separately (packed bits in vegInstancePrev),
    // so a simultaneous opposite toggle still forces a full frame — a skip
    // additionally requires both inputs steady. Skip implies no draws by
    // construction: !wantDraw forces both draw flags false (clear-only at
    // most), and !hasContent means the draw streams no-op. The main thread
    // evaluates this BEFORE enqueueing either the veg or the water task, so
    // H7 can elide the veg submit and reroute water's wait on skip frames.
    // Steady-state skip outcomes are identical to the old OR of the two
    // arbiters; transition/warmup frames always run full (safe direction).
    // The assert inside verifies the never-miss-a-draw invariant.
    static constexpr uint32_t kVegSkipWarmupFrames = 3;
    bool beginVegPass(bool wantDraw, bool hasContent);

    // ── Own offscreen framebuffer (decoupled from the solid pass) ──
    // Vegetation is rendered to its own color+depth images so it can be drawn on a
    // parallel async command buffer (mirrors the water offscreen targets). The
    // solid pass no longer shares its depth with vegetation; occlusion against
    // solid geometry is resolved at composite time (PostProcessRenderer.frag) by testing
    // the vegetation depth against the solid scene depth.
    static constexpr uint32_t VEG_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    VkImageView getVegColorView(uint32_t frameIndex) const { return (frameIndex < VEG_FRAMES) ? vegColorImageViews[frameIndex] : VK_NULL_HANDLE; }
    VkImageView getVegDepthView(uint32_t frameIndex) const { return (frameIndex < VEG_FRAMES) ? vegDepthImageViews[frameIndex] : VK_NULL_HANDLE; }
    VkImage getVegColorImage(uint32_t frameIndex) const { return (frameIndex < VEG_FRAMES) ? vegColorImages[frameIndex] : VK_NULL_HANDLE; }
    VkImage getVegDepthImage(uint32_t frameIndex) const { return (frameIndex < VEG_FRAMES) ? vegDepthImages[frameIndex] : VK_NULL_HANDLE; }
    uint32_t getVegWidth() const { return vegRenderWidth; }
    uint32_t getVegHeight() const { return vegRenderHeight; }
    VkImageLayout getVegColorLayout(uint32_t frameIndex) const { return (frameIndex < VEG_FRAMES) ? vegColorImageLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED; }
    void setVegColorLayout(uint32_t frameIndex, VkImageLayout lay) { if (frameIndex < VEG_FRAMES) vegColorImageLayouts[frameIndex] = lay; }
    VkImageLayout getVegDepthLayout(uint32_t frameIndex) const { return (frameIndex < VEG_FRAMES) ? vegDepthImageLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED; }
    void setVegDepthLayout(uint32_t frameIndex, VkImageLayout lay) { if (frameIndex < VEG_FRAMES) vegDepthImageLayouts[frameIndex] = lay; }
    
    // (The single-cascade drawShadow entry point was removed with the dead
    // code sweep — the parallel cascade path below uses drawShadowCascade.)
    // NOTE: vkCmdDrawIndexedIndirectCount is core since Vulkan 1.2 and is
    // called directly (device creation requires drawIndirectCount).

    // Stats helpers
    size_t getChunkCount() const { return chunkInstanceCounts.size(); }
    size_t getInstanceTotal() const;
    // Pool utilization telemetry (perf report 22 C1): chunk/instance counts
    // vs the fixed worst-case reservation, with a <25% warning gated on
    // non-empty. Called after scene load alongside the renderer pools.
    void logUtilization() const;

    WindSettings& getWindSettings() { return windSettings; }
    const WindSettings& getWindSettings() const { return windSettings; }
    // Shared wind-field settings (wind home). Owned here so the widget agent
    // has one getter; packed to the set=0/binding=27 UBO by
    // updateWindFieldUBO().
    WindFieldSettings& getWindFieldSettings() { return windFieldSettings; }
    const WindFieldSettings& getWindFieldSettings() const { return windFieldSettings; }
    DistanceDensitySettings& getDistanceDensitySettings() { return distanceDensitySettings; }
    const DistanceDensitySettings& getDistanceDensitySettings() const { return distanceDensitySettings; }
    // Per-frame clock (seconds). Runs updateWindFieldUBO() so the shared
    // field tracks the same cadence as SkySphere::update (called from the
    // adjacent MyApp lines) with no MyApp edit; write-on-change inside keeps
    // static scenes free of per-frame memcpys.
    void setWindTime(float timeSeconds);

    // Impostor rendering.  Call after init() once impostor views have been captured.
    // albedoArray60 and normalArray60 must be VkImageView covering 80 layers
    // (3 blade variants × 20 Fibonacci views = 60 layers).
    // depthArray60 is the captured device Z array (R32_SFLOAT, 60 layers) for depth reprojection.
    // captureInvVPBuf is a storage buffer containing per-layer inverse VP matrices.
    void setImpostorData(VulkanApp* app,                         VkImageView albedoArray60,
                         VkImageView normalArray60,
                         VkSampler sampler,
                         VkImageView depthArray60 = VK_NULL_HANDLE,
                         VkBuffer   captureInvVPBuf = VK_NULL_HANDLE,
                         // M11 (perf report 27): capture fence for the async
                         // path (captureAllAsync contract: wait it before
                         // first sampling). VK_NULL_HANDLE when no capture is
                         // in flight.
                         VkFence    captureFence = VK_NULL_HANDLE);

    // Distance beyond which vegetation instances are replaced by impostor quads.
    // Set to 0 (default) to disable impostor rendering entirely.
    void setImpostorDistance(float dist) { impostorDistance = dist; }

    // Cross-queue guard for the consolidation copy (flicker fix): the pair it
    // overwrites was last read by the previous era's vegetation draws on the
    // vegetation queue (and, transitively through tlShadow, the shadow
    // cascade draws on the cube queues — the veg task waits tlShadow every
    // frame before signaling its own timeline). The frame thread pushes the
    // vegetation timeline semaphore + latest signaled value each frame; the
    // consolidation submit waits it so the overwrite never races a reader.
    // VK_NULL_HANDLE (pre-startup) skips the wait.
    void setConsolidationWait(VkSemaphore sem, uint64_t value) {
        consolidationWaitSemaphore = sem;
        consolidationWaitValue = value;
    }

    // Build the concatenated instance buffer and per-chunk metadata for GPU
    // frustum culling. Must be called once after all chunks are generated.
    // Uses a temporary command buffer (synchronous, one-time cost).
    void consolidateChunks(VulkanApp* app);
    // Publish a fence-complete consolidation generation (completedSnapshot_
    // + vegNumChunks + table-dirty flag). Idempotent per generation; safe to
    // call every frame from prepareCull/prepareCullCascades. Returns true
    // when a NEW generation completed (caller pushes metadata). Uses the
    // ground-truth fence query, never the pump-dependent tracking list.
    bool pollConsolidation();

    // Per-instance payload: pos vec4 (xyz = world pos, w = bladeVariant +
    // rotFrac) followed by a normal vec4 (xyz = surface normal, w unused).
    // One 32-byte stride shared by the per-chunk buffers, the concatenated
    // instance buffer, and every vertex-input declaration (binding 1).
    static constexpr VkDeviceSize kInstanceStride = sizeof(glm::vec4) * 2;
    // Worst-case vegetation capacities, sized ONCE at startup so no runtime
    // vmaCreateBuffer calls occur after the first frame. 4096 matches the
    // solid slotted-mode chunk ceiling (every veg chunk keys off a solid chunk
    // id). The instance figure is the TOTAL concatenated budget divided
    // evenly across chunks; an individual coarse rung can carry more tufts
    // (placement sizes each persistent buffer from its measured count).
    static constexpr uint32_t kMaxVegChunks = 4096;
    static constexpr uint32_t kMaxVegInstancesPerChunk = 2048;
    // GPU placement scratch: the dispatch writes into a generously sized slice
    // of one shared ring buffer, then the exact per-chunk instance buffer is
    // allocated from the measured count and the slice is copied into it. This
    // keeps the persistent per-chunk buffers exactly sized while the output
    // capacity (chunk area x density, potentially thousands of tufts) is only
    // bounded by the scratch slice. Coarse ancestor rungs can reach ~1e5 tufts
    // in a wide terrain; 131k leaves headroom without a per-chunk reservation.
    static constexpr uint32_t kPlacementBatchMax = 10;
    static constexpr uint32_t kPlacementScratchPerChunk = 131072;
    static constexpr uint32_t kPlacementScratchCount = 2;
    struct PlacementScratch {
        Buffer buffer;
        std::atomic<bool> inUse{false};
    };
    std::array<PlacementScratch, kPlacementScratchCount> placementScratch;
    static constexpr VkDeviceSize kMaxVegInstances =
        static_cast<VkDeviceSize>(kMaxVegChunks) * kMaxVegInstancesPerChunk;
    // Pre-allocate ALL culling buffers to worst-case capacity in a single
    // init-time burst: concatenated instances, per-frame compact/count (main
    // + impostor), chunk-info table, and cascade buffers. Idempotent — a
    // second call with the same sizes is a no-op; different sizes assert.
    // Must be called once during SceneRenderer::init before scene loading.
    void preallocate(VulkanApp* app, uint32_t maxChunks = kMaxVegChunks,
                     uint32_t maxInstancesPerChunk = kMaxVegInstancesPerChunk);

    // GPU frustum culling: dispatch compute shader that culls chunks against
    // viewProj and compacts visible draw commands. Must be called OUTSIDE any
    // render pass (compute dispatches are illegal inside dynamic rendering).
    // Auto-cycles through triple-buffered culling slots internally.
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj);

    // Cascade-aware culling: single dispatch that culls against all 3 cascade
    // frustums simultaneously. Each chunk is culled independently per cascade.
    void prepareCullCascades(VkCommandBuffer cmd,
                             const glm::mat4 cascadeMatrices[3]);
    // Draw a specific cascade's vegetation compacted output.
    void drawShadowCascade(VulkanApp* app, VkCommandBuffer& commandBuffer,
                           VkDescriptorSet shadowDescriptorSet,
                           const glm::vec3& cameraPos,
                           uint32_t cascadeIndex);

    // Update the wind params UBO with current settings.
    // Must be called before any draw that uses wind.  Updates per-frame values
    // (camera position, falloff) so windParams on the GPU stays in sync.
    void updateWindParamsUBO(const glm::vec3& cameraPos);

    // Pack + upload the shared wind-field UBO (set=0, binding=27) from
    // WindSettings (ambient mirror) + WindFieldSettings (tornado list).
    // Write-on-change via memcmp (mirrors updateWindParamsUBO): static scenes
    // issue zero per-frame writes. Called per frame from setWindTime().
    void updateWindFieldUBO();
    // Scene-owned shared buffer for SceneRenderer::init to bind once into the
    // static/per-frame main sets (contents stream via memcpy afterwards).
    Buffer getWindFieldBuffer() const { return windFieldBuffer; }

    // Shared set=2 wind params resources. Other consumers of the vegetation
    // shader family (e.g. ImpostorCapture) bind the same layout + descriptor
    // set instead of duplicating them.
    VkDescriptorSetLayout getWindParamsDescSetLayout() const { return windParamsDescSetLayout; }
    VkDescriptorSet getWindParamsDescSet() const { return windParamsDescSet; }

private:
    
    TrackedHandle<VkPipeline> vegetationPipeline;
    TrackedHandle<VkPipeline> vegetationDepthPipeline;
    TrackedHandle<VkPipelineLayout> vegetationDepthPipelineLayout;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkPipeline> vegetationShadowPipeline;
    TrackedHandle<VkPipelineLayout> shadowPipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TextureArrayManager* vegetationTextureArrayManager = nullptr;
    VkImageView billboardAlbedoView   = VK_NULL_HANDLE;
    VkImageView billboardNormalView   = VK_NULL_HANDLE;
    VkImageView billboardOpacityView  = VK_NULL_HANDLE;
    TrackedHandle<VkSampler> billboardArraySampler;

    // Descriptor set allocated once from the app's descriptor pool and updated
    // in place when the texture arrays are (re)allocated. The layout carries
    // UPDATE_AFTER_BIND_POOL_BIT (pool has UPDATE_AFTER_BIND_BIT), so an
    // in-place vkUpdateDescriptorSets is legal even while pending command
    // buffers reference the set — no free/realloc and no deferred destruction
    // is ever needed. With VK_EXT_descriptor_buffer this update becomes a
    // plain host memory write (vkGetDescriptorEXT); no set allocation/free.
    TrackedHandle<VkDescriptorSet> vegDescriptorSet;
    bool ensureVegDescriptorSet(VulkanApp* app);
    // Rewrite the 3 billboard bindings into the existing set (event-driven
    // only: onTextureArraysReallocated — never called from the render loop,
    // so steady state issues 0 updates).
    void refreshVegDescriptors(VulkanApp* app);
    // Listener id returned from TextureArrayManager::addAllocationListener(), -1 if none
    int vegTextureListenerId = -1;

    struct InstanceBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        glm::vec3 center = glm::vec3(0.0f);
        glm::vec3 aabbMin = glm::vec3(0.0f);
        glm::vec3 aabbMax = glm::vec3(0.0f);
        size_t count = 0;
    };
    std::unordered_map<NodeID, InstanceBuffer> chunkBuffers;
    std::unordered_map<NodeID, size_t> chunkInstanceCounts;
    void destroyInstanceBuffer(NodeID chunkId, VulkanApp* app = nullptr, VkFence completionFence = VK_NULL_HANDLE);
    // Tombstone a chunk in the consolidation snapshots (count = 0, offsets
    // kept): drops it from future culls/draws without shifting the
    // concatenated layout, so in-flight and completed generations stay
    // consistent until the next submit rebuilds from the live map. Called
    // from destroy paths; also marks the chunk-info table dirty.
    void tombstoneConsolidated(NodeID chunkId);

    // Pending CPU-generation queue — chunks are enqueued by the scene loader
    // and drained 10-per-frame by draw().
    struct PendingChunk {
        NodeID chunkId;
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals; // per-vertex normals, parallel to positions
        std::vector<int> brushIndices; // per-vertex brush, parallel to positions
        std::vector<uint32_t> grassIndices;
        glm::vec3 chunkCenter;
        uint32_t instancesPerTriangle;
        uint32_t seed;
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        // GPU-resident publish (generateForGpuChunk): the mesh already lives in
        // the solid renderer's packed pools at these spans, so the vectors
        // above stay empty and processPendingChunksGPU reads the pools.
        bool gpuResident = false;
        uint32_t baseVertex = 0;
        uint32_t vertexCount = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
    };
    std::deque<PendingChunk> pendingChunks;
    mutable std::mutex pendingChunksMutex;
    // If the renderer was initialized with an app, this will be set and
    // allows immediate compute-based generation calls to run against the
    // provided `VulkanApp` instance.
    VulkanApp* appPtr = nullptr;
    // Simple VBO that provides the per-vertex 'base' used by the vegetation
    // pipeline. We use a single base vertex and expand in the shader via
    // the instance data.
    VertexBufferObject billboardVBO;

    // Separate VBO for impostor quads (4 vertices, 6 indices, unit-square
    // UV corners).  Expanded per-instance in the vertex shader without a
    // geometry shader.
    VertexBufferObject impostorVBO;

    WindSettings windSettings;
    WindFieldSettings windFieldSettings;
    DistanceDensitySettings distanceDensitySettings;
    float windTimeSeconds = 0.0f;

    // Impostor pipeline resources (populated via setImpostorData).
    TrackedHandle<VkPipeline> impostorPipeline;
    TrackedHandle<VkPipelineLayout> impostorPipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> impostorDescSetLayout;
    TrackedHandle<VkDescriptorPool> impostorDescPool;
    TrackedHandle<VkDescriptorSet> impostorDescSet;

    // Impostor depth pipeline (shadow map depth-only pass).
    TrackedHandle<VkPipeline> impostorDepthPipeline;
    TrackedHandle<VkPipelineLayout> impostorDepthPipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> impostorDepthDescSetLayout;
    TrackedHandle<VkDescriptorPool> impostorDepthDescPool;
    TrackedHandle<VkDescriptorSet> impostorDepthDescSet;
    // Impostor EVSM shadow pipeline (color + depth write, uses VegetationRendererImpostorShadow.frag)
    TrackedHandle<VkPipeline> impostorShadowPipeline;
    TrackedHandle<VkPipelineLayout> impostorShadowPipelineLayout;

    float                 impostorDistance       = 0.0f;

    // Wind params UBO (set=2, binding=0) — updated once per frame.
    Buffer                windParamsBuffer;
    // M11 (perf report 22): last payload written, so the repeated per-pass
    // calls in one frame (depth + color + per-cascade shadow draws) skip the
    // memcpy when nothing changed. All callers run on the single async task
    // thread, so the cache needs no extra synchronization.
    WindParamsUBO         windParamsCache{};
    bool                  windParamsCacheValid = false;
    // M9-(3) (report 26): exact input snapshot for the updateWindParamsUBO
    // fast path (see .cpp). The payload is a pure function of (camera,
    // windSettings, densitySettings, impostorDistance); when every field
    // compares equal to the last call, the rebuild + memcmp + memcpy would
    // all be no-ops, so the call returns early. Field-wise == is used, never
    // memcmp (the settings structs carry bool padding); it errs toward an
    // extra rebuild on NaN, never toward a missed upload.
    glm::vec3             windParamsSnapCam{0.0f};
    WindSettings          windParamsSnapWind{};
    DistanceDensitySettings windParamsSnapDensity{};
    float                 windParamsSnapImpostor = 0.0f;
    bool                  windParamsSnapValid = false;
    TrackedHandle<VkDescriptorSetLayout> windParamsDescSetLayout;
    TrackedHandle<VkDescriptorSet> windParamsDescSet;
    void*                 windParamsMapped       = nullptr;

    // Shared wind-field UBO (set=0, binding=27) — persistently mapped,
    // packed by updateWindFieldUBO(). Same write-on-change caching as above.
    Buffer                windFieldBuffer;
    WindField          windFieldCache{};
    bool                  windFieldCacheValid = false;
    void*                 windFieldMapped      = nullptr;

    // ── CPU frustum culling (indirection via concatenated instance buffer) ────
    // Double-buffered (flicker fix): consolidation copies into the INACTIVE
    // pair while draws read the ACTIVE pair (selected by the last fence-
    // complete generation in pollConsolidation). A single buffer let the
    // graphics-queue copy overwrite the bytes the vegetation-queue draws were
    // reading — and since firstInstance offsets shift between generations
    // during streaming, every consolidation produced a frame of wrong
    // instances (the reported chunk-loading flicker).
    Buffer concatenatedInstanceBuffer;     // pair A (concat + heights + bake set)
    Buffer vegBakedHeightsBuffer;
    Buffer concatenatedInstanceBufferAlt;  // pair B
    Buffer vegBakedHeightsBufferAlt;
    bool concatActiveUsesAlt = false;      // which pair the draws must bind
    bool concatSubmittedUsesAlt = false;   // pair the in-flight submit wrote
    // See setConsolidationWait (cross-queue guard for the consolidation copy).
    VkSemaphore consolidationWaitSemaphore = VK_NULL_HANDLE;
    uint64_t consolidationWaitValue = 0;
    Buffer& activeConcatenatedBuffer() { return concatActiveUsesAlt ? concatenatedInstanceBufferAlt : concatenatedInstanceBuffer; }
    Buffer& activeBakedHeightsBuffer() { return concatActiveUsesAlt ? vegBakedHeightsBufferAlt : vegBakedHeightsBuffer; }
    Buffer& inactiveConcatenatedBuffer() { return concatActiveUsesAlt ? concatenatedInstanceBuffer : concatenatedInstanceBufferAlt; }
    Buffer& inactiveBakedHeightsBuffer() { return concatActiveUsesAlt ? vegBakedHeightsBuffer : vegBakedHeightsBufferAlt; }
    // Triple-buffered culling resources to prevent CPU/GPU race conditions
    // (same pattern as IndirectRenderer::MAX_CULL_FRAMES). All DEVICE_LOCAL
    // cull outputs (GPU-only): written by the merged cull dispatch, consumed
    // by the indirect-count draws. Zeroed by createBuffer and reset each frame
    // with vkCmdFillBuffer — never mapped on the host (no CPU reader exists).
    static constexpr uint32_t VEG_CULL_FRAMES = 3;
    std::array<Buffer, VEG_CULL_FRAMES> compactedCmdBuffers;
    std::array<Buffer, VEG_CULL_FRAMES> visibleCountBuffers;

    // ── Merged main-camera vegetation cull outputs ──
    // Per-frame GPU-compacted indirect command streams (billboards +
    // impostors) written in place by the solid IndirectRenderer's merged
    // IndirectRenderer.comp dispatch (see prepareCull / solidIR->setVegetationCullData).
    std::array<Buffer, VEG_CULL_FRAMES> impostorCompactBuffers;
    std::array<Buffer, VEG_CULL_FRAMES> impostorCountBuffers;
    uint32_t vegMainCompactCapacity = 0;
    // Creates (lazily) and grows the shared GPU chunk-info table used by the
    // merged cull and the cascade cull. Re-points descriptors on growth.
    void ensureChunkInfo(VulkanApp* app);

    // ── Cascade-aware culling for vegetation shadows ──
    // GPU-side cull (veg_cascade_cull.comp): per-cascade compact + count
    // buffers for blades (indexCount=36) and impostors (indexCount=6).
    struct VegCascadeCullFrame {
        std::array<Buffer, 3> compactBuffers;        // billboard draw commands
        std::array<Buffer, 3> countBuffers;          // billboard counts (GPU atomics)
        std::array<Buffer, 3> impostorCompactBuffers; // impostor draw commands
        std::array<Buffer, 3> impostorCountBuffers;   // impostor counts (GPU atomics)
        VkDescriptorSet descSet = VK_NULL_HANDLE;
    };
    std::array<VegCascadeCullFrame, VEG_CULL_FRAMES> vegCascadeCullFrames;
    bool vegCascadeCullInited = false;
    uint32_t vegCascadeCompactCapacity = 0;
    void initCascadeCull(VulkanApp* app);
    // Writes the GPU chunk table (aabbMin/aabbMax/instanceCount/firstInstance
    // triples) via memcpy into the host-visible chunk info buffer. Iteration
    // order MUST match consolidateChunks' concatenated-instance copy order so
    // firstInstance offsets point at the right instance ranges.
    void writeVegChunkInfo();

    // Shared GPU-side chunk table for the veg cascade cull. The per-cascade
    // billboard/impostor command + count buffers (vegCascadeCullFrames) are bound
    // into the SOLID IndirectRenderer's merged IndirectRenderer.comp descriptor set (bindings
    // 24..36) via solidIR->setVegCascadeData; the old veg_cascade_cull.comp pipeline
    // is retired.
    Buffer vegChunkInfoBuffer;
    void* vegChunkInfoMapped = nullptr;

    uint32_t vegNumChunks = 0;             // number of chunks in the consolidated metadata
    uint32_t vegChunkInfoCapacity = 0;     // current chunk-info table capacity (grows as needed)
    uint32_t vegPreallocatedChunks = 0;    // worst-case chunk reservation from preallocate()
    VkDeviceSize vegPreallocatedInstances = 0; // worst-case instance reservation
    bool vegPreallocated = false;          // true once preallocate() has run
    // Baked per-instance height scales (perf report 22 C2/H4): one float per
    // concatenated instance slot, written by the bake dispatch during
    // consolidateChunks, read as an instance-rate vertex attribute
    // (binding 2, ATTR_VEG_AUX) by every vegetation VS. Same firstInstance
    // indexing as the concatenated instance pair it parallels (double-
    // buffered together; the alt twin lives with the alt instance buffer).
    TrackedHandle<VkPipeline> vegBakePipeline;
    TrackedHandle<VkPipelineLayout> vegBakePipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> vegBakeDescSetLayout;
    TrackedHandle<VkDescriptorPool> vegBakeDescPool;
    // One set per buffer pair (SAME layout): the bake dispatch must target the
    // INACTIVE pair, so each consolidation binds the set matching its copy
    // targets.
    TrackedHandle<VkDescriptorSet> vegBakeDescSet;
    TrackedHandle<VkDescriptorSet> vegBakeDescSetAlt;
    uint32_t vegCullFrameIndex = 0;        // auto-cycling frame index for triple buffering
    uint32_t vegCullCurrentSlot = 0;       // slot selected for current frame's cull + draws
    bool vegConsolidationDirty = true;     // rebuild concatenated buffer + metadata
    // M11 (perf report 22): the GPU chunk table mirrors the consolidated set,
    // so it is rewritten only when consolidation runs (set there, cleared by
    // writeVegChunkInfo). Keeps steady-state frames free of the table memcpy.
    bool vegChunkInfoDirty = true;         // rewrite vegChunkInfoBuffer

    // ── Merged-cull integration ──
    // The SOLID IndirectRenderer whose IndirectRenderer.comp dispatch also emits veg
    // blade/impostor commands. VegetationRenderer owns the output buffers and
    // the per-chunk veg metadata; it hands both to solidIR each frame and mirrors
    // its cull frame so the draw reads the slot the merged dispatch wrote.
    IndirectRenderer* solidIR = nullptr;
    // Per-solid-mesh vegetation metadata, keyed by the solid mesh id
    // (static_cast<uint32_t>(chunk NodeID)). value = vec4(instanceCount,
    // firstInstance, 0, 0). Fed to solidIR via setVegetationChunkInfo().
    std::unordered_map<uint32_t, glm::vec4> vegChunkInfoMap;
    // Cull frame to use for the MAIN-pass veg draws (mirrors solidIR's frame so
    // the draw reads the slot the merged dispatch wrote). Falls back to the local
    // counter for the shadow cascade path (which keeps its own dispatch).
    uint32_t vegFrame() const;

    // Per-frame scratch for read-barrier recording (recordReadBarriers). Might
    // be called from the shadow worker (ShadowRenderer) and the vegetation
    // worker (MyApp veg block), so the scratch lives in a thread-local inside
    // the function and the version gate below is atomic (see .cpp).
    // M9-(1) (report 26): chunk-set version. Bumped (noteChunkBuffersChanged)
    // after every chunkBuffers mutation (publish, destroy, clear); the barrier
    // batch is re-recorded only when the version moved since the last record,
    // not once per frame. Store-after-record guarantees no missed barrier.
    std::atomic<uint64_t> chunkBufferVersion{0};
    std::atomic<uint64_t> readBarrierRecordedVersion{UINT64_MAX};
    void noteChunkBuffersChanged() { chunkBufferVersion.fetch_add(1, std::memory_order_release); }

    // Pipelined consolidation: deferred callback handles fence lifecycle.
    // (TrackedHandle is a non-owning wrapper: assigning the manager-owned
    // fence handle here never double-frees; the handle is only ever passed
    // to isFencePending(), never dereferenced.)
    TrackedHandle<VkFence> consolidationFence;
    bool consolidationPending = false;
    // Generational-consistency snapshot (no-vegetation fix): the cull and the
    // draws must agree on chunk order/counts with a COMPLETED copy, while
    // publishes keep arriving. Each consolidation submit snapshots the exact
    // (chunk, buffer, count, dstOffset, AABB) rows it copies; the copy loop
    // and writeVegChunkInfo both consume the snapshot, never the live map,
    // so later publishes cannot skew them apart. The fence poll publishes a
    // generation (completedSnapshot_ + vegNumChunks) independently of later
    // publishes, so generations always terminate: no version-match is needed
    // to make progress, and draws show the last-complete prefix while
    // streaming instead of nothing.
    struct ConsolEntry {
        NodeID chunkId{};
        VkBuffer buffer = VK_NULL_HANDLE;
        uint32_t count = 0;
        VkDeviceSize dstOffset = 0;
        glm::vec3 aabbMin{0.0f}, aabbMax{0.0f};
    };
    std::vector<ConsolEntry> submittedSnapshot_;  // order copied by the in-flight submit
    std::vector<ConsolEntry> completedSnapshot_;  // order of the last fence-complete generation
    uint64_t submittedVersion_ = UINT64_MAX;      // chunkBufferVersion at last submit (mismatch = first submit pending)
    uint64_t consolEpoch_ = 0;                    // bumped by clearAllInstances: abandons in-flight generations (poll drops them instead of publishing stale rows after a reset)
    uint64_t consolEpochAtSubmit_ = 0;            // consolEpoch_ captured at the last submit (poll publishes only on match)
    // M9-(2) (report 26): latest streaming-upload fence from
    // processPendingChunks (main thread). Polled with the thread-safe
    // isFencePending() (handle compare only, never dereferenced), so a
    // destroyed/recycled handle resolves in the safe direction. All streaming
    // submits go to the same graphics queue in order, so the latest fence
    // covers every earlier one.
    VkFence streamingUploadFence = VK_NULL_HANDLE;
    // M9-(2) (report 26): serializes concurrent consolidateChunks calls from
    // the cull worker (prepareCull) and the shadow path
    // (prepareCullCascades). Short critical section only.
    std::mutex consolidationMutex;
    // H7 (report 26): worker-side instance-pass elision state for
    // beginVegPass (M12, report 27: single unified arbiter — the MyApp
    // duplicate is removed). vegInstancePrev packs the previous inputs:
    // 2 = uninitialized (first frame always runs full so target layouts +
    // clears are established before any skip), otherwise bit 0 = prev
    // wantDraw, bit 1 = prev hasContent. A change in EITHER bit is a
    // transition and forces a full frame.
    std::atomic<uint32_t> vegInstancePrev{2};
    std::atomic<uint32_t> vegInstanceWarmup{0};

    // Batched async chunk upload: one fence, deferred publish.
    // M12-(1) (report 26): the batch shares ONE staging buffer sized once
    // from the batch total (see processPendingChunks); each entry carries its
    // source offset into it. Same device bytes as the old per-chunk acquires.
    struct PendingBatchCopy {
        Buffer instBuf;
        VkDeviceSize bufSize;
        VkDeviceSize stagingOffset; // src offset into the batch staging buffer
        NodeID chunkId;
        uint32_t instanceCount;
        glm::vec3 aabbMin, aabbMax, center;
    };
    std::vector<PendingBatchCopy> pendingBatch;
    // ── Persistent staging pool (perf report 22 M10) ──────────────────────
    // Streaming used to create + destroy one host-visible staging buffer per
    // chunk (up to 10/frame): two VMA allocations (which take a global lock)
    // plus two frees per chunk. The pool hands out buffers by best-fit
    // capacity and takes them back when the batch's copy fence signals, so
    // steady-state streaming performs ZERO allocations. Entries are never
    // erased (indices stay valid for fence callbacks); the pool settles at
    // the peak concurrent need (≈ frames-in-flight × chunks-per-frame) and
    // is destroyed in destroyCulling.
    struct StagingPoolEntry {
        Buffer buffer;
        VkDeviceSize capacity = 0;
        bool inUse = false;
    };
    std::vector<StagingPoolEntry> stagingPool;
    // Acquire a host-visible TRANSFER_SRC buffer with capacity >= size
    // (best-fit; creates a new pool entry when nothing fits). Marks inUse.
    // Returns the pool index via outIndex for the deferred release.
    Buffer acquireStagingBuffer(VulkanApp* app, VkDeviceSize size, size_t& outIndex);
    // Frame-thread scratch reused by processPendingChunks for per-chunk
    // instance data (clear + reserve avoids reallocating per chunk).
    std::vector<float> instanceGenScratch;

    // ── GPU vegetation placement (compute) ────────────────────────────────
    // Replaces the per-instance CPU loop in processPendingChunks: the chunk's
    // tessellated mesh (positions/normals/brushes) plus the expanded grass
    // triangle slots are uploaded once and a compute dispatch writes the
    // 32-byte instance records DIRECTLY into the chunk's device-local buffer.
    // Only the instance count is read back (deferred, after the dispatch
    // fence), so the instance data never round-trips through the CPU.
    bool useGpuPlacement = true;
    // Upper bound on in-flight placement descriptor sets per batch (one set
    // per chunk dispatch, freed once its fence signals).
    static constexpr uint32_t kMaxVegPlacementSets = 256;
    TrackedHandle<VkPipeline> vegPlacementPipeline;
    TrackedHandle<VkPipelineLayout> vegPlacementPipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> vegPlacementDescSetLayout;
    TrackedHandle<VkDescriptorPool> vegPlacementDescPool;
    void ensurePlacementPipeline(VulkanApp* app);
    void processPendingChunksGPU(VulkanApp* app, uint32_t maxChunks);

    // ── Own offscreen color + depth targets (decoupled from solid pass) ──
    std::array<VkImage, VEG_FRAMES> vegColorImages = {};
    std::array<VmaAllocation, VEG_FRAMES> vegColorAllocations = {};
    std::array<VkDeviceMemory, VEG_FRAMES> vegColorMemories = {};
    std::array<VkImageView, VEG_FRAMES> vegColorImageViews = {};
    std::array<VkImageLayout, VEG_FRAMES> vegColorImageLayouts = {};
    std::array<VkImage, VEG_FRAMES> vegDepthImages = {};
    std::array<VmaAllocation, VEG_FRAMES> vegDepthAllocations = {};
    std::array<VkDeviceMemory, VEG_FRAMES> vegDepthMemories = {};
    std::array<VkImageView, VEG_FRAMES> vegDepthImageViews = {};
    std::array<VkImageLayout, VEG_FRAMES> vegDepthImageLayouts = {};
    uint32_t vegRenderWidth = 0, vegRenderHeight = 0;

    void destroyCulling();
    void issueVegetationDraws(VkCommandBuffer cmd, VkPipelineLayout activeLayout, VkShaderStageFlags pushConstantStages, const WindPushConstants& pc);
    void issueImpostorDraws(VkCommandBuffer cmd, VkPipelineLayout activeLayout, VkShaderStageFlags pushConstantStages, const WindPushConstants& pc);
    WindPushConstants buildWindPushConstants() const;
};
