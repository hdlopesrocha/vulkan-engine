#pragma once
#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../TextureArrayManager.hpp"
#include "../EditableTexture.hpp"
#include "../../math/Vertex.hpp"
#include "../../math/Geometry.hpp"
#include "DebugCubeRenderer.hpp"
#include "../../utils/BillboardManager.hpp"
#include "../VertexBufferObject.hpp"
#include "../../utils/Scene.hpp" // for NodeID
#include "../ubo/VegetationUBO.hpp"
#include "../ubo/FireUBO.hpp"
#include <vector>
#include <deque>
#include <unordered_map>
#include <array>
#include <mutex>
#include <glm/glm.hpp>
#include <glm/gtc/round.hpp>
#include "CommandBufferState.hpp"

class IndirectRenderer; // merged-cull integration (forward decl)

// Per-chunk vegetation instance buffer and renderer
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

    struct DistanceDensitySettings {
        bool enabled = true;
        float fullDensityDistance = 512.0f;
        float minDensityDistance = 4096.0f;
        float minDensityFactor = 0.10f;
    };

    // Animated fire billboards. Fire instances share the 6-plane crossed
    // billboard mesh with vegetation but are shaded procedurally (no atlas
    // fetch). Instances whose billboard index equals kFireBillboardIndex are
    // rendered as fire. Fire instances are created from brush-4 (lava)
    // terrain triangles (see generateForChunk/processPendingChunks) with
    // probability fireDensity.
    static constexpr uint32_t kFireBillboardIndex = 3;
    static constexpr int kFireTerrainBrushIndex = 4; // lava layer

    struct FireSettings {
        bool enabled = true;
        float size = 32.0f;          // fire quad width in world units
        float heightScale = 1.4f;   // vertical stretch of the flame quad
        float speed = 2.0f;         // animation speed multiplier
        float intensity = 1.2f;     // overall brightness multiplier
        float flicker = 0.5f;       // temporal flicker amount (0..1)
        float noiseScale = 2.5f;    // flame noise frequency
        float turbulence = 0.6f;    // sideways flame distortion
        float riseSpeed = 1.5f;     // upward scroll speed of the flames
        float windInfluence = 0.35f;// how much wind bends the flames (0..1)
        float alpha = 1.0f;         // global flame opacity multiplier
        float emissive = 1.2f;      // emissive boost (fire glows)
        float smoke = 0.25f;        // smoke veil above the flame (0..1)
        float density = 0.06f;      // fraction of brush-4 slots that become fire
        glm::vec3 innerColor = glm::vec3(1.0f, 0.95f, 0.60f); // hot core
        glm::vec3 midColor   = glm::vec3(1.0f, 0.45f, 0.10f); // mid flame
        glm::vec3 outerColor = glm::vec3(0.60f, 0.05f, 0.00f); // flame edge
    };

    float billboardScale = 10.0f;
    uint32_t billboardCount = 3; // biomes: 0=foliage, 1=grass, 2=wild (40% of instances are empty sentinel)
    explicit VegetationRenderer();
    ~VegetationRenderer();

    void setTextureArrayManager(TextureArrayManager* mgr, VulkanApp* app);
    void setBillboardArrayTextures(VkImageView albedoView, VkImageView normalView, VkImageView opacityView, VkSampler sampler, VulkanApp* app);
    // Register the SOLID IndirectRenderer whose merged indirect.comp dispatch now
    // also emits the billboard/impostor commands. VegetationRenderer supplies its
    // per-frame output buffers + per-chunk veg metadata to that renderer.
    void setSolidIndirectRenderer(IndirectRenderer* ir) { solidIR = ir; }
    void onTextureArraysReallocated(VulkanApp* app);
    void init();
    void cleanup(VulkanApp* app) override;
    void init(VulkanApp* app);
    // CPU-side instance generation — avoids GPUVM faults on RADV iGPUs where
    // the Texture Cache/Pipe cannot read from device-local or host-visible
    // storage buffers.  Enqueues the chunk and processes up to maxPerFrame
    // chunks each frame via processPendingChunks().  With neither grass nor
    // fire triangles the chunk's previous instance data is cleared instead.
    void generateChunkInstancesCPU(NodeID chunkId,
                                   const std::vector<glm::vec3>& positions,
                                   const std::vector<glm::vec3>& normals,
                                   const std::vector<uint32_t>& grassIndices,
                                   const std::vector<uint32_t>& fireIndices,
                                   const glm::vec3& chunkCenter,
                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                   uint32_t seed = 1337);
    // CPU-side per-chunk vegetation generation (moved from SceneRenderer).
    // Samples grass-flagged (brush 3) triangles from the chunk's tessellated
    // geometry, plus fire-flagged (brush 4, kFireTerrainBrushIndex) triangles
    // for animated fire billboards. Builds area-weighted virtual triangle
    // slots with unbiased stochastic rounding (area-proportional density
    // without bias), shuffles the slots per chunk (so reducing the indirect
    // instanceCount keeps a random spatial subset), then hands the result to
    // generateChunkInstancesCPU. With neither grass nor fire triangles the
    // chunk's previous instance data is cleared instead.
    void generateForChunk(VulkanApp* app, NodeID nid, const Geometry& geom);
    // Drain up to maxChunks from the pending queue.  Call every frame from
    // draw() so chunks trickle in at a controlled rate.
    void processPendingChunks(uint32_t maxChunks);
    // Number of chunks still waiting in the queue.
    size_t pendingChunkCount() const;
    void clearAllInstances();

    // (The combined render() entry point was removed with the dead code sweep,
    // perf report 22 M9: all callers use drawDepth/drawColor directly.)
    // Deferred depth test: draw vegetation + impostor depth only (no color)
    void drawDepth(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::vec3& cameraPos);
    // Deferred depth test: draw vegetation + impostor color only (LESS_OR_EQUAL, no depth write)
    void drawColor(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::vec3& cameraPos);
    void recordReadBarriers(VkCommandBuffer& commandBuffer);

    // ── Own offscreen framebuffer (decoupled from the solid pass) ──
    // Vegetation is rendered to its own color+depth images so it can be drawn on a
    // parallel async command buffer (mirrors the water offscreen targets). The
    // solid pass no longer shares its depth with vegetation; occlusion against
    // solid geometry is resolved at composite time (postprocess.frag) by testing
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
    DistanceDensitySettings& getDistanceDensitySettings() { return distanceDensitySettings; }
    const DistanceDensitySettings& getDistanceDensitySettings() const { return distanceDensitySettings; }
    FireSettings& getFireSettings() { return fireSettings; }
    const FireSettings& getFireSettings() const { return fireSettings; }
    void setWindTime(float timeSeconds) { windTimeSeconds = timeSeconds; }

    // Impostor rendering.  Call after init() once impostor views have been captured.
    // albedoArray60 and normalArray60 must be VkImageView covering 80 layers
    // (4 billboard types × 20 Fibonacci views).
    // depthArray60 is the captured device Z array (R32_SFLOAT, 80 layers) for depth reprojection.
    // captureInvVPBuf is a storage buffer containing per-layer inverse VP matrices.
    void setImpostorData(VulkanApp* app,                         VkImageView albedoArray60,
                         VkImageView normalArray60,
                         VkSampler sampler,
                         VkImageView depthArray60 = VK_NULL_HANDLE,
                         VkBuffer   captureInvVPBuf = VK_NULL_HANDLE);

    // Distance beyond which vegetation instances are replaced by impostor quads.
    // Set to 0 (default) to disable impostor rendering entirely.
    void setImpostorDistance(float dist) { impostorDistance = dist; }

    // Build the concatenated instance buffer and per-chunk metadata for GPU
    // frustum culling. Must be called once after all chunks are generated.
    // Uses a temporary command buffer (synchronous, one-time cost).
    void consolidateChunks(VulkanApp* app);

    // Per-instance payload: pos vec4 (xyz = world pos, w = billboardIndex +
    // rotFrac) followed by a normal vec4 (xyz = surface normal, w unused).
    // One 32-byte stride shared by the per-chunk buffers, the concatenated
    // instance buffer, and every vertex-input declaration (binding 1).
    static constexpr VkDeviceSize kInstanceStride = sizeof(glm::vec4) * 2;
    // Worst-case vegetation capacities, sized ONCE at startup so no runtime
    // vmaCreateBuffer calls occur after the first frame. 4096 matches the
    // solid slotted-mode chunk ceiling (SceneRenderer: every veg chunk keys
    // off a solid chunk id); per-chunk instances are bounded by the chunk
    // mesh size (1 instance per grass triangle, ≤ ~2k tris for the 512 KB
    // per-chunk vertex budget).
    static constexpr uint32_t kMaxVegChunks = 4096;
    static constexpr uint32_t kMaxVegInstancesPerChunk = 2048;
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
    // Update the fire params UBO (set=2, binding=1) with current fire settings.
    // Must be called before any draw that can render fire. Skips the memcpy
    // when the payload is unchanged (same write-on-change pattern as wind).
    void updateFireParamsUBO();

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
    // only: setBillboardArrayTextures / onTextureArraysReallocated — never
    // called from the render loop, so steady state issues 0 updates).
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

    // Pending CPU-generation queue — chunks are enqueued by the scene loader
    // and drained 10-per-frame by draw().
    struct PendingChunk {
        NodeID chunkId;
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals; // per-vertex normals, parallel to positions
        std::vector<uint32_t> grassIndices;
        std::vector<uint32_t> fireIndices; // brush-4 virtual slots → fire billboards
        glm::vec3 chunkCenter;
        uint32_t instancesPerTriangle;
        uint32_t seed;
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
    DistanceDensitySettings distanceDensitySettings;
    FireSettings fireSettings;
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
    // Impostor EVSM shadow pipeline (color + depth write, uses impostors_shadow.frag)
    TrackedHandle<VkPipeline> impostorShadowPipeline;
    TrackedHandle<VkPipelineLayout> impostorShadowPipelineLayout;

    float                 impostorDistance       = 0.0f;

    // Wind params UBO (set=2, binding=0) — updated once per frame.
    Buffer                windParamsBuffer;
    // Fire params UBO (set=2, binding=1) — updated when fire settings change.
    Buffer                fireParamsBuffer;
    // M11 (perf report 22): last payload written, so the repeated per-pass
    // calls in one frame (depth + color + per-cascade shadow draws) skip the
    // memcpy when nothing changed. All callers run on the single async task
    // thread, so the cache needs no extra synchronization.
    WindParamsUBO         windParamsCache{};
    bool                  windParamsCacheValid = false;
    FireParamsUBO         fireParamsCache{};
    bool                  fireParamsCacheValid = false;
    TrackedHandle<VkDescriptorSetLayout> windParamsDescSetLayout;
    TrackedHandle<VkDescriptorSet> windParamsDescSet;
    void*                 windParamsMapped       = nullptr;
    void*                 fireParamsMapped       = nullptr;

    // ── CPU frustum culling (indirection via concatenated instance buffer) ────
    Buffer concatenatedInstanceBuffer;  // all instances concatenated (kInstanceStride per element)
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
    // indirect.comp dispatch (see prepareCull / solidIR->setVegetationCullData).
    std::array<Buffer, VEG_CULL_FRAMES> impostorCompactBuffers;
    std::array<Buffer, VEG_CULL_FRAMES> impostorCountBuffers;
    uint32_t vegMainCompactCapacity = 0;
    // Creates (lazily) and grows the shared GPU chunk-info table used by the
    // merged cull and the cascade cull. Re-points descriptors on growth.
    void ensureChunkInfo(VulkanApp* app);

    // ── Cascade-aware culling for vegetation shadows ──
    // GPU-side cull (veg_cascade_cull.comp): per-cascade compact + count
    // buffers for billboards (indexCount=36) and impostors (indexCount=6).
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
    // into the SOLID IndirectRenderer's merged indirect.comp descriptor set (bindings
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
    // indexing as concatenatedInstanceBuffer; the two stay in lockstep
    // always (baked together, never independently modified).
    Buffer vegBakedHeightsBuffer;
    TrackedHandle<VkPipeline> vegBakePipeline;
    TrackedHandle<VkPipelineLayout> vegBakePipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> vegBakeDescSetLayout;
    TrackedHandle<VkDescriptorPool> vegBakeDescPool;
    TrackedHandle<VkDescriptorSet> vegBakeDescSet;
    uint32_t vegCullFrameIndex = 0;        // auto-cycling frame index for triple buffering
    uint32_t vegCullCurrentSlot = 0;       // slot selected for current frame's cull + draws
    bool vegConsolidationDirty = true;     // rebuild concatenated buffer + metadata
    // M11 (perf report 22): the GPU chunk table mirrors the consolidated set,
    // so it is rewritten only when consolidation runs (set there, cleared by
    // writeVegChunkInfo). Keeps steady-state frames free of the table memcpy.
    bool vegChunkInfoDirty = true;         // rewrite vegChunkInfoBuffer

    // ── Merged-cull integration ──
    // The SOLID IndirectRenderer whose indirect.comp dispatch also emits veg
    // billboard/impostor commands. VegetationRenderer owns the output buffers and
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

    // Per-frame scratch for read-barrier recording (recordReadBarriers). Runs
    // on the main frame thread only (SceneRenderer::shadowPass / preRenderPass)
    // — plain members are safe; clear() + reserve() reuse capacity across frames.
    std::vector<VkBufferMemoryBarrier2> readBarrierScratch;

    // Pipelined consolidation: deferred callback handles fence lifecycle
    TrackedHandle<VkFence> consolidationFence;
    bool consolidationPending = false;

    // Batched async chunk upload: one fence, deferred publish
    struct PendingBatchCopy {
        Buffer stagingInst, instBuf;
        VkDeviceSize bufSize;
        size_t stagingPoolIndex; // slot `stagingInst` came from (release on fence)
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
    WindPushConstants buildWindPushConstants(bool fireOnly = false) const;
};
