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
    // chunks each frame via processPendingChunks().  With no grass triangles
    // the chunk's previous instance data is cleared instead.
    void generateChunkInstancesCPU(NodeID chunkId,
                                   const std::vector<glm::vec3>& positions,
                                   const std::vector<uint32_t>& grassIndices,
                                   const glm::vec3& chunkCenter,
                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                   uint32_t seed = 1337);
    // CPU-side per-chunk vegetation generation (moved from SceneRenderer).
    // Samples grass-flagged triangles from the chunk's tessellated geometry,
    // builds area-weighted virtual triangle slots with unbiased stochastic
    // rounding (area-proportional density without bias), shuffles the slots
    // per chunk (so reducing the indirect instanceCount keeps a random spatial
    // subset), then hands the result to generateChunkInstancesCPU. With no
    // grass triangles the chunk's previous instance data is cleared instead.
    void generateForChunk(VulkanApp* app, NodeID nid, const Geometry& geom);
    // Drain up to maxChunks from the pending queue.  Call every frame from
    // draw() so chunks trickle in at a controlled rate.
    void processPendingChunks(uint32_t maxChunks);
    // Number of chunks still waiting in the queue.
    size_t pendingChunkCount() const;
    void clearAllInstances();

    // Draw all visible vegetation chunks with GPU frustum culling.
    // If queryPool != VK_NULL_HANDLE, writes GPU timestamps:
    //   queryRealIndex .. queryRealIndex+1  = real billboard passes (depth prepass + shading)
    //   queryImpostorIndex .. queryImpostorIndex+1 = impostor passes (impostor depth + color)
    void render(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet,
              const glm::mat4& viewProj, const glm::vec3& cameraPos,
              VkQueryPool queryPool = VK_NULL_HANDLE,
              uint32_t queryRealIndex = 0,
              uint32_t queryImpostorIndex = 0);
    // Deferred depth test: draw vegetation + impostor depth only (no color)
    void drawDepth(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::mat4& viewProj, const glm::vec3& cameraPos);
    // Deferred depth test: draw vegetation + impostor color only (LESS_OR_EQUAL, no depth write)
    void drawColor(VulkanApp* app, VkCommandBuffer& commandBuffer, const glm::mat4& viewProj, const glm::vec3& cameraPos);
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
    
    // Draw vegetation to shadow map using light-space matrix in the bound UBO.
    // Camera position is used for distance-based LOD; viewProj is the camera's
    // view-projection for GPU frustum culling (matching solid shadow culling).
    void drawShadow(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet shadowDescriptorSet, const glm::mat4& viewProj, const glm::vec3& cameraPos);
    PFN_vkCmdDrawIndexedIndirectCountKHR cmdDrawIndexedIndirectCount = nullptr;

    // Stats helpers
    size_t getChunkCount() const { return chunkInstanceCounts.size(); }
    size_t getInstanceTotal() const;

    WindSettings& getWindSettings() { return windSettings; }
    const WindSettings& getWindSettings() const { return windSettings; }
    DistanceDensitySettings& getDistanceDensitySettings() { return distanceDensitySettings; }
    const DistanceDensitySettings& getDistanceDensitySettings() const { return distanceDensitySettings; }
    void setWindTime(float timeSeconds) { windTimeSeconds = timeSeconds; }
    float computeDensityFactor(float distanceToCamera) const;
    std::vector<DebugCubeRenderer::CubeWithColor> getDensityDebugCubes(const glm::vec3& cameraPos) const;
    float getAverageDensityFactor(const glm::vec3& cameraPos) const;

    // Impostor rendering.  Call after init() once impostor views have been captured.
    // albedoArray60 and normalArray60 must be VkImageView covering 60 layers
    // (3 billboard types × 20 Fibonacci views).
    // depthArray60 is the captured device Z array (R32_SFLOAT, 60 layers) for depth reprojection.
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
    bool isPreallocated() const { return vegPreallocated; }

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

    // Descriptor set allocated from the app's descriptor pool and re-created when the texture arrays are (re)allocated
    TrackedHandle<VkDescriptorSet> vegDescriptorSet;
    uint32_t vegDescriptorVersion = 0;
    bool ensureVegDescriptorSet(VulkanApp* app);
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
        std::vector<uint32_t> grassIndices;
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
    VkRenderPass storedSolidRenderPass = VK_NULL_HANDLE;

    // Wind params UBO (set=2, binding=0) — updated once per frame.
    Buffer                windParamsBuffer;
    TrackedHandle<VkDescriptorSetLayout> windParamsDescSetLayout;
    TrackedHandle<VkDescriptorSet> windParamsDescSet;
    void*                 windParamsMapped       = nullptr;

    // ── CPU frustum culling (indirection via concatenated instance buffer) ────
    Buffer concatenatedInstanceBuffer;  // all instances concatenated (vec4 per element)
    // Triple-buffered culling resources to prevent CPU/GPU race conditions
    // (same pattern as IndirectRenderer::MAX_CULL_FRAMES).
    static constexpr uint32_t VEG_CULL_FRAMES = 3;
    std::array<Buffer, VEG_CULL_FRAMES> compactedCmdBuffers;
    std::array<Buffer, VEG_CULL_FRAMES> visibleCountBuffers;
    mutable std::array<uint32_t*, VEG_CULL_FRAMES> visibleCountMapped = {nullptr, nullptr, nullptr};
    mutable std::array<VkDrawIndexedIndirectCommand*, VEG_CULL_FRAMES> compactedCmdMapped = {nullptr, nullptr, nullptr};

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
    uint32_t vegCullFrameIndex = 0;        // auto-cycling frame index for triple buffering
    uint32_t vegCullCurrentSlot = 0;       // slot selected for current frame's cull + draws
    bool vegConsolidationDirty = true;     // rebuild concatenated buffer + metadata

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
        NodeID chunkId;
        uint32_t instanceCount;
        glm::vec3 aabbMin, aabbMax, center;
    };
    std::vector<PendingBatchCopy> pendingBatch;
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
    WindPushConstants buildWindPushConstants(const glm::vec3& cameraPos) const;
};
