#include "SolidRenderer.hpp"
#include "VegetationRenderer.hpp"
#include "WaterRenderer.hpp"
#include "PostProcessRenderer.hpp"
#include "SkyRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "DebugCubeRenderer.hpp"
#include "DebugSDFRenderer.hpp"
#include "WireframeRenderer.hpp"
#include "WaterBackFaceRenderer.hpp"
#include "BrushBackFaceRenderer.hpp"
#include "Solid360Renderer.hpp"
#pragma once

// Forward declarations for change handler types
class Octree;
class World;

#include <vulkan/vulkan.h>
#include "../VulkanApp.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include "../TextureArrayManager.hpp"
#include "../MaterialManager.hpp"
#include "../ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include "../../math/Vertex.hpp"
#include "../../math/BoundingCubeHasher.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <deque>
#include <vector>
#include "../../space/Model3DVersion.hpp"
#include "../../space/ThreadPool.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "IndirectRenderer.hpp"
#include "../streaming/UploadManager.hpp"   // TerrainStreamer: async streaming orchestration
#include "ShadowRenderer.hpp"
#include "WaterRenderer.hpp"
#include "../../world/World.hpp"

#include "../ubo/PassUBO.hpp"
#include "CommandBufferState.hpp"

class SceneRenderer {
    void addDebugCubeForGeometry(Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom);
public:
    // UBOs for main, shadow, and water passes
    PassUBO<UniformObject> mainPassUBO;
    PassUBO<UniformObject> shadowPassUBO;
    PassUBO<WaterParamsGPU> waterPassUBO;

    // Main uniform buffers (one per frame-in-flight)
    std::vector<Buffer> mainUniformBuffers;

    // Per-frame staging buffers for UBO uploads via vkCmdCopyBuffer
    // (replaces vkCmdUpdateBuffer to avoid implicit FULL_QUEUE barrier).
    std::vector<Buffer> uboStagingBuffers;
    
    // Materials SSBO
    Buffer materialsBuffer;
    MaterialManager* materialManagerPtr = nullptr;

    // Water params UBO (binding 7) — stored for descriptor rebinding
    Buffer waterParamsBuffer_;

    // Water render time UBO (binding 10)
    Buffer waterRenderUBOBuffer_;

    // Shadow-specific descriptor sets (one per frame). Each mirrors the main
    // descriptor set but binding 4 points to a dummy depth view to avoid
    // layout-mismatch validation errors while writing the real shadow maps.
    std::vector<VkDescriptorSet> shadowDescriptorSets;

    std::unique_ptr<ShadowRenderer> shadowMapper;
    std::unique_ptr<WaterRenderer> mainLiquidRenderer;
    std::unique_ptr<PostProcessRenderer> postProcessRenderer;
    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<SolidRenderer> mainSolidRenderer;
    std::unique_ptr<VegetationRenderer> vegetationRenderer;
    // Brush offscreen render targets (color + front depth), matching MAX_FRAMES_IN_FLIGHT
    static constexpr uint32_t BRUSH_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, BRUSH_FRAMES> brushColorImages = {};
    std::array<VmaAllocation, BRUSH_FRAMES> brushColorAllocations = {};
    std::array<VkImageView, BRUSH_FRAMES> brushColorImageViews = {};
    std::array<VkImageLayout, BRUSH_FRAMES> brushColorLayouts = {};
    std::array<VkImage, BRUSH_FRAMES> brushDepthImages = {};
    std::array<VmaAllocation, BRUSH_FRAMES> brushDepthAllocations = {};
    std::array<VkImageView, BRUSH_FRAMES> brushDepthImageViews = {};
    std::array<VkImageLayout, BRUSH_FRAMES> brushDepthLayouts = {};
    VkImageView getBrushColorView(uint32_t i) const { return brushColorImageViews[i % BRUSH_FRAMES]; }
    VkImage getBrushColorImage(uint32_t i) const { return brushColorImages[i % BRUSH_FRAMES]; }
    VkImageView getBrushDepthView(uint32_t i) const { return brushDepthImageViews[i % BRUSH_FRAMES]; }
    VkImage getBrushDepthImage(uint32_t i) const { return brushDepthImages[i % BRUSH_FRAMES]; }
    // Per-frame descriptor sets for brush depth textures (set=1)
    std::array<VkDescriptorSet, BRUSH_FRAMES> brushDepthDescriptorSets = {};
    VkDescriptorSet getBrushDepthDescriptorSet(uint32_t frameIndex) const {
        return brushDepthDescriptorSets[frameIndex % BRUSH_FRAMES];
    }

    void createBrushRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyBrushRenderTargets(VulkanApp* app);

    // Scene-owned sub-renderers for water (moved from WaterRenderer)
    std::unique_ptr<WaterBackFaceRenderer> backFaceRenderer;
    std::unique_ptr<BrushBackFaceRenderer> brushBackFaceRenderer;
    std::unique_ptr<Solid360Renderer> solid360Renderer;
    std::unique_ptr<DebugCubeRenderer> debugCubeRenderer;
    std::unique_ptr<DebugCubeRenderer> boundingBoxRenderer;
    std::unique_ptr<DebugSDFRenderer> debugSDFRenderer;
    std::unique_ptr<WireframeRenderer> solidWireframe;
    std::unique_ptr<WireframeRenderer> waterWireframe;
    // Sky settings owned by this renderer
    std::unique_ptr<SkySettings> skySettings;
    SkySettings& getSkySettings() { return *skySettings; }

    SceneRenderer();
    ~SceneRenderer();

    // Cleanup and resource destruction (accepts app for Vulkan operations)
    void cleanup(VulkanApp* app);

    // Drain generation pools.  Must be called after all Octree pools are
    // stopped (their workers may still be enqueuing tasks to these pools).
    void stopGenPools();
   
    // Pending change queues (thread-safe)
    struct PendingNode {
        Layer layer;
        OctreeNodeData node;
    };

    // Mutex protecting all chunk maps (solid, transparent, brush) and mesh operations
    std::recursive_mutex mainSolidChunksMutex;
    std::recursive_mutex mainLiquidChunksMutex;
    std::recursive_mutex brushSolidChunksMutex;
    std::recursive_mutex brushTransparentChunksMutex;
    std::recursive_mutex pendingOldBrushSolidChunksMutex;
    std::recursive_mutex pendingOldBrushLiquidChunksMutex;
    // Debug SDF cube markers are a separate map (not a chunk registry) with
    // their own guard. Written from the change handlers (build lambdas),
    // read/cloned on the render thread draw path.
    std::recursive_mutex debugSDFCubesMutex;

    // ── Legacy chunk tracking (append-based, full rebuild) ──
    // Track model ids for transparent/water meshes so we can remove them if erased/updated
    std::unordered_map<NodeID, Model3DVersion> mainLiquidChunks;
    std::unordered_map<NodeID, Model3DVersion> mainSolidChunks;

    // Brush scene chunk tracking (separate from main scene)
    std::unordered_map<NodeID, Model3DVersion> brushSolidChunks;
    std::unordered_map<NodeID, Model3DVersion> brushTransparentChunks;
    std::unordered_map<NodeID, Model3DVersion> pendingOldBrushSolidChunks;
    std::unordered_map<NodeID, Model3DVersion> pendingOldBrushLiquidChunks;

    // Slots whose chunks were erased but may be replaced (same NodeID, new
    // version). For solid/water the octree node is reused on edit, so NodeID
    // stays stable — addMeshSlotted finds the existing entry and republishes
    // it in place (new packed span per level, old span freed after the
    // replacement upload completes). The old slot must survive until the new
    // upload completes to avoid a 1-frame hole. Entries are matched by NodeID
    // and aged out after MAX_FRAMES_IN_FLIGHT frames past their birth frame.
    struct PendingDeleteEntry {
        uint32_t slotIndex = UINT32_MAX;
        uint32_t birthFrame = 0;
    };
    std::unordered_map<NodeID, PendingDeleteEntry> pendingDeleteSolidSlots;
    std::unordered_map<NodeID, PendingDeleteEntry> pendingDeleteWaterSlots;

    // ── World reference (separates world logic from rendering) ──
    // The World owns chunks, octrees, and the ChunkManager state machine.
    // The SceneRenderer only reads chunk state and produces/consumes
    // per-level meshes. Call setWorld() before scene loading.
    void setWorld(World* world) { world_ = world; }
    World* world() { return world_; }
    const World* world() const { return world_; }

    // Cached env-var flags (read once in init(), never per frame)
    bool envDisableWaterGeom = false;


    // Separate IndirectRenderer for brush solid meshes (so the brush backface
    // buffer only renders brush geometry, not all scene solids).
    IndirectRenderer brushSolidIndirectRenderer;
    // Separate IndirectRenderer for brush transparent/liquid meshes: brush
    // water must NOT share the main water IR — the brush preview is rebuilt and
    // republished independently, so its geometry belongs in a dedicated pool.
    IndirectRenderer brushLiquidIndirectRenderer;

    // Debug cubes for nodes (populated by change handlers after geometry generation)
    std::unordered_map<NodeID, DebugCubeRenderer::CubeWithColor> nodeDebugCubes;
    void addDebugCubeForNode(NodeID id, const DebugCubeRenderer::CubeWithColor& cube) { nodeDebugCubes[id] = cube; }
    void removeDebugCubeForNode(NodeID id) { nodeDebugCubes.erase(id); }
    std::vector<DebugCubeRenderer::CubeWithColor> getDebugNodeCubes() const {
        std::vector<DebugCubeRenderer::CubeWithColor> out;
        out.reserve(nodeDebugCubes.size());
        for (const auto &p : nodeDebugCubes) out.push_back(p.second);
        return out;
    }

    // SDF face debug cubes for leaf nodes under each chunk.
    std::unordered_map<NodeID, std::vector<DebugSDFRenderer::CubeSDF>> nodeDebugSDFCubes;
    void removeDebugSDFCubesForNode(NodeID id);
    void clearDebugSDFCubes();
    std::vector<DebugSDFRenderer::CubeSDF> getDebugSDFCubes();

    // Register/inspect opaque model versions (moved from SolidRenderer)
    size_t getRegisteredModelCount() const { return mainSolidChunks.size(); }

    // Remove all registered opaque meshes via IndirectRenderer and clear the map
    void removeAllRegisteredMeshes() {
        if (!mainSolidRenderer) return;
        mainSolidRenderer->getIndirectRenderer().removeAllMeshes();
        mainSolidChunks.clear();
    }

    // Remove all registered transparent/water meshes and clear the map
    void removeAllTransparentMeshes() {
        if (!mainLiquidRenderer) return;
        mainLiquidRenderer->getIndirectRenderer().removeAllMeshes();
        mainLiquidChunks.clear();
    }

    void shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, Buffer &mainUniformBuffer, const UniformObject &uboStatic, bool shadowsEnabled, bool renderSolid, bool vegetationEnabled, bool shadowTessellationEnabled = false, float lodBias = 8.0f);
    void skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj);
    void mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, bool hasWater, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool renderSolid, bool wireframeEnabled, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent);
    // Draw wireframe overlay for solid geometry on top of existing solid render
    void drawSolidWireframeOverlay(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled);
    void waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, bool waterWireframeEnabled, float waterTime, VkImageView skyView);
    void init(VulkanApp* app_, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterParams>& waterParams);
    // Re-update main descriptor set when texture arrays are (re)allocated
    void updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager);
    // cleanup declared above (accepts VulkanApp*)

    // Generate vegetation instances for a given opaque node / chunk.
    // Extracted so both legacy and slotted mode paths share the same logic.
    void generateVegetationForNode(VulkanApp* app, NodeID nid, const Geometry& geom);

    using GeometryHandler = const std::function<void(Layer, NodeID, const Octree::LoDMesh&)>&;


    
    struct PendingMeshData {
        Layer          layer;
        NodeID         nid;
        Octree::LoDMesh lodMesh;
        OctreeNodeData nodeData;   // world cube of the source node (stable band center)
        bool           isBrush = false; // brush-scene entry (own IR + slot bookkeeping)
    };

    // Process nodes from a generic per-layer NodeID->OctreeNodeData map
    // Process nodes for a single Layer (nodeMap maps NodeID->OctreeNodeData)
    void processNodeLayer(
        Scene& scene, 
        Layer layer, 
        NodeID nid, 
        OctreeNodeData& nodeData, 
        GeometryHandler onGeometry, 
        float minSize, 
        ThreadPool* poolOverride = nullptr
    );

    // ── Slotted-mode chunk processing ────────────────────────────────────────
    // Initialize the slotted rendering pipeline for solid and water layers.
    // Creates the slot pools, GPU buffers, and switches the renderers to
    // stable-slot mode. Solid and water pools are sized independently (solid is
    // the dense layer, water is sparse). Must be called once on the main thread.
    void initSlottedMode(VulkanApp* app, uint32_t maxSolidChunks,
                         uint32_t maxWaterChunks,
                         uint32_t vertexBytesPerChunk = 1u << 20,
                         uint32_t indexBytesPerChunk  = 1u << 19);

    // Process a single chunk's mesh in slotted mode.
    // To be called from the change handler callback (on a worker thread).
    // Generates the mesh and queues it for GPU upload.
    // Returns true if the chunk was processed successfully.
    bool processChunkSlotted(Layer layer, NodeID nid, const OctreeNodeData& nd,
                             const Geometry& geom, uint32_t version);

    // Drain the swap queue and atomically swap in new RenderProxies.
    // Also retires old proxies via deferred destruction.
    // Call once per frame from processPendingMeshes.
    void processChunkSwapQueue(VulkanApp* app);

    // Runtime introspection helpers for UI/debug
    size_t getTransparentModelCount();

    // Query whether a model for the given node is already registered
    bool hasModelForNode(Layer layer, NodeID nid) const;

    // The solid/water AND brush space-change lambdas are constructed by the
    // app (main.cpp) — they need world state, chunk management and debug
    // markers. No make* handler factories remain here.

    // Remove all brush meshes from GPU and clear brush chunk maps
    void clearBrushMeshes();
    void stageOldBrushChunks();

    // Resize offscreen resources when the swapchain changes
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    // Rewrite brush depth texture descriptors (bindings 14, 15) for all per-frame
    // main descriptor sets. Must be called after brush render targets are
    // recreated (on resize or after rebuildBrushScene).
    void writeBrushDepthDescriptors(VulkanApp* app);

    // ── Parallel scene loading ─────────────────────────────────────────────────
    // Drains the shared pending mesh queue (main thread) into a caller-provided
    // batch, then processes it. Call once per frame from update() before
    // recording command buffers. Drains BOTH the main scene and brush scene
    // entries (one shared queue) with a single shared per-frame budget.
    void drainPendingMeshes(std::deque<PendingMeshData>& out, size_t maxCount);

    // Unify the given pending geometry batch into the GPU. One common path for
    // the main scene AND the brush scene: each entry routes itself to the right
    // IndirectRenderer and deferred-slot bookkeeping by (layer, isBrush), so
    // solid/water geometry is processed the same way as brush geometry.
    void processPendingMeshes(VulkanApp* app, glm::vec3 cameraPos, std::deque<PendingMeshData>& batch);

    // Async streaming orchestrator (parallel per-category pools, lock-free
    // queues, drop-in staging upload manager). Currently scaffolded: its
    // update()/frame-sync runs each frame; terrain/water/brush GPU copies still
    // flow through IndirectRenderer until that path is migrated to use it.
    streaming::TerrainStreamer streamer;

private:
    // Single publish core for a pending mesh batch — every stream behaves
    // identically. Publishes each generated geometry chunk AS RECEIVED: every
    // PendingMeshData is ONE self-contained mesh (no ladder structures). Each
    // chunk occupies a slot; the mesh lands in the slot row for its LoD level
    // and publishes its own draw entry. No per-chunk reassembly. Returns the
    // number of slots published.
    //
    // The core selects, per-entry, by (layer, isBrush) which renderers are.
    // (main solid / brush solid / shared water) and whether the chunk's build
    // state runs through the ChunkManager (main scene only).
    //
    //  takeOldSlot:      resolve and consume the old slot for a chunk, or
    //                    UINT32_MAX when none. Callers free it after the new
    //                    upload completes. Receives isBrush so each stream's
    //                    deferred slot source is the SAME callback.
    //  onChunkPublished: main-thread side effect once a chunk's slot is
    //                    published (the brush stream records its Model3DVersion
    //                    map). Receives isBrush.
    //  onFinestPublished: notified with the level-0 (finest) geometry of
    //                    opaque chunks so they can drive vegetation; brush
    //                    entries pass a noop.
    size_t publishPendingMeshes(
        VulkanApp* app,
        std::deque<PendingMeshData>& batch,
        IndirectRenderer& opaqueIR,
        IndirectRenderer& brushOpaqueIR,
        IndirectRenderer& waterIR,
        IndirectRenderer& brushWaterIR,
        const std::function<uint32_t(Layer layer, NodeID nid, bool isBrush)>& takeOldSlot,
        const std::function<void(Layer layer, NodeID nid, uint32_t slotIdx, uint32_t version, bool isBrush)>& onChunkPublished,
        const std::function<void(NodeID nid, const Geometry& geom, bool isBrush)>& onFinestPublished);

    // Age out main-stream pending-delete slots older than MAX_FRAMES_IN_FLIGHT
    // (genuine deletions with no replacement chunk).
    void ageOutPendingDeletes(uint32_t curFrame, IndirectRenderer& solidIR, IndirectRenderer& waterIR);

    // World reference (null until setWorld is called).
    // The World owns ChunkManager and all chunk state.
    World* world_ = nullptr;

    // Camera position from the last processPendingMeshes call; used by the
    // shadow pass to cull with the same camPos/lodBias as the main pass so
    // shadow draws use the identical per-chunk LoD selection.
    glm::vec3 lastCameraPos_ = glm::vec3(0.0f);

public:
    // Debug helper: publish SDF cube debug markers for a (re)built chunk.
    void updateDebugSDFCubesForChunk(NodeID nid, const OctreeNodeData& nd, const Octree& tree);

    // Thread-safe mesh queue fed by tessellation on the generation pools;
    // drained on the main thread by processPendingMeshes(). ONE shared queue
    // for every stream (main solid/water + brush solid/water): entries are
    // tagged with PendingMeshData::isBrush so the drain routes each entry to
    // the right IndirectRenderer and slot bookkeeping.
    mutable std::mutex                          pendingMeshMutex;
    std::unordered_map<NodeID, PendingMeshData> pendingMeshQueue;

    // Dedicated generation pools for solid and water so both layers tessellate
    // truly in parallel: neither waits for the other to finish, and neither
    // competes for the shared scene pool. Public so the app can hand them to
    // processNodeLayer when building its own solid/water space-change lambdas.
    ThreadPool mainSolidGenPool{std::max(2u, std::thread::hardware_concurrency() / 2)};
    ThreadPool mainWaterGenPool{std::max(2u, std::thread::hardware_concurrency() / 2)};
    ThreadPool brushSolidGenPool{std::max(2u, std::thread::hardware_concurrency() / 2)};
    ThreadPool brushWaterGenPool{std::max(2u, std::thread::hardware_concurrency() / 2)};

    CommandBufferState frameCmdState;
};

// ...existing code...
