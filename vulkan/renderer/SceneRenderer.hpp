#pragma once

// Forward declarations for change handler types
class Octree;
class World;

#include <vulkan/vulkan.h>
#include "Renderer.hpp"
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
#include <array>
#include <memory>
#include <mutex>
#include <deque>
#include <vector>
#include "../../space/Model3DVersion.hpp"
#include "../../space/ThreadPool.hpp"
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
#include "BrushRenderer.hpp"
#include "Solid360Renderer.hpp"
#include "IndirectRenderer.hpp"
#include "../streaming/UploadManager.hpp"   // TerrainStreamer: async streaming orchestration
#include "../../world/World.hpp"

#include "CommandBufferState.hpp"

class SceneRenderer : public Renderer {
public:
    // Main uniform buffers (one per frame-in-flight)
    std::vector<Buffer> mainUniformBuffers;

    // Materials SSBO
    Buffer materialsBuffer;
    MaterialManager* materialManagerPtr = nullptr;

    // Water params UBO (binding 7) — stored for descriptor rebinding
    Buffer waterParamsBuffer_;
    // Byte size of waterParamsBuffer_ (vkGetDescriptorEXT forbids WHOLE_SIZE).
    VkDeviceSize waterParamsBufferSize_ = 0;

    // Shadow-specific descriptor sets (one per frame). Each mirrors the main
    // descriptor set but binding 4 points to a dummy depth view to avoid
    // layout-mismatch validation errors while writing the real shadow maps.
    std::vector<VkDescriptorSet> shadowDescriptorSets;

    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<ShadowRenderer> shadowMapper;
    std::unique_ptr<PostProcessRenderer> postProcessRenderer;
    std::unique_ptr<SolidRenderer> mainSolidRenderer;
    std::unique_ptr<WaterRenderer> mainLiquidRenderer;
    std::unique_ptr<VegetationRenderer> vegetationRenderer;
    std::unique_ptr<BrushRenderer> brushRenderer;
    std::unique_ptr<WaterBackFaceRenderer> backFaceRenderer;
    std::unique_ptr<Solid360Renderer> solid360Renderer;
    std::unique_ptr<DebugCubeRenderer> debugCubeRenderer;
    std::unique_ptr<DebugCubeRenderer> boundingBoxRenderer;
    std::unique_ptr<DebugSDFRenderer> debugSDFRenderer;
    std::unique_ptr<WireframeRenderer> waterWireframe;
    // Sky settings owned by this renderer
    std::unique_ptr<SkySettings> skySettings;
    SkySettings& getSkySettings() { return *skySettings; }

    SceneRenderer();
    ~SceneRenderer();

    // Cleanup and resource destruction (accepts app for Vulkan operations)
    void cleanup(VulkanApp* app) override;

    // Propagate the shared per-frame command state tracker to every renderer
    // that only records on the main thread. backFaceRenderer and the water
    // IndirectRenderer are deliberately excluded: the async back-face task
    // uses them on a separate thread and keeping cmdState=nullptr there
    // avoids a data race on frameCmdState.
    void setCmdState(CommandBufferState* state) override;

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

    // ── Chunk tracking ──
    // Track model ids for transparent/water meshes so we can remove them if erased/updated
    std::unordered_map<NodeID, Model3DVersion> mainLiquidChunks;
    std::unordered_map<NodeID, Model3DVersion> mainSolidChunks;

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

    void init(VulkanApp* app_, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterParams>& waterParams);
    // Re-update main descriptor set when texture arrays are (re)allocated
    void updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager);

    // ── Descriptor-buffer migration (Phase 1) ─────────────────────────────
    // Step 1: allocate 3 host-visible descriptor buffers (one per frame).
    // No-op (leaves descBuffers_.ready false) when !app->useDescriptorBuffer().
    void initDescriptorBuffers(VulkanApp* app);
    // Step 2/4: (re)write all static set-0 bindings into every frame's
    // descriptor buffer via DescriptorBuffer::writeBuffer/writeImage.
    // No-op when buffers are not ready. Sources that are not yet allocated
    // (e.g. cubemap before Solid360 init) are skipped.
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

    using GeometryHandler = const std::function<void(Layer, NodeID, const Octree::LoDMesh&)>&;


    
    struct PendingMeshData {
        Layer          layer;
        NodeID         nid;
        Octree::LoDMesh lodMesh;
        OctreeNodeData nodeData;   // world cube of the source node (stable band center)
        bool           isBrush = false; // brush-scene entry (own IR + slot bookkeeping)
        // Pad to a multiple of 8 bytes. An odd-sized value type makes the
        // std::unordered_map hash node's in-place value slot 1 byte smaller than
        // sizeof(PendingMeshData) used by the (compiler-generated) move/copy
        // constructor, so moving a PendingMeshData out of the map reads 1 byte
        // past the node and corrupts the adjacent heap chunk header (app aborts
        // on free with "corrupted size vs. prev_size").
        uint8_t        _pad[7] = {};
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

    // Resize offscreen resources when the swapchain changes
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

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

    // Descriptor-buffer migration: change guard for updateTextureDescriptorSet().
    // Static bindings (texture arrays, shadow maps, materials, water params) are
    // written once at init and re-written only when the underlying resources
    // actually change (texture realloc, resize). updateTextureDescriptorSet() is
    // event-driven (allocation listener / swapchain resize) — never called from
    // the render loop — and this signature makes repeated invocations with
    // unchanged resources a no-op, guaranteeing 0 vkUpdateDescriptorSets calls
    // in steady state. When VulkanApp::useDescriptorBuffer() is true the same
    // guard selects the GPU-side path (direct vkGetDescriptorEXT host writes
    // into the descriptor buffer, overlapped with compute) instead of the
    // classic vkUpdateDescriptorSets fallback below.
    struct StaticTextureSignature {
        VkSampler samplers[5] = {};
        VkImageView views[5] = {};
        VkSampler shadowSampler = VK_NULL_HANDLE;
        VkImageView shadowViews[3] = {};
        VkBuffer materials = VK_NULL_HANDLE;
        VkBuffer waterParams = VK_NULL_HANDLE;
        bool valid = false;
        bool matches(const StaticTextureSignature& o) const {
            for (int i = 0; i < 5; ++i)
                if (samplers[i] != o.samplers[i] || views[i] != o.views[i]) return false;
            if (shadowSampler != o.shadowSampler) return false;
            for (int i = 0; i < 3; ++i)
                if (shadowViews[i] != o.shadowViews[i]) return false;
            return materials == o.materials && waterParams == o.waterParams;
        }
    };
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
    struct Set0DescriptorBuffers {
        std::vector<Buffer> buffers;                 // one per frame
        std::vector<VkDeviceAddress> addresses;      // device address per frame
        VkDeviceSize setSize = 0;                    // aligned set-0 byte size
        std::array<VkDeviceSize, 14> bindingOffsets{}; // driver binding offsets
        bool ready = false;
    };
    Set0DescriptorBuffers descBuffers_;

    // Camera position from the last processPendingMeshes call; used by the
    // shadow pass to cull with the same camPos/lodBias as the main pass so
    // shadow draws use the identical per-chunk LoD selection.
    glm::vec3 lastCameraPos_ = glm::vec3(0.0f);

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

    CommandBufferState frameCmdState;
};

// ...existing code...
