#pragma once

#include "Renderer.hpp"
#include "../Buffer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "CommandBufferState.hpp"
#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

class Geometry; // math/Geometry.hpp (positions + brushIndex per vertex)

// ─── Required scene / params types (built by a parallel agent) ───────────────
// The includes below are mandatory per spec; each is guarded so this renderer
// still compiles standalone while sdf/gpu/SdfScene.hpp and
// vulkan/ubo/SdfUBO.hpp do not exist yet. -I. (repo root) is on the build
// include path, so the root-relative forms resolve once the files land.
#if __has_include("sdf/gpu/SdfScene.hpp")
#include "sdf/gpu/SdfScene.hpp"
#endif
#if __has_include("vulkan/ubo/SdfUBO.hpp")
#include "vulkan/ubo/SdfUBO.hpp"
#endif

// ─── GPU types (canonical, from the parallel SDF-framework agents) ──────────
// SdfDefinitionGPU / SdfInstanceGPU / SdfMaterialGPU / SdfContainerGPU /
// SdfGridCellGPU / SdfParamsUBO live in vulkan/ubo/SdfUBO.hpp and define the
// std430 contract consumed by shaders/sdf.vert(.frag) at set=1 bindings 0..6.
// The CPU scene (sdf_gpu::SdfScene) flattens into exactly these vectors via
// SdfScene::flatten().

// Generic GPU-driven SDF renderer: one instanced proxy-cube draw per SDF
// container; the fragment shader traverses definitions/materials/grid for the
// surface, volume, emissive and transparent modes. All modes share the same
// traversal + pipeline; only SdfRenderParams::timeMode.z (RenderMode)
// changes shading. Transparent mode is shaded in-shader (no blend state) as
// a placeholder until a dedicated blended pipeline variant is added.
class SdfRenderer : public Renderer {
public:
    enum class RenderMode : uint32_t { Surface = 0, Volume = 1, Emissive = 2, Transparent = 3 };

    // CPU-side counters. GPU traversal counters (cell visits, fragments) need
    // a timestamp/occlusion query pool which is not wired yet — those fields
    // are stubs kept at 0 until then.
    struct Stats {
        uint32_t containerCount = 0;
        uint32_t definitionCount = 0;
        uint32_t materialCount = 0;
        uint32_t gridCellCount = 0;
        uint32_t lastDrawInstances = 0;
        uint64_t lastCellVisits = 0; // stub: needs query pool
        uint64_t lastFragments = 0;  // stub: needs query pool
        uint32_t lavaAnchors = 0;    // flame anchors from brush-4 lava chunks
        uint32_t lavaChunks = 0;     // lava-bearing chunks currently tracked
    };

    SdfRenderer();
    ~SdfRenderer() override;

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    // ── Decoupled offscreen framebuffer (mirrors DebugSDFRenderer) ──
    // Own color+depth targets (one per frame in flight) so the pass can run
    // on its own command buffer; PostProcess composites by depth.
    static constexpr uint32_t SDF_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    VkImageView getSdfColorView(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfColorImageViews[frameIndex] : VK_NULL_HANDLE; }
    VkImageView getSdfDepthView(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfDepthImageViews[frameIndex] : VK_NULL_HANDLE; }
    VkImage getSdfColorImage(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfColorImages[frameIndex] : VK_NULL_HANDLE; }
    VkImage getSdfDepthImage(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfDepthImages[frameIndex] : VK_NULL_HANDLE; }
    VkImageLayout getSdfColorLayout(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfColorImageLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED; }
    VkImageLayout getSdfDepthLayout(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfDepthImageLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED; }
    void setSdfColorLayout(uint32_t frameIndex, VkImageLayout l) { if (frameIndex < SDF_FRAMES) sdfColorImageLayouts[frameIndex] = l; }
    void setSdfDepthLayout(uint32_t frameIndex, VkImageLayout l) { if (frameIndex < SDF_FRAMES) sdfDepthImageLayouts[frameIndex] = l; }

    // ── Scene input ──
    // setScene copies the CPU mirror and flags the GPU upload dirty. Call from
    // the render thread (or load path), NOT from a worker while prepareCull /
    // render for the same frame is recording.
    void setScene(const sdf_gpu::SdfScene& scene);
    // Re-extract GPU mirrors from the pending scene if dirty (CPU only).
    // Returns true when a re-upload was staged.
    bool rebuildGridIfDirty();
    // Convenience: rebuildGridIfDirty() now (upload itself happens in
    // prepareCull on the render thread).
    void updateScene();

    // ── Lava-anchored fire (brushIndex 4) ────────────────────────────────
    // Flame anchors are derived from the SOLID shape of lava terrain chunks
    // (same brush-4 triangles the legacy billboards use), keyed by chunk so
    // edits replace and deletions remove their flames. The SDF scene itself
    // stays generic (see SdfScene::FlameAnchor/createFireFromAnchors); this
    // collector is the only lava-aware piece, and it never touches the
    // terrain octree — it consumes published chunk geometry like the
    // vegetation generator does. Main thread only (called from
    // processPendingMeshes / octree delete handlers); guarded by sceneMutex
    // because the async sdf task reads the mirrors.
    static constexpr int kLavaBrushIndex = 4;
    static constexpr size_t kMaxLavaPerChunk = 12;
    static constexpr size_t kMaxLavaAnchors = 2048;
    void ingestLavaChunk(uintptr_t nid, const Geometry& geom);
    void removeLavaChunk(uintptr_t nid);
    void clearLava();
    // Rebuild the fire scene from collected anchors when dirty (coalesces
    // per-chunk ingests into at most one flatten per frame). Returns true
    // when a rebuild was staged.
    bool rebuildLavaIfDirty();
    // Flames per m² of lava surface at ingest (default 1 per 10000 m²)
    // and anchor scale multiplier. Apply to newly streamed chunks
    // (same limitation as the vegetation density control).
    void setLavaDensity(float d);
    float lavaDensity() const;
    void setLavaScale(float s);
    float lavaScale() const;
    // Flame shape (tapered two-radii capsule + spikes). Applied at scene
    // rebuild (new def params); marks lava dirty so the next
    // rebuildLavaIfDirty() re-flattens.
    void setLavaSpikiness(float s);
    float lavaSpikiness() const;
    void setLavaTipRadius(float r);
    float lavaTipRadius() const;
    // Tapered-capsule shape (local units, multiplied by instance scale).
    // Applied at scene rebuild; mark lava dirty so the next
    // rebuildLavaIfDirty() re-flattens.
    void setLavaBaseRadius(float r);
    float lavaBaseRadius() const;
    void setLavaHeight(float h);
    float lavaHeight() const;
    void setLavaSpikeFreq(float f);
    float lavaSpikeFreq() const;
    // Volumetric density multiplier (lower = more transparent, like real
    // flames). Applied at scene rebuild; marks lava dirty.
    void setLavaFlameDensity(float d);
    float lavaFlameDensity() const;

    // ── Per-frame parameters (packed into SdfParamsUBO) ──
    // SdfParamsUBO carries no renderer-mode field, so timeDebug.y packs two
    // uint16s as an exact float (< 2^24): high = RenderMode, low = debugFlags.
    // Shaders decode with uint(timeDebug.y): mode = v >> 16, flags = v & 0xFFFF.
    void updateParams(float timeSec, uint32_t frameIndex);
    void setFrame(uint32_t frame) { currentFrame_ = frame % SDF_FRAMES; }
    void setRenderMode(RenderMode mode);
    void setDebugFlags(uint32_t flags);
    void setMarchParams(float maxSteps, float epsilon);
    void setMarchRange(float minStep, float maxStep, float earlyTermThreshold);
    // External scene depth (main depth buffer view) used for occlusion.
    // May be called every frame; the descriptor is rewritten only when the
    // view handle actually changes (per frame slot), never blindly per frame.
    void setSceneDepth(VkImageView view, VkImageLayout layout);

    // Host->SSBO upload (pending memcpys) + Sync2 visibility barrier for this
    // frame's slot. Call OUTSIDE a render pass, before render().
    void prepareCull(VkCommandBuffer cmd);

    // Proxy-cube instanced draw (instanceCount = container count) to this
    // frame's offscreen color+depth. Ends SHADER_READ_ONLY for composite.
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet mainDescriptorSet, uint32_t frameIdx, bool enabled = true);

    const Stats& getStats() const { return stats_; }
    uint32_t getContainerCount() const { return stats_.containerCount; }
    bool hasScene() const { return stats_.containerCount > 0; }

private:
    struct CubeVertex {
        glm::vec3 position;
    };

    // One slot per frame in flight: every buffer a draw reads is slot-local,
    // so a host rewrite for frame N can never race an in-flight draw of frame
    // N-1/N-2 that references a different slot. Static scene data could share
    // one buffer, but per-slot copies keep the lifetime story trivial (same
    // pattern as DebugSDFRenderer::cullFrames) at negligible memory cost.
    // The single persistent buffer + host barrier alternative would race when
    // 3 frames are in flight (frame N+1 rewrite vs frame N draw), which is why
    // triple-buffering is used instead.
    struct FrameSlot {
        Buffer instance;    // SdfGpuInstance per container (set=1 binding 0)
        Buffer definition;  // SdfGpuDefinition (binding 1)
        Buffer material;    // SdfGpuMaterial (binding 2)
        Buffer container;   // SdfGpuContainer (binding 3)
        Buffer gridCell;    // SdfGpuGridCell (binding 4)
        Buffer gridIndex;   // uint32_t (binding 5)
        Buffer params;      // SdfParamsUBO uniform (binding 6)
        uint32_t instanceCap = 0;
        uint32_t definitionCap = 0;
        uint32_t materialCap = 0;
        uint32_t containerCap = 0;
        uint32_t gridCellCap = 0;
        uint32_t gridIndexCap = 0;
    };

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkShaderModule> vertModule;
    TrackedHandle<VkShaderModule> fragModule;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;

    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    // One descriptor set per frame slot; each is (re)written only when one of
    // its slot buffers is (re)allocated or the scene-depth view changes — never
    // blindly per frame — so a set is never updated while a CB using it is pending.
    std::array<VkDescriptorSet, SDF_FRAMES> sdfSets{};
    VkSampler depthSampler = VK_NULL_HANDLE;

    std::array<FrameSlot, SDF_FRAMES> slots;
    uint32_t currentFrame_ = 0;

    // CPU mirrors (canonical GPU types) + dirty flags (guarded; setScene may
    // come from the load path). params_ mirrors SdfParamsUBO with the
    // timeDebug.y packing documented on updateParams.
    mutable std::mutex sceneMutex;
    sdf_gpu::SdfScene pendingScene_; // last setScene() copy; flattened on rebuild
    std::vector<SdfInstanceGPU> instances_;
    std::vector<SdfDefinitionGPU> definitions_;
    std::vector<SdfMaterialGPU> materials_;
    std::vector<SdfContainerGPU> containers_;
    std::vector<SdfGridCellGPU> gridCells_;
    std::vector<uint32_t> gridIndices_;
    SdfParamsUBO params_ = {};
    RenderMode renderMode_ = RenderMode::Surface;
    uint32_t debugFlags_ = 0;
    // Per-slot dirty flags (triple-buffered slots!). A single global flag
    // breaks after the first frame: slot 0 consumes it, slots 1-2 never
    // allocate/upload yet still hit the barrier/draw with NULL buffers
    // (VUID pBufferMemoryBarriers[].buffer must not be NULL). Every setScene /
    // params change marks ALL slots dirty; flush clears only the slot written.
    std::array<bool, SDF_FRAMES> sceneDirtySlots_ = {true, true, true};
    std::array<bool, SDF_FRAMES> paramsDirtySlots_ = {true, true, true};

    // Lava-anchored flame collection (brush-4 chunk geometry -> anchors).
    struct LavaAnchor {
        glm::vec3 pos = glm::vec3(0.0f);
        glm::vec3 euler = glm::vec3(0.0f); // Y-up frame tilted onto the surface normal
        float scale = 1.0f;
        float heightScale = 1.0f; // per-instance flame stretch (0.8-1.4)
        float seed = 0.0f;
        float intensity = 1.0f;
    };
    std::unordered_map<uintptr_t, std::vector<LavaAnchor>> lavaByChunk_;
    float lavaDensity_ = 1.0f; // flames per m² of lava surface at ingest (1 per m²)
    float lavaScale_ = 32.0f;   // anchor scale multiplier at ingest
    float lavaSpikiness_ = 0.35f; // flame spike amplitude (0 = smooth rounded capsule)
    float lavaTipRadius_ = 0.125f; // flame tip radius, local units (0 = sharp cone tip)
    float lavaBaseRadius_ = 1.0f; // flame base radius, local units
    float lavaHeight_ = 1.0f;     // flame base-to-tip height, local units
    float lavaSpikeFreq_ = 2.0f;  // tongue count around the flame axis
    float lavaFlameDensity_ = 0.05f; // volumetric density (lower = glassier)
    bool lavaDirty_ = false;    // anchors changed -> rebuild staged

    VkImageView pendingDepthView_ = VK_NULL_HANDLE;
    VkImageLayout pendingDepthLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    std::array<VkImageView, SDF_FRAMES> boundDepthViews_{};

    Stats stats_;
    VulkanApp* app_ = nullptr; // stashed for buffer (re)allocation

    // Offscreen color+depth targets (one per frame in flight).
    std::array<VkImage, SDF_FRAMES> sdfColorImages{};
    std::array<VmaAllocation, SDF_FRAMES> sdfColorAllocations{};
    std::array<VkDeviceMemory, SDF_FRAMES> sdfColorMemories{};
    std::array<VkImageView, SDF_FRAMES> sdfColorImageViews{};
    std::array<VkImageLayout, SDF_FRAMES> sdfColorImageLayouts{};
    std::array<VkImage, SDF_FRAMES> sdfDepthImages{};
    std::array<VmaAllocation, SDF_FRAMES> sdfDepthAllocations{};
    std::array<VkDeviceMemory, SDF_FRAMES> sdfDepthMemories{};
    std::array<VkImageView, SDF_FRAMES> sdfDepthImageViews{};
    std::array<VkImageLayout, SDF_FRAMES> sdfDepthImageLayouts{};
    uint32_t sdfRenderWidth = 0;
    uint32_t sdfRenderHeight = 0;

    void createCubeBuffers(VulkanApp* app);
    void createDescriptorSet(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    // Flatten pendingScene_ into the gpu mirror vectors (CPU only).
    void extractFlattened();
    // Re-pack timeDebug.y from renderMode_ + debugFlags_ (see updateParams).
    void repackDebugMode();
    void ensureSlotCapacity(uint32_t slot); // grow slot buffers with headroom; rewrites slot set bindings
    void flushSlotUploads(uint32_t slot);   // memcpy dirty mirrors/params into slot buffers
    void writeSlotBinding(uint32_t slot, uint32_t binding, const Buffer& buf, VkDescriptorType type);
    void refreshDepthBinding(uint32_t slot); // rewrite binding 7 iff the view changed for this slot
};
