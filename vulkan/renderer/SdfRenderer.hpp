#pragma once

#include "Renderer.hpp"
#include "../Buffer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "CommandBufferState.hpp"
#include "sdf/types/SdfEffectConfig.hpp"
#include "sdf/types/SdfStats.hpp"
#include "SdfRendererTypes.hpp"
#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

class Geometry; // math/Geometry.hpp (positions + brushIndex per vertex)

// ─── Required scene / GPU types ────────────────────────────────────────────
// -I. (repo root) is on the build include path, so the root-relative forms
// below resolve. GPU-wire structs live in vulkan/types/ (one file per struct).
#include "sdf/types/SdfScene.hpp"
#include "vulkan/types/SdfDefinitionGPU.hpp"
#include "vulkan/types/SdfInstanceGPU.hpp"
#include "vulkan/types/SdfMaterialGPU.hpp"
#include "vulkan/types/SdfContainerGPU.hpp"
#include "vulkan/types/SdfGridCellGPU.hpp"
#include "vulkan/ubo/SdfParamsUBO.hpp"
#include "vulkan/types/SmokeFragBulletGPU.hpp"

// ─── GPU types (canonical std430 contract, one file per struct in types/) ──
// SdfDefinitionGPU / SdfInstanceGPU / SdfMaterialGPU / SdfContainerGPU /
// SdfGridCellGPU / SdfParamsUBO / SmokeFragBulletGPU define the contract consumed by
// shaders/sdf.vert(.frag) at set=1 bindings 0..8.
// The CPU scene (sdf_gpu::SdfScene) flattens into exactly these vectors via
// FlattenSdfScene (see vulkan/renderer/SdfSceneFlatten.hpp).

// Generic GPU-driven SDF renderer: one instanced proxy-cube draw per SDF
// container; the fragment shader traverses definitions/materials/grid for the
// surface, volume, emissive and transparent modes. All modes share the same
// traversal + pipeline; only SdfRenderParams::timeMode.z (RenderMode)
// changes shading. Transparent mode is shaded in-shader (no blend state) as
// a placeholder until a dedicated blended pipeline variant is added.
class SdfRenderer : public Renderer {
public:
    enum class RenderMode : uint32_t { Surface = 0, Volume = 1, Emissive = 2, Transparent = 3 };

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
    void setScene(const sdf_gpu::SdfScene& scene);    // Re-extract GPU mirrors from the pending scene if dirty (CPU only).
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
    void setLavaScale(float s);
    // Flame shape (tapered two-radii capsule + spikes). Applied at scene
    // rebuild (new def params); marks lava dirty so the next
    // rebuildLavaIfDirty() re-flattens.
    void setLavaSpikiness(float s);
    void setLavaTipRadius(float r);
    // Tapered-capsule shape (local units, multiplied by instance scale).
    // Applied at scene rebuild; mark lava dirty so the next
    // rebuildLavaIfDirty() re-flattens.
    void setLavaBaseRadius(float r);
    void setLavaHeight(float h);
    void setLavaSpikeFreq(float f);
    // Volumetric density multiplier (lower = more transparent, like real
    // flames). Applied at scene rebuild; marks lava dirty.
    void setLavaFlameDensity(float d);

    // ── Smoke bomb + bullets (second generic consumer) ──────────────────
    // A static-topology smoke scene (1 Smoke-sphere def/mat/container/
    // instance) merged with the lava scene on every rebuild. Growth, noise,
    // bullets and render tuning live in the smoke state SSBO (binding 8)
    // and NEVER rebuild geometry; only position/radius/base material touch
    // the scene. Bullet motion, wake aging and refill are pure GPU functions
    // of global time, so in-flight bullets need no CPU updates.
    void ensureSmokeScene(); // (re)build smokeScene_ from smoke params; caller holds sceneMutex, then refreshMerged()
    void setSmokeEnabled(bool e);
    void setSmokePosition(const glm::vec3& p);
    void setSmokeScale(float r);
    void setSmokeGrowthDuration(float s);
    void setSmokeLoopDuration(float s);
    void setSmokeDissipation(float d);
    void setSmokeNoiseScale(float s);
    void setSmokeNoiseStrength(float s);
    void setSmokeNoiseWarp(float s);
    void setSmokeWind(float speed, float angleDeg);
    void setSmokeDensityScale(float s);
    void setSmokeDensity(float d); // base material density (rebuilds scene)
    void setSmokeAbsorption(float a);
    void setSmokeScattering(float s);
    void setSmokeTunnel(float strength, float falloff);
    void setSmokeWake(float strength, float radius, float expansion, float length, float dissipation);
    void setSmokePressure(float radius, float strength, float waveSpeed, float waveFreq, float waveFalloff);
    void setSmokeTurbulence(float scale, float strength, float speed);
    void setSmokeShadow(int samples, float strength);
    // Smoke tint (smoke SSBO smokeColor; streams, no scene rebuild).
    void setSmokeColor(const glm::vec3& rgb);
    // Shape rig (reference port): 0 = cloud (billowy sphere), 1 = sphere,
    // 2 = cube. The cube's bounding sphere equals the smoke scale, so the
    // CPU AABB and container stay valid for every shape and orientation.
    // All three stream through the smoke SSBO; rotation is yaw/pitch/roll
    // degrees applied about the smoke center in object space.
    void setSmokeShape(int shape);
    void setSmokeRotation(float yawDeg, float pitchDeg, float rollDeg);
    // Bullet defaults (stamped into slots on fire; slot 0 = auto-loop template).
    // radiusStart = launch radius, radiusEnd = radius at the head; loopDuration
    // is the per-bullet cycle (s). Matches the BulletGPU ABI.
    void setBulletDefaults(float radiusStart, float radiusEnd, float speed, float length,
                           float angleDeg, float loopDuration);
    void setAutoFire(bool on);
    // Gold tracer shading (smoke SSBO gold0/1/2; streams, no scene rebuild).
    void setBulletGold(const glm::vec3& deep, const glm::vec3& bright,
                       float specPower, float specStrength, float fresnelBoost,
                       float warmFloor, float patternScale, float normalDistort);
    // Fire a bullet along angleDeg (default: widget angle) from an
    // auto-crossing start. Round-robins manual slots 1..7.
    void fireBullet(float angleDeg);
    void clearBullets();
    uint32_t bulletSlotsUsed() const;
    // Smoke debug views (§24: 0 = normal, 1-10 per spec list). Upper nibble
    // of debugFlags; fire views (low nibble) are preserved.
    void setSmokeDebug(uint32_t v);

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

    const SdfStats& getStats() const { return stats_; }
    uint32_t getContainerCount() const { return stats_.containerCount; }
    bool hasScene() const { return stats_.containerCount > 0; }

    // Shared effect tuning (single instance used by renderer + widgets).
    SdfEffectConfig& config() { return config_; }
    const SdfEffectConfig& config() const { return config_; }

private:
    std::array<SdfFrameSlot, SDF_FRAMES> slots;

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

    uint32_t currentFrame_ = 0;

    // CPU mirrors (canonical GPU types) + dirty flags (guarded; setScene may
    // come from the load path). params_ mirrors SdfParamsUBO with the
    // timeDebug.y packing documented on updateParams.
    mutable std::mutex sceneMutex;
    sdf_gpu::SdfScene pendingScene_; // merged lava + smoke; flattened on rebuild
    sdf_gpu::SdfScene lavaScene_;    // fire scene from lava anchors
    sdf_gpu::SdfScene smokeScene_;   // static-topology smoke bomb scene
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
    std::array<bool, SDF_FRAMES> smokeDirtySlots_ = {true, true, true};
    float lastTime_ = 0.0f; // global time of the last updateParams (bullet birth clock)

    // Lava-anchored flame collection (brush-4 chunk geometry -> anchors,
    // stored as scene FlameAnchors so chunk edits replace and deletions
    // remove their flames with no conversion step).
    std::unordered_map<uintptr_t, std::vector<sdf_gpu::SdfScene::FlameAnchor>> lavaByChunk_;
    // Shared tuning (single source of truth for renderer + widgets).
    SdfEffectConfig config_;
    bool lavaDirty_ = false;    // anchors changed -> rebuild staged

    // Smoke bomb state (second generic consumer). Scene topology is static
    // (1 def/mat/container/instance); all behavior below streams through
    // the smoke SSBO without scene rebuilds, except position/radius/base
    // material which reshape the scene.
    SmokeFragBulletGPU smokeState_ = {}; // SSBO mirror: tuning + bullets
    // Manual rounds always use slot 1 (slot 0 is the auto-loop template);
    // firing clears any previous manual round: single-flight, one at a time.

    VkImageView pendingDepthView_ = VK_NULL_HANDLE;
    VkImageLayout pendingDepthLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    std::array<VkImageView, SDF_FRAMES> boundDepthViews_{};

    SdfStats stats_;
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
    // Re-merge lava + smoke scenes into pendingScene_ and flatten (caller
    // holds sceneMutex). Marks all scene slots dirty.
    void refreshMergedLocked();
    // Re-pack timeDebug.y from renderMode_ + debugFlags_ (see updateParams).
    void repackDebugMode();
    void ensureSlotCapacity(uint32_t slot); // grow slot buffers with headroom; rewrites slot set bindings
    void flushSlotUploads(uint32_t slot);   // memcpy dirty mirrors/params into slot buffers
    void flushSmokeUpload(uint32_t slot);   // memcpy dirty smoke state into the slot buffer
    void writeSlotBinding(uint32_t slot, uint32_t binding, const Buffer& buf, VkDescriptorType type);
    void refreshDepthBinding(uint32_t slot); // rewrite binding 7 iff the view changed for this slot
    // Auto-loop bullet template (slot 0) from widget defaults (caller holds sceneMutex).
    void refreshAutoBulletLocked();
    void markSmokeSSBO(); // flag all smoke slots dirty (tuning/bullet change, no re-flatten)
    // Single-flight policy: at most one bullet at a time. A live manual
    // round suppresses the auto bullet until it dies (wake grace included).
    bool anyManualBulletLiveLocked() const;
    void syncAutoBulletLocked();
};
