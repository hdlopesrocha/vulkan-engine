#pragma once

#include "../Renderer.hpp"
#include "../../resources/Buffer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../core/TrackedHandle.hpp"
#include "../CommandBufferState.hpp"
#include "sdf/types/SdfEffectConfig.hpp"
#include "sdf/types/SdfStats.hpp"
#include "SdfRendererTypes.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

class Geometry; // math/Geometry.hpp (positions + brushIndex per vertex)

// ─── Required scene / GPU types ────────────────────────────────────────────
// -I. (repo root) is on the build include path, so the root-relative forms
// below resolve. The canonical GPU-layout structs live in sdf/types/ (one
// file per struct); each has a GLSL twin in shaders/types/ (same base name,
// identical layout). No CPU->GPU conversion step exists: the scene stores
// these structs directly and the upload memcpys them verbatim.
#include "sdf/types/SdfScene.hpp"
#include "sdf/types/SdfDefinition.hpp"
#include "sdf/types/SdfInstance.hpp"
#include "sdf/types/SdfMaterial.hpp"
#include "sdf/types/SdfContainer.hpp"
#include "sdf/types/SdfGridCell.hpp"
#include "vulkan/ubo/SdfParamsUBO.hpp"
#include "sdf/types/SmokeFragBullet.hpp"

// ─── GPU types (canonical std430 contract, one file per struct) ────────────
// SdfDefinition / SdfInstance / SdfMaterial / SdfContainer /
// SdfGridCell / SdfParamsUBO / SmokeFragBullet define the contract
// consumed by shaders/SdfRenderer.vert(.frag) at set=1 bindings 0..9
// (9 = rasterized water-surface depth for the solid+water march clamp).

// Generic GPU-driven SDF renderer: one instanced proxy-cube draw per SDF
// container; the fragment shader traverses definitions/materials/grid for the
// surface, volume, emissive and transparent modes. All modes share the same
// traversal + pipeline; only SdfRenderParams::timeMode.z (RenderMode)
// changes shading. Transparent mode is shaded in-shader (no blend state) as
// a placeholder until a dedicated blended pipeline variant is added.
// C1 (perf report 25): two fragment pipelines share one vertex module, one
// descriptor layout and one pipeline layout from the single SdfRenderer.frag
// source — the generic pipeline (all modes) plus the surface variant
// (SdfRendererSurface.frag.spv, -DSDF_VARIANT=1, smoke resolve compiled
// out), bound for Surface/Transparent modes with fallback to generic.
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
    // Read-only CPU mirror of the merged effect scene (lava flames, smoke)
    // for the Blender exporter. Call from the render thread; the scene
    // is refreshed by the regular render path.
    const sdf_gpu::SdfScene& exportScene() const { return pendingScene_; }
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

    // ── Ray-cast quality (Settings: SDF raycast pixel size) ──────────────
    // The SDF ray is cast once per NxN screen-pixel block (block center) when
    // N > 1, pixelating the SDF output; the full-resolution proxy
    // rasterization keeps the per-pixel hardware depth test against solid.
    // 1 = one ray per pixel. No scene rebuild (streamed through the params
    // UBO). Default 2 (Settings::raycastPixelSize).
    void setRaycastPixelSize(int px);
    // Ray-marching master gate (Settings::rayMarchingEnabled): when off, the
    // main SDF pass clears its targets without drawing. Lock-free: written
    // from the frame thread, read from the async SDF/shadow tasks.
    void setRayMarchingEnabled(bool on) { rayMarchingEnabled_.store(on, std::memory_order_relaxed); }
    bool isRayMarchingEnabled() const { return rayMarchingEnabled_.load(std::memory_order_relaxed); }
    // M10 (report 25): screen-space overlap gate for the solid-depth preload
    // copy. The frame thread stores the current view-projection each frame;
    // render() skips the full-screen copy when no container projects onto
    // the screen. Validation-neutral: only removes a copy + transitions,
    // end-state layouts are unchanged.
    void setOcclusionViewProj(const glm::mat4& vp);
    // C2 (report 25): smoke container sizing. true (default) keeps the
    // legacy 3700 m XZ pad (A/B baseline, zero behavior change); false fits
    // the container to the smoke ball + live bullet flights (quantized refit).
    void setSmokeWideContainer(bool wide);

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
    void setSmokeShock(float radius, float strength, float rippleAmp, float rippleFreq);
    void setSmokeTurbulence(float scale, float strength, float speed);
    void setSmokeShadow(int samples, float strength);
    void setSmokeHeatStrength(float strength);
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
    // is the per-bullet cycle (s). Matches the Bullet ABI.
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
    // SdfParamsUBO carries renderMode and debugFlags as separate integers
    // (no bit packing).
    void updateParams(float timeSec, uint32_t frameIndex);
    void setFrame(uint32_t frame) {
        currentFrameIndex_ = frame;
        currentFrame_ = frame % SDF_FRAMES;
    }
    // Refresh the per-slot depth bindings (set=1 bindings 7/9) once per frame
    // BEFORE any command buffer binds the SDF set for this frame.
    // vkUpdateDescriptorSets is illegal while a pending CB references the set
    // (VUID 03047), so the refresh must happen before the SDF CB records.
    // Main thread: call after setSceneDepth/setWaterDepth, before the shadow
    // task is enqueued. Safe: the set's previous slot use is fenced (3 frames
    // in flight).
    void refreshFrameBindings(uint32_t frameIdx);
    void setRenderMode(RenderMode mode);
    void setDebugFlags(uint32_t flags);
    void setMarchParams(float maxSteps, float epsilon);
    void setMarchRange(float minStep, float maxStep, float earlyTermThreshold);
    // March-tier setters (perf report 25 M12, Settings::sdfMaxSteps /
    // sdfSmokeSamples). Each dedupes unchanged values and marks all params
    // slots dirty, so the per-frame MyApp wiring is cheap. No idle, no
    // rebuild: the values stream through SdfParamsUBO (unlike the report-23
    // C1 texture tier, which needs a device idle + full re-upload).
    void setMaxSteps(int steps);
    void setSmokeSamples(int n);
    void setMarchTiers(int maxSteps, int smokeSamples);
    // External scene depth (main depth buffer view) used for occlusion.
    // May be called every frame; the descriptor is rewritten only when the
    // view handle actually changes (per frame slot), never blindly per frame.
    // `image`/`width`/`height` let render() pre-load the SDF depth attachment
    // with a copy of the solid depth, so the hardware depth test rejects SDF
    // fragments behind rasterized solid geometry at raster time (the in-shader
    // tExit clamp and the composite test stay as extra layers).
    void setSceneDepth(VkImageView view, VkImageLayout layout,
                       VkImage image = VK_NULL_HANDLE,
                       uint32_t width = 0, uint32_t height = 0);
    // Rasterized water-surface geometry depth (WaterRenderer::
    // getWaterGeomDepthView) used to clamp the march exit against water.
    // `enabled` gates the shader side (SdfParamsUBO::waterDepthEnabled) and
    // must be false when the water pass did not write a valid geometry depth
    // this frame (Minimal/aux-less water variant, water disabled).
    void setWaterDepth(VkImageView view, VkImageLayout layout, bool enabled);

    // Host->SSBO upload (pending memcpys) + Sync2 visibility barrier for this
    // frame's slot. Call OUTSIDE a render pass, before render().
    void prepareCull(VkCommandBuffer cmd);

    // Retired grass-only EVSM shadow API (bladed grass casts via
    // VegetationRenderer). Kept as no-ops so ShadowRenderer links unchanged.
    void prepareShadowCascade(VkCommandBuffer, uint32_t) {}
    void drawShadowCascade(VkCommandBuffer, uint32_t, const glm::mat4&, float) {}
    void setShadowLightDirection(const glm::vec3&) {}

    // Global SDF clock (last updateParams time).
    float getTime() const;

    // Proxy-cube instanced draw (instanceCount = container count) to this
    // frame's offscreen color+depth. Ends SHADER_READ_ONLY for composite.
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet mainDescriptorSet, uint32_t frameIdx, bool enabled = true);

    const SdfStats& getStats() const { return stats_; }
    uint32_t getContainerCount() const { return stats_.containerCount; }
    bool hasScene() const { return stats_.containerCount > 0; }
    // GPU march counters (set=1 binding 10). readProfile copies the slot's last
    // completed counters into stats_ — call on the frame thread: the slot's
    // previous use is fenced before reuse, so the mapped read never races the
    // GPU atomics or render()'s reset.
    void readProfile(uint32_t frameIdx);
    // Shader-side counter gate (profiling UI). supported() is false when the
    // device lacks fragmentStoresAndAtomics; the counters then stay disabled.
    void setProfileCounters(bool on) { profileCountersEnabled_.store(on, std::memory_order_relaxed); }
    bool profileCountersSupported() const { return profileCountersSupported_; }

    // Shared effect tuning (single instance used by renderer + widgets).
    SdfEffectConfig& config() { return config_; }
    const SdfEffectConfig& config() const { return config_; }

private:
    std::array<SdfFrameSlot, SDF_FRAMES> slots;

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkShaderModule> vertModule;
    TrackedHandle<VkShaderModule> fragModule;
    // C1 surface fragment variant (SdfRendererSurface.frag.spv, same vertex
    // module + pipeline layout as above). Null until its SPIR-V exists (same
    // lazy tolerance as createPipeline); render() falls back to `pipeline`.
    // L13: one vertex source/module serves both pipelines.
    TrackedHandle<VkShaderModule> fragSurfaceModule;
    TrackedHandle<VkPipeline> pipelineSurface;

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

    // SDF march counters (set=1 binding 10): one host-visible mapped buffer per
    // frame slot. render() resets + gates it; readProfile() reads the last
    // completed use.
    std::array<Buffer, SDF_FRAMES> profileBuffers_;
    bool profileCountersSupported_ = false;
    std::atomic<bool> profileCountersEnabled_{false};

    uint32_t currentFrame_ = 0;
    uint32_t currentFrameIndex_ = 0; // unmodded frame index (binding-refresh gate)
    // Ray-marching master gate (Settings::rayMarchingEnabled); atomic so the
    // frame thread can flip it while the async tasks read it.
    std::atomic<bool> rayMarchingEnabled_{true};
    // M10 occlusion gate: latest frame view-projection for the CPU 2D
    // container-overlap test (guarded by sceneMutex; render() takes it under
    // the same lock that reads the pending depth image).
    glm::mat4 occlusionViewProj_{1.0f};
    bool occlusionViewProjValid_ = false;
    // App frame index whose depth bindings were last refreshed (see
    // refreshFrameBindings). Guarded by sceneMutex.
    uint32_t bindingsFrame_ = 0xFFFFFFFFu;

    // CPU scene state + dirty flags (guarded; setScene may come from the
    // load path). The merged scene stores the canonical GPU-layout structs,
    // so the upload memcpys its vectors verbatim (no CPU->GPU conversion).
    mutable std::mutex sceneMutex;
    sdf_gpu::SdfScene pendingScene_; // merged lava + smoke (uploaded directly)
    sdf_gpu::SdfScene lavaScene_;    // fire scene from lava anchors
    sdf_gpu::SdfScene smokeScene_;   // static-topology smoke/flame shape scene
    SdfParamsUBO params_ = {};
    RenderMode renderMode_ = RenderMode::Surface;
    uint32_t debugFlags_ = 0;
    int raycastPixelSize_ = 2; // Settings::raycastPixelSize (streamed via params_)
    // Per-slot dirty flags (triple-buffered slots!). A single global flag
    // breaks after the first frame: slot 0 consumes it, slots 1-2 never
    // allocate/upload yet still hit the barrier/draw with NULL buffers
    // (VUID pBufferMemoryBarriers[].buffer must not be NULL). Every setScene /
    // params change marks ALL slots dirty; flush clears only the slot written.
    std::array<bool, SDF_FRAMES> sceneDirtySlots_ = {true, true, true};
    std::array<bool, SDF_FRAMES> paramsDirtySlots_ = {true, true, true};
    std::array<bool, SDF_FRAMES> smokeDirtySlots_ = {true, true, true};
    // M9 (perf report 25): staging -> device-local copy discipline. flush*
    // memcpys into the staging twins and stages the byte counts below; the
    // first command buffer of the frame records the vkCmdCopyBuffer copies
    // (copiesPendingSlots_ consumed there) while EVERY command buffer of the
    // frame records its own barrier via recordHostToShaderBarrier (each CB /
    // queue must: the shadow cascades, the cull CB and the SDF CB may be
    // different CBs). The *Frame_ arrays remember which frame index staged /
    // copied / flushed each slot, so steady-state frames (nothing staged
    // this frame) record no barrier at all instead of the old unconditional
    // 8-entry barrier. Per-slot dirty flags stay the upload gate.
    // TODO(M9-full): promote the copies to the existing staging pool
    // (StagingRingBuffer / streaming::UploadManager, report-23 C3 batched
    // pattern) on a dedicated transfer queue with ownership transfer; the
    // per-slot staging twins here are the safe subset that preserves the
    // triple-buffer lifetime without new fence tracking.
    // TODO(M12-distance): CPU distance-scaled maxSteps per container (near
    // full budget, far halved) needs a draw split: render() issues ONE
    // instanced draw over all containers from a single global UBO budget, so
    // per-container budgets need per-draw push constants + near/far passes
    // with A/B. Tier-only until then.
    std::array<bool, SDF_FRAMES> copiesPendingSlots_{false, false, false};
    std::array<std::array<VkDeviceSize, 6>, SDF_FRAMES> stagedBytes_{};
    std::array<uint32_t, SDF_FRAMES> sceneCopiedFrame_{};
    std::array<uint32_t, SDF_FRAMES> paramsFlushedFrame_{};
    std::array<uint32_t, SDF_FRAMES> smokeFlushedFrame_{};
    float lastTime_ = 0.0f; // global time of the last updateParams (bullet birth clock)

    // Lava-anchored flame collection (brush-4 chunk geometry -> anchors,
    // stored as scene FlameAnchors so chunk edits replace and deletions
    // remove their flames with no conversion step).
    std::unordered_map<uintptr_t, std::vector<sdf_gpu::SdfScene::FlameAnchor>> lavaByChunk_;
    // Shared tuning (single source of truth for renderer + widgets).
    SdfEffectConfig config_;
    // C2: true = legacy 3700 m XZ smoke pad (default, A/B baseline);
    // false = tight ball + live-flight fit (see fitSmokeTightLocked).
    // Guarded by sceneMutex.
    bool smokeWideContainer_ = true;
    bool lavaDirty_ = false;    // anchors changed -> rebuild staged

    // Smoke bomb state (second generic consumer). Scene topology is static
    // (1 def/mat/container/instance); all behavior below streams through
    // the smoke SSBO without scene rebuilds, except position/radius/base
    // material which reshape the scene.
    SmokeFragBullet smokeState_ = {}; // SSBO mirror: tuning + bullets
    // Manual rounds always use slot 1 (slot 0 is the auto-loop template);
    // firing clears any previous manual round: single-flight, one at a time.

    VkImageView pendingDepthView_ = VK_NULL_HANDLE;
    VkImageLayout pendingDepthLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Solid depth image + extent for the SDF-depth pre-load copy (the same
    // depth whose view is bound at binding 7).
    VkImage pendingDepthImage_ = VK_NULL_HANDLE;
    uint32_t pendingDepthWidth_ = 0;
    uint32_t pendingDepthHeight_ = 0;
    std::array<VkImageView, SDF_FRAMES> boundDepthViews_{};
    // Water-surface depth binding (set=1 binding 9): same per-slot dedupe.
    // The shader only samples it while params_.waterDepthEnabled > 0.5.
    VkImageView pendingWaterDepthView_ = VK_NULL_HANDLE;
    VkImageLayout pendingWaterDepthLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bool pendingWaterDepthEnabled_ = false;
    std::array<VkImageView, SDF_FRAMES> boundWaterDepthViews_{};

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
    // Re-merge lava + smoke scenes into pendingScene_, rebuild its
    // bounds and grids (caller holds sceneMutex). Marks all scene slots dirty.
    void refreshMergedLocked();
    // C2 (report 25): tight smoke-container fit. liveBulletBoundsLocked
    // returns the world-space AABB of every LIVE bullet's full flight segment
    // (start -> start+dir*pathLength, radius + wake margin; smoke-local frame
    // mapped to world via the shape rotation). Full path, not current
    // traveled distance, so bounds stay static during flight: one refit on
    // birth, one on death. tightSmokeBoundsLocked unions that with the smoke
    // ball and snaps out to kSmokeFitQuantum (the epsilon mechanism).
    // Caller holds sceneMutex.
    bool liveBulletBoundsLocked(glm::vec3& mn, glm::vec3& mx) const;
    void tightSmokeBoundsLocked(glm::vec3& mn, glm::vec3& mx) const;
    void fitSmokeTightLocked();
    void refitSmokeContainerIfNeededLocked();
    // M10: true when any merged container can cover a pixel (conservative:
    // true until the first view-projection arrives). Takes sceneMutex.
    bool containersOverlapScreen();
    // ── SdfModel helpers (one transform convention for every effect) ─────
    // Euler XYZ radians, R = Rx * Ry * Rz (mirrors sdfEulerMat in sdf_ops).
    // Caller holds sceneMutex.
    glm::mat3 smokeRotLocked() const;

    // Refresh params_.renderMode/debugFlags from the renderer state.
    void repackDebugMode();
    void ensureSlotCapacity(uint32_t slot); // grow slot buffers with headroom; rewrites slot set bindings
    void flushSlotUploads(uint32_t slot);   // memcpy dirty scene vectors/params into slot buffers
    void flushSmokeUpload(uint32_t slot);   // memcpy dirty smoke state into the slot buffer
    // M9: record the staged vkCmdCopyBuffer copies (stage -> gpu twins) for
    // this slot plus the transfer availability barrier (HOST -> TRANSFER for
    // the staging source, then the copies). No-op unless copies are pending:
    // the first command buffer of the frame consumes them, later ones only
    // re-record the visibility barrier below. Must be called OUTSIDE a
    // render pass, before any draw that reads the slot.
    void recordSlotCopies(VkCommandBuffer cmd, uint32_t slot);
    // HOST writes -> VERTEX/FRAGMENT reads for this slot's params/smoke plus
    // TRANSFER writes -> VERTEX/FRAGMENT reads for the scene gpu twins copied
    // this frame; entries are recorded ONLY for payload written/copied this
    // frame (tracked by paramsFlushedFrame_/smokeFlushedFrame_/
    // sceneCopiedFrame_), NULL handles are skipped. Shared by prepareCull,
    // prepareShadowCascade and render (each command buffer/queue must record
    // its own: a barrier on the cull CB does not order the SDF CB's reads).
    // Steady-state frames record nothing here (old code emitted an
    // unconditional 8-entry VK_WHOLE_SIZE barrier up to 3x/frame/slot).
    void recordHostToShaderBarrier(VkCommandBuffer cmd, uint32_t slot) const;
    void writeSlotBinding(uint32_t slot, uint32_t binding, const Buffer& buf, VkDescriptorType type);
    void refreshDepthBinding(uint32_t slot); // rewrite binding 7 iff the view changed for this slot
    // Auto-loop bullet template (slot 0) from widget defaults (caller holds sceneMutex).
    void refreshAutoBulletLocked();
    // H4: recompute the CPU-side tracer sphere from the bullet slots + time
    // (mirrors smokeBulletState GLSL); caller holds sceneMutex. Marks the
    // smoke SSBO dirty only when the sphere/valid flag changed.
    void refreshTracerLocked();
    void markSmokeSSBO(); // flag all smoke slots dirty (tuning/bullet change, no scene rebuild)
    // Single-flight policy: at most one bullet at a time. A live manual
    // round suppresses the auto bullet until it dies (wake grace included).
    bool anyManualBulletLiveLocked() const;
    void syncAutoBulletLocked();
};
