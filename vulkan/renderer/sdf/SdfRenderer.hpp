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
#include "sdf/types/SdfGrassAnchor.hpp"
#include "sdf/types/SdfGridCell.hpp"
#include "vulkan/ubo/SdfParamsUBO.hpp"
#include "sdf/types/SmokeFragBullet.hpp"

// ─── GPU types (canonical std430 contract, one file per struct) ────────────
// SdfDefinition / SdfInstance / SdfMaterial / SdfContainer /
// SdfGridCell / SdfParamsUBO / SmokeFragBullet define the contract
// consumed by shaders/SdfRenderer.vert(.frag) at set=1 bindings 0..9
// (9 = rasterized water-surface depth for the solid+water march clamp).

// Push constants for the grass-shadow pipeline (112 B; GLSL twin is the
// push_constant block in shaders/renderer/shadow/SdfGrassShadow.{vert,frag}).
// The block carries everything the grass-only EVSM caster needs without
// touching set 0: the cascade matrix, the SDF clock + march budget and the
// light-to-scene march direction.
struct SdfGrassShadowPC {
    glm::mat4 lightViewProj; // cascade light view-projection (world -> light clip)
    glm::vec4 params;        // x = time (s), y = max steps, z = epsilon, w = safety
    glm::vec4 march;         // x = max step (m), y = min step (m), z = shadow LOD camScale
                             // (>= 80 -> impostor-only grass LOD, zero blades)
    glm::vec4 lightDir;      // xyz = light-to-scene direction (world), w unused
};
static_assert(sizeof(SdfGrassShadowPC) == 112, "SdfGrassShadowPC must be 112 bytes");

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

    // ── Rock boulders (brushIndex 7) ─────────────────────────────────────
    // Rock anchors are derived from the SOLID shape of brush-7 terrain
    // chunks, keyed by chunk so edits replace and deletions remove their
    // boulders (same collector contract as lava). One rock per
    // config().rocks.spacing x spacing m of rock surface by default
    // (512 x 512 m); the rock scene merges with lava + smoke on every
    // rebuild.
    static constexpr int kRockBrushIndex = 7;
    void ingestRockChunk(uintptr_t nid, const Geometry& geom);
    void removeRockChunk(uintptr_t nid);
    void clearRocks();
    // Rebuild the rock scene from collected anchors when dirty (coalesces
    // per-chunk ingests into at most one flatten per frame). Returns true
    // when a rebuild was staged.
    bool rebuildRocksIfDirty();
    // Placement: one rock per `spacing` x `spacing` m of rock surface
    // (0 / disabled hides the boulders; candidates are retained at
    // minSpacing and decimated per rebuild, so slider edits are instant).
    void setRocksEnabled(bool on);
    void setRockSpacing(float m);
    void setRockMaxPerChunk(int n);
    // Force a rock-scene rebuild from the retained candidates (e.g. after
    // external state changes).
    void markRocksDirty();
    // Size / shape (applied at scene rebuild).
    void setRockScale(float m);
    void setRockScaleVariation(float f);
    void setRockEmbed(float f);
    void setRockNoiseScale(float s);
    void setRockNoiseAmplitude(float a);
    // Surface (applied at scene rebuild).
    void setRockTextureLayer(float layer);
    void setRockTextureTiling(float m);
    void setRockRoughness(float r);
    void setRockMetallic(float m);
    void setRockTint(const glm::vec3& rgb);

    // ── Grass clumps (existing vegetation instances) ─────────────────────
    // Grass anchors are streamed 1:1 by VegetationRenderer as chunks publish
    // (position + vegetation type/biome + smooth surface normal). Each
    // retained anchor becomes ONE SdfInstance of the procedural Grass
    // primitive, which expands into many blades in-shader — the existing
    // vegetation data stays the single source of truth and no second grass
    // placement pass exists. The scene merges with lava + rocks + smoke.
    void ingestGrassChunk(uintptr_t nid, std::vector<sdf_gpu::GrassAnchor> anchors);
    void removeGrassChunk(uintptr_t nid);
    void clearGrass();
    // Rebuild the grass scene from collected anchors when dirty (coalesces
    // per-chunk ingests into at most one flatten per frame). Returns true
    // when a rebuild was staged.
    bool rebuildGrassIfDirty();
    // Force a grass-scene rebuild from the retained anchors.
    void markGrassDirty();
    // Placement/collector controls.
    void setGrassEnabled(bool on);
    void setGrassMaxPerChunk(int n);
    // Shape (applied at scene rebuild): clump radius/blade height/width in
    // local units (1 = the per-instance vegetation scale), blade count,
    // curvature, wind lean/gain, tip width, roughness and tint.
    void setGrassClumpRadius(float r);
    void setGrassBladeHeight(float h);
    void setGrassBladeWidth(float w);
    void setGrassBladeCount(int n);
    void setGrassCurvature(float c);
    void setGrassMaxLean(float rad);
    void setGrassWindGain(float g);
    void setGrassTipWidth(float f);
    void setGrassRoughness(float r);
    void setGrassTint(const glm::vec3& rgb);
    // Shadow-caster LOD camera scale (no scene rebuild: read per shadow frame).
    void setGrassShadowLodScale(float s);

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
    void setFrame(uint32_t frame) { currentFrame_ = frame % SDF_FRAMES; }
    void setRenderMode(RenderMode mode);
    void setDebugFlags(uint32_t flags);
    void setMarchParams(float maxSteps, float epsilon);
    void setMarchRange(float minStep, float maxStep, float earlyTermThreshold);
    // External scene depth (main depth buffer view) used for occlusion.
    // May be called every frame; the descriptor is rewritten only when the
    // view handle actually changes (per frame slot), never blindly per frame.
    void setSceneDepth(VkImageView view, VkImageLayout layout);
    // Rasterized water-surface geometry depth (WaterRenderer::
    // getWaterGeomDepthView) used to clamp the march exit against water.
    // `enabled` gates the shader side (SdfParamsUBO::waterDepthEnabled) and
    // must be false when the water pass did not write a valid geometry depth
    // this frame (Minimal/aux-less water variant, water disabled).
    void setWaterDepth(VkImageView view, VkImageLayout layout, bool enabled);

    // Host->SSBO upload (pending memcpys) + Sync2 visibility barrier for this
    // frame's slot. Call OUTSIDE a render pass, before render().
    void prepareCull(VkCommandBuffer cmd);

    // ── Grass shadow casting into the EVSM cascades ──────────────────────
    // Shadow-pass interface. prepareShadowCascade() is called from each
    // cascade CB BEFORE beginShadowPass starts the dynamic rendering scope
    // (buffer barriers are not allowed inside a vkCmdBeginRendering scope):
    // the host flush is dirty-gated, so the slot upload runs exactly once per
    // frame, while every cascade CB records its own HOST->shader barrier. The
    // later SDF task then only re-records barriers and never rewrites the
    // slot while a cascade CB may read it. drawShadowCascade() records the
    // grass-only proxy draw for one cascade (set 1 = sdfSets[frameIdx] +
    // push constants; set 0 is left untouched). Only SDF_PRIM_GRASS
    // definitions ever write moments; fire/smoke/rocks have no path into the
    // EVSM output.
    void prepareShadowCascade(VkCommandBuffer cmd, uint32_t frameIdx);
    void drawShadowCascade(VkCommandBuffer cmd, uint32_t cascadeIndex,
                           const glm::mat4& lightViewProj, float time);
    // March direction for the grass shadow shader (the shadow pass owns the
    // light UBO; the SDF shadow pipeline deliberately does not bind set 0).
    void setShadowLightDirection(const glm::vec3& dir);
    // Global SDF clock (last updateParams time). The shadow task runs before
    // the SDF task, so this is one frame stale — fine for the wind phase.
    float getTime() const;

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

    // Grass-only EVSM caster for the shadow pass (SdfGrassShadow.*). Null
    // until its SPIR-V exists (same lazy tolerance as createPipeline); the
    // shadow draw no-ops while null.
    TrackedHandle<VkPipeline> shadowPipeline;
    TrackedHandle<VkPipelineLayout> shadowPipelineLayout;
    TrackedHandle<VkShaderModule> shadowVertModule;
    TrackedHandle<VkShaderModule> shadowFragModule;
    glm::vec3 shadowLightDir_ = glm::vec3(0.0f);

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

    // CPU scene state + dirty flags (guarded; setScene may come from the
    // load path). The merged scene stores the canonical GPU-layout structs,
    // so the upload memcpys its vectors verbatim (no CPU->GPU conversion).
    mutable std::mutex sceneMutex;
    sdf_gpu::SdfScene pendingScene_; // merged lava + rocks + grass + smoke (uploaded directly)
    sdf_gpu::SdfScene lavaScene_;    // fire scene from lava anchors
    sdf_gpu::SdfScene smokeScene_;   // static-topology smoke/flame shape scene
    sdf_gpu::SdfScene rocksScene_;   // rock scene from brush-7 anchors
    sdf_gpu::SdfScene grassScene_;   // grass scene from existing vegetation instances
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
    // Rock-candidate collection (brush-7 chunk geometry). Candidates are
    // ingested at config().rocks.minSpacing (the densest retained set) with
    // per-candidate variation keys; rebuildRocksIfDirty decimates them to
    // the live spacing and applies the live shape/material, so widget edits
    // never need the chunk geometry again.
    struct RockCandidate {
        glm::vec3 pos{0.0f};              // surface point (world)
        glm::vec3 normal{0.0f, 1.0f, 0.0f}; // smooth surface normal
        glm::vec3 euler{0.0f};            // random tumble (R = Rx*Ry*Rz)
        float scaleVar = 0.0f;            // [0,1) size variation key
        float seed = 0.0f;                // per-rock noise seed
        float decim = 0.0f;               // [0,1) density key (keep while < ratio)
    };
    std::unordered_map<uintptr_t, std::vector<RockCandidate>> rocksByChunk_;
    // Grass-clump collection (existing vegetation instances, keyed by the
    // vegetation chunk). Anchors are stored verbatim as the conversion
    // boundary into the generic scene; the vegetation stream stays the
    // source of truth and re-publishing a chunk replaces its clumps.
    std::unordered_map<uintptr_t, std::vector<sdf_gpu::GrassAnchor>> grassByChunk_;
    // Shared tuning (single source of truth for renderer + widgets).
    SdfEffectConfig config_;
    bool lavaDirty_ = false;    // anchors changed -> rebuild staged
    bool rocksDirty_ = false;   // anchors/shape changed -> rebuild staged
    bool grassDirty_ = false;   // vegetation clumps/shape changed -> rebuild staged

    // Smoke bomb state (second generic consumer). Scene topology is static
    // (1 def/mat/container/instance); all behavior below streams through
    // the smoke SSBO without scene rebuilds, except position/radius/base
    // material which reshape the scene.
    SmokeFragBullet smokeState_ = {}; // SSBO mirror: tuning + bullets
    // Manual rounds always use slot 1 (slot 0 is the auto-loop template);
    // firing clears any previous manual round: single-flight, one at a time.

    VkImageView pendingDepthView_ = VK_NULL_HANDLE;
    VkImageLayout pendingDepthLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    std::array<VkImageView, SDF_FRAMES> boundDepthViews_{};
    // Water-surface depth binding (set=1 binding 9): same per-slot dedupe.
    // The shader only samples it while params_.waterDepthEnabled > 0.5.
    VkImageView pendingWaterDepthView_ = VK_NULL_HANDLE;
    VkImageLayout pendingWaterDepthLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bool pendingWaterDepthEnabled_ = false;
    std::array<VkImageView, SDF_FRAMES> boundWaterDepthViews_{};

    SdfStats stats_;
    VulkanApp* app_ = nullptr; // stashed for buffer (re)allocation

    // Container range of the GRASS scene inside the merged scene (computed in
    // refreshMergedLocked from the merge order lava -> rocks -> grass ->
    // smoke). The shadow pass rasterizes exactly this range so fire/smoke/
    // rock containers are never drawn.
    uint32_t grassContainerBase_ = 0;
    uint32_t grassContainerCount_ = 0;

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
    void createShadowPipeline(VulkanApp* app);
    // Re-merge lava + rock + smoke scenes into pendingScene_, rebuild its
    // bounds and grids (caller holds sceneMutex). Marks all scene slots dirty.
    void refreshMergedLocked();
    // ── SdfModel helpers (one transform convention for every effect) ─────
    // Euler XYZ radians, R = Rx * Ry * Rz (mirrors sdfEulerMat in sdf_ops).
    // Caller holds sceneMutex.
    glm::mat3 smokeRotLocked() const;

    // Refresh params_.renderMode/debugFlags from the renderer state.
    void repackDebugMode();
    void ensureSlotCapacity(uint32_t slot); // grow slot buffers with headroom; rewrites slot set bindings
    void flushSlotUploads(uint32_t slot);   // memcpy dirty scene vectors/params into slot buffers
    void flushSmokeUpload(uint32_t slot);   // memcpy dirty smoke state into the slot buffer
    // HOST writes -> VERTEX/FRAGMENT reads for this slot's scene buffers
    // (instances/definitions/materials/containers/grids/params/smoke); NULL
    // handles are skipped. Shared by prepareCull, prepareShadowCascade and
    // render (each command buffer/queue must record its own).
    void recordHostToShaderBarrier(VkCommandBuffer cmd, uint32_t slot) const;
    void writeSlotBinding(uint32_t slot, uint32_t binding, const Buffer& buf, VkDescriptorType type);
    void refreshDepthBinding(uint32_t slot); // rewrite binding 7 iff the view changed for this slot
    // Auto-loop bullet template (slot 0) from widget defaults (caller holds sceneMutex).
    void refreshAutoBulletLocked();
    void markSmokeSSBO(); // flag all smoke slots dirty (tuning/bullet change, no scene rebuild)
    // Single-flight policy: at most one bullet at a time. A live manual
    // round suppresses the auto bullet until it dies (wake grace included).
    bool anyManualBulletLiveLocked() const;
    void syncAutoBulletLocked();
};
