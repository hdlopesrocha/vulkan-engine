#pragma once

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../../utils/OctreeRaymarchTable.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// Octree SDF raymarch terrain path (Settings::octreeRaymarchEnabled).
//
// Owns the set-2 descriptor layout (node table + layer roots + derived QEF
// payload + Phase 0 counters), the revision-shared node/root/payload buffers,
// the per-slot counter buffers and the fullscreen raymarch pipeline. The
// table covers EVERY terrain layer (Solid + Water): the shader marches all
// layer roots and keeps the nearest hit, so solid and water surfaces occlude
// each other in one pass. Solid hits shade through the shared solid surface
// body (raster-style three-brush barycentric blending, tessellation
// displacement folded into the SDF, CSM/cloud shadows, optional inline RT
// reflections/contact shadows); water hits take the water-surface shading
// branch (layer tint + Fresnel + specular, opaque because the pass writes the
// solid color/depth targets).
// The table itself is built by the CPU (Scene::buildRaymarchTable) on a debounce
// and uploaded here only when its revision differs from the slot's revision.
// The table buffers are allocated on the first upload: the mode is off by
// default, so paying ~64 MB per frame slot at startup would be wasteful.
//
// Synchronization: upload() is called from MyApp::update(), which runs after
// the frame-slot fence wait (VulkanApp::drawFrame). The slot's previous GPU
// readers are therefore complete, so the host can safely rewrite the mapped
// buffers. draw()/recordDerive() publish those writes with a HOST barrier
// because the node buffer is persistently mapped and written with plain
// memcpy; a slot that received a new revision first runs the OctreeDerive.comp
// payload pass and barriers its writes to the fragment march.
class OctreeRaymarchRenderer : public Renderer {
public:
    static constexpr uint32_t RAYMARCH_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;

    // Hard CPU-side cap: 524288 nodes * 96 B = 48 MiB per frame slot, plus
    // 16 MiB of derived QEF payload (64 MiB per slot, 192 MiB for the three
    // slots, allocated only once the mode is enabled). The
    // default procedural scene's terrain tree (solid + water) is larger than
    // this, so the breadth-first walk stores the shallow levels whole and
    // prunes the deepest surface children; the parent's trilinear field keeps
    // those regions continuous instead of leaving holes. Larger scenes
    // degrade the same way.
    static constexpr size_t NODE_CAP = 524288;
    // One root per visible terrain layer (Solid + Water). Generous for a
    // dynamic layer list; extra layers beyond this are ignored (they still
    // rasterize when the fallback path is active).
    static constexpr size_t ROOT_CAP = 256;

    struct PushConstants {
        glm::vec4 viewport;         // xy = solid target extent, zw unused
        uint32_t rootCount = 0;     // roots uploaded for this frame slot
        uint32_t tableVersion = 0;  // low 32 bits of the table revision
        uint32_t nodeCount = 0;     // records uploaded for this frame slot
        uint32_t flags = 0;         // bit0: profile counters enabled
    };
    static_assert(sizeof(PushConstants) == 32,
                  "OctreeRaymarchRenderer::PushConstants must match OctreeRaymarchPush");

    OctreeRaymarchRenderer() = default;
    ~OctreeRaymarchRenderer() override = default;

    // Builds the set-2 layout/pool/sets, the fullscreen raymarch pipeline and
    // the lazy per-frame table buffers. Safe to call once; SceneRenderer owns
    // the lifetime.
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    // Maximum node count that fits the device's storage-buffer descriptor
    // range. MyApp passes this to Scene::buildRaymarchTable so every child
    // index in the table is addressable.
    size_t maxNodes() const { return maxNodes_; }

    // Publish `table` into the frame slot's buffers. Cheap when the slot
    // already holds this revision (the common case: rebuilds are debounced).
    // An actual upload marks the slot for recordDerive().
    void upload(uint32_t frameIndex, const OctreeRaymarchTable& table);

    // True when this frame slot is ready to march (pipeline + table buffers
    // created) and holds at least one layer root. When false the caller keeps
    // the raster path and its depth prepass.
    bool hasTable(uint32_t frameIndex) const;

    // Phase 0 instrumentation. The profile shader variants (OCTREE_PROFILE)
    // are only built when the device supports fragment stores/atomics; the
    // production variants never execute the counter atomics. readProfile()
    // snapshots the slot's last completed use after its fence.
    void setProfileCountersEnabled(bool enabled) {
        profileCountersEnabled_.store(enabled, std::memory_order_relaxed);
    }
    bool profileCountersSupported() const { return profileCountersSupported_; }
    void readProfile(uint32_t frameIndex);
    const OctreeRaymarchCounters& profileStats() const { return profileStats_; }

    // Runtime RT-shading selector (mirrors SolidRenderer::setRtShadingEnabled).
    // Binds the ray-query fragment variant while reflections / local shadows
    // are enabled; otherwise the cheap non-RT variant (procedural-sky fallback)
    // is used. No-op when the device lacks ray queries (the RT pipeline was
    // never created).
    void setRtShadingEnabled(bool enabled) { rtShadingEnabled_ = enabled; }

    // Phase 2: bind the no-fallback fragment twin when the derive pass is
    // available. The normal fallback pipelines remain the default.
    void setNoFallbackEnabled(bool enabled) {
        noFallbackEnabled_.store(enabled, std::memory_order_relaxed);
    }

    // Phase 4 research: parametric child selection/exit (push-constant bit 1).
    void setParametricChildEnabled(bool enabled) {
        parametricChildEnabled_.store(enabled, std::memory_order_relaxed);
    }

    // Publish the shared revision's host-side table/root upload and, when the
    // revision has not been derived yet, run the once-per-revision payload
    // derive pass (OctreeDerive.comp) before the march. MUST be recorded OUTSIDE any
    // dynamic rendering scope: HOST is not a framebuffer-space stage, and
    // vkCmdPipelineBarrier2 / compute dispatches inside
    // vkCmdBeginRendering/vkCmdEndRendering are illegal. Call immediately
    // before beginning the solid color pass.
    void recordDerive(VkCommandBuffer cmd, uint32_t frameIndex);

    // Make the host writes to the shared revision's mapped table/root buffers
    // visible to fragment shader reads. Kept for the fragment-only fallback
    // path; MyApp records recordDerive(), which calls this when no derive pass
    // runs.
    void recordHostWriteBarrier(VkCommandBuffer cmd);

    // Record the fullscreen pass into the already-begun solid color+depth
    // scope. `width`/`height` are the solid target extent (dynamic viewport,
    // scissor and ray reconstruction).
    void draw(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex,
              uint32_t width, uint32_t height,
              VkDescriptorSet mainSet, VkDescriptorSet brushSet);

private:
    // Allocate this slot's small, slot-local resources (the Phase 0 counter
    // buffer) and point binding 3 at it. The table, roots and payload are
    // revision-shared (below), so they are bound separately. Called from
    // upload() on the main thread after the slot's fence wait.
    bool ensureSlotBuffers(uint32_t fi);

    // One immutable table revision shared by every frame slot that displays
    // it. users counts the slot descriptors currently pointing here; when it
    // reaches zero the buffers are destroyed (the last user has passed its
    // fence). This is the F18 lifetime protocol.
    struct SharedRevision {
        Buffer node;
        Buffer root;
        Buffer payload;
        uint64_t version = 0;
        uint32_t nodeCount = 0;
        uint32_t rootCount = 0;
        uint32_t users = 0;
        bool derived = false;
    };
    bool bindSlotRevision(uint32_t fi, int revision);
    void releaseSlotRevision(uint32_t fi);
    int acquireRevision();
    void destroyRevisionBuffers(SharedRevision& revision);
    // Phase 2 no-fallback pipelines: same descriptor/push-constant layout as
    // the normal twins, but the shader module has the QEF/classifier fallbacks
    // compiled out. Created only when the derive pass is available.
    bool createNoFallbackPipelines(VulkanApp* app);
    bool noFallbackVariantActive() const;

    VulkanApp* app_ = nullptr;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, RAYMARCH_FRAMES> sets_{};
    std::vector<SharedRevision> revisions_;
    std::array<int, RAYMARCH_FRAMES> slotRevision_{-1, -1, -1};
    // Host-visible Phase 0 counters (OctreeRaymarchCounters). Reset by
    // recordDerive() after the slot fence, atomically incremented by the
    // OCTREE_PROFILE fragment variant while the push-constant gate is set.
    std::array<Buffer, RAYMARCH_FRAMES> counterBuffers_{};

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    // Once-per-revision payload derive pass (OctreeDerive.comp): writes the
    // resolved material, 1/Lipschitz bound and the payload flag into the
    // record's dead bytes, and the QEF position/normal into the shared
    // revision payload. Null when the module/pipeline was unavailable; the
    // fragment march then keeps its QEF / fixed-0.5 fallback.
    VkPipeline derivePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout deriveLayout_ = VK_NULL_HANDLE;
    // RT fragment variant (same march; shadeSolidSurface with inline ray
    // queries). Created only when the device supports ray queries.
    VkPipeline pipelineRt_ = VK_NULL_HANDLE;
    // Phase 0 profile variants (OCTREE_PROFILE). Same layout/march, with the
    // gated counter atomics. Null when the device lacks fragment atomics or
    // the profile modules are unavailable; the normal variants stay active.
    VkPipeline pipelineProfile_ = VK_NULL_HANDLE;
    VkPipeline pipelineProfileRt_ = VK_NULL_HANDLE;
    // Phase 2 no-fallback twins. Null when the derive pass is unavailable or
    // the modules are missing; the normal fallback twins stay active.
    VkPipeline pipelineNoFallback_ = VK_NULL_HANDLE;
    VkPipeline pipelineNoFallbackRt_ = VK_NULL_HANDLE;
    std::atomic<bool> noFallbackEnabled_{false};
    std::atomic<bool> parametricChildEnabled_{false};
    bool profileCountersSupported_ = false;
    std::atomic<bool> profileCountersEnabled_{false};
    OctreeRaymarchCounters profileStats_{};
    bool rtShadingEnabled_ = false;
    // True when the current RT selection has a matching profile pipeline and
    // the caller enabled counters; used to gate both the pipeline choice and
    // the push-constant flag.
    bool profileVariantActive() const;
    bool rtPipelineActive() const {
        return rtShadingEnabled_ && pipelineRt_ != VK_NULL_HANDLE;
    }
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    size_t maxNodes_ = 0;
};
