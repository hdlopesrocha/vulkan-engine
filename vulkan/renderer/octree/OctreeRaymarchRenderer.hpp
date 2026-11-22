#pragma once

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../../utils/OctreeRaymarchTable.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Octree SDF raymarch terrain path (Settings::octreeRaymarchEnabled).
//
// Owns the set-2 descriptor layout (node table + layer roots + derived QEF
// payload + Phase 0 counters), the per-slot node/root/payload/counter buffers
// and the fullscreen raymarch
// pipeline. The
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

    // Publish this frame slot's host-side table upload and, when the slot
    // received a new revision, run the once-per-revision payload derive pass
    // (OctreeDerive.comp) before the march. MUST be recorded OUTSIDE any
    // dynamic rendering scope: HOST is not a framebuffer-space stage, and
    // vkCmdPipelineBarrier2 / compute dispatches inside
    // vkCmdBeginRendering/vkCmdEndRendering are illegal. Call immediately
    // before beginning the solid color pass.
    void recordDerive(VkCommandBuffer cmd, uint32_t frameIndex);

    // Make the host writes to this frame slot's mapped table buffers visible
    // to fragment shader reads. Kept for the fragment-only fallback path;
    // MyApp records recordDerive(), which calls this when no derive pass runs.
    void recordHostWriteBarrier(VkCommandBuffer cmd);

    // Record the fullscreen pass into the already-begun solid color+depth
    // scope. `width`/`height` are the solid target extent (dynamic viewport,
    // scissor and ray reconstruction).
    void draw(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex,
              uint32_t width, uint32_t height,
              VkDescriptorSet mainSet, VkDescriptorSet brushSet);

private:
    // Allocate this slot's node/root/payload buffers and point the slot's
    // descriptor set at them. Called from upload() on the main thread (after
    // the slot's fence wait) the first time a revision needs publishing.
    // Returns false on allocation failure; the caller keeps the raster path.
    bool ensureSlotBuffers(uint32_t fi);

    VulkanApp* app_ = nullptr;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, RAYMARCH_FRAMES> sets_{};
    std::array<Buffer, RAYMARCH_FRAMES> nodeBuffers_{};
    std::array<Buffer, RAYMARCH_FRAMES> rootBuffers_{};
    // Device-local, derived QEF position/normal per node (32 B). Written by
    // OctreeDerive.comp; read only by the hit ring. Never mapped by the host.
    std::array<Buffer, RAYMARCH_FRAMES> payloadBuffers_{};
    // Host-visible Phase 0 counters (OctreeRaymarchCounters). Reset by
    // recordDerive() after the slot fence, atomically incremented by the
    // OCTREE_PROFILE fragment variant while the push-constant gate is set.
    std::array<Buffer, RAYMARCH_FRAMES> counterBuffers_{};
    std::array<uint64_t, RAYMARCH_FRAMES> uploadedVersion_{};
    std::array<uint32_t, RAYMARCH_FRAMES> rootCounts_{};
    // Slot states for the payload derive pass: node count of the upload that
    // still needs its derived fields, and whether that pass is pending. Both
    // are per-slot because a revision is uploaded into each slot separately.
    std::array<uint32_t, RAYMARCH_FRAMES> nodeCounts_{};
    std::array<bool, RAYMARCH_FRAMES> needsDerive_{};

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    // Once-per-revision payload derive pass (OctreeDerive.comp): writes the
    // resolved material, 1/Lipschitz bound and the payload flag into the
    // record's dead bytes, and the QEF position/normal into payloadBuffers_.
    // Null when the module/pipeline was unavailable; the fragment march then
    // keeps its QEF / fixed-0.5 fallback.
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
