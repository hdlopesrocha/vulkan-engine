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

// Octree SDF raymarch terrain path (the ONLY solid/water renderer).
//
// Owns the set-2 descriptor layout (node table + layer roots + derived QEF
// payload), the revision-shared node/root/payload buffers and the fullscreen
// raymarch pipeline. The
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
// The table buffers are allocated on the first upload, so an empty scene pays
// no ~64 MB per-frame-slot reservation.
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
    // slots, allocated only once a table is published). The
    // default procedural scene's terrain tree (solid + water) is larger than
    // this, so the breadth-first walk stores the shallow levels whole and
    // prunes the deepest surface children; the parent's trilinear field keeps
    // those regions continuous instead of leaving holes. Larger scenes
    // degrade the same way.
    static constexpr size_t NODE_CAP = 524288;
    // One root per visible terrain layer (Solid + Water). Generous for a
    // dynamic layer list; extra layers beyond this are ignored.
    static constexpr size_t ROOT_CAP = 256;

    struct PushConstants {
        glm::vec4 viewport;         // xy = solid target extent, zw unused
        uint32_t rootCount = 0;     // roots uploaded for this frame slot
        uint32_t tableVersion = 0;  // low 32 bits of the table revision
        uint32_t nodeCount = 0;     // records uploaded for this frame slot
        uint32_t flags = 0;         // reserved
        // Shadow caster only: inverse of this cascade's light-space matrix.
        // The camera march ignores it. 96 B total keeps one push range for
        // both pipelines.
        glm::mat4 shadowInvLightSpace{1.0f};
    };
    static_assert(sizeof(PushConstants) == 96,
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

    // Record the raymarched terrain caster into one CSM cascade. Called by
    // ShadowRenderer after the rasterized casters while the cascade's dynamic
    // rendering scope is open. `mainSet` is the per-cascade shadow descriptor
    // (binding 0 carries the light-space UBO); `brushSet` is the frame's brush
    // params set (unused by the caster, present because the shared pipeline
    // layout has three sets). `invLightSpace` is the inverse of that cascade's
    // light-space matrix; the caster reconstructs orthographic light rays from
    // it. The shader reads only the node table (set 2), so no derive payload is
    // required and the pass can run before recordDerive().
    void recordShadowCascade(VkCommandBuffer cmd, uint32_t frameIndex,
                             VkDescriptorSet mainSet, VkDescriptorSet brushSet,
                             const glm::mat4& invLightSpace,
                             uint32_t shadowMapSize);
    // Host-write barrier for the node/root buffers before a shadow caster
    // reads them. Must be recorded outside dynamic rendering.
    void recordShadowHostBarrier(VkCommandBuffer cmd);

private:
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

    VulkanApp* app_ = nullptr;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, RAYMARCH_FRAMES> sets_{};
    std::vector<SharedRevision> revisions_;
    std::array<int, RAYMARCH_FRAMES> slotRevision_{-1, -1, -1};

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    // Raymarched terrain caster for the CSM cascades (shares pipelineLayout_).
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    // Once-per-revision payload derive pass (OctreeDerive.comp): writes the
    // resolved material, 1/Lipschitz bound and the payload flag into the
    // record's dead bytes, and the QEF position/normal into the shared
    // revision payload. Null when the module/pipeline was unavailable; the
    // fragment march then keeps its QEF / fixed-0.5 fallback.
    VkPipeline derivePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout deriveLayout_ = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    size_t maxNodes_ = 0;
};
