#pragma once

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../../utils/OctreeRaymarchTable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Octree SDF raymarch solid path (Settings::octreeRaymarchEnabled).
//
// Owns the set-2 descriptor layout (node table + layer roots), one host-visible
// buffer pair per frame in flight and the fullscreen raymarch pipeline. The
// shader shades through the shared solid surface body, with raster-style
// three-brush barycentric blending (the local Surface-Nets triangle is
// reconstructed at the hit), tessellation displacement folded into the marched
// SDF, CSM/cloud shadows and optional inline RT reflections/contact shadows
// (the RT fragment twin is bound while Settings::rtReflections / rtLocalShadows
// are on).
// The table itself is built by the CPU (Scene::buildRaymarchTable) on a debounce
// and uploaded here only when its revision differs from the slot's revision.
// The table buffers are allocated on the first upload: the mode is off by
// default, so paying ~48 MB per frame slot at startup would be wasteful.
//
// Synchronization: upload() is called from MyApp::update(), which runs after
// the frame-slot fence wait (VulkanApp::drawFrame). The slot's previous GPU
// readers are therefore complete, so the host can safely rewrite the mapped
// buffers. draw() records a HOST -> FRAGMENT_SHADER barrier because the node
// buffer is persistently mapped and written with plain memcpy.
class OctreeRaymarchRenderer : public Renderer {
public:
    static constexpr uint32_t RAYMARCH_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;

    // Hard CPU-side cap: 393216 nodes * 128 B = 48 MB per frame slot. The
    // default procedural scene's complete surface tree is ~358k nodes (the
    // terrain's 30-unit cells dominate; the test objects add depths 10-12), so
    // this stores it whole as long as the device range allows. Larger scenes
    // still degrade uniformly: breadth-first fills the shallow levels first
    // and the cap only prunes the deepest surface children.
    static constexpr size_t NODE_CAP = 393216;
    // One root per visible Solid layer. Generous for a dynamic layer list;
    // extra layers beyond this are ignored (they still rasterize when the
    // fallback path is active).
    static constexpr size_t ROOT_CAP = 256;

    struct PushConstants {
        glm::vec4 viewport;      // xy = solid target extent, zw unused
        uint32_t rootCount = 0;  // roots uploaded for this frame slot
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
        uint32_t pad2 = 0;
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
    void upload(uint32_t frameIndex, const OctreeRaymarchTable& table);

    // True when this frame slot is ready to march (pipeline + table buffers
    // created) and holds at least one layer root. When false the caller keeps
    // the raster path and its depth prepass.
    bool hasTable(uint32_t frameIndex) const;

    // Runtime RT-shading selector (mirrors SolidRenderer::setRtShadingEnabled).
    // Binds the ray-query fragment variant while reflections / local shadows
    // are enabled; otherwise the cheap non-RT variant (procedural-sky fallback)
    // is used. No-op when the device lacks ray queries (the RT pipeline was
    // never created).
    void setRtShadingEnabled(bool enabled) { rtShadingEnabled_ = enabled; }

    // Make the host writes to this frame slot's mapped table buffers visible
    // to fragment shader reads. MUST be recorded OUTSIDE any dynamic rendering
    // scope: HOST is not a framebuffer-space stage, and vkCmdPipelineBarrier2
    // inside vkCmdBeginRendering/vkCmdEndRendering may only use framebuffer
    // stages. Call immediately before beginning the solid color pass.
    void recordHostWriteBarrier(VkCommandBuffer cmd);

    // Record the fullscreen pass into the already-begun solid color+depth
    // scope. `width`/`height` are the solid target extent (dynamic viewport,
    // scissor and ray reconstruction).
    void draw(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex,
              uint32_t width, uint32_t height,
              VkDescriptorSet mainSet, VkDescriptorSet brushSet);

private:
    // Allocate this slot's node/root buffers and point the slot's descriptor
    // set at them. Called from upload() on the main thread (after the slot's
    // fence wait) the first time a revision needs publishing. Returns false on
    // allocation failure; the caller keeps the raster path.
    bool ensureSlotBuffers(uint32_t fi);

    VulkanApp* app_ = nullptr;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, RAYMARCH_FRAMES> sets_{};
    std::array<Buffer, RAYMARCH_FRAMES> nodeBuffers_{};
    std::array<Buffer, RAYMARCH_FRAMES> rootBuffers_{};
    std::array<uint64_t, RAYMARCH_FRAMES> uploadedVersion_{};
    std::array<uint32_t, RAYMARCH_FRAMES> rootCounts_{};

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    // RT fragment variant (same march; shadeSolidSurface with inline ray
    // queries). Created only when the device supports ray queries.
    VkPipeline pipelineRt_ = VK_NULL_HANDLE;
    bool rtShadingEnabled_ = false;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    size_t maxNodes_ = 0;
};
