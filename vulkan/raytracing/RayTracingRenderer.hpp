#pragma once

#include "../VulkanApp.hpp"
#include "../renderer/Renderer.hpp"
#include "RayTracingContext.hpp"
#include "AccelerationStructure.hpp"
#include "ShaderBindingTable.hpp"
#include <vulkan/vulkan.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cstdint>
#include <mutex>

class VegetationRenderer; // fwd decl to avoid a heavy include in this header

// Ray-tracing workload identifiers. Each maps to its own ray-tracing pipeline
// and Shader Binding Table. The distinction between Shadow / Reflection /
// Refraction is a RAY/WORKLOAD distinction (different initial rays, different
// shading), NOT a separate acceleration structure — all three trace the SAME
// unified TLAS. See RayTracingRenderer for the SBT group layout.
enum class RtWorkload : uint32_t {
    Shadow = 0,
    Reflection = 1,
    Refraction = 2,
    Render = 3,   // primary ray-traced scene render (replaces raster solid/veg colour)
    Count = 4
};

// Geometry/material kind. Drives which HIT GROUP a TLAS instance routes to (via
// instanceShaderBindingTableRecordOffset) so solid / water / vegetation can
// exhibit different ray-hit behavior without separate passes or AS hierarchies.
enum class GeometryKind : uint32_t {
    Solid = 0,
    Water = 1,
    Vegetation = 2
};

// Per-instance data for a vegetation billboard. Vegetation is registered as a
// SINGLE shared BLAS (the billboard cross-quad) referenced by MANY TLAS
// instances, each carrying its own world transform + foliage layer (so the
// any-hit can alpha-test against the correct array layer).
struct VegInstance {
    VkTransformMatrixKHR transform{};
    uint32_t customIndex = 0; // low byte = kind, next byte = foliage layer
    uint32_t sbtOffset = static_cast<uint32_t>(GeometryKind::Vegetation);
    VkGeometryInstanceFlagsKHR flags = 0;
};

// Owns the ray-tracing scene representation:
//   chunk geometry → per-chunk BLAS (rebuilt only when that chunk changes)
//                 → unified TLAS (rebuilt only when chunks are added/removed)
//                 → ray-tracing workloads (shadow / reflection / refraction)
//
// Reuses the project's existing abstractions: chunk lifecycle is driven by the
// SceneRenderer (which already knows each chunk's solid/water/vegetation kind and
// its geometry addresses from the slotted IndirectRenderer); GPU resource
// lifetime is managed through VulkanApp::deferDestroyUntilFence /
// deferDestroyUntilAllPending; buffers are allocated via VmaContext through
// RayTracingContext; synchronization reuses the app's async submit helpers.
class RayTracingRenderer : public Renderer {
public:
    RayTracingRenderer() = default;
    ~RayTracingRenderer() override = default;

    // Build RT descriptor-set layout, the three pipelines + SBTs, and the
    // unified TLAS. No-op (graceful fallback) when rtSupport.any() is false.
    void init(VulkanApp* app, VkImageView shadowOutputView);

    void cleanup(VulkanApp* app) override;

    bool supported() const { return ctx_.supported(); }

    // ── Chunk registration (called by SceneRenderer on chunk swap) ──
    // Registers (or replaces) the BLAS for a single chunk. `vertexAddress` /
    // `indexAddress` are absolute device addresses of the chunk's FIRST
    // vertex/index (computed from the slotted pool base + per-chunk offset).
    // The BLAS build is deferred to the next update() and sequenced after the
    // chunk's geometry upload via the app's async submit + extra-wait semaphore.
    void registerChunk(uint64_t chunkId, GeometryKind kind,
                       VkDeviceAddress vertexAddress, uint32_t vertexCount,
                       VkDeviceAddress indexAddress, uint32_t indexCount,
                       VkGeometryFlagsKHR geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR);

    // Remove a chunk's BLAS (deferred until GPU idle relative to it).
    void unregisterChunk(uint64_t chunkId);

    // Register vegetation billboards as alpha-tested instances in the unified
    // TLAS. Reads the consolidated billboard instance buffer (GPU→CPU readback),
    // builds ONE shared cross-quad BLAS, and emits one transformed TLAS instance
    // per billboard. No-op (graceful) when RT unsupported or vegetation absent.
    void registerVegetation(VulkanApp* app, const class VegetationRenderer* veg);

    // Bind the vegetation leaf-opacity texture (sampler2DArray) used by the
    // any-hit alpha test into the RT descriptor set (binding 14).
    void setVegetationOpacity(VulkanApp* app, VkImageView view, VkSampler sampler);

    // Rebuild only the TLAS (call after adding/removing chunks). The unified
    // TLAS is NOT rebuilt every frame — only when the instance set changes.
    void markTlasDirty() { tlasDirty_ = true; }

    // Process pending BLAS builds + (if needed) the TLAS rebuild. Call once per
    // frame from the main thread, after chunk swaps have been drained.
    void update(VulkanApp* app);

    // Record a shadow-ray trace into `shadowOutputView` (a storage image).
    // Lightweight: minimal payload, only occlusion is written.
    void traceShadow(VkCommandBuffer cmd, VkImageView shadowOutputView);

    // Record a reflection-ray trace (full shading via hit shaders).
    void traceReflection(VkCommandBuffer cmd, VkImageView outputView);
    // Record a refraction/transmission-ray trace.
    void traceRefraction(VkCommandBuffer cmd, VkImageView outputView);

    // Primary ray-traced scene render (migrated replacement for the raster
    // solid/vegetation colour pass). Writes shaded colour into outputView.
    void traceRender(VkCommandBuffer cmd, VkImageView outputView);

    // TLAS device address (for descriptor set binding / debugging).
    VkDeviceAddress tlasAddress() const { return tlas_ ? tlas_->deviceAddress() : 0; }

private:
    struct ChunkEntry {
        std::unique_ptr<AccelerationStructure> blas;
        GeometryKind kind = GeometryKind::Solid;
        bool pending = true; // BLAS build queued, not yet submitted
        // geometry inputs captured at registration time
        BlasGeometryInput geom{};
    };

    struct PipelineWorkload {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        ShaderBindingTable sbt;
        std::vector<VkShaderModule> modules;
    };

    RayTracingContext ctx_;
    bool inited_ = false;

    VkDescriptorSetLayout rtDescLayout_ = VK_NULL_HANDLE;   // set=0 (mirrors main + TLAS)
    VkDescriptorSet rtDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorPool rtPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout rtOutputLayout_ = VK_NULL_HANDLE;  // set=1 (output storage image)
    VkDescriptorSet rtOutputSets_[static_cast<uint32_t>(RtWorkload::Count)] = {};
    VkImageView shadowOutputView_ = VK_NULL_HANDLE;

    PipelineWorkload workloads_[static_cast<uint32_t>(RtWorkload::Count)];

    std::unique_ptr<AccelerationStructure> tlas_;
    std::unordered_map<uint64_t, ChunkEntry> chunks_;
    std::mutex chunksMutex_;

    // Vegetation: one shared BLAS (billboard cross-quad) + many transformed TLAS
    // instances (one per billboard). Built lazily from the VegetationRenderer.
    std::unique_ptr<AccelerationStructure> vegBlas_;
    std::vector<VegInstance> vegInstances_;
    bool vegBlasDirty_ = false;
    bool vegOpacityBound_ = false;
    uint64_t vegLastGeneration_ = 0;
    VkDeviceAddress vegLastVertexAddress_ = 0;
    VkDeviceAddress vegLastIndexAddress_ = 0;
    VkDeviceAddress vegVertexAddress_ = 0;
    VkDeviceAddress vegIndexAddress_ = 0;
    uint32_t vegVertexCount_ = 0;
    uint32_t vegIndexCount_ = 0;
    uint32_t vegVertexStride_ = 64; // sizeof(Vertex)
    VkIndexType vegIndexType_ = VK_INDEX_TYPE_UINT32;

    bool tlasDirty_ = true;

    // Helpers
    void createRtDescriptorSetLayout(VulkanApp* app);
    void createPipelines(VulkanApp* app);
    void buildTlas(VulkanApp* app, VkCommandBuffer cmd);
    void traceWorkload(VkCommandBuffer cmd, RtWorkload w, uint32_t rayGenIndex, VkImageView outputView);
};
