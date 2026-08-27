#pragma once

#include "Renderer.hpp"
#include "../Buffer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../../math/BoundingCube.hpp"
#include "../../space/Octree.hpp" // for NodeID, Octree, OctreeNode
#include "../../space/OctreeNodeData.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "CommandBufferState.hpp"

class IndirectRenderer; // forward declaration: the solid IndirectRenderer owns the merged SDF cull dispatch

// Renders leaf-node cube faces colored by their SDF sign and magnitude.
class DebugSDFRenderer : public Renderer {
public:
    struct CubeSDF {
        BoundingCube cube;
        std::array<float, 8> sdf;
        int brushIndex;
    };

    DebugSDFRenderer();
    ~DebugSDFRenderer();

    void init(VulkanApp* app);
    void setCubes(const std::vector<CubeSDF>& cubes);
    // The solid IndirectRenderer performs the SDF cube frustum cull + compaction in
    // its OWN indirect.comp dispatch (folded into the terrain cull). Point this
    // renderer at it so render() can draw from the terrain IR's SDF output buffers.
    void setIndirectRenderer(IndirectRenderer* ir) { terrainIR_ = ir; }
    // GPU frustum-cull the SDF cubes via indirect.comp (same culling the solid
    // pass uses). Call OUTSIDE a render pass, before the draw. `frame` must
    // match the value passed to setCullFrame so cull + draw share one slot.
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj,
                     glm::vec3 camPos = glm::vec3(0.0f), float lodBias = 8.0f, int maxTargetLod = 16);
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet);
    void setCullFrame(uint32_t frame) { currentCullFrame = frame % SDF_CULL_FRAMES; }
    void cleanup(VulkanApp* app) override;

    // ── Per-chunk SDF cube tracking (moved from SceneRenderer) ──
    // Populated by processNodeLayer via scene.requestSDFCubes (worker thread)
    // after chunk (re)builds; consumed as a copy on the render thread.
    void updateCubesForChunk(NodeID nid, const std::vector<CubeSDF>& cubes);
    void removeCubesForNode(NodeID id);
    void clearCubes();
    std::vector<CubeSDF> getCubes() const;

private:
    // Per-node SDF face cubes keyed by octree node id
    std::unordered_map<NodeID, std::vector<CubeSDF>> nodeDebugSDFCubes;
    mutable std::recursive_mutex cubesMutex;

    struct CubeVertex {
        glm::vec3 position;
        uint32_t cornerIndex;
    };

    // Per-cube GPU instance payload (matches debug_sdf.vert InstanceData).
    struct InstanceData {
        glm::mat4 model;
        glm::vec4 sdf0;
        glm::vec4 sdf1;
        glm::vec4 meta; // meta.x = brushIndex
    };

    static constexpr uint32_t SDF_CULL_FRAMES = 3;

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkShaderModule> vertModule;
    TrackedHandle<VkShaderModule> fragModule;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;

    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    TrackedHandle<VkDescriptorSet> descriptorSet;

    std::vector<CubeSDF> activeCubes;

    // ── Per-frame instance payload (read by the SDF vertex shader) ──
    // One set of buffers per cull frame (rotated) so an in-flight draw on the
    // same slot (3 frames in flight, FIFO) never races the host rewrite. The
    // visible DrawCmd stream + count live in the terrain IndirectRenderer's SDF
    // output buffers (folded into the terrain cull); only the per-cube instance
    // transform/value payload is stored here.
    struct CullFrame {
        Buffer instance;      // InstanceData per cube, indexed by gl_InstanceIndex
        uint32_t capacity = 0; // entries capacity
    };
    std::array<CullFrame, SDF_CULL_FRAMES> cullFrames;
    uint32_t currentCullFrame = 0;
    bool hasCubes_ = false;

    PFN_vkCmdDrawIndexedIndirectCountKHR cmdDrawIndexedIndirectCount = nullptr;
    VulkanApp* cullApp_ = nullptr; // stashed for buffer (re)allocation
    IndirectRenderer* terrainIR_ = nullptr; // owns the merged SDF cull dispatch

private:
    void createCubeBuffers(VulkanApp* app);
    void createDescriptorSet(VulkanApp* app);
    void createCullResources(VulkanApp* app);
    void ensureCullCapacity(uint32_t frame, uint32_t cubeCount);
    void writeCullFrameData(uint32_t frame, uint32_t cubeCount);
};
