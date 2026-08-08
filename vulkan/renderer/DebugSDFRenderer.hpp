#pragma once

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

// Renders leaf-node cube faces colored by their SDF sign and magnitude.
class DebugSDFRenderer {
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
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet);
    void cleanup();

    // ── Per-chunk SDF cube tracking (moved from SceneRenderer) ──
    // Populated by the space-change handlers (worker threads) after chunk
    // (re)builds; consumed as a copy on the render thread.
    void updateCubesForChunk(NodeID nid, const OctreeNodeData& nd, const Octree& tree);
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
    Buffer instanceBuffer;
    uint32_t instanceBufferCapacity = 0;

    std::vector<CubeSDF> activeCubes;
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
private:
    void createCubeBuffers(VulkanApp* app);
    void createDescriptorSet(VulkanApp* app);
    void updateInstanceBuffer(VulkanApp* app);
};
