#pragma once

#include "VulkanApp.hpp"
#include "../math/Geometry.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>

// Manages a single large vertex/index/indirect buffer and provides a simple
// CPU-side allocator for adding/removing meshes. Draws are performed via
// vkCmdDrawIndexedIndirect (one indirect command per mesh). The allocator is
// append-first and supports reclamation on remove (simple free list rebuild).
class IndirectRenderer {
public:
    struct MeshInfo {
        uint32_t id = UINT32_MAX;
        uint32_t baseVertex = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        glm::mat4 model = glm::mat4(1.0f);
        glm::vec4 boundsMin = glm::vec4(0.0f); // object-space AABB min (xyz)
        glm::vec4 boundsMax = glm::vec4(0.0f); // object-space AABB max (xyz)
        VkDeviceSize indirectOffset = 0; // byte offset into indirect buffer
        bool active = false;
        // NOTE: per-mesh buffers removed — meshes are packed into the merged buffers
    };

    IndirectRenderer();
    ~IndirectRenderer();

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    // Add mesh and return mesh id. Model matrix is stored per-mesh and uploaded
    // to a small CPU-side array used for push constants (we still push per-draw).
    uint32_t addMesh(VulkanApp* app, const Geometry& mesh, const glm::mat4& model);
    void removeMesh(uint32_t meshId);

    // Rebuild GPU backing buffers from current CPU mesh list. Call before drawing
    // if add/remove operations occurred.
    void rebuild(VulkanApp* app);

    // Bind buffers and issue an indirect draw for the given mesh id.
    // This keeps per-draw push constants outside and only performs the indirect call.
    void drawIndirect(VkCommandBuffer cmd, uint32_t meshId);

    // Bind merged vertex/index buffers once and draw all provided mesh ids.
    // This avoids binding per-mesh buffers; push constants must be set per-draw
    // by this function (it will push each mesh's model before issuing its draw).
    void drawVisibleMerged(VkCommandBuffer cmd, const std::vector<uint32_t>& visibleMeshIds, VulkanApp* app);
    void drawMergedWithCull(VkCommandBuffer cmd, const glm::mat4& viewProj, VulkanApp* app, uint32_t maxDraws = 0);
    // Run GPU culling/compaction (must be called outside any render pass).
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws = 0);
    // Issue indirect draw using the compacted indirect buffer (call inside render pass).
    void drawPrepared(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws = 0);

    // Update a descriptor set to point to the GPU-side models SSBO.
    void updateModelsDescriptorSet(VulkanApp* app, VkDescriptorSet ds);
    void setModelsDescriptorSet(VkDescriptorSet ds) { modelsDescriptorSet = ds; }

    // Accessors
    const Buffer& getVertexBuffer() const { return vertexBuffer; }
    const Buffer& getIndexBuffer() const { return indexBuffer; }
    const Buffer& getIndirectBuffer() const { return indirectBuffer; }
    const Buffer& getModelsBuffer() const { return modelsBuffer; }

    // Query mesh info (copy) for use in the app (model matrix etc.)
    MeshInfo getMeshInfo(uint32_t meshId) const;

private:
    mutable std::mutex mutex;
    uint32_t nextId = 1;
    std::vector<MeshInfo> meshes; // sparse list; index by insertion order
    std::unordered_map<uint32_t, size_t> idToIndex;

    // CPU-side combined buffers
    std::vector<Vertex> mergedVertices;
    std::vector<uint32_t> mergedIndices;
    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;

    // A temporary compact indirect buffer used to upload only visible commands
    Buffer compactIndirectBuffer;
    // GPU-side culling resources
    Buffer boundsBuffer; // vec4 per-mesh: xyz=center, w=radius
    Buffer visibleCountBuffer; // single uint counter

    // Compute pipeline objects for GPU culling
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet computeDescriptorSet = VK_NULL_HANDLE;

    // Optional device function for indirect-count draw (KHR or core 1.2)
    PFN_vkCmdDrawIndexedIndirectCountKHR cmdDrawIndexedIndirectCount = nullptr;

    // GPU buffers
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer indirectBuffer;
    Buffer modelsBuffer;

    bool dirty = false;
    VkDescriptorSet modelsDescriptorSet = VK_NULL_HANDLE;
};