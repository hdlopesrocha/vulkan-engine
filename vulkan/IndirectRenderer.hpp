// ...existing code...
// IndirectRenderer.hpp
#pragma once

#include "VulkanApp.hpp"
#include "../math/Geometry.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

// Manages a single large vertex/index/indirect buffer and provides a simple
// CPU-side allocator for adding/removing meshes. Draws are performed via
// vkCmdDrawIndexedIndirect (one indirect command per mesh). The allocator is
// append-first and supports reclamation on remove (simple free list rebuild).
class IndirectRenderer {
public:
        // Allow external code to force the dirty flag
        void setDirty(bool value) { dirty = value; }
    // Upload vertex and index data for a single mesh
    bool uploadMeshVerticesAndIndices(VulkanApp* app, uint32_t meshId);
    // Write all mesh indirect/model/bounds buffers for all active meshes
    void uploadMeshMetaBuffers(VulkanApp* app);
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
    uint32_t addMesh(VulkanApp* app, const Geometry& mesh);
    // Add mesh with a custom ID (e.g., node ID from octree). If mesh with this ID exists, it is replaced.
    uint32_t updateMesh(VulkanApp* app, const Geometry& mesh, uint32_t customId);
    void removeMesh(uint32_t meshId);

    // Rebuild GPU backing buffers from current CPU mesh list. Call before drawing
    // if add/remove operations occurred.
    void rebuild(VulkanApp* app);

    // Upload a single mesh to GPU (incremental update). Requires buffers to have capacity.
    // Returns true if upload succeeded, false if rebuild() is needed (capacity exceeded or buffers not created).
    bool uploadMesh(VulkanApp* app, uint32_t meshId);
    
    // Erase a mesh from GPU by zeroing its indirect command (prevents culling from reading trash).

        public:
            // Needed for main.cpp and other modules
    // Call after removeMesh() for runtime removals.
    void eraseMeshFromGPU(VulkanApp* app, uint32_t meshId);
    
    // Ensure GPU buffers have capacity for at least the given counts. 
    // Call this before a batch of addMesh+uploadMesh if you know the expected size.
    // Returns true if buffers are ready, false if they needed to be created/grown (triggers rebuild).
    bool ensureCapacity(VulkanApp* app, size_t vertexCount, size_t indexCount, size_t meshCount);
    
    // Check if dirty flag is set (needs rebuild or incremental uploads)
    bool isDirty() const { return dirty; }

public:
    // Bind merged vertex/index buffers once and draw all provided mesh ids.
    // This avoids binding per-mesh buffers; push constants must be set per-draw
    // by this function (it will push each mesh's model before issuing its draw).
    void drawMergedWithCull(VkCommandBuffer cmd, const glm::mat4& viewProj, VulkanApp* app, uint32_t maxDraws = 0);
    // Run GPU culling/compaction (must be called outside any render pass).
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws = 0);
    // Issue indirect draw using the compacted indirect buffer (call inside render pass).
    void drawPrepared(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws = 0);
    // Bind vertex/index buffers (call once before multiple drawIndirectOnly calls)
    void bindBuffers(VkCommandBuffer cmd);
    // Issue indirect draw only (buffers must already be bound via bindBuffers)
    void drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws = 0);
    // Issue indirect draw with custom pipeline layout for push constants
    void drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws = 0);

    // Update a descriptor set to point to the GPU-side models SSBO.
    void updateModelsDescriptorSet(VulkanApp* app, VkDescriptorSet ds);
    void setModelsDescriptorSet(VkDescriptorSet ds) { modelsDescriptorSet = ds; }

    // Accessors
    const Buffer& getVertexBuffer() const { return vertexBuffer; }
    const Buffer& getIndexBuffer() const { return indexBuffer; }
    const Buffer& getIndirectBuffer() const { return indirectBuffer; }
    const Buffer& getModelsBuffer() const { return modelsBuffer; }
    const Buffer& getCompactIndirectBuffer() const { return compactIndirectBuffer; }
    VkPipeline getComputePipeline() const { return computePipeline; }

    // Get count of active meshes
    size_t getMeshCount() const {
        std::lock_guard<std::mutex> lock(mutex);
        size_t count = 0;
        for (const auto& m : meshes) {
            if (m.second.active) ++count;
        }
        return count;
    }

    // Host-read of the GPU-visible count (requires GPU idle; stats-only).
    uint32_t readVisibleCount(VulkanApp* app) const;

    // Query mesh info (copy) for use in the app (model matrix etc.)
    MeshInfo getMeshInfo(uint32_t meshId) const;

    // Return a copy of all active mesh infos (thread-safe)
    std::vector<MeshInfo> getActiveMeshInfos() const;

private:
    mutable std::mutex mutex;
    uint32_t nextId = 1;
    std::unordered_map<uint32_t, MeshInfo> meshes; // nodeId -> MeshInfo

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
    
    // Capacity tracking (in elements, not bytes)
    size_t vertexCapacity = 0;
    size_t indexCapacity = 0;
    size_t meshCapacity = 0;

    bool dirty = false;
    bool descriptorDirty = false;  // flag for deferred descriptor update
    VkDescriptorSet pendingDescriptorSet = VK_NULL_HANDLE; // ds to update (VK_NULL_HANDLE means use/create material set)
    VkDescriptorSet modelsDescriptorSet = VK_NULL_HANDLE;
};