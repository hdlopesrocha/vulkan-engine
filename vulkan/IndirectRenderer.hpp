#pragma once

#include "VulkanApp.hpp"
#include "../math/Geometry.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>

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
        VkDeviceSize indirectOffset = 0; // byte offset into indirect buffer
        bool active = false;
        // Per-mesh GPU buffers (kept separate for simplicity)
        Buffer vertexBuffer;
        Buffer indexBuffer;
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

    // Accessors
    const Buffer& getVertexBuffer() const { return vertexBuffer; }
    const Buffer& getIndexBuffer() const { return indexBuffer; }
    const Buffer& getIndirectBuffer() const { return indirectBuffer; }

    // Query mesh info (copy) for use in the app (model matrix etc.)
    MeshInfo getMeshInfo(uint32_t meshId) const;

private:
    uint32_t nextId = 1;
    std::vector<MeshInfo> meshes; // sparse list; index by insertion order
    std::unordered_map<uint32_t, size_t> idToIndex;

    // CPU-side combined buffers
    std::vector<Vertex> mergedVertices;
    std::vector<uint32_t> mergedIndices;
    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;

    // GPU buffers
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer indirectBuffer;

    bool dirty = false;
};