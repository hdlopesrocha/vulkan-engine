#pragma once

#include "../Buffer.hpp"
#include "../../math/Geometry.hpp"
#include <glm/glm.hpp>
#include <cstdint>
#include <memory>

// An immutable snapshot of a single chunk's GPU resources.
//
// Once constructed and uploaded, a RenderProxy is never modified.
// When a chunk changes, a new RenderProxy is built from scratch and
// atomically swapped into place (see Chunk::pendingProxy swap).
// The old proxy is retained until the GPU finishes referencing it,
// then destroyed via the deferred destruction queue.
//
// Thread safety: all members are const after construction. The proxy
// itself can be read from any thread without locking.
class RenderProxy {
public:
    // Unique identifier matching the source octree node
    uint32_t chunkId = 0;

    // Octree version at the time this mesh was generated.
    // Used to detect stale proxies during swap.
    uint32_t version = 0;

    // Stable slot index in the IndirectRenderer's indirect/bounds
    // buffer. Assigned once at proxy creation and never changes.
    uint32_t slotIndex = UINT32_MAX;

    // Device-local vertex/index buffers (exclusive to this proxy
    // when the per-proxy buffer path is used; may alias to a
    // combined buffer when using the slot-allocation path).
    Buffer vertexBuffer{};
    Buffer indexBuffer{};
    VkDeviceSize vertexCount = 0;
    VkDeviceSize indexCount  = 0;

    // Object-space bounding box (used by GPU frustum culling).
    glm::vec4 boundsMin{0.0f};
    glm::vec4 boundsMax{0.0f};

    // The indirect draw command for this mesh. baseVertex/firstIndex
    // reference this proxy's vertex/index data.
    VkDrawIndexedIndirectCommand drawCmd{};

    // Default constructor for container operations.
    RenderProxy() = default;

    // Construct from a CPU mesh geometry. Does NOT upload to GPU.
    // GPU upload must happen separately (via upload() or via the
    // async upload pipeline).
    explicit RenderProxy(uint32_t chunkId_, uint32_t version_, uint32_t slotIndex_,
                         const Geometry& geom)
        : chunkId(chunkId_)
        , version(version_)
        , slotIndex(slotIndex_)
        , vertexCount(geom.vertices.size())
        , indexCount(geom.indices.size())
    {
        if (geom.vertices.empty()) {
            boundsMin = glm::vec4(0.0f);
            boundsMax = glm::vec4(0.0f);
        } else {
            glm::vec3 minp(FLT_MAX), maxp(-FLT_MAX);
            for (const auto& v : geom.vertices) {
                minp = glm::min(minp, v.position);
                maxp = glm::max(maxp, v.position);
            }
            boundsMin = glm::vec4(minp, 0.0f);
            boundsMax = glm::vec4(maxp, 0.0f);
        }

        drawCmd.indexCount    = static_cast<uint32_t>(geom.indices.size());
        drawCmd.instanceCount = 1;
        drawCmd.firstIndex    = 0;
        drawCmd.vertexOffset  = 0;
        drawCmd.firstInstance = 0;
    }

    // RenderProxy is move-only (immutable after construction).
    RenderProxy(const RenderProxy&) = delete;
    RenderProxy& operator=(const RenderProxy&) = delete;

    RenderProxy(RenderProxy&& other) noexcept
        : chunkId(other.chunkId)
        , version(other.version)
        , slotIndex(other.slotIndex)
        , vertexBuffer(other.vertexBuffer)
        , indexBuffer(other.indexBuffer)
        , vertexCount(other.vertexCount)
        , indexCount(other.indexCount)
        , boundsMin(other.boundsMin)
        , boundsMax(other.boundsMax)
        , drawCmd(other.drawCmd)
    {
        other.vertexBuffer = {};
        other.indexBuffer  = {};
        other.vertexCount = 0;
        other.indexCount  = 0;
    }

    RenderProxy& operator=(RenderProxy&& other) noexcept {
        if (this != &other) {
            chunkId     = other.chunkId;
            version     = other.version;
            slotIndex   = other.slotIndex;
            vertexBuffer = other.vertexBuffer;
            indexBuffer  = other.indexBuffer;
            vertexCount = other.vertexCount;
            indexCount  = other.indexCount;
            boundsMin   = other.boundsMin;
            boundsMax   = other.boundsMax;
            drawCmd     = other.drawCmd;

            other.vertexBuffer = {};
            other.indexBuffer  = {};
            other.vertexCount = 0;
            other.indexCount  = 0;
        }
        return *this;
    }

    // Returns true if this proxy has valid GPU resources.
    bool isValid() const {
        return vertexBuffer.buffer != VK_NULL_HANDLE || vertexCount == 0;
    }

    // Returns true if the mesh is empty (zero vertices).
    bool isEmpty() const {
        return vertexCount == 0 || indexCount == 0;
    }
};
