#include "IndirectRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "../VulkanApp.hpp"
#include "../streaming/UploadManager.hpp"
#include "../../utils/FileReader.hpp"
#include "../includes/locations.hpp"
#include "SlotAllocator.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <chrono>

// Last time a "no free slot" line was logged (throttled to 1/sec so a full
// pool doesn't spam one line per rejected mesh).
static std::chrono::steady_clock::time_point g_lastNoSlotLog = std::chrono::steady_clock::time_point::min();

namespace {
// Push constant blocks are std430: vec3 members align to 16 bytes, so the
// C++ mirrors carry explicit padding to match the shader layout exactly.
struct CullPushConstants {
    glm::mat4 viewProj;   // offset 0
    uint32_t targetLayer; // offset 64
    uint32_t numCmds;     // offset 68
    float pad0[2];        // offset 72
    glm::vec3 camPos;     // offset 80
    float lodBias;        // offset 92
    uint32_t maxTargetLod; // offset 96
    uint32_t numCmdsVeg;   // offset 100 (vegetation entry count; 0 for solid-only dispatch)
}; // 104 bytes

struct CascadeCullPushConstants {
    uint32_t numChunks;   // offset 0
    float pad0[3];        // offset 4
    glm::vec3 camPos;     // offset 16
    float lodBias;        // offset 28
}; // 32 bytes
} // namespace

// Unlocked — caller must hold `mutex`. Memoized active-mesh count; recomputed
// (full scan) only after any meshes mutation. All mutation sites set
// activeMeshCountDirty_ under the same lock, so the memoized value always
// equals a fresh scan — no behavior change vs. iterating on every call.
size_t IndirectRenderer::activeMeshCountLocked() const {
    if (activeMeshCountDirty_) {
        activeMeshCount_ = 0;
        for (const auto& kv : meshes) {
            if (kv.second.active) ++activeMeshCount_;
        }
        activeMeshCountDirty_ = false;
    }
    return activeMeshCount_;
}

// Unlocked — caller must hold `mutex`. Returns the number of draw commands
// (slots) to cull in the current mode. In slotted mode this is the fixed slot
// pool capacity (one draw entry per chunk); in legacy mode it's the active
// mesh count.
uint32_t IndirectRenderer::getCullDispatchCountLocked() const {
    if (slottedMode) {
        // One draw entry per active chunk slot; the pool capacity is the
        // fixed dispatch size (zeroed tail entries are skipped on indexCount).
        return static_cast<uint32_t>(slotAlloc.capacity());
    }
    return static_cast<uint32_t>(activeMeshCountLocked());
}

void IndirectRenderer::publishPendingTransfer(VulkanApp* app) {
    if (pendingTransfer.fence == VK_NULL_HANDLE) return;
    VkDevice dev = app->getDevice();

    // processPendingCommandBuffers owns the fence lifecycle — it will
    // free the command buffer and destroy the fence once signaled.
    // If the fence is no longer tracked, the work is already done.
    if (app->resources.find((uintptr_t)pendingTransfer.fence).has_value()) {
        VulkanApp::waitFence(dev, pendingTransfer.fence);
    }
    // Meta-buffers (indirect/draw-count) are append-only: a new mesh's
    // entry is only written once.  Calling doUploadMeshMetaBuffers
    // after the vertex/index data lands on the GPU is safe even when
    // earlier entries were written in a prior upload.
    doUploadMeshMetaBuffers(app);

    // Release the staging region. For the ring-backed path this returns the
    // suballocated region to the persistent StagingRingBuffer (no vkFreeMemory);
    // for the fallback path the dedicated staging buffer is destroyed.
    if (pendingTransfer.stagingAlloc.mappedPtr) {
        app->stagingRing.release(pendingTransfer.stagingAlloc);
        pendingTransfer.stagingAlloc = {};
    }
    if (pendingTransfer.stagingBuffer.buffer != VK_NULL_HANDLE) {
        app->resources.removeBufferVma(pendingTransfer.stagingBuffer.buffer, pendingTransfer.stagingBuffer.allocation);
        pendingTransfer.stagingBuffer = {};
    }

    // Fire deferred upload completion callbacks. These are set by the
    // legacy staging path of uploadSlot() to transition chunk state from
    // UploadingGPU → ReadyToSwap after the GPU transfer completes.
    for (auto& cb : deferredUploadCallbacks_) {
        if (cb) cb();
    }
    deferredUploadCallbacks_.clear();

    // Do NOT destroy the fence — processPendingCommandBuffers handles it.
    pendingTransfer = {};
}

void IndirectRenderer::pollPendingTransfers(VulkanApp* app) {
    if (pendingTransfer.fence == VK_NULL_HANDLE) return;
    VkDevice dev = app->getDevice();
    // If processPendingCommandBuffers already cleaned up the fence, the
    // transfer is done — skip vkGetFenceStatus on the destroyed handle.
    if (!app->resources.find((uintptr_t)pendingTransfer.fence).has_value()) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        publishPendingTransfer(app);
        return;
    }
    VkResult r = vkGetFenceStatus(dev, pendingTransfer.fence);
    if (r == VK_NOT_READY) return;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    publishPendingTransfer(app);
}

void IndirectRenderer::acquireBuffers(VkCommandBuffer cmd) {
    VkBufferMemoryBarrier2 barriers[4]{};
    uint32_t count = 0;

    auto addBarrier = [&](VkBuffer buf, VkAccessFlags2 dstAccess) {
        if (buf == VK_NULL_HANDLE) return;
        barriers[count].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[count].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[count].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[count].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        barriers[count].dstAccessMask = dstAccess;
        barriers[count].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[count].buffer = buf;
        barriers[count].offset = 0;
        barriers[count].size = VK_WHOLE_SIZE;
        ++count;
    };

    addBarrier(vertexBuffer.buffer,       VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
    addBarrier(indexBuffer.buffer,        VK_ACCESS_2_INDEX_READ_BIT);
    addBarrier(indirectBuffer.buffer,     VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    addBarrier(boundsBuffer.buffer,       VK_ACCESS_2_SHADER_READ_BIT);

    if (count > 0) {
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = count;
        depInfo.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
}

void IndirectRenderer::setVertexBufferForMesh(uint32_t meshId, Buffer vbuf) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    // For simplicity, just assign to the main vertexBuffer (per-mesh not tracked in this design)
    vertexBuffer = vbuf;
}

void IndirectRenderer::setIndexBufferForMesh(uint32_t meshId, Buffer ibuf) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    indexBuffer = ibuf;
}

uint32_t IndirectRenderer::acquireGeomSlot() {
    // Caller holds `mutex`. Lowest free index keeps the number of slots that
    // ever get allocated minimal (only as many as are concurrently in flight).
    for (uint32_t i = 0; i < MAX_GEOM_BUFFERS; i++) {
        if (!geomSlotInUse[i]) {
            geomSlotInUse[i] = true;
            return i;
        }
    }
    return UINT32_MAX;
}

void IndirectRenderer::markGeomSlotFree(uint32_t slot) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (slot < MAX_GEOM_BUFFERS) geomSlotInUse[slot] = false;
}

void IndirectRenderer::recyclePreviousGeom(VulkanApp* app, uint32_t prevSlot,
                                           Buffer prevVertex, Buffer prevIndex) {
    // Gate recycling on the current frame's fence. Graphics-queue submission is
    // FIFO, so when this frame's fence signals every earlier frame (the only
    // ones that referenced the previous buffers) has completed. The callback is
    // polled non-blocking in processPendingCommandBuffers — no vkWaitForFences,
    // hence no fence-index wraparound deadlock.
    VkFence f = app->getCurrentFrameFence();
    if (prevSlot != UINT32_MAX) {
        app->deferDestroyUntilFence(f, [this, prevSlot]() { markGeomSlotFree(prevSlot); });
    } else if (prevVertex.buffer != VK_NULL_HANDLE || prevIndex.buffer != VK_NULL_HANDLE) {
        // Previous buffers were a throwaway fallback allocation — free them.
        app->deferDestroyUntilFence(f, [app, prevVertex, prevIndex]() {
            if (prevVertex.buffer != VK_NULL_HANDLE)
                app->resources.removeBufferVma(prevVertex.buffer, prevVertex.allocation);
            if (prevIndex.buffer != VK_NULL_HANDLE)
                app->resources.removeBufferVma(prevIndex.buffer, prevIndex.allocation);
        });
    }
}

IndirectRenderer::IndirectRenderer() {}
IndirectRenderer::~IndirectRenderer() {}

void IndirectRenderer::init() {
}

void IndirectRenderer::cleanup(VulkanApp* app) {
    (void)app;
    meshes.clear();
    activeMeshCountDirty_ = true;
    vertexBuffer = {};
    indexBuffer = {};
    for (auto& b : vertexSlots) b = {};
    for (auto& b : indexSlots) b = {};
    vertexSlotCap.fill(0);
    indexSlotCap.fill(0);
    geomSlotInUse.fill(false);
    currentGeomSlot = UINT32_MAX;
    indirectBuffer = {};
    for (auto& b : compactIndirectBuffers) b = {};
    for (auto& b : visibleLodBuffers) b = {};
    visibleLodsScratch = {};
    boundsBuffer = {};
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        if (visibleCountMapped[f] && storedDevice != VK_NULL_HANDLE) {
            visibleCountBuffers[f].unmap();
            visibleCountMapped[f] = nullptr;
        }
        visibleCountBuffers[f] = {};
    }
}

uint32_t IndirectRenderer::addMesh(const Geometry& mesh) {
    if (slottedMode) {
        return UINT32_MAX;
    }
    return updateMesh(mesh, nextId++);
}

uint32_t IndirectRenderer::updateMesh(const Geometry& mesh, uint32_t customId) {
    if (slottedMode) {
        return UINT32_MAX;
    }
    std::lock_guard<std::recursive_mutex> guard(mutex);

    MeshInfo m{};
    m.id = customId;
    m.baseVertex = static_cast<uint32_t>(mergedVertices.size());
    m.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
    m.firstIndex = static_cast<uint32_t>(mergedIndices.size());
    m.indexCount = static_cast<uint32_t>(mesh.indices.size());
    m.drawIndex = static_cast<uint32_t>(indirectCommands.size());
    m.active = true;

    if (mesh.vertices.empty()) {
        // Empty mesh: set degenerate zero-sized bounds at origin
        m.boundsMin = glm::vec4(0.0f);
        m.boundsMax = glm::vec4(0.0f);
    } else {
        glm::vec3 minp(FLT_MAX), maxp(-FLT_MAX);
        for (const auto& v : mesh.vertices) {
            minp = glm::min(minp, v.position);
            maxp = glm::max(maxp, v.position);
        }
        m.boundsMin = glm::vec4(minp, 0.0f);
        m.boundsMax = glm::vec4(maxp, 0.0f);
    }

    mergedVertices.insert(mergedVertices.end(), mesh.vertices.begin(), mesh.vertices.end());
    mergedIndices.insert(mergedIndices.end(), mesh.indices.begin(), mesh.indices.end());

    VkDrawIndexedIndirectCommand cmd{};
    cmd.indexCount = m.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex = m.firstIndex;
    cmd.vertexOffset = static_cast<int32_t>(m.baseVertex);
    cmd.firstInstance = static_cast<uint32_t>(indirectCommands.size());
    indirectCommands.push_back(cmd);

    meshes[m.id] = m; // insert or replace
    activeMeshCountDirty_ = true; // active set may have changed
    dirty = true;     // adding a mesh always requires a rebuild

    return customId;
}


void IndirectRenderer::removeMesh(uint32_t meshId) {
    if (slottedMode) {
        return;
    }
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return;
    it->second.active = false;
    activeMeshCountDirty_ = true;
    dirty = true;
}

void IndirectRenderer::removeAllMeshes() {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    meshes.clear();
    activeMeshCountDirty_ = true;
    metaBuffersWrittenCount = 0;
    dirty = true;

    if (slottedMode) {
        // Pre-sized slot buffers: zero in-place instead of clearing/resizing.
        if (!mergedVertices.empty())
            for (auto& v : mergedVertices) v = Vertex{};
        if (!mergedIndices.empty())
            std::memset(mergedIndices.data(), 0, mergedIndices.size() * sizeof(uint32_t));
        if (!indirectCommands.empty())
            std::memset(indirectCommands.data(), 0,
                        indirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));

        // Zero GPU indirect and bounds buffers so culling sees indexCount=0.
        if (indirectBuffer.buffer != VK_NULL_HANDLE) {
            void* ptr = indirectBuffer.map(0);
            if (ptr) {
                std::memset(ptr, 0, indirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
                indirectBuffer.unmap();
            }
        }
        if (boundsBuffer.buffer != VK_NULL_HANDLE) {
            void* ptr = boundsBuffer.map(0);
            if (ptr) {
                std::memset(ptr, 0, slotAlloc.capacity() * 3 * sizeof(glm::vec4));
                boundsBuffer.unmap();
            }
        }
        // Zero the chosen-LoD outputs so no stale (chunk, level) pair survives
        // a scene clear.
        VkDeviceSize lodBytes = sizeof(glm::uvec2) * slotAlloc.capacity();
        for (auto& b : visibleLodBuffers) {
            if (b.buffer != VK_NULL_HANDLE) {
                void* ptr = b.map(0);
                if (ptr) {
                    std::memset(ptr, 0, (size_t)lodBytes);
                    b.unmap();
                }
            }
        }
        if (visibleLodsScratch.buffer != VK_NULL_HANDLE) {
            void* ptr = visibleLodsScratch.map(0);
            if (ptr) {
                std::memset(ptr, 0, (size_t)lodBytes);
                visibleLodsScratch.unmap();
            }
        }

        // Free all slots (collect indices first to avoid recursive locking).
        std::vector<uint32_t> active;
        slotAlloc.visitActive([&active](uint32_t idx, const auto&) { active.push_back(idx); });
        for (uint32_t idx : active) slotAlloc.free(idx);
    } else {
        mergedVertices.clear();
        mergedIndices.clear();
        mergedVertices.shrink_to_fit();
        mergedIndices.shrink_to_fit();
        indirectCommands.clear();
    }
}

bool IndirectRenderer::ensureCapacity(size_t vertexCount, size_t indexCount, size_t meshCount) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    // Add 25% headroom for future growth
    size_t neededVertexCap = vertexCount + vertexCount / 4;
    size_t neededIndexCap = indexCount + indexCount / 4;
    size_t neededMeshCap = meshCount + meshCount / 4;
    
    bool needsRebuild = false;
    
    if (vertexBuffer.buffer == VK_NULL_HANDLE || vertexCapacity < neededVertexCap) {
        needsRebuild = true;
    }
    if (indexBuffer.buffer == VK_NULL_HANDLE || indexCapacity < neededIndexCap) {
        needsRebuild = true;
    }
    if (indirectBuffer.buffer == VK_NULL_HANDLE || meshCapacity < neededMeshCap) {
        needsRebuild = true;
    }
    
    if (needsRebuild) {
        // Set target capacities - rebuild will use these
        if (neededVertexCap > vertexCapacity) vertexCapacity = neededVertexCap;
        if (neededIndexCap > indexCapacity) indexCapacity = neededIndexCap;
        if (neededMeshCap > meshCapacity) meshCapacity = neededMeshCap;
        dirty = true;
    }
    
    return !needsRebuild;
}

bool IndirectRenderer::uploadMeshes(VulkanApp* app, const std::vector<uint32_t>& meshIds, float priority) {
    app_ = app;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (meshIds.empty()) return true;

    // Per-mesh copy request gathered before any GPU work is recorded.
    struct Req {
        uint32_t meshId;
        size_t meshVertexCount;
        VkDeviceSize vertexOffset;
        VkDeviceSize vertexSize;
        VkDeviceSize indexOffset;
        VkDeviceSize indexSize;
        VkDeviceSize stagingVertexOffset;
        VkDeviceSize stagingIndexOffset;
        bool doVertex;
        bool doIndex;
    };
    std::vector<Req> reqs;
    reqs.reserve(meshIds.size());
    VkDeviceSize totalStaging = 0;
    bool anyVertex = false;
    bool anyIndex = false;

    for (uint32_t meshId : meshIds) {
        auto it = meshes.find(meshId);
        if (it == meshes.end()) {
            continue;
        }
        MeshInfo& info = it->second;
        if (!info.active) {
            continue;
        }
        if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
            return false;
        }

        // Basic bounds validations to detect corrupted mesh data early.
        size_t indicesAvailable = mergedIndices.size();
        size_t verticesAvailable = mergedVertices.size();
        if (info.firstIndex + static_cast<uint64_t>(info.indexCount) > indicesAvailable) {
            std::cerr << "[IndirectRenderer] uploadMeshes: mesh " << meshId
                      << " index range out of bounds: firstIndex=" << info.firstIndex
                      << " indexCount=" << info.indexCount
                      << " mergedIndices.size=" << indicesAvailable << std::endl;
            assert(false && "mesh index range out of bounds");
            return false;
        }

        uint32_t maxVertexIdx = 0;
        for (size_t i = info.firstIndex; i < info.firstIndex + info.indexCount; ++i) {
            uint32_t idx = mergedIndices[i];
            if (idx > maxVertexIdx) maxVertexIdx = idx;
        }
        size_t meshVertexCount = static_cast<size_t>(maxVertexIdx) + 1;

        if (info.baseVertex + meshVertexCount > verticesAvailable) {
            std::cerr << "[IndirectRenderer] uploadMeshes: mesh " << meshId
                      << " vertex range out of bounds: baseVertex=" << info.baseVertex
                      << " meshVertexCount=" << meshVertexCount
                      << " mergedVertices.size=" << verticesAvailable << std::endl;
            assert(false && "mesh vertex range out of bounds");
            return false;
        }

        if (info.baseVertex + meshVertexCount > vertexCapacity) {
            std::cerr << "[IndirectRenderer] uploadMeshes] vertex capacity exceeded (" << info.baseVertex << " + " << meshVertexCount << " > " << vertexCapacity << ")\n";
            return false;
        }
        if (info.firstIndex + info.indexCount > indexCapacity) {
            std::cerr << "[IndirectRenderer] uploadMeshes: index capacity exceeded (" << info.firstIndex << " + " << info.indexCount << " > " << indexCapacity << ")\n";
            return false;
        }

        // Ensure every index references a local vertex (before vertexOffset is applied)
        for (size_t i = info.firstIndex; i < info.firstIndex + info.indexCount; ++i) {
            uint32_t idx = mergedIndices[i];
            if (idx >= meshVertexCount) {
                std::cerr << "[IndirectRenderer] uploadMeshes: mesh " << meshId
                          << " index value out of local vertex range: indexPos=" << i
                          << " indexVal=" << idx << " meshVertexCount=" << meshVertexCount << std::endl;
                assert(false && "index value out of range for mesh");
                return false;
            }
        }

        // Check for non-finite vertex positions which indicate memory corruption or bad generation
        for (size_t v = info.baseVertex; v < info.baseVertex + meshVertexCount; ++v) {
            const Vertex &vert = mergedVertices[v];
            if (!std::isfinite(vert.position.x) || !std::isfinite(vert.position.y) || !std::isfinite(vert.position.z)) {
                std::cerr << "[IndirectRenderer] uploadMeshes: mesh " << meshId
                          << " has non-finite vertex at index=" << v << " pos=(" << vert.position.x << "," << vert.position.y << "," << vert.position.z << ")\n";
                assert(false && "non-finite vertex position");
                return false;
            }
        }

        if (info.indexCount % 3 != 0) {
            std::cerr << "[IndirectRenderer] warning: mesh " << meshId << " indexCount not multiple of 3: " << info.indexCount << std::endl;
        }
        VkDeviceSize vertexOffset = info.baseVertex * sizeof(Vertex);
        VkDeviceSize vertexSize = meshVertexCount * sizeof(Vertex);
        VkDeviceSize indexOffset = info.firstIndex * sizeof(uint32_t);
        VkDeviceSize indexSize = info.indexCount * sizeof(uint32_t);
        bool doVertexUpload = (vertexSize > 0 && info.baseVertex < mergedVertices.size());
        bool doIndexUpload = (indexSize > 0 && info.firstIndex < mergedIndices.size());
        if (doVertexUpload) anyVertex = true;
        if (doIndexUpload) anyIndex = true;

        // Direct memcpy to HOST_VISIBLE staging buffer, then vkCmdCopyBuffer
        // to device-local vertex/index buffers. Device-local memory is
        // required because HOST_VISIBLE pages on RADV iGPU lack TCP-read
        // permission in the GPU page table.
        if (doVertexUpload || doIndexUpload) {
            VkDeviceSize stagingSize = (doVertexUpload ? vertexSize : 0)
                                     + (doIndexUpload  ? indexSize  : 0);
            totalStaging += stagingSize;
            reqs.push_back({meshId, meshVertexCount, vertexOffset, vertexSize,
                            indexOffset, indexSize, 0, 0, doVertexUpload, doIndexUpload});
        }
    }

    if (reqs.empty()) return true;

    // --- Async UploadManager path -------------------------------------------
    // When an UploadManager is wired in, route the copies through it: the whole
    // validated batch is packaged as a single UploadJob (vertex + index slices)
    // and streamed via one of K concurrent staging slots. This removes the
    // single in-flight pendingTransfer slot (and its vkWaitForFences stall in
    // publishPendingTransfer) that serialized incremental uploads. Each mesh's
    // indirect/bounds meta entry is published individually when the transfer
    // retires (per-mesh, since manager transfers may complete out of order).
    // If the batch would not fit in one staging slot we fall through to the
    // legacy ring-backed path, which allocates a right-sized staging buffer.
    if (uploadMgr_ && totalStaging <= uploadMgr_->slotSize()) {
        streaming::UploadJob job;
        job.category  = streamCategory_;
        job.priority  = priority;
        job.chunkSlot = nullptr;   // merged buffers are owned by this renderer
        job.uploads.reserve(reqs.size() * 2);

        std::vector<uint32_t> batchIds;
        batchIds.reserve(reqs.size());
        for (auto& r : reqs) {
            if (r.doVertex) {
                streaming::BufferUpload bu;
                bu.dst       = vertexBuffer;
                bu.dstOffset = r.vertexOffset;
                bu.cpuData.resize(r.vertexSize);
                std::memcpy(bu.cpuData.data(), &mergedVertices[meshes[r.meshId].baseVertex], r.vertexSize);
                job.uploads.push_back(std::move(bu));
            }
            if (r.doIndex) {
                streaming::BufferUpload bu;
                bu.dst       = indexBuffer;
                bu.dstOffset = r.indexOffset;
                bu.cpuData.resize(r.indexSize);
                std::memcpy(bu.cpuData.data(), &mergedIndices[meshes[r.meshId].firstIndex], r.indexSize);
                job.uploads.push_back(std::move(bu));
            }
            batchIds.push_back(r.meshId);
        }

        job.onComplete = [this, batchIds]() {
            std::lock_guard<std::recursive_mutex> lock(mutex);
            for (uint32_t id : batchIds) publishMeshMeta(id);
        };

        uploadMgr_->enqueue(std::move(job));
        return true;
    }

    // Suballocate the staging region from the app's persistent StagingRingBuffer
    // (persistently-mapped, avoids a per-chunk vkAllocateMemory + map + free).
    // Fall back to a dedicated host-visible staging buffer only if the ring is
    // exhausted or fragmented.
    StagingRingBuffer::Allocation stagingAlloc = app->stagingRing.allocate(totalStaging);
    Buffer stagingFallback;
    void* mapped = nullptr;
    VkBuffer stagingVk = VK_NULL_HANDLE;
    VkDeviceSize stagingBase = 0;
    if (stagingAlloc.mappedPtr) {
        mapped = stagingAlloc.mappedPtr;
        stagingVk = app->stagingRing.buffer();
        stagingBase = stagingAlloc.offset;
    } else {
        stagingFallback = app->createBuffer(totalStaging,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        mapped = stagingFallback.map(0);
        stagingVk = stagingFallback.buffer;
        stagingBase = 0;
    }
    VkDeviceSize off = 0;
    for (auto& r : reqs) {
        if (r.doVertex) {
            r.stagingVertexOffset = off;
            std::memcpy(static_cast<char*>(mapped) + off, &mergedVertices[meshes[r.meshId].baseVertex], r.vertexSize);
            off += r.vertexSize;
        }
        if (r.doIndex) {
            r.stagingIndexOffset = off;
            std::memcpy(static_cast<char*>(mapped) + off, &mergedIndices[meshes[r.meshId].firstIndex], r.indexSize);
            off += r.indexSize;
        }
    }
    if (stagingFallback.buffer != VK_NULL_HANDLE) {
        stagingFallback.unmap(); // VMA persistent mapping
    }

    // Submit the staging→device-local copies asynchronously and defer the
    // meta-buffer write until the fence signals. The whole batch is coalesced
    // into a single command buffer, so one fence covers every mesh's copy:
    // when it signals, all vertex/index data has landed and publishing the
    // (append-only) meta-buffer entries for the batch is safe.  The meta-buffer
    // (indirect + bounds) is append-only, so publishing it later is safe:
    // in-flight draws that already reference earlier offsets still see valid data.
    // Publishing the previous batch first avoids overwriting its single
    // in-flight staging buffer / fence slot.
    if (pendingTransfer.fence != VK_NULL_HANDLE) {
        publishPendingTransfer(app);
    }
    pendingTransfer.fence = app->runSingleTimeCommandsAsync([&](VkCommandBuffer cmd) {
        // Barrier: prior vertex/index reads must complete before the transfers
        // write to those buffers. A single barrier per destination buffer (the
        // whole buffer) covers every disjoint copy in this batch.
        // srcStageMask must be ALL_COMMANDS: sync validation attributes a
        // draw's vertex-attribute reads to the whole pipeline span (TESS_EVAL,
        // GEOMETRY, FRAGMENT, COLOR_ATTACHMENT_OUTPUT, ... — observed for the
        // tessellated solid pipeline), so VERTEX_INPUT alone leaves the copy
        // unsynchronized against those reads (SYNC-HAZARD-WRITE-AFTER-READ).
        VkBufferMemoryBarrier2 vb{};
        vb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        vb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        vb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        vb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        vb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vb.offset = 0;
        vb.size = VK_WHOLE_SIZE;
        if (anyVertex) {
            vb.srcAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            vb.buffer = vertexBuffer.buffer;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &vb;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
        if (anyIndex) {
            vb.srcAccessMask = VK_ACCESS_2_INDEX_READ_BIT;
            vb.buffer = indexBuffer.buffer;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &vb;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
        for (auto& r : reqs) {
            if (r.doVertex) {
                VkBufferCopy vCopy{};
                vCopy.srcOffset = stagingBase + r.stagingVertexOffset;
                vCopy.dstOffset = r.vertexOffset;
                vCopy.size = r.vertexSize;
                vkCmdCopyBuffer(cmd, stagingVk, vertexBuffer.buffer, 1, &vCopy);
            }
            if (r.doIndex) {
                VkBufferCopy iCopy{};
                iCopy.srcOffset = stagingBase + r.stagingIndexOffset;
                iCopy.dstOffset = r.indexOffset;
                iCopy.size = r.indexSize;
                vkCmdCopyBuffer(cmd, stagingVk, indexBuffer.buffer, 1, &iCopy);
            }
        }
    });
    pendingTransfer.stagingAlloc = stagingAlloc;
    pendingTransfer.stagingBuffer = stagingFallback;
    return true;
}

size_t IndirectRenderer::getMergedVertexCount() const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return mergedVertices.size();
}

size_t IndirectRenderer::getMergedIndexCount() const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return mergedIndices.size();
}

bool IndirectRenderer::uploadMesh(VulkanApp* app, uint32_t meshId) {
    if (!uploadMeshes(app, std::vector<uint32_t>{meshId})) {
        return false;
    }
    // uploadMeshMetaBuffers deferred until pendingTransfer fence signals.
    return true;
}

// Write all mesh indirect/model/bounds buffers for all active meshes
void IndirectRenderer::uploadMeshMetaBuffers(VulkanApp* app) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    doUploadMeshMetaBuffers(app);
}

// Unlocked variant — caller must hold mutex.
void IndirectRenderer::doUploadMeshMetaBuffers(VulkanApp* app) {
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;

    // Write new entries (those past metaBuffersWrittenCount) in
    // indirectCommands order.  This is critical: prepareCull dispatches
    // numCmds workgroups indexed by drawIndex, and drawPrepared caps the
    // indirect draw at indirectCommands.size() — so GPU buffer positions
    // must match indirectCommands positions.  Iterating the unordered_map
    // (used before drawIndex was added) produces a different ordering,
    // causing newly-added meshes to read stale bounds at their drawIndex
    // and get incorrectly culled.
    for (size_t i = metaBuffersWrittenCount; i < indirectCommands.size(); i++) {
        const auto& cmd = indirectCommands[i];
        VkDeviceSize cmdOffset = i * sizeof(VkDrawIndexedIndirectCommand);
        void* data = indirectBuffer.map(cmdOffset);
        memcpy(data, &cmd, sizeof(cmd));
        indirectBuffer.unmap();

        // Find the MeshInfo for this position via drawIndex.
        MeshInfo* info = nullptr;
        for (auto& kv : meshes) {
            if (kv.second.active && kv.second.drawIndex == i) {
                info = &kv.second;
                break;
            }
        }

        if (info) {
            info->indirectOffset = cmdOffset;
            if (boundsBuffer.buffer != VK_NULL_HANDLE) {
                // Per draw entry the bounds buffer holds min, max and an
                // unused meta vec4 (zeroed — chunks carry no LoD band meta).
                VkDeviceSize boundsOffset = i * 3 * sizeof(glm::vec4);
                glm::vec4 bounds[3] = { info->boundsMin, info->boundsMax, glm::vec4(0.0f) };
                data = boundsBuffer.map(boundsOffset);
                memcpy(data, bounds, sizeof(bounds));
                boundsBuffer.unmap();
            }
        }
    }
    metaBuffersWrittenCount = indirectCommands.size();
}

// Caller must hold `mutex`. Queues a host-side write of one draw entry (and
// optionally its bounds) to be copied to the GPU indirect/bounds buffers by
// flushStagedMetaWrites() during the next prepareCull. Never touches the
// HOST_VISIBLE GPU buffer directly: a direct memcpy can be observed mid-write
// by an in-flight cull dispatch reading the same entry (torn DrawCmd), which
// produced the garbage-indexCount GE hang documented on the zero fills.
void IndirectRenderer::stageMeshMetaWrite(uint32_t entryIndex,
                                          const VkDrawIndexedIndirectCommand& cmd,
                                          const glm::vec4* bounds, bool boundsValid) {
    if (entryIndex >= meshCapacity) return; // must fit the GPU indirect/bounds buffers
    // Target the NEXT cull frame: stages run before setCullFrame() advances
    // (processPendingCommandBuffers precedes render), so currentCullFrame still
    // holds the previous frame's index, whose pending list was already flushed.
    const uint32_t frame = (currentCullFrame + 1u) % MAX_CULL_FRAMES;
    MetaStageRecord rec{};
    rec.entryIndex = entryIndex;
    rec.cmd = cmd;
    if (bounds && boundsValid) {
        rec.bounds[0] = bounds[0];
        rec.bounds[1] = bounds[1];
        rec.bounds[2] = bounds[2];
        rec.boundsValid = true;
    }
    metaStagePending_[frame].push_back(rec);
}

// Copy the frame's staged meta writes into the GPU indirect/bounds buffers via
// vkCmdCopyBuffer. The copies execute in frame `frame`'s command buffer on the
// graphics queue, so they are queue-ordered after every in-flight frame's cull
// reads of the same entries (no torn-read window). A barrier before the copies
// orders those prior reads (cull dispatch SHADER_STORAGE_READ, legacy indirect
// draws' INDIRECT_COMMAND_READ) against the copies' TRANSFER_WRITE — queue
// order alone is NOT a memory dependency, and without it sync-validation
// reports WRITE_AFTER_READ when a previous frame's async cull dispatch is
// still in flight. The acquireBuffers barrier that follows in prepareCull then
// orders the copies (TRANSFER_WRITE) before the cull dispatch (SHADER_READ).
// The per-frame staging buffer is only rewritten MAX_CULL_FRAMES frames later,
// by which time this frame's fence has signaled (same rotation guarantee as
// compactIndirectBuffers); host writes to it are HOST_COHERENT and ordered by
// vkQueueSubmit before the copies.
void IndirectRenderer::flushStagedMetaWrites(VkCommandBuffer cmd, uint32_t frame) {
    if (frame >= MAX_CULL_FRAMES) return;

    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (metaStagePending_[frame].empty()) return;
        metaStageFlush_.swap(metaStagePending_[frame]);
    }

    if (metaStageFlush_.empty()) return;
    if (app_ == nullptr || indirectBuffer.buffer == VK_NULL_HANDLE) {
        // No GPU buffers yet (pre-initSlots): drop the staged writes rather
        // than accumulating them; the bulk init paths re-issue publishes once
        // the buffers exist.
        metaStageFlush_.clear();
        return;
    }

    const size_t recordBytes = metaStageFlush_.size() * sizeof(MetaStageRecord);
    Buffer& sbuf = metaStageBuffers[frame];
    if (sbuf.buffer == VK_NULL_HANDLE || metaStageCapBytes[frame] < recordBytes) {
        if (sbuf.buffer != VK_NULL_HANDLE) {
            // Immediate destroy is safe: this staging buffer's last GPU use was
            // frame-3's submission, which is guaranteed signaled before frame
            // `frame`'s prepareCull runs (the frame-slot fence wait in drawFrame
            // precedes render()/prepareCull — same rotation guarantee the
            // compact buffers rely on).
            app_->resources.removeBufferVma(sbuf.buffer, sbuf.allocation);
        }
        sbuf = app_->createBuffer(recordBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        metaStageCapBytes[frame] = static_cast<uint32_t>(recordBytes);
    }

    void* mapped = sbuf.map(0);
    if (!mapped) {
        metaStageFlush_.clear();
        return;
    }
    std::memcpy(mapped, metaStageFlush_.data(), recordBytes);
    sbuf.unmap(); // VMA persistent mapping

    // Order prior reads of the shared input buffers (a previous frame's still
    // in-flight cull dispatch reading inCmds/bounds, or a legacy draw reading
    // indirectBuffer as indirect commands) before these TRANSFER_WRITE copies.
    // WRITE_AFTER_READ hazard otherwise (sync-validation, real race on RADV).
    {
        VkBufferMemoryBarrier2 flushBarriers[2] = {};
        flushBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        flushBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                  | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        flushBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                  | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        flushBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        flushBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        flushBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        flushBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        flushBarriers[0].buffer = indirectBuffer.buffer;
        flushBarriers[0].offset = 0;
        flushBarriers[0].size = VK_WHOLE_SIZE;

        flushBarriers[1] = flushBarriers[0];
        flushBarriers[1].buffer = boundsBuffer.buffer;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 2;
        depInfo.pBufferMemoryBarriers = flushBarriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // One vkCmdCopyBuffer per record: overlapping regions within a single call
    // are undefined, and zero-then-final records can target the same entry.
    for (const auto& rec : metaStageFlush_) {
        const size_t recOffset = size_t(&rec - metaStageFlush_.data()) * sizeof(MetaStageRecord);
        VkBufferCopy c{};
        c.srcOffset = recOffset + offsetof(MetaStageRecord, cmd);
        c.dstOffset = static_cast<VkDeviceSize>(rec.entryIndex) * sizeof(VkDrawIndexedIndirectCommand);
        c.size = sizeof(VkDrawIndexedIndirectCommand);
        vkCmdCopyBuffer(cmd, sbuf.buffer, indirectBuffer.buffer, 1, &c);
        if (rec.boundsValid && boundsBuffer.buffer != VK_NULL_HANDLE) {
            c.srcOffset = recOffset + offsetof(MetaStageRecord, bounds);
            c.dstOffset = static_cast<VkDeviceSize>(rec.entryIndex) * 3 * sizeof(glm::vec4);
            c.size = 3 * sizeof(glm::vec4);
            vkCmdCopyBuffer(cmd, sbuf.buffer, boundsBuffer.buffer, 1, &c);
        }
    }

    metaStageFlush_.clear();
}

// Unlocked — caller must hold mutex. Writes a single mesh's indirect command
// and bounds at its CURRENT drawIndex offset. Both indirectCommands[drawIndex]
// and meshes[id].drawIndex are read here under the lock, so they stay
// consistent even if a rebuild() reordered draw indices between the upload
// enqueue and its completion. The write is staged for the next prepareCull's
// GPU copy, so it never races in-flight cull dispatches reading the same entry.
void IndirectRenderer::publishMeshMeta(uint32_t meshId) {
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;
    auto it = meshes.find(meshId);
    if (it == meshes.end() || !it->second.active) return;
    MeshInfo& info = it->second;
    size_t i = info.drawIndex;
    if (i >= indirectCommands.size()) return;

    VkDeviceSize cmdOffset = i * sizeof(VkDrawIndexedIndirectCommand);
    const auto& cmd = indirectCommands[i];
    info.indirectOffset = cmdOffset;

    glm::vec4 bounds[3] = { info.boundsMin, info.boundsMax, glm::vec4(0.0f) };
    stageMeshMetaWrite(static_cast<uint32_t>(i), cmd, bounds, true);
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    // In slotted mode, global rebuilds are NEVER performed. Each chunk
    // updates only its own slot via uploadSlot(). If we reach here in
    // slotted mode, the caller is using the legacy API incorrectly.
    if (slottedMode) {
        app_ = app;
        return;
    }
    app_ = app;

    // A rebuild may reallocate the merged vertex/index buffers below. Any
    // UploadJob still queued or in flight in the UploadManager captured the
    // CURRENT buffer handles at uploadMeshes() time; if we destroyed those
    // buffers while such a job is pending, its vkCmdCopyBuffer would target a
    // freed VkBuffer (VUID-vkCmdCopyBuffer-dstBuffer-parameter). Drain the
    // manager first so no pending job references a soon-to-be-destroyed buffer.
    // This MUST happen BEFORE acquiring `mutex`: flush() fires each job's
    // onComplete → publishMeshMeta, which locks the same (non-recursive) mutex.
    if (uploadMgr_) uploadMgr_->flush();

    std::lock_guard<std::recursive_mutex> guard(mutex);

    // Publish any pending async upload before rebuilding.
    if (pendingTransfer.fence != VK_NULL_HANDLE) {
        publishPendingTransfer(app);
    }

    size_t activeMeshCount = 0;
    for (const auto& kv : meshes) if (kv.second.active) ++activeMeshCount;

    if (!dirty) return;

    // Defer destruction until ALL pending GPU work completes, not just
    // the current frame's fence.  Using VK_NULL_HANDLE (wait-for-all-pending)
    // avoids a fence-index wrap-around bug where (currentFrame+1)%n picks
    // an already-signaled fence, destroying buffers that are still being
    // read by in-flight indirect.comp compute dispatches — causing GPUVM
    // faults on RADV.
    //
    // The raw vkDestroyBuffer path was safe because it only destroyed the
    // VkBuffer *handle* — the underlying VkDeviceMemory remained alive and
    // GPU reads continued to work.  But vmaDestroyBuffer calls vkFreeMemory
    // too, so the memory must be guaranteed free of GPU access before we
    // call it.
    auto scheduleDestroyBuffer = [&](const Buffer &b) {
        if (b.buffer == VK_NULL_HANDLE) return;
        Buffer copy = b;
        // Gate destruction on the current frame fence (FIFO: when it signals,
        // every earlier in-flight frame that may still reference `b` has also
        // completed). This is bounded to frames-in-flight. Using
        // deferDestroyUntilAllPending(NULL fence) here would only run once ALL
        // inFlightFences are signaled, which never happens during continuous
        // rendering (a frame is always in flight) — so buffers recreated every
        // rebuild() (compact/indirect/bounds) would leak. Matches
        // recyclePreviousGeom's fence-gated recycling.
        app->deferDestroyUntilFence(app->getCurrentFrameFence(), [app, copy]() {
            if (copy.buffer != VK_NULL_HANDLE) {
                app->resources.removeBufferVma(copy.buffer, copy.allocation);
            }
        });
    };

    // Compact merged vertex/index data to reclaim space from removed meshes.
    // Without this, every brush-animation frame appends new geometry while stale
    // data from prior frames accumulates indefinitely (GPU memory leak).
    if (activeMeshCount < meshes.size()) {
        std::vector<Vertex> compactVerts;
        std::vector<uint32_t> compactIndices;
        size_t totalVerts = 0, totalIndices = 0;
        for (const auto& kv : meshes) {
            if (!kv.second.active) continue;
            totalVerts += kv.second.vertexCount;
            totalIndices += kv.second.indexCount;
        }
        compactVerts.reserve(totalVerts);
        compactIndices.reserve(totalIndices);

        for (auto& kv : meshes) {
            MeshInfo& info = kv.second;
            if (!info.active) continue;

            uint32_t oldBaseVertex = info.baseVertex;
            uint32_t oldFirstIndex = info.firstIndex;
            uint32_t vCount = info.vertexCount;
            uint32_t iCount = info.indexCount;

            info.baseVertex = static_cast<uint32_t>(compactVerts.size());
            info.firstIndex = static_cast<uint32_t>(compactIndices.size());

            auto vertStart = mergedVertices.begin() + oldBaseVertex;
            compactVerts.insert(compactVerts.end(), vertStart, vertStart + vCount);

            auto idxStart = mergedIndices.begin() + oldFirstIndex;
            compactIndices.insert(compactIndices.end(), idxStart, idxStart + iCount);
        }

        mergedVertices = std::move(compactVerts);
        mergedIndices = std::move(compactIndices);
    }

    // Calculate required capacity with 25% headroom for incremental adds
    size_t neededVertexCap = mergedVertices.size() + mergedVertices.size() / 4 + 1024;
    size_t neededIndexCap = mergedIndices.size() + mergedIndices.size() / 4 + 4096;
    size_t neededMeshCap = activeMeshCount + activeMeshCount / 4 + 64;
    
    // Save old capacities before potentially updating them (used to decide
    // whether existing GPU buffers can be reused vs. needing recreation).
    size_t oldVertexCapacity = vertexCapacity;
    size_t oldIndexCapacity = indexCapacity;
    size_t oldMeshCapacity = meshCapacity;

    // Use max of current capacity or needed capacity (never shrink)
    if (neededVertexCap > vertexCapacity) vertexCapacity = neededVertexCap;
    if (neededIndexCap > indexCapacity) indexCapacity = neededIndexCap;
    if (neededMeshCap > meshCapacity) meshCapacity = neededMeshCap;

    // Build merged GPU-side vertex and index buffers from the CPU arrays.
    // If there are no meshes, free existing buffers.
    static bool printedBufferInfo = false;
    if (!printedBufferInfo) {
        printedBufferInfo = true;
    }
    // Capture the geometry that was current BEFORE this rebuild so it can be
    // recycled once its in-flight frames retire (see recyclePreviousGeom).
    uint32_t prevSlot = currentGeomSlot;
    Buffer prevVertex = vertexBuffer;
    Buffer prevIndex = indexBuffer;
    (void)oldVertexCapacity; (void)oldIndexCapacity;

    if (mergedVertices.empty() || mergedIndices.empty()) {
        // No geometry: recycle the previous slot/buffers and clear the mirror.
        recyclePreviousGeom(app, prevSlot, prevVertex, prevIndex);
        currentGeomSlot = UINT32_MAX;
        vertexBuffer = {};
        indexBuffer = {};
        vertexCapacity = 0;
        indexCapacity = 0;
    } else {
        // Pick a free pool slot to receive the fresh full copy. The previous
        // "current" slot is still in-use (recycled below via a frame-fence
        // callback), so acquireGeomSlot never returns it.
        uint32_t slot = acquireGeomSlot();
        VkDeviceSize vertexBufferSize = vertexCapacity * sizeof(Vertex);
        VkDeviceSize indexBufferSize = indexCapacity * sizeof(uint32_t);

        if (slot != UINT32_MAX) {
            // Create the slot's buffers on first use, or grow them if capacity
            // increased (never shrink). Growth defer-destroys the old undersized
            // buffers; the common steady-state path reuses them in place.
            if (vertexSlots[slot].buffer == VK_NULL_HANDLE || vertexSlotCap[slot] < vertexCapacity) {
                if (vertexSlots[slot].buffer != VK_NULL_HANDLE) scheduleDestroyBuffer(vertexSlots[slot]);
                vertexSlots[slot] = app->createBuffer(vertexBufferSize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                vertexSlotCap[slot] = vertexCapacity;
            }
            if (indexSlots[slot].buffer == VK_NULL_HANDLE || indexSlotCap[slot] < indexCapacity) {
                if (indexSlots[slot].buffer != VK_NULL_HANDLE) scheduleDestroyBuffer(indexSlots[slot]);
                indexSlots[slot] = app->createBuffer(indexBufferSize,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                indexSlotCap[slot] = indexCapacity;
            }
            vertexBuffer = vertexSlots[slot];
            indexBuffer = indexSlots[slot];
        } else {
            // Pool exhausted (rare burst): allocate throwaway buffers reclaimed
            // once the frame that uses them retires. Bounded — never accumulates
            // like deferDestroyUntilAllPending did.
            vertexBuffer = app->createBuffer(vertexBufferSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            indexBuffer = app->createBuffer(indexBufferSize,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        currentGeomSlot = slot;

        // Recycle the previous geometry now that the new slot is current.
        recyclePreviousGeom(app, prevSlot, prevVertex, prevIndex);

        // Upload current data via staging → device-local copy into the freshly
        // selected slot. That slot is either a brand-new buffer or one whose
        // last frame has retired (guaranteed by the fence-gated recycle above),
        // so no in-flight frame is reading it — no WRITE_AFTER_READ hazard, and
        // no need for a device-wide stall.
        bool doVertexUpload = !mergedVertices.empty();
        bool doIndexUpload = !mergedIndices.empty();
        VkDeviceSize vertexDataSize = mergedVertices.size() * sizeof(Vertex);
        VkDeviceSize indexDataSize = mergedIndices.size() * sizeof(uint32_t);

        if (doVertexUpload || doIndexUpload) {
            VkDeviceSize stagingSize = (doVertexUpload ? vertexDataSize : 0)
                                     + (doIndexUpload  ? indexDataSize  : 0);
            Buffer staging = app->createBuffer(stagingSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* mapped = nullptr;
            mapped = staging.map(0);
            VkDeviceSize offset = 0;
            if (doVertexUpload) {
                std::memcpy(static_cast<char*>(mapped) + offset, mergedVertices.data(), vertexDataSize);
                offset += vertexDataSize;
            }
            if (doIndexUpload) {
                std::memcpy(static_cast<char*>(mapped) + offset, mergedIndices.data(), indexDataSize);
            }
            staging.unmap(); // VMA persistent mapping

            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                VkDeviceSize off = 0;
                if (doVertexUpload) {
                    VkBufferCopy vCopy{};
                    vCopy.size = vertexDataSize;
                    vkCmdCopyBuffer(cmd, staging.buffer, vertexBuffer.buffer, 1, &vCopy);
                    off += vertexDataSize;
                }
                if (doVertexUpload && doIndexUpload) {
                    VkBufferMemoryBarrier2 barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.buffer = vertexBuffer.buffer;
                    barrier.offset = 0;
                    barrier.size = vertexDataSize;
                    VkDependencyInfo depInfo{};
                    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depInfo.bufferMemoryBarrierCount = 1;
                    depInfo.pBufferMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(cmd, &depInfo);
                }
                if (doIndexUpload) {
                    VkBufferCopy iCopy{};
                    iCopy.srcOffset = off;
                    iCopy.size = indexDataSize;
                    vkCmdCopyBuffer(cmd, staging.buffer, indexBuffer.buffer, 1, &iCopy);
                }
            });

            app->resources.removeBufferVma(staging.buffer, staging.allocation);
        }
    }

    // Rebuild indirect command list from active meshes so GPU-side compaction matches models/bounds
    std::vector<VkDrawIndexedIndirectCommand> cmds;
    cmds.reserve(meshes.size());
    for (auto& kv : meshes) {
        MeshInfo& info = kv.second;
        if (!info.active) continue;
        info.drawIndex = static_cast<uint32_t>(cmds.size());
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
        cmd.firstInstance = info.drawIndex;
        cmds.push_back(cmd);
    }
    indirectCommands = cmds;

    // Create or update the global indirect buffer with capacity-based sizing
    // Use host-visible memory for AMD RADV driver compatibility
    VkDeviceSize indirectBufferSize = sizeof(VkDrawIndexedIndirectCommand) * meshCapacity;
    VkDeviceSize indirectDataSize = sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size();
    bool needNewIndirectBuffer = (indirectBuffer.buffer == VK_NULL_HANDLE) || (meshCapacity > oldMeshCapacity);
    if (needNewIndirectBuffer) {
        if (indirectBuffer.buffer != VK_NULL_HANDLE || indirectBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(indirectBuffer);
            indirectBuffer = {};
        }
        if (meshCapacity > 0) {
            indirectBuffer = app->createBuffer(indirectBufferSize, 
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    }
    // Write data to the (existing or newly-created) indirect buffer
    if (indirectBuffer.buffer != VK_NULL_HANDLE) {
        void* data;
        data = indirectBuffer.map(0);
        // Zero the entire buffer, then copy only the valid commands. The
        // capacity headroom must never hold allocator garbage, otherwise the
        // compute cull shader reads invalid DrawCmd entries as visible.
        memset(data, 0, (size_t)indirectBufferSize);
        if (indirectDataSize > 0) {
            memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
        }
        indirectBuffer.unmap(); // VMA persistent mapping
    }

    // Mark per-mesh indirect offsets (byte offsets inside indirect buffer).
    // `meshes` is an unordered_map keyed by mesh id, so never index it as an array.
    VkDeviceSize offsetCursor = 0;
    for (auto& kv : meshes) {
        MeshInfo& info = kv.second;
        if (!info.active) continue;
        info.indirectOffset = offsetCursor;
        offsetCursor += sizeof(VkDrawIndexedIndirectCommand);
    }

    // Models SSBO removed: shaders use identity model matrices, no modelsBuffer

    // Upload bounds SSBO (three vec4s per active mesh: min, max, lod meta)
    std::vector<glm::vec4> boundsData;
    boundsData.reserve(meshes.size() * 3);
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        boundsData.push_back(info.boundsMin);
        boundsData.push_back(info.boundsMax);
        boundsData.push_back(glm::vec4(0.0f)); // meta: unused (no LoD bands)
    }
    VkDeviceSize boundsBufferSize = sizeof(glm::vec4) * meshCapacity * 3;
    VkDeviceSize boundsDataSize = sizeof(glm::vec4) * boundsData.size();
    bool needNewBoundsBuffer = (boundsBuffer.buffer == VK_NULL_HANDLE) || (meshCapacity > oldMeshCapacity);
    if (needNewBoundsBuffer) {
        if (boundsBuffer.buffer != VK_NULL_HANDLE || boundsBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(boundsBuffer);
            boundsBuffer = {};
        }
        if (meshCapacity > 0) {
            boundsBuffer = app->createBuffer(boundsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    }
    // Write data to the (existing or newly-created) bounds buffer
    if (boundsBuffer.buffer != VK_NULL_HANDLE && boundsDataSize > 0) {
        void* bdata;
        bdata = boundsBuffer.map(0);
        memcpy(bdata, boundsData.data(), (size_t)boundsDataSize);
        boundsBuffer.unmap(); // VMA persistent mapping
    }

    // ── Vegetation per-draw table (binding 9 of indirect.comp) ──
    // Indexed by the solid draw entry index s (== MeshInfo::drawIndex). For each
    // active solid chunk that carries vegetation, store {instanceCount,
    // firstInstance, 0, 0}; entries without vegetation stay zeroed. The merged
    // dispatch reads this table inside processVegetation(s).
    {
        VkDeviceSize vegTableSize = sizeof(glm::vec4) * meshCapacity;
        bool needNewVegTable = (vegTableBuffer.buffer == VK_NULL_HANDLE) || (meshCapacity > vegTableCapacity);
        if (needNewVegTable) {
            if (vegTableBuffer.buffer != VK_NULL_HANDLE) {
                if (vegTableMapped) { vegTableBuffer.unmap(); vegTableMapped = nullptr; }
                scheduleDestroyBuffer(vegTableBuffer);
                vegTableBuffer = {};
            }
            if (meshCapacity > 0) {
                vegTableBuffer = app->createBuffer(vegTableSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                vegTableMapped = vegTableBuffer.map(0);
            }
            vegTableCapacity = meshCapacity;
        }
        if (vegTableBuffer.buffer != VK_NULL_HANDLE) {
            std::vector<glm::vec4> vt(meshCapacity, glm::vec4(0.0f));
            for (const auto& kv : meshes) {
                const MeshInfo& info = kv.second;
                if (!info.active) continue;
                auto it = vegChunkInfoMap.find(kv.first);
                if (it != vegChunkInfoMap.end() && info.drawIndex < meshCapacity)
                    vt[info.drawIndex] = it->second;
            }
            if (vegTableMapped) {
                memcpy(vegTableMapped, vt.data(), (size_t)vegTableSize);
            } else {
                void* data = vegTableBuffer.map(0);
                memcpy(data, vt.data(), (size_t)vegTableSize);
                vegTableBuffer.unmap();
            }
        }
    }

    // Create/resize compact indirect buffer (storage + indirect usage)
    // Written by compute shader every frame, read by indirect draw — DEVICE_LOCAL
    // for optimal GPU performance on discrete GPUs.
    VkDeviceSize compactSize = indirectBufferSize;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        if (compactIndirectBuffers[f].buffer != VK_NULL_HANDLE || compactIndirectBuffers[f].memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(compactIndirectBuffers[f]);
            compactIndirectBuffers[f] = {};
        }
        if (compactSize > 0) {
            compactIndirectBuffers[f] = app->createBuffer(compactSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (indirectDataSize > 0) {
                void* data;
                data = compactIndirectBuffers[f].map(0);
                // Zero the ENTIRE buffer first so any headroom (capacity beyond
                // the valid command count) is never left as uninitialized
                // allocator garbage that could be read as a giant indexCount by
                // vkCmdDrawIndexedIndirectCount and spin the GPU.
                memset(data, 0, (size_t)compactSize);
                memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
                compactIndirectBuffers[f].unmap(); // VMA persistent mapping
            }
        }
    }

    // Per-frame chosen-LoD output buffers (uvec2 per entry) + the scratch
    // buffer bound by external descriptor-set owners (cubemap/backface).
    // Same lifecycle as compactIndirectBuffers.
    VkDeviceSize lodBufSize = sizeof(glm::uvec2) * meshCapacity;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        if (visibleLodBuffers[f].buffer != VK_NULL_HANDLE || visibleLodBuffers[f].memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(visibleLodBuffers[f]);
            visibleLodBuffers[f] = {};
        }
        if (lodBufSize > 0) {
            visibleLodBuffers[f] = app->createBuffer(lodBufSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* data = visibleLodBuffers[f].map(0);
            if (data) {
                std::memset(data, 0, (size_t)lodBufSize);
                visibleLodBuffers[f].unmap();
            }
        }
    }
    if (visibleLodsScratch.buffer == VK_NULL_HANDLE && lodBufSize > 0) {
        // TRANSFER_DST required: prepareCullWithDescriptor clears it with
        // vkCmdFillBuffer each cull.
        visibleLodsScratch = app->createBuffer(lodBufSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data = visibleLodsScratch.map(0);
        if (data) {
            std::memset(data, 0, (size_t)lodBufSize);
            visibleLodsScratch.unmap();
        }
    }

    // Create or zero the per-frame visible count buffers.
    VkDeviceSize countSize = sizeof(uint32_t);
    uint32_t initialCount = static_cast<uint32_t>(indirectCommands.size());
    VkDevice dev = app->getDevice();
    storedDevice = dev;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        // Unmap old persistent mapping before destroying
        if (visibleCountMapped[f]) {
            visibleCountBuffers[f].unmap(); // VMA persistent mapping
            visibleCountMapped[f] = nullptr;
        }
        if (visibleCountBuffers[f].buffer != VK_NULL_HANDLE || visibleCountBuffers[f].memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(visibleCountBuffers[f]);
            visibleCountBuffers[f] = {};
        }
        visibleCountBuffers[f] = app->createBuffer(countSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Persistently map for host-side zeroing (avoids vkCmdFillBuffer + barrier issues on RADV)
        visibleCountMapped[f] = static_cast<uint32_t*>(visibleCountBuffers[f].map(0));
        // Initialize with full count (fallback when culling is off)
        *visibleCountMapped[f] = initialCount;
    }

    // Create compute pipeline + descriptor sets for GPU culling if not present
    if (computePipeline == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding bindings[10] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].binding = 2;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].binding = 3;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].binding = 4;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // 5,6: vegetation impostor command + count streams.
        // 7,8: vegetation billboard command + count streams.
        // 9: vegetation chunk-info table (input). When vegetation culling is
        // disabled these are bound to a dummy buffer (the shader statically
        // references them, so they must always be valid).
        bindings[5].binding = 5;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[6].binding = 6;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[7].binding = 7;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[8].binding = 8;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[9].binding = 9;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        VkDescriptorBindingFlags bindingFlags[10] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        };

        DescriptorAllocator descAlloc{app->getDevice(), app};
        computeDescriptorSetLayout = descAlloc.createLayout(
            bindings, 10,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            bindingFlags,
            "IndirectRenderer: computeDescriptorSetLayout");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(CullPushConstants); // 104 bytes: mat4 + 2*uint + pad + vec3 + float + uint

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        plinfo.pSetLayouts = &computeDescriptorSetLayout;
        plinfo.pushConstantRangeCount = 1;
        plinfo.pPushConstantRanges = &pc;

        if (vkCreatePipelineLayout(app->getDevice(), &plinfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        // central manager
        app->resources.addPipelineLayout(computePipelineLayout, "IndirectRenderer: computePipelineLayout");

        VkShaderModule compModule = app->getOrCreateShaderModule("shaders/indirect.comp.spv");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = compModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stage;
        pipelineInfo.layout = computePipelineLayout;

        if (vkCreateComputePipelines(app->getDevice(), app->getPipelineCache(), 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            // Shader module is cached by VulkanApp — do not destroy it even on error.
            throw std::runtime_error("failed to create compute pipeline!");
        }
        // track compute pipeline
        app->resources.addPipeline(computePipeline, "IndirectRenderer: computePipeline");

        VkDescriptorPoolSize irPoolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256};
        computeDescriptorPool = descAlloc.createPool(
            &irPoolSize, 1, 64,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            "IndirectRenderer: computeDescriptorPool");

        descAlloc.allocateSets(computeDescriptorPool, computeDescriptorSetLayout,
                               MAX_CULL_FRAMES, reinterpret_cast<VkDescriptorSet*>(computeDescriptorSets.data()),
                               "IndirectRenderer: computeDescriptorSet");

        // Tiny dummy bound to the vegetation bindings (5..9) on the solid-only
        // dispatch so the layout's statically-referenced bindings are always valid.
        // prepareCull re-points these to the real veg buffers when vegetation is enabled.
        if (vegDummyBuffer.buffer == VK_NULL_HANDLE) {
            vegDummyBuffer = app->createBuffer(16,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
    }

    // Update per-frame compute descriptor sets with buffer infos
    VkDescriptorBufferInfo inBuf{};
    inBuf.buffer = indirectBuffer.buffer;
    inBuf.offset = 0;
    inBuf.range = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo boundsBufInfo{};
    boundsBufInfo.buffer = boundsBuffer.buffer;
    boundsBufInfo.offset = 0;
    boundsBufInfo.range = VK_WHOLE_SIZE;

    bool anyNull = (indirectBuffer.buffer == VK_NULL_HANDLE || boundsBuffer.buffer == VK_NULL_HANDLE);
    for (uint32_t f = 0; f < MAX_CULL_FRAMES && !anyNull; f++) {
        if (compactIndirectBuffers[f].buffer == VK_NULL_HANDLE || visibleCountBuffers[f].buffer == VK_NULL_HANDLE) {
            anyNull = true;
        }
    }
    if (anyNull) {
        std::cerr << "[IndirectRenderer] Skipping compute descriptor set update: one or more buffers are VK_NULL_HANDLE" << std::endl;
    } else {
        for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
            VkDescriptorBufferInfo outBuf{};
            outBuf.buffer = compactIndirectBuffers[f].buffer;
            outBuf.offset = 0;
            outBuf.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo countBuf{};
            countBuf.buffer = visibleCountBuffers[f].buffer;
            countBuf.offset = 0;
            countBuf.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo lodBuf{};
            lodBuf.buffer = visibleLodBuffers[f].buffer;
            lodBuf.offset = 0;
            lodBuf.range = VK_WHOLE_SIZE;

            VkDescriptorSet computeDs = computeDescriptorSets[f];
            DescriptorWriter(app->getDevice())
                .writeBuffer(computeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             inBuf.buffer, inBuf.offset, inBuf.range)
                .writeBuffer(computeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             outBuf.buffer, outBuf.offset, outBuf.range)
                .writeBuffer(computeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             boundsBufInfo.buffer, boundsBufInfo.offset, boundsBufInfo.range)
                .writeBuffer(computeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             countBuf.buffer, countBuf.offset, countBuf.range)
                 .writeBuffer(computeDs, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              lodBuf.buffer, lodBuf.offset, lodBuf.range)
                 .writeBuffer(computeDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
                 .writeBuffer(computeDs, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
                 .writeBuffer(computeDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
                 .writeBuffer(computeDs, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
                 .writeBuffer(computeDs, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
                 .flush();
        }
    }

    // Try to load device function for indirect-count draws; require it.
    cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCountKHR");
    if (!cmdDrawIndexedIndirectCount) {
        cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCount");
    }
    if (!cmdDrawIndexedIndirectCount) {
        throw std::runtime_error("Required device function vkCmdDrawIndexedIndirectCountKHR is not available");
    }

    // Models SSBO removed: skip updating main descriptor set for models
    descriptorDirty = false;
    pendingDescriptorSet = VK_NULL_HANDLE;

    dirty = false;
    // rebuild() already wrote all indirect/bounds entries via memcpy above,
    // so mark every active mesh as written.  Without this the append-only
    // doUploadMeshMetaBuffers would treat the buffer as empty and rewrite
    // every entry — harmless but wasteful — and needsFullRebuild() (which
    // checks metaBuffersWrittenCount == 0) would force unnecessary rebuilds
    // on every subsequent incremental batch.
    metaBuffersWrittenCount = indirectCommands.size();
}

void IndirectRenderer::setCullFrame(uint32_t frame) {
    currentCullFrame = frame % MAX_CULL_FRAMES;
}

void IndirectRenderer::setVegetationCullData(const std::array<Buffer, MAX_CULL_FRAMES>& bbCompact,
                                             const std::array<Buffer, MAX_CULL_FRAMES>& bbCount,
                                             const std::array<Buffer, MAX_CULL_FRAMES>& impCompact,
                                             const std::array<Buffer, MAX_CULL_FRAMES>& impCount) {
    for (uint32_t i = 0; i < MAX_CULL_FRAMES; i++) {
        vegBbCompactBuf[i] = bbCompact[i].buffer;
        vegBbCountBuf[i]   = bbCount[i].buffer;
        vegImpCompactBuf[i] = impCompact[i].buffer;
        vegImpCountBuf[i]  = impCount[i].buffer;
    }
    vegCullEnabled = (vegBbCompactBuf[0] != VK_NULL_HANDLE && vegBbCountBuf[0] != VK_NULL_HANDLE &&
                      vegImpCompactBuf[0] != VK_NULL_HANDLE && vegImpCountBuf[0] != VK_NULL_HANDLE);
}

void IndirectRenderer::setVegetationChunkInfo(const std::unordered_map<uint32_t, glm::vec4>& info) {
    vegChunkInfoMap = info;
    // The veg table is rebuilt during the next buildDrawList (which knows the
    // solid draw entry indices). Force a rebuild so the map is reflected.
    setDirty(true);
}

void IndirectRenderer::updateVegTable() {
    if (meshCapacity == 0) return;
    VkDeviceSize vegTableSize = sizeof(glm::vec4) * meshCapacity;
    bool needNew = (vegTableBuffer.buffer == VK_NULL_HANDLE) || (meshCapacity > vegTableCapacity);
    if (needNew) {
        if (vegTableBuffer.buffer != VK_NULL_HANDLE) {
            if (vegTableMapped) { vegTableBuffer.unmap(); vegTableMapped = nullptr; }
            Buffer oldVegTable = vegTableBuffer;
            vegTableBuffer = {};
            app_->deferDestroyUntilAllPending([app_ = app_, oldVegTable]() {
                if (oldVegTable.buffer != VK_NULL_HANDLE)
                    app_->resources.removeBufferVma(oldVegTable.buffer, oldVegTable.allocation);
            });
        }
        if (meshCapacity > 0) {
            vegTableBuffer = app_->createBuffer(vegTableSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vegTableMapped = vegTableBuffer.map(0);
        }
        vegTableCapacity = meshCapacity;
    }
    if (vegTableBuffer.buffer == VK_NULL_HANDLE) return;
    std::vector<glm::vec4> vt(meshCapacity, glm::vec4(0.0f));
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        auto it = vegChunkInfoMap.find(kv.first);
        if (it != vegChunkInfoMap.end() && info.drawIndex < meshCapacity)
            vt[info.drawIndex] = it->second;
    }
    if (vegTableMapped) {
        memcpy(vegTableMapped, vt.data(), (size_t)vegTableSize);
    } else {
        void* data = vegTableBuffer.map(0);
        memcpy(data, vt.data(), (size_t)vegTableSize);
        vegTableBuffer.unmap();
    }
}

void IndirectRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj,
                                    glm::vec3 camPos, float lodBias, int maxTargetLod) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    // and all buffer modifications happen in rebuild() which does lock.

    // Flush staged host-side meta writes via GPU copies BEFORE any barrier or
    // dispatch: the copies (TRANSFER_WRITE) are ordered before the cull reads
    // (SHADER_READ) by the acquireBuffers barrier below, and queue-ordered
    // after every in-flight frame's reads of the same entries.
    flushStagedMetaWrites(cmd, currentCullFrame);

    Buffer& compactBuf = compactIndirectBuffers[currentCullFrame];
    Buffer& visibleCount = visibleCountBuffers[currentCullFrame];
    Buffer& visibleLods = visibleLodBuffers[currentCullFrame];
    VkDescriptorSet descSet = computeDescriptorSets[currentCullFrame];

    if (computePipeline == VK_NULL_HANDLE || compactBuf.buffer == VK_NULL_HANDLE) {
        // No meshes loaded yet (e.g. during parallel background loading). Nothing to cull.
        return;
    }

    // Acquire uploaded geometry/meta buffers (written by async vkCmdCopyBuffer /
    // host staging) so the cull dispatch and subsequent indirect draw observe
    // their TRANSFER/HOST writes. Without this, the draw's VERTEX_ATTRIBUTE_READ
    // races the async transfer write (SYNC-HAZARD-READ-AFTER-WRITE).
    acquireBuffers(cmd);

    static bool printedOnce = false;
    if (!printedOnce) {
        printedOnce = true;
    }
    
    // Barrier A: ensure prior indirect-draw reads, compute-shader atomics,
    // and prior fills of visibleCount/compactBuf are complete before we fill 0
    // again. Without this, a second cascade's vkCmdFillBuffer races with the
    // first cascade's fill (TRANSFER_WRITE→TRANSFER_WRITE hazard) and with the
    // first cascade's compute-shader atomicAdd (SHADER_WRITE→TRANSFER_WRITE).
    {
        VkBufferMemoryBarrier2 readBarriers[3] = {};
        readBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        readBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                  | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                  | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        readBarriers[0].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                                  | VK_ACCESS_2_SHADER_READ_BIT
                                  | VK_ACCESS_2_SHADER_WRITE_BIT
                                  | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        readBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        readBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        readBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        readBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        readBarriers[0].buffer = visibleCount.buffer;
        readBarriers[0].offset = 0;
        readBarriers[0].size = VK_WHOLE_SIZE;

        readBarriers[1] = readBarriers[0];
        readBarriers[1].buffer = compactBuf.buffer;
        readBarriers[2] = readBarriers[0];
        readBarriers[2].buffer = visibleLods.buffer;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 3;
        depInfo.pBufferMemoryBarriers = readBarriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Reset visible count to zero via vkCmdFillBuffer (GPU-side write) so each
    // prepareCull starts from a clean slate on the GPU timeline.  A CPU host
    // write (HOST_COHERENT) would be overwritten by the previous cascade's
    // atomicAdd before the GPU executes, causing each subsequent compute to
    // start from the accumulated count rather than 0.  vkCmdFillBuffer is used
    // instead of vkCmdUpdateBuffer to avoid the latter's implicit FULL_QUEUE
    // barrier (top-of-pipe → bottom-of-pipe) that drains the entire graphics queue.
    vkCmdFillBuffer(cmd, visibleCount.buffer, 0, sizeof(uint32_t), 0);

    // Also zero the ENTIRE compact indirect buffer so any slot the compute
    // shader does NOT write (e.g. because it early-returns or the dst index
    // lands beyond the valid command count) is a clean zeroed DrawCmd
    // (indexCount=0) instead of stale/allocator garbage.  A non-zero
    // garbage indexCount read by vkCmdDrawIndexedIndirectCount would make
    // the GE process a draw with a giant index count and never finish
    // (GPU hang observed on RADV / Radeon 680M).
    vkCmdFillBuffer(cmd, compactBuf.buffer, 0, VK_WHOLE_SIZE, 0);

    // Zero the per-frame chosen-LoD output so entries beyond the current
    // dispatch range can never be misread as a stale (chunk, level) pair.
    vkCmdFillBuffer(cmd, visibleLods.buffer, 0, VK_WHOLE_SIZE, 0);

    // ── Vegetation merged-cull outputs ──
    // When vegetation culling is enabled the same dispatch also emits billboard
    // (binding 7/8) and impostor (binding 5/6) commands, reading the per-solid-
    // drawIndex vegetation table (binding 9). The output buffers are OWNED by
    // VegetationRenderer and simply (re)bound here; we zero their count buffers
    // so the atomic-adds start from a clean state, then re-point the descriptor
    // set's veg bindings to the real buffers.
    bool vegActive = vegCullEnabled && vegTableBuffer.buffer != VK_NULL_HANDLE
                     && vegBbCountBuf[currentCullFrame] != VK_NULL_HANDLE
                     && vegImpCountBuf[currentCullFrame] != VK_NULL_HANDLE;
    if (vegActive) {
        // Refresh the per-draw vegetation table (binding 9) from the latest
        // vegChunkInfoMap. The slotted renderer updates draw indices incrementally,
        // so this is rebuilt every frame (cheap memcpy) rather than only in rebuild().
        updateVegTable();
        // Zero the veg output buffers so any slot the merged dispatch does NOT write
        // (or the atomic count limits) stays a clean zeroed DrawCmd (indexCount=0).
        vkCmdFillBuffer(cmd, vegBbCompactBuf[currentCullFrame], 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, vegImpCompactBuf[currentCullFrame], 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, vegBbCountBuf[currentCullFrame], 0, sizeof(uint32_t), 0);
        vkCmdFillBuffer(cmd, vegImpCountBuf[currentCullFrame], 0, sizeof(uint32_t), 0);
        DescriptorWriter(app_->getDevice())
            .writeBuffer(computeDescriptorSets[currentCullFrame], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegImpCompactBuf[currentCullFrame], 0, VK_WHOLE_SIZE)
            .writeBuffer(computeDescriptorSets[currentCullFrame], 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegImpCountBuf[currentCullFrame], 0, VK_WHOLE_SIZE)
            .writeBuffer(computeDescriptorSets[currentCullFrame], 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegBbCompactBuf[currentCullFrame], 0, VK_WHOLE_SIZE)
            .writeBuffer(computeDescriptorSets[currentCullFrame], 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegBbCountBuf[currentCullFrame], 0, VK_WHOLE_SIZE)
            .writeBuffer(computeDescriptorSets[currentCullFrame], 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         vegTableBuffer.buffer, 0, VK_WHOLE_SIZE)
            .flush();
    }

    // Barrier B: ensure the transfer write (zeroCount) and any prior
    // indirect-draw reads of compactBuf are complete before the compute
    // shader writes to both buffers.
    {
        VkBufferMemoryBarrier2 preBarriers[7] = {};
        preBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        preBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preBarriers[0].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        preBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarriers[0].buffer = compactBuf.buffer;
        preBarriers[0].offset = 0;
        preBarriers[0].size = VK_WHOLE_SIZE;

        preBarriers[1] = preBarriers[0];
        preBarriers[1].buffer = visibleCount.buffer;
        preBarriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        preBarriers[2] = preBarriers[0];
        preBarriers[2].buffer = visibleLods.buffer;
        preBarriers[2].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;

        uint32_t barrierCount = 3;
        if (vegActive) {
            preBarriers[3] = preBarriers[0];
            preBarriers[3].buffer = vegBbCountBuf[currentCullFrame];
            preBarriers[3].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preBarriers[3].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            preBarriers[4] = preBarriers[0];
            preBarriers[4].buffer = vegImpCountBuf[currentCullFrame];
            preBarriers[4].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preBarriers[4].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            // The veg compact output buffers were just zero-filled (TRANSFER_WRITE)
            // and are about to be written by the dispatch (COMPUTE SHADER_WRITE),
            // so they need a transfer→compute barrier too.
            preBarriers[5] = preBarriers[0];
            preBarriers[5].buffer = vegBbCompactBuf[currentCullFrame];
            preBarriers[5].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preBarriers[5].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            preBarriers[6] = preBarriers[0];
            preBarriers[6].buffer = vegImpCompactBuf[currentCullFrame];
            preBarriers[6].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preBarriers[6].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrierCount = 7;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = barrierCount;
        depInfo.pBufferMemoryBarriers = preBarriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Bind and dispatch compute cull
    if (cmdState) cmdState->bindComputePipeline(cmd, computePipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    if (cmdState) cmdState->bindComputeDescriptorSets(cmd, computePipelineLayout, 0, 1, &descSet, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, nullptr);
    uint32_t numCmds = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        numCmds = getCullDispatchCountLocked();
#ifdef DEBUG
        // DIAG: watch mesh-map growth (user-reported draw-cmd accumulation).
        // meshes.size() = total entries ever inserted (never erased),
        // activeCount = entries the cull pass actually processes.
        static std::chrono::steady_clock::time_point lastDiag{};
        auto nowD = std::chrono::steady_clock::now();
        if (nowD - lastDiag >= std::chrono::seconds(1)) {
            lastDiag = nowD;
            std::cout << "[IndirectRenderer::diag] this=" << this
                      << " meshes.size=" << meshes.size()
                      << " active=" << activeMeshCountLocked()
                      << " numCmds=" << numCmds
                      << " slotActive=" << slotAlloc.activeCount()
                      << " slotCap=" << slotAlloc.capacity()
                      << std::endl;
        }
#endif
    }
    // Fast return if nothing to cull — avoids touching the pipeline at all
    if (numCmds == 0) {
        // Count active meshes only on this rare path (used by the warn-once
        // printf below); avoids a full-map scan on every frame in the common
        // cull path. Same value as a scan: memoized under the same mutex.
        uint32_t activeInMeshes = 0;
        {
            std::lock_guard<std::recursive_mutex> lock(mutex);
            activeInMeshes = static_cast<uint32_t>(activeMeshCountLocked());
        }
        static bool warned = false;
        if (!warned) { printf("[IndirectRenderer::prepareCull] EARLY RETURN: active=%u numCmds=%u slotted=%d\n", activeInMeshes, numCmds, (int)slottedMode); warned = true; }
        return;
    }

    CullPushConstants pc{};
    pc.viewProj     = viewProj;
    pc.targetLayer  = 0;
    pc.numCmds      = numCmds;
    pc.numCmdsVeg   = 0; // solid-only dispatch: no vegetation entries consumed
    pc.camPos       = camPos;
    pc.lodBias      = lodBias;
    pc.maxTargetLod = static_cast<uint32_t>(maxTargetLod);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants), &pc);

    uint32_t groupSize = 64;
    uint32_t groups = (numCmds + groupSize - 1) / groupSize;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier to make shader writes to the compact indirect buffer and visible count visible to indirect draw
    VkBufferMemoryBarrier2 barriers[7] = {};
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    // Publish the compute write to the indirect-draw + vertex consumers AND to a
    // subsequent cull dispatch: the next prepareCull (rotated buffer reuse, or a
    // later shadow cascade) reads visibleCount via atomicAdd, which is a
    // COMPUTE_SHADER storage read. Without COMPUTE_SHADER in the destination
    // scope this cross-dispatch compute-write -> compute-read is unsynchronized
    // (was previously masked by the removed deviceWaitIdle() in rebuildBrushScene).
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = compactBuf.buffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    barriers[1] = barriers[0];
    barriers[1].buffer = visibleCount.buffer;
    barriers[1].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

    // visibleLods is a third consumer of the dispatch writes: the coarse
    // chosen-LoD pass reads it in a follow-up dispatch.
    barriers[2] = barriers[0];
    barriers[2].buffer = visibleLods.buffer;
    barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    uint32_t barrierCount = 3;
    if (vegActive) {
        barriers[3] = barriers[0];
        barriers[3].buffer = vegBbCompactBuf[currentCullFrame];
        barriers[3].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        barriers[4] = barriers[0];
        barriers[4].buffer = vegImpCompactBuf[currentCullFrame];
        barriers[4].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        barriers[5] = barriers[0];
        barriers[5].buffer = vegBbCountBuf[currentCullFrame];
        barriers[5].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        barriers[6] = barriers[0];
        barriers[6].buffer = vegImpCountBuf[currentCullFrame];
        barriers[6].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        barrierCount = 7;
    }

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.bufferMemoryBarrierCount = barrierCount;
    depInfo.pBufferMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}



void IndirectRenderer::prepareCullWithDescriptor(VkCommandBuffer cmd, const glm::mat4& viewProj, VkDescriptorSet computeDesc,
                                                 VkBuffer outCompactBuffer, VkBuffer outVisibleCountBuffer,
                                                 glm::vec3 camPos, float lodBias, int maxTargetLod) {
    if (computePipeline == VK_NULL_HANDLE) {
        // No meshes loaded yet (e.g. during parallel background loading). Nothing to cull.
        return;
    }
    if (outCompactBuffer == VK_NULL_HANDLE || computeDesc == VK_NULL_HANDLE) {
        throw std::runtime_error("IndirectRenderer::prepareCullWithDescriptor requires valid outCompactBuffer and computeDesc");
    }

    // Acquire uploaded geometry/meta buffers (async vkCmdCopyBuffer / host staging)
    // so the cull dispatch and indirect draw observe their TRANSFER/HOST writes.
    acquireBuffers(cmd);

    // Reset visible count via host mapped write (outVisibleCountBuffer is HOST_VISIBLE|HOST_COHERENT).
    // vkCmdFillBuffer + TRANSFER_BIT barrier is unreliable on RADV.
    // The caller owns the buffer; we clear it with a global memory barrier + fill.
    // Insert a TRANSFER→TRANSFER barrier before the fill so consecutive face
    // culls (same buffer, e.g. the 6 cubemap faces) don't race (WRITE_AFTER_WRITE).
    // The caller's compact buffer is zeroed too (see below): entries the
    // dispatch does NOT write (culled chunks) must read as indexCount=0, never
    // as stale/allocator garbage that vkCmdDrawIndexedIndirectCount would
    // process as a giant draw (GE hang observed on RADV / Radeon 680M).
    {
        VkBufferMemoryBarrier2 preFill[3] = {};
        preFill[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        // Drain prior compute dispatches too: the caller-owned count buffer is
        // shared across faces/frames, and a previous face's vkCmdDispatch
        // atomicAdd writes must complete before this fill overwrites them
        // (WRITE_AFTER_WRITE). Only TRANSFER_WRITE here would leave
        // dispatch→fill→dispatch unordered (sync-validation hazard).
        preFill[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT
                            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preFill[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT
                            | VK_ACCESS_2_SHADER_WRITE_BIT;
        preFill[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preFill[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preFill[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preFill[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preFill[0].buffer = outVisibleCountBuffer;
        preFill[0].offset = 0;
        preFill[0].size = VK_WHOLE_SIZE;

        // The caller's descriptor set may bind this instance's shared
        // visibleLodsScratch as the chosen-LoD output (e.g. the solid-360
        // cubemap DS). That buffer is written by every face's dispatch, so a
        // prior dispatch's writes must complete before our fill overwrites
        // them — same WRITE_AFTER_WRITE reasoning as the count buffer.
        // (Skipped when the scratch hasn't been allocated yet — VK_NULL_HANDLE
        // is not a valid barrier buffer.)
        uint32_t preFillCount = 1;
        if (visibleLodsScratch.buffer != VK_NULL_HANDLE) {
            preFill[preFillCount] = preFill[0];
            preFill[preFillCount].buffer = visibleLodsScratch.buffer;
            preFill[preFillCount].size = VK_WHOLE_SIZE;
            ++preFillCount;
        }

        // The caller-owned compact buffer (same WRITE_AFTER_WRITE reasoning:
        // a prior face's dispatch or the main pass's fill wrote it).
        preFill[preFillCount] = preFill[0];
        preFill[preFillCount].buffer = outCompactBuffer;
        preFill[preFillCount].size = VK_WHOLE_SIZE;
        ++preFillCount;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = preFillCount;
        depInfo.pBufferMemoryBarriers = preFill;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
    vkCmdFillBuffer(cmd, outVisibleCountBuffer, 0, sizeof(uint32_t), 0);
    // Zero the ENTIRE caller-owned compact buffer so any entry the cull
    // dispatch does NOT write (frustum-culled or chunk with no surviving
    // level) is a clean zeroed DrawCmd (indexCount=0) instead of stale
    // data — a garbage indexCount here would make the indirect draw's GE
    // process a giant draw and never finish.
    vkCmdFillBuffer(cmd, outCompactBuffer, 0, VK_WHOLE_SIZE, 0);
    // Zero the shared chosen-LoD scratch as well: untouched entries from a
    // previous frame must never be misread as a stale (chunk, level) pair.
    if (visibleLodsScratch.buffer != VK_NULL_HANDLE)
        vkCmdFillBuffer(cmd, visibleLodsScratch.buffer, 0, VK_WHOLE_SIZE, 0);
    {
        VkMemoryBarrier2 fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fillBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        fillBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &fillBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Barrier before dispatch: the compact buffer may have been filled by a
    // prior vkCmdFillBuffer (e.g. main pass) or written by a previous face's
    // compute dispatch. Ensure that write is visible before this dispatch
    // writes to it again (TRANSFER_WRITE/SHADER_WRITE → COMPUTE hazard).
    // The shared visibleLodsScratch (written by this dispatch via the caller's
    // descriptor set) needs the same fill→compute ordering.
    {
        VkBufferMemoryBarrier2 compactBarriers[2] = {};
        compactBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        compactBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT
                                    | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        compactBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT
                                    | VK_ACCESS_2_SHADER_WRITE_BIT;
        compactBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        compactBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        compactBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        compactBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        compactBarriers[0].buffer = outCompactBuffer;
        compactBarriers[0].offset = 0;
        compactBarriers[0].size = VK_WHOLE_SIZE;

        uint32_t compactBarrierCount = 1;
        if (visibleLodsScratch.buffer != VK_NULL_HANDLE) {
            compactBarriers[compactBarrierCount] = compactBarriers[0];
            compactBarriers[compactBarrierCount].buffer = visibleLodsScratch.buffer;
            compactBarriers[compactBarrierCount].size = VK_WHOLE_SIZE;
            ++compactBarrierCount;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = compactBarrierCount;
        depInfo.pBufferMemoryBarriers = compactBarriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Bind and dispatch compute cull using caller-provided descriptor set
    if (cmdState) cmdState->bindComputePipeline(cmd, computePipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    if (cmdState) cmdState->bindComputeDescriptorSets(cmd, computePipelineLayout, 0, 1, &computeDesc, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDesc, 0, nullptr);

    uint32_t numCmds = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        numCmds = getCullDispatchCountLocked();
    }
    // Fast return if nothing to cull — avoids touching the pipeline at all
    if (numCmds == 0) return;

    CullPushConstants pc2{};
    pc2.viewProj     = viewProj;
    pc2.targetLayer  = 0;
    pc2.numCmds      = numCmds;
    pc2.numCmdsVeg   = 0; // solid-only dispatch: no vegetation entries consumed
    pc2.camPos       = camPos;
    pc2.lodBias      = lodBias;
    pc2.maxTargetLod = static_cast<uint32_t>(maxTargetLod);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants), &pc2);

    uint32_t groupSize = 64;
    uint32_t groups = (numCmds + groupSize - 1) / groupSize;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier to make shader writes to the compact indirect buffer and visible count visible to indirect draw
    VkBufferMemoryBarrier2 barriers[2] = {};
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    // Also publish to COMPUTE_SHADER: a later cascade/face cull reuses/reads the
    // count buffer via atomicAdd (storage read). See prepareCull for rationale.
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = outCompactBuffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    barriers[1] = barriers[0];
    barriers[1].buffer = outVisibleCountBuffer;
    barriers[1].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.bufferMemoryBarrierCount = 2;
    depInfo.pBufferMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void IndirectRenderer::drawPreparedWithBuffers(VkCommandBuffer cmd, VkBuffer compactBuffer, VkBuffer visibleCountBuffer, uint32_t maxDraws) {
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
        static bool reported = false;
        if (!reported) {
            printf("[IndirectRenderer::drawPreparedWithBuffers] vertex or index buffer is NULL, skipping\n");
            reported = true;
        }
        return;
    }

    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    uint32_t bufMaxCount = static_cast<uint32_t>(meshCapacity);
    if (maxCount > bufMaxCount) {
        std::cerr << "[drawPreparedWithBuffers] CLAMPING maxCount from " << maxCount << " to meshCapacity " << bufMaxCount << std::endl;
        maxCount = bufMaxCount;
    }
    if (maxCount == 0) return; // nothing to draw — avoid calling indirect draw with 0 maxDraw

    if (!cmdDrawIndexedIndirectCount) {
        throw std::runtime_error("vkCmdDrawIndexedIndirectCountKHR not available (draw-indirect-count required)");
    }
    cmdDrawIndexedIndirectCount(cmd, compactBuffer, 0, visibleCountBuffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
}

void IndirectRenderer::drawPrepared(VkCommandBuffer cmd, uint32_t maxDraws) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
        static bool reported = false;
        if (!reported) {
            printf("[IndirectRenderer::drawPrepared] vertex or index buffer is NULL, skipping\n");
            reported = true;
        }
        return;
    }

    // Bind merged geometry
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Issue indirect-draw call; compute shader compacts only visible commands
    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    uint32_t bufMaxCount = static_cast<uint32_t>(meshCapacity);
    if (maxCount > bufMaxCount) {
        std::cerr << "[drawPrepared] CLAMPING maxCount from " << maxCount << " to meshCapacity " << bufMaxCount << std::endl;
        maxCount = bufMaxCount;
    }
    if (!cmdDrawIndexedIndirectCount) {
        throw std::runtime_error("vkCmdDrawIndexedIndirectCountKHR not available (draw-indirect-count required)");
    }
    // Use indirect-count variant to let the GPU supply the visible count from compute shader
    cmdDrawIndexedIndirectCount(cmd, compactIndirectBuffers[currentCullFrame].buffer, 0, visibleCountBuffers[currentCullFrame].buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
}

void IndirectRenderer::bindBuffers(VkCommandBuffer cmd) {
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
}

void IndirectRenderer::drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws) {
    drawIndirectOnly(cmd, app->getPipelineLayout(), maxDraws);
}

void IndirectRenderer::drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws) {
    Buffer& compactBuf = compactIndirectBuffers[currentCullFrame];
    Buffer& visibleCount = visibleCountBuffers[currentCullFrame];
    if (compactBuf.buffer == VK_NULL_HANDLE) {
        static bool reported = false;
        if (!reported) {
            printf("[IndirectRenderer::drawIndirectOnly] compactIndirectBuffer is VK_NULL_HANDLE, no draws\n");
            reported = true;
        }
        return;
    }
    // No per-draw model push-constants: models are identity in shaders.

    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    uint32_t bufMaxCount = static_cast<uint32_t>(meshCapacity);
    if (maxCount > bufMaxCount) {
        std::cerr << "[drawIndirectOnly] CLAMPING maxCount from " << maxCount << " to meshCapacity " << bufMaxCount << std::endl;
        maxCount = bufMaxCount;
    }
    if (maxCount == 0) return; // nothing to draw
    if (!cmdDrawIndexedIndirectCount) {
        throw std::runtime_error("vkCmdDrawIndexedIndirectCountKHR not available (draw-indirect-count required)");
    }
    cmdDrawIndexedIndirectCount(cmd, compactBuf.buffer, 0, visibleCount.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
}

uint32_t IndirectRenderer::readVisibleCount(VulkanApp* app) const {
    const uint32_t frame = currentCullFrame;
    const Buffer& visibleCount = visibleCountBuffers[frame];
    if (!app || visibleCount.buffer == VK_NULL_HANDLE) return 0;

    // Non-blocking read of the persistently-mapped, host-coherent count buffer.
    // The value reflects the most recent GPU cull result for this frame slot,
    // which (due to the frames-in-flight rotation) is always from an already
    // completed frame. Reading it is lock-free and never stalls the render
    // thread: no empty submit, no fence wait, no queue idle. The count shown in
    // the stats overlay lags by at most a few frames, which is invisible.
    if (visibleCountMapped[frame]) {
        return *visibleCountMapped[frame];
    }
    return 0;
}



IndirectRenderer::MeshInfo IndirectRenderer::getMeshInfo(uint32_t meshId) const {
    IndirectRenderer::MeshInfo empty;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return empty;
    return it->second;
}



// Erase a mesh's indirect command on the GPU so it will not be drawn
// before a full `rebuild()` updates the indirect buffer. This attempts
// an immediate host-write to the `indirectBuffer` (if present) to zero
// the VkDrawIndexedIndirectCommand for the specified mesh id.
void IndirectRenderer::eraseMeshFromGPU(VulkanApp* app, uint32_t meshId) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return;
    MeshInfo &info = it->second;
    if (indirectBuffer.buffer == VK_NULL_HANDLE || indirectBuffer.memory == VK_NULL_HANDLE) return;
    // If indirectOffset was assigned during the last rebuild, zero that entry.
    VkDeviceSize offset = info.indirectOffset;
    if (offset == 0 && indirectCommands.empty()) {
        // No valid indirect data available
        return;
    }
    VkDeviceSize cmdSize = sizeof(VkDrawIndexedIndirectCommand);
    // Sanity: don't write beyond current mesh capacity
    if (meshCapacity == 0) return;
    if (offset / cmdSize >= meshCapacity) return;

    VkDrawIndexedIndirectCommand zeroCmd{};
    // indexCount == 0 prevents drawing this command
    zeroCmd.indexCount = 0;
    zeroCmd.instanceCount = 0;
    zeroCmd.firstIndex = 0;
    zeroCmd.vertexOffset = 0;
    zeroCmd.firstInstance = 0;

    VkDevice dev = app ? app->getDevice() : VK_NULL_HANDLE;
    if (dev == VK_NULL_HANDLE) return;
    void* data = indirectBuffer.map(offset);
    if (data) {
        memcpy(data, &zeroCmd, cmdSize);
        std::cerr << "[IndirectRenderer] eraseMeshFromGPU: zeroed indirect cmd for mesh " << meshId << " at offset " << offset << std::endl;
        // Mark indirectOffset as invalid so future logic won't assume it
        info.indirectOffset = static_cast<VkDeviceSize>(-1);
    } else {
        std::cerr << "[IndirectRenderer] eraseMeshFromGPU: failed to map indirectBuffer memory for mesh " << meshId << std::endl;
    }
}

// ── Stable slot-based API ──────────────────────────────────────────────────

void IndirectRenderer::initSlots(VulkanApp* app,
                                 uint32_t maxActiveChunks,
                                 uint32_t totalVertexBytes,
                                 uint32_t totalIndexBytes)
{
    app_ = app;
    std::lock_guard<std::recursive_mutex> guard(mutex);

    // Packed-slot layout:
    //  - vertexCapacity/indexCapacity: TOTAL shared element pools. Each chunk
    //    packs its geometry into its OWN free-space span of these pools
    //    (PackedSpaceAllocator, best-fit) instead of a fixed worst-case
    //    per-chunk budget — so the pools host as many chunks as their BYTES
    //    allow. Small chunks share space.
    //  - slotAlloc: draw-entry allocator. Each active chunk owns ONE draw
    //    entry; the entry index IS the stable "slot" (drawIndex == slotIndex).
    // GPU buffers are pre-reserved once and never grown: a publish/erase only
    //    touches the chunk's own entry + element span — no global rebuild.
    vertexCapacity = totalVertexBytes / sizeof(Vertex);
    indexCapacity  = totalIndexBytes / sizeof(uint32_t);
    meshCapacity   = static_cast<size_t>(maxActiveChunks);

    // Draw-entry pool (one entry per chunk).
    slotAlloc.reserve(maxActiveChunks, 1, 1);

    // Packed element pools.
    spaceAlloc.reserve(static_cast<uint32_t>(vertexCapacity),
                       static_cast<uint32_t>(indexCapacity));

    // Pre-size the CPU-side merged buffers (one element per pool, fixed at
    // init). The chunk's span is written into its allocated sub-range.
    mergedVertices.resize(vertexCapacity);
    mergedIndices.resize(indexCapacity);

    // Pre-size indirect commands (one per chunk slot, initially zeroed).
    // Zeroed commands have indexCount=0, so GPU culling skips them.
    indirectCommands.resize(meshCapacity);
    std::memset(indirectCommands.data(), 0, indirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));

    // ── Create GPU buffers sized to capacity ─────────────────────────────────
    // These are created ONCE and never rebuilt. Individual slots are updated
    // in-place without touching other slots or the buffer layout.

    // Vertex buffer (device-local)
    VkDeviceSize vertexBufferSize = vertexCapacity * sizeof(Vertex);
    vertexBuffer = app->createBuffer(vertexBufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Index buffer (device-local)
    VkDeviceSize indexBufferSize = indexCapacity * sizeof(uint32_t);
    indexBuffer = app->createBuffer(indexBufferSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Indirect buffer (host-visible, persistently mapped for per-slot writes).
    // Zero the entire buffer so unallocated slots always have indexCount=0 and
    // the GPU cull shader never reads garbage bounds for slots whose meta hasn't
    // been written yet (between addMeshSlotted and the deferred writeSlotMeta).
    // TRANSFER_DST: staged meta writes land here via vkCmdCopyBuffer.
    VkDeviceSize indirectBufferSize = sizeof(VkDrawIndexedIndirectCommand) * meshCapacity;
    indirectBuffer = app->createBuffer(indirectBufferSize,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    {
        void* data = indirectBuffer.map(0);
        if (data) {
            std::memset(data, 0, (size_t)indirectBufferSize);
            indirectBuffer.unmap();
        }
    }

    // Bounds buffer (host-visible, persistently mapped). Same zeroing rationale.
    // Three vec4s per entry: min, max, meta (unused, kept zeroed for layout
    // compatibility with cascade culling).
    // TRANSFER_DST: staged bounds land here via vkCmdCopyBuffer.
    VkDeviceSize boundsBufferSize = sizeof(glm::vec4) * meshCapacity * 3;
    boundsBuffer = app->createBuffer(boundsBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    {
        void* data = boundsBuffer.map(0);
        if (data) {
            std::memset(data, 0, (size_t)boundsBufferSize);
            boundsBuffer.unmap();
        }
    }

    // Compact indirect buffers (one per cull frame, used by GPU culling)
    VkDeviceSize compactSize = indirectBufferSize;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        compactIndirectBuffers[f] = app->createBuffer(compactSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        // Zero the entire buffer so headroom is never garbage
        void* data = compactIndirectBuffers[f].map(0);
        if (data) {
            std::memset(data, 0, (size_t)compactSize);
            compactIndirectBuffers[f].unmap();
        }
    }

    // Per-frame chosen-LoD output buffers (uvec2 per draw entry: the compacted
    // firstInstance and the level, always 0 now that chunks are single-mesh)
    // and the scratch buffer bound by external descriptor-set owners.
    // TRANSFER_DST: prepareCull zeroes them with vkCmdFillBuffer each frame.
    VkDeviceSize lodBufSize = sizeof(glm::uvec2) * meshCapacity;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        visibleLodBuffers[f] = app->createBuffer(lodBufSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data = visibleLodBuffers[f].map(0);
        if (data) {
            std::memset(data, 0, (size_t)lodBufSize);
            visibleLodBuffers[f].unmap();
        }
    }
    if (visibleLodsScratch.buffer == VK_NULL_HANDLE) {
        visibleLodsScratch = app->createBuffer(lodBufSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data = visibleLodsScratch.map(0);
        if (data) {
            std::memset(data, 0, (size_t)lodBufSize);
            visibleLodsScratch.unmap();
        }
    }

    // Visible count buffers (one per cull frame, host-visible)
    VkDeviceSize countSize = sizeof(uint32_t);
    VkDevice dev = app->getDevice();
    storedDevice = dev;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        visibleCountBuffers[f] = app->createBuffer(countSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        visibleCountMapped[f] = static_cast<uint32_t*>(visibleCountBuffers[f].map(0));
        *visibleCountMapped[f] = 0;
    }

    // ── Create compute pipeline + descriptor sets for GPU culling ────────────
    // (Same as in rebuild() — factored out to share)
    {
        // Bindings (0..9): 0 input commands, 1 output commands, 2 bounds,
        // 3 count, 4 chosen-LoD output, 5/6 vegetation impostor cmd + count,
        // 7/8 vegetation billboard cmd + count, 9 vegetation chunk-info (input).
        VkDescriptorSetLayoutBinding bindings[10] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].binding = 2;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].binding = 3;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].binding = 4;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[5].binding = 5;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[6].binding = 6;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[7].binding = 7;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[8].binding = 8;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[9].binding = 9;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        VkDescriptorBindingFlags bindingFlags[10] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        };

        DescriptorAllocator descAlloc{app->getDevice(), app};
        computeDescriptorSetLayout = descAlloc.createLayout(
            bindings, 10,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            bindingFlags,
            "IndirectRenderer: computeDescriptorSetLayout");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(CullPushConstants); // 104 bytes: mat4 + 2*uint + pad + vec3 + float + uint

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        plinfo.pSetLayouts = &computeDescriptorSetLayout;
        plinfo.pushConstantRangeCount = 1;
        plinfo.pPushConstantRanges = &pc;

        if (vkCreatePipelineLayout(app->getDevice(), &plinfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        app->resources.addPipelineLayout(computePipelineLayout, "IndirectRenderer: computePipelineLayout");

        VkShaderModule compModule = app->getOrCreateShaderModule("shaders/indirect.comp.spv");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = compModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stage;
        pipelineInfo.layout = computePipelineLayout;

        if (vkCreateComputePipelines(app->getDevice(), app->getPipelineCache(), 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
        app->resources.addPipeline(computePipeline, "IndirectRenderer: computePipeline");

        VkDescriptorPoolSize irPoolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256};
        computeDescriptorPool = descAlloc.createPool(
            &irPoolSize, 1, 64,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            "IndirectRenderer: computeDescriptorPool");

        descAlloc.allocateSets(computeDescriptorPool, computeDescriptorSetLayout,
                               MAX_CULL_FRAMES, reinterpret_cast<VkDescriptorSet*>(computeDescriptorSets.data()),
                               "IndirectRenderer: computeDescriptorSet");

        // Tiny dummy bound to the vegetation bindings (5..9) on the solid-only
        // dispatch so the layout's statically-referenced bindings are always valid.
        // prepareCull re-points these to the real veg buffers when vegetation is enabled.
        if (vegDummyBuffer.buffer == VK_NULL_HANDLE) {
            vegDummyBuffer = app->createBuffer(16,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
    }

    // Update per-frame compute descriptor sets with buffer info
    VkDescriptorBufferInfo inBuf{};
    inBuf.buffer = indirectBuffer.buffer;
    inBuf.offset = 0;
    inBuf.range = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo boundsBufInfo{};
    boundsBufInfo.buffer = boundsBuffer.buffer;
    boundsBufInfo.offset = 0;
    boundsBufInfo.range = VK_WHOLE_SIZE;

    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        VkDescriptorBufferInfo outBuf{};
        outBuf.buffer = compactIndirectBuffers[f].buffer;
        outBuf.offset = 0;
        outBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo countBuf{};
        countBuf.buffer = visibleCountBuffers[f].buffer;
        countBuf.offset = 0;
        countBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo lodBuf{};
        lodBuf.buffer = visibleLodBuffers[f].buffer;
        lodBuf.offset = 0;
        lodBuf.range = VK_WHOLE_SIZE;

        VkDescriptorSet computeDs = computeDescriptorSets[f];
        DescriptorWriter(app->getDevice())
            .writeBuffer(computeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         inBuf.buffer, inBuf.offset, inBuf.range)
            .writeBuffer(computeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         outBuf.buffer, outBuf.offset, outBuf.range)
            .writeBuffer(computeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         boundsBufInfo.buffer, boundsBufInfo.offset, boundsBufInfo.range)
            .writeBuffer(computeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         countBuf.buffer, countBuf.offset, countBuf.range)
             .writeBuffer(computeDs, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           lodBuf.buffer, lodBuf.offset, lodBuf.range)
             .writeBuffer(computeDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
             .writeBuffer(computeDs, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
             .writeBuffer(computeDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
             .writeBuffer(computeDs, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
             .writeBuffer(computeDs, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           vegDummyBuffer.buffer, 0, VK_WHOLE_SIZE)
             .flush();
    }

    // Load indirect-count draw function
    cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCountKHR");
    if (!cmdDrawIndexedIndirectCount) {
        cmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetDeviceProcAddr(app->getDevice(), "vkCmdDrawIndexedIndirectCount");
    }
    if (!cmdDrawIndexedIndirectCount) {
        throw std::runtime_error("Required device function vkCmdDrawIndexedIndirectCountKHR is not available");
    }

    slottedMode = true;
    dirty = false;
    metaBuffersWrittenCount = 0;

    // Initialize cascade-aware culling resources
    initCascadeCull(app);

    std::cerr << "[IndirectRenderer::initSlots] maxActiveChunks=" << maxActiveChunks
              << " meshCapacity=" << meshCapacity
              << " indirectCommands=" << indirectCommands.size()
              << " compactBuf[0].buffer=" << (void*)compactIndirectBuffers[0].buffer
              << std::endl << std::flush;
}

void IndirectRenderer::copyGeometryToLevel(const Geometry& mesh, MeshInfo::LevelData& ld)
{
    // Copy vertex data into the level's packed span (absolute position)
    if (!mesh.vertices.empty() && ld.baseVertex + mesh.vertices.size() <= mergedVertices.size()) {
        std::memcpy(&mergedVertices[ld.baseVertex],
                    mesh.vertices.data(),
                    mesh.vertices.size() * sizeof(Vertex));
    }

    // Copy index data into the level's packed span (absolute position)
    if (!mesh.indices.empty() && ld.firstIndex + mesh.indices.size() <= mergedIndices.size()) {
        std::memcpy(&mergedIndices[ld.firstIndex],
                    mesh.indices.data(),
                    mesh.indices.size() * sizeof(uint32_t));
    }
}

uint32_t IndirectRenderer::addMeshSlotted(const Geometry& mesh, uint32_t chunkId,
                                          const glm::vec3* cubeMin, const glm::vec3* cubeMax,
                                          int level)
{
    if (!slottedMode) return UINT32_MAX;

    std::lock_guard<std::recursive_mutex> guard(mutex);

    uint32_t neededVerts = static_cast<uint32_t>(mesh.vertices.size());
    uint32_t neededIdxs  = static_cast<uint32_t>(mesh.indices.size());

    auto existing = meshes.find(chunkId);
    const bool isNewChunk = (existing == meshes.end() || !existing->second.active);

    // Allocate this chunk's draw entry on first use. The entry index is the
    // chunk's stable slot (one draw entry per chunk).
    uint32_t slotIdx = isNewChunk ? UINT32_MAX : existing->second.slotIndex;
    if (slotIdx == UINT32_MAX) {
        slotIdx = slotAlloc.allocate(1, 1);
        if (slotIdx == UINT32_MAX) {
            auto now = std::chrono::steady_clock::now();
            if (now - g_lastNoSlotLog >= std::chrono::seconds(1)) {
                g_lastNoSlotLog = now;
                std::cerr << "[IndirectRenderer] addMeshSlotted: no free draw block for chunk " << chunkId
                          << " (active=" << slotAlloc.activeCount()
                          << " capacity=" << slotAlloc.capacity()
                          << " usedVert=" << spaceAlloc.usedVertex()
                          << "/" << spaceAlloc.totalVertex()
                          << " usedIdx=" << spaceAlloc.usedIndex()
                          << "/" << spaceAlloc.totalIndex() << ")" << std::endl;
            }
            return UINT32_MAX;
        }
#ifdef DEBUG
        if (slotAlloc.peakActiveCount() >= lastPeakLogged_ + 32) {
            lastPeakLogged_ = slotAlloc.peakActiveCount();
            std::cout << "[IndirectRenderer] slot peak " << lastPeakLogged_
                      << " / " << slotAlloc.capacity() << " (this=" << this << ")" << std::endl;
        }
#endif
    }
    if (slotIdx >= slotAlloc.capacity()) {
        std::cerr << "[IndirectRenderer] addMeshSlotted: slot " << slotIdx
                  << " out of range (capacity=" << slotAlloc.capacity() << ")" << std::endl;
        return UINT32_MAX;
    }
    const uint32_t entryIndex = slotIdx; // draw entry == slot (one entry per chunk)

    // Pack this chunk's geometry into its own span of the shared element
    // pools (first-fit/best-fit free-space allocator). The old span — if any —
    // is NOT freed here: in-flight frames may still reference it, so it is
    // recorded and released once the replacement upload completes (see
    // uploadSlot). If the geometry is empty, release the old span now:
    // nothing will be uploaded and the entry is zeroed below, so the old data
    // is unreachable.
    MeshInfo* chunk = isNewChunk ? nullptr : &existing->second;
    bool effectiveEmpty = (neededVerts == 0 || neededIdxs == 0);
    if (!effectiveEmpty) {
        uint32_t newVBase = spaceAlloc.allocateVertex(neededVerts);
        uint32_t newIBase = spaceAlloc.allocateIndex(neededIdxs);
        if (newVBase == UINT32_MAX || newIBase == UINT32_MAX) {
            // Roll back the partial allocation so the pool is not fragmented.
            if (newVBase != UINT32_MAX) spaceAlloc.freeVertex(newVBase, neededVerts);
            if (newIBase != UINT32_MAX) spaceAlloc.freeIndex(newIBase, neededIdxs);
            auto now = std::chrono::steady_clock::now();
            if (now - g_lastNoSlotLog >= std::chrono::seconds(1)) {
                g_lastNoSlotLog = now;
                std::cerr << "[IndirectRenderer] addMeshSlotted: element pool exhausted for chunk "
                          << chunkId << " (verts=" << neededVerts
                          << " idxs=" << neededIdxs << "; usedVert=" << spaceAlloc.usedVertex()
                          << "/" << spaceAlloc.totalVertex()
                          << " usedIdx=" << spaceAlloc.usedIndex()
                          << "/" << spaceAlloc.totalIndex() << ")" << std::endl;
            }
            return UINT32_MAX;
        }

        // Preserve the chunk's previous span for deferred freeing. On a fresh
        // allocate (first publish) there is none. If an older span is already
        // pending (a previous republish whose upload never completed), free it
        // now: its entry was zeroed by that republish, so nothing references
        // it anymore and waiting for a completion that will never fire would
        // leak it.
        if (chunk) {
            MeshInfo::LevelData& prev = chunk->level_;
            if (prev.allocated) {
                if (prev.oldVertexBase != UINT32_MAX) {
                    spaceAlloc.freeVertex(prev.oldVertexBase, prev.oldVertexCount);
                    spaceAlloc.freeIndex(prev.oldIndexBase, prev.oldIndexCount);
                }
                prev.oldVertexBase = prev.baseVertex;
                prev.oldVertexCount = prev.vertexCount;
                prev.oldIndexBase  = prev.firstIndex;
                prev.oldIndexCount = prev.indexCount;
            }
        }

        if (isNewChunk) {
            // A brand-new slot for a removed-and-recreated node: allocate a
            // fresh entry and let the copy write into the packed span.
            MeshInfo m{};
            m.id         = chunkId;
            m.slotIndex  = slotIdx;
            m.active     = true;
            meshes[chunkId] = m;
            chunk = &meshes[chunkId];
        }

        MeshInfo::LevelData& ld = chunk->level_;
        ld.allocated    = true;
        ld.baseVertex   = newVBase;
        ld.vertexCount  = neededVerts;
        ld.firstIndex   = newIBase;
        ld.indexCount   = neededIdxs;
        ld.level        = level;
        if (cubeMin && cubeMax) {
            ld.boundsMin = glm::vec4(*cubeMin, 0.0f);
            ld.boundsMax = glm::vec4(*cubeMax, 0.0f);
        } else if (mesh.vertices.empty()) {
            ld.boundsMin = glm::vec4(0.0f);
            ld.boundsMax = glm::vec4(0.0f);
        } else {
            glm::vec3 minp(FLT_MAX), maxp(-FLT_MAX);
            for (const auto& v : mesh.vertices) {
                minp = glm::min(minp, v.position);
                maxp = glm::max(maxp, v.position);
            }
            ld.boundsMin = glm::vec4(minp, 0.0f);
            ld.boundsMax = glm::vec4(maxp, 0.0f);
        }
        // Copy geometry data into the chunk's packed span (absolute positions).
        copyGeometryToLevel(mesh, ld);

        // Publish the CPU-side indirect command for this entry (the GPU
        // buffers stay zeroed until the deferred writeSlotMeta after upload).
        indirectCommands[entryIndex].indexCount    = ld.indexCount;
        indirectCommands[entryIndex].instanceCount = 1;
        indirectCommands[entryIndex].firstIndex    = ld.firstIndex;
        indirectCommands[entryIndex].vertexOffset  = static_cast<int32_t>(ld.baseVertex);
        indirectCommands[entryIndex].firstInstance = entryIndex;

        // Also fill the mirror flat fields used by legacy-style callers that
        // read the mesh info (e.g. stats).
        chunk->baseVertex   = ld.baseVertex;
        chunk->vertexCount  = ld.vertexCount;
        chunk->firstIndex   = ld.firstIndex;
        chunk->indexCount   = ld.indexCount;
        chunk->drawIndex    = entryIndex;
        chunk->boundsMin    = ld.boundsMin;
        chunk->boundsMax    = ld.boundsMax;
    } else {
        // Empty geometry: zero the entry so the GPU never draws it, and free
        // any previous span now (no upload will reference it). If the chunk is
        // new, still register it so removeMeshSlotted can free the slot.
        if (chunk) {
            MeshInfo::LevelData& prev = chunk->level_;
            if (prev.allocated) {
                spaceAlloc.freeVertex(prev.baseVertex, prev.vertexCount);
                spaceAlloc.freeIndex(prev.firstIndex, prev.indexCount);
                prev = MeshInfo::LevelData{};
            }
        } else {
            MeshInfo m{};
            m.id        = chunkId;
            m.slotIndex = slotIdx;
            m.active    = true;
            meshes[chunkId] = m;
        }
        if (entryIndex < indirectCommands.size())
            indirectCommands[entryIndex] = VkDrawIndexedIndirectCommand{};
    }

    activeMeshCountDirty_ = true; // new or resurrected entry

    // Zero the GPU indirect + bounds for THIS entry (staged for the next
    // prepareCull) so the cull shader never sees stale/garbage data during the
    // window between this publish and the deferred writeSlotMeta (upload
    // completion). Fresh slots are redundant with initSlots zeroing, but
    // recycled slots may retain stale bounds from a prior occupant. Staged
    // (not memcpy'd): in-flight cull dispatches may still be reading this
    // entry, and a torn write would be observed as garbage by the cull.
    {
        VkDrawIndexedIndirectCommand zeroCmd{};
        glm::vec4 zeroBounds[3] = { glm::vec4(0.0f), glm::vec4(0.0f), glm::vec4(0.0f) };
        stageMeshMetaWrite(entryIndex, zeroCmd, zeroBounds, true);
    }

    return slotIdx;
}

void IndirectRenderer::removeMeshSlotted(uint32_t slotIndex)
{
    if (!slottedMode) return;

    std::lock_guard<std::recursive_mutex> guard(mutex);

    // Free the draw entry and the chunk's packed element span, then zero the
    // entry so GPU culling drops it on indexCount == 0.
    // NOTE: the packed span is freed immediately. When this is called from
    // an upload completion callback (the common chunk-replacement path) that
    // is safe: the transfer completed, so every prior frame has retired. When
    // called outside a completion (brush rebuilds), an in-flight frame that
    // already culled this chunk may still reference the span — but its entry
    // is zeroed right after, so the window only exists for already-queued
    // draws and is the same class of transient the pre-packed code accepted.
    MeshInfo* info = nullptr;
    for (auto it = meshes.begin(); it != meshes.end(); ) {
        if (it->second.active && it->second.slotIndex == slotIndex) {
            info = &it->second;
            it->second.active = false;
            activeMeshCountDirty_ = true;
            break;
        } else {
            ++it;
        }
    }

    if (info) {
        MeshInfo::LevelData& ld = info->level_;
        // A pending deferred old-span free can never fire anymore (no upload
        // of this chunk is in flight once the entry is zeroed below): release
        // it now.
        if (ld.oldVertexBase != UINT32_MAX) {
            spaceAlloc.freeVertex(ld.oldVertexBase, ld.oldVertexCount);
            spaceAlloc.freeIndex(ld.oldIndexBase, ld.oldIndexCount);
        }
        if (ld.allocated) {
            spaceAlloc.freeVertex(ld.baseVertex, ld.vertexCount);
            spaceAlloc.freeIndex(ld.firstIndex, ld.indexCount);
        }
        ld = MeshInfo::LevelData{};
    }

    // Free the slot in the allocator
    slotAlloc.free(slotIndex);

    // Zero the slot's draw entry (staged for the next prepareCull): the chunk
    // is gone, so it may not survive culling. GPU culling sees indexCount=0
    // and drops it. Staged (not memcpy'd) so an in-flight cull dispatch never
    // observes a torn entry mid-write.
    VkDrawIndexedIndirectCommand zeroCmd{};
    glm::vec4 zeroBounds[3] = { glm::vec4(0.0f), glm::vec4(0.0f), glm::vec4(0.0f) };
    if (slotIndex < indirectCommands.size()) {
        indirectCommands[slotIndex] = VkDrawIndexedIndirectCommand{};
    }
    stageMeshMetaWrite(slotIndex, zeroCmd, zeroBounds, true);
}

bool IndirectRenderer::uploadSlot(VulkanApp* app, uint32_t slotIndex, float priority,
                                   std::function<void()> onComplete)
{
    if (!slottedMode) return false;

    std::lock_guard<std::recursive_mutex> guard(mutex);

    // Find the MeshInfo for this slot
    MeshInfo* info = nullptr;
    for (auto& kv : meshes) {
        if (kv.second.active && kv.second.slotIndex == slotIndex) {
            info = &kv.second;
            break;
        }
    }
    if (!info || !info->active) return false;

    MeshInfo::LevelData& ld = info->level_;

    // If the chunk has no geometry, zero the entry (nothing to upload) and
    // run the completion immediately.
    if (!ld.allocated || ld.vertexCount == 0 || ld.indexCount == 0) {
        uint32_t entryIndex = slotIndex;
        if (entryIndex < indirectCommands.size())
            indirectCommands[entryIndex] = VkDrawIndexedIndirectCommand{};
        if (onComplete) onComplete();
        return true;
    }

    // Capture metadata for deferred write — the MeshInfo/indirectCommands are
    // stable until the slot is freed, but we capture the actual draw parameters
    // by value so the write is correct even if the callback fires after the
    // slot's MeshInfo is modified by a later republish.
    const uint32_t entryIndex      = slotIndex;
    uint32_t capEntryIndex         = entryIndex;
    uint32_t capIndexCount         = ld.indexCount;
    uint32_t capFirstIndex         = ld.firstIndex;
    int32_t  capVertexOffset       = static_cast<int32_t>(ld.baseVertex);
    glm::vec4 capBoundsMin         = ld.boundsMin;
    glm::vec4 capBoundsMax         = ld.boundsMax;
    int      capLevel              = ld.level;
    uint32_t capOldVertexBase      = ld.oldVertexBase;
    uint32_t capOldVertexCount     = ld.oldVertexCount;
    uint32_t capOldIndexBase       = ld.oldIndexBase;
    uint32_t capOldIndexCount      = ld.oldIndexCount;
    // Consume the pending old-span record: it is handed to the completion
    // callback below (freed once the replacement is resident on GPU). A second
    // republish before this upload completes would otherwise chain its old
    // span on top of this one.
    ld.oldVertexBase = UINT32_MAX;
    ld.oldIndexBase  = UINT32_MAX;

    // Calculate vertex/index byte ranges for this chunk's packed span
    VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(ld.vertexCount) * sizeof(Vertex);
    VkDeviceSize indexBytes  = static_cast<VkDeviceSize>(ld.indexCount) * sizeof(uint32_t);
    VkDeviceSize vertexOffset = static_cast<VkDeviceSize>(ld.baseVertex) * sizeof(Vertex);
    VkDeviceSize indexOffset  = static_cast<VkDeviceSize>(ld.firstIndex) * sizeof(uint32_t);

    // Build a single BufferUpload per destination (vertex + index)
    std::vector<streaming::BufferUpload> uploads;
    uploads.reserve(2);

    // Vertex data
    {
        streaming::BufferUpload vu;
        vu.dst       = vertexBuffer;
        vu.dstOffset = vertexOffset;
        vu.cpuData.resize(vertexBytes);
        std::memcpy(vu.cpuData.data(),
                    &mergedVertices[ld.baseVertex],
                    vertexBytes);
        uploads.push_back(std::move(vu));
    }
    // Index data
    {
        streaming::BufferUpload iu;
        iu.dst       = indexBuffer;
        iu.dstOffset = indexOffset;
        iu.cpuData.resize(indexBytes);
        std::memcpy(iu.cpuData.data(),
                    &mergedIndices[ld.firstIndex],
                    indexBytes);
        uploads.push_back(std::move(iu));
    }

    // Defer writeSlotMeta to the upload completion callback for ALL paths.
    // Writing meta eagerly would modify the shared indirect/bounds buffers
    // while in-flight frames may still be reading them (3 frames in flight).
    // Those frames would see the new meta but stale vertex data (upload not
    // yet executed for those frames) — causing a 1-frame hole.  The deferred
    // write runs after the upload completes, once all prior frames have retired.
    // The replaced span is freed in the same callback: the transfer completing
    // proves every prior submission (including those frames' culls) finished,
    // so the old data can no longer be referenced.
    auto deferredWriteMeta = [this, capEntryIndex, capIndexCount, capFirstIndex,
                              capVertexOffset, capBoundsMin, capBoundsMax, capLevel,
                              capOldVertexBase, capOldVertexCount,
                              capOldIndexBase, capOldIndexCount]()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (capEntryIndex >= indirectCommands.size()) return;
        if (indirectBuffer.buffer == VK_NULL_HANDLE) return;

        VkDeviceSize cmdOffset = static_cast<VkDeviceSize>(capEntryIndex) * sizeof(VkDrawIndexedIndirectCommand);
        void* cmdData = indirectBuffer.map(cmdOffset);
        if (cmdData) {
            VkDrawIndexedIndirectCommand cmd{};
            cmd.indexCount    = capIndexCount;
            cmd.instanceCount = 1;
            cmd.firstIndex    = capFirstIndex;
            cmd.vertexOffset  = capVertexOffset;
            cmd.firstInstance = capEntryIndex;
            std::memcpy(cmdData, &cmd, sizeof(cmd));
            indirectBuffer.unmap();
        }

        if (boundsBuffer.buffer != VK_NULL_HANDLE) {
            VkDeviceSize boundsOffset = static_cast<VkDeviceSize>(capEntryIndex) * 3 * sizeof(glm::vec4);
            void* bndData = boundsBuffer.map(boundsOffset);
            if (bndData) {
                // Third vec4 (meta): {cellSize, level, maxLevel, unused} —
                // cellSize is the chunk's OWN cube length (the shader derives
                // the frontier base as cellSize / 2^level); level is the
                // 0-based LoD level; maxLevel is the tree's real ladder depth
                // (set via setMaxLodLevel from LocalScene::maxChunkLod) — NOT a
                // hard-coded constant, or every chunk above that level is culled.
                const float cellSize = chunkCellSize_ > 0.0f
                                            ? chunkCellSize_
                                            : (capBoundsMax.x - capBoundsMin.x);
                const glm::vec4 lodMeta = glm::vec4(cellSize, static_cast<float>(capLevel),
                                                   static_cast<float>(maxLodLevel_), 0.0f);
                glm::vec4 bounds[3] = { capBoundsMin, capBoundsMax, lodMeta };
                std::memcpy(bndData, bounds, sizeof(bounds));
                boundsBuffer.unmap();
            }
        }

        // Free the chunk's replaced span now that the replacement is resident.
        if (capOldVertexBase != UINT32_MAX) {
            spaceAlloc.freeVertex(capOldVertexBase, capOldVertexCount);
            spaceAlloc.freeIndex(capOldIndexBase, capOldIndexCount);
        }
    };

    auto chained = [deferredWriteMeta = std::move(deferredWriteMeta),
                    onComplete = std::move(onComplete)]() mutable
    {
        deferredWriteMeta();
        if (onComplete) onComplete();
    };

    bool useUploadMgr = (uploadMgr_ != nullptr);

    if (useUploadMgr && (vertexBytes + indexBytes) <= uploadMgr_->slotSize()) {
        // ── UploadManager path (preferred) ──────────────────────────────────
        streaming::UploadJob job;
        job.category  = streamCategory_;
        job.priority  = priority;
        job.chunkSlot = nullptr;
        job.uploads   = std::move(uploads);
        job.onComplete = std::move(chained);
        uploadMgr_->enqueue(std::move(job));
    } else {
        // ── Legacy ring-backed staging path ────────────────────────────────
        deferredUploadCallbacks_.push_back(std::move(chained));
        return uploadMeshes(app, std::vector<uint32_t>{info->id}, priority);
    }

    return true;
}

// ── Cascade-aware culling ────────────────────────────────────────────────────

void IndirectRenderer::initCascadeCull(VulkanApp* app) {
    if (cascadeCullInited) return;
    cascadeCullInited = true;
    cascadeDescApp = app;

    VkDevice device = app->getDevice();

    // Create storage buffer for cascade matrices (3 mat4 = 192 bytes)
    cascadeMatrixBuffer = app->createBuffer(sizeof(glm::mat4) * 3,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Descriptor set layout: 10 bindings
    // 0: inCmds (input draw commands)
    // 1: outCmds0 (cascade 0)
    // 2: bounds
    // 3: count0
    // 4: outCmds1 (cascade 1)
    // 5: count1
    // 6: outCmds2 (cascade 2)
    // 7: count2
    // 8: cascadeMatrices
    // 9: visibleLods (chosen-LoD per draw entry, written by the main cull pass)
    std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
    VkDescriptorBindingFlags bindingFlags[10];
    for (uint32_t i = 0; i < 10; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindingFlags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    }

    DescriptorAllocator descAlloc{device, app};
    cascadeCullDescSetLayout = descAlloc.createLayout(
        bindings.data(), 10,
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        bindingFlags,
        "IndirectRenderer: cascadeCullDescSetLayout");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(CascadeCullPushConstants); // 32 bytes: uint + pad + vec3 + float

    VkPipelineLayoutCreateInfo plinfo{};
    plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plinfo.setLayoutCount = 1;
    plinfo.pSetLayouts = &cascadeCullDescSetLayout;
    plinfo.pushConstantRangeCount = 1;
    plinfo.pPushConstantRanges = &pc;

    if (vkCreatePipelineLayout(device, &plinfo, nullptr, &cascadeCullPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cascade cull pipeline layout!");
    app->resources.addPipelineLayout(cascadeCullPipelineLayout, "IndirectRenderer: cascadeCullPipelineLayout");

    VkShaderModule compModule = app->getOrCreateShaderModule("shaders/cascade_cull.comp.spv");
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = compModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = cascadeCullPipelineLayout;
    if (vkCreateComputePipelines(device, app->getPipelineCache(), 1, &pipelineInfo, nullptr, &cascadeCullPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cascade cull compute pipeline!");
    app->resources.addPipeline(cascadeCullPipeline, "IndirectRenderer: cascadeCullPipeline");

    // Descriptor pool
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
    cascadeCullDescPool = descAlloc.createPool(
        &poolSize, 1, MAX_CULL_FRAMES,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        "IndirectRenderer: cascadeCullDescPool");

    VkDescriptorSet rawSets[MAX_CULL_FRAMES];
    descAlloc.allocateSets(cascadeCullDescPool, cascadeCullDescSetLayout,
                           MAX_CULL_FRAMES, rawSets,
                           "IndirectRenderer: cascadeCullDescSet");
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        cascadeCullFrames[f].descSet = rawSets[f];
    }

    // Per-frame cascade cull resources
    VkDeviceSize compactSize = sizeof(VkDrawIndexedIndirectCommand) * meshCapacity;
    if (compactSize == 0) compactSize = sizeof(VkDrawIndexedIndirectCommand) * 1024;
    VkDeviceSize countSize = sizeof(uint32_t);

    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        for (uint32_t c = 0; c < 3; c++) {
            cascadeCullFrames[f].compactBuffers[c] = app->createBuffer(compactSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* data = cascadeCullFrames[f].compactBuffers[c].map(0);
            if (data) {
                std::memset(data, 0, (size_t)compactSize);
                cascadeCullFrames[f].compactBuffers[c].unmap();
            }
            cascadeCullFrames[f].countBuffers[c] = app->createBuffer(countSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            cascadeCullFrames[f].countMapped[c] = static_cast<uint32_t*>(cascadeCullFrames[f].countBuffers[c].map(0));
            *cascadeCullFrames[f].countMapped[c] = 0;
        }
        updateCascadeDescriptor(app, f);
    }
}

void IndirectRenderer::destroyCascadeCull() {
    if (!cascadeCullInited) return;
    cascadeCullInited = false;
    for (auto& frame : cascadeCullFrames) {
        for (uint32_t c = 0; c < 3; c++) {
            frame.compactBuffers[c] = {};
            frame.countBuffers[c] = {};
            frame.countMapped[c] = nullptr;
        }
    }
    cascadeMatrixBuffer = {};
    cascadeCullPipeline = VK_NULL_HANDLE;
    cascadeCullPipelineLayout = VK_NULL_HANDLE;
    cascadeCullDescSetLayout = VK_NULL_HANDLE;
    cascadeCullDescPool = VK_NULL_HANDLE;
}

void IndirectRenderer::updateCascadeDescriptor(VulkanApp* app, uint32_t frame) {
    VkDescriptorSet ds = cascadeCullFrames[frame].descSet;
    if (ds == VK_NULL_HANDLE) return;

    VkDescriptorBufferInfo inBuf{};
    inBuf.buffer = indirectBuffer.buffer;
    inBuf.offset = 0;
    inBuf.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo boundsBufInfo{};
    boundsBufInfo.buffer = boundsBuffer.buffer;
    boundsBufInfo.offset = 0;
    boundsBufInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo matBuf{};
    matBuf.buffer = cascadeMatrixBuffer.buffer;
    matBuf.offset = 0;
    matBuf.range = VK_WHOLE_SIZE;

    DescriptorWriter writer(app->getDevice());
    writer.writeBuffer(ds, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       inBuf.buffer, inBuf.offset, inBuf.range);
    writer.writeBuffer(ds, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       boundsBufInfo.buffer, boundsBufInfo.offset, boundsBufInfo.range);

    for (uint32_t c = 0; c < 3; c++) {
        // Shader bindings: 1=outCmds0, 2=bounds, 3=count0, 4=outCmds1, 5=count1, 6=outCmds2, 7=count2
        static const uint32_t outBindings[3] = {1, 4, 6};
        static const uint32_t cntBindings[3] = {3, 5, 7};
        writer.writeBuffer(ds, outBindings[c], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           cascadeCullFrames[frame].compactBuffers[c].buffer, 0, VK_WHOLE_SIZE);
        writer.writeBuffer(ds, cntBindings[c], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           cascadeCullFrames[frame].countBuffers[c].buffer, 0, VK_WHOLE_SIZE);
    }

    writer.writeBuffer(ds, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       matBuf.buffer, matBuf.offset, matBuf.range);
    // Binding 9: the per-frame chosen-LoD buffer written by the main cull pass.
    // Same frame index as this cascade descriptor set, so the cascade reads the
    // selection computed for the current frame in flight. visibleLodBuffers are
    // recreated only alongside indirectBuffer/boundsBuffer (initSlots/rebuild),
    // so the existing indirect/bounds refresh proxy covers them.
    writer.writeBuffer(ds, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       visibleLodBuffers[frame].buffer, 0, VK_WHOLE_SIZE);
    writer.flush();

    cascadeDescIndirectBuffer = indirectBuffer.buffer;
    cascadeDescBoundsBuffer = boundsBuffer.buffer;
}

void IndirectRenderer::refreshCascadeDescriptorsIfNeeded() {
    if (!cascadeCullInited || !cascadeDescApp) return;
    if (cascadeDescIndirectBuffer == indirectBuffer.buffer &&
        cascadeDescBoundsBuffer == boundsBuffer.buffer) return;
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        updateCascadeDescriptor(cascadeDescApp, f);
    }
}

void IndirectRenderer::prepareCullCascades(VkCommandBuffer cmd,
                                            const glm::mat4 cascadeMatrices[3],
                                            glm::vec3 camPos, float lodBias) {
    if (!cascadeCullInited || cascadeCullPipeline == VK_NULL_HANDLE) return;

    // Refresh descriptors if indirectBuffer or boundsBuffer were recreated
    refreshCascadeDescriptorsIfNeeded();

    Buffer& compactBuf = compactIndirectBuffers[currentCullFrame];
    if (compactBuf.buffer == VK_NULL_HANDLE) return;

    // Acquire uploaded geometry/meta buffers first
    acquireBuffers(cmd);

    // Upload cascade matrices to storage buffer
    {
        void* matData = cascadeMatrixBuffer.map(0);
        if (matData) {
            std::memcpy(matData, cascadeMatrices, sizeof(glm::mat4) * 3);
            cascadeMatrixBuffer.unmap();
        }
    }

    // Barrier: drain prior draws/compute before zeroing cascade outputs
    {
        VkBufferMemoryBarrier2 preFill[6]{};
        uint32_t preCount = 0;
        for (uint32_t i = 0; i < 3; i++) {
            preFill[preCount].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            preFill[preCount].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                      | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preFill[preCount].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                                      | VK_ACCESS_2_SHADER_READ_BIT
                                      | VK_ACCESS_2_SHADER_WRITE_BIT
                                      | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preFill[preCount].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            preFill[preCount].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            preFill[preCount].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preFill[preCount].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            preFill[preCount].buffer = cascadeCullFrames[currentCullFrame].compactBuffers[i].buffer;
            preFill[preCount].offset = 0;
            preFill[preCount].size = VK_WHOLE_SIZE;
            preCount++;

            preFill[preCount] = preFill[preCount - 1];
            preFill[preCount].buffer = cascadeCullFrames[currentCullFrame].countBuffers[i].buffer;
            preCount++;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = preCount;
        depInfo.pBufferMemoryBarriers = preFill;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Zero all 3 compact buffers and count buffers (cascade-specific, not the main compact buffer)
    for (uint32_t c = 0; c < 3; c++) {
        vkCmdFillBuffer(cmd, cascadeCullFrames[currentCullFrame].compactBuffers[c].buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, cascadeCullFrames[currentCullFrame].countBuffers[c].buffer, 0, sizeof(uint32_t), 0);
    }

    // Get dispatch count
    uint32_t numCmds = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        numCmds = getCullDispatchCountLocked();
    }

    // Barrier: fill → compute + indirect draw (compact + count buffers).
    // When numCmds > 0, the compute shader reads count (atomicAdd) and writes
    // compact, so SHADER_READ|SHADER_WRITE at COMPUTE stage is required.
    // When numCmds == 0, drawCascadeOnly still reads the count buffer, so
    // DRAW_INDIRECT access is also needed. Combined barrier handles both cases.
    {
        VkBufferMemoryBarrier2 barriers[6]{};
        for (uint32_t i = 0; i < 3; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barriers[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                    | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT
                                     | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].buffer = cascadeCullFrames[currentCullFrame].compactBuffers[i].buffer;
            barriers[i].offset = 0;
            barriers[i].size = VK_WHOLE_SIZE;

            barriers[3 + i] = barriers[i];
            barriers[3 + i].buffer = cascadeCullFrames[currentCullFrame].countBuffers[i].buffer;
            barriers[3 + i].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT
                                         | VK_ACCESS_2_SHADER_WRITE_BIT
                                         | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 6;
        depInfo.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    if (numCmds == 0) return;

    // Bind and dispatch cascaded culling compute shader
    VkDescriptorSet descSet = cascadeCullFrames[currentCullFrame].descSet;
    if (cmdState) cmdState->bindComputePipeline(cmd, cascadeCullPipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cascadeCullPipeline);
    if (cmdState) cmdState->bindComputeDescriptorSets(cmd, cascadeCullPipelineLayout, 0, 1, &descSet, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cascadeCullPipelineLayout, 0, 1, &descSet, 0, nullptr);

    CascadeCullPushConstants pushData{};
    pushData.numChunks = numCmds;
    pushData.camPos    = camPos;
    pushData.lodBias   = lodBias;
    vkCmdPushConstants(cmd, cascadeCullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CascadeCullPushConstants), &pushData);

    uint32_t groups = (numCmds + 63) / 64;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute shader writes → indirect draw (compact + count buffers).
    // This creates an additional dependency chain on top of the fill→draw barrier
    // above, so the draw sees whichever wrote last (fill zeros or compute atomics).
    {
        VkBufferMemoryBarrier2 barriers[6]{};
        for (uint32_t i = 0; i < 3; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[i].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                    | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                                    | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                                    | VK_ACCESS_2_SHADER_READ_BIT;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].buffer = cascadeCullFrames[currentCullFrame].compactBuffers[i].buffer;
            barriers[i].offset = 0;
            barriers[i].size = VK_WHOLE_SIZE;

            barriers[3 + i] = barriers[i];
            barriers[3 + i].buffer = cascadeCullFrames[currentCullFrame].countBuffers[i].buffer;
            barriers[3 + i].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        }

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 6;
        depInfo.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
}

void IndirectRenderer::drawCascadeOnly(VkCommandBuffer cmd, uint32_t cascadeIndex) {
    if (cascadeIndex >= 3) return;
    if (!cascadeCullInited) return;
    Buffer& compactBuf = cascadeCullFrames[currentCullFrame].compactBuffers[cascadeIndex];
    Buffer& countBuf = cascadeCullFrames[currentCullFrame].countBuffers[cascadeIndex];

    if (compactBuf.buffer == VK_NULL_HANDLE || countBuf.buffer == VK_NULL_HANDLE) return;
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;
    if (!cmdDrawIndexedIndirectCount) return;

    uint32_t maxCount = static_cast<uint32_t>(meshCapacity);
    if (maxCount == 0) maxCount = 1024;

    // Cascade compact + cascade count (full cascade buffers)
    cmdDrawIndexedIndirectCount(cmd, compactBuf.buffer, 0, countBuf.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
}
