#include "IndirectRenderer.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "../VulkanApp.hpp"
#include "../streaming/UploadManager.hpp"
#include "../../utils/FileReader.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {
// Record a buffer memory barrier to make transfer-written vertex/index data
// available. Buffers are CONCURRENT (accessible by both queues) so
// VK_QUEUE_FAMILY_IGNORED is correct. Timeline semaphore handles cross-queue
// execution ordering.
void recordTransferWriteRelease(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
    // Make transfer writes available and visible to vertex/index reads.
    // All transfers use the graphics queue with pipeline barriers.
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.bufferMemoryBarrierCount = 1;
    depInfo.pBufferMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}
} // anonymous namespace

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

    // Do NOT destroy the fence — processPendingCommandBuffers handles it.
    pendingTransfer = {};
}

void IndirectRenderer::pollPendingTransfers(VulkanApp* app) {
    if (pendingTransfer.fence == VK_NULL_HANDLE) return;
    VkDevice dev = app->getDevice();
    // If processPendingCommandBuffers already cleaned up the fence, the
    // transfer is done — skip vkGetFenceStatus on the destroyed handle.
    if (!app->resources.find((uintptr_t)pendingTransfer.fence).has_value()) {
        std::lock_guard<std::shared_mutex> lock(mutex);
        publishPendingTransfer(app);
        return;
    }
    VkResult r = vkGetFenceStatus(dev, pendingTransfer.fence);
    if (r == VK_NOT_READY) return;
    std::lock_guard<std::shared_mutex> lock(mutex);
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
    std::lock_guard<std::shared_mutex> guard(mutex);
    // For simplicity, just assign to the main vertexBuffer (per-mesh not tracked in this design)
    vertexBuffer = vbuf;
}

void IndirectRenderer::setIndexBufferForMesh(uint32_t meshId, Buffer ibuf) {
    std::lock_guard<std::shared_mutex> guard(mutex);
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
    std::lock_guard<std::shared_mutex> guard(mutex);
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

void IndirectRenderer::cleanup() {
    meshes.clear();
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
    return updateMesh(mesh, nextId++);
}

uint32_t IndirectRenderer::updateMesh(const Geometry& mesh, uint32_t customId) {
    std::lock_guard<std::shared_mutex> guard(mutex);
    //std::cout << "[IndirectRenderer::addMesh] Adding/replacing mesh ID " << customId << " with " << mesh.vertices.size() << " vertices and " << mesh.indices.size() << " indices.\n";

    MeshInfo m{};
    m.id = customId;
    m.baseVertex = static_cast<uint32_t>(mergedVertices.size());
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
    dirty = true;     // adding a mesh always requires a rebuild

    return customId;
}


void IndirectRenderer::removeMesh(uint32_t meshId) {
    std::lock_guard<std::shared_mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return;
    it->second.active = false;
    dirty = true;
}

void IndirectRenderer::removeAllMeshes() {
    std::lock_guard<std::shared_mutex> guard(mutex);
    meshes.clear();
    mergedVertices.clear();
    mergedIndices.clear();
    indirectCommands.clear();
    metaBuffersWrittenCount = 0;
    dirty = true;
}

bool IndirectRenderer::ensureCapacity(size_t vertexCount, size_t indexCount, size_t meshCount) {
    std::lock_guard<std::shared_mutex> guard(mutex);
    
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
    std::lock_guard<std::shared_mutex> guard(mutex);
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
            //printf("[IndirectRenderer::uploadMeshes] meshId %u not found\n", meshId);
            continue;
        }
        MeshInfo& info = it->second;
        if (!info.active) {
            //printf("[IndirectRenderer::uploadMeshes] meshId %u is inactive\n", meshId);
            continue;
        }
        if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
            //printf("[IndirectRenderer::uploadMeshes] buffers not created, need rebuild()\n");
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
            std::lock_guard<std::shared_mutex> lock(mutex);
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
        VkBufferMemoryBarrier2 vb{};
        vb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        vb.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
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
    std::shared_lock<std::shared_mutex> guard(mutex);
    return mergedVertices.size();
}

size_t IndirectRenderer::getMergedIndexCount() const {
    std::shared_lock<std::shared_mutex> guard(mutex);
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
    std::lock_guard<std::shared_mutex> guard(mutex);
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
                VkDeviceSize boundsOffset = i * 2 * sizeof(glm::vec4);
                glm::vec4 bounds[2] = { info->boundsMin, info->boundsMax };
                data = boundsBuffer.map(boundsOffset);
                memcpy(data, bounds, sizeof(bounds));
                boundsBuffer.unmap();
            }
        }
    }
    metaBuffersWrittenCount = indirectCommands.size();
}

// Unlocked — caller must hold mutex. Writes a single mesh's indirect command
// and bounds at its CURRENT drawIndex offset. Both indirectCommands[drawIndex]
// and meshes[id].drawIndex are read here under the lock, so they stay
// consistent even if a rebuild() reordered draw indices between the upload
// enqueue and its completion. The indirect/bounds buffers are append-only per
// slot, so writing one mesh's entry never disturbs in-flight draws of others.
void IndirectRenderer::publishMeshMeta(uint32_t meshId) {
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;
    auto it = meshes.find(meshId);
    if (it == meshes.end() || !it->second.active) return;
    MeshInfo& info = it->second;
    size_t i = info.drawIndex;
    if (i >= indirectCommands.size()) return;

    VkDeviceSize cmdOffset = i * sizeof(VkDrawIndexedIndirectCommand);
    const auto& cmd = indirectCommands[i];
    void* data = indirectBuffer.map(cmdOffset);
    memcpy(data, &cmd, sizeof(cmd));
    indirectBuffer.unmap();
    info.indirectOffset = cmdOffset;

    if (boundsBuffer.buffer != VK_NULL_HANDLE) {
        VkDeviceSize boundsOffset = i * 2 * sizeof(glm::vec4);
        glm::vec4 bounds[2] = { info.boundsMin, info.boundsMax };
        data = boundsBuffer.map(boundsOffset);
        memcpy(data, bounds, sizeof(bounds));
        boundsBuffer.unmap();
    }
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    // A rebuild may reallocate the merged vertex/index buffers below. Any
    // UploadJob still queued or in flight in the UploadManager captured the
    // CURRENT buffer handles at uploadMeshes() time; if we destroyed those
    // buffers while such a job is pending, its vkCmdCopyBuffer would target a
    // freed VkBuffer (VUID-vkCmdCopyBuffer-dstBuffer-parameter). Drain the
    // manager first so no pending job references a soon-to-be-destroyed buffer.
    // This MUST happen BEFORE acquiring `mutex`: flush() fires each job's
    // onComplete → publishMeshMeta, which locks the same (non-recursive) mutex.
    if (uploadMgr_) uploadMgr_->flush();

    std::lock_guard<std::shared_mutex> guard(mutex);

    // Publish any pending async upload before rebuilding.
    if (pendingTransfer.fence != VK_NULL_HANDLE) {
        publishPendingTransfer(app);
    }

    size_t activeMeshCount = 0;
    for (const auto& kv : meshes) if (kv.second.active) ++activeMeshCount;
#if 0
    printf("[IndirectRenderer::rebuild] Called. dirty=%d, meshes.size()=%zu, activeMeshCount=%zu, mergedVertices=%zu, mergedIndices=%zu\n",
        dirty, meshes.size(), activeMeshCount, mergedVertices.size(), mergedIndices.size());
#endif
    
    if (!dirty) return;
#if 0
    printf("[IndirectRenderer::rebuild] dirty=true, rebuilding buffers...\n");
#endif

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
#if 0
        printf("[IndirectRenderer::rebuild] mergedVertices.size()=%zu mergedIndices.size()=%zu\n", 
            mergedVertices.size(), mergedIndices.size());
        if (!mergedVertices.empty()) {
            printf("[IndirectRenderer::rebuild] Sample vertex[0]: pos=(%.2f,%.2f,%.2f)\n",
                mergedVertices[0].position.x, mergedVertices[0].position.y, mergedVertices[0].position.z);
        }
#endif
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
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
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

    // Upload bounds SSBO (two vec4s per active mesh: min, max)
    std::vector<glm::vec4> boundsData;
    boundsData.reserve(meshes.size() * 2);
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        boundsData.push_back(info.boundsMin);
        boundsData.push_back(info.boundsMax);
    }
    VkDeviceSize boundsBufferSize = sizeof(glm::vec4) * meshCapacity * 2;
    VkDeviceSize boundsDataSize = sizeof(glm::vec4) * boundsData.size();
    bool needNewBoundsBuffer = (boundsBuffer.buffer == VK_NULL_HANDLE) || (meshCapacity > oldMeshCapacity);
    if (needNewBoundsBuffer) {
        if (boundsBuffer.buffer != VK_NULL_HANDLE || boundsBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(boundsBuffer);
            boundsBuffer = {};
        }
        if (meshCapacity > 0) {
            boundsBuffer = app->createBuffer(boundsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
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

    // Create/resize compact indirect buffer (storage + indirect usage)
    // Written by compute shader every frame, read by indirect draw — DEVICE_LOCAL
    // for optimal GPU performance on discrete GPUs.
    VkDeviceSize compactSize = indirectBufferSize;
    //printf("[IndirectRenderer::rebuild] meshes=%zu activeCmds=%zu capacity=%zu\n", meshes.size(), cmds.size(), meshCapacity);
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
        VkDescriptorSetLayoutBinding bindings[4] = {};
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

        VkDescriptorBindingFlags bindingFlags[4] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        };

        DescriptorAllocator descAlloc{app->getDevice(), app};
        computeDescriptorSetLayout = descAlloc.createLayout(
            bindings, 4,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            bindingFlags,
            "IndirectRenderer: computeDescriptorSetLayout");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(glm::mat4) + sizeof(uint32_t) * 2;  // 72 bytes: mat4(64) + targetLayer(4) + numCmds(4)

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

void IndirectRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    // and all buffer modifications happen in rebuild() which does lock.
    Buffer& compactBuf = compactIndirectBuffers[currentCullFrame];
    Buffer& visibleCount = visibleCountBuffers[currentCullFrame];
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
#if 0
        uint32_t numCmds = 0;
        for (const auto& kv : meshes) if (kv.second.active) ++numCmds;
        std::cout << "[IndirectRenderer::prepareCull] RUNNING: numCmds=" << numCmds
                  << ", computePipeline=" << (void*)computePipeline
                  << ", computeDescriptorSet=" << (void*)computeDescriptorSet << std::endl;
#endif
        printedOnce = true;
    }
    
    // Barrier A: ensure prior indirect-draw reads, compute-shader atomics,
    // and prior fills of visibleCount/compactBuf are complete before we fill 0
    // again. Without this, a second cascade's vkCmdFillBuffer races with the
    // first cascade's fill (TRANSFER_WRITE→TRANSFER_WRITE hazard) and with the
    // first cascade's compute-shader atomicAdd (SHADER_WRITE→TRANSFER_WRITE).
    {
        VkBufferMemoryBarrier2 readBarriers[2] = {};
        readBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        readBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                  | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                  | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        readBarriers[0].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
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

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 2;
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

    // Barrier B: ensure the transfer write (zeroCount) and any prior
    // indirect-draw reads of compactBuf are complete before the compute
    // shader writes to both buffers.
    {
        VkBufferMemoryBarrier2 preBarriers[2] = {};
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

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 2;
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
        std::shared_lock<std::shared_mutex> lock(mutex);
        for (const auto& kv : meshes) if (kv.second.active) ++numCmds;
    }
    // Fast return if nothing to cull — avoids touching the pipeline at all
    if (numCmds == 0) return;

    uint8_t pc[72]; memcpy(pc, &viewProj, 64); uint32_t layer = 0; memcpy(pc+64, &layer, 4); memcpy(pc+68, &numCmds, 4);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 72, pc);

    uint32_t groupSize = 64;
    uint32_t groups = (numCmds + groupSize - 1) / groupSize;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier to make shader writes to the compact indirect buffer and visible count visible to indirect draw
    VkBufferMemoryBarrier2 barriers[2] = {};
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

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.bufferMemoryBarrierCount = 2;
    depInfo.pBufferMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void IndirectRenderer::prepareCullWithDescriptor(VkCommandBuffer cmd, const glm::mat4& viewProj, VkDescriptorSet computeDesc,
                                                VkBuffer outCompactBuffer, VkBuffer outVisibleCountBuffer, uint32_t maxDraws) {
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
    {
        VkBufferMemoryBarrier2 preFill{};
        preFill.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        preFill.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preFill.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preFill.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preFill.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preFill.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preFill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preFill.buffer = outVisibleCountBuffer;
        preFill.offset = 0;
        preFill.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &preFill;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
    vkCmdFillBuffer(cmd, outVisibleCountBuffer, 0, sizeof(uint32_t), 0);
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
    {
        VkBufferMemoryBarrier2 compactBarrier{};
        compactBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        compactBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT
                                    | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        compactBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT
                                    | VK_ACCESS_2_SHADER_WRITE_BIT;
        compactBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        compactBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        compactBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        compactBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        compactBarrier.buffer = outCompactBuffer;
        compactBarrier.offset = 0;
        compactBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &compactBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Bind and dispatch compute cull using caller-provided descriptor set
    if (cmdState) cmdState->bindComputePipeline(cmd, computePipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    if (cmdState) cmdState->bindComputeDescriptorSets(cmd, computePipelineLayout, 0, 1, &computeDesc, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDesc, 0, nullptr);

    uint32_t numCmds = 0;
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        for (const auto& kv : meshes) if (kv.second.active) ++numCmds;
    }
    // Fast return if nothing to cull — avoids touching the pipeline at all
    if (numCmds == 0) return;

    uint8_t pc2[72]; memcpy(pc2, &viewProj, 64); uint32_t layer2 = 0; memcpy(pc2+64, &layer2, 4); memcpy(pc2+68, &numCmds, 4);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 72, pc2);

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

    static int frameCount = 0;
    if (frameCount < 10) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        //printf("[IndirectRenderer::drawPrepared] Frame %d: vertexBuffer=%p (verts=%zu), indexBuffer=%p (indices=%zu), drawCommands=%zu\n",
        //       frameCount, (void*)vertexBuffer.buffer, mergedVertices.size(),
        //       (void*)indexBuffer.buffer, mergedIndices.size(), indirectCommands.size());
        size_t activeMeshCount = 0;
        for (const auto& kv : meshes) if (kv.second.active) ++activeMeshCount;
        //printf("[IndirectRenderer::drawPrepared] meshes.size()=%zu, activeMeshCount=%zu\n", meshes.size(), activeMeshCount);
        int meshPrint = 0;
        for (const auto& kv : meshes) {
            if (!kv.second.active) continue;
            // debugging info; keep call commented out to avoid build warnings
            // printf("  Mesh id=%u: baseVertex=%u, firstIndex=%u, indexCount=%u, boundsMin=(%.2f,%.2f,%.2f), boundsMax=(%.2f,%.2f,%.2f)\n",
            //     kv.second.id, kv.second.baseVertex, kv.second.firstIndex, kv.second.indexCount,
            //     kv.second.boundsMin.x, kv.second.boundsMin.y, kv.second.boundsMin.z,
            //     kv.second.boundsMax.x, kv.second.boundsMax.y, kv.second.boundsMax.z);
            if (++meshPrint >= 5) break;
        }
        // Print first few indirect commands
        for (size_t i = 0; i < std::min(size_t(3), indirectCommands.size()); ++i) {
            const auto& cmd = indirectCommands[i];
            //printf("  IndirectCmd[%zu]: indexCount=%u, instanceCount=%u, firstIndex=%u, vertexOffset=%d, firstInstance=%u\n",
            //    i, cmd.indexCount, cmd.instanceCount, cmd.firstIndex, cmd.vertexOffset, cmd.firstInstance);
        }
        // Check if using indirect count
        if (cmdDrawIndexedIndirectCount) {
           //printf("[IndirectRenderer::drawPrepared] Using GPU-driven count from visibleCountBuffer=%p\n",
           //       (void*)visibleCountBuffers[currentCullFrame].buffer);
        }
        frameCount++;
    }

    // Bind merged geometry
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Issue indirect-draw call; compute shader compacts only visible commands
    uint32_t maxCount = maxDraws > 0 ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
    
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
    std::shared_lock<std::shared_mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return empty;
    return it->second;
}



// Erase a mesh's indirect command on the GPU so it will not be drawn
// before a full `rebuild()` updates the indirect buffer. This attempts
// an immediate host-write to the `indirectBuffer` (if present) to zero
// the VkDrawIndexedIndirectCommand for the specified mesh id.
void IndirectRenderer::eraseMeshFromGPU(VulkanApp* app, uint32_t meshId) {
    std::lock_guard<std::shared_mutex> guard(mutex);
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
