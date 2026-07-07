#include "IndirectRenderer.hpp"
#include "../VulkanApp.hpp"
#include "../../utils/FileReader.hpp"
#include <cassert>
#include <cmath>
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
        vkWaitForFences(dev, 1, &pendingTransfer.fence, VK_TRUE, UINT64_MAX);
    }
    // Meta-buffers (indirect/draw-count) are append-only: a new mesh's
    // entry is only written once.  Calling doUploadMeshMetaBuffers
    // after the vertex/index data lands on the GPU is safe even when
    // earlier entries were written in a prior upload.
    doUploadMeshMetaBuffers(app);

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
        std::lock_guard<std::mutex> lock(mutex);
        publishPendingTransfer(app);
        return;
    }
    VkResult r = vkGetFenceStatus(dev, pendingTransfer.fence);
    if (r == VK_NOT_READY) return;
    std::lock_guard<std::mutex> lock(mutex);
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
    std::lock_guard<std::mutex> guard(mutex);
    // For simplicity, just assign to the main vertexBuffer (per-mesh not tracked in this design)
    vertexBuffer = vbuf;
}

void IndirectRenderer::setIndexBufferForMesh(uint32_t meshId, Buffer ibuf) {
    std::lock_guard<std::mutex> guard(mutex);
    indexBuffer = ibuf;
}

IndirectRenderer::IndirectRenderer() {}
IndirectRenderer::~IndirectRenderer() {}

void IndirectRenderer::init() {
}

void IndirectRenderer::cleanup() {
    // meshes no longer own per-mesh buffers; clear CPU lists
    meshes.clear();
    // Clear local handles; destruction is centralized in VulkanResourceManager.
    vertexBuffer = {};
    indexBuffer = {};
    indirectBuffer = {};
    for (auto& b : compactIndirectBuffers) b = {};
    boundsBuffer = {};
    for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
        if (visibleCountMapped[f] && storedDevice != VK_NULL_HANDLE) {
            visibleCountBuffers[f].unmap(); // VMA persistent mapping
            visibleCountMapped[f] = nullptr;
        }
        visibleCountBuffers[f] = {};
    }

    computePipeline = VK_NULL_HANDLE;
    computePipelineLayout = VK_NULL_HANDLE;
    computeDescriptorSetLayout = VK_NULL_HANDLE;
    computeDescriptorPool = VK_NULL_HANDLE;
}

uint32_t IndirectRenderer::addMesh(const Geometry& mesh) {
    return updateMesh(mesh, nextId++);
}

uint32_t IndirectRenderer::updateMesh(const Geometry& mesh, uint32_t customId) {
    std::lock_guard<std::mutex> guard(mutex);
    //std::cout << "[IndirectRenderer::addMesh] Adding/replacing mesh ID " << customId << " with " << mesh.vertices.size() << " vertices and " << mesh.indices.size() << " indices.\n";

    MeshInfo m{};
    m.id = customId;
    m.baseVertex = static_cast<uint32_t>(mergedVertices.size());
    m.firstIndex = static_cast<uint32_t>(mergedIndices.size());
    m.indexCount = static_cast<uint32_t>(mesh.indices.size());
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
    std::lock_guard<std::mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return;
    it->second.active = false;
    dirty = true;
}

bool IndirectRenderer::ensureCapacity(size_t vertexCount, size_t indexCount, size_t meshCount) {
    std::lock_guard<std::mutex> guard(mutex);
    
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

bool IndirectRenderer::uploadMeshVerticesAndIndices(VulkanApp* app, uint32_t meshId) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) {
        //printf("[IndirectRenderer::uploadMeshVerticesAndIndices] meshId %u not found\n", meshId);
        return false;
    }
    MeshInfo& info = it->second;
    if (!info.active) {
        //printf("[IndirectRenderer::uploadMeshVerticesAndIndices] meshId %u is inactive\n", meshId);
        return false;
    }
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) {
        //printf("[IndirectRenderer::uploadMeshVerticesAndIndices] buffers not created, need rebuild()\n");
        return false;
    }
    // Basic bounds validations to detect corrupted mesh data early.
    size_t indicesAvailable = mergedIndices.size();
    size_t verticesAvailable = mergedVertices.size();
    if (info.firstIndex + static_cast<uint64_t>(info.indexCount) > indicesAvailable) {
        std::cerr << "[IndirectRenderer] uploadMeshVerticesAndIndices: mesh " << meshId
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
        std::cerr << "[IndirectRenderer] uploadMeshVerticesAndIndices: mesh " << meshId
                  << " vertex range out of bounds: baseVertex=" << info.baseVertex
                  << " meshVertexCount=" << meshVertexCount
                  << " mergedVertices.size=" << verticesAvailable << std::endl;
        assert(false && "mesh vertex range out of bounds");
        return false;
    }

    if (info.baseVertex + meshVertexCount > vertexCapacity) {
        std::cerr << "[IndirectRenderer] uploadMeshVerticesAndIndices] vertex capacity exceeded (" << info.baseVertex << " + " << meshVertexCount << " > " << vertexCapacity << ")\n";
        return false;
    }
    if (info.firstIndex + info.indexCount > indexCapacity) {
        std::cerr << "[IndirectRenderer] uploadMeshVerticesAndIndices: index capacity exceeded (" << info.firstIndex << " + " << info.indexCount << " > " << indexCapacity << ")\n";
        return false;
    }

    // Ensure every index references a local vertex (before vertexOffset is applied)
    for (size_t i = info.firstIndex; i < info.firstIndex + info.indexCount; ++i) {
        uint32_t idx = mergedIndices[i];
        if (idx >= meshVertexCount) {
            std::cerr << "[IndirectRenderer] uploadMeshVerticesAndIndices: mesh " << meshId
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
            std::cerr << "[IndirectRenderer] uploadMeshVerticesAndIndices: mesh " << meshId
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
    if (doVertexUpload || doIndexUpload) {

        // Direct memcpy to HOST_VISIBLE staging buffer, then vkCmdCopyBuffer
        // to device-local vertex/index buffers. Device-local memory is
        // required because HOST_VISIBLE pages on RADV iGPU lack TCP-read
        // permission in the GPU page table.
        if (doVertexUpload || doIndexUpload) {
            VkDeviceSize stagingSize = (doVertexUpload ? vertexSize : 0)
                                     + (doIndexUpload  ? indexSize  : 0);
            Buffer staging = app->createBuffer(stagingSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* mapped = nullptr;
            mapped = staging.map(0);
            VkDeviceSize off = 0;
            if (doVertexUpload) {
                std::memcpy(static_cast<char*>(mapped) + off, &mergedVertices[info.baseVertex], vertexSize);
                off += vertexSize;
            }
            if (doIndexUpload) {
                std::memcpy(static_cast<char*>(mapped) + off, &mergedIndices[info.firstIndex], indexSize);
            }
            staging.unmap(); // VMA persistent mapping

            // Submit the staging→device-local copy asynchronously and
            // defer the meta-buffer write until the fence signals.
            // The meta-buffer (indirect + bounds) is append-only, so
            // publishing it later is safe: in-flight draws that already
            // reference earlier offsets still see valid data.
            if (pendingTransfer.fence != VK_NULL_HANDLE) {
                publishPendingTransfer(app);
            }
            pendingTransfer.fence = app->runSingleTimeCommandsAsync([&](VkCommandBuffer cmd) {
                // Barrier: prior vertex/index reads must complete before
                // the transfer writes to those buffers.
                VkBufferMemoryBarrier2 vb{};
                vb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                vb.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
                vb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                vb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                vb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vb.offset = 0;
                vb.size = VK_WHOLE_SIZE;
                if (doVertexUpload) {
                    vb.srcAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
                    vb.buffer = vertexBuffer.buffer;

                    VkDependencyInfo depInfo{};
                    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depInfo.bufferMemoryBarrierCount = 1;
                    depInfo.pBufferMemoryBarriers = &vb;
                    vkCmdPipelineBarrier2(cmd, &depInfo);
                }
                if (doIndexUpload) {
                    vb.srcAccessMask = VK_ACCESS_2_INDEX_READ_BIT;
                    vb.buffer = indexBuffer.buffer;

                    VkDependencyInfo depInfo{};
                    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depInfo.bufferMemoryBarrierCount = 1;
                    depInfo.pBufferMemoryBarriers = &vb;
                    vkCmdPipelineBarrier2(cmd, &depInfo);
                }
                VkDeviceSize o = 0;
                if (doVertexUpload) {
                    VkBufferCopy vCopy{};
                    vCopy.dstOffset = vertexOffset;
                    vCopy.size = vertexSize;
                    vkCmdCopyBuffer(cmd, staging.buffer, vertexBuffer.buffer, 1, &vCopy);
                    o += vertexSize;
                }
                if (doIndexUpload) {
                    VkBufferCopy iCopy{};
                    iCopy.srcOffset = o;
                    iCopy.dstOffset = indexOffset;
                    iCopy.size = indexSize;
                    vkCmdCopyBuffer(cmd, staging.buffer, indexBuffer.buffer, 1, &iCopy);
                }
            });
            pendingTransfer.stagingBuffer = staging;
        }
    }
    return true;
}

size_t IndirectRenderer::getMergedVertexCount() const {
    std::lock_guard<std::mutex> guard(mutex);
    return mergedVertices.size();
}

size_t IndirectRenderer::getMergedIndexCount() const {
    std::lock_guard<std::mutex> guard(mutex);
    return mergedIndices.size();
}

bool IndirectRenderer::uploadMesh(VulkanApp* app, uint32_t meshId) {
    if (!uploadMeshVerticesAndIndices(app, meshId)) {
        return false;
    }
    // uploadMeshMetaBuffers deferred until pendingTransfer fence signals.
    return true;
}

// Write all mesh indirect/model/bounds buffers for all active meshes
void IndirectRenderer::uploadMeshMetaBuffers(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);
    doUploadMeshMetaBuffers(app);
}

// Unlocked variant — caller must hold mutex.
void IndirectRenderer::doUploadMeshMetaBuffers(VulkanApp* app) {
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;

    // Append-only: skip already-written entries to avoid rewriting data
    // that in-flight GPU frames may be reading from.
    size_t activeIdx = 0;
    size_t skipped = 0;
    for (auto& kv : meshes) {
        MeshInfo& info = kv.second;
        if (!info.active) continue;
        if (skipped < metaBuffersWrittenCount) {
            ++skipped;
            ++activeIdx;
            continue;
        }

        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
        cmd.firstInstance = static_cast<uint32_t>(activeIdx);
        VkDeviceSize cmdOffset = activeIdx * sizeof(VkDrawIndexedIndirectCommand);
        VkDeviceSize cmdSize = sizeof(VkDrawIndexedIndirectCommand);
        void* data;
        data = indirectBuffer.map(cmdOffset);
        memcpy(data, &cmd, cmdSize);
        indirectBuffer.unmap(); // VMA persistent mapping
        info.indirectOffset = cmdOffset;
        if (boundsBuffer.buffer != VK_NULL_HANDLE) {
            VkDeviceSize boundsOffset = activeIdx * 2 * sizeof(glm::vec4);
            glm::vec4 bounds[2] = { info.boundsMin, info.boundsMax };
            data = boundsBuffer.map(boundsOffset);
            memcpy(data, bounds, sizeof(bounds));
            boundsBuffer.unmap(); // VMA persistent mapping
        }
        ++activeIdx;
    }
    metaBuffersWrittenCount = activeIdx;
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);

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
        app->deferDestroyUntilAllPending([app, copy]() {
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
    if (mergedVertices.empty() || mergedIndices.empty()) {
        // Free existing buffers — schedule deferred destruction so in-flight
        // GPU work completes before the memory is reclaimed.
        if (vertexBuffer.buffer != VK_NULL_HANDLE || vertexBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(vertexBuffer);
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE || indexBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(indexBuffer);
            indexBuffer = {};
        }
        vertexCapacity = 0;
        indexCapacity = 0;
    } else {
        // Determine whether the existing GPU buffers still have enough room.
        bool needNewVertexBuffer = (vertexBuffer.buffer == VK_NULL_HANDLE) || (vertexCapacity > oldVertexCapacity);
        bool needNewIndexBuffer = (indexBuffer.buffer == VK_NULL_HANDLE) || (indexCapacity > oldIndexCapacity);

        if (needNewVertexBuffer || needNewIndexBuffer) {
            // Capacity grew — destroy old buffers (deferred) and create new larger ones.
            if (vertexBuffer.buffer != VK_NULL_HANDLE || vertexBuffer.memory != VK_NULL_HANDLE) {
                scheduleDestroyBuffer(vertexBuffer);
                vertexBuffer = {};
            }
            if (indexBuffer.buffer != VK_NULL_HANDLE || indexBuffer.memory != VK_NULL_HANDLE) {
                scheduleDestroyBuffer(indexBuffer);
                indexBuffer = {};
            }

            VkDeviceSize vertexBufferSize = vertexCapacity * sizeof(Vertex);
            vertexBuffer = app->createBuffer(vertexBufferSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            VkDeviceSize indexBufferSize = indexCapacity * sizeof(uint32_t);
            indexBuffer = app->createBuffer(indexBufferSize,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        // else: existing buffers already have sufficient capacity — reuse them
        //       in-place, avoiding the memory spike of old+new allocations.

        // Upload current data via staging → device-local copy.
        // runSingleTimeCommands waits for the fence, so any in-flight GPU
        // reads of the existing buffer are guaranteed complete before the
        // transfer writes — no WRITE_AFTER_READ hazard.
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
    for (const auto& kv : meshes) {
        const MeshInfo& info = kv.second;
        if (!info.active) continue;
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
        cmd.firstInstance = static_cast<uint32_t>(cmds.size());
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
    if (indirectBuffer.buffer != VK_NULL_HANDLE && indirectDataSize > 0) {
        void* data;
        data = indirectBuffer.map(0);
        memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
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
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (indirectDataSize > 0) {
                void* data;
                data = compactIndirectBuffers[f].map(0);
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
        // Descriptor layout bindings match shaders/indirect.comp:
        //  binding 0 = inCmds, 1 = outCmds, 2 = bounds, 3 = VisibleCount
        VkDescriptorSetLayoutBinding bindings[4] = {};
        // inCmds (binding 0)
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // outCmds (binding 1)
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // bounds (binding 2)
        bindings[2].binding = 2;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // VisibleCount (binding 3)
        bindings[3].binding = 3;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        VkDescriptorBindingFlags bindingFlags[4] = {};
        bindingFlags[0] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        bindingFlags[1] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        bindingFlags[2] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        bindingFlags[3] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo{};
        flagsCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flagsCreateInfo.bindingCount = 4;
        flagsCreateInfo.pBindingFlags = bindingFlags;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 4; // bindings[0..3]
        layoutInfo.pBindings = bindings;
        layoutInfo.pNext = &flagsCreateInfo;
        // BindingFlags includes VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        // so the layout must be created with the UPDATE_AFTER_BIND_POOL flag.
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

        if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
        // central manager
        app->resources.addDescriptorSetLayout(computeDescriptorSetLayout, "IndirectRenderer: computeDescriptorSetLayout");

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

        auto compCode = FileReader::readFile("shaders/indirect.comp.spv");
        VkShaderModule compModule = app->createShaderModule(compCode);

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
            // Pipeline creation failed: unregister and destroy the shader module immediately
            app->resources.removeShaderModule(compModule);
            vkDestroyShaderModule(app->getDevice(), compModule, nullptr);
            throw std::runtime_error("failed to create compute pipeline!");
        }
        // track compute pipeline
        app->resources.addPipeline(computePipeline, "IndirectRenderer: computePipeline");
        // Clear local shader module reference; manager owns destruction
        compModule = VK_NULL_HANDLE;

        // Descriptor pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // Increase capacity to support many concurrent compute descriptor allocations
        poolSize.descriptorCount = 256;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        // Allow more descriptor sets to be allocated from this pool
        poolInfo.maxSets = 64;
        // Allow freeing descriptor sets if needed and support update-after-bind allocations
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor pool");
        }
        // track descriptor pool in central manager
        app->resources.addDescriptorPool(computeDescriptorPool, "IndirectRenderer: computeDescriptorPool");
        app->resources.addDescriptorPool(computeDescriptorPool, "IndirectRenderer: computeDescriptorPool");
        /* duplicate registration removed */

        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = computeDescriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &computeDescriptorSetLayout;
        for (uint32_t f = 0; f < MAX_CULL_FRAMES; f++) {
            if (app->allocateDescriptorSetsThreadSafe(&alloc, &computeDescriptorSets[f]) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate compute descriptor set");
            }
            app->resources.addDescriptorSet(computeDescriptorSets[f], "IndirectRenderer: computeDescriptorSet");
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

            VkWriteDescriptorSet writes[4] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = computeDescriptorSets[f];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &inBuf;

            writes[1] = writes[0];
            writes[1].dstBinding = 1; 
            writes[1].pBufferInfo = &outBuf;
            writes[2] = writes[0]; 
            writes[2].dstBinding = 2;
            writes[2].pBufferInfo = &boundsBufInfo;
            writes[3] = writes[0]; 
            writes[3].dstBinding = 3;
            writes[3].pBufferInfo = &countBuf;

            vkUpdateDescriptorSets(app->getDevice(), 4, writes, 0, nullptr);
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
    metaBuffersWrittenCount = 0; // rebuild rewrites all entries
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
    
    // Barrier A: ensure prior indirect-draw reads of visibleCount complete
    // before vkCmdFillBuffer writes 0.  Without this the write races with the
    // previous cascade's draw reading the count.
    {
        VkBufferMemoryBarrier2 readBarrier{};
        readBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        readBarrier.srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        readBarrier.srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        readBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        readBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        readBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        readBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        readBarrier.buffer = visibleCount.buffer;
        readBarrier.offset = 0;
        readBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &readBarrier;
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

    // Barrier B: ensure the transfer write (zeroCount) and any prior
    // indirect-draw reads of compactBuf are complete before the compute
    // shader writes to both buffers.
    {
        VkBufferMemoryBarrier2 preBarriers[2] = {};
        preBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        preBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preBarriers[0].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, nullptr);
    uint32_t numCmds = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
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
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
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

    // Reset visible count via host mapped write (outVisibleCountBuffer is HOST_VISIBLE|HOST_COHERENT).
    // vkCmdFillBuffer + TRANSFER_BIT barrier is unreliable on RADV.
    // The caller owns the buffer; we clear it with a global memory barrier + fill.
    vkCmdFillBuffer(cmd, outVisibleCountBuffer, 0, sizeof(uint32_t), 0);
    {
        VkMemoryBarrier2 fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fillBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        fillBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &fillBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Bind and dispatch compute cull using caller-provided descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDesc, 0, nullptr);

    uint32_t numCmds = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
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
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
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
        std::lock_guard<std::mutex> lock(mutex);
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
    const Buffer& visibleCount = visibleCountBuffers[currentCullFrame];
    if (!app || visibleCount.buffer == VK_NULL_HANDLE) return 0;
    // Stats-only: wait for GPU to finish so the counter is coherent.
    app->deviceWaitIdle();
    // Use persistent host mapping — avoid vkMapMemory on already-mapped memory
    if (visibleCountMapped[currentCullFrame]) {
        return *visibleCountMapped[currentCullFrame];
    }
    return 0;
}



IndirectRenderer::MeshInfo IndirectRenderer::getMeshInfo(uint32_t meshId) const {
    IndirectRenderer::MeshInfo empty;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = meshes.find(meshId);
    if (it == meshes.end()) return empty;
    return it->second;
}

std::vector<IndirectRenderer::MeshInfo> IndirectRenderer::getActiveMeshInfos() const {
    std::vector<MeshInfo> out;
    std::lock_guard<std::mutex> guard(mutex);
    for (const auto& kv : meshes) {
        if (kv.second.active) out.push_back(kv.second);
    }
    return out;
}

// Erase a mesh's indirect command on the GPU so it will not be drawn
// before a full `rebuild()` updates the indirect buffer. This attempts
// an immediate host-write to the `indirectBuffer` (if present) to zero
// the VkDrawIndexedIndirectCommand for the specified mesh id.
void IndirectRenderer::eraseMeshFromGPU(VulkanApp* app, uint32_t meshId) {
    std::lock_guard<std::mutex> guard(mutex);
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
