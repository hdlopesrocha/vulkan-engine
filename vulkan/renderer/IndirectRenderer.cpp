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
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
}
} // anonymous namespace

void IndirectRenderer::acquireBuffers(VkCommandBuffer cmd) {
    if (vertexBuffer.buffer == VK_NULL_HANDLE && indexBuffer.buffer == VK_NULL_HANDLE) return;

    VkBufferMemoryBarrier barriers[2]{};
    uint32_t count = 0;
    if (vertexBuffer.buffer != VK_NULL_HANDLE) {
        barriers[count].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[count].srcAccessMask = 0;
        barriers[count].dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barriers[count].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[count].buffer = vertexBuffer.buffer;
        barriers[count].offset = 0;
        barriers[count].size = VK_WHOLE_SIZE;
        ++count;
    }
    if (indexBuffer.buffer != VK_NULL_HANDLE) {
        barriers[count].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[count].srcAccessMask = 0;
        barriers[count].dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
        barriers[count].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[count].buffer = indexBuffer.buffer;
        barriers[count].offset = 0;
        barriers[count].size = VK_WHOLE_SIZE;
        ++count;
    }
    if (count > 0) {
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 0, nullptr,
            count, barriers,
            0, nullptr);
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
    compactIndirectBuffer = {};
    boundsBuffer = {};
    visibleCountBuffer = {};

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
    // Clamp brush indices in newly inserted vertices to prevent OOB shader access
    size_t newStart = m.baseVertex;
    for (size_t vi = newStart; vi < mergedVertices.size(); ++vi) {
        for (int c = 0; c < 3; ++c)
            if (mergedVertices[vi].brushIndex > 16 || mergedVertices[vi].brushIndex < 0)
                mergedVertices[vi].brushIndex = 0;
    }
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
        if (doVertexUpload) {
            for (size_t v = info.baseVertex; v < info.baseVertex + meshVertexCount; ++v) {
                if (mergedVertices[v].brushIndex > 16 || mergedVertices[v].brushIndex < 0)
                    mergedVertices[v].brushIndex = 0;
            }
        }

        // Try ring buffer first; fall back to dedicated staging if full.
        StagingRingBuffer::Allocation aV{}, aI{};
        bool useRingV = false, useRingI = false;
        Buffer fallbackStagingV{}, fallbackStagingI{};
        VkBuffer sb = app->stagingRing.buffer();

        if (doVertexUpload) {
            aV = app->stagingRing.allocate(vertexSize);
            if (aV.mappedPtr) {
                memcpy(aV.mappedPtr, &mergedVertices[info.baseVertex], vertexSize);
                useRingV = true;
            } else {
                // Ring full — create dedicated staging buffer
                fallbackStagingV = app->createBuffer(vertexSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                void* d;
                vkMapMemory(app->getDevice(), fallbackStagingV.memory, 0, vertexSize, 0, &d);
                memcpy(d, &mergedVertices[info.baseVertex], vertexSize);
                vkUnmapMemory(app->getDevice(), fallbackStagingV.memory);
            }
        }
        if (doIndexUpload) {
            aI = app->stagingRing.allocate(indexSize);
            if (aI.mappedPtr) {
                memcpy(aI.mappedPtr, &mergedIndices[info.firstIndex], indexSize);
                useRingI = true;
            } else {
                if (useRingV) { app->stagingRing.release(aV.offset, vertexSize); useRingV = false; }
                fallbackStagingI = app->createBuffer(indexSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                void* d;
                vkMapMemory(app->getDevice(), fallbackStagingI.memory, 0, indexSize, 0, &d);
                memcpy(d, &mergedIndices[info.firstIndex], indexSize);
                vkUnmapMemory(app->getDevice(), fallbackStagingI.memory);
            }
        }

        VkDevice dev = app->getDevice();
        VkFence fence = app->runSingleTimeCommandsAsyncOnTransfer([&](VkCommandBuffer cmd) {
            if (doVertexUpload) {
                VkBuffer src = useRingV ? sb : fallbackStagingV.buffer;
                VkDeviceSize srcOff = useRingV ? aV.offset : 0;
                VkBufferCopy c{}; c.srcOffset=srcOff; c.dstOffset=vertexOffset; c.size=vertexSize;
                vkCmdCopyBuffer(cmd, src, vertexBuffer.buffer, 1, &c);
                recordTransferWriteRelease(cmd, vertexBuffer.buffer, vertexOffset, vertexSize);
            }
            if (doIndexUpload) {
                VkBuffer src = useRingI ? sb : fallbackStagingI.buffer;
                VkDeviceSize srcOff = useRingI ? aI.offset : 0;
                VkBufferCopy c{}; c.srcOffset=srcOff; c.dstOffset=indexOffset; c.size=indexSize;
                vkCmdCopyBuffer(cmd, src, indexBuffer.buffer, 1, &c);
                recordTransferWriteRelease(cmd, indexBuffer.buffer, indexOffset, indexSize);
            }
        }, nullptr);

        if (useRingV) { VkDeviceSize o=aV.offset,s=vertexSize; app->deferDestroyUntilFence(fence,[this,app,o,s](){app->stagingRing.release(o,s);}); }
        if (useRingI) { VkDeviceSize o=aI.offset,s=indexSize;  app->deferDestroyUntilFence(fence,[this,app,o,s](){app->stagingRing.release(o,s);}); }
        // Destroy fallback staging buffers after the copy completes
        if (fallbackStagingV.buffer != VK_NULL_HANDLE) {
            Buffer fv = fallbackStagingV;
            app->deferDestroyUntilFence(fence, [dev, fv, app]() {
                if (fv.buffer) { app->resources.removeBuffer(fv.buffer); vkDestroyBuffer(dev, fv.buffer, nullptr); }
                if (fv.memory) { app->resources.removeDeviceMemory(fv.memory); vkFreeMemory(dev, fv.memory, nullptr); }
            });
        }
        if (fallbackStagingI.buffer != VK_NULL_HANDLE) {
            Buffer fi = fallbackStagingI;
            app->deferDestroyUntilFence(fence, [dev, fi, app]() {
                if (fi.buffer) { app->resources.removeBuffer(fi.buffer); vkDestroyBuffer(dev, fi.buffer, nullptr); }
                if (fi.memory) { app->resources.removeDeviceMemory(fi.memory); vkFreeMemory(dev, fi.memory, nullptr); }
            });
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
    uploadMeshMetaBuffers(app);
    return true;
}

// Write all mesh indirect/model/bounds buffers for all active meshes
void IndirectRenderer::uploadMeshMetaBuffers(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;
    size_t activeIdx = 0;
    for (auto& kv : meshes) {
        MeshInfo& info = kv.second;
        if (!info.active) continue;
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.baseVertex);
        cmd.firstInstance = static_cast<uint32_t>(activeIdx);
        VkDeviceSize cmdOffset = activeIdx * sizeof(VkDrawIndexedIndirectCommand);
        VkDeviceSize cmdSize = sizeof(VkDrawIndexedIndirectCommand);
        void* data;
        vkMapMemory(app->getDevice(), indirectBuffer.memory, cmdOffset, cmdSize, 0, &data);
        memcpy(data, &cmd, cmdSize);
        vkUnmapMemory(app->getDevice(), indirectBuffer.memory);
        info.indirectOffset = cmdOffset;
        // Models SSBO removed: shaders use identity model matrices, skip writing models
        if (boundsBuffer.buffer != VK_NULL_HANDLE) {
            VkDeviceSize boundsOffset = activeIdx * 2 * sizeof(glm::vec4);
            glm::vec4 bounds[2] = { info.boundsMin, info.boundsMax };
            vkMapMemory(app->getDevice(), boundsBuffer.memory, boundsOffset, sizeof(bounds), 0, &data);
            memcpy(data, bounds, sizeof(bounds));
            vkUnmapMemory(app->getDevice(), boundsBuffer.memory);
        }
        ++activeIdx;
    }
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    std::lock_guard<std::mutex> guard(mutex);
    
    size_t activeMeshCount = 0;
    for (const auto& kv : meshes) if (kv.second.active) ++activeMeshCount;
    printf("[IndirectRenderer::rebuild] Called. dirty=%d, meshes.size()=%zu, activeMeshCount=%zu, mergedVertices=%zu, mergedIndices=%zu\n",
        dirty, meshes.size(), activeMeshCount, mergedVertices.size(), mergedIndices.size());
    
    if (!dirty) return;
    printf("[IndirectRenderer::rebuild] dirty=true, rebuilding buffers...\n");

    // Helper to schedule safe destruction of old buffers via the app's
    // deferred-destroy mechanism. The callback runs only after outstanding
    // async uploads and frame fences have completed, so rebuilds can replace
    // buffers without stalling the streaming path.
    // Defer destruction to the current frame's fence. By the time this frame's
    // render completes, all async command buffers (Solid360, WaterBackFace, etc.)
    // from previous frames that reference these buffers will also have completed,
    // because the render waits on their binary semaphores.
    auto scheduleDestroyBuffer = [&](const Buffer &b) {
        if (b.buffer == VK_NULL_HANDLE && b.memory == VK_NULL_HANDLE) return;
        Buffer copy = b;
        VkDevice dev = app->getDevice();
        VkFence fence = VK_NULL_HANDLE;
        uint32_t cf = app->getCurrentFrame();
        if (cf < app->inFlightFences.size()) fence = app->inFlightFences[cf];
        app->deferDestroyUntilFence(fence, [dev, copy, app]() {
            if (copy.buffer != VK_NULL_HANDLE) {
                if (app->resources.removeBuffer(copy.buffer)) vkDestroyBuffer(dev, copy.buffer, nullptr);
            }
            if (copy.memory != VK_NULL_HANDLE) {
                if (app->resources.removeDeviceMemory(copy.memory)) vkFreeMemory(dev, copy.memory, nullptr);
            }
        });
    };

    // Calculate required capacity with 25% headroom for incremental adds
    size_t neededVertexCap = mergedVertices.size() + mergedVertices.size() / 4 + 1024;
    size_t neededIndexCap = mergedIndices.size() + mergedIndices.size() / 4 + 4096;
    size_t neededMeshCap = activeMeshCount + activeMeshCount / 4 + 64;
    
    // Use max of current capacity or needed capacity (never shrink)
    if (neededVertexCap > vertexCapacity) vertexCapacity = neededVertexCap;
    if (neededIndexCap > indexCapacity) indexCapacity = neededIndexCap;
    if (neededMeshCap > meshCapacity) meshCapacity = neededMeshCap;

    // Build merged GPU-side vertex and index buffers from the CPU arrays.
    // If there are no meshes, free existing buffers.
    static bool printedBufferInfo = false;
    if (!printedBufferInfo) {
        printf("[IndirectRenderer::rebuild] mergedVertices.size()=%zu mergedIndices.size()=%zu\n", 
            mergedVertices.size(), mergedIndices.size());
        if (!mergedVertices.empty()) {
            printf("[IndirectRenderer::rebuild] Sample vertex[0]: pos=(%.2f,%.2f,%.2f)\n",
                mergedVertices[0].position.x, mergedVertices[0].position.y, mergedVertices[0].position.z);
        }
        printedBufferInfo = true;
    }
    if (mergedVertices.empty() || mergedIndices.empty()) {
        if (vertexBuffer.buffer != VK_NULL_HANDLE || vertexBuffer.memory != VK_NULL_HANDLE) {
            // Schedule safe destruction of the old vertex buffer and its memory
            scheduleDestroyBuffer(vertexBuffer);
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE || indexBuffer.memory != VK_NULL_HANDLE) {
            // Schedule safe destruction of the old index buffer and its memory
            scheduleDestroyBuffer(indexBuffer);
            indexBuffer = {};
        }
        vertexCapacity = 0;
        indexCapacity = 0;
    } else {
        // Recreate vertex/index buffers with capacity-based sizing
        if (vertexBuffer.buffer != VK_NULL_HANDLE || vertexBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(vertexBuffer);
            vertexBuffer = {};
        }
        if (indexBuffer.buffer != VK_NULL_HANDLE || indexBuffer.memory != VK_NULL_HANDLE) {
            scheduleDestroyBuffer(indexBuffer);
            indexBuffer = {};
        }
        
        // Create vertex buffer with capacity (not just current size).
        // HOST_VISIBLE avoids large device-local buffer copies that trigger
        // RADV instability on integrated GPUs (Renoir). The data is written
        // once per rebuild via direct memcpy — no staging buffer needed.
        VkDeviceSize vertexBufferSize = vertexCapacity * sizeof(Vertex);
        vertexBuffer = app->createBuffer(vertexBufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        // Create index buffer with capacity
        VkDeviceSize indexBufferSize = indexCapacity * sizeof(uint32_t);
        indexBuffer = app->createBuffer(indexBufferSize,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        // Upload current data via direct memcpy — no staging buffers or
        // vkCmdCopyBuffer needed. HOST_VISIBLE|HOST_COHERENT memory is
        // immediately visible to the GPU on integrated/shared-memory GPUs.
        bool doVertexUpload = !mergedVertices.empty();
        bool doIndexUpload = !mergedIndices.empty();
        VkDeviceSize vertexDataSize = mergedVertices.size() * sizeof(Vertex);
        VkDeviceSize indexDataSize = mergedIndices.size() * sizeof(uint32_t);

        if (doVertexUpload) {
            // Clamp brush indices before upload.
            static int clampedCount = 0;
            for (auto& v : mergedVertices) {
                if (v.brushIndex > 16 || v.brushIndex < 0) {
                    if (clampedCount < 10) fprintf(stderr, "[rebuild] clamping brushIndex %d → 0\n", v.brushIndex);
                    v.brushIndex = 0;
                    ++clampedCount;
                }
            }
            void* data;
            vkMapMemory(app->getDevice(), vertexBuffer.memory, 0, vertexDataSize, 0, &data);
            memcpy(data, mergedVertices.data(), vertexDataSize);
            vkUnmapMemory(app->getDevice(), vertexBuffer.memory);
        }
        if (doIndexUpload) {
            void* data;
            vkMapMemory(app->getDevice(), indexBuffer.memory, 0, indexDataSize, 0, &data);
            memcpy(data, mergedIndices.data(), indexDataSize);
            vkUnmapMemory(app->getDevice(), indexBuffer.memory);
        }

        // No vkCmdCopyBuffer needed — HOST_VISIBLE|HOST_COHERENT memory is
        // immediately visible to the GPU via the map/memcpy/unmap above.
        // The host write is automatically available to VERTEX_ATTRIBUTE_READ
        // and INDEX_READ accesses thanks to HOST_COHERENT.
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
    if (indirectBuffer.buffer != VK_NULL_HANDLE || indirectBuffer.memory != VK_NULL_HANDLE) {
        scheduleDestroyBuffer(indirectBuffer);
        indirectBuffer = {};
    }
    if (meshCapacity > 0) {
        indirectBuffer = app->createBuffer(indirectBufferSize, 
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (indirectDataSize > 0) {
            void* data;
            vkMapMemory(app->getDevice(), indirectBuffer.memory, 0, indirectDataSize, 0, &data);
            memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
            vkUnmapMemory(app->getDevice(), indirectBuffer.memory);
        }
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
    if (boundsBuffer.buffer != VK_NULL_HANDLE || boundsBuffer.memory != VK_NULL_HANDLE) {
        scheduleDestroyBuffer(boundsBuffer);
        boundsBuffer = {};
    }
    if (meshCapacity > 0) {
        // Use host-visible memory for bounds - updated when meshes change
        boundsBuffer = app->createBuffer(boundsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (boundsDataSize > 0) {
            void* bdata;
            vkMapMemory(app->getDevice(), boundsBuffer.memory, 0, boundsDataSize, 0, &bdata);
            memcpy(bdata, boundsData.data(), (size_t)boundsDataSize);
            vkUnmapMemory(app->getDevice(), boundsBuffer.memory);
        }
    }

    // Create/resize compact indirect buffer (storage + indirect usage)
    // Written by compute shader every frame, read by indirect draw — DEVICE_LOCAL
    // for optimal GPU performance on discrete GPUs.
    VkDeviceSize compactSize = indirectBufferSize;
    printf("[IndirectRenderer::rebuild] meshes=%zu activeCmds=%zu capacity=%zu\n", meshes.size(), cmds.size(), meshCapacity);
    if (compactIndirectBuffer.buffer != VK_NULL_HANDLE || compactIndirectBuffer.memory != VK_NULL_HANDLE) {
        scheduleDestroyBuffer(compactIndirectBuffer);
        compactIndirectBuffer = {};
    }
    if (compactSize > 0) {
        // HOST_VISIBLE avoids sync uploads that trigger RADV instability.
        compactIndirectBuffer = app->createBuffer(compactSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (indirectDataSize > 0) {
            void* data;
            vkMapMemory(app->getDevice(), compactIndirectBuffer.memory, 0, indirectDataSize, 0, &data);
            memcpy(data, indirectCommands.data(), (size_t)indirectDataSize);
            vkUnmapMemory(app->getDevice(), compactIndirectBuffer.memory);
        }
    }

    // Create or zero the visible count buffer (single uint).
    // Must be HOST_VISIBLE — readVisibleCount() maps it for CPU stats reads.
    VkDeviceSize countSize = sizeof(uint32_t);
    if (visibleCountBuffer.buffer != VK_NULL_HANDLE || visibleCountBuffer.memory != VK_NULL_HANDLE) {
        scheduleDestroyBuffer(visibleCountBuffer);
        visibleCountBuffer = {};
    }
    visibleCountBuffer = app->createBuffer(countSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Initialize visibleCount
    uint32_t initialCount = static_cast<uint32_t>(indirectCommands.size());
    {
        void* data;
        vkMapMemory(app->getDevice(), visibleCountBuffer.memory, 0, countSize, 0, &data);
        memcpy(data, &initialCount, countSize);
        vkUnmapMemory(app->getDevice(), visibleCountBuffer.memory);
    }

    // Create compute pipeline + descriptor set for GPU culling if not present
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

        if (vkCreateComputePipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
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
        if (app->allocateDescriptorSetsThreadSafe(&alloc, &computeDescriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor set");
        }
        app->resources.addDescriptorSet(computeDescriptorSet, "IndirectRenderer: computeDescriptorSet");
    }

    // Update compute descriptor set with buffer infos
    if (computeDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo inBuf{};
        inBuf.buffer = indirectBuffer.buffer;
        inBuf.offset = 0;
        inBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo outBuf{};
        outBuf.buffer = compactIndirectBuffer.buffer;
        outBuf.offset = 0;
        outBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo boundsBuf{};
        boundsBuf.buffer = boundsBuffer.buffer;
        boundsBuf.offset = 0;
        boundsBuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo countBuf{};
        countBuf.buffer = visibleCountBuffer.buffer;
        countBuf.offset = 0;
        countBuf.range = VK_WHOLE_SIZE;

        // Check required buffers are valid before updating descriptor set
        if (indirectBuffer.buffer == VK_NULL_HANDLE ||
            compactIndirectBuffer.buffer == VK_NULL_HANDLE ||
            boundsBuffer.buffer == VK_NULL_HANDLE ||
            visibleCountBuffer.buffer == VK_NULL_HANDLE) {
            std::cerr << "[IndirectRenderer] Skipping compute descriptor set update: one or more buffers are VK_NULL_HANDLE" << std::endl;
        } else {
            VkWriteDescriptorSet writes[4] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = computeDescriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &inBuf;

            writes[1] = writes[0];
            writes[1].dstBinding = 1; 
            writes[1].pBufferInfo = &outBuf;
            // bounds is at binding 2 in the shader
            writes[2] = writes[0]; 
            writes[2].dstBinding = 2;
            writes[2].pBufferInfo = &boundsBuf;
            // visible count is at binding 3 in the shader
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
}

void IndirectRenderer::prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws) {
    // NOTE: No mutex lock here - this is only called from the main render thread
    // and all buffer modifications happen in rebuild() which does lock.
    if (computePipeline == VK_NULL_HANDLE || compactIndirectBuffer.buffer == VK_NULL_HANDLE) {
        // No meshes loaded yet (e.g. during parallel background loading). Nothing to cull.
        return;
    }
    
    static bool printedOnce = false;
    if (!printedOnce) {
        uint32_t numCmds = static_cast<uint32_t>(indirectCommands.size());
        std::cout << "[IndirectRenderer::prepareCull] RUNNING: numCmds=" << numCmds
                  << ", computePipeline=" << (void*)computePipeline
                  << ", computeDescriptorSet=" << (void*)computeDescriptorSet << std::endl;
        printedOnce = true;
    }
    
    // Reset visible count to zero using a command (clear GPU-side counter)
    vkCmdFillBuffer(cmd, visibleCountBuffer.buffer, 0, sizeof(uint32_t), 0);

    // Barrier: vkCmdFillBuffer is a TRANSFER write. The compute shader does
    // atomicAdd on visibleCountBuffer, so we must ensure the fill is visible
    // before the dispatch starts. Without this the atomicAdd can race the
    // fill and read a stale value from the previous frame.
    {
        VkBufferMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        fillBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillBarrier.buffer = visibleCountBuffer.buffer;
        fillBarrier.offset = 0;
        fillBarrier.size = sizeof(uint32_t);
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &fillBarrier, 0, nullptr);
    }

    // Bind and dispatch compute cull
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, nullptr);
    uint32_t numCmds = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        numCmds = static_cast<uint32_t>(indirectCommands.size());
    }
    // Fast return if nothing to cull — avoids touching the pipeline at all
    if (numCmds == 0) return;

    uint8_t pc[72]; memcpy(pc, &viewProj, 64); uint32_t layer = 0; memcpy(pc+64, &layer, 4); memcpy(pc+68, &numCmds, 4);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 72, pc);

    uint32_t groupSize = 64;
    uint32_t groups = (numCmds + groupSize - 1) / groupSize;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier to make shader writes to the compact indirect buffer and visible count visible to indirect draw
    VkBufferMemoryBarrier barriers[2] = {};
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = compactIndirectBuffer.buffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    barriers[1] = barriers[0];
    barriers[1].buffer = visibleCountBuffer.buffer;
    barriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 2, barriers, 0, nullptr);
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

    // Reset visible count to zero using a command (clear GPU-side counter)
    vkCmdFillBuffer(cmd, outVisibleCountBuffer, 0, sizeof(uint32_t), 0);

    // Barrier: vkCmdFillBuffer is a TRANSFER write. The compute shader does
    // atomicAdd on visibleCountBuffer, so we must ensure the fill is visible
    // before the dispatch starts.
    {
        VkBufferMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        fillBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillBarrier.buffer = outVisibleCountBuffer;
        fillBarrier.offset = 0;
        fillBarrier.size = sizeof(uint32_t);
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &fillBarrier, 0, nullptr);
    }

    // Bind and dispatch compute cull using caller-provided descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDesc, 0, nullptr);

    uint32_t numCmds = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        numCmds = static_cast<uint32_t>(indirectCommands.size());
    }
    // Fast return if nothing to cull — avoids touching the pipeline at all
    if (numCmds == 0) return;

    uint8_t pc2[72]; memcpy(pc2, &viewProj, 64); uint32_t layer2 = 0; memcpy(pc2+64, &layer2, 4); memcpy(pc2+68, &numCmds, 4);
    vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 72, pc2);

    uint32_t groupSize = 64;
    uint32_t groups = (numCmds + groupSize - 1) / groupSize;
    if (groups > 0) vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier to make shader writes to the compact indirect buffer and visible count visible to indirect draw
    VkBufferMemoryBarrier barriers[2] = {};
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = outCompactBuffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    barriers[1] = barriers[0];
    barriers[1].buffer = outVisibleCountBuffer;
    barriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 2, barriers, 0, nullptr);
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
           //       (void*)visibleCountBuffer.buffer);
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
    cmdDrawIndexedIndirectCount(cmd, compactIndirectBuffer.buffer, 0, visibleCountBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
}

void IndirectRenderer::bindBuffers(VkCommandBuffer cmd) {
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
}

void IndirectRenderer::drawAll(VkCommandBuffer cmd) {
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE) return;
    if (indirectBuffer.buffer == VK_NULL_HANDLE || indirectCommands.empty()) return;

    // Bind merged geometry
    VkBuffer vbs[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Draw ALL meshes from the original (non-culled) indirect buffer
    uint32_t drawCount = static_cast<uint32_t>(indirectCommands.size());
    if (drawCount == 0) return; // nothing to draw
    vkCmdDrawIndexedIndirect(cmd, indirectBuffer.buffer, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
}

void IndirectRenderer::drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws) {
    drawIndirectOnly(cmd, app->getPipelineLayout(), maxDraws);
}

void IndirectRenderer::drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws) {
    if (compactIndirectBuffer.buffer == VK_NULL_HANDLE) {
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
    cmdDrawIndexedIndirectCount(cmd, compactIndirectBuffer.buffer, 0, visibleCountBuffer.buffer, 0, maxCount, sizeof(VkDrawIndexedIndirectCommand));
}

uint32_t IndirectRenderer::readVisibleCount(VulkanApp* app) const {
    if (!app || visibleCountBuffer.buffer == VK_NULL_HANDLE) return 0;
    // Stats-only: wait for GPU to finish so the counter is coherent before mapping.
    app->deviceWaitIdle();
    uint32_t count = 0;
    void* data = nullptr;
    if (vkMapMemory(app->getDevice(), visibleCountBuffer.memory, 0, sizeof(uint32_t), 0, &data) == VK_SUCCESS && data) {
        memcpy(&count, data, sizeof(uint32_t));
        vkUnmapMemory(app->getDevice(), visibleCountBuffer.memory);
    }
    return count;
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

    void* data = nullptr;
    VkDevice dev = app ? app->getDevice() : VK_NULL_HANDLE;
    if (dev == VK_NULL_HANDLE) return;
    if (vkMapMemory(dev, indirectBuffer.memory, offset, cmdSize, 0, &data) == VK_SUCCESS && data) {
        memcpy(data, &zeroCmd, cmdSize);
        vkUnmapMemory(dev, indirectBuffer.memory);
        std::cerr << "[IndirectRenderer] eraseMeshFromGPU: zeroed indirect cmd for mesh " << meshId << " at offset " << offset << std::endl;
        // Mark indirectOffset as invalid so future logic won't assume it
        info.indirectOffset = static_cast<VkDeviceSize>(-1);
    } else {
        std::cerr << "[IndirectRenderer] eraseMeshFromGPU: failed to map indirectBuffer memory for mesh " << meshId << std::endl;
    }
}
