#include "AccelerationStructureManager.hpp"

#include <chrono>
#include <cstring>
#include <iostream>

#include "../VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "../../math/Vertex.hpp"

// Vertex layout contract with the RT closest-hit shaders (scalar block layout,
// byte-exact). Verified below with static_asserts so a Vertex change fails the
// build instead of silently corrupting BLAS fetches.
static constexpr VkDeviceSize kRtVertexStride = sizeof(Vertex);
static constexpr VkDeviceSize kRtPosOffset = offsetof(Vertex, position);
static constexpr VkDeviceSize kRtNormalOffset = offsetof(Vertex, normal);
static constexpr VkDeviceSize kRtUvOffset = offsetof(Vertex, texCoord);
static constexpr VkDeviceSize kRtBrushOffset = offsetof(Vertex, brushIndex);
static_assert(kRtPosOffset == 0, "Vertex::position must be at offset 0 for RT BLAS");
static_assert(sizeof(Vertex) >= 64, "Vertex unexpectedly small for RT fetch");

bool AccelerationStructureManager::init(VulkanApp* app) {
    if (!app || !app->supportsRayTracing()) {
        std::cerr << "[AS] Ray tracing not supported — AS manager disabled (rasterizer fallback)\n";
        return false;
    }
    app_ = app;
    // Scratch offset alignment is a hard VUID (device-dependent, 256 on the
    // reference RADV part). Query once; keep the spec-safe default otherwise.
    {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
        asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &asProps;
        vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &p2);
        if (asProps.minAccelerationStructureScratchOffsetAlignment > 1)
            scratchAlign_ = asProps.minAccelerationStructureScratchOffsetAlignment;
    }
    printf("[AS] incremental BLAS/TLAS ready (per-slot scratch/TLAS/instances/meta, align=%llu)\n",
           (unsigned long long)scratchAlign_);
    ready_ = true;
    tlasDirty_ = true;
    return true;
}

// Retire a buffer whose GPU lifetime may extend past this call (in-flight
// frames can still reference BLAS/TLAS storage, the instance buffer, slot
// meta or scratch being replaced). Destruction runs once no tracked command
// buffer is pending — never inline, never on a stale frame fence.
void AccelerationStructureManager::retireBuffer(VulkanApp* app, Buffer b) {
    if (b.buffer == VK_NULL_HANDLE || !app) return;
    app->deferDestroyUntilAllPending([app, b]() mutable {
        app->destroyBuffer(b);
    });
}

void AccelerationStructureManager::cleanup(VulkanApp* app) {
    if (!app) app = app_;
    if (!app) {
        blases_.clear();
        freshTlasSlot_ = UINT32_MAX;
        ready_ = false;
        return;
    }
    auto& rt = app->rtFunctions;
    for (auto& kv : blases_) {
        if (kv.second.as != VK_NULL_HANDLE && rt.destroyAS)
            rt.destroyAS(app->getDevice(), kv.second.as, nullptr);
        if (kv.second.storage.buffer != VK_NULL_HANDLE) app->destroyBuffer(kv.second.storage);
        kv.second.as = VK_NULL_HANDLE;
        kv.second.storage = {};
    }
    blases_.clear();
    pendingBlas_.clear();
    for (size_t i = 0; i < kFrameSlots; ++i) {
        if (tlasAS_[i] != VK_NULL_HANDLE && rt.destroyAS)
            rt.destroyAS(app->getDevice(), tlasAS_[i], nullptr);
        tlasAS_[i] = VK_NULL_HANDLE;
        if (tlasStorage_[i].buffer != VK_NULL_HANDLE) app->destroyBuffer(tlasStorage_[i]);
        tlasStorage_[i] = {};
        tlasStorageSize_[i] = 0;
    }
    freshTlasSlot_ = UINT32_MAX;
    for (auto& b : instanceBufs_) {
        if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
        b = {};
    }
    for (auto& b : scratchs_) {
        if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
        b = {};
    }
    for (auto& b : solidMetaBufs_) {
        if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
        b = {};
    }
    for (auto& b : waterMetaBufs_) {
        if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
        b = {};
    }
    scratchSizes_.fill(0);
    scratchAddrs_.fill(0);
    scratchUsables_.fill(0);
    ready_ = false;
}

void AccelerationStructureManager::destroyBlas(VulkanApp* app, BlasEntry& e) {
    // Both the AS object and its storage may still be referenced by in-flight
    // frames (old TLASes hold the BLAS address), so both retire deferred.
    if ((e.as != VK_NULL_HANDLE || e.storage.buffer != VK_NULL_HANDLE) && app) {
        VkAccelerationStructureKHR doomedAS = e.as;
        Buffer doomedStorage = e.storage;
        auto fnDestroy = app->rtFunctions.destroyAS;
        VkDevice dev = app->getDevice();
        app->deferDestroyUntilAllPending([app, dev, fnDestroy, doomedAS, doomedStorage]() mutable {
            if (doomedAS != VK_NULL_HANDLE && fnDestroy)
                fnDestroy(dev, doomedAS, nullptr);
            if (doomedStorage.buffer != VK_NULL_HANDLE)
                app->destroyBuffer(doomedStorage);
        });
    }
    e.as = VK_NULL_HANDLE;
    e.storage = {};
    e.storageSizeForBuild_ = 0;
    e.address = 0;
}

bool AccelerationStructureManager::ensureScratch(VulkanApp* app, VkDeviceSize bytes, uint32_t frameSlot) {
    const uint32_t slot = frameSlot % kFrameSlots;
    if (bytes == 0) return true;
    if (scratchs_[slot].buffer != VK_NULL_HANDLE && scratchUsables_[slot] >= bytes) return true;
    if (scratchs_[slot].buffer != VK_NULL_HANDLE) retireBuffer(app, scratchs_[slot]);
    scratchs_[slot] = app->createBuffer(bytes + scratchAlign_,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (scratchs_[slot].buffer == VK_NULL_HANDLE) {
        scratchSizes_[slot] = 0;
        scratchUsables_[slot] = 0;
        scratchAddrs_[slot] = 0;
        return false;
    }
    scratchSizes_[slot] = bytes + scratchAlign_;
    const VkDeviceAddress base = rt::getBufferAddress(app->getDevice(), scratchs_[slot].buffer);
    scratchAddrs_[slot] = (base + scratchAlign_ - 1) & ~(scratchAlign_ - 1);
    scratchUsables_[slot] = (base + scratchSizes_[slot]) - scratchAddrs_[slot];
    return true;
}

bool AccelerationStructureManager::ensureInstanceBuffer(VulkanApp* app, uint32_t instances) {
    if (instances == 0) return true;
    if (instanceBufs_[0].buffer != VK_NULL_HANDLE && instanceCap_ >= instances) return true;
    VkDeviceSize bytes = static_cast<VkDeviceSize>(instances) * sizeof(VkAccelerationStructureInstanceKHR);
    for (auto& b : instanceBufs_) {
        if (b.buffer != VK_NULL_HANDLE) retireBuffer(app, b);
        b = app->createBuffer(bytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer == VK_NULL_HANDLE) return false;
    }
    instanceCap_ = instances;
    return true;
}

bool AccelerationStructureManager::ensureMetaBuffer(VulkanApp* app, std::array<Buffer, kFrameSlots>& bufs, uint32_t& cap, uint32_t slots) {
    if (slots == 0) slots = 1;
    if (bufs[0].buffer != VK_NULL_HANDLE && cap >= slots) return true;
    VkDeviceSize bytes = static_cast<VkDeviceSize>(slots) * sizeof(SlotMeta);
    for (auto& b : bufs) {
        if (b.buffer != VK_NULL_HANDLE) retireBuffer(app, b);
        b = app->createBuffer(bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer == VK_NULL_HANDLE) return false;
        std::memset(b.mappedData, 0, static_cast<size_t>(bytes));
    }
    cap = slots;
    return true;
}

bool AccelerationStructureManager::syncFromScene(VulkanApp* app,
                                                const IndirectRenderer* solidIR,
                                                const IndirectRenderer* waterIR,
                                                const IndirectRenderer* /*vegHintIR*/,
                                                const LodSelect& lod,
                                                uint32_t frameSlot) {
    if (!ready_ || !app) return false;
    const auto t0 = std::chrono::steady_clock::now();
    curLod_ = lod;
    const uint32_t slot = frameSlot % kFrameSlots;

    struct LayerWork {
        const IndirectRenderer* ir;
        uint8_t layer;
        bool isWater;
        int maxLevel = 16;
    };
    LayerWork layers[2] = {{solidIR, 0, false}, {waterIR, 1, true}};
    if (solidIR) layers[0].maxLevel = solidIR->getMaxLodLevel();
    if (waterIR) layers[1].maxLevel = waterIR->getMaxLodLevel();

    // Mark-and-sweep: track which keys survive this snapshot.
    std::unordered_map<BlasKey, char, BlasKeyHash> seen;
    bool anyBlasDirty = false;

    for (auto& lw : layers) {
        if (!lw.ir) continue;
        // Capture shared-pool device addresses for the build step (no copies).
        VkDevice dev = app->getDevice();
        if (lw.isWater) {
            if (lw.ir->getVertexBuffer().buffer != VK_NULL_HANDLE)
                waterVtxAddrForBuild_ = rt::getBufferAddress(dev, lw.ir->getVertexBuffer().buffer);
            if (lw.ir->getIndexBuffer().buffer != VK_NULL_HANDLE)
                waterIdxAddrForBuild_ = rt::getBufferAddress(dev, lw.ir->getIndexBuffer().buffer);
        } else {
            if (lw.ir->getVertexBuffer().buffer != VK_NULL_HANDLE)
                solidVtxAddrForBuild_ = rt::getBufferAddress(dev, lw.ir->getVertexBuffer().buffer);
            if (lw.ir->getIndexBuffer().buffer != VK_NULL_HANDLE)
                solidIdxAddrForBuild_ = rt::getBufferAddress(dev, lw.ir->getIndexBuffer().buffer);
        }
        auto geoms = lw.ir->collectActiveSlotGeometries();
        // Size the slot-meta buffers to the pool capacity so shader indexing by
        // slot id can never run out of bounds. All three frame slots are kept
        // allocated; this frame writes its own slot (fence-guarded against
        // in-flight readers of the other slots).
        uint32_t cap = static_cast<uint32_t>(lw.ir->getMeshCapacity());
        auto& metaBufs = lw.isWater ? waterMetaBufs_ : solidMetaBufs_;
        uint32_t& metaCap = lw.isWater ? waterSlotCap_ : solidSlotCap_;
        if (!ensureMetaBuffer(app, metaBufs, metaCap, cap)) continue;
        SlotMeta* meta = static_cast<SlotMeta*>(metaBufs[slot].mappedData);
        for (auto& g : geoms) {
            BlasKey key{lw.layer, g.slotIndex};
            seen[key] = 1;
            auto it = blases_.find(key);
            const bool isNew = (it == blases_.end());
            BlasEntry* e = isNew ? &blases_[key] : &it->second;
            if (isNew) {
                *e = BlasEntry{};
                e->dirty = true;
            }
            if (e->baseVertex != g.baseVertex || e->vertexCount != g.vertexCount ||
                e->firstIndex != g.firstIndex || e->indexCount != g.indexCount || isNew) {
                e->baseVertex = g.baseVertex;
                e->vertexCount = g.vertexCount;
                e->firstIndex = g.firstIndex;
                e->indexCount = g.indexCount;
                if (!e->dirty) {
                    e->dirty = true;
                }
                anyBlasDirty = true;
                pendingBlas_.push_back(key);
            }
            // LoD band inputs ride along every sync (cheap); the TLAS gate
            // below decides rung membership from them.
            e->lodLevel = g.level;
            e->cellSize = g.boundsMax.x - g.boundsMin.x;
            e->lodBase = g.base;
            e->maxLevel = lw.maxLevel;
            if (meta && g.slotIndex < metaCap) {
                meta[g.slotIndex] = SlotMeta{g.baseVertex, g.firstIndex, g.vertexCount, g.indexCount};
            }
        }
    }

    // Sweep removed slots: destroy their BLAS (deferred) and drop instances.
    bool removed = false;
    for (auto it = blases_.begin(); it != blases_.end();) {
        if (seen.find(it->first) == seen.end()) {
            destroyBlas(app, it->second);
            it = blases_.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }

    // Rebuild the TLAS source list when membership changed, any BLAS that the
    // TLAS references was (re)built, or the LoD selection moved (camera
    // motion across a rung boundary with otherwise static geometry).
    // Incorporates new BLAS addresses after buildPending() — see the address
    // back-patch there.
    if (removed || anyBlasDirty || tlasSrc_.empty()) {
        rebuildTlasInstances();
        tlasDirty_ = true;
    }
    // LoD selection is re-evaluated every sync (a few flops per slot); only a
    // changed selected set dirties the TLAS, so a static camera issues zero
    // rebuilds in steady state.
    {
        const uint64_t selHash = rebuildTlasInstances();
        if (selHash != lastSelectionHash_) tlasDirty_ = true;
        lastSelectionHash_ = selHash;
    }

    const auto t1 = std::chrono::steady_clock::now();
    lastBlasBuildMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
    (void)lastBlasBuildMs_;
    return anyBlasDirty || tlasDirty_;
}

uint64_t AccelerationStructureManager::rebuildTlasInstances() {
    tlasSrc_.clear();
    tlasSrc_.reserve(blases_.size());
    // FNV-1a over the sorted selected (layer, slot) set.
    uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ULL;
    };
    std::vector<uint64_t> selected;
    selected.reserve(blases_.size());
    for (auto& kv : blases_) {
        const BlasEntry& e = kv.second;
        // Skip degenerate entries (no BLAS can be built for them anyway).
        if (e.vertexCount < 3 || e.indexCount < 3) continue;
        if (!lodRungSelected(e.lodLevel, e.cellSize, e.maxLevel, e.lodBase,
                             curLod_.camPos, curLod_.lodBias, curLod_.maxTargetLod))
            continue;
        TlasInstanceSrc s;
        s.key = kv.first;
        s.blasAddress = e.address;
        s.isWater = (kv.first.layer == 1);
        s.isVegetation = false;
        tlasSrc_.push_back(s);
        selected.push_back((static_cast<uint64_t>(kv.first.layer) << 32) | kv.first.slot);
    }
    std::sort(selected.begin(), selected.end());
    for (uint64_t v : selected) mix(v);
    mix(selected.size());
    // Overlap audit: group selected instances by (layer, 1m-quantized shared
    // anchor); chunk anchors sit on a 30m grid so quantization is exact for
    // column identity yet robust to float dust. More than one rung per group
    // means the band gate let overlapping rungs through.
    overlapColumns_ = 0;
    {
        std::unordered_map<uint64_t, uint32_t> levelMask;
        levelMask.reserve(tlasSrc_.size() * 2 + 1);
        for (auto& s : tlasSrc_) {
            auto bit = blases_.find(s.key);
            if (bit == blases_.end()) continue;
            const glm::vec3& b = bit->second.lodBase;
            const uint64_t qx = static_cast<uint64_t>(std::llround(b.x));
            const uint64_t qy = static_cast<uint64_t>(std::llround(b.y));
            const uint64_t qz = static_cast<uint64_t>(std::llround(b.z));
            uint64_t ck = (static_cast<uint64_t>(s.key.layer) << 56) ^
                          (qx * 73856093ULL) ^ (qy * 19349663ULL) ^ (qz * 83492791ULL);
            uint32_t& m = levelMask[ck];
            const int lv = std::clamp(bit->second.lodLevel, 0, 31);
            m |= (1u << static_cast<uint32_t>(lv));
        }
        uint32_t overlaps = 0;
        for (auto& kv : levelMask) {
            uint32_t m = kv.second;
            m = m - ((m >> 1) & 0x55555555u);
            m = (m & 0x33333333u) + ((m >> 2) & 0x33333333u);
            if ((((m + (m >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24) > 1) ++overlaps;
        }
        overlapColumns_ = overlaps;
    }
    return hash;
}

uint32_t AccelerationStructureManager::buildPending(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameSlot) {
    if (!ready_ || !app || cmd == VK_NULL_HANDLE) return 0;
    auto& rt = app->rtFunctions;
    if (!rt.cmdBuild || !rt.getBuildSizes || !rt.createAS || !rt.getAddress) return 0;
    VkDevice dev = app->getDevice();
    uint32_t built = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t slot = frameSlot % kFrameSlots;

    // No cross-submit primitives here by design: scratch, TLAS storage and
    // host-written inputs are all per frame slot (fence-ordered reuse), and
    // BLAS storage is freshly allocated per rebuild with deferred retirement.
    // Intra-CB barriers below order builds/dispatch within this submit.

    // Per-BLAS build. Each BLAS holds exactly one triangle geometry covering
    // the slot's index sub-range of the shared pool.
    VkDeviceSize maxScratch = 0;
    struct PendingBuild {
        BlasKey key;
        BlasEntry* entry;
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        VkAccelerationStructureBuildRangeInfoKHR range{};
        VkAccelerationStructureGeometryKHR geom{};
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
    };
    std::vector<PendingBuild> jobs;
    jobs.reserve(pendingBlas_.size());
    for (auto& key : pendingBlas_) {
        auto it = blases_.find(key);
        if (it == blases_.end()) continue;
        BlasEntry* e = &it->second;
        if (!e->dirty) continue;
        if (e->indexCount < 3 || e->vertexCount < 3) {
            e->dirty = false;
            continue;
        }
        // Geometry buffer addresses for this key's layer.
        VkDeviceAddress vtxAddr = (key.layer == 1) ? waterVtxAddrForBuild_ : solidVtxAddrForBuild_;
        VkDeviceAddress idxAddr = (key.layer == 1) ? waterIdxAddrForBuild_ : solidIdxAddrForBuild_;
        if (vtxAddr == 0 || idxAddr == 0) continue;
        PendingBuild j;
        j.key = key;
        j.entry = e;
        j.geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        j.geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        j.geom.flags = (key.layer == 1) ? VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR
                                        : VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& tri = j.geom.geometry.triangles;
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = vtxAddr + static_cast<VkDeviceAddress>(e->baseVertex) * kRtVertexStride + kRtPosOffset;
        tri.vertexStride = kRtVertexStride;
        tri.maxVertex = e->vertexCount;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = idxAddr + static_cast<VkDeviceAddress>(e->firstIndex) * sizeof(uint32_t);
        tri.transformData.deviceAddress = 0;
        tri.transformData.hostAddress = nullptr;
        j.buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        j.buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        j.buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        j.buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        j.buildInfo.geometryCount = 1;
        // Valid for the synchronous getBuildSizes below (j is alive here).
        // Re-patched after jobs.push_back copies j (see the build loop).
        j.buildInfo.pGeometries = &j.geom;
        j.sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        j.range.primitiveCount = e->indexCount / 3;
        j.range.primitiveOffset = 0;
        j.range.firstVertex = 0;
        j.range.transformOffset = 0;
        rt.getBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &j.buildInfo, &j.range.primitiveCount, &j.sizes);
        maxScratch = std::max(maxScratch, j.sizes.buildScratchSize);
        jobs.push_back(j);
    }

    // Allocate/resize each dirty BLAS's storage, then build one by one,
    // batching this frame slot's scratch buffer across all jobs in this frame.
    if (!jobs.empty()) {
        if (!ensureScratch(app, maxScratch, slot)) {
            pendingBlas_.clear();
            return 0;
        }
        for (auto& j : jobs) {
            j.buildInfo.pGeometries = &j.geom; // stable now (no more pushes)
            BlasEntry* e = j.entry;
            VkDeviceSize need = j.sizes.accelerationStructureSize;
            bool grown = (e->storage.buffer == VK_NULL_HANDLE) ||
                         (e->storageSizeForBuild_ < need);
            if (grown) {
                // Retire (deferred) rather than destroy: in-flight frames may
                // still traverse the old BLAS through their TLAS.
                if (e->as != VK_NULL_HANDLE || e->storage.buffer != VK_NULL_HANDLE) {
                    VkAccelerationStructureKHR oldAS = e->as;
                    Buffer oldStorage = e->storage;
                    auto fnDestroy = rt.destroyAS;
                    app->deferDestroyUntilAllPending([app, dev, fnDestroy, oldAS, oldStorage]() mutable {
                        if (oldAS != VK_NULL_HANDLE && fnDestroy)
                            fnDestroy(dev, oldAS, nullptr);
                        if (oldStorage.buffer != VK_NULL_HANDLE)
                            app->destroyBuffer(oldStorage);
                    });
                    e->as = VK_NULL_HANDLE;
                    e->storage = {};
                    e->address = 0;
                }
                e->storage = app->createBuffer(need,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                e->storageSizeForBuild_ = need;
                if (e->storage.buffer == VK_NULL_HANDLE) continue;
                VkAccelerationStructureCreateInfoKHR ci{};
                ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                ci.size = need;
                ci.buffer = e->storage.buffer;
                if (rt.createAS(dev, &ci, nullptr, &e->as) != VK_SUCCESS) {
                    e->as = VK_NULL_HANDLE;
                    continue;
                }
            }
            j.buildInfo.dstAccelerationStructure = e->as;
            j.buildInfo.scratchData.deviceAddress = scratchAddrs_[slot];
            const VkAccelerationStructureBuildRangeInfoKHR* prange = &j.range;
            rt.cmdBuild(cmd, 1, &j.buildInfo, &prange);
            // Barrier between consecutive BLAS builds sharing the scratch
            // buffer (write-after-write on scratch) and the geometry pools.
            // dst must include WRITE (next build rewrites scratch), not just
            // READ — otherwise validation reports WRITE_AFTER_WRITE.
            VkMemoryBarrier2 mem{};
            mem.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mem.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mem.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mem.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mem.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &mem;
            vkCmdPipelineBarrier2(cmd, &dep);
            e->address = rt::getASAddress(dev, rt.getAddress, e->as);
            e->dirty = false;
            ++built;
            ++totalBlasBuilds_;
        }
        pendingBlas_.clear();
        // BLAS addresses changed -> refresh TLAS sources and force a rebuild.
        rebuildTlasInstances();
        tlasDirty_ = true;
    } else {
        pendingBlas_.clear();
    }

    // TLAS build (only when dirty — never every frame in steady state).
    if (tlasDirty_ && !tlasSrc_.empty()) {
        // When BLASes were (re)built earlier in THIS command buffer, their
        // writes must complete before the TLAS build reads them back — and the
        // TLAS build reuses the same scratch buffer (write-after-write), so
        // dst needs WRITE as well as READ.
        if (built > 0) {
            VkMemoryBarrier2 mem{};
            mem.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mem.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mem.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mem.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mem.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &mem;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        const auto tt0 = std::chrono::steady_clock::now();
        uint32_t n = static_cast<uint32_t>(tlasSrc_.size());
        if (!ensureInstanceBuffer(app, n)) {
            return built;
        }
        auto* dst = static_cast<VkAccelerationStructureInstanceKHR*>(instanceBufs_[slot].mappedData);
        uint32_t w = 0;
        for (auto& s : tlasSrc_) {
            auto bit = blases_.find(s.key);
            if (bit == blases_.end() || bit->second.as == VK_NULL_HANDLE || bit->second.address == 0)
                continue;
            VkAccelerationStructureInstanceKHR inst{};
            inst.transform.matrix[0][0] = 1.0f; inst.transform.matrix[1][1] = 1.0f; inst.transform.matrix[2][2] = 1.0f;
            inst.instanceCustomIndex = rt::makeCustomIndex(s.key.slot, s.isWater, s.isVegetation);
            inst.mask = s.isWater ? 0x02 : 0x01;
            inst.instanceShaderBindingTableRecordOffset = s.isWater ? 1 : 0;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = bit->second.address;
            dst[w++] = inst;
        }
        tlasInstanceCount_ = w;
        if (w > 0) {
            VkAccelerationStructureGeometryKHR tgeom{};
            tgeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tgeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tgeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            tgeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            tgeom.geometry.instances.arrayOfPointers = VK_FALSE;
            tgeom.geometry.instances.data.deviceAddress = rt::getBufferAddress(dev, instanceBufs_[slot].buffer);
            VkAccelerationStructureBuildGeometryInfoKHR tinfo{};
            tinfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tinfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tinfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            tinfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            tinfo.geometryCount = 1;
            tinfo.pGeometries = &tgeom;
            VkAccelerationStructureBuildRangeInfoKHR trange{};
            trange.primitiveCount = w;
            VkAccelerationStructureBuildSizesInfoKHR tsizes{};
            tsizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            rt.getBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tinfo, &trange.primitiveCount, &tsizes);
            if (ensureScratch(app, tsizes.buildScratchSize, slot)) {
                // Per-slot TLAS storage: rebuilt in place when this slot is
                // recorded, reused only after this slot's frame fence signals
                // (same fence-ordered reuse as the engine's per-slot cull
                // buffers). Grows rarely; the replaced pair retires deferred.
                if (tlasStorage_[slot].buffer == VK_NULL_HANDLE ||
                    tlasStorageSize_[slot] < tsizes.accelerationStructureSize) {
                    if (tlasAS_[slot] != VK_NULL_HANDLE || tlasStorage_[slot].buffer != VK_NULL_HANDLE) {
                        VkAccelerationStructureKHR oldTlas = tlasAS_[slot];
                        Buffer oldStorage = tlasStorage_[slot];
                        auto fnDestroy = rt.destroyAS;
                        app->deferDestroyUntilAllPending([app, dev, fnDestroy, oldTlas, oldStorage]() mutable {
                            if (oldTlas != VK_NULL_HANDLE && fnDestroy)
                                fnDestroy(dev, oldTlas, nullptr);
                            if (oldStorage.buffer != VK_NULL_HANDLE)
                                app->destroyBuffer(oldStorage);
                        });
                        tlasAS_[slot] = VK_NULL_HANDLE;
                        tlasStorage_[slot] = {};
                    }
                    tlasStorage_[slot] = app->createBuffer(tsizes.accelerationStructureSize,
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    tlasStorageSize_[slot] = tsizes.accelerationStructureSize;
                    tlasAS_[slot] = VK_NULL_HANDLE;
                    if (tlasStorage_[slot].buffer != VK_NULL_HANDLE) {
                        VkAccelerationStructureCreateInfoKHR ci{};
                        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
                        ci.size = tsizes.accelerationStructureSize;
                        ci.buffer = tlasStorage_[slot].buffer;
                        rt.createAS(dev, &ci, nullptr, &tlasAS_[slot]);
                    }
                }
                if (tlasAS_[slot] != VK_NULL_HANDLE) {
                    tinfo.dstAccelerationStructure = tlasAS_[slot];
                    tinfo.scratchData.deviceAddress = scratchAddrs_[slot];
                    const VkAccelerationStructureBuildRangeInfoKHR* pr = &trange;
                    rt.cmdBuild(cmd, 1, &tinfo, &pr);
                    freshTlasSlot_ = slot;
                    ++built;
                    ++totalTlasBuilds_;
                    // TLAS build -> ray-tracing shader read barrier. Recorded
                    // here (not at dispatch) so any consumer in this command
                    // buffer observes the completed TLAS.
                    VkMemoryBarrier2 mem{};
                    mem.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mem.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                    mem.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    mem.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mem.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT;
                    VkDependencyInfo dep{};
                    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers = &mem;
                    vkCmdPipelineBarrier2(cmd, &dep);
                    const auto tt1 = std::chrono::steady_clock::now();
                    lastTlasBuildMs_ = std::chrono::duration<float, std::milli>(tt1 - tt0).count();
                }
            }
        }
        tlasDirty_ = false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    if (built > 0)
        lastBlasBuildMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return built;
}

AccelerationStructureManager::Stats AccelerationStructureManager::stats() const {
    Stats s;
    s.blasCount = static_cast<uint32_t>(blases_.size());
    s.tlasInstances = tlasInstanceCount_;
    s.overlapColumns = overlapColumns_;
    s.pendingBlasBuilds = static_cast<uint32_t>(pendingBlas_.size());
    s.tlasDirty = tlasDirty_;
    s.lastBlasBuildMs = lastBlasBuildMs_;
    s.lastTlasBuildMs = lastTlasBuildMs_;
    s.totalBlasBuilds = totalBlasBuilds_;
    s.totalTlasBuilds = totalTlasBuilds_;
    return s;
}
