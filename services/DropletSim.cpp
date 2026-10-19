#include "DropletSim.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../vulkan/core/VulkanApp.hpp"

namespace {

// Push-constant block: field-for-field mirror of `layout(push_constant) PC` in
// shaders/computation/water/DropletSim.comp (5 * vec4 = 80 bytes).
struct DropletSimPC {
    glm::vec4 spawnCenterRadius; // xyz = spawn center, w = spawn radius
    glm::vec4 regionMin;         // xyz = region min, w = water level
    glm::vec4 regionMax;         // xyz = region max, w = grid cell size
    glm::vec4 physics;           // gravity, restitution, friction, damping
    glm::vec4 misc;              // dt, particleCount, resetSeed bits, pass
};
static_assert(sizeof(DropletSimPC) == 80, "DropletSimPC must match the shader push block");

// Shader pass ids (pc.misc.w).
enum SimPass : uint32_t {
    kPassSpawn = 0,
    kPassIntegrate = 1,
    kPassGridClear = 2,
    kPassGridScatter = 3,
    kPassContacts = 4,
    kPassFinalize = 5,
};

constexpr uint32_t kLocalSize = 64;
constexpr size_t kDropletStride = 48; // vec3+float * 3 rows (see the shader struct)

uint32_t groupsFor(uint32_t items, uint32_t local) {
    return (items + local - 1u) / local;
}

} // namespace

void DropletSim::init(VulkanApp* vkApp) {
    if (initialized_) return;
    app_ = vkApp;
    createPipeline();
    createBuffers();
    initialized_ = true;
}

void DropletSim::destroy(VulkanApp* vkApp) {
    if (vkApp == nullptr) return;
    vkApp->destroyBuffer(state_);
    vkApp->destroyBuffer(shapeStage_);
    vkApp->destroyBuffer(gridCounts_);
    vkApp->destroyBuffer(gridEntries_);
    vkApp->destroyBuffer(gridExtra_);
    vkApp->destroyBuffer(statsRead_);
    vkApp->destroyBuffer(probeStage_);
    vkApp->destroyBuffer(bSdf0_);
    vkApp->destroyBuffer(bSdf1_);
    vkApp->destroyBuffer(bCube_);
    vkApp->destroyBuffer(bMeta_);
    vkApp->destroyBuffer(bChild03_);
    vkApp->destroyBuffer(bChild47_);
    // Pipeline/layout/pool/set are owned by VulkanResourceManager and released
    // by its cleanup; only the handles are dropped here.
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    descLayout_ = VK_NULL_HANDLE;
    descPool_ = VK_NULL_HANDLE;
    descSet_ = VK_NULL_HANDLE;
    nodes_.clear();
    nodeCap_ = 0;
    initialized_ = false;
}

void DropletSim::createPipeline() {
    VkDevice device = app_->getDevice();

    VkDescriptorSetLayoutBinding bindings[10] = {};
    for (int b = 0; b < 10; ++b) {
        bindings[b].binding = static_cast<uint32_t>(b);
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 10;
    dli.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("DropletSim: descriptor set layout");
    app_->resources.addDescriptorSetLayout(descLayout_, "DropletSim: descLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(DropletSimPC);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &descLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &layout_) != VK_SUCCESS)
        throw std::runtime_error("DropletSim: pipeline layout");
    app_->resources.addPipelineLayout(layout_, "DropletSim: layout");

    VkShaderModule sm = app_->getOrCreateShaderModule("shaders/computation/water/DropletSim.comp.spv");
    VkPipelineShaderStageCreateInfo ssi{};
    ssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssi.module = sm;
    ssi.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = ssi;
    cpi.layout = layout_;
    if (vkCreateComputePipelines(device, app_->getPipelineCache(), 1, &cpi, nullptr, &pipeline_) != VK_SUCCESS)
        throw std::runtime_error("DropletSim: compute pipeline");
    app_->resources.addPipeline(pipeline_, "DropletSim: pipeline");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 10;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    dpi.maxSets = 1;
    if (vkCreateDescriptorPool(device, &dpi, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("DropletSim: descriptor pool");
    app_->resources.addDescriptorPool(descPool_, "DropletSim: descPool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout_;
    if (vkAllocateDescriptorSets(device, &ai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("DropletSim: descriptor set");
    app_->resources.addDescriptorSet(descSet_, "DropletSim: descSet");
}

void DropletSim::createBuffers() {
    const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkMemoryPropertyFlags host =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Particle state: device-local, host-visible only through the one-shot
    // shape copy. Zero-init so a dispatch before the first spawn sees flags 0.
    state_ = app_->createBuffer(kMaxParticles * kDropletStride,
                                storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                deviceLocal, true);
    // Radius/mass source pattern (one entry per capacity slot); the host
    // writes it and the GPU copies it into the state buffer once per shape
    // change, so no per-frame upload ever touches particle data.
    shapeStage_ = app_->createBuffer(kMaxParticles * kDropletStride, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     host, true);
    gridCounts_ = app_->createBuffer(kGridBuckets * sizeof(uint32_t), storage, deviceLocal, true);
    gridEntries_ = app_->createBuffer(kGridBuckets * kGridMaxPerBucket * sizeof(uint32_t), storage,
                                      deviceLocal, true);
    gridExtra_ = app_->createBuffer(4 * sizeof(uint32_t),
                                    storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    deviceLocal, true);
    statsRead_ = app_->createBuffer(4 * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, true);
    probeStage_ = app_->createBuffer(kMaxParticles * kDropletStride, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     host, true);

    // Octree node tables start empty; uploadRegion() sizes them.
    writeDescriptors();
}

void DropletSim::ensureNodeTables(size_t nodeCount) {
    if (nodeCount <= nodeCap_ && bSdf0_.buffer != VK_NULL_HANDLE) return;
    const VkMemoryPropertyFlags host =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    app_->destroyBuffer(bSdf0_);
    app_->destroyBuffer(bSdf1_);
    app_->destroyBuffer(bCube_);
    app_->destroyBuffer(bMeta_);
    app_->destroyBuffer(bChild03_);
    app_->destroyBuffer(bChild47_);

    const size_t cap = std::max<size_t>(nodeCount, 1024);
    bSdf0_ = app_->createBuffer(cap * sizeof(glm::vec4), usage, host);
    bSdf1_ = app_->createBuffer(cap * sizeof(glm::vec4), usage, host);
    bCube_ = app_->createBuffer(cap * sizeof(glm::vec4), usage, host);
    bMeta_ = app_->createBuffer(cap * sizeof(glm::ivec4), usage, host);
    bChild03_ = app_->createBuffer(cap * sizeof(glm::ivec4), usage, host);
    bChild47_ = app_->createBuffer(cap * sizeof(glm::ivec4), usage, host);
    nodeCap_ = cap;
    writeDescriptors();
}

void DropletSim::uploadRegion(const std::vector<OctreeGpuNode>& nodes) {
    if (!initialized_ || nodes.empty()) return;
    const size_t count = std::min(nodes.size(), kMaxRegionNodes);
    ensureNodeTables(count);

    // Same split the topology pass uses (bindings 0..6 there): two vec4 SDF
    // rows, cube min+size, meta (lod, type, brush, flags) and the two child
    // ivec4 blocks. The node order is preserved verbatim — the shader's
    // descent depends on the children indices the builder wrote.
    auto* sdf0 = static_cast<glm::vec4*>(bSdf0_.mappedData);
    auto* sdf1 = static_cast<glm::vec4*>(bSdf1_.mappedData);
    auto* cube = static_cast<glm::vec4*>(bCube_.mappedData);
    auto* meta = static_cast<glm::ivec4*>(bMeta_.mappedData);
    auto* c03 = static_cast<glm::ivec4*>(bChild03_.mappedData);
    auto* c47 = static_cast<glm::ivec4*>(bChild47_.mappedData);
    if (sdf0 == nullptr || sdf1 == nullptr || cube == nullptr || meta == nullptr ||
        c03 == nullptr || c47 == nullptr) {
        std::cerr << "[DropletSim] node table upload failed: buffers not mapped\n";
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        const OctreeGpuNode& n = nodes[i];
        sdf0[i] = glm::vec4(n.sdf[0], n.sdf[1], n.sdf[2], n.sdf[3]);
        sdf1[i] = glm::vec4(n.sdf[4], n.sdf[5], n.sdf[6], n.sdf[7]);
        cube[i] = glm::vec4(n.cubeMin, n.cubeSize);
        meta[i] = glm::ivec4(n.lod, n.type, n.brushIndex, n.flags);
        c03[i] = glm::ivec4(n.children[0], n.children[1], n.children[2], n.children[3]);
        c47[i] = glm::ivec4(n.children[4], n.children[5], n.children[6], n.children[7]);
    }
    nodes_ = nodes;
    // Region bounds from the uploaded cubes (used by the CPU height query).
    regionMin_ = glm::vec3(1e30f);
    regionMax_ = glm::vec3(-1e30f);
    for (size_t i = 0; i < count; ++i) {
        const OctreeGpuNode& n = nodes_[i];
        regionMin_ = glm::min(regionMin_, n.cubeMin);
        regionMax_ = glm::max(regionMax_, n.cubeMin + glm::vec3(n.cubeSize));
    }
    nodesDirty_ = true;
}

void DropletSim::setParticleShape(float radius, float mass) {
    shapeRadius_ = radius;
    shapeMass_ = mass;
    shapeDirty_ = true;
}

void DropletSim::requestReset() {
    resetPending_ = true;
    ++resetSeed_; // deterministic per reset: same seed -> same initial layout
}

void DropletSim::writeDescriptors() {
    VkBuffer bufs[10] = {state_.buffer,      gridCounts_.buffer, gridEntries_.buffer,
                         gridExtra_.buffer,  bSdf0_.buffer,      bSdf1_.buffer,
                         bCube_.buffer,      bMeta_.buffer,      bChild03_.buffer,
                         bChild47_.buffer};
    // Every declared binding must name a valid buffer even before the region
    // is uploaded: fall back to the state buffer (never touched in that state).
    for (int b = 0; b < 10; ++b) {
        if (bufs[b] == VK_NULL_HANDLE) bufs[b] = state_.buffer;
    }
    VkDescriptorBufferInfo infos[10];
    VkWriteDescriptorSet writes[10] = {};
    for (int b = 0; b < 10; ++b) {
        infos[b] = {bufs[b], 0, app_->getClampedWholeSizeRange(bufs[b])};
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = descSet_;
        writes[b].dstBinding = static_cast<uint32_t>(b);
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    vkUpdateDescriptorSets(app_->getDevice(), 10, writes, 0, nullptr);
}

void DropletSim::computeBarrier(VkCommandBuffer cmd) {
    // Compute writes (particle state, hash buckets, counters) feed the next
    // dispatch's reads. One memory barrier per pass boundary keeps the
    // dependency explicit; the passes never overlap on purpose.
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void DropletSim::dispatch(VkCommandBuffer cmd, float substepSeconds, const Config& cfg) {
    if (!initialized_ || nodes_.empty() || state_.buffer == VK_NULL_HANDLE) return;

    const uint32_t count = std::clamp<uint32_t>(cfg.particleCount, 1u, kMaxParticles);
    // Grid cell = 2 particle diameters: every pair that can overlap shares a
    // cell or touches its 26 neighbours, and the hash stays sparse.
    const float gridCell = std::max(0.5f, 4.0f * std::max(0.05f, cfg.particleRadius));

    VkCommandBufferBeginInfo bi{};
    (void)bi;

    if (shapeDirty_) {
        // Host pattern -> device state (radius/mass only; positions stay
        // untouched so a shape change never teleports the particles).
        auto* dst = static_cast<uint8_t*>(shapeStage_.mappedData);
        if (dst != nullptr) {
            std::memset(dst, 0, kMaxParticles * kDropletStride);
            for (uint32_t i = 0; i < kMaxParticles; ++i) {
                uint8_t* rec = dst + static_cast<size_t>(i) * kDropletStride;
                std::memcpy(rec + 12, &shapeRadius_, sizeof(float));
                std::memcpy(rec + 28, &shapeMass_, sizeof(float));
            }
        }
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size = kMaxParticles * kDropletStride;
        vkCmdCopyBuffer(cmd, shapeStage_.buffer, state_.buffer, 1, &region);
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
        shapeDirty_ = false;
    }

    if (nodesDirty_) {
        // Host-coherent node tables -> compute reads (the region builder wrote
        // them through the mapped pointer on this thread).
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
        nodesDirty_ = false;
    }

    auto pushAndDispatch = [&](uint32_t pass, uint32_t groups, float dt, const glm::vec4& spawnCenterRadius,
                               const glm::vec4& regionMin, const glm::vec4& regionMax) {
        DropletSimPC pc{};
        pc.spawnCenterRadius = spawnCenterRadius;
        pc.regionMin = regionMin;
        pc.regionMax = regionMax;
        pc.physics = glm::vec4(cfg.gravity, cfg.restitution, cfg.friction, cfg.damping);
        pc.misc = glm::vec4(dt, static_cast<float>(count),
                            glm::uintBitsToFloat(resetSeed_), static_cast<float>(pass));
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);
    };

    const glm::vec4 spawnCenterRadius(cfg.spawnCenter, cfg.spawnRadius);
    const glm::vec4 regionMin(cfg.regionMin, cfg.waterLevel);
    const glm::vec4 regionMax(cfg.regionMax, gridCell);
    const uint32_t particleGroups = groupsFor(count, kLocalSize);
    const uint32_t bucketGroups = groupsFor(kGridBuckets, kLocalSize);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &descSet_, 0, nullptr);

    if (resetPending_) {
        // Fresh counters + seed every particle inside the spawn sphere.
        vkCmdFillBuffer(cmd, gridExtra_.buffer, 0, 4 * sizeof(uint32_t), 0u);
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
        pushAndDispatch(kPassSpawn, particleGroups, 0.0f, spawnCenterRadius, regionMin, regionMax);
        computeBarrier(cmd);
        resetPending_ = false;
    }

    const int substeps = std::clamp(cfg.substeps, 1, 8);
    const int iterations = std::clamp(cfg.solverIterations, 1, 8);
    for (int s = 0; s < substeps; ++s) {
        pushAndDispatch(kPassIntegrate, particleGroups, substepSeconds, spawnCenterRadius, regionMin, regionMax);
        computeBarrier(cmd);
        pushAndDispatch(kPassGridClear, bucketGroups, substepSeconds, spawnCenterRadius, regionMin, regionMax);
        computeBarrier(cmd);
        pushAndDispatch(kPassGridScatter, particleGroups, substepSeconds, spawnCenterRadius, regionMin, regionMax);
        computeBarrier(cmd);
        for (int it = 0; it < iterations; ++it) {
            pushAndDispatch(kPassContacts, particleGroups, substepSeconds, spawnCenterRadius, regionMin, regionMax);
            computeBarrier(cmd);
        }
        pushAndDispatch(kPassFinalize, particleGroups, substepSeconds, spawnCenterRadius, regionMin, regionMax);
        computeBarrier(cmd);
    }

    // The droplet draw reads the state buffer as a storage buffer in the vertex
    // and fragment stages: make the compute writes visible to them.
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// Interpolated SDF at `p` from the CPU copy of the node table: same descent as
// the shader (finest node containing the point, trilinear corners, INFINITY
// corners skipped). Returns a large positive value when the point is outside
// the table (no collision there).
float DropletSim::terrainHeightAt(float x, float z) const {
    if (nodes_.empty()) return 0.0f;

    auto sample = [&](const glm::vec3& p, float& cellSize) -> float {
        cellSize = 1.0f;
        uint32_t id = 0; // entry 0 = world root
        for (int guard = 0; guard < 64; ++guard) {
            const OctreeGpuNode& n = nodes_[id];
            const float half = n.cubeSize * 0.5f;
            const int child = (p.x >= n.cubeMin.x + half ? 4 : 0) +
                              (p.y >= n.cubeMin.y + half ? 2 : 0) +
                              (p.z >= n.cubeMin.z + half ? 1 : 0);
            const int32_t next = n.children[child];
            if (next < 0 || (n.flags & 1) != 0) break; // absent child or leaf
            id = static_cast<uint32_t>(next);
        }
        const OctreeGpuNode& n = nodes_[id];
        cellSize = n.cubeSize;
        if (n.type == 0) return 1e30f; // empty node: no surface data
        const glm::vec3 local = (p - n.cubeMin) / std::max(1e-6f, n.cubeSize);
        const glm::vec3 t = glm::clamp(local, 0.0f, 1.0f);
        float acc = 0.0f;
        float wsum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const float v = n.sdf[i];
            if (!std::isfinite(v) || std::fabs(v) > 1e30f) continue;
            const float wx = (i & 4) ? t.x : (1.0f - t.x);
            const float wy = (i & 2) ? t.y : (1.0f - t.y);
            const float wz = (i & 1) ? t.z : (1.0f - t.z);
            const float w = wx * wy * wz;
            acc += v * w;
            wsum += w;
        }
        if (wsum <= 1e-6f) return 1e30f;
        return acc / wsum;
    };

    // March DOWNWARD from the top of the region: the first solid sample is the
    // VISIBLE (top) surface at this XZ. Marching upward would stop at the first
    // deeper crossing (an overhang, a cave roof, or a node without data), which
    // put the spawn below the visible ground.
    const float yTop = regionMax_.y - 1.0f;
    const float yBottom = regionMin_.y + 1.0f;
    if (yTop <= yBottom) return yBottom;
    const float step = std::max(1.0f, (yTop - yBottom) / 2048.0f);
    float cellSize = 1.0f;
    for (float y = yTop; y >= yBottom; y -= step) {
        const float d = sample(glm::vec3(x, y, z), cellSize);
        if (!std::isfinite(d) || std::fabs(d) > 1e29f) continue; // no data here
        if (d < 0.0f) {
            // Inside the solid: the surface is between this sample and the one
            // above it (linear interpolation on the two distances).
            const float dAbove = sample(glm::vec3(x, y + step, z), cellSize);
            if (std::isfinite(dAbove) && std::fabs(dAbove) < 1e29f && dAbove >= 0.0f) {
                const float denom = std::max(1e-6f, dAbove - d);
                const float t = (0.0f - dAbove) / denom;
                return (y + step) + t * (-step);
            }
            return y;
        }
    }
    return yBottom;
}

DropletSim::Probe DropletSim::probeParticles() {
    Probe out{};
    if (!initialized_ || app_ == nullptr || state_.buffer == VK_NULL_HANDLE) return out;
    VkBufferCopy region{};
    region.size = kMaxParticles * kDropletStride;
    app_->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdCopyBuffer(cmd, state_.buffer, probeStage_.buffer, 1, &region);
    });
    const auto* bytes = static_cast<const uint8_t*>(probeStage_.mappedData);
    if (bytes == nullptr) return out;

    double sumY = 0.0;
    uint32_t active = 0;
    out.minY = 1e30f;
    out.maxY = -1e30f;
    for (uint32_t i = 0; i < kMaxParticles; ++i) {
        const uint8_t* rec = bytes + static_cast<size_t>(i) * kDropletStride;
        glm::vec3 pos;
        float radius = 0.0f;
        glm::vec3 vel;
        uint32_t flags = 0;
        std::memcpy(&pos, rec, sizeof(glm::vec3));
        std::memcpy(&radius, rec + 12, sizeof(float));
        std::memcpy(&vel, rec + 16, sizeof(glm::vec3));
        std::memcpy(&flags, rec + 44, sizeof(uint32_t));
        if ((flags & 1u) == 0u) continue;
        ++active;
        sumY += pos.y;
        out.minY = std::min(out.minY, pos.y);
        out.maxY = std::max(out.maxY, pos.y);
        if (out.probed < 8u) {
            out.positions[out.probed] = glm::vec4(pos, radius);
            out.velocities[out.probed] = glm::vec4(vel, glm::length(vel));
            ++out.probed;
        }
    }
    out.activeCount = active;
    out.meanY = (active > 0u) ? static_cast<float>(sumY / active) : 0.0f;
    if (active == 0u) {
        out.minY = 0.0f;
        out.maxY = 0.0f;
    }
    return out;
}

DropletSim::Stats DropletSim::readStats() {
    Stats out{};
    if (!initialized_ || app_ == nullptr || gridExtra_.buffer == VK_NULL_HANDLE) return out;
    VkBufferCopy region{};
    region.size = 4 * sizeof(uint32_t);
    app_->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdCopyBuffer(cmd, gridExtra_.buffer, statsRead_.buffer, 1, &region);
    });
    if (const auto* src = static_cast<const uint32_t*>(statsRead_.mappedData)) {
        out.gridOverflow = src[0];
        out.pairContacts = src[1];
        out.terrainAndEscaped = src[2];
        out.maxContactsPerParticle = src[3];
    }
    return out;
}
