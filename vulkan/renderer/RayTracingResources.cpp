// Hybrid RT: stable AABB-proxy (triangle-box) BLAS/TLAS + water RT pipeline.
//
// See RayTracingResources.hpp for the architecture. Key points:
//  - Proxy = triangle boxes (12 tris/box, 4096 slots), NOT AABB geometries, so
//    the hit group is a plain TRIANGLES_HIT_GROUP (no intersection shader).
//  - Face culling disabled on the TLAS instance so camera-in-volume and thin
//    boxes still hit; hit shaders derive the face normal analytically.
//  - BLAS is rebuilt (in place, same addresses) only when the staged proxy set
//    changes, throttled to at most once per 30 frames. TLAS is built once (the
//    instance references the BLAS by stable device address).
//  - All barriers use Synchronization2 with short why-comments (AGENTS.md).

#include "RayTracingResources.hpp"
#include "RendererUtils.hpp"
#include "../VulkanApp.hpp"
#include "../../math/Vertex.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// 8 corners + 36 indices (12 tris) per box. Winding MUST be outward (CCW
// from outside): cull is disabled so all faces hit, but ray queries and the
// closest-hit shader use FrontFace/HitKind to distinguish entry (frontface)
// from exit (backface). Inward winding inverts that test, flipping every
// analytic box normal inward and darkening all RT refraction/reflection
// (ndl == 0 for top faces). Reversed to outward 2026-09-09.
static constexpr uint32_t kVertsPerBox = 8;
static constexpr uint32_t kIndicesPerBox = 36;
static constexpr uint32_t kTrisPerBox = 12;
static constexpr uint32_t kBoxIndices[kIndicesPerBox] = {
    0,2,1, 0,3,2, // -Z (outward -Z)
    4,5,6, 4,6,7, // +Z (outward +Z)
    0,5,4, 0,1,5, // -Y (outward -Y)
    3,6,2, 3,7,6, // +Y (outward +Y)
    0,7,3, 0,4,7, // -X (outward -X)
    1,6,5, 1,2,6, // +X (outward +X)
};

void RayTracingResources::init(VulkanApp* app, uint32_t width, uint32_t height) {
    app_ = app;
    supported_ = app && app->rayTracingEnabled() &&
        app->fpCreateAccelerationStructureKHR && app->fpCmdBuildAccelerationStructuresKHR;
    if (!supported_) {
        std::cerr << "[HybridRT] RT unsupported — raster + CSM fallback (sky for misses)\n";
        return;
    }
    // Scratch-address alignment comes from the physical device (NOT from VMA
    // buffer placement): 256 on AMD RADV, 8 on llvmpipe. Must be known before
    // any scratch buffer is sized (slack is over-allocated for manual aligning).
    scratchAlign_ = app->accelProps.minAccelerationStructureScratchOffsetAlignment;
    if (scratchAlign_ == 0) scratchAlign_ = 256;
    printf("[HybridRT] scratch alignment: %llu bytes\n", (unsigned long long)scratchAlign_);
    try {
        createProxyBuffers(app);
        createAccelStructures(app);
        createOutputImages(app, width, height);
        createRTDescriptors(app);
        createRTPipeline(app); // graceful: pipelineReady_=false when shaders missing
    } catch (const std::exception& e) {
        std::cerr << "[HybridRT] init failed (" << e.what() << ") — RT disabled, raster fallback\n";
        supported_ = false;
        pipelineReady_ = false;
    }
    printf("[HybridRT] init: supported=%d pipeline=%d maxProxies=%u out=%ux%u\n",
        (int)supported_, (int)pipelineReady_, kMaxProxies, outWidth_, outHeight_);
}

void RayTracingResources::cleanup(VulkanApp* app) {
    if (!app) app = app_;
    destroyRTPipeline(app);
    destroyOutputImages(app);
    if (app && app->fpDestroyAccelerationStructureKHR) {
        if (blas_ != VK_NULL_HANDLE) { app->fpDestroyAccelerationStructureKHR(app->getDevice(), blas_, nullptr); blas_ = VK_NULL_HANDLE; }
        if (blasWater_ != VK_NULL_HANDLE) { app->fpDestroyAccelerationStructureKHR(app->getDevice(), blasWater_, nullptr); blasWater_ = VK_NULL_HANDLE; }
        if (tlas_ != VK_NULL_HANDLE) { app->fpDestroyAccelerationStructureKHR(app->getDevice(), tlas_, nullptr); tlas_ = VK_NULL_HANDLE; }
    }
    if (app) {
        if (sceneBlas_ != VK_NULL_HANDLE) {
            app->fpDestroyAccelerationStructureKHR(app->getDevice(), sceneBlas_, nullptr);
            sceneBlas_ = VK_NULL_HANDLE;
        }
        sceneBlasAddress_ = 0;
        if (blasBuffer_.buffer) app->destroyBuffer(blasBuffer_);
        if (blasWaterBuffer_.buffer) app->destroyBuffer(blasWaterBuffer_);
        if (sceneBlasBuffer_.buffer) app->destroyBuffer(sceneBlasBuffer_);
        if (tlasBuffer_.buffer) app->destroyBuffer(tlasBuffer_);
        if (blasScratch_.buffer) app->destroyBuffer(blasScratch_);
        if (waterScratch_.buffer) app->destroyBuffer(waterScratch_);
        if (sceneScratch_.buffer) app->destroyBuffer(sceneScratch_);
        if (tlasScratch_.buffer) app->destroyBuffer(tlasScratch_);
        if (aabbBuffer_.buffer) app->destroyBuffer(aabbBuffer_);
        if (metaBuffer_.buffer) app->destroyBuffer(metaBuffer_);
        if (scenePrimBaseBuffer_.buffer) app->destroyBuffer(scenePrimBaseBuffer_);
        if (sceneGeomInfoBuffer_.buffer) app->destroyBuffer(sceneGeomInfoBuffer_);
        if (sceneMetaBuffer_.buffer) app->destroyBuffer(sceneMetaBuffer_);
        if (tlasInstanceBuffer_.buffer) app->destroyBuffer(tlasInstanceBuffer_);
        for (auto& pb : paramsBuffers_)
            if (pb.buffer) app->destroyBuffer(pb);
        if (sbtBuffer_.buffer) app->destroyBuffer(sbtBuffer_);
        if (rtSetPool_ != VK_NULL_HANDLE) {
            app->resources.removeDescriptorPool(rtSetPool_);
            vkDestroyDescriptorPool(app->getDevice(), rtSetPool_, nullptr);
            rtSetPool_ = VK_NULL_HANDLE;
        }
        if (rtSetLayout_ != VK_NULL_HANDLE) {
            app->resources.removeDescriptorSetLayout(rtSetLayout_);
            vkDestroyDescriptorSetLayout(app->getDevice(), rtSetLayout_, nullptr);
            rtSetLayout_ = VK_NULL_HANDLE;
        }
        if (linearSampler_ != VK_NULL_HANDLE) {
            app->resources.removeSampler(linearSampler_);
            vkDestroySampler(app->getDevice(), linearSampler_, nullptr);
            linearSampler_ = VK_NULL_HANDLE;
        }
        for (auto& s : rtSets_) s = VK_NULL_HANDLE;
    }
    supported_ = false;
    pipelineReady_ = false;
}

void RayTracingResources::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!supported_) return;
    destroyOutputImages(app);
    createOutputImages(app, width, height);
    writeRTSet(app); // re-point storage-image bindings at the new views
}

// ── RT descriptors: dedicated per-slot sets ─────────────────────────────
// Layout (ray-tracing stages only, never mixed into raster pipeline layouts):
//   0 = TLAS (ACCELERATION_STRUCTURE, RAYGEN)
//   1 = reflection output (STORAGE_IMAGE, RAYGEN)
//   2 = refraction+thickness output (STORAGE_IMAGE, RAYGEN)
//   3 = params UBO (UNIFORM, RAYGEN|MISS|CLOSEST_HIT)
//   4 = proxy metadata (STORAGE, RAYGEN|CLOSEST_HIT)
//   5 = water depth, per-slot view (COMBINED_SAMPLER, RAYGEN)
//   6 = sky equirect, per-slot view (COMBINED_SAMPLER, RAYGEN|MISS)
void RayTracingResources::createRTDescriptors(VulkanApp* app) {
    VkDevice device = app->getDevice();
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    auto bind = [&](uint32_t i, VkDescriptorType t, VkShaderStageFlags stages) {
        bindings[i].binding = i;
        bindings[i].descriptorType = t;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = stages;
    };
    const VkShaderStageFlags rayStages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bind(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    bind(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    bind(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    bind(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, rayStages);
    bind(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    bind(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    bind(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = uint32_t(bindings.size());
    li.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &rtSetLayout_) != VK_SUCCESS)
        throw std::runtime_error("HybridRT: failed to create RT descriptor set layout");
    app->registerDescriptorSetLayout(rtSetLayout_, "HybridRT: rtSetLayout");

    std::array<VkDescriptorPoolSize, 5> poolSizes{{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12},
    }};
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 3;
    pi.poolSizeCount = uint32_t(poolSizes.size());
    pi.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(device, &pi, nullptr, &rtSetPool_) != VK_SUCCESS)
        throw std::runtime_error("HybridRT: failed to create RT descriptor pool");
    app->resources.addDescriptorPool(rtSetPool_, "HybridRT: rtSetPool");
    std::array<VkDescriptorSetLayout, 3> layouts{rtSetLayout_, rtSetLayout_, rtSetLayout_};
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = rtSetPool_;
    ai.descriptorSetCount = 3;
    ai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &ai, rtSets_) != VK_SUCCESS)
        throw std::runtime_error("HybridRT: failed to allocate RT descriptor sets");
    for (auto s : rtSets_) app->resources.addDescriptorSet(s, "HybridRT: rtSet");
    writeRTSet(app);
}

void RayTracingResources::writeRTSet(VulkanApp* app) {
    if (rtSets_[0] == VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();
    // TLAS may be VK_NULL_HANDLE before the first buildIfNeeded() — the
    // descriptor write is still recorded (validation allows null AS with
    // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND? No — instead skip TLAS writes
    // until the first build completes; dispatches are no-ops until then
    // because isPipelineReady() also requires a completed build).
    const bool haveTlas = (tlas_ != VK_NULL_HANDLE);
    for (int slot = 0; slot < 3; ++slot) {
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(7);
        VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
        if (haveTlas) {
            asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asInfo.accelerationStructureCount = 1;
            asInfo.pAccelerationStructures = &tlas_;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.pNext = &asInfo;
            w.dstSet = rtSets_[slot];
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            writes.push_back(w);
        }
        VkDescriptorImageInfo reflectImg{};
        reflectImg.sampler = VK_NULL_HANDLE;
        reflectImg.imageView = reflectView_;
        reflectImg.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo refractImg{};
        refractImg.sampler = VK_NULL_HANDLE;
        refractImg.imageView = refractView_;
        refractImg.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo paramsInfo{paramsBuffers_[slot].buffer, 0, sizeof(RayTracingParams)};
        VkDescriptorBufferInfo metaInfo{metaBuffer_.buffer, 0, sizeof(RTProxyMeta) * kMaxProxies};
        // NOTE: water-depth (5) + sky (6) views are per-slot scene targets owned
        // by WaterRenderer/SkyRenderer; setSceneViews() writes them (init +
        // resize). Here we write the RT-owned bindings only (0-4).
        std::vector<VkDescriptorImageInfo> imgInfos;
        imgInfos.reserve(2);
        imgInfos.push_back(reflectImg);
        imgInfos.push_back(refractImg);
        std::vector<VkDescriptorBufferInfo> bufInfos;
        bufInfos.reserve(2);
        bufInfos.push_back(paramsInfo);
        bufInfos.push_back(metaInfo);
        for (uint32_t b = 1; b <= 2; ++b) {
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = rtSets_[slot];
            w.dstBinding = b;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo = &imgInfos[b - 1];
            writes.push_back(w);
        }
        for (uint32_t b = 3; b <= 4; ++b) {
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = rtSets_[slot];
            w.dstBinding = b;
            w.descriptorCount = 1;
            w.descriptorType = (b == 3) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &bufInfos[b - 3];
            writes.push_back(w);
        }
        if (!writes.empty())
            vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    }
    // Keep the TLAS binding fresh after the first build (writeRTSet runs at
    // init before any build, so haveTlas==false then; buildIfNeeded() calls
    // this again once via ensureTlasWritten_ — see recordBuild tail).
}

void RayTracingResources::setSceneViews(VulkanApp* app, const VkImageView waterDepthViews[3],
                                        const VkImageView skyViews[3]) {
    if (!supported_ || rtSets_[0] == VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();
    // Per-slot sampled views (stable between resizes). Samplers: depth uses
    // nearest (exact hits, no filtering bleed); sky uses the RT linear sampler.
    for (int slot = 0; slot < 3; ++slot) {
        if (waterDepthViews[slot] == VK_NULL_HANDLE || skyViews[slot] == VK_NULL_HANDLE) continue;
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = linearSampler_;
        depthInfo.imageView = waterDepthViews[slot];
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo skyInfo{};
        skyInfo.sampler = linearSampler_;
        skyInfo.imageView = skyViews[slot];
        skyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = rtSets_[slot];
        writes[0].dstBinding = 5;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &depthInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = rtSets_[slot];
        writes[1].dstBinding = 6;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &skyInfo;
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
}

void RayTracingResources::createProxyBuffers(VulkanApp* app) {
    const VkDeviceSize vertBytes = VkDeviceSize(kMaxProxies) * kVertsPerBox * sizeof(float) * 3;
    const VkDeviceSize idxBytes = VkDeviceSize(kMaxProxies) * kIndicesPerBox * sizeof(uint32_t);
    // Box soup: host-visible staging (CPU writes on dirty) + device addresses
    // for BLAS builds. Coherent so memcpy is enough; visibility ordered by the
    // HOST->ACCEL_BUILD barrier recorded in recordBuild().
    VkBufferUsageFlags boxUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // +16 slack: the vertex base is aligned up to 16 (triangle vertex/index
    // input requirement) since VMA placement alone does not guarantee it.
    aabbBuffer_ = app->createBuffer(vertBytes + idxBytes + 16, boxUsage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Static index patterns, written once — vertex positions are rewritten on
    // every proxy update. Each BLAS partition uses indices relative to its own
    // vertex base (solid partition starts at vertex 0, water partition at its
    // own base), so the two BLAS geometries stay self-contained.
    {
        auto* rawBase = static_cast<char*>(aabbBuffer_.mappedData);
        VkBufferDeviceAddressInfo addrQ{};
        addrQ.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrQ.buffer = aabbBuffer_.buffer;
        const VkDeviceAddress rawAddr = vkGetBufferDeviceAddress(app->getDevice(), &addrQ);
        if (rawAddr == 0) throw std::runtime_error("box buffer device address is 0");
        boxBaseDelta_ = alignUpAddr(rawAddr, 16) - rawAddr;
        auto* base = rawBase + boxBaseDelta_;
        auto* idx = reinterpret_cast<uint32_t*>(base + vertBytes);
        // Solid partition: slots [0, kMaxSolidProxies), absolute indices.
        for (uint32_t s = 0; s < kMaxSolidProxies; ++s)
            for (uint32_t i = 0; i < kIndicesPerBox; ++i)
                idx[s * kIndicesPerBox + i] = s * kVertsPerBox + kBoxIndices[i];
        // Water partition: global slots [kWaterProxyStart, kMaxProxies), but
        // indices relative to the water vertex base (see kWaterVertBase).
        for (uint32_t w = 0; w < kMaxWaterProxies; ++w)
            for (uint32_t i = 0; i < kIndicesPerBox; ++i)
                idx[(kWaterProxyStart + w) * kIndicesPerBox + i] =
                    w * kVertsPerBox + kBoxIndices[i];
        // Degenerate verts initially (zero-area, never hit) for all slots.
        memset(base, 0, static_cast<size_t>(vertBytes));
    }
    VkBufferDeviceAddressInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    ai.buffer = aabbBuffer_.buffer;
    const VkDeviceAddress boxAddr = vkGetBufferDeviceAddress(app->getDevice(), &ai);
    aabbAddress_ = alignUpAddr(boxAddr, 16); // verts at +delta, indices at +delta+vertBytes
    if (aabbAddress_ == 0) throw std::runtime_error("box buffer device address is 0");

    metaBuffer_ = app->createBuffer(sizeof(RTProxyMeta) * kMaxProxies,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    tlasInstanceBuffer_ = app->createBuffer(3 * sizeof(VkAccelerationStructureInstanceKHR) + 16,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ai.buffer = tlasInstanceBuffer_.buffer;
    {
        const VkDeviceAddress rawInst = vkGetBufferDeviceAddress(app->getDevice(), &ai);
        if (rawInst == 0) throw std::runtime_error("instance buffer device address is 0");
        instanceDelta_ = alignUpAddr(rawInst, 16) - rawInst; // instance input needs 16
        tlasInstanceAddress_ = rawInst + instanceDelta_;
    }

    for (uint32_t f = 0; f < kParamFrames; ++f) {
        paramsBuffers_[f] = app->createBuffer(sizeof(RayTracingParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        RayTracingParams defaults{};
        if (paramsBuffers_[f].mappedData)
            memcpy(paramsBuffers_[f].mappedData, &defaults, sizeof(defaults));
    }
}

void RayTracingResources::createAccelStructures(VulkanApp* app) {
    VkDevice device = app->getDevice();
    // Partition byte layout inside the shared box buffer (verts, then indices):
    const VkDeviceSize kVertStride = sizeof(float) * 3;
    const VkDeviceSize solidVertBytes = VkDeviceSize(kMaxSolidProxies) * kVertsPerBox * kVertStride;
    const VkDeviceSize solidIdxBytes = VkDeviceSize(kMaxSolidProxies) * kIndicesPerBox * sizeof(uint32_t);
    const VkDeviceSize waterVertBase = solidVertBytes;
    const uint32_t solidTris = kMaxSolidProxies * kTrisPerBox;
    const uint32_t waterTris = kMaxWaterProxies * kTrisPerBox;

    auto makeTrisGeom = [&](VkDeviceAddress vertAddr, uint32_t maxVerts,
                            VkDeviceAddress idxAddr) {
        VkAccelerationStructureGeometryTrianglesDataKHR tris{};
        tris.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tris.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tris.vertexData.deviceAddress = vertAddr;
        tris.vertexStride = kVertStride;
        tris.maxVertex = maxVerts;
        tris.indexType = VK_INDEX_TYPE_UINT32;
        tris.indexData.deviceAddress = idxAddr;
        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tris;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR; // closest-hit only, no any-hit (perf §16)
        return geom;
    };
    auto querySizes = [&](const VkAccelerationStructureGeometryKHR& geom, uint32_t prims) {
        VkAccelerationStructureBuildGeometryInfoKHR bi{};
        bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &geom;
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        app->fpGetAccelerationStructureBuildSizesKHR(device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sizes);
        return sizes;
    };
    auto alignedScratch = [&](Buffer& dst, VkDeviceSize need, const char* what) {
        dst = app->createBuffer(std::max(need, VkDeviceSize(1)) + scratchAlign_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        // Scratch addresses MUST satisfy minAccelerationStructureScratchOffset-
        // Alignment (VUID-03710); VMA placement alone does not guarantee it, so
        // align up inside the over-allocated buffer.
        VkBufferDeviceAddressInfo addrQ{};
        addrQ.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrQ.buffer = dst.buffer;
        const VkDeviceAddress rawScratch = vkGetBufferDeviceAddress(device, &addrQ);
        if (rawScratch == 0) throw std::runtime_error(what);
        return alignUpAddr(rawScratch, scratchAlign_);
    };

    VkBufferUsageFlags asUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // Solid BLAS (partition 0).
    VkAccelerationStructureGeometryKHR solidGeom = makeTrisGeom(
        aabbAddress_, kMaxSolidProxies * kVertsPerBox,
        aabbAddress_ + VkDeviceSize(kMaxProxies) * kVertsPerBox * kVertStride);
    VkAccelerationStructureBuildSizesInfoKHR sizes = querySizes(solidGeom, solidTris);
    blasBuffer_ = app->createBuffer(sizes.accelerationStructureSize, asUsage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    blasScratchAligned_ = alignedScratch(blasScratch_, sizes.buildScratchSize, "BLAS scratch device address is 0");

    VkAccelerationStructureCreateInfoKHR blasCI{};
    blasCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    blasCI.buffer = blasBuffer_.buffer;
    blasCI.size = sizes.accelerationStructureSize;
    blasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (app->fpCreateAccelerationStructureKHR(device, &blasCI, nullptr, &blas_) != VK_SUCCESS)
        throw std::runtime_error("vkCreateAccelerationStructureKHR (BLAS) failed");
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blas_;
    blasAddress_ = app->fpGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

    // Water BLAS (partition 1: self-contained vertex/index ranges).
    VkAccelerationStructureGeometryKHR waterGeom = makeTrisGeom(
        aabbAddress_ + waterVertBase, kMaxWaterProxies * kVertsPerBox,
        aabbAddress_ + VkDeviceSize(kMaxProxies) * kVertsPerBox * kVertStride + solidIdxBytes);
    VkAccelerationStructureBuildSizesInfoKHR waterSizes = querySizes(waterGeom, waterTris);
    blasWaterBuffer_ = app->createBuffer(waterSizes.accelerationStructureSize, asUsage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VkDeviceAddress waterScratchAligned = alignedScratch(
        waterScratch_, waterSizes.buildScratchSize, "Water BLAS scratch device address is 0");

    VkAccelerationStructureCreateInfoKHR blasWaterCI{};
    blasWaterCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    blasWaterCI.buffer = blasWaterBuffer_.buffer;
    blasWaterCI.size = waterSizes.accelerationStructureSize;
    blasWaterCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (app->fpCreateAccelerationStructureKHR(device, &blasWaterCI, nullptr, &blasWater_) != VK_SUCCESS)
        throw std::runtime_error("vkCreateAccelerationStructureKHR (water BLAS) failed");
    addrInfo.accelerationStructure = blasWater_;
    blasWaterAddress_ = app->fpGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);
    waterScratchAligned_ = waterScratchAligned;

    // TLAS: three instances (solids mask 0x01, water mask 0x02, real scene
    // triangles mask 0x04), identity, cull disabled. Written at the aligned
    // offset (see instanceDelta_). Instance 2 (scene) is only added to the
    // build once its BLAS exists; the TLAS is sized for three regardless.
    auto* inst = reinterpret_cast<VkAccelerationStructureInstanceKHR*>(
        static_cast<char*>(tlasInstanceBuffer_.mappedData) + instanceDelta_);
    memset(inst, 0, 3 * sizeof(*inst));
    inst[0].transform.matrix[0][0] = 1.0f;
    inst[0].transform.matrix[1][1] = 1.0f;
    inst[0].transform.matrix[2][2] = 1.0f;
    inst[0].instanceCustomIndex = 0;
    inst[0].mask = kMaskSolid;
    inst[0].instanceShaderBindingTableRecordOffset = 0;
    inst[0].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst[0].accelerationStructureReference = blasAddress_;
    inst[1] = inst[0];
    inst[1].instanceCustomIndex = 1;
    inst[1].mask = kMaskWater;
    inst[1].accelerationStructureReference = blasWaterAddress_;
    inst[2] = inst[0];
    inst[2].instanceCustomIndex = 2;
    inst[2].mask = kMaskScene;
    inst[2].accelerationStructureReference = 0; // patched on the first scene build

    // Shader lookup buffers for the scene instance (preallocated, no resize):
    // [0] = geometry count, [1..N] = first primitive of each geometry, and one
    // vec4 average albedo per geometry.
    scenePrimBaseBuffer_ = app->createBuffer((kMaxSceneGeoms + 1) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    sceneMetaBuffer_ = app->createBuffer(kMaxSceneGeoms * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (scenePrimBaseBuffer_.mappedData)
        static_cast<uint32_t*>(scenePrimBaseBuffer_.mappedData)[0] = 0; // empty
    sceneGeomInfoBuffer_ = app->createBuffer(kMaxSceneGeoms * sizeof(glm::uvec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    VkAccelerationStructureGeometryKHR tlasGeom{};
    tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.geometry.instances = instances; // device address patched per build
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
    tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuild.geometryCount = 1;
    tlasBuild.pGeometries = &tlasGeom;
    uint32_t maxInstances = 3;
    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
    tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    app->fpGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild, &maxInstances, &tlasSizes);
    tlasBuffer_ = app->createBuffer(tlasSizes.accelerationStructureSize, asUsage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tlasScratch_ = app->createBuffer(std::max(tlasSizes.buildScratchSize, VkDeviceSize(1)) + scratchAlign_,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    {
        VkBufferDeviceAddressInfo addrQ{};
        addrQ.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrQ.buffer = tlasScratch_.buffer;
        const VkDeviceAddress rawScratch = vkGetBufferDeviceAddress(device, &addrQ);
        if (rawScratch == 0) throw std::runtime_error("TLAS scratch device address is 0");
        tlasScratchAligned_ = alignUpAddr(rawScratch, scratchAlign_);
    }
    VkAccelerationStructureCreateInfoKHR tlasCI{};
    tlasCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasCI.buffer = tlasBuffer_.buffer;
    tlasCI.size = tlasSizes.accelerationStructureSize;
    tlasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (app->fpCreateAccelerationStructureKHR(device, &tlasCI, nullptr, &tlas_) != VK_SUCCESS)
        throw std::runtime_error("vkCreateAccelerationStructureKHR (TLAS) failed");

    dirty_ = true; // consider a BLAS+TLAS build on the first buildIfNeeded()
                   // (skipped while the proxy set is still empty — see above)
}

void RayTracingResources::createOutputImages(VulkanApp* app, uint32_t width, uint32_t height) {
    outWidth_ = std::max(8u, uint32_t(float(width) * kOutputScale));
    outHeight_ = std::max(8u, uint32_t(float(height) * kOutputScale));
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    RendererUtils::createImage2DWithVma(app->getDevice(), app, outWidth_, outHeight_,
        VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT,
        "HybridRT: reflect output", reflectImage_, reflectAlloc_, reflectMem_, reflectView_);
    RendererUtils::createImage2DWithVma(app->getDevice(), app, outWidth_, outHeight_,
        VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT,
        "HybridRT: refract output", refractImage_, refractAlloc_, refractMem_, refractView_);
    reflectLayout_ = refractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    // UNDEFINED -> GENERAL initial transition. No app helper serves this pair
    // (transfer/depth-only), so record the barrier directly: fresh images need
    // no execution dependency (TOP_OF_PIPE, no access masks). Sync init path
    // may block; never in the frame path.
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier2 barriers[2]{};
        VkImage imgs[2] = {reflectImage_, refractImage_};
        for (int i = 0; i < 2; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barriers[i].srcAccessMask = 0;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = imgs[i];
            barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    });
    app->setImageLayoutTracked(reflectImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    app->setImageLayoutTracked(refractImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    app->setImageLayoutTracked(reflectImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    app->setImageLayoutTracked(refractImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    reflectLayout_ = refractLayout_ = VK_IMAGE_LAYOUT_GENERAL;
    if (linearSampler_ == VK_NULL_HANDLE)
        linearSampler_ = app->createSamplerLinearClamp("HybridRT: linearSampler");
}

void RayTracingResources::destroyOutputImages(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    if (reflectView_ != VK_NULL_HANDLE) {
        if (app->resources.removeImageView(reflectView_)) vkDestroyImageView(device, reflectView_, nullptr);
        reflectView_ = VK_NULL_HANDLE;
    }
    if (refractView_ != VK_NULL_HANDLE) {
        if (app->resources.removeImageView(refractView_)) vkDestroyImageView(device, refractView_, nullptr);
        refractView_ = VK_NULL_HANDLE;
    }
    if (reflectImage_ != VK_NULL_HANDLE) { app->destroyImageWithVma(reflectImage_, reflectAlloc_, reflectMem_); reflectImage_ = VK_NULL_HANDLE; }
    if (refractImage_ != VK_NULL_HANDLE) { app->destroyImageWithVma(refractImage_, refractAlloc_, refractMem_); refractImage_ = VK_NULL_HANDLE; }
}

void RayTracingResources::setProxies(const std::vector<RTProxyBox>& solids,
                                     const std::vector<RTProxyBox>& waters) {
    if (!supported_) return;
    const uint32_t ns = std::min<uint32_t>(uint32_t(solids.size()), kMaxSolidProxies);
    const uint32_t nw = std::min<uint32_t>(uint32_t(waters.size()), kMaxWaterProxies);
    // Fast path: identical bytes -> no rebuild (avoids BLAS churn when the
    // scene reports no real change).
    auto same = [](const std::vector<RTProxyBox>& a, const std::vector<RTProxyBox>& b, uint32_t n) {
        return n == uint32_t(b.size()) &&
            (n == 0 || memcmp(a.data(), b.data(), n * sizeof(RTProxyBox)) == 0);
    };
    if (ns == activeSolidCount_ && nw == activeWaterCount_ &&
        same(stagedSolids_, solids, ns) && same(stagedWaters_, waters, nw))
        return;
    stagedSolids_.assign(solids.begin(), solids.begin() + ns);
    stagedWaters_.assign(waters.begin(), waters.begin() + nw);
    activeSolidCount_ = ns;
    activeWaterCount_ = nw;
    dirty_ = true;
}

void RayTracingResources::setSceneGeometry(std::vector<SceneTriGeometry> geoms) {
    if (!supported_) return;
    if (geoms.size() > kMaxSceneGeoms) geoms.resize(kMaxSceneGeoms);
    sceneGeoms_ = std::move(geoms);
    sceneBlasDirty_ = true;
    dirty_ = true; // force the throttled build path even if the proxies matched
}

bool RayTracingResources::recordSceneBlas(VulkanApp* app, VkCommandBuffer cmd) {
    if (!sceneBlasDirty_) return false;
    sceneBlasDirty_ = false;
    if (sceneGeoms_.empty()) return false;
    VkDevice device = app->getDevice();
    const uint32_t n = uint32_t(sceneGeoms_.size());

    // CPU-side geometry + range arrays (one geometry per active chunk).
    std::vector<VkAccelerationStructureGeometryKHR> geoms(n);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(n);
    std::vector<uint32_t> primCounts(n);
    uint32_t* primBase = scenePrimBaseBuffer_.mappedData
        ? static_cast<uint32_t*>(scenePrimBaseBuffer_.mappedData) : nullptr;
    glm::uvec4* geomInfo = sceneGeomInfoBuffer_.mappedData
        ? static_cast<glm::uvec4*>(sceneGeomInfoBuffer_.mappedData) : nullptr;
    glm::vec4* meta = sceneMetaBuffer_.mappedData
        ? static_cast<glm::vec4*>(sceneMetaBuffer_.mappedData) : nullptr;
    uint32_t prim = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const SceneTriGeometry& s = sceneGeoms_[i];
        VkAccelerationStructureGeometryTrianglesDataKHR t{};
        t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        t.vertexData.deviceAddress = s.vertexAddress;
        t.vertexStride = sizeof(Vertex);
        t.maxVertex = s.vertexCount > 0 ? s.vertexCount - 1 : 0;
        t.indexType = VK_INDEX_TYPE_UINT32;
        t.indexData.deviceAddress = s.indexAddress;
        VkAccelerationStructureGeometryKHR g{};
        g.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.geometry.triangles = t;
        g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geoms[i] = g;
        const uint32_t pc = s.indexCount / 3;
        ranges[i].primitiveCount = pc;
        ranges[i].primitiveOffset = 0;
        ranges[i].firstVertex = 0;
        ranges[i].transformOffset = 0;
        primCounts[i] = pc;
        if (primBase) primBase[1 + i] = prim;
        if (geomInfo) geomInfo[i] = glm::uvec4(s.baseVertex, s.firstIndex, prim, 0u);
        if (meta) meta[i] = s.albedo;
        prim += pc;
    }
    if (primBase) primBase[0] = n;
    if (prim == 0) return false;

    VkAccelerationStructureBuildGeometryInfoKHR bi{};
    bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = n;
    bi.pGeometries = geoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    app->fpGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, primCounts.data(), &sizes);

    const VkBufferUsageFlags asUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (sceneBlas_ == VK_NULL_HANDLE || sizes.accelerationStructureSize > sceneBlasSize_) {
        // Retire the previous AS/buffer only after all in-flight submissions
        // complete: frames recorded earlier still reference the old TLAS, whose
        // instances point at the old BLAS address. Destroying it immediately
        // left those frames tracing freed memory — visible as corrupted
        // reflection geometry/stipple right after a rebuild.
        if (sceneBlas_ != VK_NULL_HANDLE || sceneBlasBuffer_.buffer) {
            VkAccelerationStructureKHR oldAs = sceneBlas_;
            Buffer oldBuf = sceneBlasBuffer_;
            VulkanApp* cap = app;
            app->deferDestroyUntilAllPending([cap, oldAs, oldBuf]() mutable {
                if (oldAs != VK_NULL_HANDLE)
                    cap->fpDestroyAccelerationStructureKHR(cap->getDevice(), oldAs, nullptr);
                if (oldBuf.buffer) cap->destroyBuffer(oldBuf);
            });
            sceneBlas_ = VK_NULL_HANDLE;
            sceneBlasBuffer_ = {};
        }
        sceneBlasBuffer_ = app->createBuffer(sizes.accelerationStructureSize, asUsage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = sceneBlasBuffer_.buffer;
        ci.size = sizes.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (app->fpCreateAccelerationStructureKHR(device, &ci, nullptr, &sceneBlas_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateAccelerationStructureKHR (scene BLAS) failed");
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = sceneBlas_;
        sceneBlasAddress_ = app->fpGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);
        sceneBlasSize_ = sizes.accelerationStructureSize;
    }
    if (sceneScratch_.buffer == VK_NULL_HANDLE ||
        sizes.buildScratchSize + scratchAlign_ > sceneScratchSize_) {
        // Same deferred retire as the AS buffer: an in-flight build may still
        // be using the old scratch.
        if (sceneScratch_.buffer) {
            Buffer oldScratch = sceneScratch_;
            VulkanApp* cap = app;
            app->deferDestroyUntilAllPending([cap, oldScratch]() mutable {
                if (oldScratch.buffer) cap->destroyBuffer(oldScratch);
            });
            sceneScratch_ = {};
        }
        sceneScratch_ = app->createBuffer(sizes.buildScratchSize + scratchAlign_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        sceneScratchSize_ = sizes.buildScratchSize + scratchAlign_;
    }
    VkBufferDeviceAddressInfo scratchAddrQ{};
    scratchAddrQ.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    scratchAddrQ.buffer = sceneScratch_.buffer;
    VkDeviceAddress scratchAddr = alignUpAddr(
        vkGetBufferDeviceAddress(device, &scratchAddrQ), scratchAlign_);

    bi.dstAccelerationStructure = sceneBlas_;
    bi.scratchData.deviceAddress = scratchAddr;
    const VkAccelerationStructureBuildRangeInfoKHR* pr = ranges.data();
    app->fpCmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &pr);
    return true;
}

bool RayTracingResources::buildIfNeeded(VulkanApp* app, VkCommandBuffer cmd) {
    if (!supported_ || !dirty_ || cmd == VK_NULL_HANDLE) return false;
    ++frameCounter_;
    // Never waste the initial build on an empty proxy set: an empty TLAS
    // helps nobody, yet it would flip tlasBuilt_ on, so shaders spend the
    // whole scene-load window tracing empty space (every ray misses, hence
    // visibly "no reflections" right when users look). dirty_ stays set, so
    // the first populated build fires immediately once boxes arrive (the
    // throttle below already exempts it via !lastBuiltValid_).
    const bool haveBoxes = (activeSolidCount_ + activeWaterCount_) > 0;
    if (!haveBoxes && !lastBuiltValid_) return false;
    // Throttle: at most one rebuild per 30 frames — chunk bursts (scene load /
    // brush edits) coalesce into a single build instead of one per publish.
    // Camera moves / LOD switches never mark dirty, so they never rebuild (§6).
    if (frameCounter_ - lastBuildFrame_ < 30 && lastBuiltValid_) return false;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const bool built = recordBuild(app, cmd);
    if (built) {
        lastBuildFrame_ = frameCounter_;
        lastBuiltValid_ = true;
        dirty_ = false;
        ++buildCount_;
        // tlasReady means "TLAS has content": a set emptied later (scene
        // cleared) drops back to the sky/CSM fallback instead of tracing
        // empty space.
        tlasBuilt_ = haveBoxes;
        lastBuildMs_ = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        // Rare (scene changes only, throttled): one line per rebuild.
        printf("[HybridRT] BLAS/TLAS rebuild #%u: %u solid + %u water proxies in %.2f ms (CPU record)\n",
            buildCount_, activeSolidCount_, activeWaterCount_, lastBuildMs_);
        fflush(stdout);
    }
    return built;
}

bool RayTracingResources::recordBuild(VulkanApp* app, VkCommandBuffer cmd) {
    // Defensive: never feed a zero/misaligned device address to an AS build —
    // that is a fatal validation error (VUID-03710). Skip the build instead;
    // shaders fall back to sky/CSM until a later throttled retry succeeds.
    if (blasScratchAligned_ == 0 || (blasScratchAligned_ % scratchAlign_) != 0 ||
        waterScratchAligned_ == 0 || (waterScratchAligned_ % scratchAlign_) != 0 ||
        tlasScratchAligned_ == 0 || (tlasScratchAligned_ % scratchAlign_) != 0 ||
        tlasInstanceAddress_ == 0 || (tlasInstanceAddress_ % 16) != 0 ||
        aabbAddress_ == 0 || (aabbAddress_ % 16) != 0) {
        static int guardLogs = 0;
        if (guardLogs++ < 3) {
            fprintf(stderr,
                "[HybridRT] AS build skipped: bad device address "
                "(blasScratch=0x%llx waterScratch=0x%llx tlasScratch=0x%llx instances=0x%llx boxes=0x%llx align=%llu)\n",
                (unsigned long long)blasScratchAligned_, (unsigned long long)waterScratchAligned_,
                (unsigned long long)tlasScratchAligned_, (unsigned long long)tlasInstanceAddress_,
                (unsigned long long)aabbAddress_, (unsigned long long)scratchAlign_);
        }
        return false;
    }
    // 1. Stage box verts + metadata on the host-visible buffers (coherent memcpy).
    // Solids fill slots [0, activeSolidCount_), water volumes global slots
    // [kWaterProxyStart, kWaterProxyStart + activeWaterCount_); the rest of
    // each partition stays degenerate (zero-area, never hit).
    {
        auto* verts = reinterpret_cast<float*>(
            static_cast<char*>(aabbBuffer_.mappedData) + boxBaseDelta_);
        auto* metas = static_cast<RTProxyMeta*>(metaBuffer_.mappedData);
        // Vertex positions live at GLOBAL slots (water verts physically sit in
        // slots [kWaterProxyStart, kMaxProxies) of the vertex region); only the
        // BLAS geometries view them as separate partitions. Metadata likewise.
        auto stageBox = [&](uint32_t slot, const RTProxyBox& b) {
            float* v = verts + size_t(slot) * kVertsPerBox * 3;
            const float x0 = b.minp.x, y0 = b.minp.y, z0 = b.minp.z;
            const float x1 = b.maxp.x, y1 = b.maxp.y, z1 = b.maxp.z;
            const float c[8][3] = {{x0,y0,z0},{x1,y0,z0},{x1,y1,z0},{x0,y1,z0},
                                   {x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}};
            for (int k = 0; k < 8; ++k) { v[k*3+0] = c[k][0]; v[k*3+1] = c[k][1]; v[k*3+2] = c[k][2]; }
            metas[slot].minAndMatId = glm::vec4(b.minp, b.materialId);
            metas[slot].maxAndFlags = glm::vec4(b.maxp, b.flags);
            metas[slot].albedoRough = glm::vec4(b.albedo, b.roughness);
            // Horizontal footprint lets shaders distrust coarse boxes: a box
            // tens of meters wide cannot resolve shallow-water detail, so
            // water refraction/reflection treat such hits as deep/sky.
            const float footprint = std::max(b.maxp.x - b.minp.x, b.maxp.z - b.minp.z);
            metas[slot].extra = glm::vec4(std::max(footprint, 0.0f), 0.0f, 0.0f, 0.0f);
        };
        auto zeroBox = [&](uint32_t slot) {
            float* v = verts + size_t(slot) * kVertsPerBox * 3;
            for (int k = 0; k < 24; ++k) v[k] = 0.0f; // degenerate (never hit)
            metas[slot] = RTProxyMeta{};
        };
        for (uint32_t s = 0; s < kMaxSolidProxies; ++s) {
            if (s < activeSolidCount_) stageBox(s, stagedSolids_[s]);
            else zeroBox(s);
        }
        for (uint32_t w = 0; w < kMaxWaterProxies; ++w) {
            const uint32_t slot = kWaterProxyStart + w;
            if (w < activeWaterCount_) stageBox(slot, stagedWaters_[w]);
            else zeroBox(slot);
        }
    }
    const VkDeviceSize vertBytes = VkDeviceSize(kMaxProxies) * kVertsPerBox * sizeof(float) * 3;
    const VkDeviceSize idxBytes = VkDeviceSize(kMaxProxies) * kIndicesPerBox * sizeof(uint32_t);

    // 2. Host writes -> BLAS build inputs (vertex/index/meta/instance buffers).
    {
        VkBufferMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
            | VK_ACCESS_2_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        // One barrier per buffer (same stage/access; batched in one call below
        // via three entries — written out explicitly for clarity).
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        VkBufferMemoryBarrier2 barriers[6]{barrier, barrier, barrier, barrier, barrier, barrier};
        // Ranges cover the alignment slack (writes land at +delta, delta < 16)
        // and all TLAS instances.
        barriers[0].buffer = aabbBuffer_.buffer; barriers[0].offset = 0; barriers[0].size = vertBytes + idxBytes + 16;
        barriers[1].buffer = metaBuffer_.buffer; barriers[1].offset = 0; barriers[1].size = sizeof(RTProxyMeta) * kMaxProxies;
        barriers[2].buffer = tlasInstanceBuffer_.buffer; barriers[2].offset = 0; barriers[2].size = 3 * sizeof(VkAccelerationStructureInstanceKHR) + 16;
        // Real-scene lookup buffers (host-written prim bases + per-chunk albedo)
        // are read by the fragment shader after the build.
        barriers[3].buffer = scenePrimBaseBuffer_.buffer; barriers[3].offset = 0; barriers[3].size = VK_WHOLE_SIZE;
        barriers[4].buffer = sceneMetaBuffer_.buffer; barriers[4].offset = 0; barriers[4].size = VK_WHOLE_SIZE;
        barriers[5].buffer = sceneGeomInfoBuffer_.buffer; barriers[5].offset = 0; barriers[5].size = VK_WHOLE_SIZE;
        dep.bufferMemoryBarrierCount = 6;
        dep.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 3. BLAS builds (one per layer; inactive slots degenerate). Each geometry
    // views its own partition; both builds share the stable buffer addresses.
    const VkDeviceSize kVertStride = sizeof(float) * 3;
    auto makeBuildTris = [&](VkDeviceAddress vertAddr, uint32_t maxVerts,
                             VkDeviceAddress idxAddr) {
        VkAccelerationStructureGeometryTrianglesDataKHR t{};
        t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        t.vertexData.deviceAddress = vertAddr;
        t.vertexStride = kVertStride;
        t.maxVertex = maxVerts;
        t.indexType = VK_INDEX_TYPE_UINT32;
        t.indexData.deviceAddress = idxAddr;
        return t;
    };
    auto buildBlas = [&](VkAccelerationStructureKHR dst, VkDeviceAddress scratch,
                         const VkAccelerationStructureGeometryKHR& g, uint32_t prims) {
        VkAccelerationStructureBuildGeometryInfoKHR bi{};
        bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.dstAccelerationStructure = dst;
        bi.geometryCount = 1;
        bi.pGeometries = &g;
        bi.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR r{};
        r.primitiveCount = prims;
        r.primitiveOffset = 0;
        r.firstVertex = 0;
        r.transformOffset = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* pr = &r;
        app->fpCmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &pr);
    };
    auto makeGeom = [&](const VkAccelerationStructureGeometryTrianglesDataKHR& t) {
        VkAccelerationStructureGeometryKHR g{};
        g.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.geometry.triangles = t;
        g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        return g;
    };
    const VkDeviceSize waterVertBase =
        VkDeviceSize(kMaxSolidProxies) * kVertsPerBox * kVertStride;
    const VkDeviceSize waterIdxBase =
        vertBytes + VkDeviceSize(kMaxSolidProxies) * kIndicesPerBox * sizeof(uint32_t);
    {
        VkAccelerationStructureGeometryTrianglesDataKHR tris = makeBuildTris(
            aabbAddress_, kMaxSolidProxies * kVertsPerBox, aabbAddress_ + vertBytes);
        VkAccelerationStructureGeometryKHR geom = makeGeom(tris);
        buildBlas(blas_, blasScratchAligned_, geom, kMaxSolidProxies * kTrisPerBox);
    }
    {
        VkAccelerationStructureGeometryTrianglesDataKHR tris = makeBuildTris(
            aabbAddress_ + waterVertBase, kMaxWaterProxies * kVertsPerBox,
            aabbAddress_ + waterIdxBase);
        VkAccelerationStructureGeometryKHR geom = makeGeom(tris);
        buildBlas(blasWater_, waterScratchAligned_, geom, kMaxWaterProxies * kTrisPerBox);
    }
    // 3b. Real scene-geometry BLAS (exact chunk triangles for reflection rays).
    // Built together with the proxies when chunks change.
    const bool sceneBuilt = recordSceneBlas(app, cmd);
    if (sceneBuilt) {
        static uint32_t sceneBuildCount = 0;
        printf("[HybridRT] scene BLAS rebuild #%u: %zu chunks, %.2f MB\n",
            ++sceneBuildCount, sceneGeoms_.size(),
            double(sceneBlasSize_) / (1024.0 * 1024.0));
        fflush(stdout);
    }

    // 4. BLAS writes -> TLAS read (all BLASes keep stable device addresses, so
    // the TLAS stays valid; re-recording the TLAS build is cheap (≤3 instances)
    // and keeps validation simple).
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    // Refresh the BLAS references (stable addresses, cheap) then build TLAS.
    // The real-scene instance is only added once its BLAS exists, so the TLAS
    // never references a null acceleration structure.
    {
        auto* inst = reinterpret_cast<VkAccelerationStructureInstanceKHR*>(
            static_cast<char*>(tlasInstanceBuffer_.mappedData) + instanceDelta_);
        inst[0].accelerationStructureReference = blasAddress_;
        inst[1].accelerationStructureReference = blasWaterAddress_;
        const bool haveScene = (sceneBlas_ != VK_NULL_HANDLE && sceneBlasAddress_ != 0);
        if (haveScene)
            inst[2].accelerationStructureReference = sceneBlasAddress_;
        VkAccelerationStructureGeometryInstancesDataKHR instances{};
        instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances.arrayOfPointers = VK_FALSE;
        instances.data.deviceAddress = tlasInstanceAddress_;
        VkAccelerationStructureGeometryKHR tlasGeom{};
        tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeom.geometry.instances = instances;
        tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
        tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasBuild.dstAccelerationStructure = tlas_;
        tlasBuild.geometryCount = 1;
        tlasBuild.pGeometries = &tlasGeom;
        tlasBuild.scratchData.deviceAddress = tlasScratchAligned_;
        VkAccelerationStructureBuildRangeInfoKHR trange{};
        trange.primitiveCount = haveScene ? 3u : 2u;
        const VkAccelerationStructureBuildRangeInfoKHR* pTrange = &trange;
        app->fpCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasBuild, &pTrange);
    }

    // 5. TLAS write -> ray-tracing / fragment / compute reads (ray queries in
    // main.frag sample the TLAS from the fragment stage; the water pipeline
    // reads it from the ray-tracing stage).
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    return true;
}

void RayTracingResources::updateParams(const RayTracingParams& p, uint32_t frameIndex) {
    if (!supported_) return;
    Buffer& slot = paramsBuffers_[frameIndex % kParamFrames];
    if (slot.mappedData == nullptr) return;
    memcpy(slot.mappedData, &p, sizeof(p));
}

// ── Water RT pipeline (rgen/miss/chit) + SBT ────────────────────────────
// One raygen (per-pixel reflection+refraction+thickness), one miss (sky
// equirect), one triangle hit (proxy-box shading + analytic exit thickness).
// Single bounce, no recursion (perf §16; payloads minimal).
void RayTracingResources::createRTPipeline(VulkanApp* app) {
    pipelineReady_ = false;
    if (!supported_ || !app->rayPipelineEnabled()) {
        std::cerr << "[HybridRT] ray_tracing_pipeline unavailable — water uses inline ray queries\n";
        return;
    }
    VkDevice device = app->getDevice();
    VkShaderModule rgen = VK_NULL_HANDLE, miss = VK_NULL_HANDLE, chit = VK_NULL_HANDLE;
    try {
        rgen = app->getOrCreateShaderModule("shaders/rt_water.rgen.spv");
        miss = app->getOrCreateShaderModule("shaders/rt_water.rmiss.spv");
        chit = app->getOrCreateShaderModule("shaders/rt_water.rchit.spv");
    } catch (const std::exception& e) {
        std::cerr << "[HybridRT] RT shaders missing (" << e.what() << ") — water uses inline ray queries\n";
        return;
    }
    if (rgen == VK_NULL_HANDLE || miss == VK_NULL_HANDLE || chit == VK_NULL_HANDLE) {
        std::cerr << "[HybridRT] RT shader modules null — water uses inline ray queries\n";
        return;
    }

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &rtSetLayout_;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &rtPipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("HybridRT: failed to create RT pipeline layout");
    app->resources.addPipelineLayout(rtPipelineLayout_, "HybridRT: rtPipelineLayout");

    std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
    auto stage = [&](uint32_t i, VkShaderModule m, VkShaderStageFlagBits s) {
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].module = m;
        stages[i].stage = s;
        stages[i].pName = "main";
    };
    stage(0, rgen, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    stage(1, miss, VK_SHADER_STAGE_MISS_BIT_KHR);
    stage(2, chit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};
    for (auto& g : groups) {
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader = VK_SHADER_UNUSED_KHR;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].closestHitShader = 2;

    VkRayTracingPipelineCreateInfoKHR pci{};
    pci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pci.stageCount = uint32_t(stages.size());
    pci.pStages = stages.data();
    pci.groupCount = uint32_t(groups.size());
    pci.pGroups = groups.data();
    pci.maxPipelineRayRecursionDepth = 1; // single secondary bounce (§7 default)
    pci.layout = rtPipelineLayout_;
    // NULL deferredOperation = blocking compile (init path may block; never in
    // the frame path). Pipeline cache inherited from the app for faster loads.
    pci.basePipelineHandle = VK_NULL_HANDLE;
    if (app->fpCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, app->getPipelineCache(),
            1, &pci, nullptr, &rtPipeline_) != VK_SUCCESS)
        throw std::runtime_error("HybridRT: vkCreateRayTracingPipelinesKHR failed");
    app->resources.addPipeline(rtPipeline_, "HybridRT: rtPipeline");

    // SBT: 3 records, each handleSize aligned per rtPipelineProps. Host-visible
    // + coherent; written once here, made visible by the init-time barrier in
    // the first dispatch (same-queue ordering covers it: the write completes
    // before any submit that traces).
    const auto& props = app->rtPipelineProps;
    const uint32_t handleSize = props.shaderGroupHandleSize;
    const uint32_t handleAlign = std::max(1u, props.shaderGroupHandleAlignment);
    const uint32_t baseAlign = std::max(1u, props.shaderGroupBaseAlignment);
    auto alignUp = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
    const uint32_t handleStride = alignUp(handleSize, handleAlign);
    const uint32_t recordStride = alignUp(handleStride, baseAlign);
    const VkDeviceSize sbtSize = VkDeviceSize(recordStride) * 3;
    // +baseAlign slack: SBT region base addresses must satisfy
    // shaderGroupBaseAlignment; VMA placement alone does not guarantee it.
    sbtBuffer_ = app->createBuffer(sbtSize + baseAlign,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::vector<uint8_t> handles(size_t(handleSize) * 3);
    if (app->fpGetRayTracingShaderGroupHandlesKHR(device, rtPipeline_, 0, 3,
            handles.size(), handles.data()) != VK_SUCCESS)
        throw std::runtime_error("HybridRT: vkGetRayTracingShaderGroupHandlesKHR failed");
    VkBufferDeviceAddressInfo sbtAddrQ{};
    sbtAddrQ.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    sbtAddrQ.buffer = sbtBuffer_.buffer;
    const VkDeviceAddress sbtRaw = vkGetBufferDeviceAddress(device, &sbtAddrQ);
    if (sbtRaw == 0) throw std::runtime_error("HybridRT: SBT device address is 0");
    const VkDeviceSize sbtDelta = alignUpAddr(sbtRaw, baseAlign) - sbtRaw;
    auto* dst = static_cast<uint8_t*>(sbtBuffer_.mappedData) + sbtDelta;
    memset(dst, 0, size_t(sbtSize));
    for (int i = 0; i < 3; ++i)
        memcpy(dst + size_t(i) * recordStride, handles.data() + size_t(i) * handleSize, handleSize);
    sbtAddress_ = sbtRaw + sbtDelta;
    rgenRegion_ = {sbtAddress_, recordStride, recordStride};
    missRegion_ = {sbtAddress_ + recordStride, recordStride, recordStride};
    hitRegion_ = {sbtAddress_ + recordStride * 2, recordStride, recordStride};
    callableRegion_ = {0, 0, 0};
    pipelineReady_ = true;
    printf("[HybridRT] RT pipeline ready (handle=%u stride=%u)\n", handleSize, recordStride);
}

void RayTracingResources::destroyRTPipeline(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();
    if (rtPipeline_ != VK_NULL_HANDLE) {
        if (app->resources.removePipeline(rtPipeline_)) vkDestroyPipeline(device, rtPipeline_, nullptr);
        rtPipeline_ = VK_NULL_HANDLE;
    }
    if (rtPipelineLayout_ != VK_NULL_HANDLE) {
        if (app->resources.removePipelineLayout(rtPipelineLayout_)) vkDestroyPipelineLayout(device, rtPipelineLayout_, nullptr);
        rtPipelineLayout_ = VK_NULL_HANDLE;
    }
    pipelineReady_ = false;
}

void RayTracingResources::dispatchWaterRT(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                                          const glm::mat4& invViewProj, const glm::vec3& viewPos) {
    if (!isPipelineReady() || !tlasBuilt_ || cmd == VK_NULL_HANDLE) return;
    if (reflectImage_ == VK_NULL_HANDLE || refractImage_ == VK_NULL_HANDLE) return;
    // Stream per-dispatch view state into this frame's params UBO (handle
    // stable; contents memcpy, no descriptor update). Toggles/distances/water
    // come from the caller's last updateParams(); here we only refresh the
    // view + output size (resize-safe: dispatch uses the live image extent).
    Buffer& paramSlot = paramsBuffers_[frameIdx % kParamFrames];
    if (paramSlot.mappedData) {
        auto* p = static_cast<RayTracingParams*>(paramSlot.mappedData);
        p->invViewProj = invViewProj;
        p->viewPos = glm::vec4(viewPos, 1.0f);
        p->rtResolution = glm::vec4(float(outWidth_), float(outHeight_),
            1.0f / float(outWidth_), 1.0f / float(outHeight_));
    }
    // Outputs GENERAL -> GENERAL no-op barrier with execution dependency:
    // the water geometry pass sampled them as SHADER_READ (previous frame's
    // results) earlier in this same command buffer; the layout transition to
    // GENERAL for writing is the only hazard (write-after-read, same queue).
    // Why: same-queue WAR — the fragment reads must complete before traceRays
    // overwrites the images for next frame.
    {
        VkImageMemoryBarrier2 barriers[2]{};
        for (int i = 0; i < 2; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barriers[i].srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].image = (i == 0) ? reflectImage_ : refractImage_;
            barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline_);
    VkDescriptorSet set = rtSets_[frameIdx % 3];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        rtPipelineLayout_, 0, 1, &set, 0, nullptr);
    app->fpCmdTraceRaysKHR(cmd, &rgenRegion_, &missRegion_, &hitRegion_, &callableRegion_,
        outWidth_, outHeight_, 1);
    // Write completion -> next frame's fragment sampling (same queue, ordered).
    // Why: traceRays storage writes must be visible before the next frame's
    // water fragment shader samples the images (cross-submission, same queue).
    {
        VkImageMemoryBarrier2 barriers[2]{};
        for (int i = 0; i < 2; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            barriers[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].image = (i == 0) ? reflectImage_ : refractImage_;
            barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}
