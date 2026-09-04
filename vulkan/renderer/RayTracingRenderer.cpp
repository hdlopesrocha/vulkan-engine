#include "RayTracingRenderer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#include "../MaterialManager.hpp"
#include "../TextureArrayManager.hpp"
#include "../VulkanApp.hpp"
#include "../ubo/SkyUniform.hpp"
#include "../ubo/UniformObject.hpp"
#include "DescriptorWriter.hpp"
#include "IndirectRenderer.hpp"

#ifdef USE_IMGUI
#include <imgui.h>
#endif

namespace {
constexpr uint32_t kSbtRaygenCount = 1;
constexpr uint32_t kSbtMissCount = 2; // radiance, shadow
constexpr uint32_t kSbtHitCount = 2;  // solid (rchit+rahit), water (rchit)

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) {
    return a == 0 ? v : ((v + a - 1) & ~(a - 1));
}
} // namespace

bool RayTracingRenderer::init(VulkanApp* app, TextureArrayManager* textures, MaterialManager* materials,
                              IndirectRenderer* solidIR, IndirectRenderer* waterIR,
                              const Buffer& skyUBO, const Buffer& waterParamsBuffer, uint32_t waterParamCount) {
    if (!app || !app->supportsRayTracing()) {
        std::cerr << "[RayTracing] Device lacks RT support — staying on the legacy rasterizer\n";
        available_ = false;
        return false;
    }
    app_ = app;
    textures_ = textures;
    materials_ = materials;
    solidIR_ = solidIR;
    waterIR_ = waterIR;
    skyUBO_ = skyUBO;
    waterParamsBuf_ = waterParamsBuffer;
    waterParamCount_ = waterParamCount;

    if (!asManager_.init(app)) {
        available_ = false;
        return false;
    }
    if (!createDescriptorLayout(app) || !createFrameUBOs(app)) {
        available_ = false;
        return false;
    }
    // Output images need the swapchain extent; (re)created here and on resize.
    if (!createOutputImages(app, app->getWidth(), app->getHeight())) {
        available_ = false;
        return false;
    }
    if (!createDescriptorPoolAndSets(app)) {
        available_ = false;
        return false;
    }
    // Pre-create the slot-meta buffers (sized to the pool capacities) so the
    // static descriptors below bind real buffers from the first frame. Without
    // this, bindings 11/12 would be null until the first syncFromScene grew
    // them lazily inside renderFrame.
    asManager_.syncFromScene(app, solidIR, waterIR, nullptr, AccelerationStructureManager::LodSelect(), 0);
    writeStaticDescriptors(app);
    if (!createPipeline(app) || !createShaderBindingTable(app)) {
        std::cerr << "[RayTracing] Pipeline/SBT creation failed — rasterizer fallback (RT will retry next init)\n";
        available_ = false;
        return false;
    }
    available_ = true;
    printf("[RayTracing] Initialized: HDR RT output %ux%u, mode=%d (0=legacy,1=RT,2=debug)\n",
           outWidth_, outHeight_, static_cast<int>(settings.mode));
    return true;
}

void RayTracingRenderer::cleanup(VulkanApp* app) {
    if (!app) app = app_;
    asManager_.cleanup(app);
    if (app) {
        if (sbtBuffer_.buffer != VK_NULL_HANDLE) app->destroyBuffer(sbtBuffer_);
        for (auto& b : frameUBOs_) {
            if (b.buffer != VK_NULL_HANDLE) app->destroyBuffer(b);
            b = {};
        }
        destroyOutputImages(app);
    } else {
        sbtBuffer_ = {};
        for (auto& b : frameUBOs_) b = {};
        for (auto& o : outputs_) o = {};
        for (auto& o : depthOutputs_) o = {};
    }
    sbtBuffer_ = {};
    pipeline_ = {};
    pipelineLayout_ = {};
    dsLayout_ = {};
    dsPool_ = {};
    for (auto& s : dsSets_) s = VK_NULL_HANDLE;
    for (auto& t : lastWrittenTlas_) t = VK_NULL_HANDLE;
    available_ = false;
    app_ = nullptr;
}

void RayTracingRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    if (!available_ || !app) return;
    destroyOutputImages(app);
    if (!createOutputImages(app, width, height)) return;
    // Re-point each frame set's storage-image binding at the new views
    // (resize path only — never in the render loop).
    for (uint32_t f = 0; f < kFrames; ++f) {
        if (dsSets_[f] == VK_NULL_HANDLE) continue;
        DescriptorWriter w(app->getDevice());
        if (outputs_[f].view != VK_NULL_HANDLE)
            w.writeImage(dsSets_[f], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, outputs_[f].view, VK_IMAGE_LAYOUT_GENERAL);
        if (depthOutputs_[f].view != VK_NULL_HANDLE)
            w.writeImage(dsSets_[f], 17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, depthOutputs_[f].view, VK_IMAGE_LAYOUT_GENERAL);
        w.flush();
    }
}

void RayTracingRenderer::refreshStaticDescriptors(VulkanApp* app) {
    if (!available_ || !app) return;
    writeStaticDescriptors(app);
    for (auto& t : lastWrittenTlas_) t = VK_NULL_HANDLE; // force TLAS re-point below
}

int RayTracingRenderer::activeDebugMode() const {
    if (settings.mode != Mode::Debug) return 0;
    return static_cast<int>(settings.debugView);
}

// ── Output images ───────────────────────────────────────────────────────────
bool RayTracingRenderer::createOutputImages(VulkanApp* app, uint32_t w, uint32_t h) {
    destroyOutputImages(app);
    if (w == 0 || h == 0) return false;
    outWidth_ = w;
    outHeight_ = h;
    auto makeTarget = [&](OutputTarget& o, VkFormat fmt, VkImageUsageFlags usage, const char* name) -> bool {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = fmt;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        app->createImageWithVma(ci, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                o.image, o.allocation, o.memory, name);
        if (o.image == VK_NULL_HANDLE) return false;
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = o.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(app->getDevice(), &vi, nullptr, &o.view) != VK_SUCCESS) return false;
        app->setImageLayoutTracked(o.image, VK_IMAGE_LAYOUT_UNDEFINED);
        o.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    };
    for (uint32_t f = 0; f < kFrames; ++f) {
        // HDR color: STORAGE (ray-gen write) -> SAMPLED (post-process read).
        // TRANSFER_SRC allows test readback; TRANSFER_DST serves the
        // empty-scene clear path (vkCmdClearColorImage, no TLAS yet).
        if (!makeTarget(outputs_[f], kOutputFormat,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        "rt-output")) {
            destroyOutputImages(app);
            return false;
        }
        // NDC depth companion in the raster depth space (post-process obstacle
        // tests compare it directly against raster depth targets).
        if (!makeTarget(depthOutputs_[f], kDepthFormat,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        "rt-depth")) {
            destroyOutputImages(app);
            return false;
        }
    }
    return true;
}

void RayTracingRenderer::destroyOutputImages(VulkanApp* app) {
    auto destroyOne = [&](OutputTarget& o) {
        if (!app) {
            o = {};
            return;
        }
        if (o.view != VK_NULL_HANDLE) {
            vkDestroyImageView(app->getDevice(), o.view, nullptr);
            o.view = VK_NULL_HANDLE;
        }
        if (o.image != VK_NULL_HANDLE) {
            app->destroyImageWithVma(o.image, o.allocation, o.memory);
            o.image = VK_NULL_HANDLE;
            o.allocation = VK_NULL_HANDLE;
            o.memory = VK_NULL_HANDLE;
        }
        o.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    for (auto& o : outputs_) destroyOne(o);
    for (auto& o : depthOutputs_) destroyOne(o);
}

// ── Descriptors ─────────────────────────────────────────────────────────────
bool RayTracingRenderer::createDescriptorLayout(VulkanApp* app) {
    using SF = VkShaderStageFlagBits;
    const VkShaderStageFlags allRt = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    const VkShaderStageFlags genHit = static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_RAYGEN_BIT_KHR) |
                                      static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) |
                                      static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_MISS_BIT_KHR);
    auto bind = [](uint32_t b, VkDescriptorType t, uint32_t n, VkShaderStageFlags s) {
        VkDescriptorSetLayoutBinding x{};
        x.binding = b;
        x.descriptorType = t;
        x.descriptorCount = n;
        x.stageFlags = s;
        x.pImmutableSamplers = nullptr;
        return x;
    };
    (void)static_cast<SF>(0);
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        bind(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, genHit),
        bind(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR),
        bind(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, allRt),
        bind(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
        bind(5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, allRt),
        bind(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, allRt),
        bind(7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, allRt),
        bind(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, allRt),
        bind(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, allRt),
        bind(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, allRt),
        bind(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allRt),
        bind(17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR),
    };
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = static_cast<uint32_t>(bindings.size());
    li.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(app->getDevice(), &li, nullptr, &layout) != VK_SUCCESS) return false;
    dsLayout_ = layout;
    app->resources.addDescriptorSetLayout(layout, "rtDescriptorLayout");
    app->registerDescriptorSetLayout(layout, "rtDescriptorLayout");
    return true;
}

bool RayTracingRenderer::createDescriptorPoolAndSets(VulkanApp* app) {
    std::vector<VkDescriptorPoolSize> sizes = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kFrames},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kFrames * 2},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFrames * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFrames * 8},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFrames * 5},
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = kFrames;
    pi.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pi.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(app->getDevice(), &pi, nullptr, &pool) != VK_SUCCESS) return false;
    dsPool_ = pool;
    app->resources.addDescriptorPool(pool, "rtDescriptorPool");
    std::vector<VkDescriptorSetLayout> layouts(kFrames, dsLayout_);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = kFrames;
    ai.pSetLayouts = layouts.data();
    std::array<VkDescriptorSet, kFrames> sets{};
    if (vkAllocateDescriptorSets(app->getDevice(), &ai, sets.data()) != VK_SUCCESS) return false;
    for (uint32_t f = 0; f < kFrames; ++f) {
        dsSets_[f] = sets[f];
        app->registerDescriptorSet(sets[f]);
    }
    return true;
}

bool RayTracingRenderer::createFrameUBOs(VulkanApp* app) {
    for (uint32_t f = 0; f < kFrames; ++f) {
        frameUBOs_[f] = app->createBuffer(sizeof(FrameUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (frameUBOs_[f].buffer == VK_NULL_HANDLE) return false;
    }
    return true;
}

void RayTracingRenderer::writeStaticDescriptors(VulkanApp* app) {
    if (!available_ && dsSets_[0] == VK_NULL_HANDLE) return;
    for (uint32_t f = 0; f < kFrames; ++f) {
        VkDescriptorSet ds = dsSets_[f];
        if (ds == VK_NULL_HANDLE) continue;
        DescriptorWriter w(app->getDevice());
        // Binding 1: this frame's HDR storage image (resize path rewrites it).
        if (outputs_[f].view != VK_NULL_HANDLE)
            w.writeImage(ds, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, outputs_[f].view, VK_IMAGE_LAYOUT_GENERAL);
        // Binding 17: this frame's NDC depth storage image.
        if (depthOutputs_[f].view != VK_NULL_HANDLE)
            w.writeImage(ds, 17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, depthOutputs_[f].view, VK_IMAGE_LAYOUT_GENERAL);
        // Binding 2: this frame's constants (contents stream via memcpy).
        if (frameUBOs_[f].buffer != VK_NULL_HANDLE)
            w.writeBuffer(ds, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          frameUBOs_[f].buffer, 0, sizeof(FrameUBO));
        // Binding 3: existing material SSBO (shared with the rasterizer).
        if (materials_ && materials_->getBuffer().buffer != VK_NULL_HANDLE)
            w.writeBuffer(ds, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          materials_->getBuffer().buffer, 0, VK_WHOLE_SIZE);
        // Binding 4: existing water params SSBO.
        if (waterParamsBuf_.buffer != VK_NULL_HANDLE)
            w.writeBuffer(ds, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          waterParamsBuf_.buffer, 0, VK_WHOLE_SIZE);
        // Binding 5: existing sky UBO.
        if (skyUBO_.buffer != VK_NULL_HANDLE)
            w.writeBuffer(ds, 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          skyUBO_.buffer, 0, sizeof(SkyUniform));
        // Bindings 6-10: existing texture arrays.
        if (textures_) {
            auto img = [&](uint32_t b, VkSampler s, VkImageView v) {
                if (s != VK_NULL_HANDLE && v != VK_NULL_HANDLE)
                    w.writeImage(ds, b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 s, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            };
            img(6, textures_->albedoSampler, textures_->albedoArray.view);
            img(7, textures_->normalSampler, textures_->normalArray.view);
            img(8, textures_->bumpSampler, textures_->bumpArray.view);
            img(9, textures_->roughnessSampler, textures_->roughnessArray.view);
            img(10, textures_->aoSampler, textures_->aoArray.view);
        }
        // Bindings 11-16: slot meta + packed pool views (uint). Pool views are
        // stable (pre-allocated once by initSlots); meta is per frame slot so
        // the host rewrite can never race an in-flight dispatch. Frame f's set
        // binds frame f's meta buffers (slot == frameIndex % kFrames).
        static_assert(kFrames == AccelerationStructureManager::kFrameSlots,
                      "RT frame slots must match AS manager slots");
        auto sbuf = [&](uint32_t b, VkBuffer buf) {
            if (buf != VK_NULL_HANDLE)
                w.writeBuffer(ds, b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buf, 0, VK_WHOLE_SIZE);
        };
        sbuf(11, asManager_.solidMetaBuffer(f).buffer);
        sbuf(12, asManager_.waterMetaBuffer(f).buffer);
        if (solidIR_) {
            sbuf(13, solidIR_->getVertexBuffer().buffer);
            sbuf(14, solidIR_->getIndexBuffer().buffer);
        }
        if (waterIR_) {
            sbuf(15, waterIR_->getVertexBuffer().buffer);
            sbuf(16, waterIR_->getIndexBuffer().buffer);
        }
        w.flush();
    }
}

// Point this frame's set at the freshest TLAS object across all slots (see
// tlasFresh()). Only the current slot's set is ever written (its fence makes
// that legal; other frames' sets may be in flight — cf. VUID-03047). Because
// every frame re-points its own set at the global freshest object, dispatches
// never serve stale rung selections, yet steady state issues zero updates
// (at most one single-set write on dirty frames).
void RayTracingRenderer::writeTlasDescriptor(VulkanApp* app, uint32_t frameIndex) {
    const uint32_t slot = frameIndex % kFrames;
    VkAccelerationStructureKHR tlas = asManager_.tlasFresh();
    if (tlas == VK_NULL_HANDLE || tlas == lastWrittenTlas_[slot]) return;
    VkDescriptorSet ds = dsSets_[slot];
    if (ds == VK_NULL_HANDLE) return;
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;
    VkWriteDescriptorSet ww{};
    ww.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ww.dstSet = ds;
    ww.dstBinding = 0;
    ww.descriptorCount = 1;
    ww.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    ww.pNext = &asInfo;
    DescriptorUpdateStats::noteUpdate(1);
    vkUpdateDescriptorSets(app->getDevice(), 1, &ww, 0, nullptr);
    lastWrittenTlas_[slot] = tlas;
}

// ── Pipeline + SBT ──────────────────────────────────────────────────────────
bool RayTracingRenderer::createPipeline(VulkanApp* app) {
    auto mod = [&](const char* path) { return app->getOrCreateShaderModule(path); };
    VkShaderModule rgen = mod("shaders/rt_basic.rgen.spv");
    VkShaderModule missRad = mod("shaders/rt_radiance.rmiss.spv");
    VkShaderModule missShadow = mod("shaders/rt_shadow.rmiss.spv");
    VkShaderModule chitSolid = mod("shaders/rt_solid.rchit.spv");
    VkShaderModule chitWater = mod("shaders/rt_water.rchit.spv");
    VkShaderModule ahitAlpha = mod("shaders/rt_alpha.rahit.spv");
    if (!rgen || !missRad || !missShadow || !chitSolid || !chitWater || !ahitAlpha) {
        std::cerr << "[RayTracing] Missing RT shader modules (run `make shaders`)\n";
        return false;
    }
    auto stage = [](VkShaderModule m, VkShaderStageFlagBits s) {
        VkPipelineShaderStageCreateInfo st{};
        st.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st.stage = s;
        st.module = m;
        st.pName = "main";
        return st;
    };
    std::vector<VkPipelineShaderStageCreateInfo> stages = {
        stage(rgen, VK_SHADER_STAGE_RAYGEN_BIT_KHR),
        stage(missRad, VK_SHADER_STAGE_MISS_BIT_KHR),
        stage(missShadow, VK_SHADER_STAGE_MISS_BIT_KHR),
        stage(chitSolid, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
        stage(chitWater, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
        stage(ahitAlpha, VK_SHADER_STAGE_ANY_HIT_BIT_KHR),
    };
    auto general = [](uint32_t idx) {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader = idx;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        return g;
    };
    auto triangles = [](uint32_t chit, uint32_t ahit) {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        g.generalShader = VK_SHADER_UNUSED_KHR;
        g.closestHitShader = chit;
        g.anyHitShader = ahit;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        return g;
    };
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups = {
        general(0),          // 0: raygen
        general(1),          // 1: miss radiance (sky)
        general(2),          // 2: miss shadow (lit)
        triangles(3, 5),     // 3: solid closest-hit + alpha any-hit
        triangles(4, VK_SHADER_UNUSED_KHR), // 4: water closest-hit
    };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VkDescriptorSetLayout dsl = dsLayout_;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(app->getDevice(), &pli, nullptr, &layout) != VK_SUCCESS) return false;
    pipelineLayout_ = layout;
    app->resources.addPipelineLayout(layout, "RayTracingRenderer: pipelineLayout");

    VkRayTracingPipelineCreateInfoKHR pci{};
    pci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pci.stageCount = static_cast<uint32_t>(stages.size());
    pci.pStages = stages.data();
    pci.groupCount = static_cast<uint32_t>(groups.size());
    pci.pGroups = groups.data();
    // Bounded recursion: rgen -> closest-hit -> nested reflection/refraction
    // trace (+ shadow rays, which skip closest-hit but still nest).
    pci.maxPipelineRayRecursionDepth = 4;
    pci.layout = layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = app->rtFunctions.createRTPipelines(app->getDevice(), VK_NULL_HANDLE,
                                                    app->getPipelineCache(), 1, &pci, nullptr, &pipe);
    if (r != VK_SUCCESS) {
        std::cerr << "[RayTracing] vkCreateRayTracingPipelinesKHR failed: " << r << "\n";
        return false;
    }
    pipeline_ = pipe;
    app->resources.addPipeline(pipe, "RayTracingRenderer: rayTracingPipeline");
    printf("[RayTracing] RT pipeline created (6 stages, 5 groups, recursion<=4)\n");
    return true;
}

bool RayTracingRenderer::createShaderBindingTable(VulkanApp* app) {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &p2);

    const uint32_t groupCount = kSbtRaygenCount + kSbtMissCount + kSbtHitCount;
    const uint32_t handleSize = rtProps.shaderGroupHandleSize;
    const VkDeviceSize baseAlign = rtProps.shaderGroupBaseAlignment ? rtProps.shaderGroupBaseAlignment : 1;
    const VkDeviceSize handleAlign = rtProps.shaderGroupHandleAlignment ? rtProps.shaderGroupHandleAlignment : 1;
    // Record stride: fits a handle and is a multiple of baseAlignment (hard
    // VUID on region stride).
    const uint32_t recStride = static_cast<uint32_t>(alignUp(alignUp(handleSize, handleAlign), baseAlign));
    sbtHandleSizeAligned_ = recStride;
    sbtMissStride_ = recStride;
    sbtHitStride_ = recStride;
    const VkDeviceSize raygenSize = alignUp(recStride, baseAlign);
    const VkDeviceSize missSize = alignUp(static_cast<VkDeviceSize>(kSbtMissCount) * recStride, baseAlign);
    const VkDeviceSize hitSize = alignUp(static_cast<VkDeviceSize>(kSbtHitCount) * recStride, baseAlign);
    const VkDeviceSize total = raygenSize + missSize + hitSize;

    std::vector<uint8_t> handles(static_cast<size_t>(groupCount) * handleSize);
    if (app->rtFunctions.getGroupHandles(app->getDevice(), pipeline_, 0, groupCount,
                                         handles.size(), handles.data()) != VK_SUCCESS) {
        std::cerr << "[RayTracing] getGroupHandles failed\n";
        return false;
    }
    // Every region deviceAddress must be a multiple of baseAlignment (hard
    // VUID). VMA suballocation doesn't guarantee that, so over-allocate and
    // start the table at an aligned offset inside the buffer.
    sbtBuffer_ = app->createBuffer(total + baseAlign,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (sbtBuffer_.buffer == VK_NULL_HANDLE) return false;
    const VkDeviceAddress rawBase = rt::getBufferAddress(app->getDevice(), sbtBuffer_.buffer);
    const VkDeviceAddress base = (rawBase + baseAlign - 1) & ~(baseAlign - 1);
    const VkDeviceSize baseOffset = static_cast<VkDeviceSize>(base - rawBase);
    auto* dst = static_cast<uint8_t*>(sbtBuffer_.mappedData);
    std::memset(dst, 0, static_cast<size_t>(total + baseAlign));
    auto copyRecord = [&](VkDeviceSize offset, uint32_t groupIdx) {
        std::memcpy(dst + baseOffset + offset, handles.data() + static_cast<size_t>(groupIdx) * handleSize, handleSize);
    };
    copyRecord(0, 0); // raygen
    for (uint32_t i = 0; i < kSbtMissCount; ++i) copyRecord(raygenSize + i * recStride, 1 + i);
    for (uint32_t i = 0; i < kSbtHitCount; ++i) copyRecord(raygenSize + missSize + i * recStride, 3 + i);

    sbtRaygenAddr_ = base;
    sbtMissAddr_ = base + raygenSize;
    sbtHitAddr_ = base + raygenSize + missSize;
    printf("[RayTracing] SBT ready (raygen=%llu miss=%llu hit=%llu, stride=%u)\n",
           (unsigned long long)raygenSize, (unsigned long long)missSize,
           (unsigned long long)hitSize, recStride);
    return true;
}

// ── Barriers ────────────────────────────────────────────────────────────────
void RayTracingRenderer::recordOutputBarriers(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, bool beforeTrace) {
    OutputTarget& o = outputs_[frameIndex % kFrames];
    OutputTarget& d = depthOutputs_[frameIndex % kFrames];
    VkImageMemoryBarrier2 barriers[2]{};
    auto fill = [&](VkImageMemoryBarrier2& b, OutputTarget& t) {
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = t.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (beforeTrace) {
            // HDR/depth images: previous SHADER_READ_ONLY (post-process sampled
            // them last frame) or UNDEFINED -> GENERAL for ray-gen storage writes.
            // UNDEFINED carries no prior contents, so the source masks stay empty.
            if (t.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                b.srcAccessMask = VK_ACCESS_2_NONE;
            } else {
                b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            }
            // Destination covers BOTH writers used after this barrier: the
            // ray-gen storage write (normal path) and the transfer clear
            // (empty-scene path with no TLAS yet). Without CLEAR/TRANSFER_WRITE
            // here, vkCmdClearColorImage hazards against the layout transition
            // (WRITE_AFTER_WRITE: the transition counts as a prior write).
            b.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_CLEAR_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.oldLayout = t.layout;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
            // Writers -> post-process/composite sample. Source covers both the
            // ray-gen storage write and the empty-scene transfer clear, so the
            // clear is correctly ordered before the composite reads it.
            b.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_CLEAR_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    };
    uint32_t n = 0;
    if (o.image != VK_NULL_HANDLE) fill(barriers[n++], o);
    if (d.image != VK_NULL_HANDLE) fill(barriers[n++], d);
    if (n == 0) return;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = n;
    dep.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);
    if (o.image != VK_NULL_HANDLE) {
        app->recordTrackedLayoutForCommandBuffer(cmd, o.image, barriers[0].newLayout, 0, 1);
        o.layout = barriers[0].newLayout;
    }
    if (d.image != VK_NULL_HANDLE) {
        const uint32_t di = (o.image != VK_NULL_HANDLE) ? 1 : 0;
        app->recordTrackedLayoutForCommandBuffer(cmd, d.image, barriers[di].newLayout, 0, 1);
        d.layout = barriers[di].newLayout;
    }
}

// ── Frame ───────────────────────────────────────────────────────────────────
void RayTracingRenderer::renderFrame(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                                     const UniformObject& ubo, float time, uint32_t traceCullMask,
                                     float lodBias, uint32_t maxTargetLod) {
    const auto cpuT0 = std::chrono::steady_clock::now();
    lastStats_ = Stats{};
    if (!available_ || settings.mode == Mode::Legacy || cmd == VK_NULL_HANDLE) return;
    if (pipeline_ == VK_NULL_HANDLE || !solidIR_) return;
    const uint32_t slot = frameIndex % kFrames;

    // 1. Diff live geometry -> dirty BLAS/TLAS sets (CPU only, no GPU work).
    // LoD rung selection uses the same camera/bias inputs as the GPU cull so
    // the TLAS holds exactly one rung per region (never all rungs at once).
    AccelerationStructureManager::LodSelect lod;
    lod.camPos = glm::vec3(ubo.viewPos);
    lod.lodBias = lodBias;
    lod.maxTargetLod = maxTargetLod;
    const bool needsBuild = asManager_.syncFromScene(app, solidIR_, waterIR_, nullptr, lod, slot);
    (void)needsBuild;

    // 2. Per-frame constants stream via memcpy (no descriptor updates).
    if (frameUBOs_[slot].mappedData) {
        auto* dst = static_cast<FrameUBO*>(frameUBOs_[slot].mappedData);
        dst->viewProj = ubo.viewProjection;
        dst->invViewProj = ubo.invViewProjection;
        dst->viewPos = ubo.viewPos;
        // Engine convention (see main.frag: toLight = -ubo.lightDir): the UBO
        // stores the light's travel direction (pointing down). RT shaders
        // expect the direction TO the light, so negate once here (w preserved).
        dst->lightDir = glm::vec4(-ubo.lightDir.x, -ubo.lightDir.y, -ubo.lightDir.z, ubo.lightDir.w);
        dst->lightColor = ubo.lightColor;
        dst->rtParams = glm::vec4(time, static_cast<float>(activeDebugMode()),
                                  static_cast<float>(std::clamp(settings.maxDepth, 1, 4)),
                                  settings.shadowsEnabled ? 1.0f : 0.0f);
        const bool waterOn = settings.refractionEnabled || settings.reflectionsEnabled;
        const float sigmaScale = 1.0f;
        dst->waterAbsorption = glm::vec4(settings.waterAbsorption * sigmaScale, waterOn ? 1.0f : 0.0f);
        dst->waterMisc = glm::vec4(settings.waterIOR, settings.waterReflectionStrength,
                                   settings.refractionEnabled ? 1.0f : 0.0f,
                                   settings.reflectionsEnabled ? 1.0f : 0.0f);
        dst->traceMask = glm::vec4(static_cast<float>(traceCullMask), 0.0f, 0.0f, 0.0f);
        // Raster parity: triplanar weights and map toggles mirror the scene UBO.
        dst->triplanarParams = glm::vec4(ubo.triplanarSettings.x, ubo.triplanarSettings.y, 0.0f, 0.0f);
        dst->featureToggles = glm::vec4(ubo.materialFlags.w,
                                        ubo.debugParams.y, ubo.debugParams.z, ubo.passParams.y);
    }

    // 3. Make freshly uploaded geometry + host-written AS inputs visible to
    // the AS build and the ray dispatch. Uploads land via the transfer/compute
    // stages (submit-time timeline waits already order them); the instance and
    // slot-meta buffers are host-written just above in syncFromScene. This
    // barrier documents both usage transitions for validation. Skipped in
    // steady state (no dirty geometry) to keep the barrier count minimal.
    if (needsBuild) {
        std::vector<VkBufferMemoryBarrier2> barriers;
        auto poolBarrier = [&](const Buffer& b) {
            if (b.buffer == VK_NULL_HANDLE) return;
            VkBufferMemoryBarrier2 x{};
            x.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            x.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            x.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            x.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            x.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            x.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            x.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            x.buffer = b.buffer;
            x.offset = 0;
            x.size = VK_WHOLE_SIZE;
            barriers.push_back(x);
        };
        poolBarrier(solidIR_->getVertexBuffer());
        poolBarrier(solidIR_->getIndexBuffer());
        if (waterIR_) {
            poolBarrier(waterIR_->getVertexBuffer());
            poolBarrier(waterIR_->getIndexBuffer());
        }
        if (!barriers.empty()) {
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            dep.pBufferMemoryBarriers = barriers.data();
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        // Host-written AS inputs (TLAS instance buffer + slot-meta buffers,
        // filled on the CPU in syncFromScene) -> AS build + ray-tracing reads.
        VkMemoryBarrier2 host{};
        host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        host.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        host.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        host.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        host.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo hdep{};
        hdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        hdep.memoryBarrierCount = 1;
        hdep.pMemoryBarriers = &host;
        vkCmdPipelineBarrier2(cmd, &hdep);
    }

    // 4. HDR output -> GENERAL for the storage write.
    recordOutputBarriers(app, cmd, slot, true);

    // 5. Incremental BLAS/TLAS builds (no-ops in steady state).
    const uint32_t builds = asManager_.buildPending(app, cmd, slot);
    lastStats_.blasBuildsThisFrame = builds;

    if (asManager_.tlasFresh() == VK_NULL_HANDLE) {
        // No geometry yet (empty scene): clear to sky-ish gradient via a fill
        // so post-process still composites something sane, then transition.
        VkClearColorValue clear{};
        clear.float32[0] = 0.35f; clear.float32[1] = 0.55f;
        clear.float32[2] = 0.85f; clear.float32[3] = 0.0f;
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, outputs_[slot].image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        if (depthOutputs_[slot].image != VK_NULL_HANDLE) {
            VkClearColorValue dclear{};
            dclear.float32[0] = 1.0f;
            vkCmdClearColorImage(cmd, depthOutputs_[slot].image, VK_IMAGE_LAYOUT_GENERAL, &dclear, 1, &range);
        }
        recordOutputBarriers(app, cmd, slot, false);
        auto s = asManager_.stats();
        lastStats_.blasCount = s.blasCount;
        lastStats_.overlapColumns = s.overlapColumns;
        return;
    }

    // 6. Point the frame's set at the TLAS (only when the handle changed).
    writeTlasDescriptor(app, slot);

    // 7. Trace.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout_,
                            0, 1, &dsSets_[slot], 0, nullptr);
    VkStridedDeviceAddressRegionKHR raygen{sbtRaygenAddr_, sbtHandleSizeAligned_, sbtHandleSizeAligned_};
    VkStridedDeviceAddressRegionKHR miss{sbtMissAddr_, sbtMissStride_, sbtMissStride_ * kSbtMissCount};
    VkStridedDeviceAddressRegionKHR hit{sbtHitAddr_, sbtHitStride_, sbtHitStride_ * kSbtHitCount};
    VkStridedDeviceAddressRegionKHR callable{0, 0, 0};
    app->rtFunctions.cmdTraceRays(cmd, &raygen, &miss, &hit, &callable, outWidth_, outHeight_, 1);

    // 8. HDR output -> SHADER_READ_ONLY for the existing post-process chain.
    recordOutputBarriers(app, cmd, slot, false);

    const auto cpuT1 = std::chrono::steady_clock::now();
    auto s = asManager_.stats();
    lastStats_.blasCount = s.blasCount;
    lastStats_.tlasInstances = s.tlasInstances;
    lastStats_.overlapColumns = s.overlapColumns;
    lastStats_.blasBuildMs = s.lastBlasBuildMs;
    lastStats_.tlasBuildMs = s.lastTlasBuildMs;
    lastStats_.totalBlasBuilds = s.totalBlasBuilds;
    lastStats_.totalTlasBuilds = s.totalTlasBuilds;
    lastStats_.dispatchCpuMs = std::chrono::duration<float, std::milli>(cpuT1 - cpuT0).count();
    lastStats_.primaryRays = static_cast<uint64_t>(outWidth_) * outHeight_;
    lastStats_.outWidth = outWidth_;
    lastStats_.outHeight = outHeight_;
}

VkImage RayTracingRenderer::outputImage(uint32_t frameIndex) const {
    return outputs_[frameIndex % kFrames].image;
}

VkImageView RayTracingRenderer::outputView(uint32_t frameIndex) const {
    return outputs_[frameIndex % kFrames].view;
}

VkImageView RayTracingRenderer::depthView(uint32_t frameIndex) const {
    return depthOutputs_[frameIndex % kFrames].view;
}

void RayTracingRenderer::drawUI() {
#ifdef USE_IMGUI
    if (!available_) {
        ImGui::TextDisabled("Ray tracing: unavailable on this device (rasterizer active)");
        return;
    }
    const char* modes[] = {"Legacy rasterizer", "Ray tracing", "RT debug"};
    int m = static_cast<int>(settings.mode);
    if (ImGui::Combo("RT_RENDERER_ENABLED", &m, modes, 3)) {
        settings.mode = static_cast<Mode>(m);
        if (settings.mode == Mode::Debug && settings.debugView == DebugView::Shaded)
            settings.debugView = DebugView::HitDistance;
    }
    const char* views[] = {"Shaded", "Hit distance", "World normal", "Material ID",
                           "Instance/primitive", "Fresnel", "Water thickness",
                           "Reflection", "Refraction", "Shadow", "Bounces"};
    int v = static_cast<int>(settings.debugView);
    if (ImGui::Combo("RT debug view", &v, views, 11)) settings.debugView = static_cast<DebugView>(v);
    ImGui::SliderInt("Max bounce depth", &settings.maxDepth, 1, 4);
    ImGui::Checkbox("RT shadows", &settings.shadowsEnabled);
    ImGui::Checkbox("RT reflections", &settings.reflectionsEnabled);
    ImGui::Checkbox("RT refraction", &settings.refractionEnabled);
    ImGui::SliderFloat("Water IOR", &settings.waterIOR, 1.0f, 1.6f);
    ImGui::SliderFloat("Water absorption R", &settings.waterAbsorption.x, 0.0f, 2.0f);
    ImGui::SliderFloat("Water absorption G", &settings.waterAbsorption.y, 0.0f, 2.0f);
    ImGui::SliderFloat("Water absorption B", &settings.waterAbsorption.z, 0.0f, 2.0f);
    ImGui::SliderFloat("Water reflection", &settings.waterReflectionStrength, 0.0f, 2.0f);
    ImGui::Separator();
    ImGui::Text("BLAS: %u  TLAS instances: %u  overlapCols: %u", lastStats_.blasCount, lastStats_.tlasInstances, lastStats_.overlapColumns);
    ImGui::Text("BLAS builds (frame/total): %u / %llu", lastStats_.blasBuildsThisFrame,
                (unsigned long long)lastStats_.totalBlasBuilds);
    ImGui::Text("TLAS builds (total): %llu", (unsigned long long)lastStats_.totalTlasBuilds);
    ImGui::Text("AS build CPU: BLAS %.2f ms  TLAS %.2f ms", lastStats_.blasBuildMs, lastStats_.tlasBuildMs);
    ImGui::Text("Dispatch record: %.2f ms  Primary rays: %llu", lastStats_.dispatchCpuMs,
                (unsigned long long)lastStats_.primaryRays);
    ImGui::Text("Output: %ux%u R16G16B16A16", lastStats_.outWidth, lastStats_.outHeight);
#else
    (void)0;
#endif
}
