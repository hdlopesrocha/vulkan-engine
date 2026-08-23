#include "RayTracingRenderer.hpp"

#include "../renderer/VegetationRenderer.hpp" // for VegetationRenderer + VertexBufferObject

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

// Identity 3x4 transform for TLAS instances (chunk geometry is already in world
// space in the slotted IndirectRenderer pools, so instance transforms are identity).
constexpr VkTransformMatrixKHR kIdentityTransform = {
    { { 1.0f, 0.0f, 0.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f, 0.0f },
      { 0.0f, 0.0f, 1.0f, 0.0f } }
};

} // namespace

void RayTracingRenderer::init(VulkanApp* app, VkImageView shadowOutputView) {
    if (!app->rtSupport.any()) {
        printf("[RayTracingRenderer] hardware RT unsupported — initializing software fallback\n");
        ctx_.init(app);
        useSoftware_ = true;
        shadowOutputView_ = shadowOutputView;
        createSoftDescriptorSetLayout(app);
        createSoftPipelines(app);
        // 64KB chunk info buffer (enough for 4096 chunks)
        softChunkInfoBuffer_ = ctx_.createRtBuffer(4096 * sizeof(SoftChunkInfo),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        inited_ = true;
        tlasDirty_ = true;
        softChunkInfoDirty_ = true;
        printf("[RayTracingRenderer] software fallback initialized (brute-force compute)\n");
        return;
    }
    ctx_.init(app);
    shadowOutputView_ = shadowOutputView;

    createRtDescriptorSetLayout(app);
    createPipelines(app);

    inited_ = true;
    tlasDirty_ = true;
    printf("[RayTracingRenderer] initialized (shadow / reflection / refraction workloads)\n");
}

void RayTracingRenderer::cleanup(VulkanApp* app) {
    if (!inited_) return;
    if (useSoftware_) {
        if (softRenderPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(app->getDevice(), softRenderPipeline_, nullptr);
        if (softShadowPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(app->getDevice(), softShadowPipeline_, nullptr);
        if (softPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(app->getDevice(), softPipelineLayout_, nullptr);
        if (softDescSet_ != VK_NULL_HANDLE) app->resources.removeDescriptorSet(softDescSet_);
        for (uint32_t i = 0; i < (uint32_t)RtWorkload::Count; ++i) if (softOutputSets_[i] != VK_NULL_HANDLE) app->resources.removeDescriptorSet(softOutputSets_[i]);
        if (softDescLayout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(app->getDevice(), softDescLayout_, nullptr); app->resources.removeDescriptorSetLayout(softDescLayout_); }
        if (softOutputLayout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(app->getDevice(), softOutputLayout_, nullptr); app->resources.removeDescriptorSetLayout(softOutputLayout_); }
        if (softPool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(app->getDevice(), softPool_, nullptr); app->resources.removeDescriptorPool(softPool_); }
        if (softChunkInfoBuffer_.buffer != VK_NULL_HANDLE) ctx_.destroyRtBuffer(softChunkInfoBuffer_);
        useSoftware_ = false;
    }
    for (auto& w : workloads_) {
        for (auto m : w.modules) vkDestroyShaderModule(app->getDevice(), m, nullptr);
        if (w.pipeline) vkDestroyPipeline(app->getDevice(), w.pipeline, nullptr);
        if (w.layout) vkDestroyPipelineLayout(app->getDevice(), w.layout, nullptr);
        w.sbt.destroy();
    }
    if (rtDescriptorSet_ != VK_NULL_HANDLE)
        app->resources.removeDescriptorSet(rtDescriptorSet_);
    if (rtDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(app->getDevice(), rtDescLayout_, nullptr);
        app->resources.removeDescriptorSetLayout(rtDescLayout_);
    }
    if (rtOutputLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(app->getDevice(), rtOutputLayout_, nullptr);
        app->resources.removeDescriptorSetLayout(rtOutputLayout_);
    }
    if (rtPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(app->getDevice(), rtPool_, nullptr);
        app->resources.removeDescriptorPool(rtPool_);
    }
    for (uint32_t i = 0; i < (uint32_t)RtWorkload::Count; ++i) if (rtOutputSets_[i] != VK_NULL_HANDLE) app->resources.removeDescriptorSet(rtOutputSets_[i]);
    if (tlas_) tlas_->teardown(app);
    tlas_.reset();
    if (vegBlas_) vegBlas_->teardown(app);
    vegBlas_.reset();
    for (auto& kv : chunks_) if (kv.second.blas) kv.second.blas->teardown(app);
    chunks_.clear();
    inited_ = false;
}

void RayTracingRenderer::createRtDescriptorSetLayout(VulkanApp* app) {
    // Mirror the 14 bindings of the main descriptor set (so the SAME resource
    // handles can be copied in via vkCopyDescriptorSet) but extend their stage
    // flags to include the ray-tracing stages, and add the TLAS at binding 20.
    // This reuses the existing UBO, texture/sampler arrays, materials SSBO and
    // shadow-map bindings without copying any GPU scene data.
    auto rt = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
              VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    auto ubo = [&](uint32_t b, VkDescriptorType t, VkShaderStageFlags f) {
        VkDescriptorSetLayoutBinding bd{};
        bd.binding = b; bd.descriptorCount = 1; bd.descriptorType = t;
        bd.pImmutableSamplers = nullptr; bd.stageFlags = f | rt;
        return bd;
    };
    std::array<VkDescriptorSetLayoutBinding, 16> bindings{};
    bindings[0]  = ubo(0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    bindings[1]  = ubo(1,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    bindings[2]  = ubo(2,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[3]  = ubo(3,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    bindings[4]  = ubo(4,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[5]  = ubo(5,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    bindings[6]  = ubo(6,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[7]  = ubo(7,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT|VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    bindings[8]  = ubo(8,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[9]  = ubo(9,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[10] = ubo(10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT|VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    bindings[11] = ubo(11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[12] = ubo(12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings[13] = ubo(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    // Vegetation leaf-opacity texture (sampler2DArray) for the any-hit alpha test.
    bindings[14] = ubo(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, rt);
    // TLAS at binding 20 (new).
    bindings[15] = ubo(20, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, rt);

    std::array<VkDescriptorBindingFlags, 16> bflags{};
    bflags.fill(0);
    bflags[11] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT; // match main set (env cubemap)

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = (uint32_t)bindings.size();
    flagsInfo.pBindingFlags = bflags.data();

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.pNext = &flagsInfo;
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    li.bindingCount = (uint32_t)bindings.size();
    li.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(app->getDevice(), &li, nullptr, &rtDescLayout_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create RT descriptor set layout");
    app->resources.addDescriptorSetLayout(rtDescLayout_, "RayTracingRenderer: rtDescLayout");

    // Pool for set=0 (full main + TLAS) and three set=1 output-image sets.
    std::array<VkDescriptorPoolSize, 5> poolSizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 14 * 4 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 * 4 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 4 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 },
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // rtDescLayout_ is created with UPDATE_AFTER_BIND_POOL_BIT, so the pool must
    // carry the matching flag (binding 11 = env cubemap uses update-after-bind).
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pi.maxSets = 8;
    pi.poolSizeCount = (uint32_t)poolSizes.size();
    pi.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(app->getDevice(), &pi, nullptr, &rtPool_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create RT descriptor pool");

    rtDescriptorSet_ = app->createDescriptorSet(rtDescLayout_, rtPool_);

    // Copy the static main bindings (1-13: textures, materials, sky, water
    // params, cubemap) from the static set — written once, not per-frame. Binding 0
    // (per-frame UBO) is refreshed every frame from the current main descriptor set
    // (see updateRayTracing), so it is intentionally excluded here.
    std::vector<VkCopyDescriptorSet> copies;
    for (uint32_t b = 1; b <= 13; ++b) {
        VkCopyDescriptorSet c{};
        c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        c.srcSet = app->getStaticDescriptorSet();
        c.srcBinding = b; c.srcArrayElement = 0;
        c.dstSet = rtDescriptorSet_;
        c.dstBinding = b; c.dstArrayElement = 0;
        c.descriptorCount = 1;
        copies.push_back(c);
    }
    vkUpdateDescriptorSets(app->getDevice(), 0, nullptr, (uint32_t)copies.size(), copies.data());

    // set=1 output image layout (one storage image per workload).
    VkDescriptorSetLayoutBinding outBinding{};
    outBinding.binding = 0;
    outBinding.descriptorCount = 1;
    outBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outBinding.stageFlags = rt;
    VkDescriptorSetLayoutCreateInfo oli{};
    oli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    oli.bindingCount = 1;
    oli.pBindings = &outBinding;
    if (vkCreateDescriptorSetLayout(app->getDevice(), &oli, nullptr, &rtOutputLayout_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create RT output layout");

    for (uint32_t i = 0; i < (uint32_t)RtWorkload::Count; ++i) {
        rtOutputSets_[i] = app->createDescriptorSet(rtOutputLayout_, rtPool_);
    }
}

void RayTracingRenderer::createPipelines(VulkanApp* app) {
    // Query RT properties for SBT handle size / alignment.
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 pdp2{};
    pdp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    pdp2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &pdp2);

    // Push constant: uint mode (0=shadow,1=reflection,2=refraction).
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                   VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    pc.offset = 0; pc.size = sizeof(uint32_t);

    const std::array<VkDescriptorSetLayout, 2> plSetLayouts = { rtDescLayout_, rtOutputLayout_ };

    struct WkDef { const char* rgen; const char* miss; const char* rchit; const char* anyhit; RtWorkload wk; };
    WkDef defs[] = {
        { "shaders/raytracing/shadow.rgen.spv",  "shaders/raytracing/miss_shadow.rmiss.spv", "shaders/raytracing/closesthit.rchit.spv", "shaders/raytracing/anyhit.rahit.spv",  RtWorkload::Shadow },
        { "shaders/raytracing/reflect.rgen.spv", "shaders/raytracing/miss_env.rmiss.spv",    "shaders/raytracing/closesthit.rchit.spv", "shaders/raytracing/anyhit.rahit.spv",  RtWorkload::Reflection },
        { "shaders/raytracing/refract.rgen.spv", "shaders/raytracing/miss_env.rmiss.spv",    "shaders/raytracing/closesthit.rchit.spv", "shaders/raytracing/anyhit.rahit.spv",  RtWorkload::Refraction },
        { "shaders/raytracing/render.rgen.spv",  "shaders/raytracing/miss_env.rmiss.spv",    "shaders/raytracing/render.rchit.spv",     "shaders/raytracing/render.rahit.spv",   RtWorkload::Render },
    };

    for (auto& d : defs) {
        auto& w = workloads_[(uint32_t)d.wk];
        w.modules.clear();
        auto load = [&](const char* path, VkShaderStageFlagBits stage) {
            VkShaderModule m = app->getOrCreateShaderModule(path);
            w.modules.push_back(m);
            VkPipelineShaderStageCreateInfo s{};
            s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            s.stage = stage; s.module = m; s.pName = "main";
            return s;
        };
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        stages.push_back(load(d.rgen, VK_SHADER_STAGE_RAYGEN_BIT_KHR));        // idx 0
        stages.push_back(load(d.miss, VK_SHADER_STAGE_MISS_BIT_KHR));          // idx 1
        stages.push_back(load(d.rchit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));  // idx 2
        stages.push_back(load(d.anyhit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR));     // idx 3

        // Groups: 0=raygen, 1=miss, 2=hit solid, 3=hit water, 4=hit veg
        std::array<VkRayTracingShaderGroupCreateInfoKHR, 5> groups{};
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
        groups[2].closestHitShader = 2; // solid: opaque (no any-hit)
        groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[3].closestHitShader = 2; // water: opaque (no any-hit in this first integration)
        groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[4].closestHitShader = 2;
        groups[4].anyHitShader = 3;     // vegetation: alpha-tested any-hit

        VkRayTracingPipelineCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        ci.stageCount = (uint32_t)stages.size();
        ci.pStages = stages.data();
        ci.groupCount = (uint32_t)groups.size();
        ci.pGroups = groups.data();
        ci.maxPipelineRayRecursionDepth = 2; // shadow=1; reflect/refract may recurse once
        ci.layout = w.layout; // set below

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = (uint32_t)plSetLayouts.size();
        plci.pSetLayouts = plSetLayouts.data();
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(app->getDevice(), &plci, nullptr, &w.layout) != VK_SUCCESS)
            throw std::runtime_error("RayTracingRenderer: failed to create pipeline layout");
        ci.layout = w.layout;

        VkPipeline pipe = VK_NULL_HANDLE;
        if (ctx_.dispatch.vkCreateRayTracingPipelinesKHR(app->getDevice(), VK_NULL_HANDLE, app->getPipelineCache(),
                1, &ci, nullptr, &pipe) != VK_SUCCESS)
            throw std::runtime_error("RayTracingRenderer: failed to create ray tracing pipeline");
        w.pipeline = pipe;
        app->resources.addPipeline(pipe, "RayTracingRenderer: rt pipeline");

        // SBT: raygen(0) + miss(1) + hit groups (2,3,4).
        w.sbt.init(ctx_, pipe, rtProps.shaderGroupHandleSize,
                   rtProps.shaderGroupBaseAlignment, rtProps.shaderGroupHandleAlignment);
        w.sbt.setRaygen(0);
        w.sbt.addMiss(1);
        w.sbt.addHitGroup(2);
        w.sbt.addHitGroup(3);
        w.sbt.addHitGroup(4);
        w.sbt.build();
    }
}

void RayTracingRenderer::createSoftDescriptorSetLayout(VulkanApp* app) {
    // Software fallback uses compute, not ray tracing. Bindings match the soft_*.comp shaders:
    // set 0: 0=UBO, 6=SkyUBO, 15=vertsSolid, 16=idxSolid, 17=vertsWater, 18=idxWater, 19=chunkInfos
    // set 1: 0=storage image output
    auto make = [&](uint32_t b, VkDescriptorType t) {
        VkDescriptorSetLayoutBinding bd{};
        bd.binding = b; bd.descriptorCount = 1; bd.descriptorType = t;
        bd.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        return bd;
    };
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(make(0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)); // ubo
    bindings.push_back(make(6,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)); // sky
    bindings.push_back(make(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    bindings.push_back(make(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    bindings.push_back(make(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    bindings.push_back(make(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    bindings.push_back(make(19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    std::vector<VkDescriptorBindingFlags> bflags(bindings.size(), VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    VkDescriptorSetLayoutBindingFlagsCreateInfo bflagsInfo{};
    bflagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bflagsInfo.bindingCount = (uint32_t)bflags.size();
    bflagsInfo.pBindingFlags = bflags.data();
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.pNext = &bflagsInfo;
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    li.bindingCount = (uint32_t)bindings.size();
    li.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(app->getDevice(), &li, nullptr, &softDescLayout_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create soft descriptor set layout");
    app->resources.addDescriptorSetLayout(softDescLayout_, "RayTracingRenderer: softDescLayout");

    VkDescriptorSetLayoutBinding outB{};
    outB.binding = 0; outB.descriptorCount = 1; outB.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorBindingFlags outFlag = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo outFlagsInfo{};
    outFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    outFlagsInfo.bindingCount = 1;
    outFlagsInfo.pBindingFlags = &outFlag;
    VkDescriptorSetLayoutCreateInfo oli{};
    oli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    oli.pNext = &outFlagsInfo;
    oli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    oli.bindingCount = 1; oli.pBindings = &outB;
    if (vkCreateDescriptorSetLayout(app->getDevice(), &oli, nullptr, &softOutputLayout_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create soft output layout");

    std::array<VkDescriptorPoolSize, 3> poolSizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 },
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pi.maxSets = 8;
    pi.poolSizeCount = (uint32_t)poolSizes.size();
    pi.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(app->getDevice(), &pi, nullptr, &softPool_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create soft descriptor pool");

    softDescSet_ = app->createDescriptorSet(softDescLayout_, softPool_);
    for (uint32_t i = 0; i < (uint32_t)RtWorkload::Count; ++i) softOutputSets_[i] = app->createDescriptorSet(softOutputLayout_, softPool_);
}

void RayTracingRenderer::createSoftPipelines(VulkanApp* app) {
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0; pc.size = sizeof(uint32_t);
    std::array<VkDescriptorSetLayout, 2> layouts = { softDescLayout_, softOutputLayout_ };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = (uint32_t)layouts.size();
    plci.pSetLayouts = layouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(app->getDevice(), &plci, nullptr, &softPipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to create soft pipeline layout");

    auto makePipe = [&](const char* path, VkPipeline& out) {
        VkShaderModule mod = app->getOrCreateShaderModule(path);
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = stage;
        ci.layout = softPipelineLayout_;
        if (vkCreateComputePipelines(app->getDevice(), app->getPipelineCache(), 1, &ci, nullptr, &out) != VK_SUCCESS)
            throw std::runtime_error(std::string("RayTracingRenderer: failed to create soft pipeline ") + path);
        app->resources.addPipeline(out, path);
    };
    makePipe("shaders/raytracing/soft_render.comp.spv", softRenderPipeline_);
    makePipe("shaders/raytracing/soft_shadow.comp.spv", softShadowPipeline_);
}

void RayTracingRenderer::updateSoftDescriptors(VulkanApp* app) {
    if (!useSoftware_ || softDescSet_ == VK_NULL_HANDLE) return;
    // Copy UBO and SkyUBO from the main descriptor set (bindings 0 and 6) — same as HW path
    std::vector<VkCopyDescriptorSet> copies;
    for (uint32_t b : {0u, 6u}) {
        VkCopyDescriptorSet c{};
        c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        c.srcSet = app->getMainDescriptorSet();
        c.srcBinding = b;
        c.dstSet = softDescSet_;
        c.dstBinding = b;
        c.descriptorCount = 1;
        copies.push_back(c);
    }
    vkUpdateDescriptorSets(app->getDevice(), 0, nullptr, (uint32_t)copies.size(), copies.data());

    // Bind geometry buffers if available
    auto bindBuf = [&](uint32_t binding, VkBuffer buf) {
        if (buf == VK_NULL_HANDLE) return;
        VkDescriptorBufferInfo bi{};
        bi.buffer = buf; bi.offset = 0; bi.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = softDescSet_; w.dstBinding = binding;
        w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(app->getDevice(), 1, &w, 0, nullptr);
    };
    bindBuf(15, softSolidVertex_);
    bindBuf(16, softSolidIndex_);
    bindBuf(17, softWaterVertex_);
    bindBuf(18, softWaterIndex_);
    if (softChunkInfoBuffer_.buffer != VK_NULL_HANDLE) bindBuf(19, softChunkInfoBuffer_.buffer);
}

void RayTracingRenderer::traceSoftWorkload(VkCommandBuffer cmd, RtWorkload w, VkImageView outputView) {
    if (!inited_ || !useSoftware_) return;
    VkPipeline pipe = (w == RtWorkload::Shadow) ? softShadowPipeline_ : softRenderPipeline_;
    if (pipe == VK_NULL_HANDLE) return;
    // Update output image descriptor
    VkDescriptorImageInfo ii{};
    ii.imageView = outputView; ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorSet outSet = softOutputSets_[(uint32_t)w];
    VkWriteDescriptorSet dw{};
    dw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dw.dstSet = outSet; dw.dstBinding = 0; dw.descriptorCount = 1;
    dw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; dw.pImageInfo = &ii;
    vkUpdateDescriptorSets(ctx_.device(), 1, &dw, 0, nullptr);

    VkDescriptorSet sets[2] = { softDescSet_, outSet };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, softPipelineLayout_, 0, 2, sets, 0, nullptr);
    uint32_t chunkCount = 0;
    { std::lock_guard<std::mutex> lk(chunksMutex_); chunkCount = (uint32_t)chunks_.size(); }
    vkCmdPushConstants(cmd, softPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &chunkCount);
    uint32_t wgX = (ctx_.app->getWidth() + 7) / 8;
    uint32_t wgY = (ctx_.app->getHeight() + 7) / 8;
    vkCmdDispatch(cmd, wgX, wgY, 1);
}

void RayTracingRenderer::registerChunk(uint64_t chunkId, GeometryKind kind,
                                      VkDeviceAddress vertexAddress, uint32_t vertexCount,
                                      VkDeviceAddress indexAddress, uint32_t indexCount,
                                      VkGeometryFlagsKHR geometryFlags,
                                      uint32_t baseVertex, uint32_t firstIndex,
                                      glm::vec3 aabbMin, glm::vec3 aabbMax) {
    if (!inited_) return;
    std::lock_guard<std::mutex> lk(chunksMutex_);
    ChunkEntry e;
    e.kind = kind;
    e.pending = !useSoftware_; // no BLAS build needed for software
    e.geom.vertexAddress = vertexAddress;
    e.geom.vertexCount = vertexCount;
    e.geom.vertexStride = 64; // sizeof(Vertex)
    e.geom.indexAddress = indexAddress;
    e.geom.indexCount = indexCount;
    e.geom.indexType = VK_INDEX_TYPE_UINT32;
    e.geom.geometryFlags = geometryFlags;
    e.baseVertex = baseVertex;
    e.firstIndex = firstIndex;
    e.aabbMin = aabbMin;
    e.aabbMax = aabbMax;
    // Replace any existing entry; the old BLAS is destroyed when the new one is
    // submitted (see update()).
    chunks_[chunkId] = std::move(e);
    tlasDirty_ = true;
    if (useSoftware_) softChunkInfoDirty_ = true;
}

void RayTracingRenderer::unregisterChunk(uint64_t chunkId) {
    if (!inited_) return;
    std::lock_guard<std::mutex> lk(chunksMutex_);
    auto it = chunks_.find(chunkId);
    if (it == chunks_.end()) return;
    if (it->second.blas) {
        auto old = std::move(it->second.blas);
        // Old BLAS is referenced only by the previous TLAS (in-flight frames);
        // defer teardown until all in-flight frames have retired.
        VulkanApp* app = ctx_.app;
                app->deferDestroyUntilAllPending(
                    [o = std::shared_ptr<AccelerationStructure>(std::move(old)), app]() { o->teardown(app); });
    }
    chunks_.erase(it);
    tlasDirty_ = true;
    if (useSoftware_) softChunkInfoDirty_ = true;
}

void RayTracingRenderer::registerVegetation(VulkanApp* app, const VegetationRenderer* veg) {
    if (!inited_ || !veg) return;

    // Cache the shared billboard cross-quad geometry addresses so update() can
    // build a single BLAS from them once they exist. Only mark the BLAS dirty
    // when the addresses actually change (first registration, or a rebuild of
    // the vegetation geometry), so we don't rebuild it every frame.
    const VertexBufferObject& vbo = veg->getBillboardVBO();
    if (vbo.vertexBuffer.buffer != VK_NULL_HANDLE && vbo.indexBuffer.buffer != VK_NULL_HANDLE) {
        VkBufferDeviceAddressInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        ai.buffer = vbo.vertexBuffer.buffer;
        VkDeviceAddress vaddr = vkGetBufferDeviceAddress(app->getDevice(), &ai);
        ai.buffer = vbo.indexBuffer.buffer;
        VkDeviceAddress iaddr = vkGetBufferDeviceAddress(app->getDevice(), &ai);
        if (vaddr != vegLastVertexAddress_ || iaddr != vegLastIndexAddress_) {
            vegVertexAddress_ = vaddr;
            vegIndexAddress_ = iaddr;
            vegVertexCount_ = 24;   // 6 planes × 4 corners
            vegIndexCount_  = 36;   // 6 planes × 2 triangles × 3
            vegVertexStride_ = 64;  // sizeof(Vertex)
            vegIndexType_    = VK_INDEX_TYPE_UINT32;
            vegBlasDirty_ = true;
            vegLastVertexAddress_ = vaddr;
            vegLastIndexAddress_ = iaddr;
        }
    }

    const uint64_t generation = veg->getVegetationGeneration();
    const bool geometryChanged = (generation != vegLastGeneration_) || vegInstances_.empty();
    // Skip the expensive GPU readback + TLAS rebuild unless the billboard set
    // actually changed (new consolidation) or the shared BLAS was rebuilt. This
    // keeps the common steady-state (no vegetation edits) free of per-frame
    // rebuilds; tlasDirty_ is left as-is (false) in that case.
    if (!geometryChanged && !vegBlasDirty_)
        return;
    vegLastGeneration_ = generation;

    if (geometryChanged) {
        // Read back the consolidated billboard instance buffer (GPU → CPU) and
        // emit one transformed TLAS instance per billboard. A negative w is a
        // generator sentinel (skipped biome / steep slope) and is dropped.
        vegInstances_.clear();
        const Buffer& inst = veg->getConcatenatedInstanceBuffer();
        const size_t count = veg->getConcatenatedInstanceCount();
        if (inst.buffer != VK_NULL_HANDLE && count > 0) {
            VkDeviceSize size = count * sizeof(glm::vec4);
            Buffer staging = app->createBuffer(size,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                VkBufferCopy region{};
                region.srcOffset = 0; region.dstOffset = 0; region.size = size;
                vkCmdCopyBuffer(cmd, inst.buffer, staging.buffer, 1, &region);
            });
            const glm::vec4* data = static_cast<const glm::vec4*>(staging.map(0));
            if (data) {
                const float scale = veg->getBillboardScale();
                for (size_t i = 0; i < count; ++i) {
                    glm::vec4 d = data[i];
                    if (d.w < 0.0f) continue;
                    const uint32_t layer = static_cast<uint32_t>(glm::floor(d.w));
                    const float theta = glm::fract(d.w) * 6.28318530718f;
                    // World transform: translate(worldPos) * rotateY(theta) * scale.
                    // (Per-instance height variation is approximated away; the RT
                    // billboards use a uniform scale, slightly differing in size
                    // from the rasterized ones which apply a per-position factor.)
                    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(d.x, d.y, d.z))
                                * glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0.0f, 1.0f, 0.0f))
                                * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
                    VegInstance vi{};
                    // VkTransformMatrixKHR is row-major[4][3]; glm is column-major.
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            vi.transform.matrix[r][c] = M[c][r];
                    vi.transform.matrix[3][0] = M[0][3];
                    vi.transform.matrix[3][1] = M[1][3];
                    vi.transform.matrix[3][2] = M[2][3];
                    vi.customIndex = static_cast<uint32_t>(GeometryKind::Vegetation) | (layer << 8);
                    vi.sbtOffset = static_cast<uint32_t>(GeometryKind::Vegetation);
                    // FORCE_OPAQUE: the alpha test is performed in the closest-hit
                    // shader (which reports occlusion for shadow rays), so the any-hit
                    // stage is skipped. FACING_CULL_DISABLE keeps both faces of each
                    // billboard plane able to occlude.
                    vi.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR
                             | VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                    vegInstances_.push_back(vi);
                }
            }
            app->destroyBuffer(staging);
        }
    }
    tlasDirty_ = true;
}

void RayTracingRenderer::setVegetationOpacity(VulkanApp* app, VkImageView view, VkSampler sampler) {
    if (!inited_ || view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
    if (useSoftware_ || rtDescriptorSet_ == VK_NULL_HANDLE) return;
    vegOpacityBound_ = true;
    VkDescriptorImageInfo ii{};
    ii.imageView = view;
    ii.sampler = sampler;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = rtDescriptorSet_;
    w.dstBinding = 14; w.dstArrayElement = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(app->getDevice(), 1, &w, 0, nullptr);
}

void RayTracingRenderer::setSoftwareGeometryBuffers(VkBuffer solidVertex, VkBuffer solidIndex,
                                                    VkBuffer waterVertex, VkBuffer waterIndex) {
    if (!useSoftware_) return;
    bool changed = (softSolidVertex_ != solidVertex || softSolidIndex_ != solidIndex ||
                    softWaterVertex_ != waterVertex || softWaterIndex_ != waterIndex);
    softSolidVertex_ = solidVertex;
    softSolidIndex_ = solidIndex;
    softWaterVertex_ = waterVertex;
    softWaterIndex_ = waterIndex;
    if (changed && inited_) {
        // Defer descriptor update to next update() call
    }
}

void RayTracingRenderer::update(VulkanApp* app) {
    if (!inited_) return;

    if (useSoftware_) {
        // Refresh UBO/Sky for compute shaders
        {
            VkCopyDescriptorSet c{};
            c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
            c.srcSet = app->getMainDescriptorSet();
            c.srcBinding = 0; c.srcArrayElement = 0;
            c.dstSet = softDescSet_;
            c.dstBinding = 0; c.dstArrayElement = 0;
            c.descriptorCount = 1;
            vkUpdateDescriptorSets(app->getDevice(), 0, nullptr, 1, &c);
        }
        {
            VkCopyDescriptorSet c{};
            c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
            c.srcSet = app->getMainDescriptorSet();
            c.srcBinding = 6; c.srcArrayElement = 0;
            c.dstSet = softDescSet_;
            c.dstBinding = 6; c.dstArrayElement = 0;
            c.descriptorCount = 1;
            vkUpdateDescriptorSets(app->getDevice(), 0, nullptr, 1, &c);
        }
        if (softChunkInfoDirty_) {
            std::vector<SoftChunkInfo> infos;
            {
                std::lock_guard<std::mutex> lk(chunksMutex_);
                infos.reserve(chunks_.size());
                for (auto& kv : chunks_) {
                    SoftChunkInfo info{};
                    info.baseVertex = kv.second.baseVertex;
                    info.firstIndex = kv.second.firstIndex;
                    info.indexCount = kv.second.geom.indexCount;
                    info.kind = (uint32_t)kv.second.kind;
                    info.aabbMin = kv.second.aabbMin;
                    info.aabbMax = kv.second.aabbMax;
                    infos.push_back(info);
                }
            }
            if (!infos.empty()) {
                VkDeviceSize size = infos.size() * sizeof(SoftChunkInfo);
                Buffer staging = app->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                memcpy(staging.mappedData, infos.data(), (size_t)size);
                // Ensure previous compute dispatches that read the buffer are done
                app->runSingleTimeCommands([&](VkCommandBuffer cmd){
                    VkBufferCopy r{0,0,size};
                    vkCmdCopyBuffer(cmd, staging.buffer, softChunkInfoBuffer_.buffer, 1, &r);
                });
                app->destroyBuffer(staging);
                // Also update the descriptor if buffer changed (handle same, no need)
            }
            softChunkInfoDirty_ = false;
        }
        updateSoftDescriptors(app);
        return;
    }

    // Refresh the per-frame UBO binding (0) from the current main descriptor set
    // so the RT shaders read up-to-date camera/light state. Bindings 1-13 are
    // static and were copied once at init from the static descriptor set.
    {
        VkCopyDescriptorSet c{};
        c.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        c.srcSet = app->getMainDescriptorSet();
        c.srcBinding = 0; c.srcArrayElement = 0;
        c.dstSet = rtDescriptorSet_;
        c.dstBinding = 0; c.dstArrayElement = 0;
        c.descriptorCount = 1;
        vkUpdateDescriptorSets(app->getDevice(), 0, nullptr, 1, &c);
    }

    bool havePending = false;
    {
        std::lock_guard<std::mutex> lk(chunksMutex_);
        for (auto& kv : chunks_) if (kv.second.pending) { havePending = true; break; }
    }
    if (!havePending && !tlasDirty_) return;

    // Record all pending BLAS builds + (if dirty) the TLAS rebuild into ONE
    // command buffer so the TLAS build is ordered AFTER every BLAS build it
    // references (no extra barriers / semaphores needed between them).
    VkCommandBuffer cmd = app->allocatePrimaryCommandBuffer();
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    // Build / rebuild pending chunk BLASes.
    {
        std::lock_guard<std::mutex> lk(chunksMutex_);
        for (auto& kv : chunks_) {
            ChunkEntry& e = kv.second;
            if (!e.pending) continue;
            e.pending = false;
            auto old = std::move(e.blas);
            auto blas = std::make_unique<AccelerationStructure>();
            blas->buildBlas(app, cmd, ctx_, { e.geom });
            e.blas = std::move(blas);
            if (old) {
                // Old BLAS referenced by previous TLAS only; safe to free once all
                // in-flight frames have completed.
        app->deferDestroyUntilAllPending(
            [o = std::shared_ptr<AccelerationStructure>(std::move(old)), app]() { o->teardown(app); });
            }
        }
    }

    // Build the shared vegetation BLAS (billboard cross-quad) once its geometry
    // addresses are known. Non-opaque so the any-hit alpha test is invoked.
    if (vegBlasDirty_ && vegVertexAddress_ != 0 && vegIndexAddress_ != 0) {
        BlasGeometryInput g{};
        g.vertexAddress = vegVertexAddress_;
        g.vertexCount = vegVertexCount_;
        g.vertexStride = vegVertexStride_;
        g.indexAddress = vegIndexAddress_;
        g.indexCount = vegIndexCount_;
        g.indexType = vegIndexType_;
        g.geometryFlags = 0; // non-opaque → any-hit runs
        auto old = std::move(vegBlas_);
        vegBlas_ = std::make_unique<AccelerationStructure>();
        vegBlas_->buildBlas(app, cmd, ctx_, { g });
        if (old) {
            app->deferDestroyUntilAllPending(
                [o = std::shared_ptr<AccelerationStructure>(std::move(old)), app]() { o->teardown(app); });
        }
        vegBlasDirty_ = false;
    }

    // Rebuild TLAS only when the instance set changed (add/remove/replace chunk).
    if (tlasDirty_) {
        buildTlas(app, cmd); // records into cmd and (re)builds tlas_
        tlasDirty_ = false;
    }

    // Submit the BLAS/TLAS build and block until it completes. The ray-traced
    // shadow dispatch in this frame's command buffer reads the (re)built TLAS, so
    // it must be finished before the frame is recorded/submitted. A blocking
    // submit-and-wait is both correct and simpler than a cross-submit wait
    // semaphore (which would require the build to be signaled on a separate
    // queue and risked VUID-vkQueueSubmit2-semaphore-03873 if the async submit
    // ever failed before signaling). Rebuilds only occur when chunks change, so
    // the stall is bounded and infrequent.
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("RayTracingRenderer: failed to end AS-build command buffer");
    app->submitCommandBufferAndWait(cmd);
    // The CB is freed by the submit machinery once it completes.
}

void RayTracingRenderer::buildTlas(VulkanApp* app, VkCommandBuffer cmd) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    {
        std::lock_guard<std::mutex> lk(chunksMutex_);
        instances.reserve(chunks_.size() + vegInstances_.size());
        for (auto& kv : chunks_) {
            ChunkEntry& e = kv.second;
            if (!e.blas || !e.blas->valid()) continue;
            VkAccelerationStructureInstanceKHR inst{};
            inst.transform = kIdentityTransform;
            // instanceCustomIndex encodes geometry kind in the low byte.
            inst.instanceCustomIndex = static_cast<uint32_t>(e.kind);
            inst.mask = 0xFF; // visible to all ray types; filtering is per-pipeline
            inst.instanceShaderBindingTableRecordOffset = static_cast<uint32_t>(e.kind);
            inst.flags = (e.kind == GeometryKind::Vegetation)
                             ? 0 // allow any-hit (alpha test)
                             : VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
            inst.accelerationStructureReference = e.blas->deviceAddress();
            instances.push_back(inst);
        }
        // Vegetation: many transformed instances sharing one BLAS.
        if (vegBlas_ && ctx_.supported()) {
            for (const auto& vi : vegInstances_) {
                VkAccelerationStructureInstanceKHR inst{};
                inst.transform = vi.transform;
                inst.instanceCustomIndex = vi.customIndex;
                inst.mask = 0xFF;
                inst.instanceShaderBindingTableRecordOffset = vi.sbtOffset;
                inst.flags = vi.flags;
                inst.accelerationStructureReference = vegBlas_->deviceAddress();
                instances.push_back(inst);
            }
        }
    }

    auto old = std::move(tlas_);
    tlas_ = std::make_unique<AccelerationStructure>();
    tlas_->buildTlas(app, cmd, ctx_, instances, /*update=*/false);
    if (old) {
                app->deferDestroyUntilAllPending(
                    [o = std::shared_ptr<AccelerationStructure>(std::move(old)), app]() { o->teardown(app); });
    }

    // Bind the new TLAS into the RT descriptor set (binding 20).
    VkAccelerationStructureKHR handle = tlas_->handle();
    VkWriteDescriptorSetAccelerationStructureKHR was{};
    was.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    was.accelerationStructureCount = 1;
    was.pAccelerationStructures = &handle;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.pNext = &was;
    w.dstSet = rtDescriptorSet_;
    w.dstBinding = 20; w.dstArrayElement = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(app->getDevice(), 1, &w, 0, nullptr);
}

// Record a ray-tracing dispatch for one workload. The output `view` must already
// be in VK_IMAGE_LAYOUT_GENERAL. Reuses the shared TLAS (set=0) and binds the
// per-workload output image (set=1). The ray behavior (shadow/reflection/
// refraction) is selected by the push-constant mode consumed by the shaders.
void RayTracingRenderer::traceWorkload(VkCommandBuffer cmd, RtWorkload wk, uint32_t mode,
                                       VkImageView outputView) {
    if (!inited_) return;
    if (useSoftware_) {
        // Software brute-force: Render/Shadow use dedicated compute pipelines;
        // Reflection/Refraction fall back to Render (single bounce already includes them)
        RtWorkload sw = wk;
        if (wk == RtWorkload::Reflection || wk == RtWorkload::Refraction) sw = RtWorkload::Render;
        traceSoftWorkload(cmd, sw, outputView);
        return;
    }
    if (!tlas_) return;
    auto& w = workloads_[(uint32_t)wk];
    VkDevice device = ctx_.app->getDevice();

    VkDescriptorImageInfo ii{};
    ii.imageView = outputView;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet dw{};
    dw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dw.dstSet = rtOutputSets_[(uint32_t)wk];
    dw.dstBinding = 0; dw.dstArrayElement = 0;
    dw.descriptorCount = 1;
    dw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    dw.pImageInfo = &ii;
    vkUpdateDescriptorSets(device, 1, &dw, 0, nullptr);

    VkDescriptorSet sets[2] = { rtDescriptorSet_, rtOutputSets_[(uint32_t)wk] };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, w.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, w.layout, 0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, w.layout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
        0, sizeof(uint32_t), &mode);

    VkStridedDeviceAddressRegionKHR rg = w.sbt.raygenRegion();
    VkStridedDeviceAddressRegionKHR ms = w.sbt.missRegion();
    VkStridedDeviceAddressRegionKHR hg = w.sbt.hitGroupRegion();
    VkStridedDeviceAddressRegionKHR cl = w.sbt.callableRegion();
    ctx_.dispatch.vkCmdTraceRaysKHR(cmd, &rg, &ms, &hg, &cl,
                      ctx_.app->getWidth(), ctx_.app->getHeight(), 1);
}

void RayTracingRenderer::traceShadow(VkCommandBuffer cmd, VkImageView shadowOutputView) {
    traceWorkload(cmd, RtWorkload::Shadow, 0, shadowOutputView);
}
void RayTracingRenderer::traceReflection(VkCommandBuffer cmd, VkImageView outputView) {
    traceWorkload(cmd, RtWorkload::Reflection, 1, outputView);
}
void RayTracingRenderer::traceRefraction(VkCommandBuffer cmd, VkImageView outputView) {
    traceWorkload(cmd, RtWorkload::Refraction, 2, outputView);
}
void RayTracingRenderer::traceRender(VkCommandBuffer cmd, VkImageView outputView) {
    // Primary render: rayType is carried in the payload (set to 0 in render.rgen),
    // so no push-constant mode is needed here.
    traceWorkload(cmd, RtWorkload::Render, 0, outputView);
}
