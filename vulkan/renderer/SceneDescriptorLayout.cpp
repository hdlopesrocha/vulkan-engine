#include "SceneDescriptorLayout.hpp"

#include <array>
#include <stdexcept>

#include "../VulkanApp.hpp"

void SceneDescriptorLayout::create(VulkanApp& app) {
    // binding 0 : uniform buffer (vertex shader)
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    // UBO is referenced by vertex, fragment, tessellation, and geometry stages
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT;

    // bindings 1..3: arrays of combined image samplers (albedo / normal / height)
    // bindings 1..3: one combined image sampler each (we use a texture2D array as the image view)
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    VkDescriptorSetLayoutBinding normalSamplerBinding{};
    normalSamplerBinding.binding = 2;
    normalSamplerBinding.descriptorCount = 1;
    normalSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalSamplerBinding.pImmutableSamplers = nullptr;
    normalSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding heightSamplerBinding{};
    heightSamplerBinding.binding = 3;
    heightSamplerBinding.descriptorCount = 1;
    heightSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    heightSamplerBinding.pImmutableSamplers = nullptr;
    // Height sampler is used by fragment shader and tessellation evaluation shader (for displacement)
    heightSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // binding 4: shadow map sampler
    VkDescriptorSetLayoutBinding shadowSamplerBinding{};
    shadowSamplerBinding.binding = 4;
    shadowSamplerBinding.descriptorCount = 1;
    shadowSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowSamplerBinding.pImmutableSamplers = nullptr;
    shadowSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 6: Sky UBO
    VkDescriptorSetLayoutBinding skyBinding{};
    skyBinding.binding = 6;
    skyBinding.descriptorCount = 1;
    skyBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    skyBinding.pImmutableSamplers = nullptr;
    skyBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 7: Water params SSBO (for water shader) - use storage buffer like Materials
    VkDescriptorSetLayoutBinding waterParamsBinding{};
    waterParamsBinding.binding = 7;
    waterParamsBinding.descriptorCount = 1;
    waterParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    waterParamsBinding.pImmutableSamplers = nullptr;
    // Make the water params visible to fragment, tessellation evaluation, and tessellation control shaders
    waterParamsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    // Per-instance / per-draw descriptor set uses bindings: 0 (UBO), 1..3 (samplers), 4 (shadow cascade 0),
    // 5 (Materials SSBO), 6 (Sky UBO), 7 (water params), 8 (shadow cascade 1), 9 (shadow cascade 2)
    // Note: Materials (binding 5) is declared in shaders as set=0 binding=5, so include it in the main layout.

    // binding 8: shadow map cascade 1
    VkDescriptorSetLayoutBinding shadowCascade1Binding{};
    shadowCascade1Binding.binding = 8;
    shadowCascade1Binding.descriptorCount = 1;
    shadowCascade1Binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowCascade1Binding.pImmutableSamplers = nullptr;
    shadowCascade1Binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 9: shadow map cascade 2
    VkDescriptorSetLayoutBinding shadowCascade2Binding{};
    shadowCascade2Binding.binding = 9;
    shadowCascade2Binding.descriptorCount = 1;
    shadowCascade2Binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowCascade2Binding.pImmutableSamplers = nullptr;
    shadowCascade2Binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 10: Water render UBO (time parameter for water shaders)
    VkDescriptorSetLayoutBinding waterRenderUBOBinding{};
    waterRenderUBOBinding.binding = 10;
    waterRenderUBOBinding.descriptorCount = 1;
    waterRenderUBOBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    waterRenderUBOBinding.pImmutableSamplers = nullptr;
    waterRenderUBOBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    // NOTE (hybrid RT): binding 11 (legacy 360° environment cubemap) was
    // REMOVED. Solid reflections and water reflection/refraction are now
    // hardware ray tracing (inline ray queries + water RT pipeline) with the
    // sky equirect as the miss fallback. The old Solid360 cubemap capture
    // passes, targets, sync and descriptors are deprecated (see SceneRenderer).

    // binding 12: roughness map array
    VkDescriptorSetLayoutBinding roughnessSamplerBinding{};
    roughnessSamplerBinding.binding = 12;
    roughnessSamplerBinding.descriptorCount = 1;
    roughnessSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    roughnessSamplerBinding.pImmutableSamplers = nullptr;
    roughnessSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 13: ambient occlusion map array
    VkDescriptorSetLayoutBinding aoSamplerBinding{};
    aoSamplerBinding.binding = 13;
    aoSamplerBinding.descriptorCount = 1;
    aoSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    aoSamplerBinding.pImmutableSamplers = nullptr;
    aoSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // ── Hybrid RT bindings (fragment stage: inline ray queries) ──────────
    // binding 14: TLAS (stable proxy BLAS; ray queries in main.frag/water.frag)
    VkDescriptorSetLayoutBinding tlasBinding{};
    tlasBinding.binding = 14;
    tlasBinding.descriptorCount = 1;
    tlasBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    tlasBinding.pImmutableSamplers = nullptr;
    tlasBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 15: RT water reflection output (half-res, sampled by water.frag)
    VkDescriptorSetLayoutBinding rtReflectBinding{};
    rtReflectBinding.binding = 15;
    rtReflectBinding.descriptorCount = 1;
    rtReflectBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rtReflectBinding.pImmutableSamplers = nullptr;
    rtReflectBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 16: RT water refraction + thickness output (rgb=color, a=thickness)
    VkDescriptorSetLayoutBinding rtRefractBinding{};
    rtRefractBinding.binding = 16;
    rtRefractBinding.descriptorCount = 1;
    rtRefractBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rtRefractBinding.pImmutableSamplers = nullptr;
    rtRefractBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 17: RT params UBO (toggles, distances, IOR, absorption, debug)
    VkDescriptorSetLayoutBinding rtParamsBinding{};
    rtParamsBinding.binding = 17;
    rtParamsBinding.descriptorCount = 1;
    rtParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    rtParamsBinding.pImmutableSamplers = nullptr;
    rtParamsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 18: RT proxy metadata (per-box albedo/flags for ray-query hit
    // shading in main.frag). Same buffer the RT pipeline samples at its set 0/4.
    VkDescriptorSetLayoutBinding rtMetaBinding{};
    rtMetaBinding.binding = 18;
    rtMetaBinding.descriptorCount = 1;
    rtMetaBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    rtMetaBinding.pImmutableSamplers = nullptr;
    rtMetaBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 19: previous-frame solid HDR color (screen-space reflection
    // refinement in main.frag — precise mirror samples of the real scene).
    VkDescriptorSetLayoutBinding ssrColorBinding{};
    ssrColorBinding.binding = 19;
    ssrColorBinding.descriptorCount = 1;
    ssrColorBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ssrColorBinding.pImmutableSamplers = nullptr;
    ssrColorBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 20: previous-frame solid depth (SSR march + occlusion test).
    VkDescriptorSetLayoutBinding ssrDepthBinding{};
    ssrDepthBinding.binding = 20;
    ssrDepthBinding.descriptorCount = 1;
    ssrDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ssrDepthBinding.pImmutableSamplers = nullptr;
    ssrDepthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 21: real scene-geometry primitive bases ([0]=count, [1..N]=first
    // primitive of geometry i) for the reflection BLAS (binary search).
    VkDescriptorSetLayoutBinding scenePrimBaseBinding{};
    scenePrimBaseBinding.binding = 21;
    scenePrimBaseBinding.descriptorCount = 1;
    scenePrimBaseBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    scenePrimBaseBinding.pImmutableSamplers = nullptr;
    scenePrimBaseBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 22: per-geometry average albedo (vec4 per chunk) for shading
    // real-geometry reflection hits.
    VkDescriptorSetLayoutBinding sceneMetaBinding{};
    sceneMetaBinding.binding = 22;
    sceneMetaBinding.descriptorCount = 1;
    sceneMetaBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneMetaBinding.pImmutableSamplers = nullptr;
    sceneMetaBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 23: per-geometry {baseVertex, firstIndex, primBase, 0} for real
    // triangle attribute fetches in hit shading.
    VkDescriptorSetLayoutBinding sceneGeomInfoBinding{};
    sceneGeomInfoBinding.binding = 23;
    sceneGeomInfoBinding.descriptorCount = 1;
    sceneGeomInfoBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneGeomInfoBinding.pImmutableSamplers = nullptr;
    sceneGeomInfoBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 24: merged vertex pool (float array; Vertex stride 16 floats,
    // position 0-2, normal 8-10) read by hit shading.
    VkDescriptorSetLayoutBinding sceneVertsBinding{};
    sceneVertsBinding.binding = 24;
    sceneVertsBinding.descriptorCount = 1;
    sceneVertsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneVertsBinding.pImmutableSamplers = nullptr;
    sceneVertsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 25: merged index pool (uint per index).
    VkDescriptorSetLayoutBinding sceneIndicesBinding{};
    sceneIndicesBinding.binding = 25;
    sceneIndicesBinding.descriptorCount = 1;
    sceneIndicesBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneIndicesBinding.pImmutableSamplers = nullptr;
    sceneIndicesBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding numbers are sparse by design: 11 (legacy 360 cubemap) is
    // intentionally absent.
    std::array<VkDescriptorSetLayoutBinding, 25> bindings = {
        uboLayoutBinding, samplerLayoutBinding, normalSamplerBinding, heightSamplerBinding,
        shadowSamplerBinding, /* material */ VkDescriptorSetLayoutBinding{}, skyBinding,
        waterParamsBinding, shadowCascade1Binding, shadowCascade2Binding, waterRenderUBOBinding,
        roughnessSamplerBinding, aoSamplerBinding,
        tlasBinding, rtReflectBinding, rtRefractBinding, rtParamsBinding, rtMetaBinding,
        ssrColorBinding, ssrDepthBinding, scenePrimBaseBinding, sceneMetaBinding,
        sceneGeomInfoBinding, sceneVertsBinding, sceneIndicesBinding
    };
    // Fill the material binding at position 5
    bindings[5].binding = 5;
    bindings[5].descriptorCount = 1;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].pImmutableSamplers = nullptr;
    bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // No update-after-bind bindings remain: the legacy cubemap binding 11
    // (the only UPDATE_AFTER_BIND binding, for swapchain-resize view churn)
    // is gone. RT views/TLAS are stable between resizes (rewritten only on
    // resize/recreate events, never while in flight).
    std::array<VkDescriptorBindingFlags, 25> bindingFlags{};
    bindingFlags.fill(0);

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(app.device, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // Register the main descriptor set layout for inspection/cleanup
    app.registerDescriptorSetLayout(descriptorSetLayout_, "SceneDescriptorLayout: descriptorSetLayout");

    // Allocate the main UBO/sampler/materials descriptor sets (one per frame)
    const uint32_t MAIN_DESC_SETS = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    mainDescriptorSets_.clear();
    mainDescriptorSets_.resize(MAIN_DESC_SETS);
    for (uint32_t i = 0; i < MAIN_DESC_SETS; ++i) {
        mainDescriptorSets_[i] = app.createDescriptorSet(descriptorSetLayout_);
    }

    // Allocate one static descriptor set for the scene-static bindings
    // (textures, materials, sky, water params, shadows). Written once in
    // SceneRenderer::init() and then copied into per-frame descriptor sets so
    // per-frame updates only touch binding 0 (UBO). RT bindings (14-17) are
    // written by SceneRenderer alongside (stable handles, resize-only updates).
    staticDescriptorSet_ = app.createDescriptorSet(descriptorSetLayout_);

    // Create a separate material descriptor layout used for materials only
    std::array<VkDescriptorSetLayoutBinding, 1> materialBindings = { bindings[5] };
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
    materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    materialLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialBindings.size());
    materialLayoutInfo.pBindings = materialBindings.data();

    if (vkCreateDescriptorSetLayout(app.device, &materialLayoutInfo, nullptr, &materialDescriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create material descriptor set layout!");
    }

    // Register material descriptor set layout
    app.registerDescriptorSetLayout(materialDescriptorSetLayout_, "SceneDescriptorLayout: materialDescriptorSetLayout");

    // ── Brush depth descriptor set layout (set=1, binding 0/1) ──
    // Separate from the main set so the shadow pass (which uses set=0 only)
    // doesn't need to reference these bindings. Only pipelines using main.frag
    // (graphicsPipeline, depthPrePassPipeline, deferredColorPipeline) include
    // this layout.
    std::array<VkDescriptorSetLayoutBinding, 2> brushDepthBindings{};
    brushDepthBindings[0].binding = 0;
    brushDepthBindings[0].descriptorCount = 1;
    brushDepthBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    brushDepthBindings[0].pImmutableSamplers = nullptr;
    brushDepthBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    brushDepthBindings[1].binding = 1;
    brushDepthBindings[1].descriptorCount = 1;
    brushDepthBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    brushDepthBindings[1].pImmutableSamplers = nullptr;
    brushDepthBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo brushDepthLayoutInfo{};
    brushDepthLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    brushDepthLayoutInfo.bindingCount = static_cast<uint32_t>(brushDepthBindings.size());
    brushDepthLayoutInfo.pBindings = brushDepthBindings.data();

    if (vkCreateDescriptorSetLayout(app.device, &brushDepthLayoutInfo, nullptr, &brushDepthDescriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create brush depth descriptor set layout!");
    }
    app.registerDescriptorSetLayout(brushDepthDescriptorSetLayout_, "SceneDescriptorLayout: brushDepthDescriptorSetLayout");

    // If we later add a normal map sampler (binding 2), extend bindings dynamically when required by the app.

    // ── Descriptor-buffer query layout (Phase 1) ──────────────────────────
    // Duplicate of the main set-0 bindings with DESCRIPTOR_BUFFER_BIT_EXT so
    // SceneRenderer can query the driver for the set size
    // (vkGetDescriptorSetLayoutSizeEXT) and per-binding offsets
    // (vkGetDescriptorSetLayoutBindingOffsetEXT) — both VUID-require the bit.
    // The MAIN layout deliberately keeps its classic flags: flipping it would
    // invalidate every classic vkCmdBindDescriptorSets of set 0 (VUID-08010)
    // and forbid mixing with the classic set-1/set-2 binds in the same draw.
    // That cutover (mainLayoutDescriptorBufferCapable_ = true + bind-site
    // migration) ships with the set-1/set-2 migration; until then this layout
    // is query-only (never in a pipeline layout, never used to allocate sets).
    if (app.useDescriptorBuffer()) {
#ifndef VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
#define VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT 0x00000010
#endif
        VkDescriptorSetLayoutCreateInfo queryInfo{};
        queryInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        // DESCRIPTOR_BUFFER_BIT must NOT be combined with
        // UPDATE_AFTER_BIND_POOL_BIT (VUID-flags-08002), and the
        // UPDATE_AFTER_BIND binding flag is meaningless for host-written
        // descriptor memory — so the query layout carries the DB bit alone
        // with no binding-flags pNext. This also matches the future cutover
        // main layout (direct host writes need no update-after-bind).
        queryInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        queryInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        queryInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(app.device, &queryInfo, nullptr, &descriptorBufferQueryLayout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor-buffer query layout!");
        }
        app.registerDescriptorSetLayout(descriptorBufferQueryLayout_, "SceneDescriptorLayout: descriptorBufferQueryLayout");
    }
}
