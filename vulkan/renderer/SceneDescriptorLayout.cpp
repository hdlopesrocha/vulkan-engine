#include "SceneDescriptorLayout.hpp"

#include <array>
#include <stdexcept>

#include "../core/VulkanApp.hpp"

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
    // Make the water params visible to fragment, tessellation evaluation, and
    // tessellation control shaders. VERTEX is included for the WATER_NO_TESS
    // water vertex path (C1), which loads the per-vertex layer params directly.
    waterParamsBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

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
    // VERTEX is included for the WATER_NO_TESS water vertex path (C1), which
    // reads the water time from the vertex stage.
    waterRenderUBOBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    // NOTE: binding 11 (legacy 360° environment cubemap) was REMOVED. Solid
    // and water shading use the SDF raymarched surfaces plus the sky fallback;
    // the old Solid360 cubemap capture passes/targets/descriptors are gone.

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

    // binding 19: previous-frame solid HDR color (screen-space reflection
    // refinement in SolidRenderer.frag — precise mirror samples of the real scene).
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

    // binding 27: shared wind-field UBO (ambient + up to 4 tornadoes, packed
    // by VegetationRenderer). Read via shaders/includes/vegetation/WindField.glsl by
    // every consumer (vegetation vertex, fire, SDF, volumetric clouds) so all
    // sample the identical field. Appended after 26: binding 11 stays intentionally
    // absent (legacy 360 cubemap; load-bearing hole — descriptor-buffer
    // offsets, shadow-set comments and the RT-compaction note below assume
    // it), and appending keeps the binding indices stable.
    VkDescriptorSetLayoutBinding windFieldBinding{};
    windFieldBinding.binding = 27;
    windFieldBinding.descriptorCount = 1;
    windFieldBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    windFieldBinding.pImmutableSamplers = nullptr;
    windFieldBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        | VK_SHADER_STAGE_COMPUTE_BIT
        // Water waves sample the field in the tessellation stages (control
        // probe + evaluation displacement) so wind-driven waves match the
        // rasterized surface exactly.
        | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
        | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // binding 28: per-brush image source table (types/ImageType.hpp): one uvec2
    // {ImageType, index} per brush. Lets the shaders resolve which texture set
    // and layer a brush samples (base image / TextureMixer / video).
    VkDescriptorSetLayoutBinding brushMapBinding{};
    brushMapBinding.binding = 28;
    brushMapBinding.descriptorCount = 1;
    brushMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    brushMapBinding.pImmutableSamplers = nullptr;
    brushMapBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // Binding numbers are sparse by design: 11 (legacy 360 cubemap) is
    // intentionally absent.
    std::array<VkDescriptorSetLayoutBinding, 17> bindings = {
        uboLayoutBinding, samplerLayoutBinding, normalSamplerBinding, heightSamplerBinding,
        shadowSamplerBinding, /* material */ VkDescriptorSetLayoutBinding{}, skyBinding,
        waterParamsBinding, shadowCascade1Binding, shadowCascade2Binding, waterRenderUBOBinding,
        roughnessSamplerBinding, aoSamplerBinding,
        ssrColorBinding, ssrDepthBinding,
        windFieldBinding, brushMapBinding
    };
    // Fill the material binding at position 5
    bindings[5].binding = 5;
    bindings[5].descriptorCount = 1;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].pImmutableSamplers = nullptr;
    bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    const uint32_t layoutBindingCount = static_cast<uint32_t>(bindings.size());

    // No update-after-bind bindings remain: the legacy cubemap binding 11
    // (the only UPDATE_AFTER_BIND binding, for swapchain-resize view churn)
    // is gone. SSR views are the only resizable image bindings and they are
    // rewritten only while the device is idle.
    std::array<VkDescriptorBindingFlags, 28> bindingFlags{};
    bindingFlags.fill(0);

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = layoutBindingCount;
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = layoutBindingCount;
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
    // per-frame updates only touch binding 0 (UBO).
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

    // ── Brush SDF params descriptor set layout (set=1, binding 0) ──
    // The per-frame brush description UBO (vulkan/ubo/BrushSdfUBO.hpp). Shared
    // by the brush preview raymarcher (BrushSdf.frag, set=1 binding=0) and the
    // solid shader's PAINT/REMOVE intersection test (SolidSurface.glsl). Only
    // pipelines embedding SolidRenderer.frag / BrushSdf.frag include this
    // layout; the shadow pass uses set=0 only.
    VkDescriptorSetLayoutBinding brushParamsBinding{};
    brushParamsBinding.binding = 0;
    brushParamsBinding.descriptorCount = 1;
    brushParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    brushParamsBinding.pImmutableSamplers = nullptr;
    brushParamsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo brushParamsLayoutInfo{};
    brushParamsLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    brushParamsLayoutInfo.bindingCount = 1;
    brushParamsLayoutInfo.pBindings = &brushParamsBinding;

    if (vkCreateDescriptorSetLayout(app.device, &brushParamsLayoutInfo, nullptr, &brushParamsDescriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create brush params descriptor set layout!");
    }
    app.registerDescriptorSetLayout(brushParamsDescriptorSetLayout_, "SceneDescriptorLayout: brushParamsDescriptorSetLayout");

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
        queryInfo.bindingCount = layoutBindingCount;
        queryInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(app.device, &queryInfo, nullptr, &descriptorBufferQueryLayout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor-buffer query layout!");
        }
        app.registerDescriptorSetLayout(descriptorBufferQueryLayout_, "SceneDescriptorLayout: descriptorBufferQueryLayout");
    }
}
