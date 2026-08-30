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

    // binding 11: 360° environment cubemap sampler for solid-shader reflections
    VkDescriptorSetLayoutBinding envMapBinding{};
    envMapBinding.binding = 11;
    envMapBinding.descriptorCount = 1;
    envMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    envMapBinding.pImmutableSamplers = nullptr;
    envMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

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

    std::array<VkDescriptorSetLayoutBinding, 14> bindings = {
        uboLayoutBinding, samplerLayoutBinding, normalSamplerBinding, heightSamplerBinding,
        shadowSamplerBinding, /* material */ VkDescriptorSetLayoutBinding{}, skyBinding,
        waterParamsBinding, shadowCascade1Binding, shadowCascade2Binding, waterRenderUBOBinding,
        envMapBinding, roughnessSamplerBinding, aoSamplerBinding
    };
    // Fill the material binding at position 5
    bindings[5].binding = 5;
    bindings[5].descriptorCount = 1;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].pImmutableSamplers = nullptr;
    bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // Binding flags — enable update-after-bind for binding 11 (cubemap environment map)
    // so that vkUpdateDescriptorSets can write binding 11 while a command buffer
    // referencing this descriptor set is still pending (the cubemap render path
    // swaps between a dummy cubemap and the real one every frame).
    std::array<VkDescriptorBindingFlags, 14> bindingFlags{};
    bindingFlags.fill(0);
    bindingFlags[11] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

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

    // Allocate one static descriptor set for bindings 1-13 (textures, materials, sky,
    // water params, cubemap). Written once in SceneRenderer::init() and then copied
    // into per-frame descriptor sets so per-frame updates only touch binding 0 (UBO).
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
}
