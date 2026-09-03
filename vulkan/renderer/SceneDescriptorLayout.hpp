#pragma once

#include <vulkan/vulkan.h>
#include <vector>

class VulkanApp;

// Scene-level descriptor set layouts and descriptor sets.
//
// This bundles the descriptor-set *layout* that describes the application's
// scene resources (per-frame UBO, texture/material arrays, shadow cascades,
// sky UBO, water params, environment cubemap, roughness/AO maps) together with
// the per-frame main descriptor sets and the static descriptor set.
//
// It is intentionally owned by the application (MyApp), not by the generic
// VulkanApp framework, so VulkanApp stays agnostic about *what* is rendered and
// *how* the scene bindings are laid out.
class SceneDescriptorLayout {
public:
    // Build the layouts and allocate the descriptor sets. Must be called after
    // the descriptor pool exists (VulkanApp::createDescriptorPool) and before
    // any renderer writes the static descriptor set.
    void create(VulkanApp& app);

    VkDescriptorSetLayout descriptorSetLayout() const { return descriptorSetLayout_; }
    VkDescriptorSetLayout materialDescriptorSetLayout() const { return materialDescriptorSetLayout_; }
    VkDescriptorSetLayout brushDepthDescriptorSetLayout() const { return brushDepthDescriptorSetLayout_; }
    VkDescriptorSet staticDescriptorSet() const { return staticDescriptorSet_; }

    // Descriptor-buffer query layout: identical bindings to descriptorSetLayout_
    // plus VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT. Used ONLY
    // for vkGetDescriptorSetLayoutSizeEXT / vkGetDescriptorSetLayoutBindingOffsetEXT
    // queries (both VUID-require the bit). Never in pipeline layouts, never for
    // vkAllocateDescriptorSets. VK_NULL_HANDLE when !useDescriptorBuffer().
    VkDescriptorSetLayout descriptorBufferQueryLayout() const { return descriptorBufferQueryLayout_; }

    // Whether set-0 binding goes through descriptor buffers. False in Phase 1:
    // flipping the main layout to DESCRIPTOR_BUFFER_BIT would invalidate every
    // classic vkCmdBindDescriptorSets of set 0 (VUID-08010) and classic set-1/2
    // binds in the same draws — that cutover ships with the set-1/set-2
    // migration. While false, descriptor buffers are maintained as warm mirrors
    // (written on the same events) alongside the classic sets.
    bool descriptorBufferBindActive() const { return mainLayoutDescriptorBufferCapable_; }

    // Per-frame main descriptor set (round-robin by currentFrame).
    VkDescriptorSet getMainDescriptorSet(uint32_t currentFrame) const {
        if (mainDescriptorSets_.empty()) return VK_NULL_HANDLE;
        return mainDescriptorSets_[currentFrame % static_cast<uint32_t>(mainDescriptorSets_.size())];
    }
    VkDescriptorSet getMainDescriptorSetForFrame(uint32_t idx) const {
        if (idx >= mainDescriptorSets_.size()) return VK_NULL_HANDLE;
        return mainDescriptorSets_[idx];
    }
    size_t getMainDescriptorSetCount() const { return mainDescriptorSets_.size(); }

private:
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout brushDepthDescriptorSetLayout_ = VK_NULL_HANDLE;
    // Query-only duplicate of descriptorSetLayout_ with DESCRIPTOR_BUFFER_BIT.
    VkDescriptorSetLayout descriptorBufferQueryLayout_ = VK_NULL_HANDLE;
    // True once pipeline layouts + all set-0 bind sites use the descriptor
    // buffer path (then classic set-0 writes/binds stop). False in Phase 1.
    bool mainLayoutDescriptorBufferCapable_ = false;
    VkDescriptorSet staticDescriptorSet_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> mainDescriptorSets_;
};
