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
    VkDescriptorSet staticDescriptorSet_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> mainDescriptorSets_;
};
