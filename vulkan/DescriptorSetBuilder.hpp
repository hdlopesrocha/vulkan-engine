#pragma once

#include "VulkanApp.hpp"
#include "TextureTriple.hpp"
#include "ShadowMapper.hpp"

class DescriptorSetBuilder {
public:
    DescriptorSetBuilder(VulkanApp* app, ShadowMapper* shadowMapper);

    VkDescriptorSet createMainDescriptorSet(const Triple& tr, Buffer& uniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    VkDescriptorSet createShadowDescriptorSet(const Triple& tr, Buffer& shadowUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    VkDescriptorSet createSphereDescriptorSet(const Triple& tr, Buffer& sphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    VkDescriptorSet createShadowSphereDescriptorSet(const Triple& tr, Buffer& shadowSphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    // Update material storage buffer binding for a collection of descriptor sets.
    void updateMaterialBinding(std::vector<VkDescriptorSet>& sets, Buffer& materialBuffer, VkDeviceSize elementSize);

private:
    VulkanApp* app;
    // no longer depends on TextureManager; callers supply a Triple when creating sets
    ShadowMapper* shadow;
};
