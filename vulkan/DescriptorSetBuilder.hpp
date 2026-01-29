#pragma once

#include "VulkanApp.hpp"
#include "TextureTriple.hpp"
#include "ShadowRenderer.hpp"

class DescriptorSetBuilder {
public:
    DescriptorSetBuilder(VulkanApp* app, ShadowRenderer* shadowMapper);

    VkDescriptorSet createMainDescriptorSet(const Triple& tr, Buffer& uniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    VkDescriptorSet createShadowDescriptorSet(const Triple& tr, Buffer& shadowUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    VkDescriptorSet createSphereDescriptorSet(const Triple& tr, Buffer& sphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    VkDescriptorSet createShadowSphereDescriptorSet(const Triple& tr, Buffer& shadowSphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0, Buffer* instanceBuffer = nullptr);
    // (material binding is handled via a single global material descriptor set)

private:
    VulkanApp* app;
    // no longer depends on TextureManager; callers supply a Triple when creating sets
    ShadowRenderer* shadow;
};
