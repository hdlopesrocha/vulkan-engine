#pragma once

#include "VulkanApp.hpp"
#include "TextureManager.hpp"
#include "ShadowMapper.hpp"

class DescriptorSetBuilder {
public:
    DescriptorSetBuilder(VulkanApp* app, TextureManager* texMgr, ShadowMapper* shadowMapper);

    VkDescriptorSet createMainDescriptorSet(const TextureManager::Triple& tr, Buffer& uniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0);
    VkDescriptorSet createShadowDescriptorSet(const TextureManager::Triple& tr, Buffer& shadowUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0);
    VkDescriptorSet createSphereDescriptorSet(const TextureManager::Triple& tr, Buffer& sphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0);
    VkDescriptorSet createShadowSphereDescriptorSet(const TextureManager::Triple& tr, Buffer& shadowSphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer = nullptr, VkDeviceSize materialOffset = 0);

private:
    VulkanApp* app;
    TextureManager* texMgr;
    ShadowMapper* shadow;
};
