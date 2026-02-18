#include "SkySphere.hpp"
#include "../widgets/SkySettings.hpp"
#include "../Uniforms.hpp"
#include <glm/glm.hpp>

SkySphere::SkySphere() {}

SkySphere::~SkySphere() { cleanup(); }

void SkySphere::init(VulkanApp* app, SkySettings& settings,
                     VkDescriptorSet descriptorSet) {
    skySettings = &settings;
    VkDeviceSize sbSize = sizeof(SkyUniform);

    // Defer actual destruction to VulkanResourceManager; clear local handles
    if (skyBuffer.buffer != VK_NULL_HANDLE) {
        skyBuffer.buffer = VK_NULL_HANDLE;
    }
    if (skyBuffer.memory != VK_NULL_HANDLE) {
        skyBuffer.memory = VK_NULL_HANDLE;
    }
    skyBuffer = app->createBuffer(sbSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    skyBufferSize = sbSize;

    // upload initial data
    SkyUniform data{};
    if (skySettings) {
        data.skyHorizon = glm::vec4(skySettings->horizonColor, 1.0f);
        data.skyZenith = glm::vec4(skySettings->zenithColor, 1.0f);
        data.skyParams = glm::vec4(skySettings->warmth, skySettings->exponent, skySettings->sunFlare, static_cast<float>(skySettings->mode));
        data.nightHorizon = glm::vec4(skySettings->nightHorizon, 1.0f);
        data.nightZenith = glm::vec4(skySettings->nightZenith, 1.0f);
        data.nightParams = glm::vec4(skySettings->nightIntensity, skySettings->starIntensity, 0.0f, 0.0f);
    }
    void* mapped = nullptr;
    if (vkMapMemory(app->getDevice(), skyBuffer.memory, 0, sbSize, 0, &mapped) == VK_SUCCESS) {
        memcpy(mapped, &data, static_cast<size_t>(sbSize));
        vkUnmapMemory(app->getDevice(), skyBuffer.memory);
    }

    // bind into descriptor sets (binding 6)
    VkDescriptorBufferInfo skyBufInfo{ skyBuffer.buffer, 0, sbSize };
    VkWriteDescriptorSet skyWrite{};
    skyWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    skyWrite.dstBinding = 6;
    skyWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    skyWrite.descriptorCount = 1;
    skyWrite.pBufferInfo = &skyBufInfo;

    if (descriptorSet != VK_NULL_HANDLE) {
        skyWrite.dstSet = descriptorSet;
        app->updateDescriptorSet({ skyWrite });
    }

}

void SkySphere::update(VulkanApp* app) {
    if (skyBuffer.buffer == VK_NULL_HANDLE) return;
    SkyUniform skyData;
    if (skySettings) {
        skyData.skyHorizon = glm::vec4(skySettings->horizonColor, 1.0f);
        skyData.skyZenith = glm::vec4(skySettings->zenithColor, 1.0f);
        skyData.skyParams = glm::vec4(skySettings->warmth, skySettings->exponent, skySettings->sunFlare, static_cast<float>(skySettings->mode));
        skyData.nightHorizon = glm::vec4(skySettings->nightHorizon, 1.0f);
        skyData.nightZenith = glm::vec4(skySettings->nightZenith, 1.0f);
        skyData.nightParams = glm::vec4(skySettings->nightIntensity, skySettings->starIntensity, 0.0f, 0.0f);
    } else {
        memset(&skyData, 0, sizeof(skyData));
    }
    void* mapped = nullptr;
    if (vkMapMemory(app->getDevice(), skyBuffer.memory, 0, skyBufferSize, 0, &mapped) == VK_SUCCESS) {
        memcpy(mapped, &skyData, static_cast<size_t>(skyBufferSize));
        vkUnmapMemory(app->getDevice(), skyBuffer.memory);
    }
}

void SkySphere::cleanup() {
    // Clear local handles; VulkanResourceManager will perform destruction
    skyBuffer.buffer = VK_NULL_HANDLE;
    skyBuffer.memory = VK_NULL_HANDLE;
}
