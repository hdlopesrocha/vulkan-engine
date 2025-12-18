#include "SkySphere.hpp"
#include "../widgets/SkyWidget.hpp"
#include "../Uniforms.hpp"
#include <glm/glm.hpp>

SkySphere::SkySphere(VulkanApp* app_) : app(app_) {}

SkySphere::~SkySphere() { cleanup(); }

void SkySphere::init(SkyWidget* widget,
                     VkDescriptorSet descriptorSet,
                     VkDescriptorSet shadowDescriptorSet) {
    skyWidget = widget;
    VkDeviceSize sbSize = sizeof(SkyUniform);

    if (skyBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), skyBuffer.buffer, nullptr);
        skyBuffer.buffer = VK_NULL_HANDLE;
    }
    if (skyBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), skyBuffer.memory, nullptr);
        skyBuffer.memory = VK_NULL_HANDLE;
    }
    skyBuffer = app->createBuffer(sbSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    skyBufferSize = sbSize;

    // upload initial data
    SkyUniform data{};
    if (skyWidget) {
        data.skyHorizon = glm::vec4(skyWidget->getHorizonColor(), 1.0f);
        data.skyZenith = glm::vec4(skyWidget->getZenithColor(), 1.0f);
        data.skyParams = glm::vec4(skyWidget->getWarmth(), skyWidget->getExponent(), skyWidget->getSunFlare(), 0.0f);
        data.nightHorizon = glm::vec4(skyWidget->getNightHorizon(), 1.0f);
        data.nightZenith = glm::vec4(skyWidget->getNightZenith(), 1.0f);
        data.nightParams = glm::vec4(skyWidget->getNightIntensity(), skyWidget->getStarIntensity(), 0.0f, 0.0f);
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
        app->updateDescriptorSet(descriptorSet, { skyWrite });
    }
    if (shadowDescriptorSet != VK_NULL_HANDLE) {
        skyWrite.dstSet = shadowDescriptorSet;
        app->updateDescriptorSet(shadowDescriptorSet, { skyWrite });
    }

}

void SkySphere::update() {
    if (skyBuffer.buffer == VK_NULL_HANDLE) return;
    SkyUniform skyData;
    if (skyWidget) {
        skyData.skyHorizon = glm::vec4(skyWidget->getHorizonColor(), 1.0f);
        skyData.skyZenith = glm::vec4(skyWidget->getZenithColor(), 1.0f);
        skyData.skyParams = glm::vec4(skyWidget->getWarmth(), skyWidget->getExponent(), skyWidget->getSunFlare(), 0.0f);
        skyData.nightHorizon = glm::vec4(skyWidget->getNightHorizon(), 1.0f);
        skyData.nightZenith = glm::vec4(skyWidget->getNightZenith(), 1.0f);
        skyData.nightParams = glm::vec4(skyWidget->getNightIntensity(), skyWidget->getStarIntensity(), 0.0f, 0.0f);
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
    if (skyBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), skyBuffer.buffer, nullptr);
        skyBuffer.buffer = VK_NULL_HANDLE;
    }
    if (skyBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(app->getDevice(), skyBuffer.memory, nullptr);
        skyBuffer.memory = VK_NULL_HANDLE;
    }
}
