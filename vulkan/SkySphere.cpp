#include "SkySphere.hpp"
#include "../widgets/SkySettings.hpp"
#include "../widgets/CloudSettings.hpp"
#include "ubo/SkyUniform.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cstring>

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
    // Descriptor-buffer sources need a device address for vkGetDescriptorEXT.
    VkBufferUsageFlags skyUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (app->useDescriptorBuffer())
        skyUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    skyBuffer = app->createBuffer(sbSize, skyUsage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    skyBufferSize = sbSize;

    // upload initial data
    SkyUniform data{};
    fillSkyUniform(data);
    memcpy(skyBuffer.mappedData, &data, static_cast<size_t>(sbSize));

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

void SkySphere::fillSkyUniform(SkyUniform& out) const {
    memset(&out, 0, sizeof(out));
    if (skySettings) {
        out.skyHorizon = glm::vec4(skySettings->horizonColor, 1.0f);
        out.skyZenith = glm::vec4(skySettings->zenithColor, 1.0f);
        out.skyParams = glm::vec4(skySettings->warmth, skySettings->exponent, skySettings->sunFlare, static_cast<float>(skySettings->mode));
        out.nightHorizon = glm::vec4(skySettings->nightHorizon, 1.0f);
        out.nightZenith = glm::vec4(skySettings->nightZenith, 1.0f);
        out.nightParams = glm::vec4(skySettings->nightIntensity, skySettings->starIntensity, 0.0f, 0.0f);
    }
    // Clouds share the same UBO. When no CloudSettings are attached the
    // toggles stay zero (= disabled), keeping old sky-only behavior.
    bool masterOn = false;
    if (cloudSettings) {
        masterOn = cloudSettings->enabled;
        if (hasCloudsOverride) masterOn = masterOn && cloudsEnabledOverride;
        const float windRad = glm::radians(cloudSettings->windAngleDeg);
        out.cloudToggles = glm::vec4(masterOn ? 1.0f : 0.0f,
            cloudSettings->lowEnabled ? 1.0f : 0.0f,
            cloudSettings->midEnabled ? 1.0f : 0.0f,
            cloudSettings->highEnabled ? 1.0f : 0.0f);
        out.cloudGlobal = glm::vec4(cloudSettings->densityScale, cloudSettings->windSpeed,
            windRad, cloudSettings->detailStrength);
        out.cloudTime = glm::vec4(cloudTime * std::max(cloudSettings->timeScale, 0.0f),
            cloudSettings->shadowStrength,
            static_cast<float>(cloudSettings->raymarchSteps),
            static_cast<float>(cloudSettings->lightSteps));
        out.cloudLow = glm::vec4(cloudSettings->lowCoverage, cloudSettings->lowDensity,
            cloudSettings->lowScale, cloudSettings->lowWindSpeedMul);
        out.cloudLowGeom = glm::vec4(cloudSettings->lowBaseHeight, cloudSettings->lowThickness, 0.0f, 0.0f);
        out.cloudMid = glm::vec4(cloudSettings->midCoverage, cloudSettings->midDensity,
            cloudSettings->midScale, cloudSettings->midWindSpeedMul);
        out.cloudMidGeom = glm::vec4(cloudSettings->midBaseHeight, cloudSettings->midThickness, 0.0f, 0.0f);
        out.cloudHigh = glm::vec4(cloudSettings->highCoverage, cloudSettings->highDensity,
            cloudSettings->highScale, cloudSettings->highWindSpeedMul);
        out.cloudHighGeom = glm::vec4(cloudSettings->highBaseHeight, cloudSettings->highThickness, 0.0f, 0.0f);
        out.cloudLight = glm::vec4(cloudSettings->silverLining, cloudSettings->ambientBoost,
            cloudSettings->sunForwardG, cloudSettings->exposure);
        out.cloudAnim = glm::vec4(cloudSettings->timeScale, 0.0f, 0.0f, 0.0f);
    } else {
        out.cloudToggles = glm::vec4(hasCloudsOverride && !cloudsEnabledOverride ? 0.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    }
}

void SkySphere::update(VulkanApp* app) {
    if (skyBuffer.buffer == VK_NULL_HANDLE) return;
    (void)app;
    SkyUniform skyData;
    fillSkyUniform(skyData);
    memcpy(skyBuffer.mappedData, &skyData, static_cast<size_t>(skyBufferSize));
}

void SkySphere::writeDescriptorSet(VulkanApp* app, VkDescriptorSet descriptorSet) {
    if (skyBuffer.buffer == VK_NULL_HANDLE || descriptorSet == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo skyBufInfo{ skyBuffer.buffer, 0, skyBufferSize };
    VkWriteDescriptorSet skyWrite{};
    skyWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    skyWrite.dstSet = descriptorSet;
    skyWrite.dstBinding = 6;
    skyWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    skyWrite.descriptorCount = 1;
    skyWrite.pBufferInfo = &skyBufInfo;
    app->updateDescriptorSet({ skyWrite });
}

void SkySphere::cleanup() {
    // Clear local handles; VulkanResourceManager will perform destruction
    skyBuffer.buffer = VK_NULL_HANDLE;
    skyBuffer.memory = VK_NULL_HANDLE;
}
