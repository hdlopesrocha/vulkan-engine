#include "SkySphere.hpp"
#include "../../widgets/SkySettings.hpp"
#include "../../widgets/CloudSettings.hpp"
#include "../ubo/SkyUniform.hpp"
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
        out.horizonColor = skySettings->horizonColor;
        out.zenithColor = skySettings->zenithColor;
        out.warmth = skySettings->warmth;
        out.exponent = skySettings->exponent;
        out.sunFlare = skySettings->sunFlare;
        out.skyMode = static_cast<uint32_t>(skySettings->mode);
        out.nightHorizonColor = skySettings->nightHorizon;
        out.nightZenithColor = skySettings->nightZenith;
        out.nightIntensity = skySettings->nightIntensity;
        out.starIntensity = skySettings->starIntensity;
    }
    // Clouds share the same UBO. When no CloudSettings are attached the
    // toggles stay zero (= disabled), keeping old sky-only behavior.
    bool masterOn = false;
    if (cloudSettings) {
        masterOn = cloudSettings->enabled;
        if (hasCloudsOverride) masterOn = masterOn && cloudsEnabledOverride;
        const float windRad = glm::radians(cloudSettings->windAngleDeg);
        out.cloudsEnabled = masterOn ? 1u : 0u;
        out.lowEnabled = cloudSettings->lowEnabled ? 1u : 0u;
        out.midEnabled = cloudSettings->midEnabled ? 1u : 0u;
        out.highEnabled = cloudSettings->highEnabled ? 1u : 0u;
        out.densityScale = cloudSettings->densityScale;
        out.windSpeed = cloudSettings->windSpeed;
        out.windAngleRad = windRad;
        out.detailStrength = cloudSettings->detailStrength;
        out.cloudTime = cloudTime * std::max(cloudSettings->timeScale, 0.0f);
        out.shadowStrength = cloudSettings->shadowStrength;
        out.raymarchSteps = static_cast<float>(cloudSettings->raymarchSteps);
        out.lightSteps = static_cast<float>(cloudSettings->lightSteps);
        out.lowCoverage = cloudSettings->lowCoverage;
        out.lowDensity = cloudSettings->lowDensity;
        out.lowScale = cloudSettings->lowScale;
        out.lowWindSpeedMul = cloudSettings->lowWindSpeedMul;
        out.lowBaseHeight = cloudSettings->lowBaseHeight;
        out.lowThickness = cloudSettings->lowThickness;
        out.midCoverage = cloudSettings->midCoverage;
        out.midDensity = cloudSettings->midDensity;
        out.midScale = cloudSettings->midScale;
        out.midWindSpeedMul = cloudSettings->midWindSpeedMul;
        out.midBaseHeight = cloudSettings->midBaseHeight;
        out.midThickness = cloudSettings->midThickness;
        out.highCoverage = cloudSettings->highCoverage;
        out.highDensity = cloudSettings->highDensity;
        out.highScale = cloudSettings->highScale;
        out.highWindSpeedMul = cloudSettings->highWindSpeedMul;
        out.highBaseHeight = cloudSettings->highBaseHeight;
        out.highThickness = cloudSettings->highThickness;
        out.silverLining = cloudSettings->silverLining;
        out.ambientBoost = cloudSettings->ambientBoost;
        out.sunForwardG = cloudSettings->sunForwardG;
        out.exposure = cloudSettings->exposure;
        out.timeScale = cloudSettings->timeScale;
    } else {
        out.cloudsEnabled = 0u;
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
