#include "WaterRenderer.hpp"

#include "../RendererUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <glm/glm.hpp>

namespace {

float waterPeriodToScale(float period) {
    return period > 0.0f ? 1.0f / period : 0.0f;
}

// CPU/UI stores feature periods; the shader consumes spatial scales. Convert
// exactly once at the upload boundary, and turn authored component amplitudes
// into constant steepnesses (same packing contract the raster water used).
void waterGpuPeriodsToScales(WaterParams& gpu) {
    static const WaterSettings kRef{};
    if (kRef.wavePeriod > 0.0f)
        gpu.waveComponent1.z *= gpu.waveComponent1.x / kRef.wavePeriod;

    gpu.params2.y = waterPeriodToScale(gpu.params2.y);
    gpu.waveComponent1.x = waterPeriodToScale(gpu.waveComponent1.x);
    gpu.waveComponent2.x = waterPeriodToScale(gpu.waveComponent2.x);
    gpu.waveMask.x = waterPeriodToScale(gpu.waveMask.x);
    gpu.foamNoise.x = waterPeriodToScale(gpu.foamNoise.x);
}

WaterParams makeWaterParams(const WaterSettings& p) {
    const float shoreAngle = glm::radians(p.shoreWaveAngle);
    const glm::vec2 shoreDir(std::sin(shoreAngle), std::cos(shoreAngle));

    WaterParams gpu{};
    gpu.params1 = glm::vec4(p.refractionStrength, p.fresnelPower, p.transparency, p.reflectionStrength);
    gpu.params2 = glm::vec4(p.waterTint, p.noisePeriod, static_cast<float>(p.noiseOctaves), p.noisePersistence);
    gpu.params3 = glm::vec4(p.noiseTimeSpeed, p.noiseLacunarity, p.specularIntensity, p.specularPower);
    gpu.glitterParams = glm::vec4(p.glitterIntensity, 0.0f, 0.0f, 0.0f);
    gpu.blurParams = glm::vec4(p.enableBlur ? 1.0f : 0.0f, p.blurRadius, p.blurDepthScale, 0.0f);
    gpu.waveParams = glm::vec4(p.tessNoiseInfluence, 0.0f, p.waveAmplitude, p.depthFalloff);
    gpu.reserved1 = glm::vec4(p.enableReflection ? 1.0f : 0.0f,
                              p.enableRefraction ? 1.0f : 0.0f, 0.0f, 0.0f);
    gpu.reserved2 = glm::vec4(0.0f, 0.0f, 0.0f, p.uniformReflection ? 1.0f : 0.0f);
    gpu.reserved3 = glm::vec4(0.0f);
    gpu.tessParams = glm::vec4(p.tessNearDist, p.tessFarDist, p.tessMinLevel, p.tessMaxLevel);
    gpu.causticColor = glm::vec4(p.causticColor, 0.0f);
    gpu.causticParams = glm::vec4(p.causticSoftness, p.causticIntensity, 0.0f, 0.0f);
    gpu.causticExtraParams = glm::vec4(0.0f);
    gpu.absorptionParams = glm::vec4(p.absorption, p.absorptionScale);
    gpu.refractionParams = glm::vec4(p.ior, p.maxThickness, p.shoreFadeDepth, 0.0f);
    gpu.waveToggles = glm::vec4(p.enableWaves ? 1.0f : 0.0f,
                                p.enableFoam ? 1.0f : 0.0f,
                                p.enableVolumetric ? 1.0f : 0.0f, 0.0f);
    gpu.waveZones = glm::vec4(0.0f);
    gpu.waveDirection = glm::vec4(shoreDir.x, shoreDir.y, 0.0f, 0.0f);
    gpu.waveShape = glm::vec4(p.shoreWaveSlope, p.waveCrestSharpness, 0.0f, 0.0f);
    gpu.waveShoal = glm::vec4(0.0f, 0.0f, p.waveSteepness, 0.0f);
    gpu.waveComponent1 = glm::vec4(p.wavePeriod, p.waveSpeed, 1.0f, 0.0f);
    gpu.waveComponent2 = glm::vec4(0.0f);
    gpu.waveBreaker = glm::vec4(p.windWaveResponse, p.windWaveReference,
                                p.windWaveShallowBlend, 0.0f);
    gpu.waveCurl = glm::vec4(p.rippleHeight, 0.0f, 0.0f, 0.0f);
    gpu.waveWarp = glm::vec4(0.0f, 0.0f, 0.0f, p.shoreGradientStep);
    gpu.waveMask = glm::vec4(p.shoreWaveFade, 0.0f, 0.0f, 0.0f);
    gpu.foamParams = glm::vec4(p.foamCrestThreshold, p.foamTrailPhase, p.foamDecay, p.foamColorAmount);
    gpu.foamNoise = glm::vec4(p.foamNoisePeriod, p.foamNoiseSpeed, p.foamNoiseAmount, 0.0f);
    gpu.foamExtra = glm::vec4(p.foamMaskFloor, p.foamDiffuseFloor, p.foamAmbient, 0.0f);
    gpu.foamContact = glm::vec4(p.foamContactWidth, p.foamContactAmount, p.foamContactAlpha, p.foamContactFloor);
    gpu.foamShape = glm::vec4(p.foamEdge, p.foamCoverage, 0.0f, 0.0f);
    gpu.foamColor = glm::vec4(p.foamColor, 0.0f);
    gpu.volumetricParams = glm::vec4(p.volumetricStrength, p.volumetricDensity, p.volumetricPhaseG, 0.0f);
    gpu.volumetricColor = glm::vec4(p.volumetricColor, 0.0f);
    gpu.regionShoreColor = glm::vec4(p.waterColor, 0.0f);
    gpu.regionShallowColor = glm::vec4(0.0f);
    gpu.regionBreakerColor = glm::vec4(0.0f);
    gpu.regionShoalColor = glm::vec4(0.0f);
    gpu.regionDeepColor = glm::vec4(0.0f);
    gpu.regionTintParams = glm::vec4(0.0f, p.tintShoreFadeDepth, 0.0f, 0.0f);
    gpu.musicAudio1 = glm::vec4(0.0f);
    gpu.musicAudio2 = glm::vec4(0.0f);
    return gpu;
}

} // namespace

WaterRenderer::WaterRenderer() = default;
WaterRenderer::~WaterRenderer() = default;

void WaterRenderer::init(VulkanApp* app, Buffer& paramsBuffer,
                         const std::vector<WaterSettings>& waterParams,
                         uint32_t layerCount) {
    waterParamsBuffer = paramsBuffer;
    waterParamsCount = layerCount;

    VkBufferUsageFlags uboUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (app && app->useDescriptorBuffer())
        uboUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    waterRenderUBO_ = app->createBuffer(sizeof(WaterRenderUBO), uboUsage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (waterParamsBuffer.buffer != VK_NULL_HANDLE) {
        for (uint32_t i = 0; i < waterParams.size() && i < waterParamsCount; ++i)
            updateGPUParamsForLayer(i, waterParams[i]);
    }
    ensureDummyView(app);
}

void WaterRenderer::ensureDummyView(VulkanApp* app) {
    if (!app || dummyWaterView_ != VK_NULL_HANDLE) return;
    RendererUtils::createImage2DWithVma(app->getDevice(), app, 1, 1,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, "WaterRenderer: dummyWater",
        dummyWaterImage_, dummyWaterAlloc_, dummyWaterMem_, dummyWaterView_);
    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        app->recordTransitionImageLayoutLayer(cmd, dummyWaterImage_, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
        VkClearColorValue clear{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, dummyWaterImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clear, 1, &range);
        app->recordTransitionImageLayoutLayer(cmd, dummyWaterImage_, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    });
    app->setImageLayoutTracked(dummyWaterImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
}

void WaterRenderer::updateGPUParamsForLayer(uint32_t layer, const WaterSettings& params) {
    if (layer >= waterParamsCount || waterParamsBuffer.buffer == VK_NULL_HANDLE) return;
    WaterParams gpu = makeWaterParams(params);
    waterGpuPeriodsToScales(gpu);
    void* data = waterParamsBuffer.map(static_cast<VkDeviceSize>(layer) * sizeof(WaterParams));
    if (data) std::memcpy(data, &gpu, sizeof(WaterParams));
}

void WaterRenderer::updateMusicAudio(float amplitude, float bass, float mid, float high,
                                     float beat, bool enabled) {
    if (waterParamsBuffer.buffer == VK_NULL_HANDLE ||
        waterParamsBuffer.mappedData == nullptr || waterParamsCount == 0)
        return;

    const glm::vec4 audio1(enabled ? std::clamp(amplitude, 0.0f, 1.0f) : 0.0f,
                           enabled ? std::clamp(bass, 0.0f, 1.0f) : 0.0f,
                           enabled ? std::clamp(mid, 0.0f, 1.0f) : 0.0f,
                           enabled ? std::clamp(high, 0.0f, 1.0f) : 0.0f);
    const glm::vec4 audio2(enabled ? std::clamp(beat, 0.0f, 1.0f) : 0.0f,
                           enabled ? 1.0f : 0.0f, 0.0f, 0.0f);
    const size_t off1 = offsetof(WaterParams, musicAudio1);
    const size_t off2 = offsetof(WaterParams, musicAudio2);
    for (uint32_t layer = 0; layer < waterParamsCount; ++layer) {
        char* base = static_cast<char*>(waterParamsBuffer.mappedData) +
                     static_cast<size_t>(layer) * sizeof(WaterParams);
        std::memcpy(base + off1, &audio1, sizeof(audio1));
        std::memcpy(base + off2, &audio2, sizeof(audio2));
    }
}

void WaterRenderer::cleanup(VulkanApp* app) {
    if (app) {
        VkDevice device = app->getDevice();
        if (dummyWaterView_ != VK_NULL_HANDLE) {
            if (app->resources.removeImageView(dummyWaterView_))
                vkDestroyImageView(device, dummyWaterView_, nullptr);
            dummyWaterView_ = VK_NULL_HANDLE;
        }
        if (dummyWaterImage_ != VK_NULL_HANDLE) {
            app->destroyImageWithVma(dummyWaterImage_, dummyWaterAlloc_, dummyWaterMem_);
            dummyWaterImage_ = VK_NULL_HANDLE;
        }
    }
    if (waterRenderUBO_.buffer != VK_NULL_HANDLE) waterRenderUBO_ = {};
}
