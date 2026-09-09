#pragma once
#include <glm/glm.hpp>

// GPU-side water params UBO (matches shader WaterParamsUBO layout)
struct WaterParamsGPU {
    glm::vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    glm::vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    glm::vec4 params3;  // x=noiseTimeSpeed, y=noiseLacunarity, z=specularIntensity, w=specularPower
    glm::vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    glm::vec4 deepColor; // xyz = deepColor, w = glitterIntensity
    glm::vec4 waveParams; // x=tessNoiseInfluence, y=unused, z=bumpAmplitude, w=depthFalloff
    glm::vec4 reserved1;  // x=enableReflection, y=enableRefraction, z=legacy enableBlur (unused), w=legacy blurRadius (unused)
    glm::vec4 reserved2;  // x=legacy blurSamples (unused), y=legacy volumeBlurRate (unused), z=volumeBumpRate, w=unused
    glm::vec4 reserved3;  // unused (legacy cubemap-available flag removed with Solid360)
    glm::vec4 tessParams; // x=tessNearDist, y=tessFarDist, z=tessMinLevel, w=tessMaxLevel
    glm::vec4 causticColor; // xyz = color, w = unused
    glm::vec4 causticParams; // x=scale, y=intensity, z=power, w=depthScale
    glm::vec4 causticExtraParams; // x=lineScale, y=lineMix, z = causticType (0=perlin,1=voronoi), w = causticVelocity
    glm::vec4 absorptionParams; // xyz = Beer-Lambert coeff, w = absorption scale
    glm::vec4 refractionParams; // x=IOR, y=max thickness cap, z=shore fade depth, w=unused
};
