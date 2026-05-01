#pragma once
#include <glm/glm.hpp>

// GPU-side water params UBO (matches shader WaterParamsUBO layout)
struct WaterParamsGPU {
    glm::vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    glm::vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    glm::vec4 params3;  // x=noiseTimeSpeed, y=unused, z=specularIntensity, w=specularPower
    glm::vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    glm::vec4 deepColor; // xyz = deepColor, w = glitterIntensity
    glm::vec4 waveParams; // x=unused, y=unused, z=bumpAmplitude, w=depthFalloff
    glm::vec4 reserved1;  // x=enableReflection, y=enableRefraction, z=enableBlur, w=blurRadius
    glm::vec4 reserved2;  // x=blurSamples, y=volumeBlurRate, z=volumeBumpRate, w=unused
    glm::vec4 reserved3;  // x=cube360Available(0/1), y=unused, z=unused, w=unused
    glm::vec4 causticColor; // xyz = color, w = unused
    glm::vec4 causticParams; // x=scale, y=intensity, z=power, w=depthScale
    glm::vec4 causticExtraParams; // x=lineScale, y=lineMix, z = causticType (0=perlin,1=voronoi), w = causticVelocity
};
