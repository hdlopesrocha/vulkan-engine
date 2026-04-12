#pragma once
#include <glm/glm.hpp>

// Water rendering parameters (CPU-side)
struct WaterParams {
    float waveSpeed = 0.5f;
    float waveScale = 0.03f;
    float refractionStrength = 0.03f;
    float fresnelPower = 5.0f;
    float transparency = 0.7f;
    glm::vec3 shallowColor = glm::vec3(0.1f, 0.4f, 0.5f);
    glm::vec3 deepColor = glm::vec3(0.0f, 0.15f, 0.25f);
    float depthFalloff = 0.1f;
    int noiseOctaves = 4;
    float noisePersistence = 0.5f;
    float noiseScale = 0.4f;
    float waterTint = 0.3f;
    float noiseTimeSpeed = 1.0f;

    // Reflection / specular controls
    float reflectionStrength = 0.6f;  // How much reflection mixes into the surface [0..1]
    float specularIntensity = 2.0f;   // Brightness of specular highlight
    float specularPower = 128.0f;     // Sharpness of specular highlight
    float glitterIntensity = 1.5f;    // Brightness of sun glitter sparkles

    // Vertical bump amplitude for water geometry
    float bumpAmplitude = 8.0f;

    // Depth-based wave attenuation: distance (world units) over which waves
    // transition from zero displacement (at solid surface) to full amplitude.
    // 0 = disabled (no depth-based attenuation).
    float waveDepthTransition = 20.0f;

    // Feature toggles
    bool enableReflection = true;
    bool enableRefraction = true;
    bool enableBlur = true;
    // If true, apply `reflectionStrength` uniformly across the surface
    // instead of modulating by Fresnel. Useful for debugging or stylized looks.
    bool uniformReflection = false;

    // PCF-style scene-color blur
    float blurRadius = 8.0f;    // texel radius of blur kernel
    int   blurSamples = 4;      // number of blur taps per axis (NxN kernel)

    // Volume depth-based effect transitions
    float volumeBlurRate = 0.004f;   // exponential rate: blur ramps with water thickness
    float volumeBumpRate = 0.05f;  // exponential rate: bump ramps with water thickness
    // Caustics / light focusing parameters
    glm::vec3 causticColor = glm::vec3(0.0f, 0.58f, 1.0f);
    float causticIntensity = 10.0f;    // multiplier for caustic brightness
    float causticScale = 5.0f;       // scale factor applied to Jacobian determinant
    float causticPower = 1.0f;        // exponent to sharpen caustic contrast
    // Depth scale (world units) controlling blend from surface->bottom caustic
    float causticDepthScale = 64.0f;
    // Line-shaped caustic tuning
    float causticLineScale = 1.0f;    // multiplier for anisotropy -> line strength
    float causticLineMix = 1.0f;      // 0=cloudy, 1=lines
    // Speed multiplier for caustic animation. 0 = static, 1 = normal speed.
    float causticVelocity = 1.0f;
    // Caustic generation mode: 0 = Perlin (Jacobian-based), 1 = Voronoi (Worley)
    int causticType = 0;
};

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

// GPU-side water render UBO
struct WaterRenderUBO {
    glm::vec4 timeParams; // x = waterTime, yzw = unused
};

// GPU-side water uniform buffer
struct WaterUBO {
    glm::mat4 viewProjection;
    glm::mat4 invViewProjection;
    glm::vec4 viewPos;
    glm::vec4 screenSize;    // width, height, 1/width, 1/height
};
