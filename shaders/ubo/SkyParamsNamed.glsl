#ifndef SKY_PARAMS_NAMED_GLSL
#define SKY_PARAMS_NAMED_GLSL

// Extracted from shaders/includes/ubo.glsl (single-struct GLSL type).

// Named view over the packed SkyUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct SkyParamsNamed {
    vec3 horizonColor;
    vec3 zenithColor;
    float warmth;
    float exponent;
    float sunFlare;
    vec3 nightHorizonColor;
    vec3 nightZenithColor;
    float nightIntensity;
    float starIntensity;
    // Clouds
    bool cloudsEnabled;
    bool lowEnabled;
    bool midEnabled;
    bool highEnabled;
    float densityScale;
    float windSpeed;
    float windAngleRad;
    float detailStrength;
    float cloudTime;
    float shadowStrength;
    float raymarchSteps;
    float lightSteps;
    vec4 lowTier;      // coverage, density, scale, windMul
    vec2 lowGeom;      // baseHeight, thickness
    vec4 midTier;
    vec2 midGeom;
    vec4 highTier;
    vec2 highGeom;
    float silverLining;
    float ambientBoost;
    float sunForwardG;
    float exposure;
};

#endif // SKY_PARAMS_NAMED_GLSL
