#ifndef WIND_PARAMS_NAMED_GLSL
#define WIND_PARAMS_NAMED_GLSL

// Extracted from the vegetation/impostor vertex stages (single-struct
// GLSL type). Built from the per-shader WindParamsUBO block.

// Named view over the packed WindParamsUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct WindParamsNamed {
    vec2 windDirection;
    float windStrength;
    float windBaseFrequency;
    float windSpeed;
    float gustFrequency;
    float gustStrength;
    float skewAmount;
    float trunkStiffness;
    float noiseScale;
    float verticalFlutter;
    float turbulence;
    bool densityEnabled;
    float nearDistance;
    float farDistance;
    float minFactor;
    vec3 cameraPosition;
    float densityFalloff;
};

#endif // WIND_PARAMS_NAMED_GLSL
