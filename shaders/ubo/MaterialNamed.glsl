#ifndef MATERIAL_NAMED_GLSL
#define MATERIAL_NAMED_GLSL

// Extracted from shaders/includes/ubo.glsl (single-struct GLSL type).

// Named view over the packed Material - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct MaterialNamed {
    bool skipEnvMap;
    float ambientFactor;
    bool mappingEnabled;
    float tessLevel;
    bool invertHeight;
    float tessHeightScale;
    float specularStrength;
    float shininess;
    float triplanarScaleU;
    float triplanarScaleV;
    bool triplanarEnabled;
    bool flipNormalY;
    bool swapNormalXZ;
    bool invertWidth;
    float minLevel;
    float maxLevel;
    float reflectionStrength;
    float roughnessFactor;
    float aoFactor;
    bool useAO;
};

#endif // MATERIAL_NAMED_GLSL
