#ifndef MATERIAL_NAMED_GLSL
#define MATERIAL_NAMED_GLSL

// Extracted from shaders/includes/SceneBindings.glsl (single-struct GLSL type).

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
    // Material kind (0 = texture, 1 = video, 2 = water), texture layer/slot
    // (-1 = the material's own layer), coordinate technique
    // (0 = triplanar, 1..3 = projection XYZ, 4 = angular, 5 = custom UV) and
    // blend mode (0 = normal, 1..19 = shader blend modes).
    int kind;
    int textureIndex;
    int textureMapping;
    int blendMode;
    vec4 mappingOrigin;
};

#endif // MATERIAL_NAMED_GLSL
