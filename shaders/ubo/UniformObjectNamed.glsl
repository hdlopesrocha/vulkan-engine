#ifndef UNIFORM_OBJECT_NAMED_GLSL
#define UNIFORM_OBJECT_NAMED_GLSL

// Extracted from shaders/includes/ubo.glsl (single-struct GLSL type).
// NOTE: the vegetation/impostor/capture vertex stages previously carried a
// trimmed 2-field copy (viewProjection + viewPosition) so they could avoid
// the full UBO dependency; they now use this canonical struct and fill only
// those two fields (the remaining fields are never read by those stages).

// Named view over the packed SolidParamsUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct UniformObjectNamed {
    mat4 viewProjection;
    vec3 viewPosition;
    vec3 lightDirection;
    float lightElevation;
    vec3 lightColor;
    bool cubemapCapture;
    bool normalMappingEnabled;
    bool shadowsEnabled;
    int debugMode;
    bool roughnessEnabled;
    bool ambientOcclusionEnabled;
    float triplanarThreshold;
    float triplanarExponent;
    float tessNearDist;
    float tessFarDist;
    float tessellationFactor;
    bool isShadowPass;
    bool tessellationEnabled;
    float nearPlane;
    float farPlane;
    mat4 lightSpaceMatrix;
    mat4 lightSpaceMatrix1;
    mat4 lightSpaceMatrix2;
    mat4 invViewProjection;
    float brushTextureIndex;
    float brushMode;
    float brushPhase;
    vec3 brushHsv;
};

#endif // UNIFORM_OBJECT_NAMED_GLSL
