#ifndef RAY_TRACING_PARAMS_NAMED_GLSL
#define RAY_TRACING_PARAMS_NAMED_GLSL

// Extracted from shaders/includes/rt_params.glsl (single-struct GLSL type).

struct RayTracingParamsNamed {
    bool reflectionsEnabled;
    bool refractionsEnabled;
    bool thicknessEnabled;
    bool localShadowsEnabled;
    float maxRefractDistance;
    float maxShadowDistance;
    float roughnessThreshold;
    float waterIor;
    float maxWaterThickness;
    float coarseBoxSize;
    int maxReflectionBounces;
    int debugMode;
    bool tlasReady;
    float selfSkipDist;
    bool useWaterPipeline;
    bool checkerboardReflections;
    float reflectionContribMin;
    bool singleRay;
    bool waterReflections;
    bool rayTracedWaterDepth;
    vec3 viewPosition;
    vec3 sunDirection;
    vec3 sunColor;
    mat4 invViewProj;
};

#endif // RAY_TRACING_PARAMS_NAMED_GLSL
