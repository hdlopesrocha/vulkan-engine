#ifndef RAY_TRACING_PARAMS_G_L_S_L_GLSL
#define RAY_TRACING_PARAMS_G_L_S_L_GLSL

// Extracted from shaders/includes/rt_params.glsl (single-struct GLSL type).

struct RayTracingParamsGLSL {
    vec4 toggles;      // x=reflections y=refractions z=thickness w=localShadows
    vec4 distances;    // x=maxReflect y=maxRefract z=maxShadowDist w=roughnessThreshold
    vec4 water;        // x=IOR, y=maxWaterThickness (hit clamp), z=coarseBoxSize (deep/sky fallback), w=maxReflectionBounces
    vec4 absorption;   // rgb=Beer-Lambert coeff, a=thicknessScale
    vec4 debug;        // x=DebugMode (see debug_modes.glsl), y=tlasReady, z=selfSkipDist, w=useWaterPipeline
    mat4 invViewProj;
    mat4 prevViewProj; // previous frame's view-projection (temporal SSR reprojection)
    vec4 viewPos;
    vec4 rtResolution; // xy=size, zw=1/size
    vec4 clipPlanes;   // x=near, y=far
    vec4 sunDir;       // xyz=direction TO sun
    vec4 sunColor;
    vec4 rayParams;    // x=rayScaleMode (0=full-rate, 1=checkerboard half-rate),
                       // y=contribMin (skip inline ray when lobe contribution below),
                       // z=singleRay (1=Fresnel stochastic reflection-xor-refraction,
                       //   0=dual-trace reference),
                       // w=waterReflections (1 = water reflection rays enabled)
    vec4 waterDepth;   // x=water-region depth source: 1 = ray-traced solid
                       // bottom (world-space vertical drop) in the water TES,
                       // 0 = raster only (solid scene depth + water volume back
                       // face). yzw reserved.
};

#endif // RAY_TRACING_PARAMS_G_L_S_L_GLSL
