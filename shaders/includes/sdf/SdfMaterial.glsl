// Generic SDF material + volume shading functions. Type definitions live in
// shaders/types/*.glsl (CPU twins in vulkan/types/*GPU.hpp); descriptor
// layout is declared by the shaders themselves (SdfRenderer.vert/SdfRenderer.frag):
//   set=0 binding=0 SolidParamsBlock (UniformObject/camera), set=1 bindings 0..8 as in
//   SdfRenderer (0..5 scene SSBOs, 6 params UBO, 7 scene depth, 8 smoke
//   state block).

#ifndef SDF_MATERIAL_INCLUDE_GLSL
#define SDF_MATERIAL_INCLUDE_GLSL

#include "../../types/SdfDefinition.glsl"
#include "../../types/SdfMaterial.glsl"
#include "../../types/SdfMaterialType.glsl"







float sdfUnpackSmoothK(SdfDefinition def) {
    return def.smoothK; // canonical float field (no bit-cast packing)
}

// Black-body-ish gradient: dark red -> red -> orange -> yellow -> white.
vec3 sdfTemperatureColor(float t) {
    float x = clamp(t, 0.0, 1.0);
    vec3 darkRed = vec3(0.25, 0.0, 0.0);
    vec3 red     = vec3(0.90, 0.08, 0.0);
    vec3 orange  = vec3(1.00, 0.45, 0.05);
    vec3 yellow  = vec3(1.00, 0.85, 0.40);
    vec3 white   = vec3(1.00, 1.00, 0.95);
    vec3 col = mix(darkRed, red, smoothstep(0.0, 0.30, x));
    col = mix(col, orange, smoothstep(0.30, 0.55, x));
    col = mix(col, yellow, smoothstep(0.55, 0.80, x));
    col = mix(col, white, smoothstep(0.80, 1.0, x));
    return col;
}

// Generic volume sample: (rgb = emission, a = density). Density rises inside
// the field (smoothstep falloff of -sdfDist), modulated by noise and height;
// emission follows the temperature gradient scaled by material emission.
vec4 sdfEvaluateVolume(float sdfDist, float height01, float noiseVal,
                       SdfMaterial mat, float temperature) {
    float h = clamp(height01, 0.0, 1.0);
    float n = clamp(noiseVal, 0.0, 1.0);
    float t = clamp(temperature * max(mat.tempScale, 0.0), 0.0, 1.0);
    float soft = 0.15;
    float body = 1.0 - smoothstep(-soft, soft, sdfDist);
    float topFade = 1.0 - smoothstep(0.7, 1.0, h);
    float density = body * max(mat.density, 0.0) * (0.45 + 0.55 * n) * topFade;
    vec3 emission = sdfTemperatureColor(t) * mat.emission
                  * max(mat.emissionIntensity, 0.0) * body * (0.5 + 0.5 * n);
    return vec4(emission, density);
}

#endif // SDF_MATERIAL_INCLUDE_GLSL
