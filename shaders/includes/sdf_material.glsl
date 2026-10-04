// Generic SDF GPU layouts — MUST match vulkan/types/Sdf*GPU.hpp field order.
// std430, all vec4/uvec4 aligned. CPU structs:
//   SdfDefinitionGPU { vec4 params0; vec4 params1; uvec4 meta; } // 48B, meta=(prim,op,deformFlags,bitcast(smoothK))
//   SdfInstanceGPU { vec4 posScale; vec4 rotSeed; vec4 sizeParams; uvec4 indices; vec4 boundsMin; vec4 boundsMax; } // 96B
//   SdfMaterialGPU { vec4 baseColor; vec4 surfaceParams; vec4 emission; vec4 volumeParams; vec4 extra; } // 80B
//     surfaceParams=(roughness,metallic,opacity,modeFloat), emission=(rgb,intensity),
//     volumeParams=(density,absorption,scattering,tempScale), extra=(smoothK,noiseScale,turbulence,riseSpeed)
//   SdfContainerGPU { vec4 boundsMin; vec4 boundsMax; uvec4 gridInfo; uvec4 gridOffset; } // 64B
//     gridInfo=(resX,resY,resZ,indexStart), gridOffset=(cellStart,indexStart,instanceCount,flags)
//   SdfGridCellGPU { uint offset; uint count; uint pad0; uint pad1; } // 16B
//   SdfParamsUBO { vec4 timeDebug; vec4 marchParams; vec4 fireColors0; vec4 fireColors1; } // 64B
//     timeDebug=(time,packedModeDebug,maxSteps,safety), marchParams=(minStep,maxStep,epsilon,earlyTerm)
//
// Descriptor layout (declared in sdf.vert/sdf.frag, NOT here):
//   set=0 binding=0 SolidParamsUBO (camera), set=1 bindings 0..7 as in SdfRenderer.

#ifndef SDF_MATERIAL_GLSL
#define SDF_MATERIAL_GLSL

#include "../ubo/SdfDefinitionGPU.glsl"
#include "../ubo/SdfMaterialGPU.glsl"







float sdfUnpackSmoothK(SdfDefinitionGPU def) {
    return uintBitsToFloat(def.meta.w);
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
                       SdfMaterialGPU mat, float temperature) {
    float h = clamp(height01, 0.0, 1.0);
    float n = clamp(noiseVal, 0.0, 1.0);
    float t = clamp(temperature * max(mat.volumeParams.w, 0.0), 0.0, 1.0);
    float soft = 0.15;
    float body = 1.0 - smoothstep(-soft, soft, sdfDist);
    float topFade = 1.0 - smoothstep(0.7, 1.0, h);
    float density = body * max(mat.volumeParams.x, 0.0) * (0.45 + 0.55 * n) * topFade;
    vec3 emission = sdfTemperatureColor(t) * mat.emission.rgb
                  * max(mat.emission.a, 0.0) * body * (0.5 + 0.5 * n);
    return vec4(emission, density);
}

#endif // SDF_MATERIAL_GLSL
