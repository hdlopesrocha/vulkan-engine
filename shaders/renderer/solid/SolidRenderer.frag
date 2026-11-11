
#version 460
// RT_ENABLED is defined for the SolidRendererRT.frag.spv variant (hardware RT path).
// Without it the shader uses the procedural-sky fallback with identical miss
// baselines, so non-RT hardware stays validation-clean (no TLAS/ray queries).
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif
// Shader clock for per-op RT profiling (RT_PROFILE variants only). Must be a
// top-level #extension: directives inside includes are illegal after tokens.
#ifdef RT_PROFILE
#extension GL_EXT_shader_realtime_clock : require

#include "../../ssbo/RTProxyMetaGLSL.glsl"
#include "../../ubo/RayTracingParams.glsl"
#endif
#include "../../includes/Locations.glsl"

// ── Solid terrain fragment shader ─────────────────────────────────────────
// Split out of the old shared main.frag (which selected the path with
// -DWATER_MODE). Compile-time variants:
//   * default            -> shaders/SolidRenderer.frag.spv
//   * -DRT_ENABLED       -> shaders/SolidRendererRT.frag.spv
//   * -DRT_ENABLED -DRT_PROFILE -> shaders/SolidRendererRTProf.frag.spv
layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;
layout(location = VARY_TEXTURETYPE) flat in ivec3 fragTextureTypes;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_TEXWEIGHTS) in vec3 fragTexWeights;
layout(location = VARY_HSV) in vec3 fragHSV;
layout(location = VARY_SHARPNORMAL) in vec3 fragSharpNormal; // face normal computed in TES (sharp)
layout(location = VARY_DEBUG) in vec3 fragTessLevel; // tessellation level heat (tesc output /16, DEBUG_MODE_TESS_HEAT)

#include "../../includes/SceneBindings.glsl"

// Depth-region water tint ramp: shared by the raster water surface and the
// water hit-shading inside RT rays (both include this helper).
#include "../../includes/water/WaterTint.glsl"

#include "../../includes/DebugModes.glsl"

#include "../../includes/Textures.glsl"

// Set=1 binding 0: brush SDF description (std140 UBO). PAINT/REMOVE mode
// evaluates the brush field at the fragment position to decide whether the
// fragment lies inside the brush volume — the exact SDF equivalent of the old
// rasterized front/back depth interval test.
#include "../../includes/brush/BrushSdf.glsl"

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "../../includes/Common.glsl"

#include "../../includes/solid/TangentBasis.glsl"
#include "../../includes/solid/Triplanar.glsl"

#include "../../includes/shadow/Shadows.glsl"

#include "../../includes/Hsv.glsl"

// Hybrid RT declarations (bindings 14/17/18 — TLAS, params, proxy metadata).
// The TLAS holds the stable solid-proxy boxes; water surface is excluded by
// design (origins, never targets). Shaders gate all sampling on rt.tlasReady
// (tlasReady): before the first BLAS/TLAS build completes everything falls
// back to sky/CSM so no invalid acceleration structure is ever traced.
// rt_params.glsl is included UNCONDITIONALLY (structs + rtProceduralSky): the
// sky-reflection fallback used with RT off needs the procedural sky helper;
// the RT UBO/TLAS/buffer bindings stay compiled out without RT_ENABLED.
#include "../../includes/rt/RtParams.glsl"
#ifdef RT_ENABLED
#include "../../includes/rt/RtSceneBindings.glsl"
#endif

// Screen-space reflection refinement (set 0, bindings 19/20): the *previous*
// frame's solid HDR color/depth. The proxy ray query gives plausible but
// blocky reflections; where the reflected point is on screen, marching the
// real depth buffer and sampling the real color resolves a precise mirror.
layout(set = 0, binding = 19) uniform sampler2D ssrColorTex;
layout(set = 0, binding = 20) uniform sampler2D ssrDepthTex;


// Global toggles
bool roughnessEnabled = (ubo.roughnessEnabled != 0u);
bool aoEnabled = (ubo.ambientOcclusionEnabled != 0u);

// Solid surface shading (extracted): writes outColor; debug branches return.
#include "../../includes/solid/SolidSurface.glsl"

void main() {
    bool isShadowPass = (ubo.isShadowPass != 0u);
    // Provide a default color so early debug/special-case returns still
    // produce a valid output for downstream passes.
    outColor = vec4(0.0);

    // Fast-path for shadow pass: skip expensive lighting/texture work.
    if (isShadowPass) {
        outColor = vec4(0.0);
        return;
    }

    // Solid surface shading lives in includes/solid/SolidSurface.glsl;
    // it writes outColor (debug branches return early inside it).
    shadeSolidSurface(vec3(0.0));
}
