// Water fragment stage (moved from water.frag): varyings, set-2 scene
// textures. Shared declarations (ubo, textures, RT bindings under
// RT_ENABLED, outColor, main, hsv/perlin/water_noise/voronoi) are
// provided by main.frag. Includes includes/water_surface.glsl.
// RT_ENABLED selects the hardware ray-tracing variant (ray queries + RT
// pipeline outputs). Without it the shader uses the sky-equirect fallback
// with identical miss baselines (validation-clean on non-RT hardware).


// Water fragment shader
// Samples scene color with Perlin noise-based refraction, specular lighting, and depth-based effects

layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) in vec3 fragBaseNormal;  // undisplaced base normal
layout(location = VARY_BASEPOS) in vec4 fragBasePos;          // xyz = undisplaced base position, w = TES bump amplitude
layout(location = VARY_UV) in vec2 fragTexCoord;
layout(location = VARY_POSCLIP) in vec4 fragPosClip;  // clip-space position for scene sampling
layout(location = VARY_DEBUG) in vec3 fragDebug;   // debug visual (displacement)
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace; // light-space pos (cascade 0)
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;
layout(location = VARY_HSV) in vec3 fragHSV;


// Use the same UBO as main shader


// Water offscreen pass inputs (set 2).
// Hybrid RT: the legacy solid-360 cubemap (binding 1) is REMOVED. Reflection /
// refraction come from hardware ray tracing — either the async RT pipeline
// outputs (bindings 1/2, 1-frame latency, same-queue ordered) or inline ray
// queries against the proxy TLAS (set 0, binding 14) with the sky equirect
// (binding 3) as the miss fallback. Precedence per lobe: REFLECTION — the
// inline ray (full-res exact chunk triangles) wins whenever it runs, so
// mirror positions match the scene; the pipe covers budget-skipped pixels.
// REFRACTION — pipe-first (pre-existing): a valid pipe texel
// (refract.a >= 0.0) is used as-is and its inline ray is skipped; inline
// runs only for invalid pipe texels (or when the pipeline is off). Ray
// budget (rt.rayParams): checkerboard half-rate + Fresnel stochastic
// reflection-xor + contribution gate, each applied only where the pipe
// covers the pixel; any rt.debug view except 59-61 forces full-rate
// dual-trace reference.
// The pass stays decoupled from the solid pass (no solid color/depth reads
// except the in-trace exact-hit reproject); occlusion resolves at composite.
layout(set = 2, binding = 0) uniform sampler2D waterBackDepthTex; // back-face depth for volume thickness
layout(set = 2, binding = 1) uniform sampler2D rtReflectTex;   // RT pipeline reflection (rgb, a=valid)
layout(set = 2, binding = 2) uniform sampler2D rtRefractTex;   // RT pipeline refraction (rgb, a=thickness or -1)
layout(set = 2, binding = 3) uniform sampler2D skyEquirectTex; // sky for RT miss/fallback
// Screen-space reflection refinement: the solid pass HDR color/depth. The RT
// proxy reflection is precise for on-screen scenery, blocky elsewhere; SSR
// resolves the near field per pixel and the proxy result fills the rest.
layout(set = 2, binding = 4) uniform sampler2D solidSceneColorTex;
layout(set = 2, binding = 5) uniform sampler2D solidSceneDepthTex;
// Vegetation layer (grass billboards / impostors): reflections must show the
// grass the way the composite does, otherwise grass-covered hills mirror as
// bare dirt. Binding 6 = color (alpha = coverage), 7 = depth.
layout(set = 2, binding = 6) uniform sampler2D vegColorTex;
layout(set = 2, binding = 7) uniform sampler2D vegDepthTex;






#include "water_surface.glsl"
