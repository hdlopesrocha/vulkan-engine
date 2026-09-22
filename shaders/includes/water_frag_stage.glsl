// Water fragment stage (moved from water.frag): varyings, set-2 scene
// textures. Shared declarations (ubo, textures, RT bindings under
// RT_ENABLED, outColor, main, hsv/perlin/water_noise/voronoi) are
// provided by main.frag. Includes includes/water_surface.glsl.
// RT_ENABLED selects the hardware ray-tracing variant (ray queries + RT
// pipeline outputs). Without it the shader uses the sky-equirect fallback
// with identical miss baselines (validation-clean on non-RT hardware).


// Water fragment shader
// Samples scene color with Perlin noise-based refraction, specular lighting, and depth-based effects

// H6 (perf report 19): VARY_NORMAL / VARY_UV / VARY_POSLIGHT are not declared
// here anymore — the water shading below never reads them (fragBaseNormal
// carries the undisplaced normal, the wave normal is derived per pixel from
// fragBasePos/fragWaterDepth/fragShoreDir). The wireframe debug fragment
// shader still consumes VARY_NORMAL from the TES.
layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_SHARPNORMAL) in vec3 fragBaseNormal;  // undisplaced base normal
layout(location = VARY_BASEPOS) in vec4 fragBasePos;          // xyz = undisplaced base position, w = raw bump amplitude
layout(location = VARY_WATERDEPTH) in float fragWaterDepth;   // TES-measured water thickness (-1 = unknown/deep)
layout(location = VARY_SHOREDIR) in vec2 fragShoreDir;        // unit shore direction (toward thinner water)
layout(location = VARY_POSCLIP) in vec4 fragPosClip;  // clip-space position for scene sampling
layout(location = VARY_DEBUG) in vec3 fragDebug;   // debug visual (displacement)
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;
layout(location = VARY_HSV) in vec3 fragHSV;


// Use the same UBO as main shader


// Water offscreen pass inputs (set 2).
// Hybrid RT: reflection comes exclusively from inline ray queries against the
// exact scene TLAS (set 0, binding 14) with the sky equirect (binding 3) as
// the miss fallback, refined by the screen-space march on shallow misses.
// The async RT pipeline no longer produces a reflection (its proxy output was
// never used as color — rt_water.rgen); binding 1 is retired and always
// invalid. REFRACTION — pipe-first: a valid pipe texel (retract.a >= 0.0) is
// used as-is and its inline ray is skipped; inline runs only for invalid pipe
// texels (or when the pipeline is off), with the screen-space solid sample as
// the non-RT fallback. Ray budget (rt.rayParams): the Fresnel stochastic
// single-ray xor cuts the REFRACTION lobe only (recovered from the raster
// bottom/sky); reflection always traces. Traced-result debug views
// (debugModeForcesRtReference) force full-rate dual-trace reference.
// The pass stays decoupled from the solid pass (no solid color/depth reads
// except the in-trace exact-hit reproject); occlusion resolves at composite.
layout(set = 2, binding = 0) uniform sampler2D waterBackDepthTex; // back-face depth for volume thickness
layout(set = 2, binding = 1) uniform sampler2D rtReflectTex;   // retired RT reflection (always invalid)
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
