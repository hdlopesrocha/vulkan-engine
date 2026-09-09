#version 460
// RT_ENABLED selects the hardware ray-tracing variant (ray queries + RT
// pipeline outputs). Without it the shader uses the sky-equirect fallback
// with identical miss baselines (validation-clean on non-RT hardware).
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif

#include "includes/locations.glsl"

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

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

// Use the same UBO as main shader
#include "includes/ubo.glsl"
#include "includes/textures.glsl"
#include "includes/tbn.glsl"
#include "includes/triplanar.glsl"
#include "includes/shadows.glsl"


// Water offscreen pass inputs (set 2).
// Hybrid RT: the legacy solid-360 cubemap (binding 1) is REMOVED. Reflection /
// refraction come from hardware ray tracing — either the async RT pipeline
// outputs (bindings 1/2, 1-frame latency, same-queue ordered) or inline ray
// queries against the proxy TLAS (set 0, binding 14) with the sky equirect
// (binding 3) as the miss fallback. The pass stays decoupled from the solid
// pass (no solid color/depth reads); occlusion resolves at composite time.
layout(set = 2, binding = 0) uniform sampler2D waterBackDepthTex; // back-face depth for volume thickness
layout(set = 2, binding = 1) uniform sampler2D rtReflectTex;   // RT pipeline reflection (rgb, a=valid)
layout(set = 2, binding = 2) uniform sampler2D rtRefractTex;   // RT pipeline refraction (rgb, a=thickness or -1)
layout(set = 2, binding = 3) uniform sampler2D skyEquirectTex; // sky for RT miss/fallback

#ifdef RT_ENABLED
#include "includes/rt_params.glsl"
layout(set = 0, binding = 14) uniform accelerationStructureEXT rtTlas;
layout(set = 0, binding = 17) uniform RTBlock { RayTracingParamsGLSL rt; };
layout(set = 0, binding = 18) readonly buffer RTMeta { RTProxyMetaGLSL rtMetas[]; };

// Trace one water secondary ray through the proxy TLAS.
// refraction=true: returns rgb = shaded hit (or sky on miss), a = hitT
//   (capped at rt.water.y, feathered toward deep for coarse boxes) or
//   RT_DEEP_WATER on miss (deep water).
// refraction=false: returns rgb = shaded hit feathered toward sky for coarse
//   boxes (or sky on miss/coarse); a is unused by the reflection consumer.
// Macro shadows stay CSM-owned: hits get ambient + sun diffuse only.
// Water-originated rays trace with the solid-only mask: the water surface
// itself is not in the solid BLAS (origins, never targets), and the water
// BLAS is skipped to avoid self-hits.
vec4 rtTraceWater(vec3 origin, vec3 dir, float tMax, bool refraction) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsOpaqueEXT, RT_RAY_MASK_SOLID,
        origin, 0.05, dir, tMax);
    while (rayQueryProceedEXT(rq)) {}
    vec3 sky = texture(skyEquirectTex, rtDirToEquirectUV(normalize(dir))).rgb;
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        // Deep-water marker (refraction) or plain sky (reflection).
        return refraction ? vec4(sky, RT_DEEP_WATER) : vec4(sky, -1.0);
    }
    float hitT = rayQueryGetIntersectionTEXT(rq, true);
    uint boxIdx = rtBoxIndex(uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true)),
                             rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
    RTProxyMetaGLSL meta = rtMetas[boxIdx];
    // Coarse boxes cannot resolve shallow detail (same rule as rchit): feather
    // toward deep/sky so box-size contours never print as razor lines.
    float f = rtCoarseFeather(meta.extra.x, rt.water.z);
    vec3 hitPos = origin + dir * hitT;
    bool exiting = !rayQueryGetIntersectionFrontFaceEXT(rq, true);
    vec3 boxN = rtBoxNormal(hitPos, meta.minAndMatId.xyz, meta.maxAndFlags.xyz, exiting);
    float ndl = max(dot(boxN, normalize(rt.sunDir.xyz)), 0.0);
    // Textured hit albedo: triplanar sample at the hit point with the box
    // face normal, so refracted/reflected terrain reads as real textured
    // ground (not flat proxy averages). Weights mirror main.frag.
    int hitMat = int(meta.minAndMatId.w + 0.5);
    vec3 triW = abs(boxN);
    float twt = ubo.triplanarSettings.x;
    vec3 wwt = max(vec3(0.0), triW - vec3(twt));
    float wwe = max(1.0, ubo.triplanarSettings.y);
    wwt = pow(wwt, vec3(wwe));
    triW = wwt / (wwt.x + wwt.y + wwt.z + 1e-6);
    vec3 hitAlbedo = computeTriplanarAlbedo(hitPos, triW, hitMat, boxN);
    vec3 color = hitAlbedo * (rt.sunColor.rgb * (0.55 + 0.45 * ndl) + vec3(0.09, 0.12, 0.15));
    if (!refraction) {
        // Reflection ignores thickness: feather coarse hits toward sky.
        return vec4(mix(color, sky, f), -1.0);
    }
    // Cap the reported underwater length at rt.water.y (Beer-Lambert guard),
    // then feather coarse hits toward deep thickness.
    float cap = max(rt.water.y, 0.0);
    float t = mix(min(hitT, cap), RT_DEEP_THICKNESS, f);
    return vec4(color, t);
}
#endif

// Near/far planes for linearizing depth – read from UBO passParams (z = near, w = far)
// so they always match the glm::perspective call on the CPU side.

#include "includes/hsv.glsl"

// Linearize depth from Vulkan [0,1] depth buffer to eye-space distance.
// With GLM_FORCE_DEPTH_ZERO_TO_ONE the projection maps z_eye to [0,1]:
//   d = f*(z - n) / (z*(f - n))   =>   z = n*f / (f - d*(f - n))
float linearizeDepth(float depth) {
    float nearPlane = ubo.passParams.z;
    float farPlane  = ubo.passParams.w;
    return (nearPlane * farPlane) / (farPlane - depth * (farPlane - nearPlane));
}

#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"
#include "includes/voronoi.glsl"

// Equirectangular UV for a direction (matches postprocess dirToEquirectUV).
// Local copy so both RT and non-RT variants share the miss/fallback mapping.
vec2 waterDirToEquirectUV(vec3 dir) {
    const float PI = 3.14159265358979;
    vec2 uv;
    uv.x = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return uv;
}

void main() {
    // Get water parameters from SSBO indexed by fragment brushIndex
    WaterParamsGPU wp = waterParams[fragBrushIndex];
    float time = waterRenderUBO.timeParams.x;

    // Water rendering parameters from selected water params
    float refractionStrength = wp.params1.x;
    float fresnelPower = wp.params1.y;
    float transparency = wp.params1.z;
    float waterTint = wp.params2.x;
    float noiseScale = wp.params2.y;
    int noiseOctaves = int(max(wp.params2.z, 1.0));
    float noisePersistence = wp.params2.w;
    float noiseTimeSpeed = wp.params3.x;
    float noiseLacunarity = wp.params3.y;
    float reflectionStrength = wp.params1.w;
    float specularIntensity = wp.params3.z;
    float specularPowerParam = wp.params3.w;
    float glitterIntensity = wp.deepColor.w;

    // Feature toggles and blur parameters
    bool enableReflection = wp.reserved1.x > 0.5;
    bool enableRefraction = wp.reserved1.y > 0.5;
    // During 360 cubemap capture, skip reflection/refraction to avoid feedback.
    const bool captureMode = ubo.materialFlags.x > 0.5;
    if (captureMode) { enableReflection = false; enableRefraction = false; }
    bool enableBlur       = wp.reserved1.z > 0.5;
    float blurRadius      = wp.reserved1.w;
    int   blurSamples     = max(int(wp.reserved2.x), 1);
    float volumeBlurRate  = wp.reserved2.y;

    // Apply noise time speed
    float animTime = time * noiseTimeSpeed;

    // Compute the clip → screen UV once and reuse it everywhere (the conversion
    // is identical for every use below).
    vec2 screenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;

    // NOTE: solid-occlusion discard (was: if solid depth < fragment depth,
    // discard) has been REMOVED from this shader. The water pass no longer reads
    // the solid depth texture; occlusion against solid geometry is now resolved
    // at the composite stage (postprocess.frag) where both the solid depth and
    // the water geometry depth are available.

    // === WATER VOLUME THICKNESS ===
    // Compute volume thickness from back-face depth (rendered with reversed winding)
    // before the normal computation, so we can modulate bump amplitude.
    // The solid scene depth is no longer available here; thickness is therefore
    // measured entirely from the water front/back faces. This means a flat
    // height-field surface (back-face ≈ front-face) reports ~0 thickness, which
    // is the expected "thin water" case; genuinely thick water bodies (where the
    // back-face pass renders a distant bottom) keep their measured thickness.
    float backFaceDepthRaw = texture(waterBackDepthTex, screenUV).r;
    float frontFaceLinear  = linearizeDepth(gl_FragCoord.z);
    float backFaceLinear   = linearizeDepth(backFaceDepthRaw);

    // Reconstruct world-space positions for a true view-ray thickness measurement.
    mat4 invViewProj = ubo.invViewProjection;
    vec4 backFaceWorldH = invViewProj * vec4(screenUV * 2.0 - 1.0, backFaceDepthRaw, 1.0);
    vec3 backFaceWorld = backFaceWorldH.xyz / backFaceWorldH.w;

    vec3 worldFrontPos = fragPosWorld;
    vec3 worldRayDir = normalize(worldFrontPos - ubo.viewPos.xyz);
    float backFaceThickness = max(dot(backFaceWorld - worldFrontPos, worldRayDir), 0.0);
    // A single-layer height-field surface (flat plane or tessellated waves) has
    // backFaceThickness ≈ 0 because the back-face geometry is co-planar with the
    // front face.  Comparing raw depth values with a fixed epsilon is unreliable
    // across different near/far planes and camera distances.  Instead, check the
    // already-computed world-space thickness: only trust the back face when it
    // represents a genuinely thick water body (>= 5 cm), not a thin surface.
    const float kMinVolumeThickness = 0.05; // 5 cm world-space
    // Also reject backFaceDepthRaw == 1.0 (depth-clear value = no geometry rendered).
    bool hasValidBackFace = (backFaceDepthRaw < 0.9999) && (backFaceThickness > kMinVolumeThickness);
    float waterThickness  = hasValidBackFace ? backFaceThickness : 0.0;

    // Depth-based modulation factors (exponential ramp)
        float volumeBlurFactor = (volumeBlurRate > 0.0) ? (1.0 - exp(-waterThickness * volumeBlurRate)) : 1.0;
        blurRadius *= volumeBlurFactor;
    
    
    // Common bump parameters.
    float eps = 0.5;

    // Build the surface frame from the UNDISPLACED base normal so the
    // per-fragment analytic gradient fully defines the shading normal. This
    // keeps the fine ripple detail regardless of tessellation: when tessellation
    // is enabled the geometry is displaced (silhouette/refraction look right)
    // but the normals are still evaluated per fragment at full resolution,
    // instead of being limited to the interpolated per-vertex normal.
    vec3 flatN = normalize(fragBaseNormal);
    vec3 up = abs(flatN.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, flatN));
    vec3 B  = cross(flatN, T);

    vec3 normal;
    {
        // Analytic height-field normal from a single noise evaluation (the
        // gradient replaces 5 finite-difference samples).  Uses the UNDISPLACED
        // base position (fragBasePos.xyz) and the FINAL bump amplitude
        // (fragBasePos.w) straight from the TES so the per-fragment normal is
        // evaluated on the exact same height field the displaced geometry was
        // built from — same noise, same amplitude, same depth/volume
        // attenuation.  This keeps shading normals perfectly consistent with the
        // tessellated surface.
        vec3  basePos = fragBasePos.xyz;
        float bumpAmp = fragBasePos.w;
        vec4 wave = waterWaveSample(basePos, animTime, noiseScale, noiseOctaves, noisePersistence, noiseLacunarity, bumpAmp, 1.0);
        float dhdT = dot(wave.yzw, T);
        float dhdB = dot(wave.yzw, B);

        normal = normalize(flatN - dhdT * T - dhdB * B);
    }

    
    // Normalize vectors
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    // Keep the normal facing the visible side to avoid flat/dark lighting from flipped orientation.
    if (dot(normal, viewDir) < 0.0) normal = -normal;
    vec3 lightDir = normalize(-ubo.lightDir.xyz);
    
    // Base screen UV already computed at the top of main() and reused above.
    int dbgMode = int(ubo.debugParams.x + 0.5);

    // === HYBRID RT STATE ===
    // usePipe = sample the async RT pipeline outputs (1-frame latency);
    // otherwise inline ray queries (no latency) or the sky fallback.
    // rt.debug.w carries settings.rtWaterPipeline (1=pipeline, 0=inline).
    bool rtReady = false;
    bool usePipe = false;
#ifdef RT_ENABLED
    rtReady = (rt.debug.y > 0.5);
    usePipe = rtReady && (rt.debug.w > 0.5);
#endif

    // === PERLIN NOISE-BASED REFRACTION ===
    // Generate refraction distortion from shared FBM helper.
    vec2 refractionNoise = waterRefractionNoise(
        fragPos.xyz,
        noiseScale,
        animTime,
        int(noiseOctaves),
        noisePersistence,
        noiseLacunarity
    );
    
    // Combine noise layers for complex refraction pattern
    vec2 refractionOffset = enableRefraction
        ? refractionNoise * refractionStrength
        : vec2(0.0);
    
    // Reduce refraction at edges (to avoid sampling outside screen)
    float edgeFade = smoothstep(0.0, 0.1, screenUV.x) * smoothstep(1.0, 0.9, screenUV.x) *
                     smoothstep(0.0, 0.1, screenUV.y) * smoothstep(1.0, 0.9, screenUV.y);
    refractionOffset *= edgeFade;
    
    // Sample refraction via HARDWARE RAY TRACING (§10: Snell, IOR 1.333).
    // Path selection: async RT pipeline outputs (half-res, 1-frame latency)
    // when enabled, else inline ray queries (full-res, no latency), else the
    // sky equirect. The legacy 360° capture cubemap is removed.
    // rtThickness carries the RT underwater path length when provided (>= 0).
    vec3 sceneColor = vec3(0.0);
    float rtThickness = -1.0;
#ifndef RT_ENABLED
    float waterIor = 1.333333; // approximate index of refraction for water
    float maxRefr = 300.0;
#else
    // Shared refraction constants (also read by the Beer-Lambert block below:
    // the deep-water marker substitutes maxRefr as a caustic/viz thickness).
    float maxRefr = max(rt.distances.y, 1.0);
#endif
    if (enableRefraction) {
#ifdef RT_ENABLED
        float waterIor = clamp(rt.water.x, 1.0, 2.5);
#endif
        // Approximate air->water refraction. GLSL `refract` expects the incident
        // vector (eye -> surface), i.e. -viewDir; the result is the true
        // transmitted ray pointing INTO the water toward the underwater scene.
        vec3 refrRay = refract(-viewDir, normal, 1.0 / waterIor);
        if (length(refrRay) < 1e-5) {
            // fallback to reflection if total internal reflection occurs
            refrRay = reflect(-viewDir, normal);
        }
        // Apply Perlin-based angular distortion so refractionStrength visibly
        // warps the lookup. The offset is expressed in the surface tangent
        // frame (T,B) so the distortion follows the wave orientation.
        refrRay = normalize(refrRay + T * refractionOffset.x + B * refractionOffset.y);
        bool refrResolved = false;
#ifdef RT_ENABLED
        if (usePipe && rt.toggles.y > 0.5) {
            vec4 pipeRefr = texture(rtRefractTex, screenUV);
            if (pipeRefr.a >= 0.0) {
                sceneColor = pipeRefr.rgb;
                rtThickness = pipeRefr.a;
                refrResolved = true;
            }
        }
        if (!refrResolved && rtReady && rt.toggles.y > 0.5) {
            vec4 hit = rtTraceWater(fragPosWorld - normal * 0.05, refrRay, maxRefr, true);
            sceneColor = hit.rgb;
            // a >= 0 always from rtTraceWater: capped path length on hit, or
            // RT_DEEP_WATER marker on miss (deep, unresolved water).
            rtThickness = hit.a;
            refrResolved = true;
        }
#endif
        if (!refrResolved) {
            // Sky fallback (also the non-RT path): refracted sky through the
            // surface, no thickness (back-face raster method covers thickness).
            sceneColor = texture(skyEquirectTex, waterDirToEquirectUV(refrRay)).rgb;
        }
    }

    // === RT THICKNESS + BEER-LAMBERT (§11) ===
    // Thickness source priority: the RASTER back-face measurement is the true
    // water column and — built from shared boundary vertices — is continuous
    // across chunk borders. The RT underwater path length is only a fallback
    // for fragments where the back face measured no bottom.
    //
    // Why not always prefer RT: each solid proxy box has a FLAT top per chunk,
    // so neighbouring boxes report stepped hitT values along their shared
    // faces. In shallow water those steps land inside the tint ramp and print
    // the proxy grid onto the water as filled tiles ("overlayed chunks").
    // Attenuate the refracted light BEFORE the tint mix:
    // T = exp(-absorption * thickness).
    //
    // Deep-water handling: a refraction ray that misses the terrestrial
    // proxies is marked RT_DEEP_WATER. It substitutes the deep-water tint as
    // the base color (never attenuated to black), keeping the raster thickness
    // when a real bottom was measured.
    vec3 absorbCoeff = vec3(0.35, 0.12, 0.08);
    float absorbScale = 1.0;
    bool rtDeepMiss = false;
    // Water tint colors from UBO (declared here — the deep-miss path needs
    // them, and the tint composition below reuses them).
    vec3 deepTint = wp.deepColor.rgb;
    vec3 shallowTint = wp.shallowColor.rgb;
#ifdef RT_ENABLED
    // Hash-dither resolved RT hit lengths (±0.3 m). Per-chunk flat box tops
    // quantize the true depth into steps; undithered, those steps print as
    // terrace bands. Dithered they degrade to grain, which reads as water
    // noise. Miss marker and "no RT" (-1) are never touched.
    if (rtReady && rt.toggles.z > 0.5 && rtThickness >= 0.0 && rtThickness < RT_DEEP_WATER) {
        float h = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        rtThickness = max(rtThickness + (h - 0.5) * 0.6, 0.0);
    }
    if (rtReady && rt.toggles.z > 0.5 && rtThickness >= 0.0) {
        rtDeepMiss = (rtThickness >= RT_DEEP_WATER);
        absorbCoeff = rt.absorption.rgb;
        absorbScale = 1.0;
        if (rtDeepMiss) {
            sceneColor = deepTint;
            // No raster bottom: keep the full path as a proxy thickness so
            // depth-dependent effects (caustics, debug views) stay sane.
            if (!hasValidBackFace)
                waterThickness = maxRefr * max(rt.absorption.a, 0.0);
            // else: raster back-face thickness stands (continuous, true depth).
        } else if (!hasValidBackFace) {
            // RT hit length is the only depth signal available.
            // (Already capped at rt.water.y upstream: rgen + rtTraceWater.)
            waterThickness = rtThickness * max(rt.absorption.a, 0.0);
        }
        // else: raster back-face thickness stands (continuous, true depth).
    } else {
        absorbCoeff = rt.absorption.rgb;
    }
#endif
    // Clamp the optical thickness so transmittance never falls below ~e^-2.5:
    // deep hits stay readable dark teal instead of blacking out, while the
    // depth gradient is preserved. (Shallow and mid ranges never reach the
    // clamp, so their look is unchanged.)
    vec3 transmittance = exp(-min(absorbCoeff * max(waterThickness * absorbScale, 0.0),
                                  vec3(2.5)));
    if (!rtDeepMiss) sceneColor *= transmittance;

    // sceneDepthRaw already sampled once at the top of main() and reused.
    // Sample g-buffer attachments produced by the main pass (if available)

    // === DEPTH-BASED EFFECTS ===
    float waterDepthRaw = gl_FragCoord.z;

    // Depth difference between the water back face and the water front face. The
    // solid scene depth is no longer available here (occlusion is handled at
    // composite time), so the "water column" is measured purely from the water
    // volume. For flat water (no valid back face) this collapses to 0, i.e. the
    // thinnest possible water, which is the correct degenerate case.
    float backFaceDiff = max(backFaceLinear - frontFaceLinear, 0.0);
    float depthDiff = hasValidBackFace ? backFaceDiff : 0.0;
    
    // Depth-based color fade (deeper = more tinted)
    float depthFalloff = wp.waveParams.w;
    if (depthFalloff <= 0.0) depthFalloff = 0.02;
    float depthFade = 1.0 - exp(-depthDiff * depthFalloff);
    
    // === FRESNEL EFFECT ===
    // Schlick approximation anchored at the physical air->water base
    // reflectance F0 = 0.02: looking straight down reflects ~2% of the
    // environment, grazing angles approach a full mirror. fresnelPower
    // (default 5 = standard Schlick) shapes the transition curve.
    float fresnelCurve = pow(1.0 - max(dot(viewDir, normal), 0.0), clamp(fresnelPower, 1.0, 8.0));
    float fresnel = clamp(0.02 + 0.98 * fresnelCurve, 0.0, 1.0);
    
    // === SPECULAR LIGHTING (Perlin noise-based) ===
    vec3 halfDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(normal, halfDir), 0.0);
    
    // Main specular highlight with noise perturbation
    float specNoise = 0.8 + 0.4 * waterFbmNoise(fragPos.xyz, noiseScale, animTime, 1.0,
                                                max(int(noiseOctaves), 1), noisePersistence, noiseLacunarity, vec3(0.0));
    float specular = pow(specAngle, specularPowerParam) * specNoise;
    vec3 specularColor = ubo.lightColor.xyz * specular * specularIntensity;
    
    // Sun glitter: high-frequency noise-based sparkles
    if (glitterIntensity > 0.0) {
        float glitterNoise = waterFbmNoise(fragPos.xyz, noiseScale * 3.0, animTime, 3.0,
                                           max(int(noiseOctaves) - 2, 1), noisePersistence, noiseLacunarity, vec3(0.0));
        float glitterThreshold = 0.7 + 0.2 * waterFbmNoise(fragPos.xyz, noiseScale * 0.5, animTime, 0.5,
                                                           max(int(noiseOctaves), 1), noisePersistence, noiseLacunarity, vec3(0.0));
        float glitter = smoothstep(glitterThreshold, 1.0, glitterNoise) * pow(specAngle, 32.0);
        specularColor += ubo.lightColor.xyz * glitter * glitterIntensity;
    }
    
    // === REFLECTION (hardware RT §9) ===
    // Standard convention: reflect the eye-to-surface incident (-viewDir).
    // Pipeline outputs (half-res, 1-frame latency) win when enabled and valid;
    // else inline ray queries; else the sky equirect. No 360 cubemap.
    vec3 reflectDir = reflect(-viewDir, normal);

    vec3 skyColor = vec3(0.0);
    bool reflResolved = false;
#ifdef RT_ENABLED
    if (usePipe && rt.toggles.x > 0.5) {
        vec4 pipeRefl = texture(rtReflectTex, screenUV);
        if (pipeRefl.a > 0.5) {
            skyColor = pipeRefl.rgb;
            reflResolved = true;
        }
    }
    if (!reflResolved && rtReady && rt.toggles.x > 0.5) {
        float maxRefl = max(rt.distances.x, 1.0);
        vec4 hit = rtTraceWater(fragPosWorld + normal * 0.05, normalize(reflectDir), maxRefl, false);
        skyColor = hit.rgb;
        reflResolved = true;
    }
#endif
    if (!reflResolved) {
        skyColor = texture(skyEquirectTex, waterDirToEquirectUV(normalize(reflectDir))).rgb;
    }

    // === AERIAL DETAIL FADE (§10/§11) ===
    // Blend ray-traced detail toward analytic fallbacks with fragment distance.
    // Proxy boxes are per-chunk flats: beyond the near field their tops,
    // footprints and hit/miss classification imprint box-shaped steps onto
    // refraction color, reflection color and thickness — and every such step
    // is a potential razor line (LOD frontiers are straight, full-width and
    // camera-following). Distance is continuous, so fading by distance cannot
    // create edges by construction; it only removes them. Near field (<120 m,
    // where boxes are tightest) keeps pixel-identical RT detail; far field
    // converges to deep tint + sky, i.e. honest aerial perspective. Debug
    // views below read the pre-fade snapshots.
    vec3 dbgSceneColor = sceneColor;
    vec3 dbgReflColor = skyColor;
    // Translucency (final alpha) must use the TRUE local thickness, not the
    // faded one: the fade inflates distant shallows toward deep, which would
    // force distant shores opaque. Snapshot before fading.
    float thicknessForAlpha = waterThickness;
    {
        const float fadeStart = 120.0;
        const float fadeEnd = 400.0;
        float fragDist = length(fragPosWorld - ubo.viewPos.xyz);
        float detailFade = smoothstep(fadeStart, fadeEnd, fragDist);
        if (detailFade > 0.0) {
            vec3 skyRef = texture(skyEquirectTex, waterDirToEquirectUV(normalize(reflectDir))).rgb;
            skyColor = mix(skyColor, skyRef, detailFade);
            sceneColor = mix(sceneColor, deepTint, detailFade);
            waterThickness = mix(waterThickness, maxRefr, detailFade);
        }
    }

    // Uniform reflection toggle: when set, apply reflectionStrength uniformly
    // instead of modulating by Fresnel. This flag is stored in reserved2.w
    // (see WaterParamsGPU.reserved2.w).
    bool uniformReflection = wp.reserved2.w > 0.5;


    // === SHADOW ON WATER ===
    // Direct shadow map sampling is disabled for water because the water
    // surface sits at a different height than the terrain, causing the
    // EVSM shadow to misalign with the terrain shadow visible through
    // refraction.  This misalignment creates a visible bright halo around
    // vegetation shadows.  The refracted scene (sceneColorTex) already
    // carries the correct terrain/vegetation shadows, so the water
    // surface is darkened naturally through refraction.
    float shadow = 0.0;
    
    // === WATER COLOR COMPOSITION ===
    // Water tint colors from UBO (declared in the Beer-Lambert block above).


    // Caustic parameters
    vec3 causticColor = wp.causticColor.rgb;
    float causticScale = wp.causticParams.x;
    float causticIntensity = wp.causticParams.y;
    float causticPower = wp.causticParams.z;
    int causticType = int(round(clamp(wp.causticExtraParams.z, 0.0, 1.0)));
    float causticVelocity = wp.causticExtraParams.w;
    float causticAnimTime = animTime * causticVelocity;

    // Compute a volume-based factor from measured water thickness.
    // Use causticParams.w (depth-scale) as a per-layer reference distance; if unset,
    // fall back to a small stable value so division is safe.
    float tintDepthScale = max(wp.causticParams.w, 0.0001);
    float volumeFactor = 1.0f;

    // Water tint color transitions from shallow → deep depending on volume.
    vec3 waterTintColor = mix(shallowTint, deepTint, volumeFactor);

    // Attenuate tint effect when the measured thickness is very small (near zero).
    // This reduces color influence when the front and back faces are nearly coincident.
    float thicknessAttenuation = smoothstep(0.0, max(0.005, tintDepthScale * 0.25), waterThickness);

// Blend scene color with water tint based on both the depth-based fade (depthFade)
// and the measured volume. The `waterTint` parameter scales overall tint strength.
// `transparency` (params1.z, Water widget slider) caps how much tint may cover
// the refraction: 1 = crystal clear (bottom fully visible), 0 = fully
// tintable. At defaults (0.7) the cap is 0.3, matching the previous effective
// range, so existing looks are preserved while the slider comes alive.
    float tintMax = clamp(1.0 - transparency, 0.0, 1.0);
    float tintBlend = clamp(depthFade * waterTint * volumeFactor * thicknessAttenuation, 0.0, tintMax);
    vec3 refractedColor = mix(sceneColor, waterTintColor, tintBlend);
    
    // Mix refracted color with reflection. By default, use Fresnel weighting
    // to increase reflection at grazing angles. If `uniformReflection` is
    // enabled, use `reflectionStrength` directly so reflection appears
    // across all pixels uniformly (useful for debugging/stylized look).
    vec3 waterColor;
    if (captureMode) {
        // 360 capture: no env feedback. Use the base tint so water is not
        // black (refraction/reflection are disabled, leaving sceneColor zero).
        waterColor = waterTintColor;
    } else if (enableReflection) {
        float reflMix = uniformReflection ? reflectionStrength : (fresnel * reflectionStrength);
        waterColor = mix(refractedColor, skyColor, reflMix);
    } else {
        waterColor = refractedColor;
    }
    
    // Add specular highlights (suppressed in shadow)
    waterColor += specularColor * (1.0 - shadow);

    // Darken diffuse water color in shadow
    waterColor *= mix(1.0, 0.55, shadow);

    // (Volume light accumulation removed — caustics only)

    // === CAUSTICS / LIGHT FOCUSING ===
    // Estimate local Jacobian of the refraction offset field by finite-difference
    // along the surface tangent frame (T,B). Negative determinant indicates
    // local focusing (area contraction) which produces brighter caustics.
    // Compute incidence/angle and a simple depth-based ramp for caustic strength
    // Incidence term for caustic modulation
    float lightIncidenceCaust = max(dot(normal, lightDir), 0.0);
    float angularCaust = (causticPower > 0.0) ? pow(lightIncidenceCaust, causticPower) : 1.0;

    // Depth-based ramps: keep a small exponential ramp as an additional softening
    float depthRampCaust = 1.0 - exp(-waterThickness * 0.02);

    // Volume-aware caustics: evaluate the refraction noise Jacobian at both
    // the front surface and at the back-face (bottom) and blend according
    // to water thickness. This approximates how focusing changes through the
    // water column and lets caustics appear where the volume causes stronger
    // focusing on the bottom.
    float causticDepthScale = wp.causticParams.w; // w = depth-scale (world units)
    float depthInfluence = (causticDepthScale > 0.0) ? clamp(waterThickness / causticDepthScale, 0.0, 1.0) : 1.0;

    // Back-face (bottom) sampling: march along the view ray from the front
    // position by the measured water thickness to approximate the bottom
    // world position — used by both caustic modes.
    vec3 backPos = fragPosWorld + worldRayDir * waterThickness;

    // Line-shaped measure parameters
    float lineScale = wp.causticExtraParams.x;
    float lineMix = clamp(wp.causticExtraParams.y, 0.0, 1.0);

    // Prepare outputs that debug and later code expect
    float caustFront = 0.0;
    float caustBack = 0.0;
    float lineFrontRaw = 0.0;
    float lineBackRaw = 0.0;
    float cloudFinal = 0.0;
    float lineFinal = 0.0;
    float lineCombined = 0.0;

    // Skip the entire caustic block when caustics are effectively disabled
    // (intensity ≈ 0) and we are not visualizing them in a debug mode. This
    // avoids hundreds of redundant 4D-noise evaluations per deep-water fragment
    // with zero visual change where caustics are off.
    bool causticDebugMode = (dbgMode >= 39 && dbgMode <= 42);
    if (causticIntensity > 0.001 || causticDebugMode) {

    // Reuse the refraction noise already computed above (same fragPos, same
    // octaves/scale/time) for the front-face caustic Jacobian instead of
    // recomputing waterRefractionNoise a second time this fragment.
    vec2 caustRef0 = refractionNoise * refractionStrength;

    // Compute only the selected caustic noise per-fragment
    if (causticType == 1) {
        // VORONOI-based measures (Worley noise) — jitter feature points using FBM
        vec2 vorFront = voronoi3d(fragPos * causticScale, causticAnimTime, noiseScale, 0.5, noiseOctaves, noisePersistence, noiseLacunarity);
        vec2 vorBack  = voronoi3d(backPos * causticScale, causticAnimTime, noiseScale, 0.5, noiseOctaves, noisePersistence, noiseLacunarity);
        float f1f = vorFront.x;
        float f2f = vorFront.y;
        float f1b = vorBack.x;
        float f2b = vorBack.y;

        caustFront = max(1.0 - f1f, 0.0);
        caustBack  = max(1.0 - f1b, 0.0);
        lineFrontRaw = max(1.0 - (f2f - f1f) * lineScale, 0.0);
        lineBackRaw  = max(1.0 - (f2b - f1b) * lineScale, 0.0);

        // Compose final cloud/line measures and apply power/intensity
        float cloudCombined = mix(caustFront, caustBack, depthInfluence);
        cloudFinal = pow(max(cloudCombined, 1e-6), causticPower);
        lineCombined = mix(lineFrontRaw, lineBackRaw, depthInfluence);
        lineFinal = pow(max(lineCombined, 1e-6), causticPower);
    } else {
        // PERLIN-based measures (existing Jacobian method)
        // Front face: reuse caustRef0 (= waterRefractionNoise(fragPos)) and only
        // compute the two tangent-perturbed samples that differ.
        vec2 refT = waterRefractionNoise(fragPos + eps * T, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence, noiseLacunarity) * refractionStrength;
        vec2 refB = waterRefractionNoise(fragPos + eps * B, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence, noiseLacunarity) * refractionStrength;
        vec2 ddT = (refT - caustRef0) / eps;
        vec2 ddB = (refB - caustRef0) / eps;
        float detJFront = ddT.x * ddB.y - ddT.y * ddB.x;
        float trFront = ddT.x + ddB.y;
        float anisFront = sqrt(max(trFront * trFront - 4.0 * detJFront, 0.0));

        caustFront = max(-detJFront * causticScale, 0.0);
        lineFrontRaw  = max(anisFront * causticScale * lineScale, 0.0);

        // Back face: only needed for genuinely thick water volumes. For flat /
        // co-planar surfaces (hasValidBackFace == false) the bottom Jacobian
        // equals the surface Jacobian, so reuse the front estimate instead of
        // issuing the full back-face FBM sample set.
        if (hasValidBackFace) {
            vec2 caustRef0b = waterRefractionNoise(backPos.xyz, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence, noiseLacunarity) * refractionStrength;
            vec2 refTb = waterRefractionNoise(backPos + eps * T, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence, noiseLacunarity) * refractionStrength;
            vec2 refBb = waterRefractionNoise(backPos + eps * B, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence, noiseLacunarity) * refractionStrength;
            vec2 ddTb = (refTb - caustRef0b) / eps;
            vec2 ddBb = (refBb - caustRef0b) / eps;
            float detJBack = ddTb.x * ddBb.y - ddTb.y * ddBb.x;
            float trBack = ddTb.x + ddBb.y;
            float anisBack = sqrt(max(trBack * trBack - 4.0 * detJBack, 0.0));

            caustBack = max(-detJBack * causticScale, 0.0);
            lineBackRaw = max(anisBack * causticScale * lineScale, 0.0);
        } else {
            caustBack = caustFront;
            lineBackRaw = lineFrontRaw;
        }

        // Compose final cloud/line measures and apply power/intensity
        float cloudCombined = mix(caustFront, caustBack, depthInfluence);
        cloudFinal = pow(max(cloudCombined, 1e-6), causticPower);
        lineCombined = mix(lineFrontRaw, lineBackRaw, depthInfluence);
        lineFinal = pow(max(lineCombined, 1e-6), causticPower);
    }

    } // end caustic-intensity / debug guard

    // Blend cloud vs line patterns, then apply intensity and modulations
    float caustRaw = mix(cloudFinal, lineFinal, lineMix);
    float caustic = caustRaw * causticIntensity * depthRampCaust * angularCaust * edgeFade * (1.0 - shadow);

    waterColor += causticColor * caustic;

    // Apply per-vertex HSV: rotate hue, offset saturation, scale value
    vec3 hsvColor = fragHSV;
    vec3 texHSV = rgbToHsv(waterColor);
    texHSV.x = mod(texHSV.x + hsvColor.x, 360.0);
    texHSV.y = clamp(texHSV.y * (hsvColor.y * 2.0), 0.0, 1.0);
    texHSV.z *= hsvColor.z * 2.0;
    waterColor = hsvToRgb(texHSV);

    // === FINAL OUTPUT ===
    // True translucency through the composite blend (mix(baseColor,
    // waterColor, waterAlpha)): shallow water reveals the bright rasterized
    // bottom beneath instead of replacing it with the dark proxy color, while
    // deep water stays opaque. Driven by the Transparency slider so 1.0 gives
    // crystal shallows and 0.0 restores legacy fully-opaque water. Capture
    // mode keeps alpha 1 (no solid backdrop is composited there).
    float thicknessFrac = clamp(thicknessForAlpha / 3.0, 0.0, 1.0); // ~3 m -> opaque
    float alpha = mix(1.0, thicknessFrac, clamp(transparency, 0.0, 1.0));
    if (captureMode) alpha = 1.0;
    outColor = vec4(waterColor, alpha);

    // Debug: visual displacement color when debug mode set to 32 ("Water Displacement")
    if (dbgMode == 32) {
        // Prefer tessellation-provided debug value when available (fragDebug).
        // But also compute a per-fragment approximation of the bump displacement so the debug
        // mode works even when tessellation is disabled.
        float timeDebug = waterRenderUBO.timeParams.x;
        float waveScaleDbg = 1.0;  // No longer in passParams (z=nearPlane now)

        float bumpAmpDbg = wp.waveParams.z;

        float animTimeDbg = timeDebug * wp.params3.x;
        float waveDisplacementDbg = waterWaveDisplacement(
            fragPos.xyz,
            animTimeDbg,
            noiseScale,
            noiseOctaves,
            noisePersistence,
            noiseLacunarity,
            bumpAmpDbg,
            waveScaleDbg
        );

        float maxExpected = bumpAmpDbg * waveScaleDbg * 1.5;
        float normDisp = clamp((waveDisplacementDbg / maxExpected) * 0.5 + 0.5, 0.0, 1.0);

        vec3 debugCol = fragDebug;
        // If tessellation wasn't producing a debug value (likely zero), prefer computed color
        if (length(debugCol) < 0.001) debugCol = vec3(normDisp);
        outColor = vec4(debugCol, 1.0);
    }

    // Debug mode 33: raw sky equirect (reflection of view dir) — verifies the
    // water pass reaches the sky fallback it uses for RT misses.
    if (dbgMode == 33) {
        vec3 sc = texture(skyEquirectTex,
            waterDirToEquirectUV(normalize(reflect(-viewDir, normal)))).rgb;
        outColor = vec4(sc, 1.0);
    }

    // Debug mode 34: screen UV — verifies correct clip → UV conversion.
    if (dbgMode == 34) {
        outColor = vec4(screenUV, 0.0, 1.0);
    }


   // Debug mode 35: water noise
    if (int(ubo.debugParams.x) == 35) {
        outColor = vec4(refractionNoise, 0.5 + 0.5 * (refractionNoise.x - refractionNoise.y), 1.0);
    }

    // Debug mode 36: final displaced normal used by shading.
    if (int(ubo.debugParams.x) == 36) {
        vec3 n = normalize(normal);
        outColor = vec4(n * 0.5 + 0.5, 1.0);
    }

    // --- Reflection sampling debug helpers ---
    // Use the global debug mode (ubo.debugParams.x) to visualize reflection
    // computation steps and RT sampling. Helpful to diagnose orientation.
    if (dbgMode == 37) {
        // Visualize reflection vector (packed to [0,1])
        vec3 vis = reflectDir * 0.5 + 0.5;
        outColor = vec4(vis, 1.0);
    }
    if (dbgMode == 38) {
        // Show RT/sky reflection color actually used by shading
        outColor = vec4(skyColor, 1.0);
    }

    if (dbgMode == 39) {
        vec3 maps = vec3(clamp(caustFront, 0.0, 1.0), clamp(caustBack, 0.0, 1.0), clamp(mix(caustFront, caustBack, depthInfluence), 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 40) {
        vec3 maps = vec3(clamp(lineFrontRaw, 0.0, 1.0), clamp(lineBackRaw, 0.0, 1.0), clamp(lineCombined, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 41) {
        vec3 maps = vec3(clamp(cloudFinal, 0.0, 1.0), clamp(lineFinal, 0.0, 1.0), clamp(caustRaw, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 42) {
        outColor = vec4(vec3(clamp(caustic, 0.0, 1.0)), 1.0);
    }

    // --- Water thickness / depth debug modes (43..48) ---
    // 43: Back-face raw depth (texture sample)
    if (dbgMode == 43) {
        outColor = vec4(vec3(backFaceDepthRaw), 1.0);
    }
    // 44: Front-face linear depth (normalized to [0,1])
    if (dbgMode == 44) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((frontFaceLinear - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 45: Back-face linear depth (normalized to [0,1])
    if (dbgMode == 45) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((backFaceLinear - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 46: Water thickness (normalized by per-layer caustic depth scale or 1.0)
    // (Former debug modes 46/47 showed solid scene depth, which water no longer
    // samples; they have been removed.)
    if (dbgMode == 46) {
        float denom = max(wp.causticParams.w, 1.0);
        float v = clamp(waterThickness / denom, 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }

    // ── Hybrid RT debug views (settings.rtDebugView mirrors) ──
    // 50 = RT/pipeline reflection only, 51 = refraction only,
    // 52 = RT thickness, 53 = Fresnel, 54 = Beer-Lambert transmittance.
    // 50/51 read the pre-aerial-fade snapshots so diagnostics show raw RT.
    if (dbgMode == 50) {
        outColor = vec4(dbgReflColor, 1.0);
    }
    if (dbgMode == 51) {
        outColor = vec4(dbgSceneColor, 1.0);
    }
    if (dbgMode == 52) {
        float thickDenom = 300.0;
#ifdef RT_ENABLED
        thickDenom = max(rt.distances.y, 1.0);
#endif
        outColor = vec4(vec3(clamp(waterThickness / thickDenom, 0.0, 1.0)), 1.0);
    }
    if (dbgMode == 53) {
        outColor = vec4(vec3(clamp(fresnel, 0.0, 1.0)), 1.0);
    }
    if (dbgMode == 54) {
        outColor = vec4(clamp(transmittance, 0.0, 1.0), 1.0);
    }


    // Final outputs: only write the composited water color (RGBA)
    // Normal/mask outputs removed — they are no longer produced by this pass.


}
