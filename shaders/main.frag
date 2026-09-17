
#version 460
// RT_ENABLED is defined for the main_rt.frag.spv variant (hardware RT path).
// Without it the shader uses the procedural-sky fallback with identical miss
// baselines, so non-RT hardware stays validation-clean (no TLAS/ray queries).
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif
#include "includes/locations.glsl"

layout(location = VARY_COLOR) in vec3 fragColor;
layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_TEXWEIGHTS) in vec3 fragTexWeights;
layout(location = VARY_HSV) in vec3 fragHSV;
layout(location = VARY_SHARPNORMAL) in vec3 fragSharpNormal; // face normal computed in TES (sharp)
layout(location = VARY_DEBUG) in vec3 fragTessLevel; // tessellation level heat (tesc output /16, debug 60)

#include "includes/ubo.glsl"

#include "includes/textures.glsl"

#ifndef BRUSH_PASS
layout(set = 1, binding = 0) uniform sampler2D brushDepthTex;
layout(set = 1, binding = 1) uniform sampler2D brushBackFaceDepthTex;
#endif

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "includes/common.glsl"

#include "includes/tbn.glsl"
#include "includes/triplanar.glsl"

#ifndef BRUSH_PASS
#include "includes/shadows.glsl"
#endif

#include "includes/hsv.glsl"

// Hybrid RT declarations (bindings 14/17/18 — TLAS, params, proxy metadata).
// The TLAS holds the stable solid-proxy boxes; water surface is excluded by
// design (origins, never targets). Shaders gate all sampling on rt.debug.y
// (tlasReady): before the first BLAS/TLAS build completes everything falls
// back to sky/CSM so no invalid acceleration structure is ever traced.
// Compiled out entirely without RT_ENABLED (non-RT hardware fallback).
#ifdef RT_ENABLED
#include "includes/rt_params.glsl"
layout(set = 0, binding = 14) uniform accelerationStructureEXT rtTlas;
layout(set = 0, binding = 17) uniform RTBlock { RayTracingParamsGLSL rt; };
layout(set = 0, binding = 18) readonly buffer RTMeta { RTProxyMetaGLSL rtMetas[]; };
// Real scene-geometry reflection lookups: rtScenePrimBase[0] = geometry count,
// [1..N] = first primitive of geometry i (binary-searched for a hit's owner);
// rtSceneAlbedo[i] = that chunk's average albedo.
layout(set = 0, binding = 21) readonly buffer RTScenePrimBase { uint rtScenePrimBase[]; };
layout(set = 0, binding = 22) readonly buffer RTSceneAlbedo { vec4 rtSceneAlbedo[]; };
// Real triangle attributes for hit shading: geometry bases, the merged vertex
// pool (Vertex = 16 floats: position 0-2, normal 8-10) and the index pool.
layout(set = 0, binding = 23) readonly buffer RTSceneGeomInfo { uvec4 rtSceneGeomInfo[]; };
layout(set = 0, binding = 24) readonly buffer RTSceneVerts { float rtSceneVerts[]; };
layout(set = 0, binding = 25) readonly buffer RTSceneIndices { uint rtSceneIndices[]; };
#include "includes/rt_scene_sample.glsl"
#endif

// Screen-space reflection refinement (set 0, bindings 19/20): the *previous*
// frame's solid HDR color/depth. The proxy ray query gives plausible but
// blocky reflections; where the reflected point is on screen, marching the
// real depth buffer and sampling the real color resolves a precise mirror.
layout(set = 0, binding = 19) uniform sampler2D ssrColorTex;
layout(set = 0, binding = 20) uniform sampler2D ssrDepthTex;

// March `dir` from `origin` against the previous frame's solid depth, using the
// previous frame's view-projection so camera motion does not stipple the hit
// test. Returns rgb = reflected scene color, a = confidence (0 = no usable
// hit). Falls back to the proxy/sky result for off-screen occluded rays.
// Marching front to back, the first depth crossing is the first real
// intersection, so any crossing counts as a hit and a binary search refines
// it (step-size independent); the self-UV guard rejects the reflector's own
// surface. `eyeDir` is the unit vector from the surface to the camera: rays
// nearly tangent to the view direction (silhouettes) are where screen-space
// marching is least reliable, so they fade out.
vec4 traceSSR(vec3 origin, vec3 dir, vec3 eyeDir, mat4 prevVP, vec2 selfUV) {
    float nearP = ubo.passParams.z;
    float farP  = ubo.passParams.w;
    float facing = clamp(abs(dot(dir, eyeDir)), 0.0, 1.0);
    float prevT = 0.0;
    float t = 0.25;
    for (int i = 0; i < 64; ++i) {
        // Near field: 1 m steps (thin silhouettes); far field: 13% geometric
        // growth so the remaining steps reach ~4 km without huge near steps.
        t += (i < 20) ? 1.0 : max(2.0, t * 0.13);
        vec3 P = origin + dir * t;
        vec4 clip = prevVP * vec4(P, 1.0);
        if (clip.w <= 0.001 || clip.w > farP * 2.0) break;
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
        float d = textureLod(ssrDepthTex, uv, 0.0).r;
        if (d < 1.0) {
            float sceneEye = (nearP * farP) / (farP - d * (farP - nearP));
            float rayEye = clip.w; // GLM perspective: clip.w == eye depth
            if (rayEye > sceneEye + 0.05 && distance(uv, selfUV) > 0.02) {
                // First crossing: refine with a binary search (5 iterations).
                float lo = prevT, hi = t;
                for (int j = 0; j < 5; ++j) {
                    float mid = 0.5 * (lo + hi);
                    vec4 cm = prevVP * vec4(origin + dir * mid, 1.0);
                    vec2 uvm = cm.xy / cm.w * 0.5 + 0.5;
                    float dm = textureLod(ssrDepthTex, uvm, 0.0).r;
                    float em = (nearP * farP) / (farP - dm * (farP - nearP));
                    if (cm.w > em) { hi = mid; uv = uvm; rayEye = cm.w; sceneEye = em; }
                    else lo = mid;
                }
                // The refined UV may have moved back next to the fragment (a
                // grazing ray that never truly left its own surface): reject.
                if (distance(uv, selfUV) < 0.02) { prevT = t; continue; }
                float edge = smoothstep(0.0, 0.06, uv.x) * smoothstep(0.0, 0.06, 1.0 - uv.x)
                           * smoothstep(0.0, 0.06, uv.y) * smoothstep(0.0, 0.06, 1.0 - uv.y);
                // Crossings further behind the surface are less certain
                // (ray nearly parallel to it): soften instead of hard-cutting.
                float gapFade = 1.0 - clamp((rayEye - sceneEye) / max(1.0, sceneEye * 0.25), 0.0, 0.5);
                return vec4(textureLod(ssrColorTex, uv, 0.0).rgb,
                            edge * gapFade * smoothstep(0.03, 0.35, facing));
            }
        }
        prevT = t;
    }
    return vec4(0.0);
}

// Global toggles
bool roughnessEnabled = ubo.debugParams.y > 0.5;
bool aoEnabled = ubo.debugParams.z > 0.5;

void main() {
    bool isShadowPass = ubo.passParams.x > 0.5;
    // Provide a default color so early debug/special-case returns still
    // produce a valid output for downstream passes.
    outColor = vec4(0.0);

    // Fast-path for shadow pass: skip expensive lighting/texture work.
    if (isShadowPass) {
        outColor = vec4(0.0);
        return;
    }

    // Texture indices, optionally overridden by PAINT/REMOVE mode
    ivec3 texIndices = fragTexIndices;
    vec3 hsvColor = fragHSV;
    float brushRedFade = 0.0;
#ifndef BRUSH_PASS
    bool isPaintMode = ubo.brushParams.y > 1.5;
    bool isRemoveMode = ubo.brushParams.y > 0.5 && ubo.brushParams.y < 1.5;
    if (isPaintMode || isRemoveMode) {
        vec2 brushUV = gl_FragCoord.xy / vec2(textureSize(brushDepthTex, 0));
        float brushFront = texture(brushDepthTex, brushUV).r;
        float brushBack = texture(brushBackFaceDepthTex, brushUV).r;
        float fragDepth = gl_FragCoord.z;
        if (fragDepth >= brushFront && fragDepth <= brushBack) {
            int brushTexIndex = int(ubo.brushParams.x + 0.5);
            texIndices = ivec3(brushTexIndex);
            // Override vertex HSV with the brush's HSV so painted areas get the brush tint
            hsvColor = ubo.brushHSV.xyz;
            if (isRemoveMode) {
                brushRedFade = (sin(ubo.brushParams.w * 6.28318) + 1.0) * 0.5;
            }
        }
    }
#endif

    // Use the three tex indices and barycentric weights provided by the TES for blending
    vec3 w = fragTexWeights;
    vec2 uv = fragUV;
    bool usedTriplanar = false;

    // Geometry normal (world-space) and geometric face normal from derivatives
    vec3 N = normalize(fragNormal);
    vec3 geomN = normalize(cross(dFdx(fragPosWorld), dFdy(fragPosWorld)));
    if (length(geomN) < 1e-5) geomN = N;
    vec3 worldNormal = N;

    // Precompute triplanar blend weights from geometric normal (abs^2 normalized)
    // Compute triplanar weights with a configurable dead-zone threshold and adjustable steepness
    vec3 triW = abs(geomN);

    // Subtract threshold and clamp so small components remain zero until threshold is exceeded
    float t = ubo.triplanarSettings.x; // threshold (0..1)
    vec3 wt = max(vec3(0.0), triW - vec3(t));

    // Apply exponent to make transitions steeper
    float e = max(1.0, ubo.triplanarSettings.y);
    wt = pow(wt, vec3(e));
    float triWSum = wt.x + wt.y + wt.z + 1e-6;
    triW = wt / triWSum;
    
    // Sample albedo texture (triplanar when enabled)
    vec3 albedoColor;
    vec3 tripNormal0;
    vec3 tripNormal1;
    vec3 tripNormal2;

    // Mix triplanar flag across the three materials
    float triFlag = dot(vec3(materials[texIndices.x].triplanarParams.z, materials[texIndices.y].triplanarParams.z, materials[texIndices.z].triplanarParams.z), w);

    // Compute triplanar UVs once per material and reuse across all map fetches below.
    vec2 uv0X, uv0Y, uv0Z;
    vec2 uv1X, uv1Y, uv1Z;
    vec2 uv2X, uv2Y, uv2Z;
    if (w.x > 0.0) computeTriplanarUVs(fragPosWorldNotDisplaced, texIndices.x, geomN, uv0X, uv0Y, uv0Z);
    if (w.y > 0.0) computeTriplanarUVs(fragPosWorldNotDisplaced, texIndices.y, geomN, uv1X, uv1Y, uv1Z);
    if (w.z > 0.0) computeTriplanarUVs(fragPosWorldNotDisplaced, texIndices.z, geomN, uv2X, uv2Y, uv2Z);

    if (triFlag > 0.5) {
        usedTriplanar = true;
        // compute triplanar albedo per-layer then blend
        vec3 a0 = w.x > 0.0 ? computeTriplanarAlbedoUVs(triW, texIndices.x, uv0X, uv0Y, uv0Z) : vec3(0.0);
        vec3 a1 = w.y > 0.0 ? computeTriplanarAlbedoUVs(triW, texIndices.y, uv1X, uv1Y, uv1Z) : vec3(0.0);
        vec3 a2 = w.z > 0.0 ? computeTriplanarAlbedoUVs(triW, texIndices.z, uv2X, uv2Y, uv2Z) : vec3(0.0);
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
        // If normal mapping/triplanar normal enabled per-material or global, compute blended triplanar normal
        float mapFlag0 = materials[texIndices.x].mappingParams.x;
        float mapFlag1 = materials[texIndices.y].mappingParams.x;
        float mapFlag2 = materials[texIndices.z].mappingParams.x;
        if ((mapFlag0 * w.x + mapFlag1 * w.y + mapFlag2 * w.z) > 0.5 || ubo.materialFlags.w > 0.5) {
            tripNormal0 = w.x > 0.0 ? computeTriplanarNormalUVs(triW, texIndices.x, N, uv0X, uv0Y, uv0Z) : vec3(0.0);
            tripNormal1 = w.y > 0.0 ? computeTriplanarNormalUVs(triW, texIndices.y, N, uv1X, uv1Y, uv1Z) : vec3(0.0);
            tripNormal2 = w.z > 0.0 ? computeTriplanarNormalUVs(triW, texIndices.z, N, uv2X, uv2Y, uv2Z) : vec3(0.0);
            vec3 blended = tripNormal0 * w.x + tripNormal1 * w.y + tripNormal2 * w.z;
            worldNormal = normalize(blended);
            
            
            // Diagnostic: detect invalid/degenerate normals and show red so we can find broken pixels
            if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
                outColor = vec4(1.0, 0.0, 0.0, 1.0);
                return;
            }
        }
    } else {
        // Sample albedo from each layer and blend by barycentric weights
        vec3 a0 = texture(albedoArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(texIndices.z))).rgb;
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
    }

    // Compute normal mapping if enabled (per-material or global toggle)
    if (!usedTriplanar && ((materials[texIndices.x].mappingParams.x * w.x + materials[texIndices.y].mappingParams.x * w.y + materials[texIndices.z].mappingParams.x * w.z) > 0.5 || ubo.materialFlags.w > 0.5)) {
        // Sample normal map per-layer and blend in tangent space
        vec3 n0 = texture(normalArray, vec3(uv, float(texIndices.x))).rgb * 2.0 - 1.0;
        vec3 n1 = texture(normalArray, vec3(uv, float(texIndices.y))).rgb * 2.0 - 1.0;
        vec3 n2 = texture(normalArray, vec3(uv, float(texIndices.z))).rgb * 2.0 - 1.0;
        vec3 nmap = normalize(n0 * w.x + n1 * w.y + n2 * w.z);
        // Build TBN matrix from geometry for UV-space normal mapping
        vec3 T = normalize(dFdx(fragPosWorld));
        vec3 B = normalize(cross(N, T));
        T = normalize(cross(B, N)); // re-orthogonalize
        mat3 TBN = mat3(T, B, N);
        worldNormal = normalize(TBN * nmap);
        if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    // Sample roughness map (R channel)
    float roughnessValue;
    if (usedTriplanar) {
        float r0 = w.x > 0.0 ? computeTriplanarRoughnessUVs(triW, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
        float r1 = w.y > 0.0 ? computeTriplanarRoughnessUVs(triW, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
        float r2 = w.z > 0.0 ? computeTriplanarRoughnessUVs(triW, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
        roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    } else {
        float r0 = texture(roughnessArray, vec3(uv, float(texIndices.x))).r;
        float r1 = texture(roughnessArray, vec3(uv, float(texIndices.y))).r;
        float r2 = texture(roughnessArray, vec3(uv, float(texIndices.z))).r;
        roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    }
    if (!roughnessEnabled) roughnessValue = 0.0;

    // Sample ambient occlusion map (R channel)
    float ambientOcclusion;
    if (usedTriplanar) {
        float ao0 = w.x > 0.0 ? computeTriplanarAOUVs(triW, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
        float ao1 = w.y > 0.0 ? computeTriplanarAOUVs(triW, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
        float ao2 = w.z > 0.0 ? computeTriplanarAOUVs(triW, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
        ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);
    } else {
        float ao0 = texture(aoArray, vec3(uv, float(texIndices.x))).r;
        float ao1 = texture(aoArray, vec3(uv, float(texIndices.y))).r;
        float ao2 = texture(aoArray, vec3(uv, float(texIndices.z))).r;
        ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);
    }

    // Lighting calculation
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, toLight), 0.0);

    // Shadow calculation (CSM authoritative macro sun-shadow — NEVER replaced
    // by RT; see §2/§21). RT adds an optional selective local/contact term
    // below, combined without double-darkening.
    float shadow = 0.0;
#ifndef BRUSH_PASS
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    if (ubo.shadowEffects.w > 0.5) {
        if (NdotL > 0.01) {
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, fragPosWorld, bias);
        } else {
            shadow = 1.0;
        }
    }
#endif
    // RT local/contact shadows (default OFF — CSM-only is authoritative).
    // Only where CSM says lit: nearby proxy geometry adds high-frequency
    // contact occlusion the cascades cannot resolve. CSM-shadowed pixels keep
    // the CSM result (no double-darkening: combined as independent occluders).
    // Compiled out without RT_ENABLED (rtLocalShadow stays 0 = CSM-only).
    float rtLocalShadow = 0.0;
#if !defined(BRUSH_PASS) && defined(RT_ENABLED)
    bool rtReady = (rt.debug.y > 0.5);
    if (rtReady && rt.toggles.w > 0.5 && shadow < 0.5 && NdotL > 0.01) {
        vec3 sunDir = -normalize(ubo.lightDir.xyz);
        float shadowDist = max(rt.distances.z, 0.5);
        rayQueryEXT shadowRQ;
        rayQueryInitializeEXT(shadowRQ, rtTlas, gl_RayFlagsOpaqueEXT, RT_RAY_MASK_ALL,
            fragPosWorld + worldNormal * 0.05, 0.05, sunDir, shadowDist);
        while (rayQueryProceedEXT(shadowRQ)) {}
        if (rayQueryGetIntersectionTypeEXT(shadowRQ, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            float hitT = rayQueryGetIntersectionTEXT(shadowRQ, true);
            // Contact falloff: full occlusion at contact, fading to lit at maxDist.
            rtLocalShadow = clamp(1.0 - hitT / shadowDist, 0.0, 1.0);
        }
    }
#endif
    float totalShadow = shadow;
#ifndef BRUSH_PASS
    // Independent-occluder combine: CSM lit + RT lit = lit; either occluded =
    // occluded. RT never brightens CSM shadows and never double-darkens.
    totalShadow = 1.0 - (1.0 - shadow) * (1.0 - clamp(rtLocalShadow, 0.0, 1.0));
#endif

    // Blend material parameters (ambient/specular) by barycentric weights
    vec4 matFlags0 = materials[texIndices.x].materialFlags;
    vec4 matFlags1 = materials[texIndices.y].materialFlags;
    vec4 matFlags2 = materials[texIndices.z].materialFlags;
    vec4 blendedMatFlags = matFlags0 * w.x + matFlags1 * w.y + matFlags2 * w.z;
    // Ambient occlusion: sample texture value, blend with useAO flag and aoFactor
    // Blend roughnessAOParams across materials
    vec4 ra0 = materials[texIndices.x].roughnessAOParams;
    vec4 ra1 = materials[texIndices.y].roughnessAOParams;
    vec4 ra2 = materials[texIndices.z].roughnessAOParams;
    vec4 blendedRA = ra0 * w.x + ra1 * w.y + ra2 * w.z;
    
    float useAOf = blendedRA.z;
    float aoFactor = blendedRA.y;
    float roughnessFactor = blendedRA.x;
    
    float aoBlend = (useAOf > 0.5 && aoEnabled) ? ambientOcclusion : 1.0;
    aoBlend = mix(1.0, aoBlend, aoFactor);
    vec3 ambient = albedoColor * blendedMatFlags.z * aoBlend;
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);

    // Specular
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    vec4 spec0 = materials[texIndices.x].specularParams;
    vec4 spec1 = materials[texIndices.y].specularParams;
    vec4 spec2 = materials[texIndices.z].specularParams;
    vec4 blendedSpec = spec0 * w.x + spec1 * w.y + spec2 * w.z;
    // Gate specular on NdotL to prevent non-physical specular on dark-side surfaces
    // (normal-mapped worldNormal can make NdotL == 0 while still reflecting toward viewer).
    // Clamp shininess to >= 1.0 so pow(x, 0) == 1 never fires.
    float shininess = max(blendedSpec.y, 1.0);
    // Roughness modulates specular exponent: 0 = smooth (glossy), 1 = rough (diffuse)
    float specPower = mix(shininess, 1.0, roughnessValue * roughnessFactor);
    specPower = max(specPower, 1.0);
    float spec = (NdotL > 0.0) ? pow(max(dot(viewDir, reflectDir), 0.0), specPower) : 0.0;
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * blendedSpec.x;

    // Environment reflection via HARDWARE RAY TRACING (hybrid RT §7).
    // The rasterized world position + shading normal seed one secondary ray
    // through the stable proxy TLAS. On hit the proxy-box material shades the
    // reflection; on miss the procedural sky is evaluated directly. The old
    // 360° cubemap capture is removed. Single bounce, roughness-gated for
    // performance (§16); reflection rays are uncapped (RT_NO_LIMIT) so
    // distant scenery still mirrors. Very rough materials keep the
    // cheap sky approximation instead of tracing.
    vec3 envReflection = vec3(0.0);
    float blendedRefStrength = 0.0;
    float envFresnelFactor = 0.0;
    if (ubo.materialFlags.x < 0.5) {
        float refStrength0 = materials[texIndices.x].tessLevelParams.z;
        float refStrength1 = materials[texIndices.y].tessLevelParams.z;
        float refStrength2 = materials[texIndices.z].tessLevelParams.z;
        blendedRefStrength = refStrength0 * w.x + refStrength1 * w.y + refStrength2 * w.z;
        // Skip non-reflective surfaces (the mix factor below collapses to zero).
        if (blendedRefStrength > 1e-4) {
            // Use the smooth surface normal for the reflection ray: the
            // normal-mapped worldNormal carries per-pixel detail that scatters
            // RT rays (parts of the surface reflect the sky instead of the
            // scene), while the displaced derivative normal would warp the
            // reflection along every displacement bump. fragNormal is the
            // interpolated base-surface normal — clean and stable.
            vec3 reflN = normalize(fragNormal);
            vec3 reflV = normalize(viewDir);
            float cosTheta = clamp(dot(reflN, reflV), 0.0, 1.0);
            float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
            float rough = clamp(roughnessValue * roughnessFactor, 0.0, 1.0);
            vec3 reflDir = reflect(-reflV, reflN);
            // Procedural sky fallback (dielectric F0=0.04 gradient using the
            // scene sky colors — also the RT miss value, so toggling RT never
            // pops the miss baseline).
            vec3 skyApprox = vec3(0.35, 0.5, 0.65);
#ifdef RT_ENABLED
            skyApprox = rtProceduralSky(normalize(reflDir), sky.skyHorizon.rgb,
                                        sky.skyZenith.rgb, sky.skyParams.y);
#endif
            skyApprox *= aoBlend * (1.0 - rough * 0.5);
            vec3 rtColor = skyApprox;
            // Set when the ray-query resolved the reflection to a WATER proxy:
            // the SSR refinement samples the previous frame's SOLID render,
            // which contains the lake BOTTOM terrain (water is a transparent
            // pass), so it would overwrite the water surface tint with the
            // underwater ground — "water not visible in the reflection".
            bool waterHit = false;
            // Roughness gate + distance limit + global toggle + TLAS readiness
            // (ray queries compiled out without RT_ENABLED — sky fallback).
            float roughThreshold = 0.6;
#ifdef RT_ENABLED
            roughThreshold = clamp(rt.distances.w, 0.0, 1.0);
            bool doRTTrace = (rt.debug.y > 0.5) && rt.toggles.x > 0.5 && rough <= roughThreshold;
#else
            bool doRTTrace = false;
#endif
            if (doRTTrace) {
#ifdef RT_ENABLED
                // The BLAS holds the UNDISPLACED CPU mesh — use the undisplaced
                // position + base normal so the ray starts on the BLAS surface.
                // A small bias clears it; no displacement-dependent bias is
                // needed (the BLAS itself must not displace, per design).
                float selfSkip = max(rt.debug.z, 0.15);
                vec3 origin = fragPosWorldNotDisplaced + reflN * selfSkip;
                rayQueryEXT rq;
                // Mirror reflections trace the real scene-geometry instance
                // ONLY (exact chunk triangles, including the real water mesh
                // via the waterChunk flag below). The water proxy boxes are
                // deliberately excluded: they are coarse thickness slabs whose
                // side walls stick up above the lake surface while neighbouring
                // chunks leave vertical gaps — grazing mirror rays either slam
                // into a wall (a flat water block, faded to sky at the top,
                // where distant terrain should be) or thread a gap (sky leak),
                // printing chunk-sized shards and cracks across the reflection.
                rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsOpaqueEXT,
                    RT_RAY_MASK_SCENE,
                    origin, 0.05, normalize(reflDir), RT_NO_LIMIT);
                while (rayQueryProceedEXT(rq)) {}
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                    float hitT = rayQueryGetIntersectionTEXT(rq, true);
                    // Own-surface guard: hits closer than selfSkip are the
                    // reflector's own triangles, not true scenery.
                    if (hitT >= selfSkip) {
                        // Mask-selected: only the real scene-geometry instance can
                        // report hits (see the SCENE-only mask above).
                        const uint inst = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
                        if (inst == RT_SCENE_INSTANCE) {
                        // The primitive index is LOCAL to the hit geometry
                        // (per GLSL_EXT_ray_query: "the index of the primitive
                        // within the geometry of the BLAS"). The geometry index
                        // comes from the ray query directly — never binary
                        // search cumulative primBase (that maps local indices
                        // onto the wrong chunk for every geometry after the
                        // first, corrupting brushIndex/UV/normal reads).
                        const uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
                        const uint lo = uint(rayQueryGetIntersectionGeometryIndexEXT(rq, true));
                        // Real interpolated triangle normal from the merged
                        // vertex pool (Vertex stride 16 floats: position 0-2,
                        // normal 8-10) via the hit barycentrics.
                        vec3 hitPos = origin + normalize(reflDir) * hitT;
                        // (Screen-space color lookup removed: it pasted the
                        // previous frame's solid render — including the sky —
                        // over the accurate RT reflection. The ray-query
                        // shading below is the real reflection.)
                        {
                        uvec4 gi = rtSceneGeomInfo[lo];
                        // prim is already local to this geometry (ray-query
                        // semantics) — do NOT subtract the cumulative base.
                        uint localPrim = prim;
                        const uint kVertStride = 16u;
                        vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
                        // Real water mesh (gi.w): shade with a constant UP
                        // normal. Bindings 24/25 address the SOLID vertex
                        // pools while water vertices live in the WATER pools,
                        // so fetching here would light the water with another
                        // chunk's normals (faceted shards). Water is a
                        // heightfield, so UP matches the surface.
                        vec3 hitN = vec3(0.0, 1.0, 0.0);
                        uint i0 = 0u, i1 = 0u, i2 = 0u;
                        if (gi.w == 0u) {
                            i0 = rtSceneIndices[gi.y + localPrim * 3u + 0u] + gi.x;
                            i1 = rtSceneIndices[gi.y + localPrim * 3u + 1u] + gi.x;
                            i2 = rtSceneIndices[gi.y + localPrim * 3u + 2u] + gi.x;
                            vec3 n0 = vec3(rtSceneVerts[i0 * kVertStride + 8u],
                                           rtSceneVerts[i0 * kVertStride + 9u],
                                           rtSceneVerts[i0 * kVertStride + 10u]);
                            vec3 n1 = vec3(rtSceneVerts[i1 * kVertStride + 8u],
                                           rtSceneVerts[i1 * kVertStride + 9u],
                                           rtSceneVerts[i1 * kVertStride + 10u]);
                            vec3 n2 = vec3(rtSceneVerts[i2 * kVertStride + 8u],
                                           rtSceneVerts[i2 * kVertStride + 9u],
                                           rtSceneVerts[i2 * kVertStride + 10u]);
                            hitN = normalize(n0 * (1.0 - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);
                        }
                        if (dot(hitN, reflDir) > 0.0) hitN = -hitN; // face the incoming ray
                        // Water chunk marker (geomInfo.w): the real water mesh
                        // is in the scene BLAS; shade it as water (sky
                        // reflection + tint), never the terrain albedo lookup
                        // (a water chunk's brushIndex addresses water params).
                        if (gi.w > 0u) {
                            // Reflected water: transparent water look computed
                            // from the water's OWN params (water.frag formula):
                            // tint = mix(shallow, deep, volume) blended over
                            // the sky by waterTint*transparency. No recursive
                            // reflection/refraction.
                            // Stable water layer id: chunk-dominant, carried in
                            // rtSceneAlbedo[lo].w (per-vertex brushIndex can
                            // vary within a triangle → color flicker).
                            int wLayer = clamp(int(rtSceneAlbedo[lo].w + 0.5), 0, 31);
                            WaterParamsGPU wp = waterParams[wLayer];
                            vec3 shallowTint = wp.shallowColor.rgb;
                            vec3 deepTint = wp.deepColor.rgb;
                            float waterTintStr = wp.params2.x;
                            float transparency = wp.params1.z;
                            float depthFalloff = wp.waveParams.w;
                            float thickness = max(wp.refractionParams.y, 0.0);
                            float tintDepthScale = max(wp.causticParams.w, 0.0001);
                            float volumeFactor = 1.0 - exp(-thickness / tintDepthScale);
                            vec3 waterTintColor = mix(shallowTint, deepTint, volumeFactor);
                            // Beer-Lambert absorption (water.frag): the sky seen
                            // through the transparent water is attenuated by the
                            // water column.
                            vec3 transmittance = exp(-min(
                                wp.absorptionParams.rgb
                                    * max(thickness * wp.absorptionParams.a, 0.0),
                                vec3(2.5)));
                            float depthFade = 1.0 - exp(-thickness * depthFalloff);
                            float tintMax = clamp(1.0 - transparency, 0.0, 1.0);
                            float tintBlend = clamp(depthFade * waterTintStr, 0.0, tintMax);
                            vec3 skyR = rtProceduralSky(normalize(reflDir),
                                sky.skyHorizon.rgb, sky.skyZenith.rgb, sky.skyParams.y);
                            vec3 toLight = -normalize(ubo.lightDir.xyz);
                            float ndl = max(dot(hitN, toLight), 0.0);
                            vec3 waterColor = mix(skyR * transmittance, waterTintColor, tintBlend)
                                * (ubo.lightColor.rgb * (0.55 + 0.45 * ndl)
                                   + vec3(0.09, 0.12, 0.15));
                            rtColor = waterColor;
                            // A mirror's reflection is not occluded by AO nor dimmed by the
                            // surface roughness (the RT roughness gate already
                            // handles scatter): the reflection is full-strength.
                            rtColor *= 1.0;
                            waterHit = true;
                        } else {
                        // Real painted material at this triangle (Vertex float
                        // offset 11 = brushIndex): the proxy registry only
                        // knows the chunk's DOMINANT brush, which loses the
                        // ground-cover mix painted per vertex.
                        int maxLayer = max(int(textureSize(albedoArray, 0).z) - 1, 0);
                        // brushIndex is an int stored in the float-typed vertex pool: read its
                        // bit pattern (reading it as a float yields a denormal, decoding to 0).
                        const int matId = clamp(floatBitsToInt(rtSceneVerts[i0 * kVertStride + 11u]), 0, maxLayer);
                        // Exact raster texture lookup: three corner materials
                        // compressed into unique slots and blended by the hit
                        // barycentrics (main.tesc + main.frag) — fixes wrong
                        // material identity at in-triangle boundaries.
                        vec3 hitAlbedo = rtSceneSampleReflectionAlbedo(
                            i0, i1, i2, bary, hitPos, hitN, maxLayer);
                        // Macro shadows stay CSM-owned (no RT shadow recompute).
                        // The sky ambient fill matches water.frag's hit shading
                        // so reflections read as lit scenery, not dark plates.
                        // Full shading for the reflected hit: real texture
                        // albedo, real interpolated normal, sun diffuse (CSM
                        // shadowed) + sky ambient.
                        vec3 toLight = -normalize(ubo.lightDir.xyz);
                        float ndl = max(dot(hitN, toLight), 0.0);
                        float hitShadow = ShadowCalculation(
                            ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
                        rtColor = hitAlbedo * (ubo.lightColor.rgb * ndl * (1.0 - hitShadow)
                                               + vec3(0.09, 0.12, 0.15));
                        // A mirror's reflection is not occluded by AO nor dimmed by the
                            // surface roughness (the RT roughness gate already
                            // handles scatter): the reflection is full-strength.
                            rtColor *= 1.0;
                        } // terrain else
                        } // !ssHit
                    }
                }
            }
#endif
            }
            // SSR refinement: where the reflected scene is on screen, the
            // previous frame's real color/depth resolve the mirror per pixel;
            // the proxy/sky result above stays the fallback for the rest.
#ifdef RT_ENABLED
            // (Screen-space refinement removed: the inline ray query hits the
            // real geometry and shades it accurately. The previous-frame SSR
            // pasted the old solid render — including the sky — on top of the
            // reflection, progressively replacing it with stale content.)
#endif
            envReflection = rtColor;
            // The rasterized material roughness controls the reflectivity:
            // smooth (rough ≈ 0) = full mirror, rough (≈ 1) = matte lit color
            // only. The RT roughness gate above already skips the trace for
            // very rough surfaces; this factor fades the result smoothly.
            envFresnelFactor = 1.0 - clamp(rough, 0.0, 1.0);
        }
    }

    // Debug visualisation modes (0 = normal render)
    int debugMode = int(ubo.debugParams.x + 0.5);
    if (debugMode == 1) {
        vec3 gn = normalize(fragNormal);
        outColor = vec4(gn * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 2) {
        vec3 nm = normalize(worldNormal);
        outColor = vec4(nm * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 3) {
        outColor = vec4(uv.x, uv.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 4) {
        outColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 5) {
        vec3 ra0 = texture(albedoArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 ra1 = texture(albedoArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 ra2 = texture(albedoArray, vec3(uv, float(texIndices.z))).rgb;
        vec3 rawAlbedo = ra0 * w.x + ra1 * w.y + ra2 * w.z;
        outColor = vec4(rawAlbedo, 1.0);
        return;
    }
    if (debugMode == 6) {
        vec3 rn0 = texture(normalArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 rn1 = texture(normalArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 rn2 = texture(normalArray, vec3(uv, float(texIndices.z))).rgb;
        vec3 rawNormalTex = rn0 * w.x + rn1 * w.y + rn2 * w.z;
        outColor = vec4(rawNormalTex, 1.0);
        return;
    }
    if (debugMode == 7) {
        float h0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
        float h1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
        float h2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
        float h = h0 * w.x + h1 * w.y + h2 * w.z;
        outColor = vec4(vec3(h), 1.0);
        return;
    }
    if (debugMode == 8) {
        outColor = vec4(NdotL, totalShadow, 0.0, 1.0);
        return;
    }
    if (debugMode == 9) {
        vec3 normalToShow = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
        outColor = vec4(normalToShow * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 10) {
        vec3 tl = normalize(toLight);
        outColor = vec4(tl * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 11) {
        outColor = vec4(vec3(NdotL), 1.0);
        return;
    }
    if (debugMode == 12) {
        outColor = vec4(shadow, 0.0, totalShadow, 1.0);
        return;
    }
    if (debugMode == 13) {
        // Visualize triplanar blend weights RGB (X/Y/Z projections)
        outColor = vec4(triW, 1.0);
        return;
    }

    if (debugMode == 14) {
        // Map each corner brushIndex to a distinct color from a small palette, then blend by barycentric weights
        const int PALETTE_SIZE = 16;
        const vec3 palette[PALETTE_SIZE] = vec3[](
            vec3(0.90, 0.10, 0.10), // red
            vec3(0.10, 0.90, 0.10), // green
            vec3(0.10, 0.10, 0.90), // blue
            vec3(0.90, 0.90, 0.10), // yellow
            vec3(0.90, 0.10, 0.90), // magenta
            vec3(0.10, 0.90, 0.90), // cyan
            vec3(1.00, 0.55, 0.10), // orange
            vec3(0.55, 0.35, 0.15), // brown
            vec3(0.60, 0.20, 0.80), // purple
            vec3(1.00, 0.50, 0.70), // pink
            vec3(0.70, 1.00, 0.30), // lime
            vec3(0.00, 0.45, 0.55), // teal
            vec3(0.05, 0.10, 0.35), // navy
            vec3(0.45, 0.50, 0.10), // olive
            vec3(0.60, 0.60, 0.60), // gray
            vec3(1.00, 1.00, 1.00)  // white
        );

        vec3 c0 = palette[int(mod(float(texIndices.x), float(PALETTE_SIZE)) + 0.5)];
        vec3 c1 = palette[int(mod(float(texIndices.y), float(PALETTE_SIZE)) + 0.5)];
        vec3 c2 = palette[int(mod(float(texIndices.z), float(PALETTE_SIZE)) + 0.5)];
        vec3 blended = c0 * w.x + c1 * w.y + c2 * w.z;
        outColor = vec4(blended, 1.0);
        return;
    }

    if (debugMode == 15) {
        // Visualize barycentric weights directly as RGB
        outColor = vec4(clamp(w, 0.0, 1.0), 1.0);
        return;
    }

    if (debugMode == 16) {
        // Show the raw albedo samples for each corner packed into RGB (a0.r, a1.r, a2.r)
        vec3 a0 = texture(albedoArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(texIndices.z))).rgb;
        outColor = vec4(a0.r, a1.r, a2.r, 1.0);
        return;
    }

    if (debugMode == 17) {
        // Visualize triplanar-sampled albedo blended across the three material indices
        vec3 ta0 = computeTriplanarAlbedo(fragPosWorld, triW, texIndices.x, N);
        vec3 ta1 = computeTriplanarAlbedo(fragPosWorld, triW, texIndices.y, N);
        vec3 ta2 = computeTriplanarAlbedo(fragPosWorld, triW, texIndices.z, N);
        vec3 tAlbedo = ta0 * w.x + ta1 * w.y + ta2 * w.z;
        outColor = vec4(tAlbedo, 1.0);
        return;
    }

    if (debugMode == 18) {
        // Show per-projection triplanar heights for each corner packed into RGB
        vec2 tScale = vec2(materials[texIndices.x].triplanarParams.x, 
                            materials[texIndices.x].triplanarParams.y);
        float th0x = texture(heightArray, vec3(fragPosWorld.yz * tScale, float(texIndices.x))).r;
        float th0y = texture(heightArray, vec3(fragPosWorld.xz * tScale, float(texIndices.x))).r;
        float th0z = texture(heightArray, vec3(fragPosWorld.xy * tScale, float(texIndices.x))).r;
        // Pack the three projection samples as RGB for the first material (useful to see which projection contributes height)
        outColor = vec4(th0x, th0y, th0z, 1.0);
        return;
    }

    if (debugMode == 19) {
        // Show difference between UV-blended height and triplanar-blended height (abs difference)
        float h_uv0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
        float h_uv1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
        float h_uv2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
        float h_uv = h_uv0 * w.x + h_uv1 * w.y + h_uv2 * w.z;
        float h_tri0 = sampleHeightTriplanar(fragPosWorld, geomN, texIndices.x);
        float h_tri1 = sampleHeightTriplanar(fragPosWorld, geomN, texIndices.y);
        float h_tri2 = sampleHeightTriplanar(fragPosWorld, geomN, texIndices.z);
        float h_tri = h_tri0 * w.x + h_tri1 * w.y + h_tri2 * w.z;
        float d = abs(h_uv - h_tri);
        outColor = vec4(vec3(d * 5.0), 1.0); // amplify differences for visibility
        return;
    }

    if (debugMode == 20) {
        // Visualize triplanar-sampled normal blended across the three material indices
        vec3 tn0 = computeTriplanarNormal(fragPosWorld, triW, texIndices.x, geomN, N);
        vec3 tn1 = computeTriplanarNormal(fragPosWorld, triW, texIndices.y, geomN, N);
        vec3 tn2 = computeTriplanarNormal(fragPosWorld, triW, texIndices.z, geomN, N);
        vec3 blended = tn0 * w.x + tn1 * w.y + tn2 * w.z;
        vec3 tNormal = reorientNormal(blended, geomN);
        outColor = vec4(tNormal * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 21) {
        // Show per-projection triplanar normals for the first material packed into RGB
        vec3 nX = computeTriplanarNormal(fragPosWorld, vec3(1.0, 0.0, 0.0), texIndices.x, geomN, N);
        vec3 nY = computeTriplanarNormal(fragPosWorld, vec3(0.0, 1.0, 0.0), texIndices.x, geomN, N);
        vec3 nZ = computeTriplanarNormal(fragPosWorld, vec3(0.0, 0.0, 1.0), texIndices.x, geomN, N);
        // Pack single components of each projection to RGB so we can visually inspect contributions
        outColor = vec4(nX.x * 0.5 + 0.5, nY.y * 0.5 + 0.5, nZ.z * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 22) {
        // Visualize triplanar-sampled bump (height) blended across the three material indices
        float b0 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.x);
        float b1 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.y);
        float b2 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.z);
        float b = b0 * w.x + b1 * w.y + b2 * w.z;
        outColor = vec4(vec3(b), 1.0);
        return;
    }

    if (debugMode == 23) {
        // Show per-projection triplanar heights using sampleHeightTriplanar for the first material packed into RGB
        float ph0x = sampleHeightTriplanar(fragPosWorld, vec3(1.0, 0.0, 0.0), texIndices.x);
        float ph0y = sampleHeightTriplanar(fragPosWorld, vec3(0.0, 1.0, 0.0), texIndices.x);
        float ph0z = sampleHeightTriplanar(fragPosWorld, vec3(0.0, 0.0, 1.0), texIndices.x);
        outColor = vec4(ph0x, ph0y, ph0z, 1.0);
        return;
    }

    if (debugMode == 24) {
        // Show difference between UV-blended height and triplanar-blended height using worldNormal (abs difference)
        float h_uv0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
        float h_uv1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
        float h_uv2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
        float h_uv = h_uv0 * w.x + h_uv1 * w.y + h_uv2 * w.z;
        float h_tri0 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.x);
        float h_tri1 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.y);
        float h_tri2 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.z);
        float h_tri = h_tri0 * w.x + h_tri1 * w.y + h_tri2 * w.z;
        float d = abs(h_uv - h_tri);
        outColor = vec4(vec3(d * 5.0), 1.0); // amplify differences for visibility
        return;
    }

    if (debugMode == 25) {
        // Visualize triplanar UV for X projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, texIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvX);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 26) {
        // Visualize triplanar UV for Y projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, texIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvY);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 27) {
        // Visualize triplanar UV for Z projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, texIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvZ);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }

    if (debugMode == 28) {
        tripNormal0 = computeTriplanarNormal(fragPosWorldNotDisplaced, triW, texIndices.x, geomN, N);
        outColor = vec4(normalize(tripNormal0) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 29) {
        tripNormal1 = computeTriplanarNormal(fragPosWorldNotDisplaced, triW, texIndices.y, geomN, N);
        outColor = vec4(normalize(tripNormal1) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 30) {
        tripNormal2 = computeTriplanarNormal(fragPosWorldNotDisplaced, triW, texIndices.z, geomN, N);
        outColor = vec4(normalize(tripNormal2) * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 31) {
        // Visualize TES-provided face normal (sharp per-triangle normal computed in tessellation evaluation shader)
        vec3 s = normalize(fragSharpNormal);
        outColor = vec4(s * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 32) {
        outColor = vec4(vec3(roughnessValue), 1.0);
        return;
    }
    if (debugMode == 33) {
        outColor = vec4(vec3(ambientOcclusion), 1.0);
        return;
    }
    if (debugMode == 49) {
        // Environment reflection contribution — the cubemap sample weighted
        // by the Fresnel factor actually mixed into the final colour.
        outColor = vec4(envReflection * envFresnelFactor, 1.0);
        return;
    }
    // ── Hybrid RT debug views (also selectable via settings.rtDebugView) ──
    // 55 = CSM-only shadows, 56 = RT-local-only, 57 = CSM+RT combined shadow.
    if (debugMode == 55) {
        outColor = vec4(vec3(shadow), 1.0);
        return;
    }
    if (debugMode == 56) {
        outColor = vec4(vec3(clamp(rtLocalShadow, 0.0, 1.0)), 1.0);
        return;
    }
    if (debugMode == 57) {
        outColor = vec4(vec3(totalShadow), 1.0);
        return;
    }
    // ── Tessellation-level heatmap (tess levels / 16: black = 1 inactive,
    // blue→green→red = increasing subdivision). Proves per-fragment which
    // materials/dstances actually subdivide.
    if (debugMode == 58) {
        float t = clamp(fragTessLevel.x, 0.0, 4.0);
        vec3 heat = mix(vec3(0.0), vec3(0.0, 0.0, 1.0), clamp(t * 4.0, 0.0, 1.0));
        heat = mix(heat, vec3(0.0, 1.0, 0.0), clamp((t - 0.25) * 4.0, 0.0, 1.0));
        heat = mix(heat, vec3(1.0, 0.0, 0.0), clamp((t - 0.5) * 2.0, 0.0, 1.0));
        outColor = vec4(heat, 1.0);
        return;
    }

    // Energy-conserving blend between lit color and environment reflection.
    // reflectionStrength=0 → lit color only; =1 → physical Fresnel mirror
    // (~4% at normal incidence, full mirror at grazing angles).
    vec3 litColor = ambient + diffuse + specular;
    vec3 finalColor = mix(litColor, envReflection, envFresnelFactor);

    // REMOVE mode: smoothly fade between red tint and brush texture at 1Hz
    if (brushRedFade > 0.0) {
        vec3 redTint = vec3(finalColor.r, 0.0, 0.0);
        finalColor = mix(redTint, finalColor, brushRedFade);
    }
    
    // DEBUG: Visualize lighting components
    // Uncomment to debug:
    // if (length(albedoColor) < 0.01) { outColor = vec4(1.0, 0.0, 0.0, 1.0); return; } // Red if no albedo
    // if (length(finalColor) < 0.01) { outColor = vec4(0.0, 1.0, 0.0, 1.0); return; } // Green if no lighting
    // outColor = vec4(albedoColor, 1.0); return; // Show raw albedo
    // outColor = vec4(vec3(NdotL), 1.0); return; // Show N·L term
    
    // Apply per-vertex HSV: rotate hue, offset saturation, scale value
    // In paint mode, hsvColor is overridden with the brush's HSV from the UBO
    vec3 texHSV = rgbToHsv(finalColor);
    texHSV.x = mod(texHSV.x + hsvColor.x, 360.0);
    texHSV.y = clamp(texHSV.y * (hsvColor.y * 2.0), 0.0, 1.0);
    texHSV.z *= hsvColor.z * 2.0;
    finalColor = hsvToRgb(texHSV);
    
    // Final output (single color target)
    outColor = vec4(finalColor, 1.0);
}