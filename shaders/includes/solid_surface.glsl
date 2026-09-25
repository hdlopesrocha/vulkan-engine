#include "sky_view.glsl"
// Solid (terrain) surface shading — extracted from main.frag so the
// fragment entry point stays a thin dispatcher (see Phase-1 water-in-main
// migration). Writes the global outColor; early returns (debug views,
// shadow fast path callers) behave exactly as before.
// Requires: ubo, textures, varyings, shadows, triplanar, hsv (as declared
// by main.frag before this include).
//
// Reflection SSR helper (previous-frame solid color/depth). Kept with the
// solid logic: it exists solely for solid-surface reflections.
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
    float nearP = ubo.nearPlane;
    float farP  = ubo.farPlane;
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

void shadeSolidSurface() {
    // Texture indices, optionally overridden by PAINT/REMOVE mode
    ivec3 texIndices = fragTexIndices;
    vec3 hsvColor = fragHSV;
    float brushRedFade = 0.0;
#ifndef BRUSH_PASS
    bool isPaintMode = ubo.brushMode > 1.5;
    bool isRemoveMode = ubo.brushMode > 0.5 && ubo.brushMode < 1.5;
    if (isPaintMode || isRemoveMode) {
        vec2 brushUV = gl_FragCoord.xy / vec2(textureSize(brushDepthTex, 0));
        float brushFront = texture(brushDepthTex, brushUV).r;
        float brushBack = texture(brushBackFaceDepthTex, brushUV).r;
        float fragDepth = gl_FragCoord.z;
        if (fragDepth >= brushFront && fragDepth <= brushBack) {
            int brushTexIndex = int(ubo.brushTextureIndex + 0.5);
            texIndices = ivec3(brushTexIndex);
            // Override vertex HSV with the brush's HSV so painted areas get the brush tint
            hsvColor = ubo.brushHsv;
            if (isRemoveMode) {
                brushRedFade = (sin(ubo.brushPhase * 6.28318) + 1.0) * 0.5;
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
    float t = ubo.triplanarThreshold; // threshold (0..1)
    vec3 wt = max(vec3(0.0), triW - vec3(t));

    // Apply exponent to make transitions steeper
    float e = max(1.0, ubo.triplanarExponent);
    wt = pow(wt, vec3(e));
    float triWSum = wt.x + wt.y + wt.z + 1e-6;
    triW = wt / triWSum;
    
    // Sample albedo texture (triplanar when enabled)
    vec3 albedoColor;
    vec3 tripNormal0;
    vec3 tripNormal1;
    vec3 tripNormal2;

    // Mix triplanar flag across the three materials
    float triFlag = dot(vec3(float(materialNamed(materials[texIndices.x]).triplanarEnabled), float(materialNamed(materials[texIndices.y]).triplanarEnabled), float(materialNamed(materials[texIndices.z]).triplanarEnabled)), w);

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
        float mapFlag0 = float(materialNamed(materials[texIndices.x]).mappingEnabled);
        float mapFlag1 = float(materialNamed(materials[texIndices.y]).mappingEnabled);
        float mapFlag2 = float(materialNamed(materials[texIndices.z]).mappingEnabled);
        if ((mapFlag0 * w.x + mapFlag1 * w.y + mapFlag2 * w.z) > 0.5 || ubo.normalMappingEnabled) {
            // C3: blend texels first, transform once (same weights/texels).
            tripNormal0 = w.x > 0.0 ? computeTriplanarNormalBlended(triW, texIndices.x, N, uv0X, uv0Y, uv0Z) : vec3(0.0);
            tripNormal1 = w.y > 0.0 ? computeTriplanarNormalBlended(triW, texIndices.y, N, uv1X, uv1Y, uv1Z) : vec3(0.0);
            tripNormal2 = w.z > 0.0 ? computeTriplanarNormalBlended(triW, texIndices.z, N, uv2X, uv2Y, uv2Z) : vec3(0.0);
            vec3 blended = tripNormal0 * w.x + tripNormal1 * w.y + tripNormal2 * w.z;
            worldNormal = normalize(blended);
            
            
            // Diagnostic: detect invalid/degenerate normals and show red so we can find broken pixels
            if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
                outColor = vec4(1.0, 0.0, 0.0, 1.0);
                return;
            }
        }
    } else {
        // Sample albedo from each layer and blend by barycentric weights.
        // Zero-weight slots are skipped (C3): weight 0 contributes 0, so the
        // fetch is provably dead — single-material interiors pay 1, not 3.
        vec3 a0 = w.x > 0.0 ? texture(albedoArray, vec3(uv, float(texIndices.x))).rgb : vec3(0.0);
        vec3 a1 = w.y > 0.0 ? texture(albedoArray, vec3(uv, float(texIndices.y))).rgb : vec3(0.0);
        vec3 a2 = w.z > 0.0 ? texture(albedoArray, vec3(uv, float(texIndices.z))).rgb : vec3(0.0);
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
    }

    // Compute normal mapping if enabled (per-material or global toggle)
    if (!usedTriplanar && ((float(materialNamed(materials[texIndices.x]).mappingEnabled) * w.x + float(materialNamed(materials[texIndices.y]).mappingEnabled) * w.y + float(materialNamed(materials[texIndices.z]).mappingEnabled) * w.z) > 0.5 || ubo.normalMappingEnabled)) {
        // Sample normal map per-layer and blend in tangent space (C3: dead
        // slots skipped, same argument as albedo above).
        vec3 n0 = w.x > 0.0 ? texture(normalArray, vec3(uv, float(texIndices.x))).rgb * 2.0 - 1.0 : vec3(0.0);
        vec3 n1 = w.y > 0.0 ? texture(normalArray, vec3(uv, float(texIndices.y))).rgb * 2.0 - 1.0 : vec3(0.0);
        vec3 n2 = w.z > 0.0 ? texture(normalArray, vec3(uv, float(texIndices.z))).rgb * 2.0 - 1.0 : vec3(0.0);
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

    // Sample roughness map (R channel). Triplanar path (C3): one fetch on
    // the dominant projection per material instead of three.
    float roughnessValue;
    if (usedTriplanar) {
        float r0 = w.x > 0.0 ? computeTriplanarRoughnessDominant(triW, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
        float r1 = w.y > 0.0 ? computeTriplanarRoughnessDominant(triW, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
        float r2 = w.z > 0.0 ? computeTriplanarRoughnessDominant(triW, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
        roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    } else {
        float r0 = w.x > 0.0 ? texture(roughnessArray, vec3(uv, float(texIndices.x))).r : 0.0;
        float r1 = w.y > 0.0 ? texture(roughnessArray, vec3(uv, float(texIndices.y))).r : 0.0;
        float r2 = w.z > 0.0 ? texture(roughnessArray, vec3(uv, float(texIndices.z))).r : 0.0;
        roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    }
    if (!roughnessEnabled) roughnessValue = 0.0;

    // Sample ambient occlusion map (R channel). Triplanar path (C3): one
    // fetch on the dominant projection per material instead of three.
    float ambientOcclusion;
    if (usedTriplanar) {
        float ao0 = w.x > 0.0 ? computeTriplanarAODominant(triW, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
        float ao1 = w.y > 0.0 ? computeTriplanarAODominant(triW, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
        float ao2 = w.z > 0.0 ? computeTriplanarAODominant(triW, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
        ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);
    } else {
        float ao0 = w.x > 0.0 ? texture(aoArray, vec3(uv, float(texIndices.x))).r : 0.0;
        float ao1 = w.y > 0.0 ? texture(aoArray, vec3(uv, float(texIndices.y))).r : 0.0;
        float ao2 = w.z > 0.0 ? texture(aoArray, vec3(uv, float(texIndices.z))).r : 0.0;
        ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);
    }

    // Lighting calculation
    vec3 toLight = -normalize(ubo.lightDirection);
    float NdotL = max(dot(worldNormal, toLight), 0.0);

    // Shadow calculation (CSM authoritative macro sun-shadow — NEVER replaced
    // by RT; see §2/§21). RT adds an optional selective local/contact term
    // below, combined without double-darkening.
    float shadow = 0.0;
#ifndef BRUSH_PASS
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    if (ubo.shadowsEnabled) {
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
    bool rtReady = rt.tlasReady;
    if (rtReady && rt.localShadowsEnabled && shadow < 0.5 && NdotL > 0.01) {
        vec3 sunDir = -normalize(ubo.lightDirection);
        float shadowDist = max(rt.maxShadowDistance, 0.5);
        RT_PROF_BEGIN(rtProfShadow, RT_PROFILE_OP_CONTACT_SHADOW);
        rayQueryEXT shadowRQ;
        rayQueryInitializeEXT(shadowRQ, rtTlas, gl_RayFlagsOpaqueEXT, RT_RAY_MASK_ALL,
            fragPosWorld + worldNormal * 0.05, 0.05, sunDir, shadowDist);
        while (rayQueryProceedEXT(shadowRQ)) {}
        RT_PROF_END(rtProfShadow);
        if (rayQueryGetIntersectionTypeEXT(shadowRQ, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            RT_PROF_HIT(RT_PROFILE_OP_CONTACT_SHADOW);
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
    vec3 diffuse = albedoColor * ubo.lightColor * NdotL * (1.0 - totalShadow);

    // Specular
    vec3 viewDir = normalize(ubo.viewPosition - fragPosWorld);
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
    vec3 specular = ubo.lightColor * spec * (1.0 - totalShadow) * blendedSpec.x;

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
    // Ray-budget mask for DEBUG_MODE_RAY_MASK (ray-query pixel ratio): 0 = no
    // reflective surface, 1 = skipped by roughness/contrib gate, 2 = skipped
    // by checkerboard half-rate, 3 = inline ray traced.
    float rtTraceMask = 0.0;
    if (!ubo.cubemapCapture) {
        float refStrength0 = materialNamed(materials[texIndices.x]).reflectionStrength;
        float refStrength1 = materialNamed(materials[texIndices.y]).reflectionStrength;
        float refStrength2 = materialNamed(materials[texIndices.z]).reflectionStrength;
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
            // Sky-reflection fallback: used whenever the RT mirror ray is not
            // traced (RT reflections off / TLAS not ready / roughness gate).
            // Procedural horizon/zenith gradient from the sky UBO — also the
            // RT miss value, so toggling RT never pops the miss baseline, and
            // solid reflections still come "from the sky" in raster-only mode.
            vec3 skyApprox = rtProceduralSky(normalize(reflDir), sky.horizonColor,
                                             sky.zenithColor, sky.exponent);
            skyApprox *= aoBlend * (1.0 - rough * 0.5);
            vec3 rtColor = skyApprox;
            // Set when the ray-query resolved the reflection to a WATER proxy:
            // the SSR refinement samples the previous frame's SOLID render,
            // which contains the lake BOTTOM terrain (water is a transparent
            // pass), so it would overwrite the water surface tint with the
            // underwater ground — "water not visible in the reflection".
            bool waterHit = false;
            // Ray-budget gate (perf: 2-4x fewer inline rays, no visible change):
            // roughness gate (existing) + contribution gate + checkerboard
            // half-rate + global toggle + TLAS readiness (ray queries compiled
            // out without RT_ENABLED — sky fallback). contrib = the exact lobe
            // weight mixed into the final color below, so skipping contrib <
            // contribMin (rt.reflectionContribMin, default 0.02) only drops rays whose
            // result would be invisible. Checkerboard (rt.checkerboardReflections)
            // traces even (x+y) pixels only; strong mirrors (contrib > 0.5)
            // stay full-rate so polished surfaces never dither, and any debug
            // view of a traced result (see debugModeForcesRtReference) forces
            // full-rate reference. The null-TLAS skip path (rt.tlasReady) is
            // unchanged.
            float roughThreshold = 0.6;
#ifdef RT_ENABLED
            roughThreshold = clamp(rt.roughnessThreshold, 0.0, 1.0);
            float contrib = clamp(blendedRefStrength, 0.0, 1.0) * fresnel
                * (1.0 - clamp(rough, 0.0, 1.0));
            float contribMin = clamp(rt.reflectionContribMin, 0.0, 1.0);
            // Ray mask / depth source visualize the budgeted behavior itself,
            // so they must not force reference (otherwise the counters could
            // never show the live behavior).
            bool refMode = debugModeForcesRtReference(rt.debugMode);
            bool checkerOn = rt.checkerboardReflections && !refMode;
            // Checkerboard only claims pixels that survived every other gate
            // (else gated pixels would misreport as half-rate in the mask).
            bool gatedOut = !((rt.tlasReady) && rt.reflectionsEnabled
                && rough <= roughThreshold && contrib >= contribMin);
            bool checkerSkip = checkerOn && !gatedOut
                && ((int(gl_FragCoord.x) + int(gl_FragCoord.y)) & 1) == 1
                && contrib <= 0.5;
            bool doRTTrace = !gatedOut && !checkerSkip;
            // Mask for DEBUG_MODE_RAY_MASK.
            rtTraceMask = checkerSkip ? 2.0 : ((doRTTrace ? 3.0 : 1.0));
#else
            bool doRTTrace = false;
            rtTraceMask = 1.0;
#endif
            if (doRTTrace) {
#ifdef RT_ENABLED
                // The BLAS holds the UNDISPLACED CPU mesh — use the undisplaced
                // position + base normal so the ray starts on the BLAS surface.
                // A small bias clears it; no displacement-dependent bias is
                // needed (the BLAS itself must not displace, per design).
                float selfSkip = max(rt.selfSkipDist, 0.15);
                vec3 origin = fragPosWorldNotDisplaced + reflN * selfSkip;
                RT_PROF_BEGIN(rtProfRefl, RT_PROFILE_OP_SOLID_REFLECTION);
                rayQueryEXT rq;
                // Mirror reflections trace the real scene-geometry instances
                // ONLY (exact chunk triangles: solids + the real water mesh,
                // the latter flagged by geomInfo.w). The water proxy boxes are
                // deliberately excluded: they are coarse thickness slabs whose
                // side walls stick up above the lake surface while neighbouring
                // chunks leave vertical gaps — grazing mirror rays either slam
                // into a wall (a flat water block, faded to sky at the top,
                // where distant terrain should be) or thread a gap (sky leak),
                // printing chunk-sized shards and cracks across the reflection.
                // Cull ray-front faces: scene triangles wind CW-outward (the
                // rasterizer draws them with BACK+CW culling), and ray-front
                // is fixed CCW — so this keeps exactly the rasterizer-visible
                // faces and skips inward faces the main pass would cull.
                rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsOpaqueEXT |
                    gl_RayFlagsCullFrontFacingTrianglesEXT,
                    RT_RAY_MASK_SCENE | RT_RAY_MASK_SCENE_WATER,
                    origin, 0.05, normalize(reflDir), RT_NO_LIMIT);
                while (rayQueryProceedEXT(rq)) {}
                RT_PROF_END(rtProfRefl);
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                    RT_PROF_HIT(RT_PROFILE_OP_SOLID_REFLECTION);
                    float hitT = rayQueryGetIntersectionTEXT(rq, true);
                    // Own-surface guard: hits closer than selfSkip are the
                    // reflector's own triangles, not true scenery.
                    if (hitT >= selfSkip) {
                        // Mask-selected: only the real scene-geometry instances
                        // can report hits (see the SCENE masks above).
                        const uint inst = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
                        if (rtIsSceneInstance(inst)) {
                        // The primitive index is LOCAL to the hit geometry
                        // (per GLSL_EXT_ray_query: "the index of the primitive
                        // within the geometry of the BLAS"). The geometry index
                        // maps through rtSceneGeomIndex() onto the combined
                        // solid/water lookup buffers — never binary search
                        // cumulative primBase (that maps local indices onto
                        // the wrong chunk for every geometry after the first,
                        // corrupting brushIndex/UV/normal reads).
                        const uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
                        const uint lo = rtSceneGeomIndex(inst,
                            uint(rayQueryGetIntersectionGeometryIndexEXT(rq, true)));
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
                        // Mirror strength of this primary hit (drives the
                        // in-reflection bounce below); set by each branch.
                        float hitReflectivity = 0.0;
                        // Water chunk marker (geomInfo.w): the real water mesh
                        // is in the scene BLAS; shade it as water (sky
                        // reflection + tint), never the terrain albedo lookup
                        // (a water chunk's brushIndex addresses water params).
                        if (gi.w > 0u) {
                            // Reflected water: SAME composition as the raster
                            // water surface (tint base + Fresnel/strength sky
                            // reflection) so water inside a mirror matches the
                            // water itself. The underwater bottom is not traced
                            // from inside a mirror, so the refraction term
                            // collapses to the tint (refraction-off water).
                            int nWLM = max(waterParams.length(), 1);
                            int wIdM = int(rtSceneAlbedo[lo].w + 0.5);
                            int wLayer = (wIdM >= 0 && wIdM < nWLM) ? wIdM : 0;
                            WaterParamsNamed wp = waterParamsNamed(waterParams[wLayer]);
                            rtColor = rtResolveWaterHit(wp, hitPos, hitN, reflDir);
                            waterHit = true;
                            // No bounce off water hits (see rtHitReflectivity):
                            // the water look already includes its mirror and
                            // recursive water rays self-intersect the flat
                            // BLAS mesh (water-on-water triangle noise).
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
                            i0, i1, i2, bary, hitPos, hitN, maxLayer, 0.0);
                        // Macro shadows stay CSM-owned (no RT shadow recompute).
                        // The sky ambient fill matches water.frag's hit shading
                        // so reflections read as lit scenery, not dark plates.
                        // Full shading for the reflected hit: real texture
                        // albedo, real interpolated normal, sun diffuse (CSM
                        // shadowed) + sky ambient.
                        vec3 toLight = -normalize(ubo.lightDirection);
                        float ndl = max(dot(hitN, toLight), 0.0);
                        float hitShadow = ShadowCalculation(
                            ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
                        // Albedo-scaled sky ambient (raster convention), not
                        // a dark constant: grazing/off-screen terrain hits in
                        // mirrors no longer read as near-black plates.
                        rtColor = hitAlbedo * (ubo.lightColor * ndl * (1.0 - hitShadow)
                                               + vec3(0.26));
                        // A mirror's reflection is not occluded by AO nor dimmed by
                            // the surface roughness (the RT roughness gate already
                            // handles scatter): the reflection is full-strength.
                            rtColor *= 1.0;
                        // Chunk mirror strength (packed in rtSceneAlbedo[].w by
                        // the proxy/geometry builder) gates the bounce below.
                        hitReflectivity = clamp(rtSceneAlbedo[lo].w, 0.0, 1.0);
                        } // terrain else
                        // Reflection-inside-reflection: when the primary hit is
                        // itself reflective, chain extra mirror rays up to the
                        // configured bounce count (rt.maxReflectionBounces, 0 = single).
                        {
                            int extraBounces = clamp(rt.maxReflectionBounces, 0, 3) - 1;
                            if (hitReflectivity > 0.02 && extraBounces >= 0) {
                                vec3 nextDir = normalize(reflect(normalize(reflDir), hitN));
                                vec3 bounceCol = rtTraceMirror(hitPos + hitN * 0.05,
                                                               nextDir, extraBounces,
                                                               selfSkip);
                                rtColor = mix(rtColor, bounceCol, hitReflectivity);
                            }
                        }
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
            // Reflectivity: the material's reflectionStrength lerps between a
            // physical Fresnel mirror (0) and a full mirror at every angle
            // (1) — same convention as the water reflectionStrength — shaped
            // by the Schlick Fresnel term, then faded by the rasterized
            // material roughness (smooth ≈ full mirror, rough ≈ matte lit
            // color only). The RT roughness gate above already skips the trace
            // for very rough surfaces; this factor fades the result smoothly.
            envFresnelFactor = mix(fresnel, 1.0, clamp(blendedRefStrength, 0.0, 1.0))
                * (1.0 - clamp(rough, 0.0, 1.0));
        }
    }

    // ── Unified debug views (IDs shared with the water path) ──
    // Canonical IDs live in includes/debug_modes.glsl (mirror of
    // vulkan/includes/DebugModes.hpp). Water-only views (sky/refraction/
    // noise/displacement/thickness/absorption/caustics/depth/compose) fall
    // through to normal shading here; 0 = normal render.
    int debugMode = ubo.debugMode;
    if (debugMode == DEBUG_MODE_SCENE_DEPTH) {
        // Linear eye-space depth / far (white = far/clear). Shows whether the
        // opaque pass actually writes the terrain (e.g. the lake bed) behind
        // the water.
        float nearP = max(ubo.nearPlane, 1e-4);
        float farP = max(ubo.farPlane, nearP + 1.0);
        float zEye = (nearP * farP) / (farP - gl_FragCoord.z * (farP - nearP));
        outColor = vec4(vec3(clamp(zEye / farP, 0.0, 1.0)), 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_SHADING_NORMAL) {
        // Final material-perturbed normal actually used for lighting.
        vec3 nm = normalize(worldNormal);
        outColor = vec4(nm * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_GEOMETRIC_NORMAL) {
        // Interpolated base-surface normal before material perturbation.
        outColor = vec4(normalize(N) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_FACE_NORMAL) {
        // Rasterizer-visible facet normal from screen-space derivatives.
        vec3 fn = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
        outColor = vec4(fn * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_ALBEDO) {
        outColor = vec4(albedoColor, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_NORMAL_MAP) {
        // Raw normal-map samples (UV addressing) blended by barycentric weights.
        vec3 rn0 = texture(normalArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 rn1 = texture(normalArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 rn2 = texture(normalArray, vec3(uv, float(texIndices.z))).rgb;
        outColor = vec4(rn0 * w.x + rn1 * w.y + rn2 * w.z, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_HEIGHT_MAP) {
        // Final height/bump sample driving displacement: triplanar when the
        // material uses triplanar mapping, UV addressing otherwise.
        float h;
        if (usedTriplanar) {
            float b0 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.x);
            float b1 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.y);
            float b2 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.z);
            h = b0 * w.x + b1 * w.y + b2 * w.z;
        } else {
            float h0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
            float h1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
            float h2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
            h = h0 * w.x + h1 * w.y + h2 * w.z;
        }
        outColor = vec4(vec3(h), 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_ROUGHNESS) {
        // Roughness that actually drives the specular/mirror mix.
        outColor = vec4(vec3(clamp(roughnessValue * roughnessFactor, 0.0, 1.0)), 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_AMBIENT_OCCLUSION) {
        outColor = vec4(vec3(ambientOcclusion), 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_TRIPLANAR_WEIGHTS) {
        // Triplanar blend weights RGB (X/Y/Z projections).
        outColor = vec4(triW, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_MATERIAL_INDEX) {
        // Palette-blended brush/texture index: each corner's index selects a
        // color, barycentric weights blend them.
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
        outColor = vec4(c0 * w.x + c1 * w.y + c2 * w.z, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_UV) {
        outColor = vec4(uv.x, uv.y, 0.0, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_N_DOT_L) {
        outColor = vec4(vec3(NdotL), 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_LIGHT_VECTOR) {
        vec3 tl = normalize(toLight);
        outColor = vec4(tl * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_SHADOW) {
        // R = CSM, G = RT local/contact, B = combined term actually shaded.
        outColor = vec4(shadow, clamp(rtLocalShadow, 0.0, 1.0), totalShadow, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_REFLECTION_COLOR) {
        // Environment reflection contribution mixed into the final color.
        outColor = vec4(envReflection * envFresnelFactor, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_REFLECTION_VECTOR) {
        vec3 rv = normalize(reflect(-viewDir, worldNormal));
        outColor = vec4(rv * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_FRESNEL) {
        outColor = vec4(vec3(envFresnelFactor), 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_RAY_MASK) {
        // Ray-query budget mask: white = inline traced, mid-grey =
        // checkerboard-skipped (pipeline/sky fallback), dark = gated
        // (roughness/contrib), black = non-reflective. The traced-pixel
        // fraction must drop >=50% vs full-rate with default settings.
        vec3 maskCol = vec3(0.0);
        if (rtTraceMask > 2.5) maskCol = vec3(1.0);
        else if (rtTraceMask > 1.5) maskCol = vec3(0.45);
        else if (rtTraceMask > 0.5) maskCol = vec3(0.12);
        outColor = vec4(maskCol, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_TESS_HEAT) {
        // Tessellation-level heatmap (tess levels / 16: black = 1 inactive,
        // blue→green→red = increasing subdivision). Proves per-fragment which
        // materials/distances actually subdivide.
        float t = clamp(fragTessLevel.x, 0.0, 4.0);
        vec3 heat = mix(vec3(0.0), vec3(0.0, 0.0, 1.0), clamp(t * 4.0, 0.0, 1.0));
        heat = mix(heat, vec3(0.0, 1.0, 0.0), clamp((t - 0.25) * 4.0, 0.0, 1.0));
        heat = mix(heat, vec3(1.0, 0.0, 0.0), clamp((t - 0.5) * 2.0, 0.0, 1.0));
        outColor = vec4(heat, 1.0);
        return;
    }

    // (legacy per-corner/per-projection material debug views removed — the
    // canonical MaterialIndex / NormalMap / HeightMap / TriplanarWeights
    // views above cover them)

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
    // In paint mode, hsvColor is overridden with the brush's HSV from the UBO.
    // H5: skip the round-trip for the identity tint (0, 0.5, 0.5), which the
    // conversion pair leaves unchanged. Baked non-identity tints
    // (Simple/NormalBrush paintHSV) and brush-tint pixels still convert; the
    // compare is uniform wherever the tint is uniform, and inexact
    // interpolation safely falls into the conversion path.
    if (hsvColor.x != 0.0 || hsvColor.y != 0.5 || hsvColor.z != 0.5) {
        vec3 texHSV = rgbToHsv(finalColor);
        texHSV.x = mod(texHSV.x + hsvColor.x, 360.0);
        texHSV.y = clamp(texHSV.y * (hsvColor.y * 2.0), 0.0, 1.0);
        texHSV.z *= hsvColor.z * 2.0;
        finalColor = hsvToRgb(texHSV);
    }
    
    // Final output (single color target)
    outColor = vec4(finalColor, 1.0);
}
