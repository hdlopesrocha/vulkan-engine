
#include "../../types/WaterParamsNamed.glsl"
#include "../sky/SkyView.glsl"
#include "../noise/Perlin.glsl"
#include "../sky/Clouds.glsl"
#include "../TextureMapping.glsl"
// Solid (terrain) surface shading — extracted from SolidRenderer.frag so the
// fragment entry point stays a thin dispatcher (see Phase-1 water-in-main
// migration). Writes the global outColor; early returns (debug views,
// shadow fast path callers) behave exactly as before.
// Requires: ubo, textures, varyings, shadows, triplanar, hsv (as declared
// by SolidRenderer.frag before this include).
//
// Reflection SSR helper (previous-frame solid color/depth). Kept with the
// solid logic: it exists solely for solid-surface reflections.
// March `dir` from `origin` against the previous frame's solid depth, using the
// previous frame's view-projection so camera motion does not stipple the hit
// test. Returns rgb = reflected scene color, a = confidence (0 = no usable
// hit). Falls back to the proxy/sky result for off-screen occluded rays.
// `minHitY` rejects hits below a world height (the water raymarch uses the
// water level, so the lake surface / submerged bed is never painted as its own
// reflection).
// Marching front to back, the first depth crossing is the first real
// intersection, so any crossing counts as a hit and a binary search refines
// it (step-size independent); the self-UV guard rejects the reflector's own
// surface. `eyeDir` is the unit vector from the surface to the camera: rays
// nearly tangent to the view direction (silhouettes) are where screen-space
// marching is least reliable, so they fade out.
vec4 traceSSR(vec3 origin, vec3 dir, vec3 eyeDir, mat4 prevVP, vec2 selfUV,
              float minHitY) {
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
                // Reject hits below the caller's world-height floor.
                if (minHitY > -1e29) {
                    float dHit = textureLod(ssrDepthTex, uv, 0.0).r;
                    vec4 wh = ubo.invViewProjection * vec4(uv * 2.0 - 1.0, dHit, 1.0);
                    if (wh.w > 1e-6 && (wh.y / wh.w) < minHitY) { prevT = t; continue; }
                }
                float edge = smoothstep(0.0, 0.06, uv.x) * smoothstep(0.0, 0.06, 1.0 - uv.x)
                           * smoothstep(0.0, 0.06, uv.y) * smoothstep(0.0, 0.06, 1.0 - uv.y);
                // Crossings further behind the surface are less certain
                // (ray nearly parallel to it): soften instead of hard-cutting.
                float gapFade = 1.0 - clamp((rayEye - sceneEye) / max(1.0, sceneEye * 0.25), 0.0, 0.5);
                // Grazing reflections are physically the strongest (water), so
                // the silhouette fade is kept narrow.
                return vec4(textureLod(ssrColorTex, uv, 0.0).rgb,
                            edge * gapFade * smoothstep(0.0, 0.08, facing));
            }
        }
        prevT = t;
    }
    return vec4(0.0);
}

void shadeSolidSurface(vec3 geomNormalOverride) {
    // Texture indices, optionally overridden by PAINT/REMOVE mode
    ivec3 texIndices = fragTexIndices;
    vec3 hsvColor = fragHSV;
    float brushRedFade = 0.0;
    bool isPaintMode = ubo.brushMode > 1u;
    bool isRemoveMode = ubo.brushMode == 1u;
    if (isPaintMode || isRemoveMode) {
        // Direct SDF intersection: the fragment is inside the brush volume
        // when the brush field is negative at its world position. This is the
        // exact evaluation the brush preview raymarches, so preview and
        // painted result always agree (sweep + effects included).
        if (brushInsideBounds(fragPosWorld) && brushDistance(fragPosWorld) < 0.0) {
            int brushTexIndex = clamp(int(brush.xFlags.z), 0,
                                      max(textureSize(albedoArray, 0).z - 1, 0));
            texIndices = ivec3(brushTexIndex);
            // Override vertex HSV with the brush's HSV so painted areas get the brush tint
            hsvColor = brush.xHsv.xyz;
            if (isRemoveMode) {
                brushRedFade = (sin(ubo.brushPhase * 6.28318) + 1.0) * 0.5;
            }
        }
    }

    // Use the three tex indices and barycentric weights provided by the TES for blending
    vec3 w = fragTexWeights;
    vec2 uv = fragUV;
    bool usedTriplanar = false;

    // Geometry normal (world-space) and geometric face normal from derivatives.
    // A raymarched caller may either pass its analytic SDF normal or zero; zero
    // differentiates the marched hit positions exactly like the raster path
    // differentiates its displaced triangles (the raymarch uses zero for full
    // parity, including the tessellation bump).
    vec3 N = normalize(fragNormal);
    vec3 geomN = (dot(geomNormalOverride, geomNormalOverride) > 0.0)
        ? normalize(geomNormalOverride)
        : normalize(cross(dFdx(fragPosWorld), dFdy(fragPosWorld)));
    if (length(geomN) < 1e-5) geomN = N;
    vec3 worldNormal = N;

    // Precompute triplanar blend weights from geometric normal (abs^2 normalized)
    // Compute triplanar weights with a configurable dead-zone threshold and adjustable steepness
    vec3 triW = abs(geomN);

    // Subtract threshold and clamp so small components remain zero until threshold is exceeded
    float t = ubo.triplanarThreshold; // threshold (0..1)
    vec3 wt = max(vec3(0.0), triW - vec3(t));

    // Apply exponent to make transitions steeper. L13: skip the per-pixel
    // pow() at the default exponent 1.0 — uniform branch (no divergence),
    // exact for wt >= 0 (pow(x, 1) == x).
    float e = max(1.0, ubo.triplanarExponent);
    if (e > 1.0) wt = pow(wt, vec3(e));
    float triWSum = wt.x + wt.y + wt.z + 1e-6;
    triW = wt / triWSum;

    // Per-corner material views: each corner of the triangle may use a
    // different texture mapping mode (triplanar / single projection /
    // SdfGiven sphere / custom UV).
    MaterialNamed matA = materialNamed(materials[texIndices.x]);
    MaterialNamed matB = materialNamed(materials[texIndices.y]);
    MaterialNamed matC = materialNamed(materials[texIndices.z]);

    // Mix the triplanar flag across the three corners using each corner's
    // per-vertex texture type.
    ivec3 texTypes = fragTextureTypes;
    float triFlag = dot(vec3(float(mappingUsesTriplanar(texTypes.x)),
                             float(mappingUsesTriplanar(texTypes.y)),
                             float(mappingUsesTriplanar(texTypes.z))), w);
    usedTriplanar = triFlag > 0.5;

    // Compute the UVs and blend weights for each corner's own mapping mode
    // and reuse them across all map fetches below.
    vec2 uv0X = vec2(0.0), uv0Y = vec2(0.0), uv0Z = vec2(0.0);
    vec2 uv1X = vec2(0.0), uv1Y = vec2(0.0), uv1Z = vec2(0.0);
    vec2 uv2X = vec2(0.0), uv2Y = vec2(0.0), uv2Z = vec2(0.0);
    vec3 mw0 = vec3(1.0, 0.0, 0.0);
    vec3 mw1 = vec3(1.0, 0.0, 0.0);
    vec3 mw2 = vec3(1.0, 0.0, 0.0);
    if (w.x > 0.0) computeMaterialUVs(texTypes.x, texIndices.x, fragPosWorldNotDisplaced, geomN, uv, ubo.triplanarThreshold, ubo.triplanarExponent, uv0X, uv0Y, uv0Z, mw0);
    if (w.y > 0.0) computeMaterialUVs(texTypes.y, texIndices.y, fragPosWorldNotDisplaced, geomN, uv, ubo.triplanarThreshold, ubo.triplanarExponent, uv1X, uv1Y, uv1Z, mw1);
    if (w.z > 0.0) computeMaterialUVs(texTypes.z, texIndices.z, fragPosWorldNotDisplaced, geomN, uv, ubo.triplanarThreshold, ubo.triplanarExponent, uv2X, uv2Y, uv2Z, mw2);

    // Sample albedo per corner with that corner's own mapping weights. One-hot
    // weights (projection / SdfGiven / custom) collapse to a single fetch.
    vec3 a0 = w.x > 0.0 ? computeTriplanarAlbedoUVs(mw0, texIndices.x, uv0X, uv0Y, uv0Z) : vec3(0.0);
    vec3 a1 = w.y > 0.0 ? computeTriplanarAlbedoUVs(mw1, texIndices.y, uv1X, uv1Y, uv1Z) : vec3(0.0);
    vec3 a2 = w.z > 0.0 ? computeTriplanarAlbedoUVs(mw2, texIndices.z, uv2X, uv2Y, uv2Z) : vec3(0.0);
    vec3 albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;

    // Compute normal mapping if enabled (per-material or global toggle).
    // Triplanar-dominant patches use each corner's mapping weights; the
    // non-triplanar path samples each corner at the UV of its dominant axis.
    float mapFlag0 = float(matA.mappingEnabled);
    float mapFlag1 = float(matB.mappingEnabled);
    float mapFlag2 = float(matC.mappingEnabled);
    if ((mapFlag0 * w.x + mapFlag1 * w.y + mapFlag2 * w.z) > 0.5 || (ubo.normalMappingEnabled != 0u)) {
        if (usedTriplanar) {
            // C3: blend texels first, transform once (same weights/texels).
            vec3 tripNormal0 = w.x > 0.0 ? computeTriplanarNormalBlended(mw0, texIndices.x, N, uv0X, uv0Y, uv0Z) : vec3(0.0);
            vec3 tripNormal1 = w.y > 0.0 ? computeTriplanarNormalBlended(mw1, texIndices.y, N, uv1X, uv1Y, uv1Z) : vec3(0.0);
            vec3 tripNormal2 = w.z > 0.0 ? computeTriplanarNormalBlended(mw2, texIndices.z, N, uv2X, uv2Y, uv2Z) : vec3(0.0);
            vec3 blended = tripNormal0 * w.x + tripNormal1 * w.y + tripNormal2 * w.z;
            worldNormal = normalize(blended);
        } else {
            // Sample the normal map per corner at its dominant mapped UV and
            // blend in tangent space (C3: dead slots skipped).
            vec2 nuv0 = dominantMappingUV(mw0, uv0X, uv0Y, uv0Z);
            vec2 nuv1 = dominantMappingUV(mw1, uv1X, uv1Y, uv1Z);
            vec2 nuv2 = dominantMappingUV(mw2, uv2X, uv2Y, uv2Z);
            vec3 n0 = w.x > 0.0 ? texture(normalArray, vec3(nuv0, float(brushLayer(texIndices.x)))).rgb * 2.0 - 1.0 : vec3(0.0);
            vec3 n1 = w.y > 0.0 ? texture(normalArray, vec3(nuv1, float(brushLayer(texIndices.y)))).rgb * 2.0 - 1.0 : vec3(0.0);
            vec3 n2 = w.z > 0.0 ? texture(normalArray, vec3(nuv2, float(brushLayer(texIndices.z)))).rgb * 2.0 - 1.0 : vec3(0.0);
            vec3 nmap = normalize(n0 * w.x + n1 * w.y + n2 * w.z);
            // Build TBN matrix from geometry for UV-space normal mapping
            vec3 T = normalize(dFdx(fragPosWorld));
            vec3 B = normalize(cross(N, T));
            T = normalize(cross(B, N)); // re-orthogonalize
            mat3 TBN = mat3(T, B, N);
            worldNormal = normalize(TBN * nmap);
        }

        // Diagnostic: detect invalid/degenerate normals and show red so we can find broken pixels.
        // L14: gated on any debug view (uniform branch). Skipped in the
        // normal view, where it never fires on valid assets — a degenerate
        // normal there would render black instead of red, equally visible
        // if assets ever regress.
        if (ubo.debugMode != 0 && (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6)) {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    // Sample roughness map (R channel). One fetch on the dominant projection
    // of each corner's own mapping (triplanar or single projected UV).
    float r0 = w.x > 0.0 ? computeTriplanarRoughnessDominant(mw0, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
    float r1 = w.y > 0.0 ? computeTriplanarRoughnessDominant(mw1, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
    float r2 = w.z > 0.0 ? computeTriplanarRoughnessDominant(mw2, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
    float roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    if (!roughnessEnabled) roughnessValue = 0.0;

    // Sample ambient occlusion map (R channel), same dominant-projection
    // fetch per corner as roughness above.
    float ao0 = w.x > 0.0 ? computeTriplanarAODominant(mw0, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
    float ao1 = w.y > 0.0 ? computeTriplanarAODominant(mw1, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
    float ao2 = w.z > 0.0 ? computeTriplanarAODominant(mw2, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
    float ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);

    // Lighting calculation
    vec3 toLight = -normalize(ubo.lightDirection);
    float NdotL = max(dot(worldNormal, toLight), 0.0);

    // Shadow calculation. The raster solid path keeps the CSM authoritative
    // macro sun-shadow. The octree raymarcher defines SOLID_SHADOW_TRACE and
    // replaces the CSM
    // sample with a second SDF ray march through the same node table: no
    // terrain chunk meshes exist any more, so the cascades can no longer cast
    // terrain shadows.
    float shadow = 0.0;
    // Primary cascade hint for the secondary-hit fast path below (H6).
    // Default 0 reproduces today's secondary behavior wherever unused.
    int solidCascadeHint = 0;
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    if (ubo.shadowsEnabled != 0u) {
        if (NdotL > 0.01) {
#ifdef SOLID_SHADOW_TRACE
            shadow = solidShadowTrace(fragPosWorld, worldNormal, toLight, solidCascadeHint);
#else
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, fragPosWorld, bias, solidCascadeHint);
#endif
        } else {
            shadow = 1.0;
        }
    }
    float totalShadow = shadow;
    // Volumetric cloud shadows: project the fragment along the sun direction
    // onto the three cloud slabs and attenuate direct light. Combined as an
    // independent occluder with CSM (clouds never brighten the shadow term).
    // Skipped in the shadow pass (isShadowPass fast-path returned above) and
    // when the surface faces away (NdotL == 0 keeps full shadow = 1.0).
    float cloudShadow = 0.0;
    if ((sky.cloudsEnabled != 0u) && NdotL > 0.01) {
        cloudShadow = cloudShadowAt(fragPosWorld);
        totalShadow = 1.0 - (1.0 - totalShadow) * (1.0 - cloudShadow);
    }

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

    // Environment reflection: procedural sky gradient plus the cheap
    // single-sample cloud approximation, evaluated along the mirror direction.
    // Terrain is SDF raymarched, so there is no rasterized scene color to
    // trace against; the sky is the environment for every reflective surface.
    vec3 envReflection = vec3(0.0);
    float blendedRefStrength = 0.0;
    float envFresnelFactor = 0.0;
    if (ubo.cubemapCapture == 0u) {
        float refStrength0 = materialNamed(materials[texIndices.x]).reflectionStrength;
        float refStrength1 = materialNamed(materials[texIndices.y]).reflectionStrength;
        float refStrength2 = materialNamed(materials[texIndices.z]).reflectionStrength;
        blendedRefStrength = refStrength0 * w.x + refStrength1 * w.y + refStrength2 * w.z;
        // Skip non-reflective surfaces (the mix factor below collapses to zero).
        if (blendedRefStrength > 1e-4) {
            vec3 reflN = normalize(fragNormal);
            vec3 reflV = normalize(viewDir);
            float cosTheta = clamp(dot(reflN, reflV), 0.0, 1.0);
            float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
            float rough = clamp(roughnessValue * roughnessFactor, 0.0, 1.0);
            vec3 reflDir = reflect(-reflV, reflN);
            vec3 skyApprox = proceduralSkyGradient(normalize(reflDir), sky.horizonColor,
                                             sky.zenithColor, sky.exponent);
            skyApprox *= aoBlend * (1.0 - rough * 0.5);
            if (sky.cloudsEnabled != 0u) {
                float reflDay = smoothstep(-0.2, 0.2, -ubo.lightElevation);
                skyApprox += cloudApproxForReflection(normalize(reflDir),
                    normalize(-ubo.lightDirection), reflDay) * aoBlend;
            }
            envReflection = skyApprox;
            // Reflectivity: the material's reflectionStrength lerps between a
            // physical Fresnel mirror (0) and a full mirror at every angle
            // (1) — same convention as the water reflectionStrength — shaped
            // by the Schlick Fresnel term, then faded by the rasterized
            // material roughness (smooth ≈ full mirror, rough ≈ matte lit
            // color only).
            envFresnelFactor = mix(fresnel, 1.0, clamp(blendedRefStrength, 0.0, 1.0))
                * (1.0 - clamp(rough, 0.0, 1.0));
        }
    }

    // ── Unified debug views (IDs shared with the water path) ──
    // Canonical IDs live in includes/DebugModes.glsl (mirror of
    // vulkan/includes/debug/DebugModes.hpp). Water-only views (sky/refraction/
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
        // Raw normal-map samples at each corner's dominant mapped UV, blended
        // by barycentric weights.
        vec2 nuv0 = dominantMappingUV(mw0, uv0X, uv0Y, uv0Z);
        vec2 nuv1 = dominantMappingUV(mw1, uv1X, uv1Y, uv1Z);
        vec2 nuv2 = dominantMappingUV(mw2, uv2X, uv2Y, uv2Z);
        vec3 rn0 = texture(normalArray, vec3(nuv0, float(brushLayer(texIndices.x)))).rgb;
        vec3 rn1 = texture(normalArray, vec3(nuv1, float(brushLayer(texIndices.y)))).rgb;
        vec3 rn2 = texture(normalArray, vec3(nuv2, float(brushLayer(texIndices.z)))).rgb;
        outColor = vec4(rn0 * w.x + rn1 * w.y + rn2 * w.z, 1.0);
        return;
    }
    if (debugMode == DEBUG_MODE_HEIGHT_MAP) {
        // Final height/bump sample driving displacement. Triplanar materials
        // keep the world-space triplanar fetch; every other mapping mode
        // samples at its dominant mapped UV.
        float h0 = 0.0;
        float h1 = 0.0;
        float h2 = 0.0;
        if (w.x > 0.0) {
            h0 = mappingUsesTriplanar(texTypes.x)
                ? sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.x, uv)
                : texture(heightArray, vec3(dominantMappingUV(mw0, uv0X, uv0Y, uv0Z), float(brushLayer(texIndices.x)))).r;
        }
        if (w.y > 0.0) {
            h1 = mappingUsesTriplanar(texTypes.y)
                ? sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.y, uv)
                : texture(heightArray, vec3(dominantMappingUV(mw1, uv1X, uv1Y, uv1Z), float(brushLayer(texIndices.y)))).r;
        }
        if (w.z > 0.0) {
            h2 = mappingUsesTriplanar(texTypes.z)
                ? sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.z, uv)
                : texture(heightArray, vec3(dominantMappingUV(mw2, uv2X, uv2Y, uv2Z), float(brushLayer(texIndices.z)))).r;
        }
        float h = h0 * w.x + h1 * w.y + h2 * w.z;
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
        vec3 c0 = palette[int(mod(float(brushLayer(texIndices.x)), float(PALETTE_SIZE)) + 0.5)];
        vec3 c1 = palette[int(mod(float(brushLayer(texIndices.y)), float(PALETTE_SIZE)) + 0.5)];
        vec3 c2 = palette[int(mod(float(brushLayer(texIndices.z)), float(PALETTE_SIZE)) + 0.5)];
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
        // R = CSM, G = (unused, kept for channel stability), B = combined
        // term actually shaded.
        outColor = vec4(shadow, shadow, totalShadow, 1.0);
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
