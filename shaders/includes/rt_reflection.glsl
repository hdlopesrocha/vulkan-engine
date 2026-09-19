// Multi-bounce mirror tracing for the hybrid RT reflection paths.
//
// Requires (declared by the includer, all under RT_ENABLED): rt_params.glsl,
// ubo.glsl (ubo / waterParams / sky), textures.glsl (albedoArray),
// shadows.glsl (ShadowCalculation) and rt_scene_sample.glsl. Included from
// main.frag right after those declarations.
//
// A reflection ray that hits a reflective surface (solid with a material
// mirror strength, or a water mesh with its layer reflection enabled) spawns
// another mirror ray from the hit point, up to the configured bounce count
// (rt.water.w). GLSL forbids recursion, so the bounce chain is an iterative
// loop carrying a throughput factor.

#ifndef RT_REFLECTION_GLSL
#define RT_REFLECTION_GLSL

// Mirror strength of a scene-geometry hit. Water hits return 0: a water
// surface's mirror is already composed inside rtWaterSurfaceLook, and letting
// a reflection ray bounce off the flat BLAS water mesh spawned secondary rays
// that self-intersect neighbouring water triangles (water-on-water triangle
// noise). Solid hits carry the chunk's average material mirror strength
// (packed by SceneRenderer::rebuildProxySet) and can bounce.
float rtHitReflectivity(uint lo) {
    if (rtSceneGeomInfo[lo].w != 0u) return 0.0;
    return clamp(rtSceneAlbedo[lo].w, 0.0, 1.0);
}

// Shade a water-mesh hit with the SAME composition as the raster water
// surface (shadeWaterSurface): tint*(depthFade*waterTint) as the refraction
// base (the underwater bottom is not traced inside a reflection) blended with
// the sky reflection along reflect(incident, hitN) by the layer's
// Fresnel/strength mirror mix. This keeps water seen inside a reflection
// looking like the water surface itself.
vec3 rtWaterSurfaceLook(WaterParamsGPU wp, vec3 hitN, vec3 incidentDir,
                        vec3 skyAtReflection) {
    float thickness = max(wp.refractionParams.y, 0.0);
    float tintScale = max(wp.causticParams.w, 1e-3);
    float volumeFactor = 1.0 - exp(-thickness / tintScale);
    vec3 tintColor = mix(wp.shallowColor.rgb, wp.deepColor.rgb, volumeFactor);
    float oceanF = 1.0 - exp(-max(thickness - wp.oceanColor.w, 0.0)
                              / max(wp.oceanParams.x, 1e-3));
    tintColor = mix(tintColor, wp.oceanColor.rgb, oceanF);
    float depthFade = 1.0 - exp(-thickness * max(wp.waveParams.w, 1e-4));
    float tintMax = clamp(1.0 - wp.params1.z, 0.0, 1.0);
    float tintBlend = clamp(depthFade * wp.params2.x, 0.0, tintMax);
    vec3 refractTerm = tintColor * tintBlend;
    if (wp.reserved1.x < 0.5) return refractTerm; // reflection disabled
    float viewCos = clamp(dot(hitN, -normalize(incidentDir)), 0.0, 1.0);
    float fres = 0.02 + 0.98 * pow(1.0 - viewCos, clamp(wp.params1.y, 1.0, 8.0));
    float reflMix = (wp.reserved2.w > 0.5)
        ? clamp(wp.params1.w, 0.0, 1.0)
        : mix(fres, 1.0, clamp(wp.params1.w, 0.0, 1.0));
    return mix(refractTerm, skyAtReflection, reflMix);
}

// Convenience wrapper for the bounce tracer (procedural sky everywhere).
vec3 rtShadeWaterHit(uint lo, vec3 hitN, vec3 incoming) {
    int nWL = max(waterParams.length(), 1);
    int layer = clamp(int(rtSceneAlbedo[lo].w + 0.5), 0, nWL - 1);
    WaterParamsGPU wp = waterParams[layer];
    vec3 skyR = rtProceduralSky(normalize(reflect(incoming, hitN)),
                                sky.skyHorizon.rgb, sky.skyZenith.rgb, sky.skyParams.y);
    return rtWaterSurfaceLook(wp, hitN, incoming, skyR);
}

// Trace one mirror chain. `extraBounces` is the number of ADDITIONAL rays
// beyond this call's own trace (0 = trace once, no bounce; 1 = reflection
// inside the reflection; ... clamped to 3). Returns the accumulated color,
// sky on miss. `minHit` biases the origin out of the reflector's own surface.
vec3 rtTraceMirror(vec3 origin, vec3 dir, int extraBounces, float minHit) {
    vec3 accum = vec3(0.0);
    float throughput = 1.0;
    int traces = clamp(extraBounces, 0, 3) + 1;
    for (int b = 0; b < traces; ++b) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, rtTlas,
            gl_RayFlagsOpaqueEXT | gl_RayFlagsCullFrontFacingTrianglesEXT,
            RT_RAY_MASK_SCENE, origin, max(minHit, 0.05), dir, RT_NO_LIMIT);
        while (rayQueryProceedEXT(rq)) {}

        vec3 skyCol = rtProceduralSky(normalize(dir),
            sky.skyHorizon.rgb, sky.skyZenith.rgb, sky.skyParams.y);
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT ||
            rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true) != RT_SCENE_INSTANCE) {
            accum += throughput * skyCol;
            break;
        }

        uint lo = uint(rayQueryGetIntersectionGeometryIndexEXT(rq, true));
        uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
        vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        float t = rayQueryGetIntersectionTEXT(rq, true);
        vec3 hitPos = origin + dir * t;
        uvec4 gi = rtSceneGeomInfo[lo];

        const uint kVertStride = 16u;
        vec3 hitN = vec3(0.0, 1.0, 0.0);
        uint i0 = 0u, i1 = 0u, i2 = 0u;
        if (gi.w == 0u) {
            i0 = rtSceneIndices[gi.y + prim * 3u + 0u] + gi.x;
            i1 = rtSceneIndices[gi.y + prim * 3u + 1u] + gi.x;
            i2 = rtSceneIndices[gi.y + prim * 3u + 2u] + gi.x;
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
        if (dot(hitN, dir) > 0.0) hitN = -hitN; // face the incoming ray

        vec3 lit;
        if (gi.w != 0u) {
            lit = rtShadeWaterHit(lo, hitN, dir);
        } else {
            int maxLayer = max(int(textureSize(albedoArray, 0).z) - 1, 0);
            vec3 albedo = rtSceneSampleReflectionAlbedo(i0, i1, i2, bary, hitPos, hitN, maxLayer, 0.0);
            vec3 toSun = normalize(-ubo.lightDir.xyz);
            float ndl = max(dot(hitN, toSun), 0.0);
            float hitShadow = ShadowCalculation(
                ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
            lit = albedo * (ubo.lightColor.rgb * ndl * (1.0 - hitShadow) + vec3(0.26));
        }

        float refl = rtHitReflectivity(lo);
        bool bounceOn = (b + 1 < traces) && (refl > 0.02);
        if (bounceOn) {
            // Outgoing = diffuse part now, mirror part through the next ray
            // (blended when the chain resolves).
            accum += throughput * lit * (1.0 - refl);
            throughput *= refl;
            origin = hitPos + hitN * 0.05;
            dir = normalize(reflect(dir, hitN));
        } else {
            accum += throughput * lit;
            break;
        }
    }
    return accum;
}

// Resolve a water surface hit seen inside a ray (reflection/refraction path):
// trace the Snell refraction ray for the underwater scene + water column and
// one bounded mirror chain for the surface reflection, then compose exactly
// like the raster water surface (tint + Beer-Lambert over the refracted
// scene, mixed with the reflected scene by Fresnel/strength). CSM shadows are
// evaluated on the refracted bottom and inside the mirror chain, so shadowed
// water stays dark when seen in a reflection.
//
// Not recursive: the mirror chain shades water hits with rtShadeWaterHit()
// (look only), never with this function.
vec3 rtResolveWaterHit(WaterParamsGPU wp, vec3 hitPos, vec3 hitN, vec3 incidentDir) {
    vec3 inc = normalize(incidentDir);
    float ior = clamp(wp.refractionParams.x, 1.0, 2.5);
    float thickCap = max(wp.refractionParams.y, 0.0);

    // --- Refraction: nearest scene triangle behind the surface along Snell ---
    vec3 refrDir = refract(inc, hitN, 1.0 / ior);
    if (dot(refrDir, refrDir) < 1e-6) refrDir = reflect(inc, hitN); // TIR
    refrDir = normalize(refrDir);
    vec3 refrColor = vec3(0.0);
    float thickness = 0.0;
    bool haveBottom = false;
    {
        float tMax = max(thickCap * 3.0, 8.0);
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsNoOpaqueEXT, RT_RAY_MASK_SCENE,
                              hitPos + hitN * 0.05, 0.05, refrDir, tMax);
        float bestT = -1.0;
        uint bestPrim = 0u, bestLo = 0u;
        vec2 bestBary = vec2(0.0);
        bool bestWater = false;
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCommittedIntersectionNoneEXT) continue;
            if (rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) != RT_SCENE_INSTANCE) continue;
            float t = rayQueryGetIntersectionTEXT(rq, false);
            uint lo = uint(rayQueryGetIntersectionGeometryIndexEXT(rq, false));
            if (bestT < 0.0 || t < bestT) {
                bestT = t;
                bestLo = lo;
                bestPrim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, false));
                bestBary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
                bestWater = (rtSceneGeomInfo[lo].w != 0u);
            }
        }
        if (bestT >= 0.0) {
            thickness = min(bestT, thickCap);
            if (!bestWater) {
                uvec4 gi = rtSceneGeomInfo[bestLo];
                const uint kVertStride = 16u;
                uint i0 = rtSceneIndices[gi.y + bestPrim * 3u + 0u] + gi.x;
                uint i1 = rtSceneIndices[gi.y + bestPrim * 3u + 1u] + gi.x;
                uint i2 = rtSceneIndices[gi.y + bestPrim * 3u + 2u] + gi.x;
                vec3 n0 = vec3(rtSceneVerts[i0 * kVertStride + 8u],
                               rtSceneVerts[i0 * kVertStride + 9u],
                               rtSceneVerts[i0 * kVertStride + 10u]);
                vec3 n1 = vec3(rtSceneVerts[i1 * kVertStride + 8u],
                               rtSceneVerts[i1 * kVertStride + 9u],
                               rtSceneVerts[i1 * kVertStride + 10u]);
                vec3 n2 = vec3(rtSceneVerts[i2 * kVertStride + 8u],
                               rtSceneVerts[i2 * kVertStride + 9u],
                               rtSceneVerts[i2 * kVertStride + 10u]);
                vec3 bN = normalize(n0 * (1.0 - bestBary.x - bestBary.y)
                                    + n1 * bestBary.x + n2 * bestBary.y);
                if (dot(bN, refrDir) > 0.0) bN = -bN;
                vec3 hp = hitPos + refrDir * bestT;
                int maxLayer = max(int(textureSize(albedoArray, 0).z) - 1, 0);
                vec3 alb = rtSceneSampleReflectionAlbedo(i0, i1, i2, bestBary, hp, bN,
                                                         maxLayer, 0.0);
                vec3 toSun = normalize(-ubo.lightDir.xyz);
                float ndl = max(dot(bN, toSun), 0.0);
                float bShadow = ShadowCalculation(
                    ubo.lightSpaceMatrix * vec4(hp, 1.0), hp, 0.0015);
                refrColor = alb * (ubo.lightColor.rgb * ndl * (1.0 - bShadow)
                                   + vec3(0.26));
                haveBottom = true;
            }
        }
    }

    // --- Water tint / absorption over the refracted column ---
    float tintScale = max(wp.causticParams.w, 1e-3);
    float volumeFactor = 1.0 - exp(-thickness / tintScale);
    vec3 tintColor = mix(wp.shallowColor.rgb, wp.deepColor.rgb, volumeFactor);
    float oceanF = 1.0 - exp(-max(thickness - wp.oceanColor.w, 0.0)
                              / max(wp.oceanParams.x, 1e-3));
    tintColor = mix(tintColor, wp.oceanColor.rgb, oceanF);
    // CSM shadow evaluated AT THE WATER SURFACE: darken the composed water
    // (volume tint + surface mirror), matching the raster water convention
    // (waterColor *= mix(1.0, 0.55, shadow)). The refracted bottom below
    // carries its own CSM shadow too, so shallow lakes show terrain shadows
    // through the water while deep water darkens via this factor. Without it,
    // water seen inside a reflection stayed fully lit while the raster surface
    // beside it sat in shadow.
    float waterShadow = ShadowCalculation(
        ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
    float depthFade = 1.0 - exp(-thickness * max(wp.waveParams.w, 1e-4));
    float tintMax = clamp(1.0 - wp.params1.z, 0.0, 1.0);
    float tintBlend = clamp(depthFade * wp.params2.x, 0.0, tintMax);

    vec3 refractTerm;
    if (haveBottom) {
        vec3 transmittance = exp(-min(
            wp.absorptionParams.rgb * max(thickness * wp.absorptionParams.a, 0.0),
            vec3(2.5)));
        refractTerm = mix(refrColor * transmittance, tintColor, tintBlend);
    } else {
        // Deep / no bottom resolved: sky along the refraction direction,
        // saturated to the deep tint (primary water's miss continuity).
        vec3 deepScene = rtProceduralSky(refrDir, sky.skyHorizon.rgb,
                                         sky.skyZenith.rgb, sky.skyParams.y);
        refractTerm = mix(deepScene, tintColor, max(tintBlend, 0.85));
    }

    if (wp.reserved1.x < 0.5) return refractTerm * mix(1.0, 0.55, waterShadow); // reflection disabled

    // --- Reflection: one bounded mirror chain (water hits there are look-only) ---
    vec3 reflDir = normalize(reflect(inc, hitN));
    vec3 reflColor = rtTraceMirror(hitPos + hitN * 0.05, reflDir, 0, 0.05);
    float viewCos = clamp(dot(hitN, -inc), 0.0, 1.0);
    float fres = 0.02 + 0.98 * pow(1.0 - viewCos, clamp(wp.params1.y, 1.0, 8.0));
    float reflMix = (wp.reserved2.w > 0.5)
        ? clamp(wp.params1.w, 0.0, 1.0)
        : mix(fres, 1.0, clamp(wp.params1.w, 0.0, 1.0));
    return mix(refractTerm, reflColor, reflMix) * mix(1.0, 0.55, waterShadow);
}

#endif // RT_REFLECTION_GLSL
