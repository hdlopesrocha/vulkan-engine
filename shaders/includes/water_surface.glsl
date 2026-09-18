// Water surface shading + ray helpers — extracted from water.frag so the
// water pass and the future main-pass water variant share one
// implementation (Phase-1 water-in-main migration). Writes the global
// outColor (alpha = blend factor for the main-pass path).
// Requires (declared by the includer before this file): ubo /
// waterRenderUBO / waterParams, water varyings, set-2 scene textures,
// RT bindings (RT_ENABLED), hsv + perlin + water_noise + voronoi.
#ifdef RT_ENABLED
// Sample the same-frame raster bottom where Snell ray (O, S-normalized)
// would land on it — the zero-ray recovery for pixels whose refraction ray
// missed or was budget-skipped. Unprojects the view-ray bottom to world,
// intersects S with the horizontal plane through it, and samples THAT
// landing point with the hit path's depth-tolerance lookup (+vegetation
// composite), so traced hits and ray-less pixels shade the same texel
// identically. Returns rgb=bottom color, a=along-ray distance (>=0);
// a=-1 when no raster bottom stands beneath the pixel (thin slip).
// Explicit LOD (callable under divergent control flow).
vec4 rtRasterBottom(vec3 O, vec3 S, vec2 suv) {
    float dB = textureLod(solidSceneDepthTex, suv, 0.0).r;
    if (dB >= 1.0) return vec4(0.0, 0.0, 0.0, -1.0);
    vec4 bwH = ubo.invViewProjection * vec4(suv * 2.0 - 1.0, dB, 1.0);
    vec3 Bw = bwH.xyz / max(bwH.w, 1e-6);
    // Raw view-ray sample: guaranteed bottom-below-pixel color used when
    // the plane landing below fails its consistency check.
    vec3 landColor = textureLod(solidSceneColorTex, suv, 0.0).rgb;
    float landT = max(dot(Bw - O, S), 0.0);
    // Intersect S with the horizontal plane through Bw: t* matches hitT's
    // definition (along-ray distance from the shared undisplaced entry)
    // for one locally-flat bottom.
    if (S.y < -1e-4 && Bw.y < O.y) {
        float tPlane = (Bw.y - O.y) / S.y;
        vec3 L = O + S * tPlane;
        vec4 clipL = ubo.viewProjection * vec4(L, 1.0);
        if (clipL.w > 0.001) {
            vec2 huvL = clipL.xy / clipL.w * 0.5 + 0.5;
            if (huvL.x >= 0.0 && huvL.x <= 1.0 && huvL.y >= 0.0 && huvL.y <= 1.0) {
                float hdL = textureLod(solidSceneDepthTex, huvL, 0.0).r;
                if (hdL < 1.0) {
                    const float nearB = ubo.passParams.z;
                    const float farB = ubo.passParams.w;
                    float sceneEyeL = (nearB * farB) / (farB - hdL * (farB - nearB));
                    if (abs(clipL.w - sceneEyeL) < max(2.0, sceneEyeL * 0.02)) {
                        // Snell-consistent sample (+veg layer, exactly like
                        // the hit path).
                        landColor = textureLod(solidSceneColorTex, huvL, 0.0).rgb;
                        float hitNdcZL = clipL.z / clipL.w;
                        float vegDL = textureLod(vegDepthTex, huvL, 0.0).r;
                        vec4 vegCL = textureLod(vegColorTex, huvL, 0.0);
                        if (vegCL.a > 0.0 && !(hitNdcZL < vegDL)) {
                            landColor = mix(landColor, vegCL.rgb, vegCL.a);
                        }
                        landT = tPlane;
                    }
                }
            }
        }
    }
    return vec4(landColor, landT);
}
#endif

#ifdef RT_ENABLED
// Trace one water secondary ray. Both paths trace the real scene-geometry
// instance (exact chunk triangles, water mesh flagged in geomInfo.w).
// refraction=true: downward Snell ray from the TRUE displaced surface,
//   enumerated without culling order, resolving the lake-bottom color from
//   the nearest SOLID triangle (water candidates skipped) with a =
//   underwater path length (capped at thickCap) or RT_DEEP_WATER when no
//   solid triangle exists along the ray.
// refraction=false: mirror ray resolving the reflected scene the staged way
//   (forward culled, reverse culled, forward unculled); a = 1 on a hit,
//   0 on a miss.
// Macro shadows stay CSM-owned: hits get ambient + sun diffuse only.
vec4 rtTraceWater(vec3 origin, vec3 dir, float tMax, bool refraction, float thickCap) {
    rayQueryEXT rq;
    // Refraction rays start AT the displaced water surface and go down.
    // tMin stays tiny (1 cm): the filtered pass skips water-mesh candidates,
    // so coplanar touch is the only self-guard needed, and shoreline
    // shallows (<5 cm deep) still hit the lake bottom underneath instead of
    // missing to deep-water tint. Reflection keeps the 5 cm tMin (self-hit
    // guard, origin biased above the surface at the call site).
    float tMin = refraction ? 0.01 : 0.05;
    // Both paths trace the real scene-geometry instance only (exact chunk
    // triangles, including the real water mesh flagged in geomInfo.w): proxy
    // boxes are excluded like in main.frag (their coarse flat tops imprint
    // the dominant material and stepped heights on refraction hits). Both
    // stay opaque and cull ray-front faces so only rasterizer-visible
    // triangles report (scene meshes wind CW-outward for the BACK+CW
    // rasterizer; ray-front is fixed CCW).
    rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsOpaqueEXT |
        gl_RayFlagsCullFrontFacingTrianglesEXT, RT_RAY_MASK_SCENE,
        origin, tMin, dir, tMax);
    while (rayQueryProceedEXT(rq)) {}
    // Explicit LOD: reachable under per-fragment control flow (pipe validity
    // / toggles / hit-vs-miss differ per pixel), where implicit-LOD texture()
    // has undefined derivatives.
    vec3 sky = textureLod(skyEquirectTex, rtDirToEquirectUV(normalize(dir)), 0.0).rgb;
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        // Forward miss. For refraction, validate the local water column
        // from the opposite direction before accepting it (see haveHit
        // selection below); reflection misses stay plain sky, and the
        // refraction deep marker is resolved against the raster bottom at
        // the call site (miss continuity), not here.
        if (!refraction) {
            return vec4(sky, 0.0);
        }
    }
    // Hit selection: refraction uses a FILTERED pass, reflection the staged
    // chain below. The BLAS holds undisplaced geometry while the raster
    // shows tessellated/displaced surfaces (waves up to bumpAmplitude off
    // the base plane), so any fixed cull set is blind to flipped faces and
    // any base-plane origin mismeasures the visible column. Refraction
    // therefore traces from the TRUE displaced surface with NO culling
    // order, enumerating every candidate and keeping the nearest SOLID
    // triangle; own water surface (geomInfo.w) is skipped so trough walls
    // and grazing crests can never report air gaps as water depth. The
    // reported length IS the visible water column — no correction needed.
    bool haveHit = false;
    float hitT = 0.0;
    uint prim = 0u;
    uint lo = 0u;
    vec2 hitBary = vec2(0.0);
    if (refraction) {
        // FILTERED refraction pass (caller passes the displaced surface
        // point as origin). NoOpaque: every candidate is enumerated (an
        // opaque query would stop at the first triangle); no cull flags:
        // displacement flips faces, so no fixed cull set is trusted.
        // Water-mesh candidates COUNT (user): the nearest WATER and nearest
        // SOLID triangles are tracked separately and whichever bounds the
        // water column first wins — min(water backface, solid front face).
        // The shared shading below handles both winners (gi.w water-look
        // vs triplanar bottom). Ties break toward solid (deterministic, no
        // flicker on coplanar water/solid).
        rayQueryEXT rqf;
        rayQueryInitializeEXT(rqf, rtTlas, gl_RayFlagsNoOpaqueEXT, RT_RAY_MASK_SCENE,
            origin, tMin, dir, tMax);
        float bestTS = -1.0;
        uint bestPrimS = 0u;
        uint bestLoS = 0u;
        vec2 bestBaryS = vec2(0.0);
        float bestTW = -1.0;
        uint bestPrimW = 0u;
        uint bestLoW = 0u;
        vec2 bestBaryW = vec2(0.0);
        while (rayQueryProceedEXT(rqf)) {
            if (rayQueryGetIntersectionTypeEXT(rqf, false) == gl_RayQueryCommittedIntersectionNoneEXT) continue;
            if (rayQueryGetIntersectionInstanceCustomIndexEXT(rqf, false) != RT_SCENE_INSTANCE) continue;
            uint loC = uint(rayQueryGetIntersectionGeometryIndexEXT(rqf, false));
            float tC = rayQueryGetIntersectionTEXT(rqf, false);
            if (rtSceneGeomInfo[loC].w != 0u) {
                if (bestTW < 0.0 || tC < bestTW) {
                    bestTW = tC;
                    bestPrimW = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rqf, false));
                    bestLoW = loC;
                    bestBaryW = rayQueryGetIntersectionBarycentricsEXT(rqf, false);
                }
            } else {
                if (bestTS < 0.0 || tC < bestTS) {
                    bestTS = tC;
                    bestPrimS = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rqf, false));
                    bestLoS = loC;
                    bestBaryS = rayQueryGetIntersectionBarycentricsEXT(rqf, false);
                }
            }
        }
        if (bestTS >= 0.0 && (bestTW < 0.0 || bestTS <= bestTW)) {
            hitT = bestTS;
            prim = bestPrimS;
            lo = bestLoS;
            hitBary = bestBaryS;
            haveHit = true;
        } else if (bestTW >= 0.0) {
            hitT = bestTW;
            prim = bestPrimW;
            lo = bestLoW;
            hitBary = bestBaryW;
            haveHit = true;
        }
    } else {
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
        rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true) == RT_SCENE_INSTANCE) {
        hitT = rayQueryGetIntersectionTEXT(rq, true);
        prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
        lo = uint(rayQueryGetIntersectionGeometryIndexEXT(rq, true));
        hitBary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        haveHit = true;
    } else {
        // Opposite-direction validation of the local column (reflection;
        // refraction uses the filtered pass above). Water is allowed beyond
        // the same dead zone the forward tMin enforces (5.5 cm > 5 cm bias
        // + float noise): distant lake surfaces still mirror, but the
        // origin's own plane can never rediscover itself as "sky".
        float backSpan = min(tMax, max(thickCap * 2.0, 4.0));
        vec3 backOrigin = origin + dir * backSpan;
        rayQueryEXT rq2;
        rayQueryInitializeEXT(rq2, rtTlas, gl_RayFlagsOpaqueEXT |
            gl_RayFlagsCullFrontFacingTrianglesEXT, RT_RAY_MASK_SCENE,
            backOrigin, 0.01, -dir, backSpan);
        while (rayQueryProceedEXT(rq2)) {}
        if (rayQueryGetIntersectionTypeEXT(rq2, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
            rayQueryGetIntersectionInstanceCustomIndexEXT(rq2, true) == RT_SCENE_INSTANCE) {
            uint lo2 = uint(rayQueryGetIntersectionGeometryIndexEXT(rq2, true));
            float fwdT = backSpan - rayQueryGetIntersectionTEXT(rq2, true);
            bool waterOk = (rtSceneGeomInfo[lo2].w == 0u)
                || (!refraction && fwdT > 0.055);
            if (waterOk && fwdT >= 0.0) {
                hitT = fwdT;
                prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq2, true));
                lo = lo2;
                hitBary = rayQueryGetIntersectionBarycentricsEXT(rq2, true);
                haveHit = true;
            }
        }
        if (!haveHit) {
            // Final fallback: same forward ray with NO face culling. The two
            // culled rays above partition all faces on the near segment
            // (forward set + complement), so this ray's unique contribution
            // is flipped faces anywhere along the FULL segment — displaced
            // overhangs and flipped panels beyond the local backSpan that
            // neither cull set can report. Water rule mirrors stage 1:
            // refraction skips its own surface (seen from above within tMin
            // like the forward ray); reflection likewise relies on its
            // origin bias + tMin, so its mesh may report like any hit.
            rayQueryEXT rq3;
            rayQueryInitializeEXT(rq3, rtTlas, gl_RayFlagsOpaqueEXT, RT_RAY_MASK_SCENE,
                origin, tMin, dir, tMax);
            while (rayQueryProceedEXT(rq3)) {}
            if (rayQueryGetIntersectionTypeEXT(rq3, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
                rayQueryGetIntersectionInstanceCustomIndexEXT(rq3, true) == RT_SCENE_INSTANCE) {
                uint lo3 = uint(rayQueryGetIntersectionGeometryIndexEXT(rq3, true));
                if (rtSceneGeomInfo[lo3].w == 0u || !refraction) {
                    hitT = rayQueryGetIntersectionTEXT(rq3, true);
                    prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq3, true));
                    lo = lo3;
                    hitBary = rayQueryGetIntersectionBarycentricsEXT(rq3, true);
                    haveHit = true;
                }
            }
        }
    } // end backward-else
    } // end refraction(filtered) / reflection(staged) split
    if (!haveHit) {
        // Deep-water marker (refraction filtered miss) or plain sky
        // (reflection triple miss). The deep marker is resolved against the
        // raster bottom at the call site, not here: this function has no
        // view of the solid depth target.
        return refraction ? vec4(sky, RT_DEEP_WATER) : vec4(sky, 0.0);
    }
    vec3 hitPos = origin + dir * hitT;
    // Underwater length cap (Beer-Lambert guard), shared by all hit returns
    // below (the old coarse-box feather toward deep is obsolete: exact
    // triangles need no terracing workarounds).
    float cap = max(thickCap, 0.0);

    { // haveHit guarantees a scene-geometry hit from either direction.
        // Real triangle hit: shade with the owning chunk's average albedo.
        // The primitive index is LOCAL to the hit geometry (GLSL_EXT_ray_query
        // semantics); the geometry index comes from the ray query directly.
        // A cumulative primBase binary search would map local indices onto the
        // wrong chunk for every geometry after the first (corrupting
        // brushIndex/UV/normal reads → wrong textures in reflections).
        const uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
        const uint lo = uint(rayQueryGetIntersectionGeometryIndexEXT(rq, true));
        // Screen-space color lookup FIRST: sample this frame's solid render at
        // the reflected hit point so the mirror shows the terrain exactly as
        // it appears on screen (mixed ground cover, shadows, detail) instead
        // of the chunk's single dominant-material sample (a flat dirt blob).
        // The water pass runs after the solid pass, so the targets are current.
        {
            const float nearP = ubo.passParams.z;
            const float farP = ubo.passParams.w;
            // World-to-clip projection is viewProjection (NOT its inverse:
            // invViewProjection maps clip->world, e.g. sky_fullscreen.vert;
            // main.frag's traceSSR projects with prevVP * world the same
            // way). Using the inverse here scatters huv across the target
            // (wrong reflection/refraction textures wherever the depth check
            // happens to pass) and starves the exact path everywhere else.
            vec4 hc = ubo.viewProjection * vec4(hitPos, 1.0);
            if (hc.w > 0.001) {
                vec2 huv = hc.xy / hc.w * 0.5 + 0.5;
                if (huv.x >= 0.0 && huv.x <= 1.0 && huv.y >= 0.0 && huv.y <= 1.0) {
                    float hd = textureLod(solidSceneDepthTex, huv, 0.0).r;
                    if (hd < 1.0) {
                        float sceneEye = (nearP * farP) / (farP - hd * (farP - nearP));
                        if (abs(hc.w - sceneEye) < max(2.0, sceneEye * 0.02)) {
                            vec3 hitColor = textureLod(solidSceneColorTex, huv, 0.0).rgb;
                            // Composite the vegetation layer exactly like
                            // postprocess.frag: use it when it sits in front of
                            // the reflected surface (else grass-covered hills
                            // would mirror as bare terrain).
                            float hitNdcZ = hc.z / hc.w;
                            float vegD = textureLod(vegDepthTex, huv, 0.0).r;
                            vec4 vegC = textureLod(vegColorTex, huv, 0.0);
                            if (vegC.a > 0.0 && !(hitNdcZ < vegD)) {
                                hitColor = mix(hitColor, vegC.rgb, vegC.a);
                            }
                            // Refraction reports the underwater path length as
                            // thickness (capped); reflection reports a plain hit.
                            return vec4(hitColor, refraction ? min(hitT, cap) : 1.0);
                        }
                    }
                }
            }
        }
        // Off-screen fallback: real interpolated triangle normal (see
        // main.frag) for relief and correct shading; albedo is the triplanar
        // sample at the true hit position, blended to the chunk average with
        // distance.
        uvec4 gi = rtSceneGeomInfo[lo];
        // prim is already local to this geometry (ray-query semantics) — do
        // NOT subtract the cumulative base.
        uint localPrim = prim;
        const uint kVertStride = 16u;
        uint i0 = rtSceneIndices[gi.y + localPrim * 3u + 0u] + gi.x;
        uint i1 = rtSceneIndices[gi.y + localPrim * 3u + 1u] + gi.x;
        uint i2 = rtSceneIndices[gi.y + localPrim * 3u + 2u] + gi.x;
        vec3 n0 = vec3(rtSceneVerts[i0 * kVertStride + 8u],
                       rtSceneVerts[i0 * kVertStride + 9u],
                       rtSceneVerts[i0 * kVertStride + 10u]);
        vec3 n1 = vec3(rtSceneVerts[i1 * kVertStride + 8u],
                       rtSceneVerts[i1 * kVertStride + 9u],
                       rtSceneVerts[i1 * kVertStride + 10u]);
        vec3 n2 = vec3(rtSceneVerts[i2 * kVertStride + 8u],
                       rtSceneVerts[i2 * kVertStride + 9u],
                       rtSceneVerts[i2 * kVertStride + 10u]);
        vec2 bary = hitBary;
        vec3 hitN = normalize(n0 * (1.0 - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);
        if (dot(hitN, dir) > 0.0) hitN = -hitN;
        // Water chunk marker (geomInfo.w): the real water mesh is in the scene
        // BLAS — a water surface reflecting another water surface. Shade it as
        // water (sky reflection + tint); its brushIndex addresses water
        // params, not scene materials.
        if (gi.w > 0u) {
            // Reflected water: transparent water look computed from the
            // water's OWN params (water.frag formula): tint = mix(shallow,
            // deep, volume) blended over the sky by waterTint*transparency.
            // No recursive reflection/refraction.
            // Stable water layer id: chunk-dominant, carried in
            // rtSceneAlbedo[lo].w (per-vertex brushIndex can vary within a
            // triangle → color flicker). Guarded to the water SSBO range
            // (default layer 0), like the fragment entry.
            int nWLL = max(waterParams.length(), 1);
            int wId = int(rtSceneAlbedo[lo].w + 0.5);
            int wLayer = (wId >= 0 && wId < nWLL) ? wId : 0;
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
            vec3 transmittance = exp(-min(
                wp.absorptionParams.rgb * max(thickness * wp.absorptionParams.a, 0.0),
                vec3(2.5)));
            float depthFade = 1.0 - exp(-thickness * depthFalloff);
            float tintMax = clamp(1.0 - transparency, 0.0, 1.0);
            float tintBlend = clamp(depthFade * waterTintStr, 0.0, tintMax);
            vec3 toSun = normalize(rt.sunDir.xyz);
            float ndl = max(dot(hitN, toSun), 0.0);
            // CSM shadow at the reflected hit, like the terrain branch below:
            // without it a shadowed lake still reflects fully lit in mirrors
            // and other water surfaces.
            float hitShadow = ShadowCalculation(
                ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
            vec3 waterColor = mix(sky * transmittance, waterTintColor, tintBlend)
                * (rt.sunColor.rgb * (0.55 + 0.45 * ndl) * (1.0 - hitShadow) + vec3(0.09, 0.12, 0.15));
            return vec4(waterColor, refraction ? min(hitT, cap) : 1.0);
        }
        int maxLayer = max(int(textureSize(albedoArray, 0).z) - 1, 0);
        // Distance-based texture LOD (this helper runs under per-fragment
        // control flow where implicit-LOD texture() is undefined): refraction
        // looks down at possibly distant bottoms, so clamp shimmer there.
        // Reflections keep 0.0 to match main.frag exactly.
        float hitLod = refraction ? clamp(log2(1.0 + hitT * 0.02), 0.0, 4.0) : 0.0;
        // Real painted material at this triangle (Vertex float offset 11 =
        // brushIndex). The proxy registry only knows the chunk's DOMINANT
        // brush, which loses the ground-cover mix painted per vertex — the
        // main source of "wrong textures" in reflections.
        // brushIndex is an int stored in the float-typed vertex pool: read its
        // bit pattern (reading it as a float yields a denormal, decoding to 0).
        const int matId = clamp(floatBitsToInt(rtSceneVerts[i0 * kVertStride + 11u]), 0, maxLayer);
        // Exact raster texture lookup: the three corner materials are
        // compressed into unique slots and blended by the hit barycentrics
        // (main.tesc + main.frag). This is what fixes "dirt where grass
        // should be" at material boundaries inside triangles.
        vec3 albedo = rtSceneSampleReflectionAlbedo(i0, i1, i2, bary, hitPos, hitN, maxLayer, hitLod);
        // Full shading for the hit: real texture albedo, the real
        // interpolated normal, sun diffuse (CSM shadowed) + sky ambient.
        // Refraction and reflection share it: both show lit scenery, not a
        // flat fill.
        vec3 toSun = normalize(rt.sunDir.xyz);
        float ndl = max(dot(hitN, toSun), 0.0);
        float hitShadow = ShadowCalculation(
            ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
        vec3 color = albedo * (rt.sunColor.rgb * ndl * (1.0 - hitShadow)
                               + vec3(0.09, 0.12, 0.15));
        return vec4(color, refraction ? min(hitT, cap) : 1.0);
    }

    // Unreachable with the current masks (both paths select the scene
    // instance only), but keep a well-defined miss-shaped fallback so a
    // future mask change cannot fall out of a non-void function.
    return refraction ? vec4(sky, RT_DEEP_WATER) : vec4(sky, 0.0);
}
#endif

// Near/far planes for linearizing depth – read from UBO passParams (z = near, w = far)
// so they always match the glm::perspective call on the CPU side.


// Linearize depth from Vulkan [0,1] depth buffer to eye-space distance.
// With GLM_FORCE_DEPTH_ZERO_TO_ONE the projection maps z_eye to [0,1]:
//   d = f*(z - n) / (z*(f - n))   =>   z = n*f / (f - d*(f - n))
float linearizeDepth(float depth) {
    float nearPlane = ubo.passParams.z;
    float farPlane  = ubo.passParams.w;
    return (nearPlane * farPlane) / (farPlane - depth * (farPlane - nearPlane));
}

// Screen-space reflection march over the solid pass depth (set 2, binding 5).
// Returns hit color in rgb and a confidence in a (0 = no usable hit). The ray
// is projected with the camera inverse view-projection (the water pass runs
// after the solid pass, so this frame's targets are current). Marching front
// to back, the first depth crossing is the first real intersection, so any
// crossing counts as a hit and a binary search refines it — this is
// step-size independent, unlike a thin thickness window. Covers up to ~2 km
// so distant mirrors (the polished spheres) reflect on-screen scenery at the
// right positions instead of the flat proxy boxes. `eyeDir` is the unit
// vector from the surface to the camera: rays nearly tangent to the view
// direction are where screen-space marching is least reliable, so they fade.
vec4 traceSSR(vec3 origin, vec3 dir, vec3 eyeDir) {
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
        // World-to-clip is viewProjection (see the note at the rtTraceWater
        // lookup). NOTE: traceSSR is currently uncalled (the inline trace
        // hits exact triangles directly); kept consistent so a future caller
        // does not inherit the old inverse-matrix projection bug.
        vec4 clip = ubo.viewProjection * vec4(P, 1.0);
        if (clip.w <= 0.001 || clip.w > farP * 2.0) break;
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
        float d = textureLod(solidSceneDepthTex, uv, 0.0).r;
        if (d < 1.0) {
            float sceneEye = (nearP * farP) / (farP - d * (farP - nearP));
            float rayEye = clip.w; // GLM perspective: clip.w == eye depth
            if (rayEye > sceneEye + 0.05) {
                // First crossing: refine with a binary search (5 iterations).
                float lo = prevT, hi = t;
                for (int j = 0; j < 5; ++j) {
                    float mid = 0.5 * (lo + hi);
                    vec4 cm = ubo.viewProjection * vec4(origin + dir * mid, 1.0);
                    vec2 uvm = cm.xy / cm.w * 0.5 + 0.5;
                    float dm = textureLod(solidSceneDepthTex, uvm, 0.0).r;
                    float em = (nearP * farP) / (farP - dm * (farP - nearP));
                    if (cm.w > em) { hi = mid; uv = uvm; rayEye = cm.w; sceneEye = em; }
                    else lo = mid;
                }
                float edge = smoothstep(0.0, 0.06, uv.x) * smoothstep(0.0, 0.06, 1.0 - uv.x)
                           * smoothstep(0.0, 0.06, uv.y) * smoothstep(0.0, 0.06, 1.0 - uv.y);
                // Crossings further behind the surface are less certain
                // (ray nearly parallel to it): soften instead of hard-cutting.
                float gapFade = 1.0 - clamp((rayEye - sceneEye) / max(1.0, sceneEye * 0.25), 0.0, 0.5);
                return vec4(textureLod(solidSceneColorTex, uv, 0.0).rgb,
                            edge * gapFade * smoothstep(0.03, 0.35, facing));
            }
        }
        prevT = t;
    }
    return vec4(0.0);
}

vec2 waterDirToEquirectUV(vec3 dir) {
    const float PI = 3.14159265358979;
    vec2 uv;
    uv.x = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return uv;
}

void shadeWaterSurface() {
    // Get water parameters from SSBO indexed by fragment brushIndex.
    // Brush ids are TERRAIN paint ids (0..N textures) while the water SSBO
    // holds only a few layers: out-of-range ids (e.g. painted shore rings)
    // would read past the allocation (undefined values: black opaque water
    // with RT disabled). Fall back to layer 0 (the default water look) so
    // every pixel renders defined water. fragBrushIndex itself is left raw
    // so debug view 61 can still show the true id distribution.
    int nWaterLayers = max(waterParams.length(), 1);
    int waterLi = (fragBrushIndex >= 0 && fragBrushIndex < nWaterLayers) ? fragBrushIndex : 0;
    WaterParamsGPU wp = waterParams[waterLi];
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

    // Feature toggles
    bool enableReflection = wp.reserved1.x > 0.5;
    bool enableRefraction = wp.reserved1.y > 0.5;
    // During 360 cubemap capture, skip reflection/refraction to avoid feedback.
    const bool captureMode = ubo.materialFlags.x > 0.5;
    if (captureMode) { enableReflection = false; enableRefraction = false; }

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
    // Per-lobe precedence is documented in the header above (reflection:
    // exact-inline-wins with pipe covering budget skips; refraction:
    // pipe-first with inline fallback). rt.debug.w carries
    // settings.rtWaterPipeline.
    bool rtReady = false;
    bool usePipe = false;
#ifdef RT_ENABLED
    rtReady = (rt.debug.y > 0.5);
    usePipe = (rt.debug.w > 0.5);
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
    
    // Sample refraction via HARDWARE RAY TRACING (§10: Snell).
    // Path selection: async RT pipeline outputs (half-res, 1-frame latency)
    // when enabled, else inline ray queries (full-res, no latency), else the
    // sky equirect. The legacy 360° capture cubemap is removed.
    // rtThickness carries the RT underwater path length when provided (>= 0).
    // Water look (IOR, absorption, thickness cap) comes from the per-layer
    // water params — the single source of truth (no RT-global duplicates).
    vec3 sceneColor = vec3(0.0);
    float rtThickness = -1.0;
    // True when rtThickness below came from an exact scene-triangle hit
    // (not a proxy average): the hash dither against proxy terracing must
    // not touch it.
    bool rtThickFromScene = false;
    float waterIor = clamp(wp.refractionParams.x, 1.0, 2.5);
    float refrThickCap = max(wp.refractionParams.y, 0.0);
    vec3 absorbCoeff = wp.absorptionParams.rgb;
    float absorbScaleBase = max(wp.absorptionParams.a, 0.0);
#ifdef RT_ENABLED
    // Shared ray constant (also read by the Beer-Lambert block below:
    // the deep-water marker substitutes maxRefr as a caustic/viz thickness).
    float maxRefr = max(rt.distances.y, 1.0);
#else
    float maxRefr = 300.0;
#endif
    // === RAY-BUDGET DECISIONS (perf: fewer inline rays, exactness kept) ===
    // Early Fresnel estimate (identical formula to the composite below) for
    // Fresnel-based single-ray selection. reflMixEst is the lobe weight the
    // composite will use (uniform flag ? reflectionStrength : Fresnel mix).
    // Rule: a budget skip (xor/checker) applies ONLY when the async pipeline
    // covers that lobe's pixel — an uncovered skip would replace an exact
    // hit with sky. Refraction is pipe-first (pre-existing): its inline ray
    // runs only for invalid pipe texels, so xor/checker act on the
    // reflection inline ray (the dominant inline cost); refraction keeps its
    // pipe-or-trace behavior plus the negligible-lobe gate.
    float fresnelEarlyCurve = pow(1.0 - clamp(dot(viewDir, normal), 0.0, 1.0),
                                 clamp(fresnelPower, 1.0, 8.0));
    float fresnelEarly = clamp(0.02 + 0.98 * fresnelEarlyCurve, 0.0, 1.0);
    bool uniformEarly = wp.reserved2.w > 0.5;
    float reflMixEst = uniformEarly
        ? clamp(reflectionStrength, 0.0, 1.0)
        : mix(fresnelEarly, 1.0, clamp(reflectionStrength, 0.0, 1.0));
    // Blue-noise-ish stochastic selector (FragCoord hash): single-ray mode
    // traces reflection XOR refraction with probability = reflMixEst
    // (Schlick-weighted) to save a ray on hits. A missed first ray always
    // recovers the other lobe (bottom via zero-cost raster recovery, mirror
    // via a real ray), so no pixel ends with two empty lobes ("just
    // tinted"). Reference mode (any rt.debug view except 59-61, the
    // diagnostic masks themselves) forces dual-trace + full-rate.
    float hash01 = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    bool waterRefMode = false;
    bool waterSingleRay = false;
    bool waterChecker = false;
    float waterContribMin = 0.02;
#ifdef RT_ENABLED
    // Views 59-61 visualize the budgeted behavior itself, so they must not
    // force reference (otherwise the counters could never show the live cut).
    waterRefMode = (rt.debug.x != 0.0)
        && (rt.debug.x < 58.5 || rt.debug.x > 61.5);
    waterSingleRay = (rt.rayParams.z > 0.5) && !waterRefMode;
    waterContribMin = clamp(rt.rayParams.y, 0.0, 1.0);
    waterChecker = (rt.rayParams.x > 0.5) && !waterRefMode
        && ((int(gl_FragCoord.x) + int(gl_FragCoord.y)) & 1) == 1;
#endif
    bool wantReflInline = true;
    bool wantRefrInline = true;
    if (waterSingleRay && enableReflection && enableRefraction && !captureMode) {
        bool pickRefl = hash01 < clamp(reflMixEst, 0.0, 1.0);
        wantReflInline = pickRefl;
        wantRefrInline = !pickRefl;
    }
    // Per-lobe contribution gates (negligible lobes skip the inline ray and
    // keep the cheap fallback): reflection matters by reflMixEst, refraction
    // by 1-reflMixEst. Disabled lobes report 0 contribution upstream.
    float reflContribEst = enableReflection
        ? (enableRefraction ? reflMixEst : 1.0) : 0.0;
    float refrContribEst = enableRefraction
        ? (enableReflection ? (1.0 - reflMixEst) : 1.0) : 0.0;
    // Thick-water policy (user): where the raster back-face measures a real
    // column (genuinely thick volumes — wavy/displaced water reports meters
    // here), refraction AND reflection always trace: force dual-trace +
    // full-rate by clearing the xor pick and the checkerboard. Flat/calm
    // water (no raster column) keeps the budgeted path above. Contrib gates
    // stay (a negligible lobe is invisible traced or not); reference mode
    // already forces full, so this is a no-op there.
    if (hasValidBackFace) {
        wantReflInline = true;
        wantRefrInline = true;
        waterChecker = false;
    }
    // Checkerboard pattern flag; reflection additionally keeps full rate
    // for strong mirrors (reflMixEst > 0.7) so grazing water never dithers.
    // (Refraction is pipe-first, so its inline fallback is not checker-gated:
    // an invalid pipe texel has no cover and must trace.)
    bool reflCheckerSkip = waterChecker && (reflMixEst <= 0.7);
    // Ray-mask for debug view 59 (pixel-ratio counter): per lobe 0=disabled,
    // 1=sky fallback, 2=pipe hit, 3=inline traced, 4=skipped by budget gate.
    float refrMask = 0.0;
    float reflMaskDbg = 0.0;
    // Bottom-lobe content flags for miss-recovery (reflection recovers when
    // the bottom ended content-free): pipe cover, real inline hit
    // (a < RT_DEEP_WATER, i.e. not the miss marker), raster recovery.
    bool pipeRefrValid = false;
    bool refrInlineHitReal = false;
    // Thickness-source id for debug view 60 (which branch set the water
    // column): 0=raster back-face MISSING / no RT, 1=RT inline hit length,
    // 2=miss continuity (raster bottom measured along the Snell ray),
    // 3=miss with no raster bottom->thin, 4=(retired),
    // 5=pipeline refraction output, 6=raster BACK-FACE stood (overrides RT).
    float depthSource = 0.0;
    // Hoisted Snell direction (normalized): the miss-continuity path in the
    // deep branch measures the raster bottom along THIS ray so hit pixels
    // (hitT along refrRay) and miss pixels share one world-space depth.
    vec3 refrRayW = vec3(0.0);
    bool haveRefrRayW = false;
    if (enableRefraction) {
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
        refrRayW = refrRay;
        haveRefrRayW = true;
        bool refrResolved = false;
#ifdef RT_ENABLED
        if (usePipe && rt.toggles.y > 0.5) {
            // Half-res single-mip pipeline output: explicit LOD 0 (also safe
            // under the per-fragment pipe-validity branch).
            vec4 pipeRefr = textureLod(rtRefractTex, screenUV, 0.0);
            if (pipeRefr.a >= 0.0) {
                sceneColor = pipeRefr.rgb;
                rtThickness = pipeRefr.a;
                refrResolved = true;
                refrMask = 2.0;
                depthSource = 5.0;
                pipeRefrValid = true;
            }
        }
        // Inline fallback: runs only when the pipe cannot cover this texel
        // (invalid). The xor/checker skips are pipe-gated by construction —
        // with no pipe cover the exact bottom must be traced, never replaced
        // with sky. Only the negligible-lobe gate still applies (an
        // invisible lobe stays skipped). Never black.
        bool refrBudgetSkip = (refrContribEst < waterContribMin);
        if (waterRefMode) refrBudgetSkip = false;
        if (!refrResolved && rtReady && rt.toggles.y > 0.5 && !refrBudgetSkip) {
            float refrSceneTMax = hasValidBackFace
                ? (backFaceThickness * 1.5 + 2.0)
                : min(maxRefr, max(refrThickCap * 3.0, 8.0));
            // Origin on the TRUE displaced surface (fragPosWorld): the
            // filtered pass skips water candidates, so the tMin self-guard
            // only needs to clear coplanar touch — and the reported length
            // is the visible water column, not base-to-bottom.
            vec4 hit = rtTraceWater(fragPosWorld, refrRay, refrSceneTMax, true, refrThickCap);
            sceneColor = hit.rgb;
            // a >= 0 always from rtTraceWater: capped path length on hit, or
            // RT_DEEP_WATER marker on miss (deep, unresolved water). Only a
            // real triangle hit counts as bottom content for miss-recovery.
            rtThickness = hit.a;
            rtThickFromScene = true;
            refrResolved = true;
            refrMask = 3.0;
            depthSource = 1.0;
            refrInlineHitReal = (hit.a < RT_DEEP_WATER);
        } else if (!refrResolved && rtReady && rt.toggles.y > 0.5 && refrBudgetSkip) {
            refrMask = 4.0;
        }
#endif
#ifdef RT_ENABLED
        // Waterline overshoot clamp: a ray hit can report distant bottom at
        // true-zero-depth pixels when the ray slips past the waterline lip
        // through the bottom-mesh edge and lands meters out (the layer cap
        // does not catch this — 6 m of column at the waterline still prints
        // an opaque dark strip). The raster bottom directly behind this
        // pixel bounds it: a reported column far deeper than the raster
        // column is an overshoot, never a measurement. The 2x disagreement
        // guard keeps this from touching agreeing signals (warp wobble,
        // oblique paths) — it only fires on gross lip overshoots. Marker
        // (deep) values are excluded: they belong to miss continuity below.
        if (refrResolved && (pipeRefrValid || refrInlineHitReal)
            && rtThickness >= 0.0 && rtThickness < RT_DEEP_WATER) {
            float sD = textureLod(solidSceneDepthTex, screenUV, 0.0).r;
            if (sD < 1.0) {
                float gR = linearizeDepth(sD) - frontFaceLinear;
                if (gR >= 0.0 && gR < rtThickness * 0.5) {
                    rtThickness = gR;
                    rtThickFromScene = true; // raster-exact, skip dither
                    depthSource = 2.0;
                }
            }
        }
#endif
        if (!refrResolved) {
            // Skipped-ray raster recovery (miss-recovery without a second
            // ray): the budget gates saved a ray for this lobe, but the
            // pixel still needs real bottom content — sample the raster
            // bottom at the Snell landing (zero rays, same helper as miss
            // continuity, same displaced-surface entry as the ray). Only
            // when the lobe matters and RT could have traced (same
            // readiness as the inline path); otherwise the sky fallback
            // stands (non-RT build, RT off, negligible lobe).
            bool refrRecovered = false;
#ifdef RT_ENABLED
            if (rtReady && rt.toggles.y > 0.5 && haveRefrRayW
                && refrContribEst >= waterContribMin) {
                vec4 rb = rtRasterBottom(fragPosWorld, refrRayW, screenUV);
                if (rb.a >= 0.0) {
                    sceneColor = rb.rgb;
                    rtThickness = min(rb.a, max(refrThickCap, 0.0));
                    rtThickFromScene = true;
                    refrRecovered = true;
                    refrMask = 4.0; // ray still skipped — ratio honest
                    depthSource = 2.0;
                }
            }
#endif
            if (!refrRecovered) {
                // Sky fallback (also the non-RT path): refracted sky through
                // the surface, no thickness (back-face raster method covers
                // thickness). Explicit LOD: per-fragment control flow.
                sceneColor = textureLod(skyEquirectTex, waterDirToEquirectUV(refrRay), 0.0).rgb;
                if (refrMask == 0.0) refrMask = 1.0;
            }
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
    // Deep-water handling: a refraction ray that misses carries the
    // RT_DEEP_WATER marker, but depth is never defaulted from it. The miss
    // is resolved against the same-frame raster bottom (continuity path in
    // the deep branch): raster color + verticalized gap, capped like a hit.
    // Far bottoms saturate the shared Beer-Lambert/tint/alpha ramps to the
    // deep look by themselves, so there is no thin/deep threshold and no
    // waterline seam. Absorption comes from the per-layer water params
    // (single source of truth); rt.* carries only ray-technical state
    // (toggles, distances, coarse size).
    float absorbScale = absorbScaleBase;
    bool rtDeepMiss = false;
    // Water tint colors from UBO (declared here — the deep-miss path needs
    // them, and the tint composition below reuses them).
    vec3 deepTint = wp.deepColor.rgb;
    vec3 shallowTint = wp.shallowColor.rgb;
#ifdef RT_ENABLED
    // Hash-dither resolved RT hit lengths (±0.3 m). Per-chunk flat box tops
    // quantize the true depth into steps; undithered, those steps print as
    // terrace bands. Dithered they degrade to grain, which reads as water
    // noise. Miss marker and "no RT" (-1) are never touched. Skipped for
    // exact scene-triangle hits (rtThickFromScene): dithering those would
    // reintroduce the noise the exact geometry just removed.
    if (rtReady && rt.toggles.z > 0.5 && !rtThickFromScene && rtThickness >= 0.0 && rtThickness < RT_DEEP_WATER) {
        float h = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        rtThickness = max(rtThickness + (h - 0.5) * 0.6, 0.0);
    }
    if (rtReady && rt.toggles.z > 0.5 && rtThickness >= 0.0) {
        rtDeepMiss = (rtThickness >= RT_DEEP_WATER);
        if (rtDeepMiss) {
            // Miss continuity (no thresholds, hence no seams): a refraction
            // ray that misses is resolved against the same-frame raster
            // bottom — at the point where THIS Snell ray would land on it,
            // not straight below the pixel. Hit pixels report hitT along
            // refrRay and shade the raster bottom at their landing point;
            // here the raster bottom is unprojected to world (clip->world
            // is the correct invViewProjection direction), intersected with
            // refrRay (horizontal-plane approx through the unprojected
            // point), and sampled at THAT landing point with the same
            // depth-tolerance lookup the hit path uses. Same ray, same
            // origin, same sample point, same cap: color and thickness are
            // continuous across the hit/miss border by construction. Far
            // bottoms still read as deep water by themselves through the
            // shared ramp saturation below (Beer-Lambert floor, tint/alpha
            // saturation, aerial fade) — no deepTint substitution, no 300 m
            // default. A 300 m column is never defaulted: with no usable
            // raster bottom (clear depth) the miss stays thin (far pixels
            // are backstopped by the aerial fade). Covers inline-ray and
            // pipeline misses alike (both arrive here via rtThickness).
            // Explicit LOD: per-fragment control flow.
            // Shared raster-bottom helper (same Snell landing + lookup as
            // ray-less skipped pixels above): hits, misses and skips all
            // shade one consistent bottom. Entry matches the ray origin
            // (displaced surface). Covers inline-ray and pipeline misses
            // alike (both arrive here via rtThickness).
            vec4 rb = haveRefrRayW
                ? rtRasterBottom(fragPosWorld, refrRayW, screenUV)
                : vec4(0.0, 0.0, 0.0, -1.0);
            if (rb.a >= 0.0) {
                sceneColor = rb.rgb;
                rtDeepMiss = false;
                rtThickness = min(rb.a, max(refrThickCap, 0.0));
                rtThickFromScene = true;
                depthSource = 2.0;
            } else {
                // No raster bottom to stand on: thin slip, not an abyss
                // (far pixels are backstopped by the aerial fade below).
                rtDeepMiss = false;
                rtThickness = 0.0;
                rtThickFromScene = true;
                depthSource = 3.0;
            }
        }
        // NOTE: rtDeepMiss is always resolved above (raster continuity or
        // thin slip), so no legacy deepTint/maxRefr substitution remains;
        // depth grades continuously through the shared path below.
        if (!hasValidBackFace) {
            // RT hit length is the only depth signal available.
            // (Already capped at refrThickCap upstream: rgen + rtTraceWater.)
            waterThickness = rtThickness * absorbScaleBase;
        } else {
            // Raster back-face thickness stands, bounded by the raster solid
            // bottom directly behind this pixel (same view ray, so directly
            // comparable): the water column can never run deeper than the
            // bottom behind it. On wavy/grazing shores the back-face pass
            // reports the next swell underside meters out while the true
            // column is centimeters — min() keeps the true one. Occluders
            // (gap < 0) and missing bottoms (clear depth) leave the back
            // face standing; the composite occludes those pixels anyway.
            // Final clamp to the per-layer hit cap keeps unbounded deep
            // columns from attenuating to black.
            float sDB = textureLod(solidSceneDepthTex, screenUV, 0.0).r;
            if (sDB < 1.0) {
                float gRB = linearizeDepth(sDB) - frontFaceLinear;
                if (gRB >= 0.0) {
                    waterThickness = min(waterThickness, gRB);
                }
            }
            waterThickness = min(waterThickness, refrThickCap);
            depthSource = 6.0; // DIAG-60: raster back-face stood (overrides RT)
        }
    }
#else
    // Non-RT build: same raster bound on the back-face column (the band is
    // identical there — it never involved ray tracing).
    if (hasValidBackFace) {
        float sDB = textureLod(solidSceneDepthTex, screenUV, 0.0).r;
        if (sDB < 1.0) {
            float gRB = linearizeDepth(sDB) - frontFaceLinear;
            if (gRB >= 0.0) {
                waterThickness = min(waterThickness, gRB);
            }
        }
        waterThickness = min(waterThickness, refrThickCap);
        depthSource = 6.0;
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
    
    // Depth-based color fade (deeper = more tinted). Uses the best available
    // depth signal: the smooth raster backface diff where a real volume was
    // measured, else the RT thickness fallback — otherwise Water Tint / Depth
    // Falloff could never affect flat heightfield water (no backface, so
    // depthDiff is 0 there). Both signals collapse to 0 at the shoreline, so
    // the fade (like alpha) vanishes at the waterline. RT steps arrive
    // pre-dithered as grain, the same tradeoff Beer-Lambert already accepts.
    float depthFalloff = wp.waveParams.w;
    if (depthFalloff <= 0.0) depthFalloff = 0.02;
    float tintDepth = max(depthDiff, waterThickness);
    float depthFade = 1.0 - exp(-tintDepth * depthFalloff);
    
    // === FRESNEL EFFECT ===
    // Schlick approximation anchored at the physical air->water base
    // reflectance F0 = 0.02: looking straight down reflects ~2% of the
    // environment, grazing angles approach a full mirror. fresnelPower
    // (default 5 = standard Schlick) shapes the transition curve.
    // NOTE: dot() is clamped ABOVE as well: two normalized vectors can dot
    // to 1.0000001 in floating point, and pow(negative, x) is undefined
    // (NaN on most drivers) — which used to black out calm top-down water
    // where dot(viewDir, normal) rounds to exactly ~1.0.
    float fresnelCurve = pow(1.0 - clamp(dot(viewDir, normal), 0.0, 1.0), clamp(fresnelPower, 1.0, 8.0));
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
    // Precedence (see HYBRID RT STATE): the inline ray (full-res exact chunk
    // triangles) wins whenever it runs — it resolves mirror positions the
    // coarse proxy pipeline cannot. The pipeline output covers budget-skipped
    // pixels. Miss-recovery (below): when the bottom lobe ended content-free
    // the mirror traces even off-pick, so no pixel ends with two empty lobes
    // ("just tinted"). Sky covers the rest. No 360 cubemap.
    //
    // Miss-recovery trigger: the bottom lobe has no pipe cover, no real
    // inline hit, and no meaningful raster recovery — while refraction is
    // enabled. (Refraction disabled means bottom-sky by user choice, not a
    // miss, so the mirror keeps its own pick/gates.)
    bool bottomEmpty = enableRefraction && !pipeRefrValid && !refrInlineHitReal
        && !(depthSource > 1.5 && depthSource < 2.5 && rtThickness > 0.001);
    vec3 reflectDir = reflect(-viewDir, normal);

    vec3 skyColor = vec3(0.0);
    bool reflResolved = false;
#ifdef RT_ENABLED
    // Sample the pipe first (one cheap tap) so the budget gate below can
    // tell covered pixels (pipe valid) from uncovered ones. The pipe color
    // is only USED if the inline ray does not run or misses.
    vec3 pipeReflCol = vec3(0.0);
    bool pipeReflValid = false;
    if (usePipe && rt.toggles.x > 0.5) {
        vec4 pipeRefl = textureLod(rtReflectTex, screenUV, 0.0);
        if (pipeRefl.a > 0.5) {
            pipeReflCol = pipeRefl.rgb;
            pipeReflValid = true;
        }
    }
    // Budget gate: single-ray xor + checkerboard half-rate (strong mirrors
    // exempt) + negligible-lobe skip. Skips apply ONLY when the pipe covers
    // the pixel — an uncovered skip would replace an exact hit with sky, a
    // visible regression. Covered skips reuse the pipe; uncovered pixels
    // always trace (when ready/toggled), so exact reflections are never
    // sacrificed for the budget. Miss-recovery (bottomEmpty) traces
    // regardless of xor/checker — a guarantee, not a saving — but keeps the
    // contribution gate (negligible lobes stay skipped) and never overrides
    // reference mode. Never black.
    bool reflXorSkip = !wantReflInline && pipeReflValid;
    bool reflCheckerEff = reflCheckerSkip && pipeReflValid;
    bool reflBudgetSkip = reflXorSkip || reflCheckerEff
        || (reflContribEst < waterContribMin);
    if (waterRefMode) reflBudgetSkip = false;
    bool reflRecover = bottomEmpty && enableReflection && !waterRefMode
        && reflContribEst >= waterContribMin;
    bool reflDidTrace = false;
    vec4 reflHit = vec4(0.0);
    if ((!reflBudgetSkip || reflRecover) && rtReady && rt.toggles.x > 0.5) {
        // Origin on the UNDISPLACED base surface, biased along the base
        // normal (mirrors main.frag): the BLAS holds the undisplaced CPU
        // mesh, so tracing from the displaced (tessellated wave) surface
        // starts the ray off-BLAS-surface. Wave troughs sit BELOW the BLAS
        // plane, and their upward/scattered rays then immediately self-hit
        // the neighboring undisplaced water triangles from underneath
        // (backfaces are not culled from below) — shading as flat water
        // tint and printing chunk-bordered blobs whose borders move with
        // the per-chunk tessellation LOD. The base plane + bias restores
        // the self-guard the tMin was designed for. Direction keeps the
        // rippled normal (sparkle detail is unaffected).
        vec3 reflBaseN = normalize(fragBaseNormal);
        if (dot(reflBaseN, viewDir) < 0.0) reflBaseN = -reflBaseN;
        reflHit = rtTraceWater(fragBasePos.xyz + reflBaseN * 0.05, normalize(reflectDir), RT_NO_LIMIT, false, 0.0);
        reflDidTrace = true;
        if (reflHit.a > 0.5) {
            skyColor = reflHit.rgb;
            reflResolved = true;
            reflMaskDbg = 3.0;
        }
        // Miss: fall through to pipe/sky below (the trace's own sky equals
        // the fallback sky; the pipe may still hold a proxy hit the exact
        // triangles missed).
    }
    if (!reflResolved && pipeReflValid) {
        skyColor = pipeReflCol;
        reflResolved = true;
        reflMaskDbg = 2.0;
    }
    if (!reflResolved && reflDidTrace) {
        // Traced, missed, no pipe cover: sky (same value the trace saw).
        skyColor = reflHit.rgb;
        reflResolved = true;
        reflMaskDbg = 3.0;
    } else if (!reflResolved && rtReady && rt.toggles.x > 0.5 && reflBudgetSkip) {
        reflMaskDbg = 4.0;
    }
#endif
    if (!reflResolved) {
        // Explicit LOD: per-fragment fallback branch (see refraction above).
        skyColor = textureLod(skyEquirectTex, waterDirToEquirectUV(normalize(reflectDir)), 0.0).rgb;
        if (reflMaskDbg == 0.0) reflMaskDbg = 1.0;
    }

    // (Screen-space refinement removed: the inline trace above hits the real
    // chunk triangles directly, so a depth-march pass is redundant.)

    // === AERIAL DETAIL FADE (§10/§11) ===
    // Refraction/thickness fade: proxy boxes are per-chunk flats, so beyond
    // the near field their tops and hit/miss classification imprint box-shaped
    // steps onto refraction color and thickness — and every such step is a
    // potential razor line (LOD frontiers are straight, full-width and
    // camera-following). Distance is continuous, so fading by distance cannot
    // create edges by construction; it only removes them. Near field (<120 m,
    // where boxes are tightest) keeps pixel-identical RT detail; far field
    // converges to deep tint, i.e. honest aerial perspective.
    // Reflection is NOT faded: a mirror must keep reflecting the scenery no
    // matter how far the water pixel is from the camera.
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

    // Tint color ramps shallow → deep with measured water thickness around the
    // per-layer reference distance (Caustic Depth Scale doubles as the depth
    // reference here). depthFade above already forces the blend to 0 at the
    // shoreline, so shallowTint never paints the waterline.
    float tintDepthScale = max(wp.causticParams.w, 0.0001);
    float volumeFactor = 1.0 - exp(-waterThickness / tintDepthScale);

    // Water tint color transitions from shallow → deep depending on volume.
    vec3 waterTintColor = mix(shallowTint, deepTint, volumeFactor);

// Blend scene color with water tint: depthFade (Depth Falloff over the best
// depth signal) sets the amount, Water Tint scales it, Transparency caps it
// (1 = crystal clear keeps the refracted bottom, 0 = fully tintable).
    float tintMax = clamp(1.0 - transparency, 0.0, 1.0);
    float tintBlend = clamp(depthFade * waterTint, 0.0, tintMax);
    vec3 refractedColor = mix(sceneColor, waterTintColor, tintBlend);
    
    // Mix refracted color with reflection. By default, use Fresnel weighting
    // to increase reflection at grazing angles. If `uniformReflection` is
    // enabled, use `reflectionStrength` directly so reflection appears
    // across all pixels uniformly (useful for debugging/stylized look).
    vec3 waterColor;
    // Mirror presence for the translucency below: the Fresnel surface mirror
    // lives at the interface, not in the volume.
    float mirrorPresence = 0.0;
    if (captureMode) {
        // 360 capture: no env feedback. Use the base tint so water is not
        // black (refraction/reflection are disabled, leaving sceneColor zero).
        waterColor = waterTintColor;
    } else if (enableReflection) {
        // Reflection Strength is the mirror amount: 1 = full mirror at every
        // angle (polished spheres), 0 = physical Fresnel-only water. Fresnel
        // still shapes partial strengths so low values keep the grazing
        // falloff instead of popping a flat reflection over the whole surface.
        float reflMix = uniformReflection
            ? reflectionStrength
            : mix(fresnel, 1.0, clamp(reflectionStrength, 0.0, 1.0));
        waterColor = mix(refractedColor, skyColor, reflMix);
        mirrorPresence = clamp(reflMix, 0.0, 1.0);
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
    bool causticDebugMode = (dbgMode >= 42 && dbgMode <= 45);
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
    // Shoreline fade: zero-depth water is no water. When a real depth signal
    // exists (raster volume or RT hit), force fully transparent AT the
    // waterline so the shore shows the bottom with no water color, ramping to
    // the normal depth-driven alpha over shoreFadeDepth meters. Without any
    // depth signal (non-RT flat water reports 0 everywhere) the fade is
    // skipped so the water stays visible via the transparency floor above.
    float thicknessFrac = clamp(thicknessForAlpha / 3.0, 0.0, 1.0); // ~3 m -> opaque
    float alpha = mix(1.0, thicknessFrac, clamp(transparency, 0.0, 1.0));
    // The Fresnel surface mirror is not volume translucency: a strong mirror
    // (grazing angles) must composite even where the water is thin, or
    // shallows and puddles lose their sky entirely (real puddles mirror!).
    // Top-down views are unaffected (mirrorPresence ≈ 0 there).
    alpha = max(alpha, mirrorPresence);
    float shoreWidth = max(wp.refractionParams.z, 0.0);
    if (thicknessForAlpha > 1e-4 && shoreWidth > 1e-6) {
        alpha *= smoothstep(0.0, shoreWidth, thicknessForAlpha);
    }
    if (captureMode) alpha = 1.0;
    outColor = vec4(waterColor, alpha);

    // Debug: visual displacement color when debug mode set to 38 ("Water Displacement")
    if (dbgMode == 38) {
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

    // Debug mode 39: raw sky equirect (reflection of view dir) — verifies the
    // water pass reaches the sky fallback it uses for RT misses.
    if (dbgMode == 39) {
        vec3 sc = texture(skyEquirectTex,
            waterDirToEquirectUV(normalize(reflect(-viewDir, normal)))).rgb;
        outColor = vec4(sc, 1.0);
    }

    // Debug mode 35: screen UV — verifies correct clip → UV conversion.
    if (dbgMode == 35) {
        outColor = vec4(screenUV, 0.0, 1.0);
    }


   // Debug mode 36: water noise
    if (int(ubo.debugParams.x) == 36) {
        outColor = vec4(refractionNoise, 0.5 + 0.5 * (refractionNoise.x - refractionNoise.y), 1.0);
    }

    // Debug mode 37: final displaced normal used by shading.
    if (int(ubo.debugParams.x) == 37) {
        vec3 n = normalize(normal);
        outColor = vec4(n * 0.5 + 0.5, 1.0);
    }

    // --- Reflection sampling debug helpers ---
    // Use the global debug mode (ubo.debugParams.x) to visualize reflection
    // computation steps and RT sampling. Helpful to diagnose orientation.
    if (dbgMode == 40) {
        // Visualize reflection vector (packed to [0,1])
        vec3 vis = reflectDir * 0.5 + 0.5;
        outColor = vec4(vis, 1.0);
    }
    if (dbgMode == 41) {
        // Show RT/sky reflection color actually used by shading
        outColor = vec4(skyColor, 1.0);
    }

    if (dbgMode == 42) {
        vec3 maps = vec3(clamp(caustFront, 0.0, 1.0), clamp(caustBack, 0.0, 1.0), clamp(mix(caustFront, caustBack, depthInfluence), 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 43) {
        vec3 maps = vec3(clamp(lineFrontRaw, 0.0, 1.0), clamp(lineBackRaw, 0.0, 1.0), clamp(lineCombined, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 44) {
        vec3 maps = vec3(clamp(cloudFinal, 0.0, 1.0), clamp(lineFinal, 0.0, 1.0), clamp(caustRaw, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 45) {
        outColor = vec4(vec3(clamp(caustic, 0.0, 1.0)), 1.0);
    }

    // --- Water thickness / depth debug modes (46..49) ---
    // 43: Back-face raw depth (texture sample)
    if (dbgMode == 46) {
        outColor = vec4(vec3(backFaceDepthRaw), 1.0);
    }
    // 44: Front-face linear depth (normalized to [0,1])
    if (dbgMode == 47) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((frontFaceLinear - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 45: Back-face linear depth (normalized to [0,1])
    if (dbgMode == 48) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((backFaceLinear - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 46: Water thickness (normalized by per-layer caustic depth scale or 1.0)
    // (Ancient modes 46/47 showed solid scene depth, which water no longer
    // samples; long removed — 46..49 are water depth/thickness now.)
    if (dbgMode == 49) {
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
    if (dbgMode == 59) {
        // Ray-query pixel ratio (per lobe): R = reflection inline traced,
        // G = refraction inline traced, B = pipeline hit. Budget-skipped
        // pixels (checker/single-ray/contrib) stay dark; sky fallback is
        // near-black. The lit-pixel fraction must drop >=50% vs full-rate.
        // Masks: 0=disabled, 1=sky, 2=pipe, 3=inline, 4=budget-skipped.
        vec3 maskCol = vec3(0.0);
        if (reflMaskDbg > 2.5 && reflMaskDbg < 3.5) maskCol.r = 1.0;
        else if (reflMaskDbg > 1.5 && reflMaskDbg < 2.5) maskCol.b += 0.5;
        if (refrMask > 2.5 && refrMask < 3.5) maskCol.g = 1.0;
        else if (refrMask > 1.5 && refrMask < 2.5) maskCol.b += 0.5;
        outColor = vec4(maskCol, 1.0);
    }
    if (dbgMode == 60) {
        // Water-column source: which branch set this pixel's thickness.
        // Read with RT on (toggles + pipeline as in the failing view).
        // grey=raster back-face/no-RT, green=RT inline hit length,
        // cyan=miss continuity (raster bottom), magenta=miss with no
        // raster bottom->thin, blue=pipeline refraction output.
        // (Red/proven-deep is retired: depth now grades continuously.)
        vec3 dc = vec3(0.25);
        if (depthSource > 0.5 && depthSource < 1.5) dc = vec3(0.0, 1.0, 0.0);
        else if (depthSource < 2.5 && depthSource > 1.5) dc = vec3(0.0, 1.0, 1.0);
        else if (depthSource < 3.5 && depthSource > 2.5) dc = vec3(1.0, 0.0, 1.0);
        else if (depthSource > 4.5) dc = vec3(0.0, 0.0, 1.0);
        outColor = vec4(dc, 1.0);
    }
    if (dbgMode == 61) {
        // Water brush/layer id per pixel (golden-ratio hue). MAGENTA =
        // out of the waterParams SSBO range: those pixels previously read
        // past the allocation (undefined values → black opaque water with
        // RT disabled) and now fall back to layer 0.
        int nLB = max(waterParams.length(), 1);
        vec3 bidCol;
        if (fragBrushIndex < 0 || fragBrushIndex >= nLB) {
            bidCol = vec3(1.0, 0.0, 1.0);
        } else {
            float hh = fract(float(fragBrushIndex) * 0.61803398875);
            bidCol = clamp(0.5 + 0.5 * cos(6.2831853 * (hh + vec3(0.0, 0.33, 0.67))), 0.0, 1.0);
        }
        outColor = vec4(bidCol, 1.0);
    }


    // Final outputs: only write the composited water color (RGBA)
    // Normal/mask outputs removed — they are no longer produced by this pass.


}
