#include "water_render_view.glsl"
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
                    const float nearB = ubo.nearPlane;
                    const float farB = ubo.farPlane;
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
// instances (exact chunk triangles): solids via RT_SCENE_INSTANCE and the
// water mesh via RT_SCENE_WATER_INSTANCE (geomInfo.w marks the partition).
// refraction=true: downward Snell ray from the TRUE displaced surface with
//   no culling, resolving the lake bottom against SOLID triangles only. The
//   water mesh in the BLAS is the UNDISPLACED base surface: treating its
//   triangles as a water/air boundary reports phantom columns at every wave
//   crest (tiny thickness → transparent "ice" patches) and flips winner per
//   base triangle (hard cuts between the water look and the bottom). A miss
//   is deep water. One OPAQUE early-out query, so cost tracks the nearest
//   solid hit instead of every candidate along the ray. a = underwater path
//   length (capped at thickCap) or RT_DEEP_WATER when no solid exists.
// refraction=false: mirror ray resolving the reflected scene the staged way
//   (forward culled, reverse culled, forward unculled); it DOES see the
//   water mesh (other water bodies must appear in mirrors); a = 1 on a hit,
//   0 on a miss.
// Macro shadows stay CSM-owned: hits get ambient + sun diffuse only.
vec4 rtTraceWater(vec3 origin, vec3 dir, float tMax, bool refraction, float thickCap, float waterMinHit) {
    // Refraction rays start AT the displaced water surface and go down.
    // tMin stays tiny (1 cm): the solid query can never see the water mesh
    // (separate instance/mask), so coplanar touch is the only self-guard
    // needed, and shoreline shallows (<5 cm deep) still hit the lake bottom
    // underneath instead of missing to deep-water tint. Reflection keeps the
    // 5 cm tMin (self-hit guard, origin biased above the surface at the
    // call site).
    float tMin = refraction ? 0.01 : 0.05;
    // RT per-op profiling op id (no-op without RT_PROFILE): this helper serves
    // both water lobes, so every query it issues is attributed to its lobe.
    const uint rtProfOp = refraction ? RT_PROFILE_OP_WATER_REFRACTION
                                     : RT_PROFILE_OP_WATER_REFLECTION;
    // Hit selection: refraction runs one OPAQUE early-out query over solids;
    // reflection keeps the staged chain. The BLAS holds undisplaced geometry
    // while the raster shows tessellated/displaced surfaces (waves up to
    // waveAmplitude off the base plane), so refraction uses NO cull flags:
    // opacity and face culling are independent, and not culling keeps
    // flipped (displaced) faces visible while hardware still early-outs at
    // the nearest accepted hit. The reported length IS the visible water
    // column — no correction needed.
    //
    // The shading block consumes the outer prim/lo/hitBary that the WINNING
    // query stored alongside hitT — indices, distance and barycentrics are
    // provably from the same query.
    bool haveHit = false;
    float hitT = 0.0;
    uint prim = 0u;
    uint lo = 0u;
    vec2 hitBary = vec2(0.0);
    if (refraction) {
        // Nearest SOLID triangle: opaque (hardware early-out) and no cull
        // flags, so displaced/flipped faces still register.
        RT_PROF_BEGIN(rtProfRqS, rtProfOp);
        rayQueryEXT rqS;
        rayQueryInitializeEXT(rqS, rtTlas, gl_RayFlagsOpaqueEXT, RT_RAY_MASK_SCENE,
            origin, tMin, dir, tMax);
        while (rayQueryProceedEXT(rqS)) {}
        RT_PROF_END(rtProfRqS);
        if (rayQueryGetIntersectionTypeEXT(rqS, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
            rayQueryGetIntersectionInstanceCustomIndexEXT(rqS, true) == RT_SCENE_INSTANCE) {
            uint loC = rtSceneGeomIndex(RT_SCENE_INSTANCE,
                uint(rayQueryGetIntersectionGeometryIndexEXT(rqS, true)));
            // Post-commit classification: the mask already selects the solid
            // partition; verify geomInfo.w agrees before committing.
            if (rtSceneGeomInfo[loC].w == 0u) {
                RT_PROF_HIT(rtProfOp);
                hitT = rayQueryGetIntersectionTEXT(rqS, true);
                prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rqS, true));
                lo = loC;
                hitBary = rayQueryGetIntersectionBarycentricsEXT(rqS, true);
                haveHit = true;
            }
        }
    } else {
    // Stage 1: forward ray (cull front faces). For reflection, a water-mesh
    // hit closer than waterMinHit is the reflector's own / nearby surface —
    // the BLAS water plane is flat and coplanar per chunk, so a rippled
    // normal that tips the mirror ray slightly down hits it immediately and
    // shades the water as if it reflected itself (flat sky where scenery is
    // due: the "reflection missing at the shore" report). Skip it; the
    // later stages can still find real scenery. Distant water (a lake across
    // the valley), i.e. beyond the threshold, stays a valid reflector.
    //
    // The staged chain traces the real scene-geometry instances only (exact
    // chunk triangles: solids + the real water mesh via geomInfo.w): proxy
    // boxes are excluded like in main.frag (their coarse flat tops imprint
    // the dominant material and stepped heights on hits). It stays opaque
    // and culls ray-front faces so only rasterizer-visible triangles report
    // (scene meshes wind CW-outward for the BACK+CW rasterizer; ray-front is
    // fixed CCW).
    RT_PROF_BEGIN(rtProfRq, rtProfOp);
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsOpaqueEXT |
        gl_RayFlagsCullFrontFacingTrianglesEXT,
        RT_RAY_MASK_SCENE | RT_RAY_MASK_SCENE_WATER,
        origin, tMin, dir, tMax);
    while (rayQueryProceedEXT(rq)) {}
    RT_PROF_END(rtProfRq);
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        // Forward miss: reflection misses stay plain sky (the refraction
        // deep marker never reaches this branch — it comes from the
        // split-instance pass's miss below and is resolved against the raster
        // bottom at the call site). The sky fetch is miss-only (Issue
        // L11); explicit LOD because this helper runs under per-fragment
        // control flow, where implicit-LOD texture() has undefined
        // derivatives.
        return vec4(textureLod(skyEquirectTex, rtDirToEquirectUV(normalize(dir)), 0.0).rgb, 0.0);
    }
    // Committed type != None is guaranteed by the early-out above; only
    // the scene-instance filter remains.
    {
        uint instC = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
        if (rtIsSceneInstance(instC)) {
            float tC = rayQueryGetIntersectionTEXT(rq, true);
            uint loC = rtSceneGeomIndex(instC, uint(rayQueryGetIntersectionGeometryIndexEXT(rq, true)));
            bool selfWater = (!refraction) && (rtSceneGeomInfo[loC].w != 0u) && (tC < waterMinHit);
            if (!selfWater) {
                hitT = tC;
                prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
                lo = loC;
                hitBary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
                haveHit = true;
            }
        }
    }
    if (!haveHit) {
        // Opposite-direction validation of the local column (reflection;
        // refraction uses the split-instance pass above). Water is allowed
        // beyond the same dead zone the forward tMin enforces (5.5 cm > 5 cm
        // bias + float noise): distant lake surfaces still mirror, but the
        // origin's own plane can never rediscover itself as "sky".
        float backSpan = min(tMax, max(thickCap * 2.0, 4.0));
        vec3 backOrigin = origin + dir * backSpan;
        RT_PROF_BEGIN(rtProfRq2, rtProfOp);
        rayQueryEXT rq2;
        rayQueryInitializeEXT(rq2, rtTlas, gl_RayFlagsOpaqueEXT |
            gl_RayFlagsCullFrontFacingTrianglesEXT,
            RT_RAY_MASK_SCENE | RT_RAY_MASK_SCENE_WATER,
            backOrigin, 0.01, -dir, backSpan);
        while (rayQueryProceedEXT(rq2)) {}
        RT_PROF_END(rtProfRq2);
        if (rayQueryGetIntersectionTypeEXT(rq2, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            uint instC2 = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq2, true));
            if (rtIsSceneInstance(instC2)) {
                uint lo2 = rtSceneGeomIndex(instC2, uint(rayQueryGetIntersectionGeometryIndexEXT(rq2, true)));
                float fwdT = backSpan - rayQueryGetIntersectionTEXT(rq2, true);
                bool waterOk = (rtSceneGeomInfo[lo2].w == 0u)
                    || (!refraction && fwdT >= waterMinHit);
                if (waterOk && fwdT >= 0.0) {
                    hitT = fwdT;
                    prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq2, true));
                    lo = lo2;
                    hitBary = rayQueryGetIntersectionBarycentricsEXT(rq2, true);
                    haveHit = true;
                }
            }
        }
        if (!haveHit) {
            // Final fallback: same forward ray with NO face culling. The two
            // culled rays above partition all faces on the near segment
            // (forward set + complement), so this ray's unique contribution
            // is flipped faces anywhere along the FULL segment — displaced
            // overhangs and flipped panels beyond the local backSpan that
            // neither cull set can report. Water rule mirrors stage 1:
            // reflection relies on its origin bias + tMin, so its mesh may
            // report like any hit.
            RT_PROF_BEGIN(rtProfRq3, rtProfOp);
            rayQueryEXT rq3;
            rayQueryInitializeEXT(rq3, rtTlas, gl_RayFlagsOpaqueEXT,
                RT_RAY_MASK_SCENE | RT_RAY_MASK_SCENE_WATER,
                origin, tMin, dir, tMax);
            while (rayQueryProceedEXT(rq3)) {}
            RT_PROF_END(rtProfRq3);
            if (rayQueryGetIntersectionTypeEXT(rq3, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                uint instC3 = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq3, true));
                if (rtIsSceneInstance(instC3)) {
                    uint lo3 = rtSceneGeomIndex(instC3, uint(rayQueryGetIntersectionGeometryIndexEXT(rq3, true)));
                    float tC3 = rayQueryGetIntersectionTEXT(rq3, true);
                    if (rtSceneGeomInfo[lo3].w == 0u || (!refraction && tC3 >= waterMinHit)) {
                        hitT = tC3;
                        prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq3, true));
                        lo = lo3;
                        hitBary = rayQueryGetIntersectionBarycentricsEXT(rq3, true);
                        haveHit = true;
                    }
                }
            }
        }
    } // end backward-else
    } // end refraction(split) / reflection(staged) split
    // One hit per call (not per staged query): rays - hits is the miss count.
    if (haveHit) RT_PROF_HIT(rtProfOp);
    if (!haveHit) {
        // Deep-water marker (refraction miss) or plain sky (reflection
        // triple miss). The deep marker is resolved against the raster
        // bottom at the call site, not here: this function has no view of
        // the solid depth target. The sky fetch is miss-only (Issue L11):
        // hit pixels never touch the equirect.
        vec3 sky = textureLod(skyEquirectTex, rtDirToEquirectUV(normalize(dir)), 0.0).rgb;
        return refraction ? vec4(sky, RT_DEEP_WATER) : vec4(sky, 0.0);
    }
    vec3 hitPos = origin + dir * hitT;
    // Underwater length cap (Beer-Lambert guard), shared by all hit returns
    // below (the old coarse-box feather toward deep is obsolete: exact
    // triangles need no terracing workarounds).
    float cap = max(thickCap, 0.0);

    { // haveHit guarantees a scene-geometry hit from the winning query.
        // Real triangle hit: shade with the owning chunk's data. prim/lo/
        // hitBary are the OUTER variables stored by the same query that
        // produced hitT (the split-instance refraction pass or the winning
        // reflection stage) — never re-read from another traversal, whose
        // committed triangle can disagree with the winner (the old shadowing
        // re-reads from the lead query paired one triangle's hitT/bary with
        // another triangle's indices → wrong textures in reflections).
        // The primitive index is LOCAL to the hit geometry
        // (GLSL_EXT_ray_query semantics); a cumulative primBase binary
        // search would map local indices onto the wrong chunk for every
        // geometry after the first.
        // Water-mesh hits are classified from geomInfo.w. The screen-space
        // bottom lookup below is valid for solid hits AND for a REFRACTED
        // water-mesh hit (the projected point's solid depth is the underwater
        // terrain), but never for a REFLECTED water hit: pasting the on-screen
        // terrain behind the lake made reflected water read as land. Water
        // vertices also live in the WATER pools, so the solid vertex/index
        // reads in the off-screen fallback are only valid for gi.w == 0.
        uvec4 gi = rtSceneGeomInfo[lo];
        // Below-waterline guard (reflection only): wave slopes can tip a
        // grazing mirror ray slightly below the surface, where it hits the
        // submerged lake bed instead of the scenery above the water. Shading
        // that hit paints the underwater terrain over the water — the
        // "far geometry has no reflection / dark far water" artifact. Treat
        // any SOLID hit clearly below the reflector's water level as a miss so
        // the mirror falls back to the sky (the ray direction is left
        // untouched, so genuine above-water terrain hits still reflect
        // normally). The reference is the UNDISPLACED base position
        // (fragBasePos.y = the local waterline), not the displaced origin:
        // in shallow water a wave trough can dip below the bed, which would
        // let a bed hit sit above the displaced surface and slip through.
        if (!refraction && gi.w == 0u && hitPos.y < fragBasePos.y - 0.05) {
            // Mirror ray dipped below the waterline: return the horizon sky as
            // a resolved reflection (a=0.6 > 0.5, source=guard for the
            // reflection-source debug view) so the fallback does not re-march
            // the solid depth and land on the bed again.
            vec3 skyDir = normalize(vec3(dir.x, max(dir.y, 0.0), dir.z));
            vec3 skyB = textureLod(skyEquirectTex, rtDirToEquirectUV(skyDir), 0.0).rgb;
            return vec4(skyB, 0.6);
        }
        if (gi.w == 0u || refraction) {
            // Screen-space color lookup. Sample this frame's solid render at
            // the reflected/refracted hit point so the mirror shows the
            // terrain exactly as it appears on screen (mixed ground cover,
            // shadows, detail) instead of the chunk's single dominant-material
            // sample (a flat dirt blob). The water pass runs after the solid
            // pass, so the targets are current.
            const float nearP = ubo.nearPlane;
            const float farP = ubo.farPlane;
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
        if (gi.w > 0u) {
            // Self-water guard for reflection: a wave-trough fragment's origin
            // sits BELOW the undisplaced (flat) water BLAS plane, so a grazing
            // upward mirror ray can hit the water mesh from below at a distance
            // far beyond waterMinHit. Shading that hit as water reflects the
            // ray about the surface, pointing the mirror chain DOWN into the
            // volume where it lands on the lake bed — the far-water "sky is not
            // reflected" artifact (steep near rays hit within waterMinHit and
            // were already skipped). A reflection ray travelling UP that hits
            // the water mesh is always the reflector's own surface: treat it as
            // a sky miss.
            if (!refraction && dir.y > 0.0) {
                vec3 skyW = textureLod(skyEquirectTex, rtDirToEquirectUV(normalize(dir)), 0.0).rgb;
                return vec4(skyW, 0.6); // source=guard (self-water)
            }
            // Reflected/refracted water: use the SAME composition as the
            // raster water surface (tint base + Fresnel/strength sky
            // reflection) so water inside a ray matches the water itself. The
            // underwater bottom is traced by rtResolveWaterHit's own Snell
            // ray, so the term is the water column, not a solid surface.
            //
            // Normal: the real water triangle's vertices live in the WATER
            // pools (not bindings 24/25), so no interpolated normal is
            // available here. Water is a heightfield: use a constant UP normal
            // (matching the solid path).
            vec3 hitN = vec3(0.0, 1.0, 0.0);
            int nWLL = max(waterParams.length(), 1);
            int wId = int(rtSceneAlbedo[lo].w + 0.5);
            int wLayer = (wId >= 0 && wId < nWLL) ? wId : 0;
            WaterParamsNamed wp = waterParamsNamed(waterParams[wLayer]);
            vec3 waterColor = rtResolveWaterHit(wp, hitPos, hitN, dir);
            // No bounce off water hits: the water look already includes its
            // mirror, and recursive water rays self-intersect the flat BLAS
            // mesh (water-on-water triangle noise).
            // a=0.8 marks a resolved reflected-water hit (source=water in the
            // reflection-source debug view); refraction carries thickness.
            return vec4(waterColor, refraction ? min(hitT, cap) : 0.8);
        }
        // Off-screen fallback: real interpolated triangle normal (see
        // main.frag) for relief and correct shading; albedo is the triplanar
        // sample at the true hit position, blended to the chunk average with
        // distance.
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
        vec3 toSun = normalize(rt.sunDirection);
        float ndl = max(dot(hitN, toSun), 0.0);
        float hitShadow = ShadowCalculation(
            ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, 0.0015);
        // Sky ambient scaled by the hit albedo (the raster's convention is
        // albedo * ambient): the old dark constant left grazing/off-screen
        // hits near-black — the mirror darkening near the shore.
        vec3 color = albedo * (rt.sunColor * ndl * (1.0 - hitShadow)
                               + vec3(0.26));
        // Reflection-inside-reflection: a reflective solid hit chains extra
        // mirror rays using the chunk's mirror strength (packed in
        // rtSceneAlbedo[].w).
        if (!refraction) {
            float refl = clamp(rtSceneAlbedo[lo].w, 0.0, 1.0);
            int extraBounces = clamp(rt.maxReflectionBounces, 0, 3) - 1;
            if (refl > 0.02 && extraBounces >= 0) {
                vec3 nextDir = normalize(reflect(normalize(dir), hitN));
                vec3 bounceCol = rtTraceMirror(hitPos + hitN * 0.05, nextDir,
                                               extraBounces, waterMinHit);
                color = mix(color, bounceCol, refl);
            }
        }
        return vec4(color, refraction ? min(hitT, cap) : 1.0);
    }

    // Unreachable with the current masks (both paths select the scene
    // instance only), but keep a well-defined miss-shaped fallback so a
    // future mask change cannot fall out of a non-void function.
    vec3 sky = textureLod(skyEquirectTex, rtDirToEquirectUV(normalize(dir)), 0.0).rgb;
    return refraction ? vec4(sky, RT_DEEP_WATER) : vec4(sky, 0.0);
}
#endif

// Near/far planes for linearizing depth – read from UBO passParams (z = near, w = far)
// so they always match the glm::perspective call on the CPU side.


// Linearize depth from Vulkan [0,1] depth buffer to eye-space distance.
// With GLM_FORCE_DEPTH_ZERO_TO_ONE the projection maps z_eye to [0,1]:
//   d = f*(z - n) / (z*(f - n))   =>   z = n*f / (f - d*(f - n))
float linearizeDepth(float depth) {
    float nearPlane = ubo.nearPlane;
    float farPlane  = ubo.farPlane;
    return (nearPlane * farPlane) / (farPlane - depth * (farPlane - nearPlane));
}

// Screen-space reflection march over the solid pass depth (set 2, binding 5).
// Returns the hit color in rgb and a confidence in a (0 = no usable hit);
// `hitPos` receives the world position of the solid surface at the hit. This
// fallback covers the on-screen solids the exact-triangle ray misses (coarse
// LoD / undisplaced-BLAS slips): without it those solids simply vanish from
// the reflection. The CALLER rejects hits below the local water level, so the
// submerged bed can never be painted as a reflection (the far-water terrain
// artifact). `maxT` bounds the march; the camera projection is viewProjection
// (invViewProjection maps clip->world).
vec4 traceSSR(vec3 origin, vec3 dir, float maxT, out vec3 hitPos) {
    hitPos = vec3(0.0);
    float nearP = ubo.nearPlane;
    float farP  = ubo.farPlane;
    float prevT = 0.0;
    float t = 0.25;
    for (int i = 0; i < 64; ++i) {
        // Near field: 1 m steps (thin silhouettes); far field: 13% geometric
        // growth so the remaining steps reach the cap without huge near steps.
        t += (i < 20) ? 1.0 : max(2.0, t * 0.13);
        if (t > maxT) break;
        vec3 P = origin + dir * t;
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
                    if (cm.w > em) { hi = mid; uv = uvm; }
                    else lo = mid;
                }
                float edge = smoothstep(0.0, 0.06, uv.x) * smoothstep(0.0, 0.06, 1.0 - uv.x)
                           * smoothstep(0.0, 0.06, uv.y) * smoothstep(0.0, 0.06, 1.0 - uv.y);
                float dHit = textureLod(solidSceneDepthTex, uv, 0.0).r;
                vec4 wh = ubo.invViewProjection * vec4(uv * 2.0 - 1.0, dHit, 1.0);
                hitPos = wh.xyz / max(wh.w, 1e-6);
                return vec4(textureLod(solidSceneColorTex, uv, 0.0).rgb, edge);
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
    // Default aux outputs: every early-return debug view leaves a defined
    // value (zeros = no body, no weighted contribution, no measured depth).
    // WATER_NO_BODY (H4): compiled out with the output declarations.
#ifndef WATER_NO_BODY
    outWaterBody = vec4(0.0);
    outWaterColumn = vec4(0.0);
#endif

    // Get water parameters from SSBO indexed by fragment brushIndex.
    // Brush ids are TERRAIN paint ids (0..N textures) while the water SSBO
    // holds only a few layers: out-of-range ids (e.g. painted shore rings)
    // would read past the allocation (undefined values: black opaque water
    // with RT disabled). Fall back to layer 0 (the default water look) so
    // every pixel renders defined water. fragBrushIndex itself is left raw
    // so DEBUG_MODE_MATERIAL_INDEX can still show the true id distribution.
    int nWaterLayers = max(waterParams.length(), 1);
    int waterLi = (fragBrushIndex >= 0 && fragBrushIndex < nWaterLayers) ? fragBrushIndex : 0;
    WaterParamsNamed wp = waterParamsNamed(waterParams[waterLi]);
    float time = waterRenderUBO.waterTime;

    // Water rendering parameters from selected water params
    float refractionStrength = wp.refractionStrength;
    float fresnelPower = wp.fresnelPower;
    float transparency = wp.transparency;
    float waterTint = wp.waterTint;
    float noiseScale = wp.noiseScale;
    int noiseOctaves = int(max(wp.noiseOctaves, 1.0));
    float noisePersistence = wp.noisePersistence;
    float noiseTimeSpeed = wp.noiseTimeSpeed;
    float noiseLacunarity = wp.noiseLacunarity;
    float reflectionStrength = wp.reflectionStrength;
    float specularIntensity = wp.specularIntensity;
    float specularPowerParam = wp.specularPower;
    float glitterIntensity = wp.glitterIntensity;

    // Feature toggles
    bool enableReflection = wp.enableReflection;
    bool enableRefraction = wp.enableRefraction;
    // NOTE: do NOT fold waterRenderUBO.refractionAllowed into this. That flag is
    // settings.rtRefractions - it exists to gate the RAY-TRACED refraction paths
    // (which need the pipeline/TLAS). The raster Snell-landing refraction below
    // works without any ray tracing, so gating it on the RT toggle disabled
    // water refraction completely whenever the RT preset was off (the default),
    // which is exactly what "refraction not working" looked like.
    // Blur is a two-way gate like refraction: the per-material flag
    // (blurParams.x) AND the global Settings toggle (waterRenderUBO.blurAllowed)
    // must both be on, so the Minimal preset can disable blur without
    // overwriting the authored per-layer params.
    bool enableBlur = wp.enableBlur && waterRenderUBO.blurAllowed;
    // Global ray-path gates from Settings, delivered via the water render UBO
    // so they apply in BOTH fragment variants (the non-RT variant has no `rt`
    // block). Refraction folds the settings gate in: Minimal ships
    // rtRefractions = false and must not refract at all, while Maximum sets it
    // true and gets the full Snell path (inline ray, or the raster landing when
    // the RT is unavailable). The REFLECTION deliberately does not fold it in -
    // with the ray off it falls back to the sky equirect, which is exactly the
    // sky-only mirror Minimal wants. The surface shading (Gerstner swell + FBM
    // ripples) is preset-independent and identical in both modes.
    enableRefraction = enableRefraction && waterRenderUBO.refractionAllowed;
    // Reflection stays ON with the RT ray off: the mirror falls back to the
    // sky equirect (sky-only reflection), which is the requested raster
    // behavior. The ray itself is gated at the trace site (`rt.waterReflections`).
    // During 360 cubemap capture, skip reflection/refraction to avoid feedback.
    const bool captureMode = ubo.cubemapCapture;
    if (captureMode) { enableReflection = false; enableRefraction = false; }

    // animTime is the RAW water time. Wave Speed alone drives the swell (the
    // Gerstner phase and the shore drift); Noise Time Speed alone drives the
    // noise chains, which multiply it in explicitly below. Folding the noise
    // speed in here coupled the two and double-scaled the noise axis.
    float animTime = time;

    // Compute the clip → screen UV once and reuse it everywhere (the conversion
    // is identical for every use below).
    vec2 screenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;

    // === SOLID-OCCLUSION REJECTION (perf report 20 C5) ===
    // The water pass is a separate pass that runs AFTER the solid pass (the
    // water task waits tlSolid) and solidSceneDepthTex is bound and in
    // SHADER_READ_ONLY, so a water fragment the terrain already covers can be
    // rejected here instead of being fully shaded and thrown away by the
    // composite's identical test (postprocess.frag: obstacleDepth <
    // waterGeomDepth -> waterAlpha = 0). Nothing is rejected that the
    // composite would have kept: the eye-space ordering below is the same
    // ordering the composite compares, the 5 cm bias only makes this test
    // stricter, and a discarded fragment leaves the water targets at their
    // clear values (colour alpha 0, depth 1.0) - exactly what the composite
    // expects of an uncovered pixel.
    //
    // Compared in EYE SPACE with a 5 cm bias: an absolute NDC epsilon is
    // useless at this near/far ratio (1e-5 NDC is ~100 m at 1 km), and the
    // bias keeps two surfaces that are within a few centimetres of each other
    // (the shoreline, where the water meets the bank) from flickering between
    // accepted and rejected. depthParams.x gates the whole test: it is set
    // only when solidSceneDepthTex holds THIS frame's solid depth. The
    // water-in-main variant binds the PREVIOUS frame's depth for its in-trace
    // lookups, so a discard there would punch holes under camera motion, and
    // in that variant the hardware depth test already does this job.
    //
    // Cost: one depth fetch and two linearisations, against the whole water
    // shading stack for every occluded pixel. The `discard` does disable
    // hardware early-Z for this pipeline, but the water depth target starts
    // cleared and the surface is a single layer, so the early-Z it loses is
    // self-occlusion that barely happens.
    // Debug views must see the water: the rejection below discards fragments,
    // and a discard returns before the debug dispatch, so a mis-firing test
    // showed up as black water in every view (Reflection Vector first).
    if (ubo.debugMode == 0 && waterRenderUBO.solidDepthIsCurrent) {
        float solidDepthRaw = textureLod(solidSceneDepthTex, screenUV, 0.0).r;
        if (solidDepthRaw < 1.0) {
            float solidEye = linearizeDepth(solidDepthRaw);
            float waterEye = linearizeDepth(gl_FragCoord.z);
            // Tolerance in eye space proportional to distance: at range the
            // depth precision alone exceeds a fixed 5 cm, which rejected whole
            // distant water areas. 0.5% of the eye depth plus the fixed floor.
            float tol = 0.05 + 0.005 * waterEye;
            if (solidEye < waterEye - tol) {
                discard;
                return;
            }
        }
    }

    // === WATER VOLUME THICKNESS ===
    // Compute volume thickness from back-face depth (rendered with reversed winding)
    // before the normal computation, so we can modulate bump amplitude.
    // The thickness is measured entirely from the water front/back faces (the
    // solid depth is only used for the occlusion rejection above). This means
    // a flat height-field surface (back-face ≈ front-face) reports ~0 thickness, which
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
    vec3 worldRayDir = normalize(worldFrontPos - ubo.viewPosition);
    float backFaceThickness = max(dot(backFaceWorld - worldFrontPos, worldRayDir), 0.0);
    // A single-layer height-field surface (flat plane or tessellated waves) has
    // backFaceThickness ≈ 0 because the back-face geometry is co-planar with the
    // front face.  Comparing raw depth values with a fixed epsilon is unreliable
    // across different near/far planes and camera distances.  Instead, check the
    // already-computed world-space thickness: only trust the back face when it
    // represents a genuinely thick water body (>= 5 cm), not a thin surface.
    const float kMinVolumeThickness = 0.05; // 5 cm world-space
    // Also reject backFaceDepthRaw == 1.0 (depth-clear value = no geometry rendered).
    bool hasValidBackFace = (backFaceDepthRaw < 1.0) && (backFaceThickness > kMinVolumeThickness);
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

    // ── Wave detail LOD: NO distance cutoff (perf report 20 C4, revised) ──
    // The wave field is the single sine swell and is evaluated at EVERY
    // distance; the normal is never faded to the flat base normal. The octave
    // budget below feeds the FOAM NOISE chain only (a sine has no octaves), so
    // the swell's shape and its alias-free analytic slope survive at any
    // range instead of turning into a flat mirror.
    //
    // Cost: one evaluation normally, two inside the band (the coarse one is
    // cheap), one coarse past it — still less than the pre-LOD code, which ran
    // the full spectrum at every distance.
    const float kWaterDetailNear = 200.0;
    const float kWaterDetailFar  = 1000.0;
    float camDist = length(ubo.viewPosition - fragPosWorld);
    float detail = 1.0 - smoothstep(kWaterDetailNear, kWaterDetailFar, camDist);
    // Foam-noise octave budget (the swell itself needs no octaves).
    int waveFieldOct = noiseOctaves;
    // Sub-pixel noise terms (highlight perturbation, glitter, caustics) fade
    // over the tail of the band so they stop without a step where the detail
    // band ends. The WAVES do not use this: the field and its normal run at
    // every distance.
    float waveNoiseTail = 1.0 - smoothstep(0.8 * kWaterDetailFar, kWaterDetailFar, camDist);

    vec3 normal;
    // Same single thickness-zoned wave field the TES displaced the geometry
    // with: identical undisplaced base position, animation time, water
    // thickness (fragWaterDepth forwarded by the TES) and raw amplitude
    // (fragBasePos.w), so the per-pixel analytic normal matches the
    // rasterized surface exactly while still resolving detail far below the
    // tessellation density. waveField.foam carries the whitewater coverage
    // used by the foam shading below.
    // Screen-space world footprint of this pixel, for the analytic band AA.
    vec2 waveDx = dFdx(fragBasePos.xz);
    vec2 waveDy = dFdy(fragBasePos.xz);
    WaterWaveField waveField = waterWaveField(
        fragBasePos.xyz, 
        animTime, 
        fragWaterDepth, 
        fragBasePos.w, 
        fragShoreDir, 
        wp, 
        true,
        waveFieldOct,
        waveDx,
        waveDy);
    // The swell is a single sine: it has no octave spectrum, so no coarse/
    // full blend is needed - the closed form IS the wave at every distance.
    {
        float dhdT = dot(waveField.grad, T);
        float dhdB = dot(waveField.grad, B);

        normal = normalize(flatN - dhdT * T - dhdB * B);
    }

    // Ripples: directional FBM detail, PER PIXEL only. The vertices keep the
    // Gerstner swell; this is the sub-tessellation roughness, so it exists on
    // the shading normal alone. It advects toward the shore with the same
    // motion as the swell (the noise is sampled upstream of the shore drift),
    // its strength follows the wave steepness, and it fades with the distance
    // LOD so its finest octaves can never alias.
    {
        const int kRippleOctaves = 3;
        // Ripple HEIGHT in metres, faded by the distance LOD and scaled by the
        // steepness. Treating it as a height (not a slope) is what lets the
        // normal be the one the surface would have with infinite tessellation:
        // the displacement the vertices cannot afford is still in the normal.
        float rippleHeight = max(wp.rippleHeight, 0.0)
                           * clamp(wp.waveSteepness, 0.0, 1.0) * detail;
        if (rippleHeight > 0.0 && wp.noiseScale > 0.0) {
            vec3 drift = vec3(fragShoreDir.x, 0.0, fragShoreDir.y) * (wp.waveSpeed * animTime);
            vec4 n = fbmGrad4D(vec4((fragBasePos.xyz - drift) * wp.noiseScale,
                                    animTime * wp.noiseTimeSpeed),
                               kRippleOctaves, wp.noisePersistence, wp.noiseLacunarity);
            vec3 rg = vec3(n.y, 0.0, n.w) * wp.noiseScale;   // d/dx, d/dz
            normal = normalize(normal - dot(rg, T) * rippleHeight * T
                                      - dot(rg, B) * rippleHeight * B);
        }
    }

    // Calm-patch gate for the wave-derived NOISE chains (see waterWaveField:
    // the mask returns early there). dbg.y is the calm mask, and it is 0
    // whenever the field did not run for any reason - calm patch, waves off,
    // octave budget 0 - so everything wave-derived can be skipped together.
    bool wavesActive = waveField.calmMask > 0.0;

    
    // Normalize vectors
    vec3 viewDir = normalize(ubo.viewPosition - fragPosWorld);
    // Two-sided shading: which side of the surface we are looking at is decided
    // by the UNDISPLACED base normal (the back-face pass sees the underside).
    // Testing the wave-perturbed normal instead flips every steep wave face that
    // points away from the camera, lighting the backs of the crests like the
    // fronts - that is what made the surface read as crumpled / striped.
    if (dot(flatN, viewDir) < 0.0) normal = -normal;

    vec3 lightDir = normalize(-ubo.lightDirection);
    
    // Base screen UV already computed at the top of main() and reused above.
    int dbgMode = ubo.debugMode;

    // === HYBRID RT STATE ===
    // Per-lobe precedence is documented in the header above (reflection:
    // exact-inline-wins with pipe covering budget skips; refraction:
    // pipe-first with inline fallback). rt.useWaterPipeline carries
    // settings.rtWaterPipeline.
    bool rtReady = false;
    bool usePipe = false;
#ifdef RT_ENABLED
    rtReady = rt.tlasReady;
    usePipe = rt.useWaterPipeline;
#endif

    // === NOISE-BASED REFRACTION (perf report 20 C2) ===
    // One low-octave 3D FBM pair instead of four chained 4D FBM layers. Only
    // evaluated when the distortion can contribute at all (refraction on AND a
    // non-zero authored strength, or the debug view that visualizes it), and
    // faded out once a pixel's world-space footprint exceeds the finest
    // retained noise feature: past that point the offset is sub-pixel, so the
    // noise is pure cost. The fade scales the offset rather than cutting it,
    // so no pop appears at the fade distance.
    // One wave: the refraction samples the scene straight through the surface -
    // the FBM screen-space wobble (an extra ripple system on top of the sine)
    // is gone. `refractionNoise` is kept for the noise debug view.
    vec2 refractionNoise = vec2(0.0);   // noise debug view only
    
    // Reduce refraction at edges (to avoid sampling outside screen)
    float edgeFade = smoothstep(0.0, 0.1, screenUV.x) * smoothstep(1.0, 0.9, screenUV.x) *
                     smoothstep(0.0, 0.1, screenUV.y) * smoothstep(1.0, 0.9, screenUV.y);
    
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
    float waterIor = clamp(wp.waterIor, 1.0, 2.5);
    float refrThickCap = max(wp.maxThickness, 0.0);
    vec3 absorbCoeff = wp.absorption;
    float absorbScaleBase = max(wp.absorptionScale, 0.0);
#ifdef RT_ENABLED
    // Shared ray constant (also read by the Beer-Lambert block below:
    // the deep-water marker substitutes maxRefr as a caustic/viz thickness).
    float maxRefr = max(rt.maxRefractDistance, 1.0);
#else
    float maxRefr = 300.0;
#endif
#ifdef RT_ENABLED
    // === RAY-BUDGET DECISIONS (perf: fewer inline rays, exactness kept) ===
    // Early Fresnel estimate (identical formula to the composite below) for
    // Fresnel-weighted single-ray selection. reflMixEst is the lobe weight the
    // composite will use (uniform flag ? reflectionStrength : Fresnel mix).
    // The reflection lobe always traces (a skipped mirror is a missing
    // mirror); the single-ray xor cuts the REFRACTION lobe, which recovers
    // from the raster bottom (zero rays) or sky where skipped.
    float fresnelEarlyCurve = pow(1.0 - clamp(dot(viewDir, normal), 0.0, 1.0),
                                 clamp(fresnelPower, 1.0, 8.0));
    float fresnelEarly = clamp(0.02 + 0.98 * fresnelEarlyCurve, 0.0, 1.0);
    bool uniformEarly = wp.uniformReflection;
    float reflMixEst = uniformEarly
        ? clamp(reflectionStrength, 0.0, 1.0)
        : mix(fresnelEarly, 1.0, clamp(reflectionStrength, 0.0, 1.0));
    // Blue-noise-ish stochastic selector (FragCoord hash): single-ray mode
    // traces reflection XOR refraction with probability = reflMixEst
    // (Schlick-weighted) to save a ray on hits. The refraction lobe recovers
    // from the raster bottom when skipped, so no pixel ends with two empty
    // lobes ("just tinted"). Traced-result debug views (see
    // debugModeForcesRtReference) force dual-trace + full-rate.
    float hash01 = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    bool waterRefMode = false;
    bool waterSingleRay = false;
    float waterContribMin = 0.02;
    // Ray mask / depth source visualize the budgeted behavior itself, so they
    // must not force reference (otherwise they could never show the live cut).
    waterRefMode = debugModeForcesRtReference(rt.debugMode);
    waterSingleRay = rt.singleRay && !waterRefMode;
    waterContribMin = clamp(rt.reflectionContribMin, 0.0, 1.0);
    // Single-ray ray budget (rt.singleRay): traces the reflection XOR
    // refraction stochastically with probability = reflMixEst (Schlick-weight)
    // to save one full-resolution ray on dual-lobe pixels. Only the
    // REFRACTION lobe honors the cut: a skipped refraction ray falls back to
    // the raster bottom (zero rays) or sky, whereas a skipped reflection is a
    // missing mirror (the shoreline artifact the reflection lobe must not
    // reintroduce), so reflection always traces. Checkerboard
    // (rt.checkerboardReflections) is intentionally NOT applied to water: an invalid
    // pipe texel has no cover and must trace, and dithering mirrors is
    // visible. Solid reflections keep their own checkerboard gate.
    bool wantRefrInline = true;
    if (waterSingleRay && enableReflection && enableRefraction && !captureMode) {
        bool pickRefl = hash01 < clamp(reflMixEst, 0.0, 1.0);
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
    // here), refraction AND reflection always trace: force dual-trace by
    // clearing the xor pick. Flat/calm water (no raster column) keeps the
    // budgeted path above. Contrib gates stay (a negligible lobe is invisible
    // traced or not); reference mode already forces full, so this is a no-op
    // there.
    if (hasValidBackFace) {
        wantRefrInline = true;
    }
#endif
    // Ray-mask for DEBUG_MODE_RAY_MASK (pixel-ratio counter): per lobe 0=disabled,
    // 1=sky fallback, 2=pipe hit, 3=inline traced, 4=skipped by budget gate.
    float refrMask = 0.0;
    float reflMaskDbg = 0.0;
    // Bottom-lobe content flags for miss-recovery (reflection recovers when
    // the bottom ended content-free): pipe cover, real inline hit
    // (a < RT_DEEP_WATER, i.e. not the miss marker), raster recovery.
    bool pipeRefrValid = false;
    bool refrInlineHitReal = false;
    // Thickness-source id for DEBUG_MODE_DEPTH_SOURCE (which branch set the water
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
    // Refraction perturbation: the SAME directional FBM that shades the ripples
    // also perturbs the transmitted ray, so the refracted image is distorted by
    // the surface noise - the ripple slopes warp the lookup, exactly like the
    // normal they were derived from. One FBM evaluation, sampled on the same
    // shore-directed drift as the swell and the ripples.
    vec2 refractionOffset = vec2(0.0);
    if (enableRefraction && wp.noiseScale > 0.0) {
        const int   kRefrNoiseOctaves = 3;
        // The distortion runs finer than the surface ripples (the classic water
        // refraction warp): 2.5x the ripple frequency.
        const float kRefrNoiseScale = 2.5;
        float refrNoiseScale = wp.noiseScale * kRefrNoiseScale;
        vec3 drift = vec3(fragShoreDir.x, 0.0, fragShoreDir.y) * (wp.waveSpeed * animTime);
        // fbmGrad4D returns vec4(value, d/dx, d/dy, d/dz) over (x, y, z, t), so
        // (n.y, n.w) is the world-xz gradient; scaling by the sample scale gives
        // the world-space slope, which is what bends the ray.
        vec4 n = fbmGrad4D(vec4((fragBasePos.xyz - drift) * refrNoiseScale,
                                animTime * wp.noiseTimeSpeed),
                           kRefrNoiseOctaves, wp.noisePersistence, wp.noiseLacunarity);
        refractionOffset = vec2(n.y, n.w) * refrNoiseScale * 0.5;
    }
    if (enableRefraction) {
        // Approximate air->water refraction. GLSL `refract` expects the incident
        // vector (eye -> surface), i.e. -viewDir; the result is the true
        // transmitted ray pointing INTO the water toward the underwater scene.
        vec3 refrRay = refract(-viewDir, normal, 1.0 / waterIor);
        if (length(refrRay) < 1e-5) {
            // fallback to reflection if total internal reflection occurs
            refrRay = reflect(-viewDir, normal);
        }
        // Refraction Strength is the amount of bending: 1 = the full Snell ray,
        // 0 = straight through (no refraction), in between = the classic partial
        // refraction. This is the layer's control over the effect - it used to
        // only scale the removed Perlin distortion, which left the knob dead and
        // the refraction fixed at full strength for every layer.
        float refrAmount = clamp(wp.refractionStrength, 0.0, 1.0);
        refrRay = normalize(mix(normalize(-viewDir), refrRay, refrAmount));
        // Noise perturbation in the surface tangent frame: the ripple noise bends
        // the ray too, so the bottom warps with the same detail the surface has.
        refrRay = normalize(refrRay + (T * refractionOffset.x + B * refractionOffset.y) * refrAmount);
        refrRayW = refrRay;
        haveRefrRayW = true;
        bool refrResolved = false;
#ifdef RT_ENABLED
        if (usePipe && rt.refractionsEnabled) {
            // Half-res single-mip pipeline output: explicit LOD 0 (also safe
            // under the per-fragment pipe-validity branch).
            vec4 pipeRefr = textureLod(rtRefractTex, screenUV, 0.0);
            if (pipeRefr.a >= 0.0) {
                sceneColor = pipeRefr.rgb;
                // Thickness only when the RT-thickness toggle is on: the
                // ray still runs for the refracted color, but its path
                // length is not consumed as a water column when disabled.
                rtThickness = rt.thicknessEnabled ? pipeRefr.a : -1.0;
                refrResolved = true;
                refrMask = 2.0;
                depthSource = 5.0;
                pipeRefrValid = true;
            }
        }
        // Inline fallback: runs when the pipe cannot cover this texel (invalid)
        // and the single-ray xor did not pick the reflection lobe for it. A
        // skipped refraction ray falls through to the raster recovery below
        // (zero-ray raster bottom sample) or the sky fallback, so the exact
        // bottom is never replaced with black.
        bool refrBudgetSkip = (refrContribEst < waterContribMin) || !wantRefrInline;
        if (waterRefMode) refrBudgetSkip = false;
        if (!refrResolved && rtReady && rt.refractionsEnabled && !refrBudgetSkip) {
            float refrSceneTMax = hasValidBackFace
                ? (backFaceThickness * 1.5 + 2.0)
                : min(maxRefr, max(refrThickCap * 3.0, 8.0));
            // Origin on the TRUE displaced surface (fragPosWorld): the
            // filtered pass skips water candidates, so the tMin self-guard
            // only needs to clear coplanar touch — and the reported length
            // is the visible water column, not base-to-bottom.
            vec4 hit = rtTraceWater(fragPosWorld, refrRay, refrSceneTMax, true, refrThickCap, 0.0);
            sceneColor = hit.rgb;
            // a >= 0 always from rtTraceWater: capped path length on hit, or
            // RT_DEEP_WATER marker on miss (deep, unresolved water). Only a
            // real triangle hit counts as bottom content for miss-recovery.
            rtThickness = rt.thicknessEnabled ? hit.a : -1.0;
            rtThickFromScene = true;
            refrResolved = true;
            refrMask = 3.0;
            depthSource = 1.0;
            refrInlineHitReal = (hit.a < RT_DEEP_WATER);
        } else if (!refrResolved && rtReady && rt.refractionsEnabled && refrBudgetSkip) {
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
            if (rtReady && rt.refractionsEnabled && haveRefrRayW
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
                // Non-RT / no-cover fallback, split by RT availability:
                //  * RT refraction unavailable (non-RT build, RT off, or the
                //    refraction ray path disabled): recover the bottom in
                //    screen space — sample the solid pass color at the
                //    Perlin-distorted UV, exactly like the legacy raster
                //    water. The reflection lobe below stays the sky fallback,
                //    so both lobes are still combined by Fresnel/strength.
                //  * RT available but this pixel was budget-skipped with no
                //    raster recovery: sample the refracted sky (the exact ray
                //    was intentionally not cast).
                bool rtRefrAvailable = false;
#ifdef RT_ENABLED
                rtRefrAvailable = rtReady && rt.refractionsEnabled;
#endif
                bool refrServed = false;
                if (!rtRefrAvailable) {
                    // One wave: the lookup lands where the SNELL ray meets the
                    // real bottom. refrRayW is produced by refract() against the
                    // per-pixel Gerstner normal, so the bottom is displaced by
                    // the wave and its ripples exactly like the shading - no
                    // noise offset involved. Only when no ray is available does
                    // this fall back to the straight-through sample.
#ifdef RT_ENABLED
                    if (haveRefrRayW) {
                        // rtRasterBottom() measures the column along the ray but
                        // samples the color at THIS pixel. Refraction needs the
                        // lateral shift, so take its thickness, land the Snell
                        // ray on the bottom and sample there. The ray is bent by
                        // the per-pixel Gerstner normal (refrRayW), so the
                        // displaced bottom is exactly what the shading sees.
                        vec4 rbFb = rtRasterBottom(fragPosWorld, refrRayW, screenUV);
                        if (rbFb.a >= 0.0) {
                            vec3 landing = fragPosWorld + refrRayW * rbFb.a;
                            vec4 clipL = ubo.viewProjection * vec4(landing, 1.0);
                            if (clipL.w > 1e-4) {
                                vec2 uvL = clamp((clipL.xy / clipL.w) * 0.5 + 0.5, 0.001, 0.999);
                                sceneColor = textureLod(solidSceneColorTex, uvL, 0.0).rgb;
                                refrServed = true;
                            }
                        }
                    }
#endif
                    if (!refrServed) {
                        vec2 refrUV = clamp(screenUV, 0.001, 0.999);
                        if (textureLod(solidSceneDepthTex, refrUV, 0.0).r < 1.0) {
                            sceneColor = textureLod(solidSceneColorTex, refrUV, 0.0).rgb;
                            refrServed = true;
                        }
                    }
                }
                if (!refrServed) {
                    // Sky fallback (also the no-bottom / skipped-ray path):
                    // refracted sky through the surface, no thickness
                    // (back-face raster method covers thickness). Explicit
                    // LOD: per-fragment control flow.
                    sceneColor = textureLod(skyEquirectTex, waterDirToEquirectUV(refrRay), 0.0).rgb;
                }
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
#ifdef RT_ENABLED
    // Hash-dither resolved RT hit lengths (±0.3 m). Per-chunk flat box tops
    // quantize the true depth into steps; undithered, those steps print as
    // terrace bands. Dithered they degrade to grain, which reads as water
    // noise. Miss marker and "no RT" (-1) are never touched. Skipped for
    // exact scene-triangle hits (rtThickFromScene): dithering those would
    // reintroduce the noise the exact geometry just removed.
    if (rtReady && rt.thicknessEnabled && !rtThickFromScene && rtThickness >= 0.0 && rtThickness < RT_DEEP_WATER) {
        float h = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        rtThickness = max(rtThickness + (h - 0.5) * 0.6, 0.0);
    }
    if (rtReady && rt.thicknessEnabled && rtThickness >= 0.0) {
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
    float depthFalloff = wp.depthFalloff;
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
    
    // Main specular highlight with noise perturbation. The noise FBM only
    // runs when the highlight/glitter can contribute (intensity > 0) AND the
    // layer actually has a wave field to perturb: with waves off
    // (!wp.enableWaves) waterWaveField() returns identically zero, so
    // the FBM perturbation and the glitter sparkles only add noise to a flat
    // surface. The unperturbed analytic highlight is kept instead.
    //
    // Unlike the wave field itself (which is evaluated at every distance), the
    // highlight PERTURBATION stops at the detail band: at that range the
    // highlight is sub-pixel, so the FBM would only add shimmer on top of it,
    // never visible sparkle. This is not a wave cutoff.
    vec3 specularColor = vec3(0.0);
    // Debug channels for DEBUG_MODE_WATER_SPECULAR_NOISE: R = specular
    // perturbation, G = glitter (both 0.5 = the chain did not run), B = the
    // sun-lobe gate that lets them run at all. They stay at their neutral
    // values wherever the lobe gates the chains off, which is exactly what that
    // view exists to show.
    vec3 dbgHighlightNoise = vec3(0.5, 0.5, 0.0);
    // wavesActive: a calm patch has no wave field to perturb, so it takes the
    // unperturbed analytic highlight below (same branch as waves-off).
    if (wp.enableWaves && detail > 0.0 && wavesActive) {
        // Lobe terms FIRST (perf report 20 M10): every noise chain below exists
        // only to perturb the highlight, so outside the sun lobe it is multiplied
        // by a number that is already zero. pow(specAngle, 128) is zero for all
        // but a ~22 deg cone around the mirror direction and pow(specAngle, 32)
        // for all but ~42 deg, so most water pixels now pay no noise at all
        // (previously 10 four-dimensional Perlin evaluations each). The
        // threshold is on the final contribution, in linear light: 1e-4 is
        // ~0.03/255, i.e. invisible.
        const float kSpecNoiseMin = 1e-4;
        float specLobe = pow(specAngle, specularPowerParam);
        float glitterLobe = pow(specAngle, 32.0);
        dbgHighlightNoise.z = max(specLobe, glitterLobe);

        // One wave: the highlight is the analytic specular lobe on the sine
        // normal - the FBM perturbation and the glitter sparkles (two more
        // ripple systems) are gone.
        dbgHighlightNoise.x = 0.5;
        dbgHighlightNoise.y = 0.5;
        if (specularIntensity > 0.0 && specularIntensity * specLobe > kSpecNoiseMin) {
            specularColor = ubo.lightColor * specLobe * specularIntensity;
        }
    } else if (specularIntensity > 0.0) {
        // Waves off: no field to perturb, so no FBM and no glitter; the
        // unperturbed analytic highlight is the whole specular lobe.
        specularColor = ubo.lightColor * pow(specAngle, specularPowerParam) * specularIntensity;
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
    // Reflection direction, taken about the normal that faces the VIEWER. The
    // water volume is drawn front and back, and the back-face pass carries a
    // downward normal: reflecting about it pointed the mirror into the ground
    // half of the equirect and rendered the reflection (and the Reflection
    // Vector debug view) black.
    vec3 reflNormal = (dot(normal, viewDir) < 0.0) ? -normal : normal;
    vec3 reflectDir = reflect(-viewDir, reflNormal);

    vec3 skyColor = vec3(0.0);
    bool reflResolved = false;
    // Which branch resolved the reflection (DEBUG_MODE_REFLECTION_SOURCE):
    // 0=disabled, 1=inline solid hit, 2=inline water hit, 3=guard, 4=SSR hit,
    // 5=inline sky miss, 6=equirect fallback.
    float reflSource = 0.0;
#ifdef RT_ENABLED
    // Reflection source: the INLINE exact-triangle ray only. The async proxy
    // pipeline no longer produces a reflection (its flat proxy output was
    // never used as color — see rt_water.rgen), so the old pipe-coverage
    // debug sample is gone with it.
    // Budget: only the negligible-contribution gate remains for reflection.
    // A skipped mirror is a missing mirror (exactly the shore bug), so the
    // xor/checkerboard cuts do not apply to the reflection lobe: every pixel
    // that can trace does trace. Reference mode clears the gate.
    // The per-layer Reflection toggle disables the lobe entirely: no inline
    // ray, no SSR fallback (capture mode forces it off too).
    bool reflBudgetSkip = (reflContribEst < waterContribMin);
    if (waterRefMode) reflBudgetSkip = false;
    bool reflDidTrace = false;
    vec4 reflHit = vec4(0.0);
    vec3 reflOrigin = vec3(0.0);
    if (enableReflection && !reflBudgetSkip && rtReady && rt.waterReflections) {
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
        // Origin on the VISIBLE displaced surface (not the base plane): near
        // the shore the base plane can sit under the bank, and an
        // underground origin hits terrain backfaces/self-water and paints
        // bogus reflections. The self-water distance guard below replaces
        // the old base-plane trick for keeping the reflector's own surface
        // out of the result.
        reflOrigin = fragPosWorld + reflBaseN * 0.05;
        float reflWaterMinHit = length(fragPosWorld - fragBasePos.xyz) + 2.0;
        reflHit = rtTraceWater(reflOrigin, normalize(reflectDir), RT_NO_LIMIT, false, 0.0, reflWaterMinHit);
        reflDidTrace = true;
        if (reflHit.a > 0.5) {
            skyColor = reflHit.rgb;
            reflResolved = true;
            reflMaskDbg = 3.0;
            reflSource = (reflHit.a > 0.9) ? 1.0 : ((reflHit.a > 0.7) ? 2.0 : 3.0);
        }
    }
    if (!reflResolved && reflDidTrace) {
        // Inline miss. Screen-space fallback for the on-screen solids the
        // exact-triangle ray misses (coarse LoD / undisplaced-BLAS slips):
        // without it those solids vanish from the reflection. Hits BELOW the
        // local water level are the submerged bed and are rejected (sky), so
        // the far-water "terrain painted as reflection" artifact cannot come
        // back. The march is distance-capped.
        const float kSsrMaxDist = 2000.0;
        vec3 ssrWorld;
        vec4 ssr = traceSSR(reflOrigin, normalize(reflectDir), kSsrMaxDist, ssrWorld);
        if (ssr.a > 0.02 && ssrWorld.y >= fragBasePos.y - 0.05) {
            skyColor = ssr.rgb;
            reflSource = 4.0;
        } else {
            skyColor = reflHit.rgb; // trace's own sky (miss baseline)
            reflSource = 5.0;
        }
        reflMaskDbg = 3.0;
        reflResolved = true;
    } else if (!reflResolved && rtReady && rt.waterReflections && reflBudgetSkip) {
        reflMaskDbg = 4.0;
    }
#endif
    if (!reflResolved) {
        // Explicit LOD: per-fragment fallback branch (see refraction above).
        // ALWAYS sampled: when the rays are off this is the water's only light
        // source, and leaving it conditional meant a false `enableReflection`
        // (or any future gate) rendered the water black. One texture fetch on
        // the ray-less path is cheap; a black lake is not.
        skyColor = textureLod(skyEquirectTex, waterDirToEquirectUV(normalize(reflectDir)), 0.0).rgb;
        reflSource = 6.0;
        if (reflMaskDbg == 0.0) reflMaskDbg = 1.0;
    }

    // (Screen-space refinement removed: the inline trace above hits the real
    // chunk triangles directly, so a depth-march pass is redundant.)
    //
    // (Aerial detail fade removed: it was a proxy-era workaround for box-step
    // classification edges, and it inflated waterThickness toward maxRefr (a
    // RAY-RANGE bound, not a water column) with camera distance. That tripped
    // the deep-ocean color stop at range, so distant water darkened
    // even when it was shallow. Refraction now traces exact scene triangles
    // with a continuous hit/miss path, so distance no longer needs to fade
    // depth/color. Debug snapshots (raw RT before composition) stay.)
    vec3 dbgSceneColor = sceneColor;
    vec3 dbgReflColor = skyColor;
    float thicknessForAlpha = waterThickness;

    // Uniform reflection toggle: when set, apply reflectionStrength uniformly
    // instead of modulating by Fresnel. This flag is stored in reserved2.w
    // (see wp.uniformReflection).
    bool uniformReflection = wp.uniformReflection;


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


    // Caustic parameters (physical, wave-shape driven — see the caustics
    // block below). Only color, strength and the fold softness remain: the
    // pattern, its spatial scale and its animation all come from the wave
    // height field itself (no separate noise/scale/velocity knobs).
    vec3 causticColor = wp.causticColor;
    float causticIntensity = wp.causticIntensity;
    float causticSoftness = clamp(wp.causticSoftness, 0.02, 1.0);

    // Tint color: 5-stop depth-region ramp keyed to the shore-wave zone
    // boundaries, so the tint color follows the measured depth bands (shore
    // line → foam band → breaker line → shoaling band → open ocean) via the
    // shared waterRegionTint() helper.
    // Depth signal the regions are defined on: the TES-measured vertical
    // world-space drop (fragWaterDepth, always finite), with the composed
    // thickness signal as a fallback when the measured value is unusable.
    float regionDepth = (fragWaterDepth >= 0.0) ? fragWaterDepth : tintDepth;
    vec3 waterTintColor = waterRegionTint(wp, regionDepth);

    // Shoreline tint fade: the tint weight ramps to exactly 0 at the
    // waterline over the per-layer tint shore fade depth, so shore water near
    // the border is transparent and shows the bottom with no water color.
    float tintShoreFade = 1.0;
    if (wp.tintShoreFadeDepth > 0.0 && regionDepth > 1e-4) {
        tintShoreFade = smoothstep(0.0, max(wp.tintShoreFadeDepth, 1e-4), regionDepth);
    }

// Blend scene color with water tint: depthFade (Depth Falloff over the best
// depth signal) sets the amount, Water Tint scales it, Transparency caps it
// (1 = crystal clear keeps the refracted bottom, 0 = fully tintable).
    float tintMax = clamp(1.0 - transparency, 0.0, 1.0);
    float tintBlend = clamp(depthFade * waterTint * tintShoreFade, 0.0, tintMax);
    // No refraction (Minimal): nothing filled the refracted color, and leaving
    // it at its zero value rendered the water black-transparent. Fall back to
    // the sky along the view direction, so the water is tinted transparency -
    // the sky shows through it - instead of black.
    if (!enableRefraction) {
        // Sky source for the transmission term. The view direction points down
        // and an equirect's lower half is ground/black, so this samples the
        // sky along the reflected direction - the same sky the mirror uses, so
        // both terms stay lit and vary per pixel with the Gerstner + FBM normal.
        sceneColor = textureLod(skyEquirectTex, waterDirToEquirectUV(normalize(reflectDir)), 0.0).rgb;
    }
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
        // RT on/off: skyColor is the traced scene when the RT lobes are
        // available, else the sky-equirect fallback along the same reflect
        // direction; refractedColor is the traced Snell bottom with RT on and
        // the screen-space solid sample with RT off. The SAME Fresnel/strength
        // weighting composes both cases, so only the lobe sources change.
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

    // === CAUSTICS (physical, driven by the wave height field) ===
    // Sunlight refracted at the wavy surface converges on the lake bottom.
    // For a height field h(x,z), the ray entering at the surface lands
    // displaced horizontally by D = d·(dh/du)·K along the sun azimuth u,
    // with the exact first-order Snell coefficient
    //     K = cos(theta_i) / (n · cos^3(theta_t)),
    // where d is the water column. The bottom irradiance is the inverse
    // Jacobian of the map x -> x + D(x):
    //     E/E0 = 1 / |1 + d · K · d2h/du2|
    // so the caustic pattern is produced ONLY by the wave field — the same
    // field that displaces the surface and builds the analytic shading
    // normal. No independent caustic noise, scale, or animation
    // clock: the pattern rides the waves, moves with them, and inherits
    // their spectrum by construction. The added light is the EXCESS over
    // the flat-surface case and is attenuated by the same Beer-Lambert
    // transmittance as the bottom behind it.
    float caustic = 0.0;
    float causticGain = 1.0;   // bottom irradiance ratio 1/|J|
    // The caustic pattern feeds the final mask debug view, so keep it
    // computed while that view is selected even if the effect is disabled.
    bool causticDebugMode = (dbgMode == DEBUG_MODE_CAUSTICS);
    // Enter only when the effect can contribute: the pattern is produced ONLY
    // by the wave field (its curvature is identically zero when
    // !wp.enableWaves), and the column must be
    // deep enough to focus sunlight (flat/no-volume water reports
    // waterThickness = 0, which makes the Jacobian exactly 1 and the excess
    // exactly 0). The debug view still forces the block so its mask stays
    // populated.
    // Attenuated by the water column (the focused light travels down to the
    // bottom and back to the eye through the same absorption).
    waterColor += causticColor * (caustic * transmittance);

    // Apply per-vertex HSV: rotate hue, offset saturation, scale value
    vec3 hsvColor = fragHSV;
    vec3 texHSV = rgbToHsv(waterColor);
    texHSV.x = mod(texHSV.x + hsvColor.x, 360.0);
    texHSV.y = clamp(texHSV.y * (hsvColor.y * 2.0), 0.0, 1.0);
    texHSV.z *= hsvColor.z * 2.0;
    waterColor = hsvToRgb(texHSV);

    // === VOLUMETRIC SCATTERING ===
    // Single-scattering approximation of sunlight scattered inside the
    // measured water column toward the eye: Henyey-Greenstein phase (g),
    // saturating with thickness (density), attenuated by the same
    // Beer-Lambert transmittance the bottom light travels through. The phase
    // is scaled by 4*pi so 1.0 is the isotropic limit (g = 0). Amount, tint,
    // density and anisotropy are all per-layer parameters.
    if (wp.enableVolumetric && wp.volumetricStrength > 0.0) {
        float volDepth = max(waterThickness, 0.0);
        float volAtt = 1.0 - exp(-volDepth * max(wp.volumetricDensity, 0.0));
        float volCos = clamp(dot(viewDir, lightDir), -1.0, 1.0);
        float volG = clamp(wp.volumetricPhaseG, -0.95, 0.95);
        float volDenom = max(1.0 + volG * volG - 2.0 * volG * volCos, 1e-4);
        float volPhase = (1.0 - volG * volG) / pow(volDenom, 1.5);
        waterColor += wp.volumetricColor * ubo.lightColor
                    * volPhase * volAtt * wp.volumetricStrength * transmittance;
    }

    // === FOAM / WHITEWATER ===
    // Whitewater coverage from the same shore-wave field that displaced and
    // shaded the surface (waveField.foam). Lit by the sun with a configurable
    // ambient floor, composited over the water color per layer. Foam is a
    // surface effect, so it lands after the volume terms (caustics/scatter).
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
    // Refraction off keeps the surface TRANSPARENT: the (undistorted) solid
    // bottom must stay visible through the water via the composite alpha
    // blend. Only the refraction distortion/Snell path is disabled above.
    // The Fresnel surface mirror is not volume translucency: a strong mirror
    // (grazing angles) must composite even where the water is thin, or
    // shallows and puddles lose their sky entirely (real puddles mirror!).
    // Top-down views are unaffected (mirrorPresence ≈ 0 there).
    alpha = max(alpha, mirrorPresence);
#ifndef RT_ENABLED
    // The sky mirror is a surface, not volume translucency: keep the Minimal
    // water opaque (the shoreline fade below still dissolves it at the
    // waterline).
    alpha = 1.0;
#endif
    float shoreWidth = max(wp.shoreFadeDepth, 0.0);
    if (thicknessForAlpha > 1e-4 && shoreWidth > 1e-6) {
        alpha *= smoothstep(0.0, shoreWidth, thicknessForAlpha);
    }
    // The shoreline tint fade is also a coverage fade:
    // the last water pixels approaching the border are transparent so the
    // bottom shows with no water color. Only applied where a real depth
    // signal exists (flat unmeasurable water keeps the transparency-floor
    // alpha above); the contact-foam line is re-maxed after, so it survives
    // the fade.
    if (regionDepth > 1e-4) alpha *= tintShoreFade;
    // Shoreline contact foam is a surface line, not volume translucency: it
    // must stay visible where the water meets the solid even when the alpha
    // shoreline fade would otherwise erase the last water pixels.
    if (captureMode) alpha = 1.0;
    outColor = vec4(waterColor, alpha);

    // ── Water aux outputs (color attachments 1 and 2) ──
    // body (RGB) = the refraction + tint BODY (refractedColor, pre-reflection).
    // body weight (A) = composite coverage times the body's share of the
    // final mix, i.e. coverage * (1 - reflection mix). The composite blurs
    // only this body and re-inserts it with this weight, so the reflection
    // lobe, specular highlights, caustics and foam stay sharp while the
    // refracted bottom and its tint soften.
    // column (RG) = measured water depth (m) and this material's blur radius
    // in pixels (0 = crisp). The radius grows with the measured depth up to
    // the per-material cap, so the blur is per water material (layer); it is
    // 0 whenever the material flag OR the global Settings blur toggle is off.
    // WATER_NO_BODY (H4): the single-color-attachment variant has no aux
    // attachments, so only the writes are compiled out.
#ifndef WATER_NO_BODY
    float bodyWeight = (enableReflection
        ? clamp(1.0 - mirrorPresence, 0.0, 1.0)
        : 1.0) * clamp(alpha, 0.0, 1.0);
    float blurPx = enableBlur
        ? clamp(regionDepth * max(wp.blurDepthScale, 0.0), 0.0, max(wp.blurRadius, 0.0))
        : 0.0;
    outWaterBody = vec4(refractedColor, bodyWeight);
    outWaterColumn = vec4(min(max(regionDepth, 0.0), 60000.0), blurPx, 0.0, 0.0);
#endif

    // ── Unified debug views (IDs shared with the solid path) ──
    // Canonical IDs live in includes/debug_modes.glsl (mirror of
    // vulkan/includes/DebugModes.hpp). Solid-only material views (albedo,
    // normal/height maps, roughness, AO, triplanar, UV, tess heat) fall
    // through to the composited water color above; 0 = normal render.
    if (dbgMode == DEBUG_MODE_SHADING_NORMAL) {
        outColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_GEOMETRIC_NORMAL) {
        outColor = vec4(normalize(flatN) * 0.5 + 0.5, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_FACE_NORMAL) {
        vec3 fn = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
        outColor = vec4(fn * 0.5 + 0.5, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_MATERIAL_INDEX) {
        // Water brush/layer id per pixel (golden-ratio hue). MAGENTA = out of
        // the waterParams SSBO range (those pixels fall back to layer 0).
        int nLB = max(waterParams.length(), 1);
        vec3 bidCol;
        if (fragBrushIndex < 0 || fragBrushIndex >= nLB) {
            bidCol = vec3(1.0, 0.0, 1.0);
        } else {
            float hh = fract(float(fragBrushIndex) * 0.61803398875);
            bidCol = clamp(0.5 + 0.5 * cos(6.2831853 * (hh + vec3(0.0, 0.33, 0.67))), 0.0, 1.0);
        }
        outColor = vec4(bidCol, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_N_DOT_L) {
        outColor = vec4(vec3(max(dot(normal, lightDir), 0.0)), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_LIGHT_VECTOR) {
        outColor = vec4(normalize(lightDir) * 0.5 + 0.5, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_SHADOW) {
        // Water never samples the shadow map (see SHADOW ON WATER above):
        // always 0 by design.
        outColor = vec4(vec3(shadow), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_REFLECTION_COLOR) {
        // Reflection color actually used (RT/SSR hit or sky fallback).
        outColor = vec4(skyColor, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_REFLECTION_SOURCE) {
        // Which branch resolved the reflection:
        //   black   = reflection disabled
        //   red     = inline ray hit SOLID terrain (above the waterline)
        //   yellow  = inline ray hit reflected WATER
        //   magenta = guard (below-waterline bed / upward self-water)
        //   green   = SSR near-field hit
        //   blue    = inline ray miss (sky)
        //   cyan    = equirect sky fallback (no inline trace)
        vec3 srcCol = vec3(0.0);
        if (reflSource < 0.5)       srcCol = vec3(0.0);
        else if (reflSource < 1.5)  srcCol = vec3(1.0, 0.0, 0.0);
        else if (reflSource < 2.5)  srcCol = vec3(1.0, 1.0, 0.0);
        else if (reflSource < 3.5)  srcCol = vec3(1.0, 0.0, 1.0);
        else if (reflSource < 4.5)  srcCol = vec3(0.0, 1.0, 0.0);
        else if (reflSource < 5.5)  srcCol = vec3(0.0, 0.0, 1.0);
        else                        srcCol = vec3(0.0, 1.0, 1.0);
        outColor = vec4(srcCol, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_REFLECTION_VECTOR) {
        // Computed right here from the shaded surface instead of reading
        // reflectDir: that variable is filled by the reflection bookkeeping,
        // which the ray-less build compiles out, so the view rendered black on
        // Minimal. viewDir and the wave normal are always valid.
        vec3 nVis = (dot(normal, viewDir) < 0.0) ? -normal : normal;
        outColor = vec4(reflect(-viewDir, nVis) * 0.5 + 0.5, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_FRESNEL) {
        outColor = vec4(vec3(clamp(fresnel, 0.0, 1.0)), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_RAY_MASK) {
        // Ray-query pixel ratio (per lobe): R = reflection inline traced,
        // G = refraction inline traced, B = pipeline hit. Budget-skipped
        // pixels (checker/single-ray/contrib) stay dark; sky fallback is
        // near-black. Masks: 0=disabled, 1=sky, 2=pipe, 3=inline, 4=budget.
        vec3 maskCol = vec3(0.0);
        if (reflMaskDbg > 2.5 && reflMaskDbg < 3.5) maskCol.r = 1.0;
        else if (reflMaskDbg > 1.5 && reflMaskDbg < 2.5) maskCol.b += 0.5;
        if (refrMask > 2.5 && refrMask < 3.5) maskCol.g = 1.0;
        else if (refrMask > 1.5 && refrMask < 2.5) maskCol.b += 0.5;
        outColor = vec4(maskCol, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_SKY_REFLECTION) {
        // Raw sky equirect along the reflection direction — verifies the
        // fallback the water pass uses for RT misses.
        vec3 sc = texture(skyEquirectTex,
            waterDirToEquirectUV(normalize(reflect(-viewDir, normal)))).rgb;
        outColor = vec4(sc, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_REFRACTION_COLOR) {
        // Refracted scene color before the aerial-distance fade.
        outColor = vec4(dbgSceneColor, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_NOISE) {
        // The refracted-ray perturbation as actually applied (tangent frame):
        //   R/G = the offset components (grey 0.5 = zero), B = its magnitude.
        // If this is BLACK the noise is not reaching the ray; if it has colour
        // the perturbation is live and the problem is downstream of the ray.
        outColor = vec4(clamp(refractionOffset.x * 2.0 + 0.5, 0.0, 1.0),
                        clamp(refractionOffset.y * 2.0 + 0.5, 0.0, 1.0),
                        clamp(length(refractionOffset) * 4.0, 0.0, 1.0), 1.0);
        return;
    }
    // ── The single sine swell and its gate, so the wave can be inspected:
    //    the amplitude mask (calm bands), the sine itself and its slope. ──
    if (dbgMode == DEBUG_MODE_WATER_SLOPE) {
        // Along-shore slope of the swell: R = signed (grey 0.5 = crest or
        // trough, where the slope is zero), G = |slope| (1 = max steepness).
        outColor = vec4(0.5 + 0.5 * waveField.swellSlope,
                        clamp(abs(waveField.swellSlope), 0.0, 1.0), 0.0, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_CONTACT) {
        // Shore sine: R = swell profile (signed), G = shoreline contact foam.
        outColor = vec4(0.5 + 0.5 * waveField.swell,
                        clamp(waveField.contact, 0.0, 1.0), 0.0, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_GERSTNER) {
        // The per-pixel Gerstner wave - exactly what the shading normal is
        // built from, evaluated at this pixel's base position and depth.
        //   R = wave height (grey 0.5 = mean level)
        //   G = |analytic slope| (the normal's tilt, 2.0 = white)
        //   B = horizontal pinch (the Gerstner displacement that sharpens
        //       the crests), normalised by the crest height
        float maxExpected = max(fragBasePos.w * max(wp.waveAmplitude, 1e-3), 1e-3);
        outColor = vec4(0.5 + 0.5 * clamp(waveField.height / maxExpected, -1.0, 1.0),
                        clamp(length(waveField.grad) * 0.5, 0.0, 1.0),
                        clamp(length(waveField.disp) / maxExpected, 0.0, 1.0), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_SWELL) {
        // The sine swell, signed (grey 0.5 = mean level):
        // R = profile (sin), G = along-shore slope (cos).
        outColor = vec4(0.5 + 0.5 * waveField.swell,
                        0.5 + 0.5 * waveField.swellSlope, 0.0, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_FOAM_MASK) {
        // R = whitewater coverage, G = shoreline contact line. Black where the
        // layer has foam off or the zone never breaks.
        outColor = vec4(waveField.foam, waveField.contact, 0.0, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_SPECULAR_NOISE) {
        // Highlight noise inside the sun lobe: R = specular perturbation,
        // G = glitter (0.5 = the chain did not run), B = the lobe gate itself.
        // Black means the lobe gated both chains off there (perf report 20 M10).
        outColor = vec4(dbgHighlightNoise, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_SHORE_DIRECTION) {
        // The shore direction: the direction of DECREASING water depth, which
        // the wave field uses as the movement direction of every component -
        // both ridged trains, the chop, the calm mask drift and the foam drift
        // - in EVERY depth region (the zones only change the amplitude).
        //   R/G = dir.x / dir.z packed to 0..1 (same convention as Light Vector)
        //   B   = 1 where the direction came from the measured water-depth
        //         gradient, 0 where it fell back to the configured Shore
        //         Direction angle.
        // The flag is exact: the vertex stage that solved the gradient reports
        // it through fragDebug.z. If B is 0 everywhere, either the water has no
        // measurable depth slope (deep water: a clear solid depth and a clear
        // back face), or Shore Gradient Step is 0, which disables the gradient
        // by design.
        float shoreMeasured = (fragDebug.z > 0.5) ? 1.0 : 0.0;
        outColor = vec4(fragShoreDir.x * 0.5 + 0.5,
                        fragShoreDir.y * 0.5 + 0.5,
                        shoreMeasured, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_DISPLACEMENT) {
        // Prefer the tessellation-provided debug value (fragDebug); fall back
        // to a per-fragment evaluation of the same single wave field so the
        // view works without tessellation.
        float timeDebug = waterRenderUBO.waterTime;
        float bumpAmpDbg = wp.bumpAmplitude;
        float animTimeDbg = timeDebug;
        vec4 waveDbg = waterWaveSample(
            fragPos.xyz, animTimeDbg, waterThickness, bumpAmpDbg, fragShoreDir, wp,
            waveFieldOct, waveDx, waveDy);
        float maxExpected = max(bumpAmpDbg * wp.waveAmplitude, 1e-3);
        float normDisp = clamp((waveDbg.x / maxExpected) * 0.5 + 0.5, 0.0, 1.0);
        // fragDebug.xy carries the displacement; .z carries the shore-direction
        // source and must not leak into this view.
        vec2 debugCol = fragDebug.xy;
        if (length(debugCol) < 0.001) debugCol = vec2(normDisp);
        outColor = vec4(vec3(debugCol.x), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_THICKNESS) {
        // Water column thickness normalized by the per-layer refraction cap.
        float thickN = clamp(waterThickness / max(refrThickCap, 1.0), 0.0, 1.0);
        outColor = vec4(vec3(thickN), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_ABSORPTION) {
        // Beer-Lambert transmittance through the water column.
        outColor = vec4(clamp(transmittance, 0.0, 1.0), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_CAUSTICS) {
        outColor = vec4(vec3(clamp(caustic, 0.0, 1.0)), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_DEPTH_SOURCE) {
        // Water-column source: which branch set this pixel's thickness.
        // grey=raster back-face/no-RT, green=RT inline hit length,
        // cyan=miss continuity (raster bottom), magenta=miss with no raster
        // bottom->thin, blue=pipeline refraction output.
        vec3 dc = vec3(0.25);
        if (depthSource > 0.5 && depthSource < 1.5) dc = vec3(0.0, 1.0, 0.0);
        else if (depthSource < 2.5 && depthSource > 1.5) dc = vec3(0.0, 1.0, 1.0);
        else if (depthSource < 3.5 && depthSource > 2.5) dc = vec3(1.0, 0.0, 1.0);
        else if (depthSource > 4.5) dc = vec3(0.0, 0.0, 1.0);
        outColor = vec4(dc, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_COMPOSE) {
        // R = tintBlend (water tint dominance), G = mirrorPresence
        // (reflection mix), B = thickness / layer cap.
        float thickN = clamp(waterThickness / max(refrThickCap, 1.0), 0.0, 1.0);
        outColor = vec4(clamp(tintBlend, 0.0, 1.0),
                        clamp(mirrorPresence, 0.0, 1.0),
                        thickN, 1.0);
        return;
    }


    if (dbgMode == DEBUG_MODE_WATER_COLOR) {
        // One region: the single water colour.
        outColor = vec4(wp.waterColor, 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_WATER_DEPTH_SOURCES) {
        // Unambiguous classification of the raster sources:
        //   RED    (1,0,0) = solid depth target CLEAR at this pixel
        //   YELLOW (1,1,0) = solid sample ABOVE the water surface (bank)
        //   otherwise      = real data, scaled to 1/4 of the deep zone so it
        //                    can never saturate into a flag color:
        //                    R = solid drop, G = back-face drop, B = final depth
        float zScale = 4.0 * max(wp.shoreWaveFade, 1.0);   // depth display scale
        float sd = textureLod(solidSceneDepthTex, screenUV, 0.0).r;
        if (sd >= 1.0) {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
        vec4 sw = ubo.invViewProjection * vec4(screenUV * 2.0 - 1.0, sd, 1.0);
        float sDropSigned = fragPosWorld.y - sw.y / sw.w;
        if (sDropSigned < 0.0) {
            outColor = vec4(1.0, 1.0, 0.0, 1.0);
            return;
        }
        float bd = textureLod(waterBackDepthTex, screenUV, 0.0).r;
        float bDrop = 0.0;
        if (bd < 1.0) {
            vec4 bw = ubo.invViewProjection * vec4(screenUV * 2.0 - 1.0, bd, 1.0);
            bDrop = max(fragPosWorld.y - bw.y / bw.w, 0.0);
        }
        outColor = vec4(clamp(sDropSigned / zScale, 0.0, 1.0),
                        clamp(bDrop / zScale, 0.0, 1.0),
                        clamp(fragWaterDepth / zScale, 0.0, 1.0), 1.0);
        return;
    }
    if (dbgMode == DEBUG_MODE_SCENE_DEPTH) {
        // Solid scene depth behind the water (linear eye-space / far).
        // White = clear: the opaque pass wrote no terrain at this pixel.
        float sd = textureLod(solidSceneDepthTex, screenUV, 0.0).r;
        float farP = max(ubo.farPlane, 1.0);
        outColor = (sd >= 1.0)
            ? vec4(1.0, 1.0, 1.0, 1.0)
            : vec4(vec3(clamp(linearizeDepth(sd) / farP, 0.0, 1.0)), 1.0);
        return;
    }


    // Final outputs: only write the composited water color (RGBA)
    // Normal/mask outputs removed — they are no longer produced by this pass.


}
