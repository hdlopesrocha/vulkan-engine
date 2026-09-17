// Real-geometry reflection sample: replicate the raster's terrain texture
// lookup byte-for-byte (main.tesc unique-slot compression + fragment
// barycentric weights + per-material triplanar/UV branch), so reflected
// ground cover matches what the direct view paints at every triangle.
//
// Requires: rtSceneVerts (float pool, Vertex stride 16 floats: pos 0-2,
// uv 6-7, normal 8-10, brushIndex 11 int bits), albedoArray, materials[],
// computeTriplanarAlbedoLod.

vec3 rtSceneSampleReflectionAlbedo(uint i0, uint i1, uint i2, vec2 bary,
                                   vec3 hitPos, vec3 hitN, int maxLayer) {
    const uint kVertStride = 16u;
    // Corner brush indices (int bit patterns in the float pool).
    int bm0 = floatBitsToInt(rtSceneVerts[i0 * kVertStride + 11u]);
    int bm1 = floatBitsToInt(rtSceneVerts[i1 * kVertStride + 11u]);
    int bm2 = floatBitsToInt(rtSceneVerts[i2 * kVertStride + 11u]);
    // Unique-slot compression, identical to main.tesc.
    int um0 = (bm0 >= 0) ? bm0 : -1;
    int um1 = -1;
    int um2 = -1;
    if (bm1 >= 0 && bm1 != um0) um1 = bm1;
    if (bm2 >= 0 && bm2 != um0 && bm2 != um1) um2 = bm2;
    vec3 sw0 = (bm0 < 0) ? vec3(0.0) : ((bm0 == um0) ? vec3(1, 0, 0) : ((bm0 == um1) ? vec3(0, 1, 0) : vec3(0, 0, 1)));
    vec3 sw1 = (bm1 < 0) ? vec3(0.0) : ((bm1 == um0) ? vec3(1, 0, 0) : ((bm1 == um1) ? vec3(0, 1, 0) : vec3(0, 0, 1)));
    vec3 sw2 = (bm2 < 0) ? vec3(0.0) : ((bm2 == um0) ? vec3(1, 0, 0) : ((bm2 == um1) ? vec3(0, 1, 0) : vec3(0, 0, 1)));
    vec3 blendW = sw0 * (1.0 - bary.x - bary.y) + sw1 * bary.x + sw2 * bary.y;

    // Interpolated UV for the UV-mapped material path (same for all slots,
    // like main.frag's `uv`).
    vec2 uv0 = vec2(rtSceneVerts[i0 * kVertStride + 6u], rtSceneVerts[i0 * kVertStride + 7u]);
    vec2 uv1 = vec2(rtSceneVerts[i1 * kVertStride + 6u], rtSceneVerts[i1 * kVertStride + 7u]);
    vec2 uv2 = vec2(rtSceneVerts[i2 * kVertStride + 6u], rtSceneVerts[i2 * kVertStride + 7u]);
    vec2 uvInt = uv0 * (1.0 - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;

    vec3 acc = vec3(0.0);
    float accW = 0.0;
    // Slot 0
    if (um0 >= 0 && blendW.x > 1e-5) {
        int m = clamp(um0, 0, maxLayer);
        vec3 c = (materials[m].triplanarParams.z > 0.5)
            ? computeTriplanarAlbedoLod(hitPos, abs(hitN), m, hitN, 0.0)
            : textureLod(albedoArray, vec3(uvInt, float(m)), 0.0).rgb;
        acc += c * blendW.x;
        accW += blendW.x;
    }
    // Slot 1
    if (um1 >= 0 && blendW.y > 1e-5) {
        int m = clamp(um1, 0, maxLayer);
        vec3 c = (materials[m].triplanarParams.z > 0.5)
            ? computeTriplanarAlbedoLod(hitPos, abs(hitN), m, hitN, 0.0)
            : textureLod(albedoArray, vec3(uvInt, float(m)), 0.0).rgb;
        acc += c * blendW.y;
        accW += blendW.y;
    }
    // Slot 2
    if (um2 >= 0 && blendW.z > 1e-5) {
        int m = clamp(um2, 0, maxLayer);
        vec3 c = (materials[m].triplanarParams.z > 0.5)
            ? computeTriplanarAlbedoLod(hitPos, abs(hitN), m, hitN, 0.0)
            : textureLod(albedoArray, vec3(uvInt, float(m)), 0.0).rgb;
        acc += c * blendW.z;
        accW += blendW.z;
    }
    if (accW <= 1e-5) {
        // All-underground corner case: raster falls back to material 0.
        return textureLod(albedoArray, vec3(uvInt, 0.0), 0.0).rgb;
    }
    return acc / max(accW, 1e-5);
}

// Hit-point displacement refinement: the scene BLAS holds the UNDISPLACED
// base mesh while the raster draws the TES-displaced surface ((h - 0.5) *
// scale along the vertex normal, up to +/-16 m for rock materials). Shading
// the raw ray hit therefore prints the smooth low-poly ghost of the terrain
// (distorted geometry) and samples triplanar albedo meters away from the
// raster's sample points (distorted/stretched UVs). This replicates
// main.tese's applyDisplacement (see displacement.glsl) at the hit
// barycentrics so albedo/shadow lookups land on the rendered surface.
// hitN must be the UNFLIPPED interpolated vertex normal (TES displaces along
// it, not along the ray-facing copy). Slot compression, weights and UVs
// follow main.tesc / rtSceneSampleReflectionAlbedo above. Returns the
// world-space offset to ADD to hitPos (zero when mapping is disabled or the
// scale is negligible); lod is the explicit height-mip level.
// Requires: rtSceneVerts, materials[], ubo, sampleHeightLod,
// sampleHeightTriplanarWLod, computeTriplanarWeights.
vec3 rtDisplaceHit(uint i0, uint i1, uint i2, vec2 bary, vec3 hitPos, vec3 hitN, float lod) {
    const uint kVertStride = 16u;
    int bm0 = floatBitsToInt(rtSceneVerts[i0 * kVertStride + 11u]);
    int bm1 = floatBitsToInt(rtSceneVerts[i1 * kVertStride + 11u]);
    int bm2 = floatBitsToInt(rtSceneVerts[i2 * kVertStride + 11u]);
    // Unique-slot compression, identical to main.tesc (TES then clamps empty
    // slots to material 0 with zero weight — replicated below via um/max).
    int um0 = (bm0 >= 0) ? bm0 : -1;
    int um1 = -1;
    int um2 = -1;
    if (bm1 >= 0 && bm1 != um0) um1 = bm1;
    if (bm2 >= 0 && bm2 != um0 && bm2 != um1) um2 = bm2;
    vec3 sw0 = (bm0 < 0) ? vec3(0.0) : ((bm0 == um0) ? vec3(1, 0, 0) : ((bm0 == um1) ? vec3(0, 1, 0) : vec3(0, 0, 1)));
    vec3 sw1 = (bm1 < 0) ? vec3(0.0) : ((bm1 == um0) ? vec3(1, 0, 0) : ((bm1 == um1) ? vec3(0, 1, 0) : vec3(0, 0, 1)));
    vec3 sw2 = (bm2 < 0) ? vec3(0.0) : ((bm2 == um0) ? vec3(1, 0, 0) : ((bm2 == um1) ? vec3(0, 1, 0) : vec3(0, 0, 1)));
    vec3 blendW = sw0 * (1.0 - bary.x - bary.y) + sw1 * bary.x + sw2 * bary.y;
    // TES: texIndices = max(corner0 slots, 0); weights zero out empty slots.
    ivec3 texIndices = max(ivec3(um0, um1, um2), ivec3(0));

    vec2 uv0 = vec2(rtSceneVerts[i0 * kVertStride + 6u], rtSceneVerts[i0 * kVertStride + 7u]);
    vec2 uv1 = vec2(rtSceneVerts[i1 * kVertStride + 6u], rtSceneVerts[i1 * kVertStride + 7u]);
    vec2 uv2 = vec2(rtSceneVerts[i2 * kVertStride + 6u], rtSceneVerts[i2 * kVertStride + 7u]);
    vec2 uvInt = uv0 * (1.0 - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;

    // Blended triplanar flag (weighted by barycentric weights).
    float triFlag = materials[texIndices.x].triplanarParams.z * blendW.x +
                    materials[texIndices.y].triplanarParams.z * blendW.y +
                    materials[texIndices.z].triplanarParams.z * blendW.z;

    // Per-material heights (default neutral 0.5 when mapping is disabled).
    vec3 triW = computeTriplanarWeights(hitN);
    float h0 = 0.5;
    float h1 = 0.5;
    float h2 = 0.5;
    if (materials[texIndices.x].mappingParams.x > 0.5) {
        if (materials[texIndices.x].triplanarParams.z > 0.5 || triFlag > 0.5) {
            h0 = sampleHeightTriplanarWLod(hitPos, hitN, triW, texIndices.x, lod);
        } else {
            h0 = sampleHeightLod(uvInt, texIndices.x, lod);
        }
    }
    if (materials[texIndices.y].mappingParams.x > 0.5) {
        if (materials[texIndices.y].triplanarParams.z > 0.5 || triFlag > 0.5) {
            h1 = sampleHeightTriplanarWLod(hitPos, hitN, triW, texIndices.y, lod);
        } else {
            h1 = sampleHeightLod(uvInt, texIndices.y, lod);
        }
    }
    if (materials[texIndices.z].mappingParams.x > 0.5) {
        if (materials[texIndices.z].triplanarParams.z > 0.5 || triFlag > 0.5) {
            h2 = sampleHeightTriplanarWLod(hitPos, hitN, triW, texIndices.z, lod);
        } else {
            h2 = sampleHeightLod(uvInt, texIndices.z, lod);
        }
    }

    float h = h0 * blendW.x + h1 * blendW.y + h2 * blendW.z;
    float scale = materials[texIndices.x].mappingParams.w * blendW.x +
                  materials[texIndices.y].mappingParams.w * blendW.y +
                  materials[texIndices.z].mappingParams.w * blendW.z;
    if (abs(scale) < 1e-6) return vec3(0.0);
    // Global mapping toggle, exactly like main.tese.
    float mappingFlag = (materials[texIndices.x].mappingParams.x * blendW.x +
                         materials[texIndices.y].mappingParams.x * blendW.y +
                         materials[texIndices.z].mappingParams.x * blendW.z) * ubo.passParams.y;
    if (mappingFlag <= 0.5) return vec3(0.0);

    float displacement = (h - 0.5) * scale;
    return hitN * displacement;
}
