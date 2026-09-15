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
