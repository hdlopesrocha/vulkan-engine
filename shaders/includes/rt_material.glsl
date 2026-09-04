// Raster-equivalent material evaluation for ray tracing.
// Mirrors shaders/main.frag + includes/triplanar.glsl + includes/common.glsl +
// includes/tbn.glsl + includes/hsv.glsl + includes/displacement.glsl EXACTLY
// (same formulas, same blend order, same per-corner barycentric weights), so
// ray-traced solids shade like the legacy rasterizer. Two documented
// deviations (both forced, both small):
//   1. No dFdx/dFdy in ray tracing: the non-triplanar normal-map TBN basis is
//      built from a stable up-cross instead of UV derivatives.
//   2. TES displacement is approximated by offsetting the hit point along the
//      smooth normal with the same height/scale formula (traversal stays on
//      the base mesh; micro-scale silhouette differs accordingly).
#ifndef RT_MATERIAL_GLSL
#define RT_MATERIAL_GLSL

// ── Triplanar weights from the GEOMETRIC normal (mirrors computeTriplanarWeights) ──
vec3 rtTriplanarWeights(vec3 geomN) {
    vec3 w = abs(geomN);
    float t = rt.triplanarParams.x;
    vec3 wt = max(vec3(0.0), w - vec3(t));
    float e = max(1.0, rt.triplanarParams.y);
    wt = pow(wt, vec3(e));
    float sum = wt.x + wt.y + wt.z + 1e-6;
    return wt / sum;
}

// ── Per-material triplanar UVs (mirrors computeTriplanarUVs) ──
void rtTriplanarUVs(vec3 worldPos, int matId, vec3 geomN, out vec2 uvX, out vec2 uvY, out vec2 uvZ) {
    vec2 scale = vec2(materials[matId].triplanarParams.x, materials[matId].triplanarParams.y);
    if (geomN.x >= 0.0) uvX = vec2(-worldPos.z, -worldPos.y) * scale;
    else                uvX = vec2( worldPos.z, -worldPos.y) * scale;
    if (geomN.y >= 0.0) uvY = vec2(worldPos.x,  worldPos.z) * scale;
    else                uvY = vec2(worldPos.x, -worldPos.z) * scale;
    if (geomN.z >= 0.0) uvZ = vec2( worldPos.x, -worldPos.y) * scale;
    else                uvZ = vec2(-worldPos.x, -worldPos.y) * scale;
}

// ── Height with per-material flips (mirrors sampleHeight) ──
float rtSampleHeight(vec2 tcIn, int matId) {
    vec2 tc = tcIn;
    if (materials[matId].mappingParams.z > 0.5) tc.y = 1.0 - tc.y;
    if (materials[matId].normalParams.z > 0.5) tc.x = 1.0 - tc.x;
    return clamp(texture(heightArray, vec3(tc, float(matId))).r, 0.0, 1.0);
}

// ── Triplanar height with flips (mirrors sampleHeightTriplanarW) ──
float rtSampleHeightTriplanarW(vec3 worldPos, vec3 normal, vec3 w, int matId) {
    vec2 scale = vec2(materials[matId].triplanarParams.x, materials[matId].triplanarParams.y);
    if (materials[matId].mappingParams.z > 0.5) scale.y = -scale.y;
    if (materials[matId].normalParams.z > 0.5) scale.x = -scale.x;
    float hX = 0.0, hY = 0.0, hZ = 0.0;
    if (w.x > 0.0) {
        vec2 uvX = (normal.x >= 0.0) ? vec2(-worldPos.z, -worldPos.y) : vec2(worldPos.z, -worldPos.y);
        hX = texture(heightArray, vec3(uvX * scale, float(matId))).r;
    }
    if (w.y > 0.0) {
        vec2 uvY = (normal.y >= 0.0) ? vec2(worldPos.x, worldPos.z) : vec2(worldPos.x, -worldPos.z);
        hY = texture(heightArray, vec3(uvY * scale, float(matId))).r;
    }
    if (w.z > 0.0) {
        vec2 uvZ = (normal.z >= 0.0) ? vec2(worldPos.x, -worldPos.y) : vec2(-worldPos.x, -worldPos.y);
        hZ = texture(heightArray, vec3(uvZ * scale, float(matId))).r;
    }
    return clamp(hX * w.x + hY * w.y + hZ * w.z, 0.0, 1.0);
}

// ── Normal convention (mirrors applyNormalConvention) ──
vec3 rtApplyNormalConvention(vec3 n, vec4 normalParams) {
    vec3 nn = n;
    if (normalParams.x > 0.5) nn.y = -nn.y;
    if (normalParams.y > 0.5) nn = vec3(nn.z, nn.y, nn.x);
    return nn;
}

// ── Single-projection world normal (mirrors computeProjectionNormal) ──
vec3 rtProjectionNormal(vec2 uv, int matId, vec3 surfaceN) {
    vec3 nSample = texture(normalArray, vec3(uv, float(matId))).rgb * 2.0 - 1.0;
    nSample = normalize(rtApplyNormalConvention(nSample, materials[matId].normalParams));
    vec3 axis = surfaceN;
    vec3 up = abs(axis.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, axis));
    vec3 B = cross(axis, T);
    return normalize(nSample.x * T + nSample.y * B + nSample.z * surfaceN);
}

// ── Triplanar world normal per corner (mirrors computeTriplanarNormalUVs) ──
vec3 rtTriplanarNormalUVs(vec3 triW, int matId, vec3 surfaceN, vec2 uvX, vec2 uvY, vec2 uvZ) {
    float wsum = triW.x + triW.y + triW.z + 1e-6;
    vec3 w = triW / wsum;
    vec3 nmX = triW.x > 0.0 ? rtProjectionNormal(uvX, matId, surfaceN) : vec3(0.0);
    vec3 nmY = triW.y > 0.0 ? rtProjectionNormal(uvY, matId, surfaceN) : vec3(0.0);
    vec3 nmZ = triW.z > 0.0 ? rtProjectionNormal(uvZ, matId, surfaceN) : vec3(0.0);
    return normalize(nmX * w.x + nmY * w.y + nmZ * w.z);
}

// ── TES displacement approximation (mirrors applyDisplacement + the global
// tessellation gate in main.tese: mappingFlag *= passParams.y): same height
// sampling, same (h-0.5)*scale, along the smooth normal. Returns the displaced
// point and the signed offset (worldPos = basePos + N*disp) so ray origins
// can be lifted clear of the traversed base mesh on both sides.
// Without the global gate, RT would bump shade (and bump self-shadow) while
// the rasterizer renders flat.
vec3 rtDisplace(vec3 worldPos, vec3 N, vec2 uv, ivec3 brush, vec3 bary, vec3 geomN,
                out float disp) {
    disp = 0.0;
    int n = int(materials.length());
    ivec3 ids = ivec3(n > 0 ? clamp(brush.x, 0, n - 1) : 0,
                      n > 0 ? clamp(brush.y, 0, n - 1) : 0,
                      n > 0 ? clamp(brush.z, 0, n - 1) : 0);
    float mapBlend = materials[ids.x].mappingParams.x * bary.x
                   + materials[ids.y].mappingParams.x * bary.y
                   + materials[ids.z].mappingParams.x * bary.z;
    if (mapBlend * rt.featureToggles.w <= 0.5) return worldPos;
    float triFlag = materials[ids.x].triplanarParams.z * bary.x
                  + materials[ids.y].triplanarParams.z * bary.y
                  + materials[ids.z].triplanarParams.z * bary.z;
    vec3 tw = rtTriplanarWeights(N);
    float h0 = 0.5, h1 = 0.5, h2 = 0.5;
    if (materials[ids.x].mappingParams.x > 0.5)
        h0 = (materials[ids.x].triplanarParams.z > 0.5 || triFlag > 0.5)
             ? rtSampleHeightTriplanarW(worldPos, N, tw, ids.x) : rtSampleHeight(uv, ids.x);
    if (materials[ids.y].mappingParams.x > 0.5)
        h1 = (materials[ids.y].triplanarParams.z > 0.5 || triFlag > 0.5)
             ? rtSampleHeightTriplanarW(worldPos, N, tw, ids.y) : rtSampleHeight(uv, ids.y);
    if (materials[ids.z].mappingParams.x > 0.5)
        h2 = (materials[ids.z].triplanarParams.z > 0.5 || triFlag > 0.5)
             ? rtSampleHeightTriplanarW(worldPos, N, tw, ids.z) : rtSampleHeight(uv, ids.z);
    float h = h0 * bary.x + h1 * bary.y + h2 * bary.z;
    float scale = materials[ids.x].mappingParams.w * bary.x
                + materials[ids.y].mappingParams.w * bary.y
                + materials[ids.z].mappingParams.w * bary.z;
    disp = (h - 0.5) * scale;
    return worldPos + N * disp;
}

// Ray origin lifted clear of the traversed base mesh. The shading point sits
// at basePos + N*disp (disp may be negative = inside the mesh), so a plain
// epsilon can start under the surface and self-hit. Lifts to basePos +
// N*eps on the outside in all cases.
vec3 rtSurfaceOrigin(vec3 worldPos, vec3 N, float disp) {
    return worldPos + N * (RAY_ORIGIN_EPS + max(-disp, 0.0));
}

// ── HSV converts (mirror includes/hsv.glsl) ──
vec3 rtHsvToRgb(vec3 hsv) {
    vec3 c = clamp(hsv, vec3(0.0), vec3(360.0, 1.0, 1.0));
    float h = c.x / 60.0;
    float s = c.y;
    float v = c.z;
    float hi = floor(h);
    float f = h - hi;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    int i = int(hi) % 6;
    if (i == 0) return vec3(v, t, p);
    if (i == 1) return vec3(q, v, p);
    if (i == 2) return vec3(p, v, t);
    if (i == 3) return vec3(p, q, v);
    if (i == 4) return vec3(t, p, v);
    return vec3(v, p, q);
}

vec3 rtRgbToHsv(vec3 rgb) {
    vec3 c = rgb;
    float cmax = max(c.r, max(c.g, c.b));
    float cmin = min(c.r, min(c.g, c.b));
    float delta = cmax - cmin;
    float v = cmax;
    float s = cmax == 0.0 ? 0.0 : delta / cmax;
    float h = 0.0;
    if (delta > 0.0001) {
        if (cmax == c.r) h = 60.0 * mod((c.g - c.b) / delta, 6.0);
        else if (cmax == c.g) h = 60.0 * ((c.b - c.r) / delta + 2.0);
        else h = 60.0 * ((c.r - c.g) / delta + 4.0);
        if (h < 0.0) h += 360.0;
    }
    return vec3(h, s, v);
}

#endif // RT_MATERIAL_GLSL
