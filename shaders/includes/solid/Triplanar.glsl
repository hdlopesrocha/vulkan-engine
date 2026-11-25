// Triplanar projection utilities for albedo and normal mapping

#include "../TextureMapping.glsl"

// Compute triplanar UVs from world-space position. `scale` is the per-vertex UV
// scale stored in Vertex::texCoord (painted by the scene's TextureCoordPainter).
// Uses the geometric normal sign to match CPU-side orientation. The per-axis
// projection itself lives in TextureMapping.glsl (shared with the single-axis
// mapping modes).
void computeTriplanarUVs(in vec3 fragPosWorld, in vec3 geomN, in vec2 scale, out vec2 uvX, out vec2 uvY, out vec2 uvZ) {
    uvX = textureMappingUV(TEXTURE_AXIS_X, fragPosWorld, geomN, scale);
    uvY = textureMappingUV(TEXTURE_AXIS_Y, fragPosWorld, geomN, scale);
    uvZ = textureMappingUV(TEXTURE_AXIS_Z, fragPosWorld, geomN, scale);
}

// Sample albedo using precomputed triplanar UVs (weights already blended).
// The zero-weight guards (triW.x > 0.0) skip unused projections.
vec3 computeTriplanarAlbedoUVs(in vec3 triW, in int brushIndex, in vec2 uvX, in vec2 uvY, in vec2 uvZ) {
    vec3 cX = triW.x > 0.0 ? texture(albedoArray, vec3(uvX, float(brushLayer(brushIndex)))).rgb : vec3(0.0);
    vec3 cY = triW.y > 0.0 ? texture(albedoArray, vec3(uvY, float(brushLayer(brushIndex)))).rgb : vec3(0.0);
    vec3 cZ = triW.z > 0.0 ? texture(albedoArray, vec3(uvZ, float(brushLayer(brushIndex)))).rgb : vec3(0.0);
   
    return cX * triW.x + cY * triW.y + cZ * triW.z;
}

// Helper: compute normal from a single projection with given tangent basis
// (factored for the C3 blended path; identical operations, identical order:
// the Y-flip / XZ-swap convention ops are norm-preserving, so
// normalize-then-convert equals the original convert-then-normalize).
vec3 sampleNormalTexel(vec2 uv, int brushIndex) {
    vec3 nSample = texture(normalArray, vec3(uv, float(brushLayer(brushIndex)))).rgb * 2.0 - 1.0;
    return applyNormalConvention(normalize(nSample), materials[brushIndex].normalParams);
}

// Projection-independent basis transform shared by the per-projection and
// blended normal paths.
vec3 transformProjectionNormal(vec3 nSample, vec3 surfaceN) {
    vec3 axis = surfaceN;
    vec3 up = abs(axis.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, axis));
    vec3 B = cross(axis, T);
    return normalize(nSample.x * T + nSample.y * B + nSample.z * surfaceN);
}

vec3 computeProjectionNormal(vec2 uv, int brushIndex, vec3 surfaceN) {
    return transformProjectionNormal(sampleNormalTexel(uv, brushIndex), surfaceN);
}

// Compute triplanar normal by blending the normal-map TEXELS first, then a
// single basis transform (perf report 21 C3). One T/B build and
// two normalizes instead of three + one; the three texel fetches remain
// (roughness/AO below are where fetches drop). The basis is
// projection-independent, so no dominant-projection choice is needed.
vec3 computeTriplanarNormalBlended(in vec3 triW, in int brushIndex, in vec3 surfaceN, in vec2 uvX, in vec2 uvY, in vec2 uvZ) {
    // Normalize incoming triplanar weights and use them directly for blending
    float wsum = triW.x + triW.y + triW.z + 1e-6;
    vec3 w = triW / wsum;

    vec3 t = vec3(0.0);
    if (triW.x > 0.0) t += sampleNormalTexel(uvX, brushIndex) * w.x;
    if (triW.y > 0.0) t += sampleNormalTexel(uvY, brushIndex) * w.y;
    if (triW.z > 0.0) t += sampleNormalTexel(uvZ, brushIndex) * w.z;
    return transformProjectionNormal(normalize(t), surfaceN);
}

// Single-projection roughness (perf report 21 C3): roughness is a
// low-frequency signal that needs no per-projection sampling. One fetch on
// the dominant (max-weight) projection, selected branchlessly so all lanes
// stay converged. Ties fall back to X, then Y — deterministic.
float computeTriplanarRoughnessDominant(in vec3 triW, in int brushIndex, in vec2 uvX, in vec2 uvY, in vec2 uvZ) {
    float dx = step(triW.y, triW.x) * step(triW.z, triW.x);
    float dy = (1.0 - dx) * step(triW.z, triW.y);
    vec2 uv = uvX * dx + uvY * dy + uvZ * (1.0 - dx - dy);
    return texture(roughnessArray, vec3(uv, float(brushLayer(brushIndex)))).r;
}

// Single-projection AO (perf report 21 C3): same dominant-projection
// argument as roughness above; AO modulates ambient only.
float computeTriplanarAODominant(in vec3 triW, in int brushIndex, in vec2 uvX, in vec2 uvY, in vec2 uvZ) {
    float dx = step(triW.y, triW.x) * step(triW.z, triW.x);
    float dy = (1.0 - dx) * step(triW.z, triW.y);
    vec2 uv = uvX * dx + uvY * dy + uvZ * (1.0 - dx - dy);
    return texture(aoArray, vec3(uv, float(brushLayer(brushIndex)))).r;
}

