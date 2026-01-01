// Triplanar projection utilities for albedo and normal mapping

// Compute triplanar UVs from world-space position using triplanarParams.x/y as scale
// Compute triplanar UVs from world-space position using triplanarParams.x/y as scale.
// Uses the geometric normal sign to match CPU-side orientation.
void computeTriplanarUVs(in vec3 fragPosWorld, in int texIndex, in vec3 geomN, out vec2 uvX, out vec2 uvY, out vec2 uvZ) {
    vec2 scale = vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
    // X projection -> sample YZ. Match CPU mapping: positive X => (-z, -y), negative X => (z, -y)
    if (geomN.x >= 0.0) uvX = vec2(-fragPosWorld.z, -fragPosWorld.y) * scale;
    else               uvX = vec2( fragPosWorld.z, -fragPosWorld.y) * scale;

    // Y projection -> sample XZ. CPU mapping: positive Y => (x, z), negative Y => (x, -z)
    if (geomN.y >= 0.0) uvY = vec2(fragPosWorld.x,  fragPosWorld.z) * scale;
    else               uvY = vec2(fragPosWorld.x, -fragPosWorld.z) * scale;

    // Z projection -> sample XY. CPU mapping: positive Z => (x, -y), negative Z => (-x, -y)
    if (geomN.z >= 0.0) uvZ = vec2( fragPosWorld.x, -fragPosWorld.y) * scale;
    else               uvZ = vec2(-fragPosWorld.x, -fragPosWorld.y) * scale;
}

// Sample albedo using triplanar blending weights
vec3 computeTriplanarAlbedo(in vec3 fragPosWorld, in vec3 triW, in int texIndex, in vec3 geomN) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, texIndex, geomN, uvX, uvY, uvZ);
    vec3 cX = triW.x > 0.0 ? texture(albedoArray, vec3(uvX, float(texIndex))).rgb : vec3(0.0);
    vec3 cY = triW.y > 0.0 ? texture(albedoArray, vec3(uvY, float(texIndex))).rgb : vec3(0.0);
    vec3 cZ = triW.z > 0.0 ? texture(albedoArray, vec3(uvZ, float(texIndex))).rgb : vec3(0.0);
   
    return cX * triW.x + cY * triW.y + cZ * triW.z;
}

// Reoriented Normal Mapping (RNM) helper: reorient a blended world-space normal
// toward the geometric normal while preserving surface detail.
vec3 reorientNormal(in vec3 blendedWorld, in vec3 geomN) {
    vec3 b = normalize(blendedWorld);
    float d = dot(b, geomN);
    return normalize(b + geomN * (1.0 - d));
}

// Helper: compute normal from a single projection with given tangent basis
vec3 computeProjectionNormal(vec2 uv, int texIndex, vec3 surfaceN) {
    vec3 nSample = texture(normalArray, vec3(uv, float(texIndex))).rgb * 2.0 - 1.0;
    nSample = normalize(applyNormalConvention(nSample, materials[texIndex].normalParams));
    vec3 axis = surfaceN;
    vec3 up = abs(axis.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, axis));
    vec3 B = cross(axis, T);
    return normalize(nSample.x * T + nSample.y * B + nSample.z * surfaceN);
}

// Compute triplanar normal by sampling the normal map for each projection and transforming each to world-space.
// `geomN` is the geometric world-space normal (for UV computation), `surfaceN` is the smooth interpolated normal (TBN base).
vec3 computeTriplanarNormal(in vec3 fragPosWorld, in vec3 triW, in int texIndex, in vec3 geomN, in vec3 surfaceN) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, texIndex, geomN, uvX, uvY, uvZ);
    vec3 nmX = vec3(0.0);
    vec3 nmY = vec3(0.0);
    vec3 nmZ = vec3(0.0);

    // Normalize incoming triplanar weights and use them directly for blending
    float wsum = triW.x + triW.y + triW.z + 1e-6;
    vec3 w = triW / wsum;

    // X projection
    if (triW.x > 0.0) {
        nmX = computeProjectionNormal(uvX, texIndex, surfaceN);
    }

    // Y projection
    if (triW.y > 0.0) {
        nmY = computeProjectionNormal(uvY, texIndex, surfaceN);
    }

    // Z projection
    if (triW.z > 0.0) {
        nmZ = computeProjectionNormal(uvZ, texIndex, surfaceN);
    }

    // Blend the per-projection world-space normals using the normalized triW weights
    return normalize(nmX * w.x + nmY * w.y + nmZ * w.z);
}

vec4 computeTriplanarTangent(in ivec3 texIndices, in vec3 matWeights, in vec3 geomN, in vec3 triW, in vec3 fragPosWorld, in vec2 fragUV, in vec3 N, out vec3 T, out vec3 B) {
    // Calculate tangent and bitangent from UV derivatives
    vec3 dpdx = dFdx(fragPosWorld);
    vec3 dpdy = dFdy(fragPosWorld);
    vec2 duvdx = dFdx(fragUV);
    vec2 duvdy = dFdy(fragUV);

    // Tangent: direction of increasing U
    T = normalize(duvdx.x * dpdx + duvdy.x * dpdy);
    // Bitangent: direction of increasing V
    B = normalize(duvdx.y * dpdx + duvdy.y * dpdy);

    // Safeguard against invalid vectors (e.g., if derivatives are zero)
    if (T.x != T.x || T.y != T.y || T.z != T.z || length(T) < 1e-6) {
        vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        T = normalize(cross(up, N));
    }
    if (B.x != B.x || B.y != B.y || B.z != B.z || length(B) < 1e-6) {
        B = normalize(cross(N, T));
    }

    // Ensure T and B are orthogonal to N
    T = normalize(T - dot(T, N) * N);
    B = normalize(B - dot(B, N) * N);

    // Compute handedness
    float handed = sign(dot(cross(N, T), B));
    if (handed == 0.0 || handed != handed) handed = 1.0;
    return vec4(T, handed);
}
