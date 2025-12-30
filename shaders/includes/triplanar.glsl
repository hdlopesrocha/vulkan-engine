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

// Compute triplanar normal by sampling the normal map for each projection and transforming each to world-space
// Build a stable tangent/bitangent basis for a given projection axis (axis must be unit-length)
void buildTBFromAxis(in vec3 axis, out vec3 T, out vec3 B) {
    vec3 up = abs(axis.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    T = normalize(cross(up, axis));
    B = cross(axis, T);
}

// Compute triplanar normal by sampling the normal map for each projection and transforming each to world-space.
// `geomN` is the geometric world-space normal of the fragment (used to preserve axis sign/orientation).
// This variant also attempts to use the fragment TBN (vertex tangent or derivative-based) when available
// to convert the most-significant projection's normal into world-space, reducing seams where appropriate.
vec3 computeTriplanarNormal(in vec3 fragPosWorld, in vec3 triW, in int texIndex, in vec3 geomN, in vec4 fragTangent) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, texIndex, geomN, uvX, uvY, uvZ);

    vec3 nmX = vec3(0.0);
    vec3 nmY = vec3(0.0);
    vec3 nmZ = vec3(0.0);

    // We rely on vertex tangents supplied by the pipeline; build per-axis TB bases

    // Determine dominant projection axis (largest triplanar weight)
    int dominant = 0;
    if (triW.y > triW.x && triW.y >= triW.z) dominant = 1;
    else if (triW.z > triW.x && triW.z > triW.y) dominant = 2;

    if(triW.x > 0.0) {
        vec3 nX = texture(normalArray, vec3(uvX, float(texIndex))).rgb * 2.0 - 1.0;
        nX = normalize(nX);
        // Axis uses same convention as CPU code (pointing opposite to geomN sign as before)
        vec3 axisX = vec3(geomN.x >= 0.0 ? -1.0 : 1.0, 0.0, 0.0);
        // CPU mapping: X projection samples (u,v) = (-z, -y) for +X, (z, -y) for -X
        vec3 uDirX = vec3(0.0, 0.0, -(geomN.x >= 0.0 ? 1.0 : -1.0));
        vec3 vDirX = vec3(0.0, -1.0, 0.0);
        vec3 tX = normalize(uDirX - axisX * dot(axisX, uDirX));
        vec3 bX = normalize(cross(axisX, tX));
        if (dot(bX, vDirX) < 0.0) bX = -bX;
        // Convert sampled normal from projection tangent-space into world-space using the projection basis
        nmX = normalFromNormalMap(nX, tX, bX, axisX);
    }
    if(triW.y > 0.0) {
        vec3 nY = texture(normalArray, vec3(uvY, float(texIndex))).rgb * 2.0 - 1.0;
        nY = normalize(nY);
        vec3 axisY = vec3(0.0, geomN.y >= 0.0 ? -1.0 : 1.0, 0.0);
        // CPU mapping: Y projection samples (u,v) = (x, z) for +Y, (x, -z) for -Y
        vec3 uDirY = vec3(1.0, 0.0, 0.0);
        vec3 vDirY = vec3(0.0, 0.0, (geomN.y >= 0.0 ? 1.0 : -1.0));
        vec3 tY = normalize(uDirY - axisY * dot(axisY, uDirY));
        vec3 bY = normalize(cross(axisY, tY));
        if (dot(bY, vDirY) < 0.0) bY = -bY;
        // Convert sampled normal from projection tangent-space into world-space using the projection basis
        nmY = normalFromNormalMap(nY, tY, bY, axisY);
    }
    if(triW.z > 0.0) {
        vec3 nZ = texture(normalArray, vec3(uvZ, float(texIndex))).rgb * 2.0 - 1.0;
        nZ = normalize(nZ);
        vec3 axisZ = vec3(0.0, 0.0, geomN.z >= 0.0 ? -1.0 : 1.0);
        // CPU mapping: Z projection samples (u,v) = (x, -y) for +Z, (-x, -y) for -Z
        vec3 uDirZ = vec3((geomN.z >= 0.0 ? 1.0 : -1.0), 0.0, 0.0);
        vec3 vDirZ = vec3(0.0, -1.0, 0.0);
        vec3 tZ = normalize(uDirZ - axisZ * dot(axisZ, uDirZ));
        vec3 bZ = normalize(cross(axisZ, tZ));
        if (dot(bZ, vDirZ) < 0.0) bZ = -bZ;
        // Convert sampled normal from projection tangent-space into world-space using the projection basis
        nmZ = normalFromNormalMap(nZ, tZ, bZ, axisZ);
    }


    return normalize(nmX * triW.x + nmY * triW.y + nmZ * triW.z);
}
