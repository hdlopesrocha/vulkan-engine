// Triplanar projection utilities for albedo and normal mapping

// Compute triplanar UVs from world-space position using triplanarParams.x/y as scale
void computeTriplanarUVs(in vec3 fragPosWorld, in int texIndex, out vec2 uvX, out vec2 uvY, out vec2 uvZ) {
    uvX = fragPosWorld.yz * vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
    uvY = fragPosWorld.xz * vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
    uvZ = fragPosWorld.xy * vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
}

// Sample albedo using triplanar blending weights
vec3 computeTriplanarAlbedo(in vec3 fragPosWorld, in vec3 triW, in int texIndex) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, texIndex, uvX, uvY, uvZ);
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
vec3 computeTriplanarNormal(in vec3 fragPosWorld, in vec3 triW, in int texIndex, in vec3 geomN) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, texIndex, uvX, uvY, uvZ);

    vec3 nmX = vec3(0.0);
    vec3 nmY = vec3(0.0);
    vec3 nmZ = vec3(0.0);

    if(triW.x > 0.0) {
        vec3 tX, bX;
        vec3 nX = texture(normalArray, vec3(uvX, float(texIndex))).rgb * 2.0 - 1.0;
        nX = normalize(nX);
        if (nX.z < 0.0) nX = -nX;
        vec3 axisX = vec3(geomN.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
        buildTBFromAxis(axisX, tX, bX);
        nmX = normalFromNormalMap(nX, tX, bX, axisX);
    }
    if(triW.y > 0.0) {
        vec3 tY, bY;
        vec3 nY = texture(normalArray, vec3(uvY, float(texIndex))).rgb * 2.0 - 1.0;
        nY = normalize(nY);
        if (nY.z < 0.0) nY = -nY;
        vec3 axisY = vec3(0.0, geomN.y >= 0.0 ? 1.0 : -1.0, 0.0);
        buildTBFromAxis(axisY, tY, bY);
        nmY = normalFromNormalMap(nY, tY, bY, axisY);
    }
    if(triW.z > 0.0) {
        vec3 tZ, bZ;
        vec3 nZ = texture(normalArray, vec3(uvZ, float(texIndex))).rgb * 2.0 - 1.0;
        nZ = normalize(nZ);
        if (nZ.z < 0.0) nZ = -nZ;
        vec3 axisZ = vec3(0.0, 0.0, geomN.z >= 0.0 ? 1.0 : -1.0);
        buildTBFromAxis(axisZ, tZ, bZ);
        nmZ = normalFromNormalMap(nZ, tZ, bZ, axisZ);
    }


    return normalize(nmX * triW.x + nmY * triW.y + nmZ * triW.z);
}
