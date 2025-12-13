// Triplanar projection utilities for albedo and normal mapping

// Compute triplanar UVs from world-space position using triplanarParams.x/y as scale
void computeTriplanarUVs(in vec3 fragPosWorld, out vec2 uvX, out vec2 uvY, out vec2 uvZ) {
    uvX = fragPosWorld.yz * vec2(ubo.triplanarParams.x, ubo.triplanarParams.y);
    uvY = fragPosWorld.xz * vec2(ubo.triplanarParams.x, ubo.triplanarParams.y);
    uvZ = fragPosWorld.xy * vec2(ubo.triplanarParams.x, ubo.triplanarParams.y);
}

// Sample albedo using triplanar blending weights
vec3 computeTriplanarAlbedo(in vec3 fragPosWorld, in vec3 triW, in int texIndex) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, uvX, uvY, uvZ);
    vec3 cX = texture(albedoArray, vec3(uvX, float(texIndex))).rgb;
    vec3 cY = texture(albedoArray, vec3(uvY, float(texIndex))).rgb;
    vec3 cZ = texture(albedoArray, vec3(uvZ, float(texIndex))).rgb;
    return cX * triW.x + cY * triW.y + cZ * triW.z;
}

// Compute triplanar normal by sampling the normal map for each projection and transforming each to world-space
vec3 computeTriplanarNormal(in vec3 fragPosWorld, in vec3 triW, in int texIndex) {
    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(fragPosWorld, uvX, uvY, uvZ);
    vec3 nX = texture(normalArray, vec3(uvX, float(texIndex))).rgb * 2.0 - 1.0;
    vec3 nY = texture(normalArray, vec3(uvY, float(texIndex))).rgb * 2.0 - 1.0;
    vec3 nZ = texture(normalArray, vec3(uvZ, float(texIndex))).rgb * 2.0 - 1.0;
    vec3 tX = vec3(0.0, 1.0, 0.0);
    vec3 bX = vec3(0.0, 0.0, 1.0);
    vec3 nmX = normalFromNormalMap(nX, tX, bX, vec3(1.0, 0.0, 0.0));
    vec3 tY = vec3(1.0, 0.0, 0.0);
    vec3 bY = vec3(0.0, 0.0, 1.0);
    vec3 nmY = normalFromNormalMap(nY, tY, bY, vec3(0.0, 1.0, 0.0));
    vec3 tZ = vec3(1.0, 0.0, 0.0);
    vec3 bZ = vec3(0.0, 1.0, 0.0);
    vec3 nmZ = normalFromNormalMap(nZ, tZ, bZ, vec3(0.0, 0.0, 1.0));
    return normalize(nmX * triW.x + nmY * triW.y + nmZ * triW.z);
}
