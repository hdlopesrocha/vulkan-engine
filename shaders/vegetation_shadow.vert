#version 450

#include "ubo/UniformObjectNamed.glsl"
#include "ubo/WindParamsNamed.glsl"

// Vegetation EVSM shadow vertex shader.
// Expands 24-corner billboard mesh into world space.
// Only outputs world position for the fragment shader.

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inLocalPos;
layout(location = ATTR_COLOR) in vec3 inLocalTangent;
layout(location = ATTR_BRUSH_INDEX) in int inCornerNormalData;
layout(location = ATTR_INSTANCE) in vec4 instanceData;
layout(location = ATTR_VEG_NORMAL) in vec3 inInstanceNormal; // surface normal: tilt frame onto it (matches vegetation.vert)
// Baked height scale (binding 2, perf report 22 C2/H4); see vegetation.vert.
layout(location = ATTR_VEG_AUX) in float inBakedHeight;

layout(location = VARY_POSWORLD) out vec3 outWorldPos;

layout(set = 2, binding = 0) uniform WindParamsUBO {
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
} windParamsPacked;


WindParamsNamed windParamsNamed() {
    WindParamsNamed n;
    n.windDirection = windParamsPacked.windDirAndStrength.xz;
    n.windStrength = windParamsPacked.windDirAndStrength.w;
    n.windBaseFrequency = windParamsPacked.windNoise.x;
    n.windSpeed = windParamsPacked.windNoise.y;
    n.gustFrequency = windParamsPacked.windNoise.z;
    n.gustStrength = windParamsPacked.windNoise.w;
    n.skewAmount = windParamsPacked.windShape.x;
    n.trunkStiffness = windParamsPacked.windShape.y;
    n.noiseScale = windParamsPacked.windShape.z;
    n.verticalFlutter = windParamsPacked.windShape.w;
    n.turbulence = windParamsPacked.windTurbulence.x;
    n.densityEnabled = windParamsPacked.densityParams.x > 0.5;
    n.nearDistance = windParamsPacked.densityParams.y;
    n.farDistance = windParamsPacked.densityParams.z;
    n.minFactor = windParamsPacked.densityParams.w;
    n.cameraPosition = windParamsPacked.cameraPosAndFalloff.xyz;
    n.densityFalloff = windParamsPacked.cameraPosAndFalloff.w;
    return n;
}

WindParamsNamed windParams = windParamsNamed();

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} uboPacked;


UniformObjectNamed uniformObjectNamed() {
    UniformObjectNamed n;
    n.viewProjection = uboPacked.viewProjection;
    n.viewPosition = uboPacked.viewPos.xyz;
    return n;
}

UniformObjectNamed ubo = uniformObjectNamed();

#include "includes/perlin2d.glsl"
#include "includes/vegetation_common.glsl"

void main() {
    // Sentinel: instance was skipped by generator (empty biome or steep slope).
    if (instanceData.w < 0.0) {
        outWorldPos = vec3(0.0);
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    int cornerType = inCornerNormalData & 0xFF;

    vec3 worldPos = instanceData.xyz;

    if (impostorDistance > 0.0 && distance(windParams.cameraPosition, worldPos) >= impostorDistance) {
        outWorldPos = worldPos;
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float rotFrac = fract(instanceData.w);
    float theta = rotFrac * 6.28318530718;
    float cosT = cos(theta);
    float sinT = sin(theta);

    float heightScale = inBakedHeight;
    float scale = billboardScale * heightScale;
    vec3 localPos = rotateY(inLocalPos, cosT, sinT) * scale;
    vec3 tangent = rotateY(inLocalTangent, cosT, sinT);

    float heightFactor = (cornerType == 2 || cornerType == 3) ? 1.0 : 0.0;

    // Wind (same as vegetation.vert)
    vec2 windDirXZ = windParams.windDirection;
    float dirLen = length(windDirXZ);
    if (dirLen > 0.0001) windDirXZ /= dirLen; else windDirXZ = vec2(0.0);
    float amplitude = windParams.windStrength;
    float baseFreq = windParams.windBaseFrequency;
    float speed = windParams.windSpeed;
    float gustFreq = windParams.gustFrequency;
    float gustStrength = windParams.gustStrength;
    float skewAmount = windParams.skewAmount;
    float trunkStiffness = windParams.trunkStiffness;
    float noiseScale = windParams.noiseScale;
    float verticalFlutter = windParams.verticalFlutter;
    float turbulence = windParams.turbulence;
    float bendWeight = pow(clamp(heightFactor, 0.0, 1.0), mix(4.0, 1.0, clamp(trunkStiffness, 0.0, 1.0)));
    vec2 windMotion = windDirXZ * (windTime * speed);
    vec2 p = worldPos.xz * (baseFreq * noiseScale);
    float nBase = perlin2(p + windMotion);
    float nSkew = perlin2(worldPos.xz * (baseFreq * noiseScale * 2.7) + vec2(windTime * 0.37, -windTime * 0.29));
    float sway = (nBase + perlin2(worldPos.xz * (gustFreq * noiseScale) - windMotion * 0.35) * gustStrength) * amplitude;
    float skew = nSkew * skewAmount * amplitude;
    vec2 turbulentDir = vec2(-windDirXZ.y, windDirXZ.x) * (nBase * turbulence * 0.35);
    vec3 horizontal = vec3((windDirXZ + turbulentDir) * sway, 0.0);
    vec3 skewOffset = tangent * (skew * bendWeight);
    vec3 vertical = vec3(0.0, abs(nSkew * verticalFlutter * amplitude) * bendWeight, 0.0);
    vec3 windOffset = (horizontal + skewOffset + vertical) * bendWeight;

    // Normal-aligned frame (same as vegetation.vert): tilt the corner offsets
    // onto the surface normal; the wind offset stays a world-space delta.
    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    localPos = tiltToNormal(localPos, surfN);

    vec3 finalPos = worldPos + localPos + windOffset;
    outWorldPos = finalPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
