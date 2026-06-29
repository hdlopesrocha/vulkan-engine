#version 450

// Vegetation EVSM shadow vertex shader.
// Expands 24-corner billboard mesh into world space but only outputs
// fragPosWorld (the only varying needed by shadow_evsm.frag).

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inLocalPos;
layout(location = ATTR_COLOR) in vec3 inLocalTangent;
layout(location = ATTR_UV) in vec2 inCornerUV;
layout(location = ATTR_BRUSH_INDEX) in int inCornerNormalData;
layout(location = ATTR_INSTANCE) in vec4 instanceData;

layout(location = VARY_POSWORLD) out vec3 outWorldPos;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
};

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

#include "includes/perlin2d.glsl"
#include "includes/vegetation_common.glsl"

void main() {
    int cornerType = inCornerNormalData & 0xFF;

    vec3 worldPos = instanceData.xyz;
    float rotFrac = fract(instanceData.w);
    float theta = rotFrac * 6.28318530718;
    float cosT = cos(theta);
    float sinT = sin(theta);

    float heightScale = vegetationHeightScale(worldPos.xz);
    float scale = billboardScale * heightScale;
    vec3 localPos = rotateY(inLocalPos, cosT, sinT) * scale;
    vec3 tangent = rotateY(inLocalTangent, cosT, sinT);

    float heightFactor = (cornerType == 2 || cornerType == 3) ? 1.0 : 0.0;

    // Wind (same as vegetation.vert)
    vec2 windDirXZ = windDirAndStrength.xz;
    float dirLen = length(windDirXZ);
    if (dirLen > 0.0001) windDirXZ /= dirLen; else windDirXZ = vec2(0.0);
    float amplitude = windDirAndStrength.w;
    float baseFreq = windNoise.x;
    float speed = windNoise.y;
    float gustFreq = windNoise.z;
    float gustStrength = windNoise.w;
    float skewAmount = windShape.x;
    float trunkStiffness = windShape.y;
    float noiseScale = windShape.z;
    float verticalFlutter = windShape.w;
    float turbulence = windTurbulence.x;
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

    vec3 finalPos = worldPos + localPos + windOffset;
    outWorldPos = finalPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
