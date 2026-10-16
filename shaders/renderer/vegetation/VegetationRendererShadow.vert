#version 450

#include "../../ubo/UniformObject.glsl"
#include "../../ubo/WindParamsUBO.glsl"

// Bladed grass EVSM shadow vertex shader.
// Expands the blade-tuft base mesh into world space (same math as
// VegetationRenderer.vert, light VP). Only outputs world position.

#include "../../includes/Locations.glsl"

layout(location = ATTR_POS) in vec3 inLocalPos;       // x=bladeIdx, y=heightT, z=side
layout(location = ATTR_COLOR) in vec3 inLocalTangent; // canonical blade dir
layout(location = ATTR_BRUSH_INDEX) in int inCornerNormalData;
layout(location = ATTR_INSTANCE) in vec4 instanceData;
layout(location = ATTR_VEG_NORMAL) in vec3 inInstanceNormal;
layout(location = ATTR_VEG_AUX) in float inBakedHeight;

layout(location = VARY_POSWORLD) out vec3 outWorldPos;

// Canonical wind params block (set=2 binding=0)
layout(std140, set = 2, binding = 0) uniform WindParamsBlock {
    WindParamsUBO windParams;
};

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

// Canonical scene UBO (set=0 binding=0)
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

#include "../../includes/noise/Perlin2D.glsl"
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/vegetation/WindField.glsl"
#include "../../includes/vegetation/VegetationCommon.glsl"
#include "../../includes/vegetation/BladeGrass.glsl"

void main() {
    if (instanceData.w < 0.0) {
        outWorldPos = vec3(0.0);
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPos = instanceData.xyz;

    if (impostorDistance > 0.0 && distance(windParams.cameraPosition, worldPos) >= impostorDistance) {
        outWorldPos = worldPos;
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    int bladeIdx = (inCornerNormalData >> 8) & 0xFF;
    float heightT = inLocalPos.y;
    float side = inLocalPos.z;

    float rotFrac = fract(instanceData.w);
    float theta = rotFrac * 6.28318530718;
    float cosT = cos(theta);
    float sinT = sin(theta);

    float heightScale = inBakedHeight;
    float scale = billboardScale * heightScale;

    vec4 bh = bladeHash(float(bladeIdx));
    float bladeHeightVar = 0.70 + 0.50 * bh.x;
    float leanAmt = (0.10 + 0.35 * bh.y);
    float leanAng = bh.z * 6.28318530718;
    float widthVar = 0.75 + 0.50 * bh.w;

    vec3 baseDir = rotateY(inLocalTangent, cosT, sinT);
    {
        float j = (bh.z - 0.5) * 0.9;
        float cj = cos(j), sj = sin(j);
        baseDir = rotateY(baseDir, cj, sj);
    }
    vec3 widthDir = normalize(vec3(-baseDir.z, 0.0, baseDir.x) + vec3(1e-5, 0.0, 0.0));
    vec3 leanDir = rotateY(vec3(cos(leanAng), 0.0, sin(leanAng)), cosT, sinT);

    float tuftRadius = 0.07;
    float halfWidth = 0.05 * widthVar;
    float bladeH = (0.48 + 0.22 * bh.x) * bladeHeightVar;

    vec3 discOffset;
    float leanScale = 1.0;
    if (bladeIdx >= 8) {
        float sat = float(bladeIdx) - 8.0;
        float satAng = sat * 1.5707963 + 0.7853982 + (bh.y - 0.5) * 0.6;
        float satR = 0.20 + 0.06 * bh.w;
        discOffset = rotateY(vec3(cos(satAng), 0.0, sin(satAng)), cosT, sinT) * satR;
    } else {
        discOffset = baseDir * (tuftRadius * (0.35 + 0.65 * bh.y));
    }
    if (bladeIdx >= 8) leanScale = 0.5;
    vec3 localPos;
    if (heightT < 0.5) {
        localPos = discOffset + widthDir * (side * halfWidth);
        localPos.y = 0.0;
    } else {
        localPos = discOffset + vec3(0.0, bladeH, 0.0)
                 + leanDir * (leanAmt * leanScale * bladeH)
                 + baseDir * (0.12 * bladeH * bh.w);
    }
    localPos *= scale;

    // Wind (same as the main blade shader)
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
    float bendWeight = pow(clamp(heightT, 0.0, 1.0), mix(4.0, 1.0, clamp(trunkStiffness, 0.0, 1.0)));
    vec2 windMotion = windDirXZ * (windTime * speed);
    vec2 p = worldPos.xz * (baseFreq * noiseScale);
    float nBase = perlin2(p + windMotion);
    float nSkew = perlin2(worldPos.xz * (baseFreq * noiseScale * 2.7) + vec2(windTime * 0.37, -windTime * 0.29));
    float sway = (nBase + perlin2(worldPos.xz * (gustFreq * noiseScale) - windMotion * 0.35) * gustStrength) * amplitude;
    float skew = nSkew * skewAmount * amplitude;
    vec2 turbulentDir = vec2(-windDirXZ.y, windDirXZ.x) * (nBase * turbulence * 0.35);
    vec3 horizontal = vec3((windDirXZ + turbulentDir) * sway, 0.0);
    vec3 skewOffset = baseDir * (skew * bendWeight);
    vec3 vertical = vec3(0.0, abs(nSkew * verticalFlutter * amplitude) * bendWeight, 0.0);
    vec3 svfSh = (windEnabled >= 0.5) ? windSVF(worldPos, windTime) : vec3(0.0);
    vec2 svfShXZ = clamp(svfSh.xz, vec2(-15.0), vec2(15.0));
    vec3 sharedWind = vec3(svfShXZ.x, 0.0, svfShXZ.y) * 0.2;
    vec3 windOffset = (horizontal + sharedWind + skewOffset + vertical) * bendWeight;

    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    localPos = tiltToNormal(localPos, surfN);

    vec3 finalPos = worldPos + localPos + windOffset;
    outWorldPos = finalPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
