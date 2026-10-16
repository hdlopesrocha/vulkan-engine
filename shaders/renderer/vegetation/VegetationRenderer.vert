#version 450

#include "../../ubo/UniformObject.glsl"
#include "../../ubo/WindParamsUBO.glsl"

// Bladed grass vertex shader — one triangle per blade, 8 blades per tuft.
// Replaces the old 6-plane billboard expansion. Base mesh (24 verts) encodes:
//   inLocalPos.x = blade index [0,8), .y = heightT (0=base, 1=tip), .z = side (-1=left, +1=right, 0=tip)
//   inLocalTangent = canonical blade base direction (XZ disc, rotated per instance)
//   inCornerUV = (side*0.5, heightT) for frag gradient
//   inCornerNormalData = (bladeIdx<<8)|corner (0=left,1=right,2=tip)
// Per-instance data identical to before (pos + variant+rotFrac, normal, baked height).

#include "../../includes/Locations.glsl"

// Per-blade vertex attributes (24 verts in the base VBO)
layout(location = ATTR_POS) in vec3 inLocalPos;       // x=bladeIdx, y=heightT, z=side
layout(location = ATTR_COLOR) in vec3 inLocalTangent; // canonical blade dir (XZ)
layout(location = ATTR_UV) in vec2 inCornerUV;        // (side*0.5, heightT)
layout(location = ATTR_BRUSH_INDEX) in int inCornerNormalData; // encoded: hi=blade idx, lo=corner type
// Per-instance data (binding 1)
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz = world position, w = variantIndex + rotFrac
#ifndef VEG_CAPTURE
// Per-instance surface normal (binding 1, offset 16): tuft tilts onto this.
layout(location = ATTR_VEG_NORMAL) in vec3 inInstanceNormal;
#endif
#ifndef VEG_CAPTURE
// Baked height scale (binding 2): same values the old billboards used.
layout(location = ATTR_VEG_AUX) in float inBakedHeight;
#endif

layout(location = VARY_UV) out vec3 fragTexCoord;     // x=side, y=heightT, z=variant layer
layout(location = VARY_BRUSHPATCH) flat out int outBrushIndex;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_PLANE_NORMAL) flat out vec3 outPlaneNormal;
layout(location = VARY_POSLIGHT) flat out vec3 outTangentWS;
#ifndef VEG_CAPTURE
layout(location = VARY_TANGENTWS) flat out vec3 outLightDirN;
#endif

// Canonical scene UBO (set=0 binding=0)
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

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

#include "../../includes/noise/Perlin2D.glsl"
#ifndef VEG_CAPTURE
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/vegetation/WindField.glsl"
#endif
#include "../../includes/vegetation/VegetationCommon.glsl"
#include "../../includes/vegetation/BladeGrass.glsl"

// Copy of applyWindSkew from the old billboard shader (height-weighted).
vec3 applyWindSkew(vec3 basePos, vec3 right, float heightFactor) {
    if (windEnabled < 0.5) return vec3(0.0);

    vec2 windDirXZ = windParams.windDirection;
    float dirLen = length(windDirXZ);
    if (dirLen > 0.0001) {
        windDirXZ /= dirLen;
    } else {
        windDirXZ = vec2(0.0, 0.0);
    }
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
    vec2 p = basePos.xz * (baseFreq * noiseScale);

    float nBase = perlin2(p + windMotion);
    float nGust = perlin2(basePos.xz * (gustFreq * noiseScale) - windMotion * 0.35);
    float nSkew = perlin2(basePos.xz * (baseFreq * noiseScale * 2.7) + vec2(windTime * 0.37, -windTime * 0.29));

    float sway = (nBase + nGust * gustStrength) * amplitude;
    float skew = nSkew * skewAmount * amplitude;
    vec2 turbulentDir = vec2(-windDirXZ.y, windDirXZ.x) * (nBase * turbulence * 0.35);
    float flutter = nSkew * verticalFlutter * amplitude;

    vec3 horizontal = vec3((windDirXZ + turbulentDir) * sway, 0.0);
    vec3 skewOffset = right * (skew * bendWeight);
    vec3 vertical = vec3(0.0, abs(flutter) * bendWeight, 0.0);
    vec3 sharedWind = vec3(0.0);
#ifndef VEG_CAPTURE
    {
        vec3 svfVeg = windSVF(basePos, windTime);
        vec2 svfVegXZ = clamp(svfVeg.xz, vec2(-15.0), vec2(15.0));
        sharedWind = vec3(svfVegXZ.x, 0.0, svfVegXZ.y) * 0.2;
    }
#endif
    return (horizontal + sharedWind + skewOffset + vertical) * bendWeight;
}

void main() {
    // Sentinel: instance was skipped by generator.
    if (instanceData.w < 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPos = instanceData.xyz;
    vec3 camPos = ubo.viewPosition;

    int variantIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);

    // Impostor hand-off: blades own [0, impostorDistance), impostor quads own
    // [impostorDistance, ...). Direct per-instance swap, no dithered fade.
    if (impostorDistance > 0.0 && distance(worldPos, camPos) >= impostorDistance) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    int bladeIdx = (inCornerNormalData >> 8) & 0xFF;
    float heightT = inLocalPos.y;
    float side = inLocalPos.z;

    float theta = rotFrac * 6.28318530718;
    float cosT = cos(theta);
    float sinT = sin(theta);

#ifdef VEG_CAPTURE
    float heightScale = vegetationHeightScale(worldPos.xz);
#else
    float heightScale = inBakedHeight;
#endif

    // Per-blade deterministic variation (height, lean, width, phase).
    // Canonical in the blade index only (see BladeGrass.glsl): the capture
    // bakes this exact layout, so runtime blades and their impostors agree.
    vec4 bh = bladeHash(float(bladeIdx));
    float bladeHeightVar = 0.70 + 0.50 * bh.x;   // 0.70..1.20
    float leanAmt = (0.10 + 0.35 * bh.y);        // tip lean as fraction of height
    float leanAng = bh.z * 6.28318530718;        // lean direction
    float widthVar = 0.75 + 0.50 * bh.w;         // base half-width variation

    float scale = billboardScale * heightScale;

    // Canonical blade base direction, rotated by the instance rotation plus a
    // small per-blade jitter so blades within a tuft do not align.
    vec3 baseDir = rotateY(inLocalTangent, cosT, sinT);
    {
        float j = (bh.z - 0.5) * 0.9;
        float cj = cos(j), sj = sin(j);
        baseDir = rotateY(baseDir, cj, sj);
    }

    // Blade frame: width direction is perpendicular to the blade direction.
    vec3 widthDir = normalize(vec3(-baseDir.z, 0.0, baseDir.x) + vec3(1e-5, 0.0, 0.0));
    vec3 leanDir = vec3(cos(leanAng), 0.0, sin(leanAng));
    leanDir = rotateY(leanDir, cosT, sinT);

    // Tuft footprint and blade dimensions in unit space (scaled by `scale`).
    // 12 wide blades per tuft. Worst-case static extents stay inside the
    // impostor quad (half-width 0.75, top edge 1.0 in these units): max
    // blade height 0.84, max horizontal reach ~0.61, leaving margin for
    // wind flutter/sway. With billboardScale=10 and heightScale~1: ~0.7 m
    // radius tufts, ~0.5-1.0 m wide blades, ~3-8 m tall.
    float tuftRadius = 0.07;
    float halfWidth = 0.05 * widthVar;
    float bladeH = (0.48 + 0.22 * bh.x) * bladeHeightVar;

    // Root placement: blades 0-7 form the central tuft; blades 8-11 are
    // satellite plants distributed around the instance origin at the four
    // diagonal compass points (plus hash jitter), so the extra plants do
    // not grow from the same spot. Satellite lean is halved to keep the
    // whole patch inside the impostor quad. Everything rotates rigidly
    // with the instance rotation (impostor layer matching assumes a rigid
    // tuft under Y rotation).
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
    // Local blade vertex before tilt: base corners spread by width, tip rises
    // with lean + slight forward curve.
    vec3 localPos;
    if (heightT < 0.5) {
        localPos = discOffset + widthDir * (side * halfWidth);
        localPos.y = 0.0;
    } else {
        // Tip: rise + lean, plus a small mid-curve bow along the blade dir.
        localPos = discOffset + vec3(0.0, bladeH, 0.0)
                 + leanDir * (leanAmt * leanScale * bladeH)
                 + baseDir * (0.12 * bladeH * bh.w);
    }
    localPos *= scale;

    // Blade facing normal (for lighting): perpendicular to the blade plane,
    // i.e. cross of blade-up and width dir, rotated by instance rotation.
    vec3 bladeNormal = normalize(cross(widthDir, vec3(0.0, 1.0, 0.0)) + vec3(1e-4, 0.0, 0.0));
    bladeNormal = rotateY(bladeNormal, cosT, sinT);

    // Wind displacement, weighted by height (tips sway, bases stay).
    vec3 windOffset = applyWindSkew(worldPos + localPos, baseDir, heightT);

    // Tilt the tuft frame onto the surface normal so grass grows along terrain.
#ifndef VEG_CAPTURE
    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    localPos = tiltToNormal(localPos, surfN);
    baseDir = tiltToNormal(baseDir, surfN);
    bladeNormal = tiltToNormal(bladeNormal, surfN);
#else
    vec3 surfN = vec3(0.0, 1.0, 0.0);
#endif
    vec3 finalPos = worldPos + localPos + windOffset;

    outWorldPos = finalPos;
    outBrushIndex = variantIdx;
    outPlaneNormal = bladeNormal;
    outTangentWS = baseDir;
#ifndef VEG_CAPTURE
    outLightDirN = normalize(-ubo.lightDirection);
#endif

    // x=side (-1..1), y=heightT (0..1), z=variant for the frag palette.
    fragTexCoord = vec3(side, heightT, float(variantIdx));

    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
