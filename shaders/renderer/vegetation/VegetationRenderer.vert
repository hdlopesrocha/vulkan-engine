#version 450

#include "../../ubo/UniformObject.glsl"
#include "../../ubo/WindParamsUBO.glsl"

// Vertex-shader billboard expansion — replaces the old geometry-shader approach.
// 24 pre-computed corner vertices (6 planes × 4 corners) are drawn as a
// triangle strip and transformed per-instance in the vertex shader.

#include "../../includes/Locations.glsl"

// Per-corner vertex attributes (x24 in the base VBO)
layout(location = ATTR_POS) in vec3 inLocalPos;       // pre-rotation local offset from center
layout(location = ATTR_COLOR) in vec3 inLocalTangent; // pre-rotation tangent (stored in Vertex.color)
layout(location = ATTR_UV) in vec2 inCornerUV;        // UV for this corner
layout(location = ATTR_BRUSH_INDEX) in int inCornerNormalData; // encoded: hi=plane index, lo=corner type
// Per-instance data (binding 1)
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz = world position, w = billboardIndex + rotFrac
#ifndef VEG_CAPTURE
// Per-instance surface normal (binding 1, offset 16): the billboard frame
// tilts onto this so grass grows along the terrain normal. Excluded from the
// VEG_CAPTURE variant (capture pipeline provides no such attribute).
layout(location = ATTR_VEG_NORMAL) in vec3 inInstanceNormal;
#endif
#ifndef VEG_CAPTURE
// Baked height scale (binding 2, perf report 22 C2/H4): vegetationHeightScale
// evaluated once per chunk version by the bake dispatch instead of per vertex
// per pass. Same values the VS would compute (same device, same source).
// Undefined for VEG_CAPTURE (ImpostorCapture binds no aux buffer).
layout(location = ATTR_VEG_AUX) in float inBakedHeight;
#endif

layout(location = VARY_UV) out vec3 fragTexCoord;     // xy=uv, z=array layer
layout(location = VARY_BRUSHPATCH) flat out int outBrushIndex;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_PLANE_NORMAL) flat out vec3 outPlaneNormal;
layout(location = VARY_POSLIGHT) flat out vec3 outTangentWS;

// Canonical scene UBO (set=0 binding=0): the shared struct IS the block
// layout — no packed/named-view split.
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

// Canonical wind params block (set=2 binding=0): the shared struct IS the
// block layout — no packed/named-view split.
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
// Shared wind field (set 0, binding 27): ambient + tornadoes sampled with
// the push-constant clock. Requires perlin.glsl (perlinNoise3D) before
// wind_field.glsl. Excluded from VEG_CAPTURE: the capture pipeline binds
// its own set 0 (camera UBO, binding 0 only, no binding 27), so declaring
// the block there would break pipeline creation; capture is static by
// design (windEnabled=0) so nothing is lost.
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/vegetation/WindField.glsl"
#endif
#include "../../includes/vegetation/VegetationCommon.glsl"

// Pre-computed per-plane data for the 6 billboard planes (θ=0 frame).
// Plane 0-3: tilted 45°, Plane 4-5: vertical.
// Encoding: normalData.y = plane index (0-5), normalData.z = corner type (0=BL,1=BR,2=TL,3=TR)
// The vertex shader uses plane index to reconstruct the outward direction for tilted planes
// and computes the plane normal from tangent + outward.

vec3 getOutwardForPlane(int planeIdx) {
    // Outward directions for the 4 tilted planes (indices 0-3)
    vec3 outdirs[4] = vec3[4](
        vec3( 1.0, 0.0, 0.0),
        vec3( 0.0, 0.0, 1.0),
        vec3(-1.0, 0.0, 0.0),
        vec3( 0.0, 0.0,-1.0)
    );
    if (planeIdx < 4) return outdirs[planeIdx];  // tilted
    return vec3(0.0);  // vertical planes have no outward tilt
}

vec3 computePlaneNormal(vec3 tangent, int planeIdx, vec3 worldUp) {
    if (planeIdx < 4) {
        // Tilted plane: normal = cross(tangent, normalize(up + outward))
        vec3 outward = getOutwardForPlane(planeIdx);
        return normalize(cross(tangent, normalize(worldUp + outward)));
    } else {
        // Vertical plane: normal = cross(tangent, up)
        return normalize(cross(tangent, worldUp));
    }
}

// Copy of applyWindSkew from the old geometry shader
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
    // Shared-field supplement: horizontal (XZ) windSVF at the instance
    // position on the push-constant clock. Under the same windEnabled gate
    // (the early-out at the top returns 0 when disabled, so this adds
    // exactly 0 then; VEG_CAPTURE excludes the block entirely and keeps 0).
    // Gain 0.2: the shared ambient mirrors these same wind sliders, so 0.2
    // keeps it a modest ~20% supplement to the existing sway; tornado cores
    // are clamped to +/-15 m/s so funnels bend blades without flinging them.
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
    // Sentinel: instance was skipped by generator (empty biome or steep slope).
    if (instanceData.w < 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPos = instanceData.xyz;
    vec3 camPos = ubo.viewPosition;

    int billboardIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);

    // Distance-based culling, hoisted above per-instance decode (perf report
    // 22 H4): the condition uses only worldPos/camPos, so culled instances
    // skip the trig, height fetch and index decodes with identical output
    // (degenerate triangle either way). No-op when impostors are off
    // (impostorDistance == 0).
    //
    // This is the impostor hand-off point: billboards own [0,
    // impostorDistance) and impostors own [impostorDistance, ...) (see
    // impostors.vert's matching < impostorDistance cull). It is a DIRECT
    // per-instance swap - no dithered cross-fade. The previous 0.50x-1.15x
    // complementary dither dissolved the billboards into sparse dots that
    // the impostors did not visually replace, reading as a bare ring; the
    // half-res Minimal vegetation target made each 4x4 Bayer block cover
    // 8x8 screen pixels, so the ring was unmissable.
    if (impostorDistance > 0.0 && distance(worldPos, camPos) >= impostorDistance) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    int planeIdx  = (inCornerNormalData >> 8) & 0xFF;
    int cornerType = inCornerNormalData & 0xFF; // 0=BL, 1=BR, 2=TL, 3=TR

    float theta = rotFrac * 6.28318530718;
    float cosT = cos(theta);
    float sinT = sin(theta);

    // Per-instance height variation: baked (see inBakedHeight decl).
    // Identical to vegetationHeightScale(worldPos.xz); the bake covers all
    // 24 corner invocations with one evaluation per chunk version.
    // VEG_CAPTURE (ImpostorCapture's canonical single instance, no aux
    // buffer bound) evaluates locally, exactly as before the bake.
#ifdef VEG_CAPTURE
    float heightScale = vegetationHeightScale(worldPos.xz);
#else
    float heightScale = inBakedHeight;
#endif

    // Rotate tangent and local position by Y-rotation
    vec3 tangent = rotateY(inLocalTangent, cosT, sinT);

    // Compute height factor for wind: 0 = bottom, 1 = top
    float heightFactor = (cornerType == 2 || cornerType == 3) ? 1.0 : 0.0;

    // Scale the pre-computed corner offsets by the per-instance billboard size.
    // Base corners use hs=0.5, h=1.0, tilt=1.0 — scale by billboardScale * heightVariation.
    float scale = billboardScale * heightScale;
    vec3 localPos = rotateY(inLocalPos, cosT, sinT) * scale;

    // Apply wind displacement
    vec3 windOffset = applyWindSkew(worldPos + localPos, tangent, heightFactor);

    // Final world position. The billboard frame tilts onto the surface normal
    // (shortest arc from +Y) so grass grows along the terrain, not world-up.
    // Wind was evaluated in the un-tilted frame above; the offset itself is a
    // world-space delta and stays as-is (mirrored by VegetationRendererShadow.vert).
#ifndef VEG_CAPTURE
    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    localPos = tiltToNormal(localPos, surfN);
    tangent = tiltToNormal(tangent, surfN);
#else
    vec3 surfN = vec3(0.0, 1.0, 0.0);
#endif
    vec3 finalPos = worldPos + localPos + windOffset;

    // Compute plane normal (needs rotated + tilted tangent)
    vec3 planeNormal = computePlaneNormal(tangent, planeIdx, surfN);

    outWorldPos = finalPos;
    outBrushIndex = billboardIdx;
    outPlaneNormal = planeNormal;
    outTangentWS = tangent;

    // UV: from vertex attribute, array layer = billboard index
    fragTexCoord = vec3(inCornerUV, float(billboardIdx));

    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
