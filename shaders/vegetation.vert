#version 450

// Vertex-shader billboard expansion — replaces the old geometry-shader approach.
// 24 pre-computed corner vertices (6 planes × 4 corners) are drawn as a
// triangle strip and transformed per-instance in the vertex shader.

#include "includes/locations.glsl"

// Per-corner vertex attributes (x24 in the base VBO)
layout(location = ATTR_POS) in vec3 inLocalPos;       // pre-rotation local offset from center
layout(location = ATTR_COLOR) in vec3 inLocalTangent; // pre-rotation tangent (stored in Vertex.color)
layout(location = ATTR_UV) in vec2 inCornerUV;        // UV for this corner
layout(location = ATTR_BRUSH_INDEX) in int inCornerNormalData; // encoded: hi=plane index, lo=corner type
// Per-instance data (binding 1)
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz = world position, w = billboardIndex + rotFrac

layout(location = VARY_UV) out vec3 fragTexCoord;     // xy=uv, z=array layer
layout(location = VARY_BRUSHPATCH) flat out int outBrushIndex;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_PLANE_NORMAL) flat out vec3 outPlaneNormal;
layout(location = VARY_POSLIGHT) flat out vec3 outTangentWS;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

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

#include "includes/perlin2d.glsl"
#include "includes/vegetation_common.glsl"

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

    vec2 windDirXZ = windDirAndStrength.xz;
    float dirLen = length(windDirXZ);
    if (dirLen > 0.0001) {
        windDirXZ /= dirLen;
    } else {
        windDirXZ = vec2(0.0, 0.0);
    }
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
    return (horizontal + skewOffset + vertical) * bendWeight;
}

void main() {
    int planeIdx  = (inCornerNormalData >> 8) & 0xFF;
    int cornerType = inCornerNormalData & 0xFF; // 0=BL, 1=BR, 2=TL, 3=TR

    vec3 worldPos = instanceData.xyz;
    vec3 camPos = ubo.viewPos.xyz;

    // Sentinel: empty-biome instances have w < 0
    if (instanceData.w < 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Decode biome-driven height scale, billboard type, and rotation
    // from the packed instance word.
    int billboardIdx = decodeBillboardIndex(instanceData.w);
    float heightScale = decodeHeightScale(instanceData.w);
    float rotFrac = decodeRotFrac(instanceData.w);
    float theta = rotFrac * 6.28318530718;
    float cosT = cos(theta);
    float sinT = sin(theta);

    bool shadowPass = windEnabled < 0.0;

    // Distance-based culling (same as old geometry shader)
    if (impostorDistance > 0.0 && distance(worldPos, camPos) >= impostorDistance) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Rotate tangent and local position by Y-rotation
    vec3 tangent = rotateY(inLocalTangent, cosT, sinT);

    // Scale the pre-computed corner offsets by the per-instance billboard size.
    // Base corners use hs=0.5, h=1.0, tilt=1.0 — scale by billboardScale * heightVariation.
    float scale = billboardScale * heightScale;
    vec3 localPos = rotateY(inLocalPos, cosT, sinT) * scale;

    // Compute height factor for wind: 0 = bottom, 1 = top
    float heightFactor = (cornerType == 2 || cornerType == 3) ? 1.0 : 0.0;

    // Apply wind displacement
    vec3 windOffset = applyWindSkew(worldPos + localPos, tangent, heightFactor);

    // Final world position
    vec3 finalPos = worldPos + localPos + windOffset;

    // Compute plane normal (needs rotated tangent)
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    vec3 planeNormal = computePlaneNormal(tangent, planeIdx, worldUp);

    outWorldPos = finalPos;
    outBrushIndex = billboardIdx;
    outPlaneNormal = planeNormal;
    outTangentWS = tangent;

    // UV: from vertex attribute, array layer = billboard index
    fragTexCoord = vec3(inCornerUV, float(billboardIdx));

    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
