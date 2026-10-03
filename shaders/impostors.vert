#version 450

#include "includes/locations.glsl"

layout(location = ATTR_UV) in vec2 inCornerUV;
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz=world pos, w=billboard index + rotFrac
layout(location = ATTR_VEG_NORMAL) in vec3 inInstanceNormal; // surface normal: re-anchor quad onto it
// Baked height scale (binding 2, perf report 22 C2/H4); see vegetation.vert.
layout(location = ATTR_VEG_AUX) in float inBakedHeight;

layout(location = VARY_UV) out vec3 outTexCoord;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_FACE_NORMAL) flat out vec3 outFaceNormal;
layout(location = VARY_ROTFRAC) flat out float outRotFrac;
layout(location = VARY_POSLIGHT) flat out vec3 outInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} uboPacked;

// Named view over the packed SolidParamsUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct UniformObjectNamed {
    mat4 viewProjection;
    vec3 viewPosition;
};

UniformObjectNamed uniformObjectNamed() {
    UniformObjectNamed n;
    n.viewProjection = uboPacked.viewProjection;
    n.viewPosition = uboPacked.viewPos.xyz;
    return n;
}

UniformObjectNamed ubo = uniformObjectNamed();

layout(set = 2, binding = 0) uniform WindParamsUBO {
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
} windParamsPacked;

// Named view over the packed WindParamsUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct WindParamsNamed {
    vec2 windDirection;
    float windStrength;
    float windBaseFrequency;
    float windSpeed;
    float gustFrequency;
    float gustStrength;
    float skewAmount;
    float trunkStiffness;
    float noiseScale;
    float verticalFlutter;
    float turbulence;
    bool densityEnabled;
    float nearDistance;
    float farDistance;
    float minFactor;
    vec3 cameraPosition;
    float densityFalloff;
};

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

// Fire params (set=2, binding=1) for procedural fire-quad sizing. Must match
// FireParamsUBO (vulkan/ubo/FireUBO.hpp). Values only — no flame functions
// (those need fbm, unavailable with perlin2d.glsl here).
layout(set = 2, binding = 1) uniform FireParamsUBO {
    vec4 fireEnabledSizeSpeedIntensity; // x=enabled, y=size, z=speed, w=intensity
    vec4 fireShape;                     // x=flicker, y=noiseScale, z=heightScale, w=turbulence
    vec4 fireMotion;                    // x=riseSpeed, y=windInfluence, z=alpha, w=emissive
    vec4 fireInnerColor;
    vec4 fireMidColor;
    vec4 fireOuterColor;
    vec4 fireExtra;                     // x=smoke
} fireParamsImp;

#include "includes/perlin2d.glsl"
#include "includes/vegetation_common.glsl"

void main() {
    vec3 worldPos = instanceData.xyz;
    int billboardIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);
    outRotFrac = rotFrac;

    // Fire impostors shade procedurally (same as fire billboards): no capture
    // exists for flames, which stay animated at every distance. When fire is
    // disabled in the widget, fire instances collapse (atlas layer 3 is out
    // of bounds for the 3-layer capture arrays).
    bool isFire = (billboardIdx == FIRE_BILLBOARD_INDEX);
    bool fireOn = fireParamsImp.fireEnabledSizeSpeedIntensity.x > 0.5;
    if (isFire && !fireOn) {
        outTexCoord = vec3(0.0); outWorldPos = worldPos; outFaceNormal = vec3(0.0, 1.0, 0.0);
        outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec3 camPos = ubo.viewPosition;
    float dist = distance(worldPos, camPos);

    // Direct hand-off: billboards own [0, impostorDistance), impostors own
    // [impostorDistance, ...). No dithered cross-fade (see vegetation.frag):
    // the fade dissolved the grass before the impostors read as vegetation.
    if (impostorDistance <= 0.0 || dist < impostorDistance) {
        outTexCoord = vec3(0.0); outWorldPos = worldPos; outFaceNormal = vec3(0.0, 1.0, 0.0);
        outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float densityFactor = densityFactorForDistance(dist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) {
            outTexCoord = vec3(0.0); outWorldPos = worldPos; outFaceNormal = vec3(0.0, 1.0, 0.0);
            outInstanceOffset = worldPos;
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }
    }

    outInstanceOffset = worldPos;

    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int NUM_VIEWS = 20;

    vec3 toCamera = normalize(camPos - worldPos);

    float instTheta = rotFrac * 6.28318530718;
    float cI = cos(instTheta);
    float sI = sin(instTheta);
    vec3 toCamera_canonical = vec3(
         cI * toCamera.x + sI * toCamera.z,
        toCamera.y,
        -sI * toCamera.x + cI * toCamera.z
    );

    int bestIdx = 0;
    float bestDot = -2.0;
    for (int i = 0; i < NUM_VIEWS; i++) {
        float y = 1.0 - (float(i) + 0.5) / float(NUM_VIEWS) * 2.0;
        float r = sqrt(max(0.0, 1.0 - y * y));
        float theta = goldenAngle * float(i);
        vec3 d = vec3(cos(theta) * r, y, sin(theta) * r);
        float dt = dot(toCamera_canonical, d);
        if (dt > bestDot) { bestDot = dt; bestIdx = i; }
    }

    int layerIdx = clamp(billboardIdx, 0, 3) * NUM_VIEWS + bestIdx;

    float hs = inBakedHeight;

    vec3 center;
    float quadHalfW;
    float quadHalfH;
    float fireCenterH = 0.0f;
    if (isFire) {
        // Procedural fire quad: sized to the CAPTURE FRAME (not the flame
        // core) so the full-frame snapshot UVs map 1:1 with no shrinking.
        // Same formula as ImpostorCapture::capture (frame half-extent from
        // live settings); transparent margins are discarded, so the bigger
        // quad costs only a few transparent pixels. Flame center anchors the
        // quad so base and tip land on their world footprint.
        // Width additionally tracks the per-instance variation the billboards
        // use (snapshots bake wVar 0.85 from the canonical rotFrac 0), so
        // wide instances don't shrink at the hand-off.
        float fSize = max(fireParamsImp.fireEnabledSizeSpeedIntensity.y, 0.1);
        float fHeightScale = max(fireParamsImp.fireShape.z, 0.1);
        float fireHLive = fSize * fHeightScale;
        float frameHE = 0.5773503 * max(fireHLive * 0.92, fSize * 1.0);
        float wVarLive = (0.85 + 0.3 * rotFrac) / 0.85;
        quadHalfW = frameHE * wVarLive;
        quadHalfH = frameHE;
        fireCenterH = fireHLive * 0.5;
    }

    // Match the capture setup: the plant was captured at heightScale=1.0 with a
    // fixed-size square framebuffer. The plant occupies only 34.6% of the image
    // height (top/bottom 32.7% are clear). The quad and its centre scale
    // uniformly with the runtime height scale (hs), exactly like the billboard
    // (vegetation.vert scales its corners by billboardScale * hs), so the
    // impostor is the same size as the plant it replaces. The UV crop maps the
    // captured bbox (a fixed fraction of the image) to the whole quad, so the
    // texture scales with the quad; it must NOT scale with hs itself.
    // The quad stays glued to sloped ground by rising along the surface normal.
    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    if (isFire) {
        center = worldPos + surfN * fireCenterH;
    } else {
        center = worldPos + surfN * (billboardScale * hs * 0.5);
    }
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    vec3 right;
    float sinElev = length(vec2(toCamera.x, toCamera.z));
    if (sinElev > 0.001) {
        right = normalize(cross(worldUp, toCamera));
    } else {
        right = vec3(1.0, 0.0, 0.0);
    }
    vec3 upDir = normalize(cross(toCamera, right));

    // Quad size: the captured plant bbox in world units, scaled by hs.
    // (The old code scaled the width by hs but left the height at
    // billboardScale and scaled uFrac by hs, which cancelled out and rendered
    // every impostor at the capture size - tall plants shrank and short ones
    // were cropped at the hand-off.)
    // Fire quads were sized above (full-quad procedural flames, no crop).
    if (!isFire) {
        quadHalfW = 0.75 * billboardScale * hs;
        quadHalfH = 0.5  * billboardScale * hs;
    }
    right = right * quadHalfW;
    vec3 up = upDir * quadHalfH;

    vec3 faceNorm = toCamera;
    outFaceNormal = faceNorm;

    float u = inCornerUV.x;
    float v = inCornerUV.y;
    vec3 offset = (u - 0.5) * 2.0 * right + (0.5 - v) * 2.0 * up;
    vec3 finalPos = center + offset;

    // Crop UV to the plant's bounding box within the captured image.
    // The plant occupies UV.V in [0.327, 0.673] (vertical) and
    // UV.U in [0.5 ± 1.5/2.8867] (horizontal), independent of hs.
    // Fire skips the crop: flames shade procedurally from the raw quad UV
    // (impostors.frag derives the fire type from the layer index).
    if (isFire) {
        outTexCoord = vec3(inCornerUV, float(layerIdx));
    } else {
        float uFrac = 1.5 / 2.886751346;      // 1.5 / (2 * 2.5 * tan(30°))
        float vFrac = 1.0 / 2.886751346;      // billboardScale / (2 * 2.5 * billboardScale * tan(30°))
        float vOff  = 0.5 - 0.5 / 2.886751346;
        outTexCoord = vec3(0.5 + (inCornerUV.x - 0.5) * uFrac,
                           inCornerUV.y * vFrac + vOff,
                           float(layerIdx));
    }
    outWorldPos = finalPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}
