#version 450
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

layout(location = 0) in vec3 fragTexCoordIn[];
layout(location = 1) flat in int fragTexIndexIn[];
layout(location = 2) in vec3 fragWorldPosIn[];

layout(location = 0) out vec3 inTexCoord;
layout(location = 1) flat out int inTexIndex;
layout(location = 2) out      vec3 outWorldPos;    // interpolated world-space vertex position
layout(location = 3) flat out vec3 outPlaneNormal; // face normal of the billboard plane
layout(location = 4) flat out vec3 outTangentWS;   // tangent along billboard width

// Must match SolidParamsUBO — only read the first two fields.
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

vec2 fade2(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 grad2(vec2 ip) {
    float a = hash12(ip) * 6.28318530718;
    return vec2(cos(a), sin(a));
}

float perlin2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = fade2(f);

    float n00 = dot(grad2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float n10 = dot(grad2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float n01 = dot(grad2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float n11 = dot(grad2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    float nx0 = mix(n00, n10, u.x);
    float nx1 = mix(n01, n11, u.x);
    return mix(nx0, nx1, u.y);
}

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float densityFactorForDistance(float distanceToCamera) {
    if (densityParams.x < 0.5) return 1.0;

    float nearDistance = max(0.0, densityParams.y);
    float minFactor = clamp(densityParams.w, 0.0, 1.0);
    float falloff = cameraPosAndFalloff.w;
    if (distanceToCamera <= nearDistance || minFactor >= 1.0 || falloff <= 0.0) {
        return 1.0;
    }

    float density = exp(-falloff * (distanceToCamera - nearDistance));
    return clamp(density, minFactor, 1.0);
}

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
    // Sentinel: w=-1 means this instance was skipped by the compute shader (slope filter).
    // Discard rather than rendering garbage geometry from uninitialized memory.
    if (fragTexIndexIn[0] < 0) return;

    vec3 worldPos = fragWorldPosIn[0];
    vec3 camPos   = ubo.viewPos.xyz;

    // If impostor rendering is enabled, skip instances that are far enough away –
    // the impostor pipeline will render them as camera-facing quads instead.
    if (impostorDistance > 0.0 && distance(worldPos, camPos) >= impostorDistance) return;

    float densityFactor = densityFactorForDistance(distance(cameraPosAndFalloff.xyz, worldPos));
    if (densityFactor < 0.9999) {
        float keep = hash13(vec3(worldPos.xz * 0.03125, fragTexCoordIn[0].z + worldPos.y * 0.0078125));
        if (keep > densityFactor) return;
    }

    vec3 worldUp  = vec3(0.0, 1.0, 0.0);
    float hs = billboardScale * 0.5;
    float h = billboardScale;
    // 45 degree inclination: tan(45) = 1, so top offset equals height.
    float tilt = h;
    int   ti = fragTexIndexIn[0];
    float layer = fragTexCoordIn[0].z;

    vec3 outwardDirs[4] = vec3[4](
        vec3( 1.0, 0.0, 0.0),
        vec3( 0.0, 0.0, 1.0),
        vec3(-1.0, 0.0, 0.0),
        vec3( 0.0, 0.0,-1.0)
    );

    vec3 tangents[6] = vec3[6](
        vec3( 0.0, 0.0, 1.0),
        vec3(-1.0, 0.0, 0.0),
        vec3( 0.0, 0.0,-1.0),
        vec3( 1.0, 0.0, 0.0),
        vec3( 1.0, 0.0, 0.0),
        vec3( 0.0, 0.0, 1.0)
    );

    // 4 planes with 45-degree inclination rotated laterally.
    for (int p = 0; p < 4; ++p) {
        vec3 tangent = tangents[p];
        vec3 outward = outwardDirs[p];

        vec3 bl = worldPos - tangent * hs + applyWindSkew(worldPos, tangent, 0.0);
        vec3 br = worldPos + tangent * hs + applyWindSkew(worldPos, tangent, 0.0);
        vec3 tl = worldPos - tangent * hs + worldUp * h + outward * tilt + applyWindSkew(worldPos, tangent, 1.0);
        vec3 tr = worldPos + tangent * hs + worldUp * h + outward * tilt + applyWindSkew(worldPos, tangent, 1.0);

        // Plane normal = cross(tangent, planeUpDir).
        // For 45-degree tilt (tilt == h), planeUpDir = normalize(worldUp + outward).
        outPlaneNormal = normalize(cross(tangent, normalize(worldUp + outward)));
        outTangentWS   = tangent;

        gl_Position = ubo.viewProjection * vec4(bl, 1.0); inTexCoord = vec3(0.0, 1.0, layer); inTexIndex = ti; outWorldPos = bl; EmitVertex();
        gl_Position = ubo.viewProjection * vec4(br, 1.0); inTexCoord = vec3(1.0, 1.0, layer); inTexIndex = ti; outWorldPos = br; EmitVertex();
        gl_Position = ubo.viewProjection * vec4(tl, 1.0); inTexCoord = vec3(0.0, 0.0, layer); inTexIndex = ti; outWorldPos = tl; EmitVertex();
        gl_Position = ubo.viewProjection * vec4(tr, 1.0); inTexCoord = vec3(1.0, 0.0, layer); inTexIndex = ti; outWorldPos = tr; EmitVertex();
        EndPrimitive();
    }

    // 2 vertical planes perpendicular to ground and perpendicular to each other.
    for (int p = 4; p < 6; ++p) {
        vec3 tangent = tangents[p];

        vec3 bl = worldPos - tangent * hs + applyWindSkew(worldPos, tangent, 0.0);
        vec3 br = worldPos + tangent * hs + applyWindSkew(worldPos, tangent, 0.0);
        vec3 tl = worldPos - tangent * hs + worldUp * h + applyWindSkew(worldPos, tangent, 1.0);
        vec3 tr = worldPos + tangent * hs + worldUp * h + applyWindSkew(worldPos, tangent, 1.0);

        // Vertical plane: planeUpDir = worldUp, so normal = cross(tangent, worldUp).
        outPlaneNormal = normalize(cross(tangent, worldUp));
        outTangentWS   = tangent;

        gl_Position = ubo.viewProjection * vec4(bl, 1.0); inTexCoord = vec3(0.0, 1.0, layer); inTexIndex = ti; outWorldPos = bl; EmitVertex();
        gl_Position = ubo.viewProjection * vec4(br, 1.0); inTexCoord = vec3(1.0, 1.0, layer); inTexIndex = ti; outWorldPos = br; EmitVertex();
        gl_Position = ubo.viewProjection * vec4(tl, 1.0); inTexCoord = vec3(0.0, 0.0, layer); inTexIndex = ti; outWorldPos = tl; EmitVertex();
        gl_Position = ubo.viewProjection * vec4(tr, 1.0); inTexCoord = vec3(1.0, 0.0, layer); inTexIndex = ti; outWorldPos = tr; EmitVertex();
        EndPrimitive();
    }
}
