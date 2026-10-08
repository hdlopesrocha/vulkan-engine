#version 450

// ── Brush preview raymarcher ────────────────────────────────────────────────
// Renders the selected brush's SDF volume into the offscreen brush color +
// depth targets (composited by PostProcess, depth-tested against the scene).
// This replaces the old tessellated brush-octree raster pass: the preview is
// the SDF itself, so it always matches the shape applied to the octree.
//
// Shading mirrors the solid terrain shader (SolidSurface.glsl) for a single
// material: the brush's material index drives the same triplanar material
// arrays (albedo/normal/roughness/AO) and the same direct-light + HSV tint
// formulas, minus what cannot apply to preview geometry (CSM/RT/cloud
// shadows, reflections) — the old BRUSH_PASS variant skipped those too.
//
// The fullscreen pass extends the inherited ray/triangle helpers only through
// Fullscreen.vert; the ray is reconstructed from invViewProjection.

#include "../../includes/Locations.glsl"
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Textures.glsl"
#include "../../includes/Common.glsl"
#include "../../includes/solid/TangentBasis.glsl"
#include "../../includes/solid/Triplanar.glsl"
#include "../../includes/Hsv.glsl"
#include "../../includes/brush/BrushSdf.glsl"

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

const int kBrushMaxSteps = 192;

bool brushRaySphere(vec3 ro, vec3 rd, vec3 c, float r, out float t0, out float t1) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float cc = dot(oc, oc) - r * r;
    float h = b * b - cc;
    if (h < 0.0) return false;
    h = sqrt(h);
    t0 = -b - h;
    t1 = -b + h;
    return t1 > 0.0;
}

// 4-tap tetrahedral gradient (same scheme as SdfRenderer.frag).
vec3 brushGradient(vec3 p, float h) {
    vec2 e = vec2(h, 0.0);
    vec3 n = vec3(
        brushDistance(p + e.xyy) - brushDistance(p - e.xyy),
        brushDistance(p + e.yxy) - brushDistance(p - e.yxy),
        brushDistance(p + e.yyx) - brushDistance(p - e.yyx));
    float l = length(n);
    return (l > 1e-9) ? n / l : vec3(0.0, 1.0, 0.0);
}

// Single-material copy of the SolidSurface material path: triplanar sampling
// from the shared arrays when the material enables it, dominant-axis planar
// mapping otherwise (SDF surfaces carry no mesh UVs).
vec3 shadeBrushSurface(vec3 P, vec3 geomN) {
    int mi = clamp(int(brush.xFlags.z), 0, max(textureSize(albedoArray, 0).z - 1, 0));
    vec3 N = normalize(geomN);
    vec3 worldNormal = N;

    vec3 triW = abs(geomN);
    float t = ubo.triplanarThreshold;
    vec3 wt = max(vec3(0.0), triW - vec3(t));
    float e = max(1.0, ubo.triplanarExponent);
    if (e > 1.0) wt = pow(wt, vec3(e));
    triW = wt / (wt.x + wt.y + wt.z + 1e-6);

    vec2 uvX, uvY, uvZ;
    computeTriplanarUVs(P, mi, geomN, uvX, uvY, uvZ);

    MaterialNamed mn = materialNamed(materials[mi]);
    bool triplanar = mn.triplanarEnabled;
    float dx = step(triW.y, triW.x) * step(triW.z, triW.x);
    float dy = (1.0 - dx) * step(triW.z, triW.y);
    vec2 planarUV = uvX * dx + uvY * dy + uvZ * (1.0 - dx - dy);

    vec3 albedoColor;
    if (triplanar) {
        albedoColor = computeTriplanarAlbedoUVs(triW, mi, uvX, uvY, uvZ);
        if (mn.mappingEnabled || ubo.normalMappingEnabled != 0u) {
            worldNormal = computeTriplanarNormalBlended(triW, mi, N, uvX, uvY, uvZ);
        }
    } else {
        albedoColor = texture(albedoArray, vec3(planarUV, float(mi))).rgb;
        if (mn.mappingEnabled || ubo.normalMappingEnabled != 0u) {
            worldNormal = computeProjectionNormal(planarUV, mi, N);
        }
    }

    float roughnessValue;
    float ambientOcclusion;
    if (triplanar) {
        roughnessValue = computeTriplanarRoughnessDominant(triW, mi, uvX, uvY, uvZ);
        ambientOcclusion = computeTriplanarAODominant(triW, mi, uvX, uvY, uvZ);
    } else {
        roughnessValue = texture(roughnessArray, vec3(planarUV, float(mi))).r;
        ambientOcclusion = texture(aoArray, vec3(planarUV, float(mi))).r;
    }
    if (ubo.roughnessEnabled == 0u) roughnessValue = 0.0;

    // Direct lighting: same formulas as SolidSurface.glsl (no CSM/RT/clouds).
    vec3 toLight = -normalize(ubo.lightDirection);
    float NdotL = max(dot(worldNormal, toLight), 0.0);
    vec4 materialFlags = materials[mi].materialFlags;
    vec4 roughnessAOParams = materials[mi].roughnessAOParams;
    float useAOf = roughnessAOParams.z;
    float aoFactor = roughnessAOParams.y;
    float roughnessFactor = roughnessAOParams.x;
    float aoBlend = (useAOf > 0.5 && ubo.ambientOcclusionEnabled != 0u) ? ambientOcclusion : 1.0;
    aoBlend = mix(1.0, aoBlend, aoFactor);
    vec3 ambient = albedoColor * materialFlags.z * aoBlend;
    vec3 diffuse = albedoColor * ubo.lightColor * NdotL;
    vec3 viewDir = normalize(ubo.viewPosition - P);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    vec4 specularParams = materials[mi].specularParams;
    float shininess = max(specularParams.y, 1.0);
    float specPower = max(mix(shininess, 1.0, roughnessValue * roughnessFactor), 1.0);
    float spec = (NdotL > 0.0) ? pow(max(dot(viewDir, reflectDir), 0.0), specPower) : 0.0;
    vec3 specular = ubo.lightColor * spec * specularParams.x;
    vec3 finalColor = ambient + diffuse + specular;

    // HSV tint: identical to SolidSurface.glsl's per-vertex HSV application.
    vec3 hsvColor = brush.xHsv.xyz;
    if (hsvColor.x != 0.0 || hsvColor.y != 0.5 || hsvColor.z != 0.5) {
        vec3 texHSV = rgbToHsv(finalColor);
        texHSV.x = mod(texHSV.x + hsvColor.x, 360.0);
        texHSV.y = clamp(texHSV.y * (hsvColor.y * 2.0), 0.0, 1.0);
        texHSV.z *= hsvColor.z * 2.0;
        finalColor = hsvToRgb(texHSV);
    }
    return finalColor;
}

void main() {
    // radius 0 = no brush selected / nothing to preview.
    if (brush.xBounds.w <= 0.0) discard;

    vec2 uv = gl_FragCoord.xy / max(brush.xViewport.xy, vec2(1.0));
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 farPoint = ubo.invViewProjection * vec4(ndc, 1.0, 1.0);
    if (abs(farPoint.w) < 1e-8) discard;
    vec3 ro = ubo.viewPosition;
    vec3 rd = normalize(farPoint.xyz / farPoint.w - ro);

    float t0, t1;
    if (!brushRaySphere(ro, rd, brush.xBounds.xyz, brush.xBounds.w, t0, t1)) discard;
    t0 = max(t0, 0.0);
    if (t0 >= t1) discard;

    float eps = max(brush.xBounds.w * 1e-4, 0.01);
    // Front-surface-only preview: from inside the volume there is no front
    // face, matching the old front-face raster pass (the composite hides the
    // overlay where no surface depth was written).
    if (brushDistance(ro + rd * (t0 + eps)) < 0.0) discard;

    float t = t0;
    bool hit = false;
    for (int i = 0; i < kBrushMaxSteps; ++i) {
        vec3 p = ro + rd * t;
        float d = brushDistance(p);
        if (d < eps) { hit = true; break; }
        t += max(d, eps);
        if (t > t1) break;
    }
    if (!hit) discard;

    vec3 hp = ro + rd * t;
    vec3 geomN = brushGradient(hp, max(eps * 2.0, brush.xBounds.w * 5e-4));
    if (dot(geomN, rd) > 0.0) geomN = -geomN;  // always face the viewer

    vec3 color = shadeBrushSurface(hp, geomN);

    vec4 clip = ubo.viewProjection * vec4(hp, 1.0);
    float depth = (clip.w > 1e-6) ? clamp(clip.z / clip.w, 0.0, 1.0) : 1.0;
    gl_FragDepth = depth;
    outColor = vec4(color, 1.0);
}
