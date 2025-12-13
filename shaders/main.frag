#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 5) flat in int fragTexIndex;
layout(location = 4) in vec3 fragPosWorld;
layout(location = 6) in vec4 fragPosLightSpace;
layout(location = 9) in vec4 fragTangent;

#include "includes/ubo.glsl"

#include "includes/textures.glsl"

layout(location = 0) out vec4 outColor;

#include "includes/common.glsl"

#include "includes/shadows.glsl"

void main() {
    // Always use layer 0 since each descriptor set points to different textures
    int texIndex = 0;

    vec2 uv = fragUV;

    // Geometry normal (world-space)
    vec3 N = normalize(fragNormal);
    vec3 worldNormal = N;
    // Tangent & bitangent (world-space) from vertex tangents (preferred), otherwise compute via derivatives
    vec3 T = vec3(0.0, 0.0, 0.0);
    vec3 B = vec3(0.0, 0.0, 0.0);
    bool haveTB = false;
    // Sample albedo texture
    vec3 albedoColor = texture(albedoArray, vec3(uv, float(texIndex))).rgb;

    // Compute normal mapping if enabled (per-material or global toggle)
    if (ubo.mappingParams.x > 0.5 || ubo.materialFlags.w > 0.5) {
        // Prefer vertex tangent if available
        if (length(fragTangent.xyz) > 1e-6) {
            vec3 t = normalize(fragTangent.xyz);
            vec3 b = normalize(cross(N, t) * fragTangent.w);
            T = t; B = b; haveTB = true;
            // Sample tangent-space normal from texture and transform to world-space
            vec3 nmap = texture(normalArray, vec3(uv, float(texIndex))).rgb * 2.0 - 1.0;
            worldNormal = normalize(nmap.x * T + nmap.y * B + nmap.z * N);
        } else {
            // Use screen-space derivatives to compute tangent & bitangent in world-space
            vec3 dp1 = dFdx(fragPosWorld);
            vec3 dp2 = dFdy(fragPosWorld);
            vec2 duv1 = dFdx(fragUV);
            vec2 duv2 = dFdy(fragUV);
            float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (abs(det) > 1e-6) {
                float r = 1.0 / det;
                vec3 t = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
                vec3 b = normalize((dp2 * duv1.x - dp1 * duv2.x) * r);
                // Orthonormalize against geometry normal
                t = normalize(t - N * dot(N, t));
                b = normalize(cross(N, t));
                // Sample tangent-space normal from texture and transform to world-space
                vec3 nmap = texture(normalArray, vec3(uv, float(texIndex))).rgb * 2.0 - 1.0;
                worldNormal = normalize(nmap.x * t + nmap.y * b + nmap.z * N);
                T = t; B = b; haveTB = true;
            }
        }
    }

    // Lighting calculation
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, toLight), 0.0);

    // Shadow calculation
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    float shadow = 0.0;
    if (ubo.shadowEffects.w > 0.5) {
        if (NdotL > 0.01) {
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, bias);
        } else {
            shadow = 1.0;
        }
    }
    float totalShadow = shadow;

    vec3 ambient = albedoColor * ubo.materialFlags.z;
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);

    // Specular
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), ubo.specularParams.y);
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * ubo.specularParams.x;

    // Debug visualisation modes (0 = normal render)
    int debugMode = int(ubo.debugParams.x + 0.5);
    if (debugMode == 1) {
        vec3 gn = normalize(fragNormal);
        outColor = vec4(gn * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 2) {
        vec3 nm = normalize(worldNormal);
        outColor = vec4(nm * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 3) {
        vec2 fuv = fract(fragUV);
        outColor = vec4(fuv.x, fuv.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 4) {
        // Show tangent if available, else geometry normal
        vec3 tcol;
        if (haveTB) {
            tcol = normalize(T) * 0.5 + 0.5;
        } else {
            tcol = normalize(N) * 0.5 + 0.5;
        }
        outColor = vec4(tcol, 1.0);
        return;
    }
    if (debugMode == 5) {
        // Show bitangent if available, else geometry normal
        vec3 bcol;
        if (haveTB) {
            bcol = normalize(B) * 0.5 + 0.5;
        } else {
            bcol = normalize(N) * 0.5 + 0.5;
        }
        outColor = vec4(bcol, 1.0);
        return;
    }
    if (debugMode == 6) {
        vec3 ncol = normalize(N) * 0.5 + 0.5;
        outColor = vec4(ncol, 1.0);
        return;
    }
    if (debugMode == 7) {
        vec3 rawAlbedo = texture(albedoArray, vec3(fragUV, float(texIndex))).rgb;
        outColor = vec4(rawAlbedo, 1.0);
        return;
    }
    if (debugMode == 8) {
        vec3 rawNormalTex = texture(normalArray, vec3(fragUV, float(texIndex))).rgb;
        outColor = vec4(rawNormalTex, 1.0);
        return;
    }
    if (debugMode == 9) {
        float h = texture(heightArray, vec3(fragUV, float(texIndex))).r;
        outColor = vec4(vec3(h), 1.0);
        return;
    }
    if (debugMode == 10) {
        outColor = vec4(NdotL, totalShadow, 0.0, 1.0);
        return;
    }
    if (debugMode == 11) {
        vec3 normalToShow = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
        outColor = vec4(normalToShow * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 12) {
        vec3 tl = normalize(toLight);
        outColor = vec4(tl * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 13) {
        outColor = vec4(vec3(NdotL), 1.0);
        return;
    }
    if (debugMode == 14) {
        outColor = vec4(shadow, 0.0, totalShadow, 1.0);
        return;
    }

    // Fallback to normal rendering
    outColor = vec4(ambient + diffuse + specular, 1.0);
}
