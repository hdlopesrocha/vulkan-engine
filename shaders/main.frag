#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 5) flat in int fragTexIndex;
layout(location = 4) in vec3 fragPosWorld;
layout(location = 6) in vec4 fragPosLightSpace;

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

    // Sample albedo texture
    vec3 albedoColor = texture(albedoArray, vec3(uv, float(texIndex))).rgb;

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
        // Tangent unavailable — show geometry normal instead
        vec3 tcol = normalize(N) * 0.5 + 0.5;
        outColor = vec4(tcol, 1.0);
        return;
    }
    if (debugMode == 5) {
        // Bitangent unavailable — show geometry normal instead
        vec3 bcol = normalize(N) * 0.5 + 0.5;
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
