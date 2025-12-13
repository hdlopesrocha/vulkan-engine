#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 sharpNormal;
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
    vec3 viewDirT = vec3(0.0, 0.0, 1.0);
    float adjHeightScale = 0.0;
    float adjMinLayers = 1.0;
    float adjMaxLayers = 1.0;

    // Compute geometry normal
    vec3 N = normalize(fragNormal);
    
    // Compute tangent and bitangent from derivatives (no vertex tangent attribute)
    vec3 dpdx = dFdx(fragPosWorld);
    vec3 dpdy = dFdy(fragPosWorld);
    vec2 duvdx = dFdx(fragUV);
    vec2 duvdy = dFdy(fragUV);
    float denom = duvdx.x * duvdy.y - duvdy.x * duvdx.y;
    vec3 T;
    if (abs(denom) < 1e-6) {
        // Robust fallback tangent: pick any vector orthogonal to N
        if (abs(N.z) < 0.9) T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
        else T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    } else {
        float r = 1.0 / denom;
        T = (dpdx * duvdy.y - dpdy * duvdx.y) * r;
        // Orthonormalize T against N (Gram-Schmidt)
        T = normalize(T - N * dot(N, T));
        // If orthonormalization produced a degenerate tangent, fall back
        if (length(T) < 1e-6) {
            if (abs(N.z) < 0.9) T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
            else T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
        }
    }
    vec3 B = normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
    
    // Compute world-space normal
    vec3 worldNormal = N; // Default to geometry normal
    
    // Apply normal mapping if enabled (global setting stored in materialFlags.w)
    bool mappingEnabled = (ubo.materialFlags.w > 0.5);
    if (mappingEnabled) {
        vec3 normalMap = texture(normalArray, vec3(uv, float(texIndex))).rgb;
        
        // Transform normal from [0,1] to [-1,1] range
        normalMap = normalize(normalMap * 2.0 - 1.0);
        
        worldNormal = normalize(TBN * normalMap);
    }
    
    // Sample albedo texture (after UV displacement if any height-based displacement was applied)
    vec3 albedoColor = texture(albedoArray, vec3(uv, float(texIndex))).rgb;

    // Lighting calculation
    // ubo.lightDir is sent as a vector FROM the light TOWARD the surface (light->surface).
    // Negate it here to get the conventional surface->light vector used for lighting calculations.
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    
    // Use normal-mapped normal for lighting calculations
        float NdotL = max(dot(worldNormal, toLight), 0.0);
    
    // Adjust shadow position based on height displacement (disabled)
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    // Displacement-based shadow adjustment removed: do not modify adjustedPosLightSpace

    // Calculate shadow with adaptive bias based on surface angle.
    // If global shadows are disabled (ubo.shadowEffects.w == 0) we skip shadowing entirely.
    float shadow = 0.0;

    if (ubo.shadowEffects.w > 0.5) {
        // Only calculate shadows for surfaces facing the light (use normal-mapped normal)
        if (NdotL > 0.01) {
            // Increased bias since shadow map no longer uses displacement adjustments
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, bias);
        } else {
            // Back-facing surfaces should be in full shadow
            shadow = 1.0;
        }
    }
    
    // Combine global shadow
    float totalShadow = shadow;
    
    vec3 ambient = albedoColor * ubo.materialFlags.z; // ambient factor
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);
    
    // Specular lighting (Phong) - uses normal-mapped surface normal
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), ubo.specularParams.y); // shininess from uniform
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * ubo.specularParams.x; // specular strength from uniform
    
    // Debug visualisation modes (0 = normal render)
    int debugMode = int(ubo.debugParams.x + 0.5);
    if (debugMode == 1) {
        // Fragment geometry normal (world-space geometry normal)
        vec3 gn = normalize(fragNormal);
        outColor = vec4(gn * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 2) {
        // Normal map visualised in world space
        vec3 nm = normalize(worldNormal);
        outColor = vec4(nm * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 3) {
        // UV coordinates (wrap using fract)
        vec2 fuv = fract(fragUV);
        outColor = vec4(fuv.x, fuv.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 4) {
        // Tangent (world-space)
        vec3 tcol = normalize(T) * 0.5 + 0.5;
        outColor = vec4(tcol, 1.0);
        return;
    }
    if (debugMode == 5) {
        // Bitangent (world-space)
        vec3 bcol = normalize(B) * 0.5 + 0.5;
        outColor = vec4(bcol, 1.0);
        return;
    }
    if (debugMode == 6) {
        // Normal (world-space)
        vec3 ncol = normalize(N) * 0.5 + 0.5;
        outColor = vec4(ncol, 1.0);
        return;
    }
    if (debugMode == 7) {
        // Raw albedo texture (no lighting)
        vec3 rawAlbedo = texture(albedoArray, vec3(fragUV, float(texIndex))).rgb;
        outColor = vec4(rawAlbedo, 1.0);
        return;
    }
    if (debugMode == 8) {
        // Raw normal map colors (as stored in the texture, no remap)
        vec3 rawNormalTex = texture(normalArray, vec3(fragUV, float(texIndex))).rgb;
        outColor = vec4(rawNormalTex, 1.0);
        return;
    }
    if (debugMode == 9) {
        // Bump/height map visualization (grayscale)
        float h = texture(heightArray, vec3(fragUV, float(texIndex))).r;
        outColor = vec4(vec3(h), 1.0);
        return;
    }
    if (debugMode == 10) {
        // Lighting comparison: NdotL and shadow
        outColor = vec4(NdotL, totalShadow, 0.0, 1.0);
        return;
    }
    if (debugMode == 11) {
        outColor = vec4(sharpNormal * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 12) {
        // Visualize surface->light vector (mapped to 0..1) to inspect direction
        vec3 tl = normalize(toLight);
        outColor = vec4(tl * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 13) {
        // Show NdotL as grayscale (how much each fragment is lit)
        outColor = vec4(vec3(NdotL), 1.0);
        return;
    }
    if (debugMode == 14) {
        // Shadow diagnostics visualization:
        // R = global shadow (from shadow map sampling),
        // B = combined totalShadow used to darken lighting.
        outColor = vec4(shadow, 0.0, totalShadow, 1.0);
        return;
    }

    // Fallback to normal rendering
    outColor = vec4(ambient + diffuse + specular, 1.0);
}
