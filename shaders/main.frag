#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec4 fragTangent;
layout(location = 5) flat in int fragTexIndex;
layout(location = 4) in vec3 fragPosWorld;
layout(location = 6) in vec4 fragPosLightSpace;

#include "includes/ubo.glsl"

#include "includes/textures.glsl"

layout(location = 0) out vec4 outColor;

#include "includes/common.glsl"

#include "includes/parallax.glsl"

#include "includes/shadows.glsl"

void main() {
    // Always use layer 0 since each descriptor set points to different textures
    int texIndex = 0;
    
    vec2 uv = fragUV;
    vec3 viewDirT = vec3(0.0, 0.0, 1.0);
    
    // Prepare adjusted POM parameters (default = material values)
    float adjHeightScale = ubo.pomParams.x;
    float adjMinLayers = ubo.pomParams.y;
    float adjMaxLayers = ubo.pomParams.z;

    // Parallax Occlusion Mapping (if enabled AND mapping mode == parallax)
    int mappingMode = int(ubo.mappingParams.x + 0.5);
    if (mappingMode == 1 && ubo.pomParams.w > 0.5) {
        // Compute tangent-space view direction for POM
        vec3 N = normalize(fragNormal);
        vec3 T = normalize(fragTangent.xyz);
        float vertHand = fragTangent.w;

        // Handle normal Y flipping
        if (ubo.pomFlags.x > 0.5) {
            N.y = -N.y;
        }
        
        // Compute bitangent using per-vertex handedness (and optional global flip)
        float handed = vertHand * (ubo.pomFlags.y > 0.5 ? -1.0 : 1.0);
        vec3 B = cross(N, T) * handed;
        // TBN matrix transforms from tangent space to world space
        mat3 TBN = mat3(T, B, N);
        
        vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
        vec3 viewDirT = normalize(transpose(TBN) * viewDir);

        // Compute smooth LOD fade in the fragment shader using the parallaxLOD values
        // parallaxLOD.x = near, parallaxLOD.y = far, parallaxLOD.z = reductionAtFar
        float parallaxNear = ubo.parallaxLOD.x;
        float parallaxFar = ubo.parallaxLOD.y;
        float reductionAtFar = ubo.parallaxLOD.z;

        // compute world-space distance from camera to fragment
        float dist = length(ubo.viewPos.xyz - fragPosWorld);
        // t = 0 -> near (full detail), t = 1 -> far (reduced)
        float t = 0.0;
        if (parallaxFar > parallaxNear) {
            t = clamp((dist - parallaxNear) / (parallaxFar - parallaxNear), 0.0, 1.0);
        }
        float lodFactor = mix(1.0, reductionAtFar, t);

        // apply smooth scaling to height and layers (keep at least 1 layer)
        adjHeightScale = max(0.0, adjHeightScale * lodFactor);
        adjMinLayers = max(1.0, adjMinLayers * lodFactor);
        adjMaxLayers = max(1.0, adjMaxLayers * lodFactor);

        // Apply parallax occlusion mapping with adjusted parameters computed on GPU
        uv = ParallaxOcclusionMapping(fragUV, viewDirT, texIndex, adjHeightScale, adjMinLayers, adjMaxLayers);
    }
    
    // Compute geometry normal
    vec3 N = normalize(fragNormal);
    
    // Handle normal Y flipping for geometry normal (consistent with POM block)
    if (ubo.pomFlags.x > 0.5) {
        N.y = -N.y;
    }
    
    vec3 T = normalize(fragTangent.xyz);
    float vertHand_main = fragTangent.w;
    float handed_main = vertHand_main * (ubo.pomFlags.y > 0.5 ? -1.0 : 1.0);
    vec3 B = cross(N, T) * handed_main;
    mat3 TBN = mat3(T, B, N);
    
    // Compute world-space normal
    vec3 worldNormal = N; // Default to geometry normal
    
    // Apply normal mapping if enabled (mappingMode >= 1)
    if (mappingMode >= 1) {
        vec3 normalMap = texture(normalArray, vec3(uv, float(texIndex))).rgb;
        
        // Transform normal from [0,1] to [-1,1] range
        normalMap = normalize(normalMap * 2.0 - 1.0);
        
        // Handle normal Y flipping for normal map
        if (ubo.pomFlags.x > 0.5) {
            normalMap.y = -normalMap.y;
        }
        worldNormal = normalize(TBN * normalMap);
    }
    
    // Sample albedo texture (after UV displacement if parallax was applied)
    vec3 albedoColor = texture(albedoArray, vec3(uv, float(texIndex))).rgb;

    // Lighting calculation
    // ubo.lightDir is sent as a vector FROM the light TOWARD the surface (light->surface).
    // Negate it here to get the conventional surface->light vector used for lighting calculations.
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    
    // Use normal-mapped normal for lighting calculations
    float NdotL = max(dot(N, toLight), 0.0);
    
    // Adjust shadow position based on parallax displacement (if enabled)
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    if (int(ubo.mappingParams.x + 0.5) == 1 && ubo.pomParams.w > 0.5 && ubo.shadowEffects.y > 0.5) { // Check shadowDisplacement setting
        // Get height at the parallax-displaced UV
        float height = sampleHeight(uv, texIndex);
        // Use the adjusted heightScale computed earlier so shadow displacement follows the GPU LOD
        float heightScale = adjHeightScale;

        // Use parallaxFade (computed earlier) to scale shadow displacement; avoid shifting shadows when parallax is disabled
        float parallaxFadeLocal = clamp((viewDirT.z - 0.12) / (0.35 - 0.12), 0.0, 1.0);

        // Offset world position along the geometry normal by the height displacement (scaled by fade)
        // Use geometry normal (not normal-map detail) so shadow displacement follows actual geometry.
        // Height is interpreted so larger values mean deeper displacement into the surface;
        // move the shadow sample point along -N to project the displaced surface toward the light correctly.
        vec3 offset = N * height * heightScale * parallaxFadeLocal;
        vec3 adjustedWorldPos = fragPosWorld - offset;

        // Transform adjusted position to light space
        adjustedPosLightSpace = ubo.lightSpaceMatrix * vec4(adjustedWorldPos, 1.0);
    }
    
    // Calculate shadow with adaptive bias based on surface angle.
    // If global shadows are disabled (ubo.shadowEffects.w == 0) we skip shadowing entirely.
    float shadow = 0.0;
    float selfShadow = 1.0;

    if (ubo.shadowEffects.w > 0.5) {
        // Only calculate shadows for surfaces facing the light (use normal-mapped normal)
        if (NdotL > 0.01) {
            // Increased bias since shadow map no longer uses parallax displacement
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, bias);

            // Add parallax self-shadowing (if POM and self-shadowing are enabled)
            if (int(ubo.mappingParams.x + 0.5) == 1 && ubo.pomParams.w > 0.5 && ubo.shadowEffects.x > 0.5) { // Check selfShadowing setting
                // Transform light direction to tangent space
                vec3 lightDirT = normalize(transpose(TBN) * toLight);

                // Get current height at the parallax-offset UV
                float currentHeight = sampleHeight(uv, texIndex);

                // Only compute self-shadow if light is reasonably above the surface
                // Use a small threshold to avoid computing shadows for grazing angles
                float parallaxFadeLocal = clamp((viewDirT.z - 0.12) / (0.35 - 0.12), 0.0, 1.0);
                if (lightDirT.z > 0.1 && parallaxFadeLocal > 0.01) {
                    // use adjusted heightScale and minLayers computed earlier for self-shadow ray-marching
                    float rawSelf = ParallaxSelfShadow(uv, lightDirT, currentHeight, texIndex, adjHeightScale, adjMinLayers);
                    // Blend self-shadow with no-shadow (1.0) based on parallax fade so it smoothly appears
                    selfShadow = mix(1.0, rawSelf, parallaxFadeLocal);
                }
                // If light is at grazing angle or below, just don't apply self-shadow (leave at 1.0)
            }
        } else {
            // Back-facing surfaces should be in full shadow
            shadow = 1.0;
        }
    } else {
        // Shadows globally disabled: leave shadow=0 (fully lit) and selfShadow=1 (no self-shadow)
        shadow = 0.0;
        selfShadow = 1.0;
    }
    
    // Combine global shadow and self-shadow (multiply to get darker result)
    float totalShadow = max(shadow, 1.0 - selfShadow);
    
    vec3 ambient = albedoColor * ubo.pomFlags.z; // ambient factor
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);
    
    // Specular lighting (Phong) - uses normal-mapped surface normal
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), ubo.specularParams.y); // shininess from uniform
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * ubo.specularParams.x; // specular strength from uniform
    
    // Debug visualisation modes (0 = normal render)
    int debugMode = int(ubo.debugParams.x + 0.5);
    if (debugMode == 0) {
        outColor = vec4(ambient + diffuse + specular, 1.0);
        return;
    }
    if (debugMode == 1) {
        // Geometry normal (world-space geometry normal)
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
        // G = self-shadow occlusion amount (1.0 - selfShadow),
        // B = combined totalShadow used to darken lighting.
        outColor = vec4(shadow, 1.0 - selfShadow, totalShadow, 1.0);
        return;
    }
    if (debugMode == 11) {
        // Normal computed from derivatives of world position (dFdx/dFdy)
        vec3 dPdx = dFdx(fragPosWorld);
        vec3 dPdy = dFdy(fragPosWorld);
        vec3 derivNormal = normalize(cross(dPdx, dPdy));
        outColor = vec4(derivNormal * 0.5 + 0.5, 1.0);
        return;
    }
    // Fallback to normal rendering
    outColor = vec4(ambient + diffuse + specular, 1.0);
}
