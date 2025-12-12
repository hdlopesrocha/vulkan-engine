#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragTangent;
layout(location = 5) flat in int fragTexIndex;
layout(location = 4) in vec3 fragPosWorld;
layout(location = 6) in vec4 fragPosLightSpace;

#include "includes/ubo.glsl"

#include "includes/textures.glsl"

layout(location = 0) out vec4 outColor;

#include "includes/common.glsl"

vec2 ParallaxOcclusionMapping(vec2 texCoords, vec3 viewDirT, int texIndex, float heightScale, float minLayers, float maxLayers) {
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0,0.0,1.0), viewDirT)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    // parallax direction: when flipParallaxDirection is OFF (0.0), black carves inward (standard)
    // when flipParallaxDirection is ON (1.0), black goes outward (inverted)
    vec2 P = (ubo.pomFlags.w > 0.5) ? -viewDirT.xy * heightScale : viewDirT.xy * heightScale;
    vec2 deltaTex = P / numLayers;
    vec2 currentTex = texCoords;
    // Sample height (respect per-material interpretation)
    float currentDepthMapValue = sampleHeight(currentTex, texIndex);
    vec2 prevTex = currentTex;
    float prevDepth = currentDepthMapValue;
    // linear search through layers
    for (int i = 0; i < int(numLayers); ++i) {
        prevTex = currentTex;
        prevDepth = currentDepthMapValue;
        currentTex -= deltaTex;
    currentDepthMapValue = sampleHeight(currentTex, texIndex);
        currentLayerDepth += layerDepth;
        if (currentLayerDepth >= currentDepthMapValue) break;
    }
    // linear interpolation between prev and current (initial estimate)
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = prevDepth - (currentLayerDepth - layerDepth);
    float weight = 0.0;
    if ((afterDepth - beforeDepth) != 0.0) weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTex = mix(prevTex, currentTex, weight);

    // Binary-search refinement between prevTex and currentTex to improve intersection precision
    // Use a small fixed number of iterations (5) for a good quality/cost tradeoff.
    vec2 lowTex = prevTex;
    vec2 highTex = currentTex;
    float lowLayerDepth = currentLayerDepth - layerDepth;
    float highLayerDepth = currentLayerDepth;

    const int NUM_BINARY_ITERS = 5;
    for (int i = 0; i < NUM_BINARY_ITERS; ++i) {
        vec2 midTex = (lowTex + highTex) * 0.5;
    float midDepthMap = sampleHeight(midTex, texIndex);
        float midLayerDepth = (lowLayerDepth + highLayerDepth) * 0.5;
        if (midLayerDepth >= midDepthMap) {
            // intersection is between low and mid
            highTex = midTex;
            highLayerDepth = midLayerDepth;
        } else {
            // intersection is between mid and high
            lowTex = midTex;
            lowLayerDepth = midLayerDepth;
        }
    }

    // take midpoint of refined interval as final UV
    finalTex = (lowTex + highTex) * 0.5;
    return finalTex; // Don't clamp - let sampler repeat mode handle tiling
}

// Parallax self-shadowing: ray-march from surface point towards light
// Returns 0.0 (full shadow) to 1.0 (no shadow)
// Height convention: black (texture.r=0) = deep (1.0 after inversion), white (texture.r=1) = surface (0.0 after inversion)
// ParallaxSelfShadow now accepts adjusted, per-fragment heightScale and minLayers
float ParallaxSelfShadow(vec2 texCoords, vec3 lightDirT, float currentHeight, int texIndex, float heightScale, float minLayers) {
    float qualityMultiplier = ubo.shadowEffects.z; // quality setting from UI
    
    // Number of samples for shadow ray-march (adjusted by quality)
    float numSamples = minLayers * qualityMultiplier;
    numSamples = max(numSamples, 4.0); // At least 4 samples for decent quality
    
    // March in the direction of the light in texture space
    // Use the SAME logic as POM: respect the flipParallaxDirection flag
    // Negate X component
    vec2 lightXY = vec2(-lightDirT.x, lightDirT.y);
    vec2 L = (ubo.pomFlags.w > 0.5) ? -lightXY * heightScale : lightXY * heightScale;
    vec2 rayStep = L / numSamples;
    
    // Start from current displaced position
    vec2 currentTexCoords = texCoords;
    
    // Height system: 0.0 = surface (white), 1.0 = deepest (black)
    // We're at currentHeight and march towards the light
    float rayHeight = currentHeight;
    
    // Height change per step - as we go towards light (up), height decreases
    float layerDepth = lightDirT.z / numSamples;
    
    // March from current position towards the light source
    for (int i = 1; i <= int(numSamples); ++i) {
        // Move along the ray
        currentTexCoords -= rayStep; // Subtract like POM does
        rayHeight -= layerDepth; // Move up (decrease depth)
        
        // If we've reached the surface, no more shadow
        if (rayHeight <= 0.0) break;
        
        // Sample height at this position
        float sampledHeight = sampleHeight(currentTexCoords, texIndex);
        
        // If sampled point is shallower (less deep) than our ray, it blocks the light
        float bias = 0.02;
        if (sampledHeight < rayHeight - bias) {
            // Blocking geometry found
            float occlusionDepth = (rayHeight - sampledHeight) / heightScale;
            float shadowFactor = clamp(occlusionDepth * 2.0, 0.0, 1.0);
            return 1.0 - shadowFactor;
        }
    }
    
    // No occlusion - fully lit
    return 1.0;
}

// Shadow calculation with PCF (Percentage Closer Filtering)
float ShadowCalculation(vec4 fragPosLightSpace, float bias) {
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // With GLM_FORCE_DEPTH_ZERO_TO_ONE, Z is already in [0,1]
    // But XY still need transformation from [-1,1] to [0,1]
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    
    // Outside shadow map bounds = no shadow
    if(projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;
    
    // Get closest depth value from light's perspective
    float currentDepth = projCoords.z;
    
    // PCF (Percentage Closer Filtering) for soft shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    // Use larger 5x5 kernel to blur over gaps at cube edges
    for(int x = -2; x <= 2; ++x)
    {
        for(int y = -2; y <= 2; ++y)
        {
            vec2 offset = vec2(x, y) * texelSize;
            float pcfDepth = texture(shadowMap, projCoords.xy + offset).r;
            // Shadow test with bias to prevent shadow acne
            shadow += currentDepth > (pcfDepth + bias) ? 1.0 : 0.0;
        }
    }
    shadow /= 25.0; // Average the 25 samples (5x5)
    
    return shadow;
}

void main() {
    // Always use layer 0 since each descriptor set points to different textures
    int texIndex = 0;
    
    vec2 uv = fragUV;
    
    // Prepare adjusted POM parameters (default = material values)
    float adjHeightScale = ubo.pomParams.x;
    float adjMinLayers = ubo.pomParams.y;
    float adjMaxLayers = ubo.pomParams.z;

    // Parallax Occlusion Mapping (if enabled AND mapping mode == parallax)
    int mappingMode = int(ubo.mappingParams.x + 0.5);
    if (mappingMode == 1 && ubo.pomParams.w > 0.5) {
        // Compute tangent-space view direction for POM
        vec3 N = normalize(fragNormal);
        vec3 T = normalize(fragTangent);
        
        // Handle normal Y flipping
        if (ubo.pomFlags.x > 0.5) {
            N.y = -N.y;
        }
        
        // Handle tangent handedness flipping  
        if (ubo.pomFlags.y > 0.5) {
            T = -T;
        }
        
        // Compute bitangent correctly: cross(T, N) for right-handed coordinate system
        vec3 B = cross(T, N);
        // TBN matrix transforms from tangent space to world space
        // Columns are: tangent, bitangent, normal
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
    
    // Sample textures
    vec3 albedoColor = texture(albedoArray, vec3(uv, float(texIndex))).rgb;
    vec3 normalMap = texture(normalArray, vec3(uv, float(texIndex))).rgb;
    
    // Transform normal from [0,1] to [-1,1] range
    normalMap = normalMap * 2.0 - 1.0;
    
    // Handle normal Y flipping for normal map
    if (ubo.pomFlags.x > 0.5) {
        normalMap.y = -normalMap.y;
    }
    
    // Compute world-space normal from normal map
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    
    // Handle tangent handedness
    if (ubo.pomFlags.y > 0.5) {
        T = -T;
    }
    
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    vec3 worldNormal = normalize(TBN * normalMap);

    
    
    // Use geometry normal for shadow plane detection (not mapped normal)
    vec3 geometryNormal = normalize(fragNormal);
    
    // Lighting calculation
    // ubo.lightDir points from surface to light
    vec3 toLight = normalize(ubo.lightDir.xyz);
    
    // Use geometry normal for base lighting to get consistent light direction
    // Then add normal map detail on top
    float geometryNdotL = dot(geometryNormal, toLight);
    float normalMappedNdotL = max(dot(worldNormal, toLight), 0.0);
    
    // Blend between geometry-based and normal-mapped lighting
    // This gives consistent light direction while preserving normal map detail
    float baseLighting = max(geometryNdotL, 0.0);
    float detailLighting = normalMappedNdotL;
    // Use geometry lighting as base (70%), modulate with normal map detail (30%)
    float NdotL = baseLighting * 0.7 + detailLighting * 0.3;
    
    // Adjust shadow position based on parallax displacement (if enabled)
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    if (int(ubo.mappingParams.x + 0.5) == 1 && ubo.pomParams.w > 0.5 && ubo.shadowEffects.y > 0.5) { // Check shadowDisplacement setting
        // Get height at the parallax-displaced UV
        float height = sampleHeight(uv, texIndex);
        // Use the adjusted heightScale computed earlier so shadow displacement follows the GPU LOD
        float heightScale = adjHeightScale;

        // Offset world position along the normal by the height displacement
        // This makes shadows "follow" the height variations
        vec3 offset = worldNormal * height * heightScale;
        vec3 adjustedWorldPos = fragPosWorld + offset;

        // Transform adjusted position to light space
        adjustedPosLightSpace = ubo.lightSpaceMatrix * vec4(adjustedWorldPos, 1.0);
    }
    
    // Calculate shadow with adaptive bias based on surface angle
    float shadow = 0.0;
    float selfShadow = 1.0;
    
    // Only calculate shadows for surfaces facing the light (use geometry normal to avoid artifacts)
    if (geometryNdotL > 0.01) {
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
        if (lightDirT.z > 0.1) {
            // use adjusted heightScale and minLayers computed earlier for self-shadow ray-marching
            selfShadow = ParallaxSelfShadow(uv, lightDirT, currentHeight, texIndex, adjHeightScale, adjMinLayers);
        }
        // If light is at grazing angle or below, just don't apply self-shadow (leave at 1.0)
        }
    } else {
        // Back-facing surfaces should be in full shadow
        shadow = 1.0;
    }
    
    // Combine global shadow and self-shadow (multiply to get darker result)
    float totalShadow = max(shadow, 1.0 - selfShadow);
    
    vec3 ambient = albedoColor * ubo.pomFlags.z; // ambient factor
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);
    
    // Specular lighting (Blinn-Phong) - uses normal-mapped surface normal
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 halfwayDir = normalize(toLight + viewDir);
    float spec = pow(max(dot(worldNormal, halfwayDir), 0.0), ubo.specularParams.y); // shininess from uniform
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * ubo.specularParams.x; // specular strength from uniform
    
    outColor = vec4(ambient + diffuse + specular, 1.0);
}
