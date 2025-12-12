// Parallax Occlusion Mapping helper
vec2 ParallaxOcclusionMapping(vec2 texCoords, vec3 viewDirT, int texIndex, float heightScale, float minLayers, float maxLayers) {
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0,0.0,1.0), viewDirT)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    // parallax direction: project the tangent-space view direction onto the texture plane
    // P = (viewDirT.xy / viewDirT.z) * heightScale; guard against small z
    float vz = max(abs(viewDirT.z), 1e-6);
    vec2 P = (ubo.pomFlags.w > 0.5) ? -(viewDirT.xy / vz) * heightScale : (viewDirT.xy / vz) * heightScale;
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
float ParallaxSelfShadow(vec2 texCoords, vec3 lightDirT, float currentHeight, int texIndex, float heightScale, float minLayers) {
    float qualityMultiplier = ubo.shadowEffects.z; // quality setting from UI
    
    // Number of samples for shadow ray-march (adjusted by quality)
    float numSamples = minLayers * qualityMultiplier;
    numSamples = max(numSamples, 4.0); // At least 4 samples for decent quality
    
    // March in the direction of the light in texture space
    // Use the same projection as POM: project light direction into texture plane
    float lz = max(abs(lightDirT.z), 1e-6);
    vec2 L = (ubo.pomFlags.w > 0.5) ? -(lightDirT.xy / lz) * heightScale : (lightDirT.xy / lz) * heightScale;
    vec2 rayStep = L / numSamples;
    
    // Start from current displaced position
    vec2 currentTexCoords = texCoords;
    
    // Height system: 0.0 = surface (white), 1.0 = deepest (black)
    // We're at currentHeight and march towards the light
    float rayHeight = currentHeight * heightScale;
    
    // Height change per step - scale by heightScale to match
    float layerDepth = (max(abs(lightDirT.z), 1e-6) * heightScale) / numSamples;
    
    // March from current position towards the light source
    for (int i = 1; i <= int(numSamples); ++i) {
        // Move along the ray
        currentTexCoords -= rayStep; // Subtract like POM does
        rayHeight -= layerDepth; // Move up (decrease depth)
        
        // If we've reached the surface, no more shadow
        if (rayHeight <= 0.0) break;
        
        // Sample height at this position
        float sampledHeight = sampleHeight(currentTexCoords, texIndex) * heightScale;
        
        // If sampled point is shallower (less deep) than our ray, it blocks the light
        float bias = heightScale * 0.02;
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