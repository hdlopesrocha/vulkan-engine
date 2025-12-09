#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragTangent;
layout(location = 5) flat in int fragTexIndex;

// UBO must match binding 0 defined in vertex shader / host code
// UBO must match CPU-side UniformObject layout:
// mat4 mvp; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor;
// UBO must match CPU-side UniformObject layout:
// mat4 mvp; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor; vec4 pomParams; vec4 pomFlags;
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=unused
} ubo;

// texture arrays: binding 1 = albedo array, 2 = normal array, 3 = height array
layout(binding = 1) uniform sampler2DArray albedoArray;
layout(binding = 2) uniform sampler2DArray normalArray;
layout(binding = 3) uniform sampler2DArray heightArray;

layout(location = 0) out vec4 outColor;
layout(location = 4) in vec3 fragPosWorld;

// Parallax Occlusion Mapping helper
vec2 ParallaxOcclusionMapping(vec2 texCoords, vec3 viewDirT, int texIndex) {
    float heightScale = ubo.pomParams.x;
    float minLayers = ubo.pomParams.y;
    float maxLayers = ubo.pomParams.z;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0,0.0,1.0), viewDirT)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    // parallax direction selectable via ubo.pomFlags.w (0.0 = normal, 1.0 = flipped)
    vec2 P = (ubo.pomFlags.w > 0.5) ? -viewDirT.xy * heightScale : viewDirT.xy * heightScale;
    vec2 deltaTex = P / numLayers;
    vec2 currentTex = texCoords;
    float currentDepthMapValue = texture(heightArray, vec3(currentTex, float(texIndex))).r;
    vec2 prevTex = currentTex;
    float prevDepth = currentDepthMapValue;
    // linear search through layers
    for (int i = 0; i < int(numLayers); ++i) {
        prevTex = currentTex;
        prevDepth = currentDepthMapValue;
        currentTex -= deltaTex;
    currentDepthMapValue = texture(heightArray, vec3(currentTex, float(texIndex))).r;
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
    float midDepthMap = texture(heightArray, vec3(midTex, float(texIndex))).r;
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
    return clamp(finalTex, 0.0, 1.0);
}

void main() {
    // Always use layer 0 since each descriptor set points to different textures
    int texIndex = 0;
    
    vec2 uv = fragUV;
    
    // Parallax Occlusion Mapping (if enabled)
    if (ubo.pomParams.w > 0.5) {
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
        
        vec3 B = cross(N, T);
        mat3 TBN = mat3(T, B, N);
        
        vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
        vec3 viewDirT = normalize(transpose(TBN) * viewDir);
        
        // Apply parallax occlusion mapping
        uv = ParallaxOcclusionMapping(fragUV, viewDirT, texIndex);
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
    
    // Lighting calculation
    vec3 lightDir = normalize(ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, lightDir), 0.0);
    
    vec3 ambient = albedoColor * ubo.pomFlags.z; // ambient factor
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL;
    
    outColor = vec4(ambient + diffuse, 1.0);
}
