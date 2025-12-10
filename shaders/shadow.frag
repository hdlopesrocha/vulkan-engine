#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTangent;
layout(location = 3) in vec3 fragPosWorld;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=unused, w=flipParallaxDirection
} push;

layout(binding = 3) uniform sampler2DArray heightArray;

// Simplified parallax occlusion mapping for shadow pass
vec2 ParallaxOcclusionMapping(vec2 texCoords, vec3 viewDirT, int texIndex) {
    float heightScale = push.pomParams.x;
    float minLayers = push.pomParams.y;
    float maxLayers = push.pomParams.z;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0,0.0,1.0), viewDirT)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    
    vec2 P = (push.pomFlags.w > 0.5) ? -viewDirT.xy * heightScale : viewDirT.xy * heightScale;
    vec2 deltaTex = P / numLayers;
    vec2 currentTex = texCoords;
    float currentDepthMapValue = 1.0 - texture(heightArray, vec3(currentTex, float(texIndex))).r;
    
    // Linear search
    for (int i = 0; i < int(numLayers); ++i) {
        if (currentLayerDepth >= currentDepthMapValue) break;
        currentTex -= deltaTex;
        currentDepthMapValue = 1.0 - texture(heightArray, vec3(currentTex, float(texIndex))).r;
        currentLayerDepth += layerDepth;
    }
    
    return currentTex;
}

void main() {
    // Apply parallax mapping if enabled
    if (push.pomParams.w > 0.5) {
        int texIndex = 0;
        
        // Compute tangent-space view direction
        vec3 N = normalize(fragNormal);
        vec3 T = normalize(fragTangent);
        
        if (push.pomFlags.x > 0.5) {
            N.y = -N.y;
        }
        if (push.pomFlags.y > 0.5) {
            T = -T;
        }
        
        vec3 B = cross(T, N);
        mat3 TBN = mat3(T, B, N);
        
        vec3 viewDir = normalize(push.viewPos.xyz - fragPosWorld);
        vec3 viewDirT = normalize(transpose(TBN) * viewDir);
        
        // Apply parallax and check if we should discard
        vec2 parallaxUV = ParallaxOcclusionMapping(fragUV, viewDirT, texIndex);
        
        // Sample height at parallax-displaced position
        float height = 1.0 - texture(heightArray, vec3(parallaxUV, float(texIndex))).r;
        
        // Discard if significantly occluded (this creates proper shadow silhouette)
        // The threshold determines how aggressive the culling is
        if (height > 0.9) {
            discard;
        }
    }
    
    // Depth is automatically written by the pipeline
}
