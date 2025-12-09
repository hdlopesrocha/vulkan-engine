#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragTangent;

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

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform sampler2D heightMap;

layout(location = 0) out vec4 outColor;
layout(location = 4) in vec3 fragPosWorld;

// Parallax Occlusion Mapping helper
vec2 ParallaxOcclusionMapping(vec2 texCoords, vec3 viewDirT) {
    float heightScale = ubo.pomParams.x;
    float minLayers = ubo.pomParams.y;
    float maxLayers = ubo.pomParams.z;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0,0.0,1.0), viewDirT)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 P = viewDirT.xy * heightScale;
    vec2 deltaTex = P / numLayers;
    vec2 currentTex = texCoords;
    float currentDepthMapValue = texture(heightMap, currentTex).r;
    vec2 prevTex = currentTex;
    float prevDepth = currentDepthMapValue;
    // linear search through layers
    for (int i = 0; i < int(numLayers); ++i) {
        prevTex = currentTex;
        prevDepth = currentDepthMapValue;
        currentTex -= deltaTex;
        currentDepthMapValue = texture(heightMap, currentTex).r;
        currentLayerDepth += layerDepth;
        if (currentLayerDepth >= currentDepthMapValue) break;
    }
    // linear interpolation between prev and current
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = prevDepth - (currentLayerDepth - layerDepth);
    float weight = 0.0;
    if ((afterDepth - beforeDepth) != 0.0) weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTex = mix(prevTex, currentTex, weight);
    return clamp(finalTex, 0.0, 1.0);
}

void main() {
    // compute view direction in world space
    vec3 viewDirWorld = normalize(ubo.viewPos.xyz - fragPosWorld);
    // construct TBN
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    vec3 B = normalize(cross(N, T));
    // view direction in tangent space
    vec3 Vt = normalize(vec3(dot(viewDirWorld, T), dot(viewDirWorld, B), dot(viewDirWorld, N)));
    // compute parallax occlusion mapping to get final texture coords
    vec2 finalUV = ParallaxOcclusionMapping(fragUV, Vt);

    // sample textures with parallax-corrected UV
    vec4 tex = texture(texSampler, finalUV);
    vec3 mapN = texture(normalMap, finalUV).xyz * 2.0 - 1.0; // tangent-space normal
    // optional flip Y channel
    if (ubo.pomFlags.x > 0.5) mapN.y = -mapN.y;
    // transform normal to world space via TBN; allow flipping tangent-handedness
    vec3 B_use = (ubo.pomFlags.y > 0.5) ? normalize(cross(T, N)) : normalize(cross(N, T));
    vec3 mapped = normalize(mapN.x * T + mapN.y * B_use + mapN.z * N);

    vec3 L = normalize(ubo.lightDir.xyz);
    float diff = max(dot(mapped, L), 0.0);
    float ambient = ubo.pomFlags.z; // ambient stored in pomFlags.z
    vec3 lighting = (ambient + diff * ubo.lightColor.w) * ubo.lightColor.rgb; // ambient + diffuse*intensity
    vec3 shaded = tex.rgb * fragColor * lighting;
    outColor = vec4(shaded, tex.a);
}
