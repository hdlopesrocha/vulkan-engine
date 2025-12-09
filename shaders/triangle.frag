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
    // compute view direction in world space
    vec3 viewDirWorld = normalize(ubo.viewPos.xyz - fragPosWorld);
    // construct TBN
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    vec3 B = normalize(cross(N, T));
    // view direction in tangent space
    vec3 Vt = normalize(vec3(dot(viewDirWorld, T), dot(viewDirWorld, B), dot(viewDirWorld, N)));
    // compute parallax occlusion mapping to get final texture coords
    vec2 finalUV = ParallaxOcclusionMapping(fragUV, Vt, fragTexIndex);

    // sample textures with parallax-corrected UV from texture2D arrays
    vec4 tex = texture(albedoArray, vec3(finalUV, float(fragTexIndex)));
    vec3 mapN = texture(normalArray, vec3(finalUV, float(fragTexIndex))).xyz * 2.0 - 1.0; // tangent-space normal
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
