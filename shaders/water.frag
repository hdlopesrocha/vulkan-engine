#version 450

// Water fragment shader
// Implements refraction, specular lighting, and depth-based edge foam

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragPosClip;  // clip-space position for depth lookup

layout(location = 0) out vec4 outColor;

// Use the same UBO as main shader (includes/ubo.glsl style)
layout(set = 1, binding = 0) uniform UniformBufferObject {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 materialFlags;
    mat4 lightSpaceMatrix;
    vec4 shadowEffects;
    vec4 debugParams;
    vec4 triplanarSettings;
    vec4 tessParams;   // x=nearDist, y=farDist, z=minLevel, w=maxLevel
    vec4 passParams;   // x=time, y=tessEnabled, z=waveScale, w=noiseScale
} ubo;

// Scene depth texture for edge foam (set 2, binding 0)
layout(set = 2, binding = 0) uniform sampler2D sceneDepthTex;

// Near/far planes for linearizing depth
const float nearPlane = 0.1;
const float farPlane = 8192.0;

// Linearize depth from [0,1] NDC to view-space distance
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // NDC
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

// Perlin noise functions for detail
vec3 hash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float perlinNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    float n000 = dot(hash33(i + vec3(0,0,0)), f - vec3(0,0,0));
    float n001 = dot(hash33(i + vec3(0,0,1)), f - vec3(0,0,1));
    float n010 = dot(hash33(i + vec3(0,1,0)), f - vec3(0,1,0));
    float n011 = dot(hash33(i + vec3(0,1,1)), f - vec3(0,1,1));
    float n100 = dot(hash33(i + vec3(1,0,0)), f - vec3(1,0,0));
    float n101 = dot(hash33(i + vec3(1,0,1)), f - vec3(1,0,1));
    float n110 = dot(hash33(i + vec3(1,1,0)), f - vec3(1,1,0));
    float n111 = dot(hash33(i + vec3(1,1,1)), f - vec3(1,1,1));
    
    float nx00 = mix(n000, n100, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    
    return mix(nxy0, nxy1, u.z);
}

float fbm(vec3 p, int octaves) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++) {
        total += perlinNoise3D(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return total / maxValue;
}

void main() {
    // Get water parameters from UBO
    float time = ubo.passParams.x;
    float noiseScale = ubo.passParams.w;
    float waveScale = ubo.passParams.z;
    
    // Water rendering parameters (can be moved to UBO for UI control)
    float fresnelPower = 5.0;
    float transparency = 0.6;
    float foamDepthThreshold = 2.0;  // Distance in world units for edge foam
    
    // Normalize vectors
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPos);
    vec3 lightDir = normalize(-ubo.lightDir.xyz);
    
    // Sample scene depth and compute edge foam
    // Convert clip position to NDC screen UV
    vec2 screenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
    float sceneDepthRaw = texture(sceneDepthTex, screenUV).r;
    
    // Linearize depths for comparison
    float sceneDepthLinear = linearizeDepth(sceneDepthRaw);
    float waterDepthLinear = linearizeDepth(gl_FragCoord.z);
    
    // Depth difference: positive when water is in front of scene geometry
    float depthDiff = sceneDepthLinear - waterDepthLinear;
    
    // Edge foam: bright white when water is close to terrain
    float edgeFoam = 1.0 - smoothstep(0.0, foamDepthThreshold, depthDiff);
    edgeFoam = pow(edgeFoam, 0.5);  // Soften the falloff
    
    // Add high-frequency detail noise to normal
    vec3 detailNoisePos = vec3(fragPos.x * noiseScale * 0.5, time * 0.5, fragPos.z * noiseScale * 0.5);
    float detailNoise = fbm(detailNoisePos, 3);
    vec3 detailNormal = normalize(normal + vec3(detailNoise * 0.15, 0.0, detailNoise * 0.15));
    
    // Water color - deep blue-green
    vec3 deepColor = vec3(0.0, 0.1, 0.2);
    vec3 shallowColor = vec3(0.0, 0.4, 0.5);
    
    // Fresnel effect - more reflective at grazing angles
    float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), fresnelPower);
    fresnel = clamp(fresnel, 0.0, 1.0);
    
    // Sky reflection approximation
    vec3 reflectDir = reflect(-viewDir, detailNormal);
    vec3 skyColor = mix(vec3(0.5, 0.6, 0.8), vec3(0.8, 0.85, 0.95), max(reflectDir.y, 0.0));
    
    // Refraction-based color distortion
    float refractionNoise = perlinNoise3D(vec3(fragPos.xz * noiseScale * 0.2, time * 0.3));
    vec3 refractedColor = mix(deepColor, shallowColor, 0.5 + refractionNoise * 0.5);
    
    // Specular highlight (Blinn-Phong)
    vec3 halfDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(detailNormal, halfDir), 0.0);
    float specular = pow(specAngle, 128.0); // High shininess for water
    vec3 specularColor = ubo.lightColor.xyz * specular * 1.5;
    
    // Secondary specular for sun glitter
    float glitterNoise = fbm(vec3(fragPos.xz * noiseScale * 2.0, time * 2.0), 2);
    float glitter = pow(specAngle, 64.0) * (0.5 + glitterNoise * 0.5);
    specularColor += ubo.lightColor.xyz * glitter * 0.5;
    
    // Combine: mix refracted water color with reflected sky based on fresnel
    vec3 waterColor = mix(refractedColor, skyColor, fresnel);
    
    // Add specular highlights
    waterColor += specularColor;
    
    // Ambient contribution
    waterColor += refractedColor * 0.1;
    
    // Procedural foam based on noise (wave peaks)
    float foamNoise = fbm(vec3(fragPos.xz * noiseScale * 0.3, time * 0.2), 3);
    float proceduralFoam = smoothstep(0.6, 0.8, foamNoise) * 0.3;
    
    // Combine procedural foam with edge foam
    float totalFoam = max(proceduralFoam, edgeFoam);
    waterColor = mix(waterColor, vec3(0.95, 0.98, 1.0), totalFoam);
    
    // Calculate alpha - more opaque with foam
    float alpha = mix(transparency, 0.95, max(fresnel, totalFoam));
    
    outColor = vec4(waterColor, alpha);
}
