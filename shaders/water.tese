#version 450

// Water tessellation evaluation shader
// Applies wave displacement using Perlin noise

layout(triangles, equal_spacing, ccw) in;

layout(location = 0) in vec3 inPos[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec2 inTexCoord[];

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragPosClip;  // clip-space position for depth lookup

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

// Perlin noise functions
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
    // Interpolate position
    vec3 pos = gl_TessCoord.x * inPos[0] + 
               gl_TessCoord.y * inPos[1] + 
               gl_TessCoord.z * inPos[2];
    
    // Interpolate normal
    vec3 normal = normalize(gl_TessCoord.x * inNormal[0] + 
                           gl_TessCoord.y * inNormal[1] + 
                           gl_TessCoord.z * inNormal[2]);
    
    // Interpolate texture coordinates
    fragTexCoord = gl_TessCoord.x * inTexCoord[0] + 
                   gl_TessCoord.y * inTexCoord[1] + 
                   gl_TessCoord.z * inTexCoord[2];
    
    // Get water parameters
    float time = ubo.passParams.x;
    float waveScale = ubo.passParams.z;
    float noiseScale = ubo.passParams.w;
    
    // Wave parameters - using reasonable defaults
    float waveSpeed = 0.5;
    float waveHeight = 0.5;
    
    // Calculate wave displacement using world-space Perlin noise
    vec3 noisePos1 = vec3(pos.x * noiseScale * 0.1, time * waveSpeed, pos.z * noiseScale * 0.1);
    vec3 noisePos2 = vec3(pos.x * noiseScale * 0.07 + 50.0, time * waveSpeed * 0.8, pos.z * noiseScale * 0.07 + 50.0);
    
    float wave1 = fbm(noisePos1, 4);
    float wave2 = fbm(noisePos2, 3);
    float waveDisplacement = (wave1 + wave2 * 0.5) * waveHeight;
    
    // Displace position along normal (Y-up for water surface)
    pos.y += waveDisplacement;
    
    // Calculate perturbed normal from wave gradient
    float eps = 0.1;
    vec3 noisePos1X = vec3((pos.x + eps) * noiseScale * 0.1, time * waveSpeed, pos.z * noiseScale * 0.1);
    vec3 noisePos1Z = vec3(pos.x * noiseScale * 0.1, time * waveSpeed, (pos.z + eps) * noiseScale * 0.1);
    
    float waveX = fbm(noisePos1X, 4);
    float waveZ = fbm(noisePos1Z, 4);
    
    float dX = (waveX - wave1) / eps * waveHeight;
    float dZ = (waveZ - wave1) / eps * waveHeight;
    
    // Perturbed normal
    fragNormal = normalize(vec3(-dX, 1.0, -dZ));
    
    fragPos = pos;
    vec4 clipPos = ubo.viewProjection * vec4(pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}
