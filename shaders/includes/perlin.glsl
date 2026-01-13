// Perlin noise utilities for GLSL shaders
// 3D and 4D Perlin noise with fractional Brownian motion (FBM)

vec3 hash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

vec4 hash44(vec4 p) {
    p = vec4(dot(p, vec4(127.1, 311.7, 74.7, 269.5)),
             dot(p, vec4(269.5, 183.3, 246.1, 127.1)),
             dot(p, vec4(113.5, 271.9, 124.6, 311.7)),
             dot(p, vec4(271.9, 124.6, 113.5, 183.3)));
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

float perlinNoise4D(vec4 p) {
    vec4 i = floor(p);
    vec4 f = fract(p);
    vec4 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    // 16 corners of a 4D hypercube
    float n0000 = dot(hash44(i + vec4(0,0,0,0)), f - vec4(0,0,0,0));
    float n0001 = dot(hash44(i + vec4(0,0,0,1)), f - vec4(0,0,0,1));
    float n0010 = dot(hash44(i + vec4(0,0,1,0)), f - vec4(0,0,1,0));
    float n0011 = dot(hash44(i + vec4(0,0,1,1)), f - vec4(0,0,1,1));
    float n0100 = dot(hash44(i + vec4(0,1,0,0)), f - vec4(0,1,0,0));
    float n0101 = dot(hash44(i + vec4(0,1,0,1)), f - vec4(0,1,0,1));
    float n0110 = dot(hash44(i + vec4(0,1,1,0)), f - vec4(0,1,1,0));
    float n0111 = dot(hash44(i + vec4(0,1,1,1)), f - vec4(0,1,1,1));
    float n1000 = dot(hash44(i + vec4(1,0,0,0)), f - vec4(1,0,0,0));
    float n1001 = dot(hash44(i + vec4(1,0,0,1)), f - vec4(1,0,0,1));
    float n1010 = dot(hash44(i + vec4(1,0,1,0)), f - vec4(1,0,1,0));
    float n1011 = dot(hash44(i + vec4(1,0,1,1)), f - vec4(1,0,1,1));
    float n1100 = dot(hash44(i + vec4(1,1,0,0)), f - vec4(1,1,0,0));
    float n1101 = dot(hash44(i + vec4(1,1,0,1)), f - vec4(1,1,0,1));
    float n1110 = dot(hash44(i + vec4(1,1,1,0)), f - vec4(1,1,1,0));
    float n1111 = dot(hash44(i + vec4(1,1,1,1)), f - vec4(1,1,1,1));
    
    // Interpolate along x
    float nx000 = mix(n0000, n1000, u.x);
    float nx001 = mix(n0001, n1001, u.x);
    float nx010 = mix(n0010, n1010, u.x);
    float nx011 = mix(n0011, n1011, u.x);
    float nx100 = mix(n0100, n1100, u.x);
    float nx101 = mix(n0101, n1101, u.x);
    float nx110 = mix(n0110, n1110, u.x);
    float nx111 = mix(n0111, n1111, u.x);
    
    // Interpolate along y
    float nxy00 = mix(nx000, nx100, u.y);
    float nxy01 = mix(nx001, nx101, u.y);
    float nxy10 = mix(nx010, nx110, u.y);
    float nxy11 = mix(nx011, nx111, u.y);
    
    // Interpolate along z
    float nxyz0 = mix(nxy00, nxy10, u.z);
    float nxyz1 = mix(nxy01, nxy11, u.z);
    
    // Interpolate along w (time)
    return mix(nxyz0, nxyz1, u.w);
}

float fbm(vec3 p, int octaves, float persistence) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++) {
        total += perlinNoise3D(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return total / maxValue;
}

float fbm(vec4 p, int octaves, float persistence) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++) {
        total += perlinNoise4D(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return total / maxValue;
}

// Legacy version with hardcoded persistence for backward compatibility
float fbm(vec3 p, int octaves) {
    return fbm(p, octaves, 0.5);
}
