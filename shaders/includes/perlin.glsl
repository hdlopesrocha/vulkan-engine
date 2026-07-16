// Perlin noise utilities for GLSL shaders
// 3D and 4D Perlin noise with fractional Brownian motion (FBM)

// PCG-style integer hash (xsh rs) plus a uint->[0,1) float conversion.
// Bit-mixing replaces the classic fract(sin(dot(...))) hash: it avoids
// transcendental sin evaluations (a known SFU bottleneck) and the precision
// artefacts that idiom shows at large coordinates on some vendors.
// Shared between perlin.glsl and voronoi.glsl via the include guard below.
#ifndef PERLIN_GLSL_PCG_HELPERS
#define PERLIN_GLSL_PCG_HELPERS
uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float uintToUnitFloat(uint x) {
    // Map the top 23 bits into the mantissa of a [1.0, 2.0) float, then bias
    // down to [0.0, 1.0). Pure bit manipulation, no transcendentals.
    return uintBitsToFloat(0x3f800000u | (x >> 9u)) - 1.0;
}
#endif

vec3 hash33(vec3 p) {
    uvec3 u = floatBitsToUint(p);
    uint s = pcgHash(u.x);
    s = pcgHash(s ^ u.y);
    s = pcgHash(s ^ u.z);
    return vec3(uintToUnitFloat(pcgHash(s + 0x9E3779B9u)),
                uintToUnitFloat(pcgHash(s + 0x85EBCA6Bu)),
                uintToUnitFloat(pcgHash(s + 0xC2B2AE35u))) * 2.0 - 1.0;
}

vec4 hash44(vec4 p) {
    uvec4 u = floatBitsToUint(p);
    uint s = pcgHash(u.x);
    s = pcgHash(s ^ u.y);
    s = pcgHash(s ^ u.z);
    s = pcgHash(s ^ u.w);
    return vec4(uintToUnitFloat(pcgHash(s + 0x9E3779B9u)),
                uintToUnitFloat(pcgHash(s + 0x85EBCA6Bu)),
                uintToUnitFloat(pcgHash(s + 0xC2B2AE35u)),
                uintToUnitFloat(pcgHash(s + 0x27D4EB2Fu))) * 2.0 - 1.0;
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

float fbm(vec3 p, int octaves, float persistence, float lacunarity) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++) {
        total += perlinNoise3D(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return total / maxValue;
}

float fbm(vec4 p, int octaves, float persistence, float lacunarity) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++) {
        total += perlinNoise4D(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return total / maxValue;
}

// Analytic gradient of perlinNoise4D.
// Returns vec4(value, d/dx, d/dy, d/dz).  The w (time) axis is treated as a
// fixed offset, so its derivative is dropped: we only need the spatial gradient
// of the water height field.  Each corner gradient is the (constant) random
// vector hash44(i+c); the per-octave spatial gradient is g_c, and the mix tree
// accumulates it with the Hermite derivative du = 30 f^2 (1-f)^2 along the axis
// currently being interpolated.
vec4 perlinNoise4DGrad(vec4 p) {
    vec4 I = floor(p);
    vec4 f = fract(p);
    vec4 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec4 du = 30.0 * f * f * (1.0 - f) * (1.0 - f);

    vec4 g0000 = hash44(I + vec4(0,0,0,0));
    vec4 g0001 = hash44(I + vec4(0,0,0,1));
    vec4 g0010 = hash44(I + vec4(0,0,1,0));
    vec4 g0011 = hash44(I + vec4(0,0,1,1));
    vec4 g0100 = hash44(I + vec4(0,1,0,0));
    vec4 g0101 = hash44(I + vec4(0,1,0,1));
    vec4 g0110 = hash44(I + vec4(0,1,1,0));
    vec4 g0111 = hash44(I + vec4(0,1,1,1));
    vec4 g1000 = hash44(I + vec4(1,0,0,0));
    vec4 g1001 = hash44(I + vec4(1,0,0,1));
    vec4 g1010 = hash44(I + vec4(1,0,1,0));
    vec4 g1011 = hash44(I + vec4(1,0,1,1));
    vec4 g1100 = hash44(I + vec4(1,1,0,0));
    vec4 g1101 = hash44(I + vec4(1,1,0,1));
    vec4 g1110 = hash44(I + vec4(1,1,1,0));
    vec4 g1111 = hash44(I + vec4(1,1,1,1));

    float n0000 = dot(g0000, f);
    float n0001 = dot(g0001, f - vec4(0,0,0,1));
    float n0010 = dot(g0010, f - vec4(0,0,1,0));
    float n0011 = dot(g0011, f - vec4(0,0,1,1));
    float n0100 = dot(g0100, f - vec4(0,1,0,0));
    float n0101 = dot(g0101, f - vec4(0,1,0,1));
    float n0110 = dot(g0110, f - vec4(0,1,1,0));
    float n0111 = dot(g0111, f - vec4(0,1,1,1));
    float n1000 = dot(g1000, f - vec4(1,0,0,0));
    float n1001 = dot(g1001, f - vec4(1,0,0,1));
    float n1010 = dot(g1010, f - vec4(1,0,1,0));
    float n1011 = dot(g1011, f - vec4(1,0,1,1));
    float n1100 = dot(g1100, f - vec4(1,1,0,0));
    float n1101 = dot(g1101, f - vec4(1,1,0,1));
    float n1110 = dot(g1110, f - vec4(1,1,1,0));
    float n1111 = dot(g1111, f - vec4(1,1,1,1));

    // Interpolate along x (axis 0)
    float nx000 = mix(n0000, n1000, u.x);
    vec4  gx000 = mix(g0000, g1000, u.x); gx000.x += (n1000 - n0000) * du.x;
    float nx001 = mix(n0001, n1001, u.x);
    vec4  gx001 = mix(g0001, g1001, u.x); gx001.x += (n1001 - n0001) * du.x;
    float nx010 = mix(n0010, n1010, u.x);
    vec4  gx010 = mix(g0010, g1010, u.x); gx010.x += (n1010 - n0010) * du.x;
    float nx011 = mix(n0011, n1011, u.x);
    vec4  gx011 = mix(g0011, g1011, u.x); gx011.x += (n1011 - n0011) * du.x;
    float nx100 = mix(n0100, n1100, u.x);
    vec4  gx100 = mix(g0100, g1100, u.x); gx100.x += (n1100 - n0100) * du.x;
    float nx101 = mix(n0101, n1101, u.x);
    vec4  gx101 = mix(g0101, g1101, u.x); gx101.x += (n1101 - n0101) * du.x;
    float nx110 = mix(n0110, n1110, u.x);
    vec4  gx110 = mix(g0110, g1110, u.x); gx110.x += (n1110 - n0110) * du.x;
    float nx111 = mix(n0111, n1111, u.x);
    vec4  gx111 = mix(g0111, g1111, u.x); gx111.x += (n1111 - n0111) * du.x;

    // Interpolate along y (axis 1)
    float nxy00 = mix(nx000, nx010, u.y);
    vec4  gxy00 = mix(gx000, gx010, u.y); gxy00.y += (nx010 - nx000) * du.y;
    float nxy01 = mix(nx001, nx011, u.y);
    vec4  gxy01 = mix(gx001, gx011, u.y); gxy01.y += (nx011 - nx001) * du.y;
    float nxy10 = mix(nx100, nx110, u.y);
    vec4  gxy10 = mix(gx100, gx110, u.y); gxy10.y += (nx110 - nx100) * du.y;
    float nxy11 = mix(nx101, nx111, u.y);
    vec4  gxy11 = mix(gx101, gx111, u.y); gxy11.y += (nx111 - nx101) * du.y;

    // Interpolate along z (axis 2)
    float nxyz0 = mix(nxy00, nxy10, u.z);
    vec4  gxyz0 = mix(gxy00, gxy10, u.z); gxyz0.z += (nxy10 - nxy00) * du.z;
    float nxyz1 = mix(nxy01, nxy11, u.z);
    vec4  gxyz1 = mix(gxy01, gxy11, u.z); gxyz1.z += (nxy11 - nxy01) * du.z;

    // Interpolate along w (axis 3, time) — only the w component of the gradient
    // is affected, and we discard it.
    float v = mix(nxyz0, nxyz1, u.w);
    vec4  g = mix(gxyz0, gxyz1, u.w);     g.w += (nxyz1 - nxyz0) * du.w;

    return vec4(v, g.xyz);
}

// Analytic gradient of fbm(vec4(p, ...), ...).  Returns
// vec4(value, d/dx, d/dy, d/dz).  The per-octave chain rule contributes an
// extra `frequency` factor because each octave samples perlinNoise4D(p*frequency).
vec4 fbmGrad4D(vec4 p, int octaves, float persistence, float lacunarity) {
    float total = 0.0;
    vec3  grad = vec3(0.0);
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        vec4 r = perlinNoise4DGrad(p * frequency);
        total += r.x * amplitude;
        grad  += r.yzw * (amplitude * frequency);
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return vec4(total / maxValue, grad / maxValue);
}

// Legacy versions with hardcoded defaults for backward compatibility
float fbm(vec3 p, int octaves, float persistence) {
    return fbm(p, octaves, persistence, 2.0);
}
float fbm(vec4 p, int octaves, float persistence) {
    return fbm(p, octaves, persistence, 2.0);
}
float fbm(vec3 p, int octaves) {
    return fbm(p, octaves, 0.5, 2.0);
}
