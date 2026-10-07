// 2D Perlin noise helpers shared by vegetation and impostor shaders.
// Depends on nothing — include before any code that calls perlin2().

vec2 perlin2d_fade(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float perlin2d_hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float perlin2d_hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 8 fixed unit directions. Indexed by an integer hash so the gradient is
// selected without any transcendental cos/sin in the hot path (mirrors the
// SH3 fix applied to the 3D/4D noise in perlin.glsl).
const vec2 perlin2d_grad_table[8] = vec2[8](
    vec2( 1.0,         0.0),
    vec2( 0.70710678,  0.70710678),
    vec2( 0.0,         1.0),
    vec2(-0.70710678,  0.70710678),
    vec2(-1.0,         0.0),
    vec2(-0.70710678, -0.70710678),
    vec2( 0.0,        -1.0),
    vec2( 0.70710678, -0.70710678)
);

vec2 perlin2d_grad(vec2 ip) {
    int idx = int(perlin2d_hash12(ip) * 8.0) & 7;
    return perlin2d_grad_table[idx];
}

float perlin2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = perlin2d_fade(f);

    float n00 = dot(perlin2d_grad(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float n10 = dot(perlin2d_grad(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float n01 = dot(perlin2d_grad(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float n11 = dot(perlin2d_grad(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    float nx0 = mix(n00, n10, u.x);
    float nx1 = mix(n01, n11, u.x);
    return mix(nx0, nx1, u.y);
}
