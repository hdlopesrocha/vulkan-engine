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

vec2 perlin2d_grad(vec2 ip) {
    float a = perlin2d_hash12(ip) * 6.28318530718;
    return vec2(cos(a), sin(a));
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
