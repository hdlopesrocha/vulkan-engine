// Shortest-arc tilt mapping +Y onto a (unit) surface normal, mirroring
// tiltToNormal in VegetationCommon.glsl (which needs perlin + windParams and
// therefore cannot be included by texture-free fragment shaders). Used by
// the impostor path to apply the SAME terrain tilt to reconstructed capture
// positions and normals that the blade vertex shader bakes into its frame.
#ifndef NORMAL_TILT_GLSL
#define NORMAL_TILT_GLSL

// Rotate a direction/vector around the world-up (Y) axis. Canonical
// definition shared by VegetationCommon.glsl (blade VS) and the impostor
// fragment shaders, so the reconstruction applies the same rotation the
// blade vertex shader baked into its frame.
vec3 rotateY(vec3 v, float c, float s) {
    return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

// Identity for +Y; safe for degenerate (zero) normals.
vec3 grassTiltToNormal(vec3 v, vec3 n) {
    vec3 k = vec3(n.z, 0.0, -n.x); // cross(+Y, n)
    float s = length(k);
    if (s < 1e-4) return (n.y >= 0.0) ? v : vec3(v.x, -v.y, v.z);
    k /= s;
    float c = clamp(n.y, -1.0, 1.0);
    return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

#endif // NORMAL_TILT_GLSL
