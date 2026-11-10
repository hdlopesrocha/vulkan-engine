#ifndef SDF_OPERATION_MIRROR_GLSL
#define SDF_OPERATION_MIRROR_GLSL

// Folds space onto the positive side of the plane dot(p, n) = offset, so one
// primitive instance renders mirrored on both sides. A degenerate normal
// falls back to the XZ ground plane.
vec3 opMirror(vec3 p, vec3 n, float offset) {
    vec3 nn = (length(n) > 1e-8) ? normalize(n) : vec3(0.0, 1.0, 0.0);
    float d = dot(p, nn) - offset;
    return p - nn * (min(d, 0.0) * 2.0);
}

#endif // SDF_OPERATION_MIRROR_GLSL
