#ifndef SDF_OPERATION_REPEAT_GLSL
#define SDF_OPERATION_REPEAT_GLSL

// Tiles space with period rep. A non-positive component disables tiling on
// that axis (avoids divide-by-zero, keeps the coordinate untouched).
vec3 opRepeat(vec3 p, vec3 rep) {
    vec3 q = p;
    if (rep.x > 1e-6) {
        q.x = mod(p.x + 0.5 * rep.x, rep.x) - 0.5 * rep.x;
    }
    if (rep.y > 1e-6) {
        q.y = mod(p.y + 0.5 * rep.y, rep.y) - 0.5 * rep.y;
    }
    if (rep.z > 1e-6) {
        q.z = mod(p.z + 0.5 * rep.z, rep.z) - 0.5 * rep.z;
    }
    return q;
}

#endif // SDF_OPERATION_REPEAT_GLSL
