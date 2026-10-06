// Generic SDF combinators and space transforms (IQ-style exact/polynomial).
// No textures, no vendor-specific behavior.

#ifndef SDF_OPS_GLSL
#define SDF_OPS_GLSL

// Operator ids live in types/SdfOpType.glsl (CPU twin sdf/types/SdfOpType.hpp).
#include "../types/SdfOpType.glsl"

float opUnion(float d1, float d2) {
    return min(d1, d2);
}

float opIntersection(float d1, float d2) {
    return max(d1, d2);
}

// Subtracts d1 from d2 (keeps the part of d2 outside d1).
float opSubtraction(float d1, float d2) {
    return max(d2, -d1);
}

// Polynomial smooth-min family (IQ). k is clamped to [0, 2]; k <= 0 falls
// back to the hard op. Callers pass the material smooth factor
// (SdfDefinition.smoothK, clamped here again for safety).
float opSmoothUnion(float d1, float d2, float k) {
    float kk = clamp(k, 0.0, 2.0);
    if (kk <= 1e-6) {
        return min(d1, d2);
    }
    float h = clamp(0.5 + 0.5 * (d2 - d1) / kk, 0.0, 1.0);
    return mix(d2, d1, h) - kk * h * (1.0 - h);
}

float opSmoothIntersection(float d1, float d2, float k) {
    float kk = clamp(k, 0.0, 2.0);
    if (kk <= 1e-6) {
        return max(d1, d2);
    }
    float h = clamp(0.5 - 0.5 * (d2 - d1) / kk, 0.0, 1.0);
    return mix(d2, d1, h) + kk * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k) {
    float kk = clamp(k, 0.0, 2.0);
    if (kk <= 1e-6) {
        return max(d2, -d1);
    }
    float h = clamp(0.5 - 0.5 * (d2 + d1) / kk, 0.0, 1.0);
    return mix(d2, -d1, h) + kk * h * (1.0 - h);
}

// Combines two fields with the op selector from SdfDefinition.op.
float sdfCombine(float a, float b, uint op, float k) {
    if (op == SDF_OP_INTERSECTION) {
        return opIntersection(a, b);
    }
    if (op == SDF_OP_SUBTRACTION) {
        return opSubtraction(a, b);
    }
    if (op == SDF_OP_SMOOTH_UNION) {
        return opSmoothUnion(a, b, k);
    }
    if (op == SDF_OP_SMOOTH_INTERSECTION) {
        return opSmoothIntersection(a, b, k);
    }
    if (op == SDF_OP_SMOOTH_SUBTRACTION) {
        return opSmoothSubtraction(a, b, k);
    }
    return opUnion(a, b);
}

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

// Folds space onto the positive side of the plane dot(p, n) = offset, so one
// primitive instance renders mirrored on both sides. A degenerate normal
// falls back to the XZ ground plane.
vec3 opMirror(vec3 p, vec3 n, float offset) {
    vec3 nn = (length(n) > 1e-8) ? normalize(n) : vec3(0.0, 1.0, 0.0);
    float d = dot(p, nn) - offset;
    return p - nn * (min(d, 0.0) * 2.0);
}

// Rotation from XYZ euler angles (radians). Order: R = Rx * Ry * Rz, i.e.
// intrinsic XYZ. Shared by the generic SdfModel (sdf_model.glsl) so every
// GPU SDF uses one rotation convention.
mat3 sdfEulerMat(vec3 euler) {
    float cx = cos(euler.x);
    float sx = sin(euler.x);
    float cy = cos(euler.y);
    float sy = sin(euler.y);
    float cz = cos(euler.z);
    float sz = sin(euler.z);
    mat3 rx = mat3(1.0, 0.0, 0.0,
                  0.0, cx, sx,
                  0.0, -sx, cx);
    mat3 ry = mat3(cy, 0.0, -sy,
                  0.0, 1.0, 0.0,
                  sy, 0.0, cy);
    mat3 rz = mat3(cz, sz, 0.0,
                  -sz, cz, 0.0,
                  0.0, 0.0, 1.0);
    return rx * ry * rz;
}

#endif // SDF_OPS_GLSL
