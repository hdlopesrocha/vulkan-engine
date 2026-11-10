#ifndef SDF_OPERATION_COMBINE_GLSL
#define SDF_OPERATION_COMBINE_GLSL

// Generic SDF combinator dispatcher (operator ids live in
// types/SdfOpType.glsl; CPU twin sdf/types/SdfOpType.hpp).

#include "../../../types/SdfOpType.glsl"

#include "Union.glsl"
#include "Intersection.glsl"
#include "Subtraction.glsl"
#include "SmoothUnion.glsl"
#include "SmoothIntersection.glsl"
#include "SmoothSubtraction.glsl"
#include "Interpolation.glsl"

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
    if (op == SDF_OP_INTERPOLATION) {
        return opInterpolation(a, b, k);
    }
    return opUnion(a, b);
}

#endif // SDF_OPERATION_COMBINE_GLSL
