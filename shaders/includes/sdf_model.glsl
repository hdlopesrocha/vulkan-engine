#ifndef SDF_MODEL_INCLUDE_GLSL
#define SDF_MODEL_INCLUDE_GLSL

// Generic SDF model transform helpers. Every GPU SDF is evaluated in its
// own local frame; these functions map that frame to world space
// (translate * rotate * scale) and back. Primitives never transform points
// themselves: the evaluator (sdfEvalInstance) builds the model once per
// instance and passes local coordinates down.
//
// Type: shaders/types/SdfModel.glsl
// Requires: shaders/includes/sdf_ops.glsl (sdfEulerMat) included first, and
// the SdfInstance wire type (shaders/types/SdfInstance.glsl).
#include "../types/SdfModel.glsl"
#include "../types/SdfInstance.glsl"

// Builds the model from the canonical instance transform fields
// (SdfInstance): position + rotation + per-axis scale.
SdfModel sdfModelFromInstance(SdfInstance inst) {
    float uni = max(inst.scale, 1e-4);
    float rSc = max(inst.radiusScale, 1e-4);
    float hSc = max(inst.heightScale, 1e-4);
    SdfModel m;
    m.translation = inst.position;
    m.rotation = inst.rotation;
    m.scale = vec3(uni * rSc, uni * hSc, uni * rSc);
    return m;
}

// local -> world: scale, then rotate, then translate.
vec3 sdfModelToWorld(SdfModel m, vec3 lpos) {
    return sdfEulerMat(m.rotation) * (lpos * m.scale) + m.translation;
}

// world -> local (inverse TRS). outScale is the conservative local->world
// distance scale (min axis): a local distance d maps to at most d * outScale,
// so sphere tracing never oversteps the world-space field.
vec3 sdfModelToLocal(SdfModel m, vec3 wpos, out float outScale) {
    vec3 s = max(abs(m.scale), vec3(1e-4));
    vec3 q = transpose(sdfEulerMat(m.rotation)) * (wpos - m.translation);
    outScale = min(min(s.x, s.y), s.z);
    return q / s;
}

// Direction-only variant (no translation), for rotating vectors that live in
// world space (wind, sun direction) into an instance's local frame.
vec3 sdfModelDirToLocal(SdfModel m, vec3 wdir) {
    vec3 s = max(abs(m.scale), vec3(1e-4));
    return transpose(sdfEulerMat(m.rotation)) * (wdir / s);
}

#endif // SDF_MODEL_INCLUDE_GLSL
